"""End-to-end tests for Path-spec StageExample (issue #387, Option B)."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import polars as pl
import pytest

from MEDS_transforms.stages.examples import (
    ComparatorFn,
    CompareContext,
    StageExample,
    _compare_parquet,
)


@pytest.fixture
def simple_parquet(tmp_path: Path) -> Path:
    """A tiny parquet file at a stable path for reuse."""
    fp = tmp_path / "data.parquet"
    pl.DataFrame({"a": [1, 2]}).write_parquet(fp)
    return fp


def _write_yaml_spec(fp: Path, contents: str) -> Path:
    fp.write_text(contents)
    return fp


# ---------------------------------------------------------------------------------------------
# Path-as-want_data / want_metadata
# ---------------------------------------------------------------------------------------------


def test_both_want_fields_as_paths_permitted(tmp_path: Path):
    """XOR rule only applies to MEDSDataset + DataFrame pair; Path specs may describe both."""
    spec1 = _write_yaml_spec(tmp_path / "out_data.yaml", "data/x.parquet: |\n  a,b\n  1,2\n")
    spec2 = _write_yaml_spec(tmp_path / "out_metadata.yaml", "metadata/codes.parquet: |\n  code\n  foo\n")

    # Should not raise.
    ex = StageExample(stage_name="test", want_data=spec1, want_metadata=spec2)
    assert ex.want_data == spec1
    assert ex.want_metadata == spec2


def test_xor_rule_still_applies_for_medsdataset_and_dataframe(tmp_path: Path):
    """Two non-Path types simultaneously is still a misuse."""
    from meds_testing_helpers.dataset import MEDSDataset
    from meds_testing_helpers.static_sample_data import SIMPLE_STATIC_SHARDED_BY_SPLIT

    ds = MEDSDataset.from_yaml(SIMPLE_STATIC_SHARDED_BY_SPLIT)
    md = pl.DataFrame({"code": ["x"]})

    with pytest.raises(ValueError, match="but not both"):
        StageExample(stage_name="test", want_data=ds, want_metadata=md)


def test_from_dir_falls_back_to_path_for_non_meds_output(tmp_path: Path):
    """out_data.yaml that isn't a MEDS spec is stored as Path (mirrors in_data behavior)."""
    import yaml

    example_dir = tmp_path / "example"
    example_dir.mkdir()
    # A yaml_to_disk spec that isn't MEDS-shaped.
    (example_dir / "out_data.yaml").write_text(
        yaml.dump({"raw/chunk.parquet": None, "sidecar/manifest.json": {"v": 1}})
    )
    (example_dir / "cfg.yaml").write_text(yaml.dump({}))

    ex = StageExample.from_dir("test", ".", example_dir)
    assert isinstance(ex.want_data, Path)
    assert ex.want_data == example_dir / "out_data.yaml"


def test_from_dir_still_parses_valid_meds_output(tmp_path: Path):
    """A MEDS-shaped out_data.yaml is still parsed as a MEDSDataset (no regression)."""
    from meds_testing_helpers.dataset import MEDSDataset

    example_dir = tmp_path / "example"
    example_dir.mkdir()
    (example_dir / "out_data.yaml").write_text(
        "data/train/0: |-2\n"
        "  subject_id,time,code,numeric_value\n"
        "  1,,GENDER//M,\n"
        "metadata/codes.parquet: |-2\n"
        "  code,description,parent_codes\n"
        "  GENDER//M,,\n"
    )
    (example_dir / "cfg.yaml").write_text("")

    ex = StageExample.from_dir("test", ".", example_dir)
    assert isinstance(ex.want_data, MEDSDataset)


# ---------------------------------------------------------------------------------------------
# check_outputs dispatching through the comparator map
# ---------------------------------------------------------------------------------------------


def test_check_outputs_path_spec_passes_when_parquet_matches(tmp_path: Path):
    spec = _write_yaml_spec(tmp_path / "out_data.yaml", "data/shard.csv: |\n  a\n  1\n  2\n")
    # Create an "actual" output that matches the spec byte-for-byte using yaml_disk directly.
    from yaml_to_disk import yaml_disk

    actual = tmp_path / "actual"
    actual.mkdir()
    yaml_disk(spec, root_dir=actual)

    ex = StageExample(
        stage_name="test",
        want_data=spec,
        suffix_comparators={".csv": lambda a, b, ctx: None},  # accept anything
    )
    ex.check_outputs(actual, is_resolved_dir=True)  # must not raise


def test_check_outputs_path_spec_mismatch_propagates_comparator_error(tmp_path: Path):
    """A comparator-raised AssertionError surfaces cleanly from check_outputs."""
    spec = _write_yaml_spec(tmp_path / "want.yaml", "data/a.csv: |\n  x\n  1\n")
    actual = tmp_path / "actual"
    (actual / "data").mkdir(parents=True)
    (actual / "data" / "a.csv").write_text("x\n9\n")

    def _strict_text_compare(exp_fp: Path, act_fp: Path, ctx: CompareContext) -> None:
        assert exp_fp.read_text() == act_fp.read_text(), f"CSV mismatch at {ctx.rel}"

    ex = StageExample(
        stage_name="test",
        want_data=spec,
        suffix_comparators={".csv": _strict_text_compare},
    )
    with pytest.raises(AssertionError, match="CSV mismatch"):
        ex.check_outputs(actual, is_resolved_dir=True)


def test_check_outputs_path_spec_missing_file_raises(tmp_path: Path):
    spec = _write_yaml_spec(tmp_path / "want.yaml", "data/shard.csv: |\n  a\n  1\n")
    actual = tmp_path / "actual"
    actual.mkdir()

    ex = StageExample(stage_name="test", want_data=spec)
    with pytest.raises(AssertionError, match="not found"):
        ex.check_outputs(actual, is_resolved_dir=True)


def test_unknown_suffix_raises_runtime_error(tmp_path: Path):
    """No comparator for the suffix → explicit RuntimeError pointing at registration."""
    spec = _write_yaml_spec(tmp_path / "want.yaml", "data/x.weird: foo\n")
    actual = tmp_path / "actual"
    (actual / "data").mkdir(parents=True)
    (actual / "data" / "x.weird").write_text("foo\n")

    ex = StageExample(stage_name="test", want_data=spec)
    with pytest.raises(RuntimeError, match=r"No comparator registered for '\.weird'"):
        ex.check_outputs(actual, is_resolved_dir=True)


# ---------------------------------------------------------------------------------------------
# Policy fields: skip lists, tolerate_unexpected
# ---------------------------------------------------------------------------------------------


def test_tolerate_unexpected_false_flags_extra_files(tmp_path: Path):
    spec = _write_yaml_spec(tmp_path / "want.yaml", "data/a.csv: |\n  x\n  1\n")
    actual = tmp_path / "actual"
    (actual / "data").mkdir(parents=True)
    (actual / "data" / "a.csv").write_text("x\n1\n")
    (actual / "data" / "leftover.csv").write_text("y\n9\n")

    ex = StageExample(
        stage_name="test",
        want_data=spec,
        suffix_comparators={".csv": lambda a, b, ctx: None},
    )
    with pytest.raises(AssertionError, match="Unexpected files"):
        ex.check_outputs(actual, is_resolved_dir=True)


def test_tolerate_unexpected_true_ignores_extra_files(tmp_path: Path):
    spec = _write_yaml_spec(tmp_path / "want.yaml", "data/a.csv: |\n  x\n  1\n")
    actual = tmp_path / "actual"
    (actual / "data").mkdir(parents=True)
    (actual / "data" / "a.csv").write_text("x\n1\n")
    (actual / "data" / "leftover.csv").write_text("y\n9\n")

    ex = StageExample(
        stage_name="test",
        want_data=spec,
        suffix_comparators={".csv": lambda a, b, ctx: None},
        tolerate_unexpected=True,
    )
    ex.check_outputs(actual, is_resolved_dir=True)  # must not raise


def test_default_skip_dirs_ignores_hydra_and_logs(tmp_path: Path):
    spec = _write_yaml_spec(tmp_path / "want.yaml", "data/a.csv: |\n  x\n  1\n")
    actual = tmp_path / "actual"
    (actual / "data").mkdir(parents=True)
    (actual / "data" / "a.csv").write_text("x\n1\n")
    # Simulate Hydra-emitted noise alongside the real outputs.
    (actual / ".logs").mkdir()
    (actual / ".logs" / "run.log").write_text("...\n")
    (actual / ".hydra").mkdir()
    (actual / ".hydra" / "config.yaml").write_text("x: 1\n")

    ex = StageExample(
        stage_name="test",
        want_data=spec,
        suffix_comparators={".csv": lambda a, b, ctx: None},
    )
    # .logs and .hydra must be silently ignored even under strict tolerate_unexpected=False.
    ex.check_outputs(actual, is_resolved_dir=True)


def test_custom_skip_files_ignores_named_file(tmp_path: Path):
    spec = _write_yaml_spec(tmp_path / "want.yaml", "data/a.csv: |\n  x\n  1\n")
    actual = tmp_path / "actual"
    (actual / "data").mkdir(parents=True)
    (actual / "data" / "a.csv").write_text("x\n1\n")
    (actual / "data" / "README").write_text("stage notes\n")

    ex = StageExample(
        stage_name="test",
        want_data=spec,
        suffix_comparators={".csv": lambda a, b, ctx: None},
        skip_files=frozenset({"README"}),
    )
    ex.check_outputs(actual, is_resolved_dir=True)


# ---------------------------------------------------------------------------------------------
# Comparator registry: class default + per-instance override
# ---------------------------------------------------------------------------------------------


def test_class_level_comparator_registration(tmp_path: Path):
    """Subclasses register format-specific comparators via SUFFIX_COMPARATORS."""
    calls = {"json": 0}

    def _compare_json_contents(exp_fp, act_fp, ctx: CompareContext) -> None:
        calls["json"] += 1
        # Strip trailing whitespace so we don't trip over yaml_to_disk materialization nuances.
        assert exp_fp.read_text().strip() == act_fp.read_text().strip(), f"JSON mismatch at {ctx.rel}"

    class JSONAwareExample(StageExample):
        SUFFIX_COMPARATORS: ClassVar[dict] = {
            **StageExample.SUFFIX_COMPARATORS,
            ".json": _compare_json_contents,
        }

    # Use yaml_to_disk and a byte-equal actual copy so we don't have to guess at yaml_to_disk's
    # quoting/escaping behavior for JSON literal contents.
    from yaml_to_disk import yaml_disk

    spec = _write_yaml_spec(tmp_path / "want.yaml", "extra/m.json: |\n  manifest line\n")
    actual = tmp_path / "actual"
    actual.mkdir()
    yaml_disk(spec, root_dir=actual)

    ex = JSONAwareExample(stage_name="test", want_data=spec)
    ex.check_outputs(actual, is_resolved_dir=True)
    assert calls["json"] == 1


def test_per_instance_comparator_overrides_class_default(tmp_path: Path):
    """Instance-level suffix_comparators wins over the class-level SUFFIX_COMPARATORS."""
    spec = _write_yaml_spec(tmp_path / "want.yaml", "x.parquet: 0\n")
    actual = tmp_path / "actual"
    actual.mkdir()
    pl.DataFrame({"a": [1, 2]}).write_parquet(actual / "x.parquet")

    # Custom comparator that always says "mismatch" — proves the class-level .parquet default
    # was NOT used.
    def _always_fail(exp_fp, act_fp, ctx):
        raise AssertionError(f"custom comparator: {ctx.rel}")

    ex = StageExample(
        stage_name="test",
        want_data=spec,
        suffix_comparators={".parquet": _always_fail},
    )
    with pytest.raises(AssertionError, match="custom comparator"):
        ex.check_outputs(actual, is_resolved_dir=True)


# ---------------------------------------------------------------------------------------------
# Default parquet comparator behavior
# ---------------------------------------------------------------------------------------------


def test_compare_parquet_read_failure_raises_assertion(tmp_path: Path):
    """Corrupt / unreadable parquet surfaces as AssertionError, not a raw polars exception."""
    (tmp_path / "want.parquet").write_bytes(b"not a parquet file")
    pl.DataFrame({"a": [1]}).write_parquet(tmp_path / "got.parquet")
    ctx = CompareContext(rel=Path("want.parquet"), tolerances={".parquet": {}})
    with pytest.raises(AssertionError, match=r"Failed to read parquet at want\.parquet"):
        _compare_parquet(tmp_path / "want.parquet", tmp_path / "got.parquet", ctx)


def test_from_dir_falls_back_to_path_for_non_meds_metadata(tmp_path: Path):
    """out_metadata.yaml that isn't a MEDS-shaped metadata spec is stored as Path."""
    import yaml

    example_dir = tmp_path / "example"
    example_dir.mkdir()
    # A yaml_to_disk spec that isn't in metadata/codes.parquet shape.
    (example_dir / "out_metadata.yaml").write_text(yaml.dump({"extra/manifest.json": {"v": 1}}))
    (example_dir / "cfg.yaml").write_text(yaml.dump({}))

    ex = StageExample.from_dir("test", ".", example_dir)
    assert isinstance(ex.want_metadata, Path)
    assert ex.want_metadata == example_dir / "out_metadata.yaml"


def test_check_outputs_dispatches_path_spec_on_want_metadata(tmp_path: Path):
    """want_metadata as Path goes through _check_path_spec just like want_data."""
    from yaml_to_disk import yaml_disk

    spec = _write_yaml_spec(tmp_path / "out_metadata.yaml", "extra/m.csv: |\n  x\n  1\n")
    actual = tmp_path / "actual"
    actual.mkdir()
    yaml_disk(spec, root_dir=actual)

    ex = StageExample(
        stage_name="test",
        want_metadata=spec,
        suffix_comparators={".csv": lambda a, b, ctx: None},
    )
    ex.check_outputs(actual, is_resolved_dir=True)  # must not raise


def test_compare_parquet_tolerance_bridge(tmp_path: Path):
    """df_check_kwargs flows through CompareContext.tol_for('.parquet')."""
    # Values differ by more than default atol but within loose rtol — strict would fail, loose passes.
    pl.DataFrame({"a": [1.0]}).write_parquet(tmp_path / "want.parquet")
    pl.DataFrame({"a": [1.001]}).write_parquet(tmp_path / "got.parquet")

    strict_ctx = CompareContext(rel=Path("a"), tolerances={".parquet": {"rel_tol": 0.0, "abs_tol": 0.0}})
    with pytest.raises(AssertionError, match="Parquet mismatch"):
        _compare_parquet(tmp_path / "want.parquet", tmp_path / "got.parquet", strict_ctx)

    loose_ctx = CompareContext(rel=Path("a"), tolerances={".parquet": {"rel_tol": 1e-2, "abs_tol": 1e-2}})
    _compare_parquet(tmp_path / "want.parquet", tmp_path / "got.parquet", loose_ctx)  # no raise


def test_medsdataset_branch_unchanged(tmp_path: Path):
    """Legacy MEDSDataset check_outputs path still works exactly as before."""
    from meds_testing_helpers.dataset import MEDSDataset
    from meds_testing_helpers.static_sample_data import SIMPLE_STATIC_SHARDED_BY_SPLIT

    ds = MEDSDataset.from_yaml(SIMPLE_STATIC_SHARDED_BY_SPLIT)
    ex = StageExample(stage_name="test", want_data=ds)

    output_dir = tmp_path / "cohort"
    (output_dir / "data").mkdir(parents=True)
    ds.write(output_dir)

    ex.check_outputs(output_dir)  # must not raise


# ---------------------------------------------------------------------------------------------
# _resolve_comparators behavior
# ---------------------------------------------------------------------------------------------


def test_resolve_comparators_class_default(tmp_path: Path):
    ex = StageExample(stage_name="test", want_metadata=pl.DataFrame({"code": ["x"]}))
    resolved = ex._resolve_comparators()
    assert ".parquet" in resolved
    assert resolved[".parquet"] is _compare_parquet


def test_resolve_comparators_instance_override(tmp_path: Path):
    def my_parquet(*args, **kwargs):
        pass

    fake: ComparatorFn = my_parquet
    ex = StageExample(
        stage_name="test",
        want_metadata=pl.DataFrame({"code": ["x"]}),
        suffix_comparators={".parquet": fake, ".foo": fake},
    )
    resolved = ex._resolve_comparators()
    assert resolved[".parquet"] is fake
    assert resolved[".foo"] is fake
