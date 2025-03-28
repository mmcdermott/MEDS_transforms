"""Tests the fit vocabulary indices script."""

from pathlib import Path

from tests import FIT_VOCABULARY_INDICES_SCRIPT
from tests.transform_tester_base import MEDS_CODE_METADATA_SCHEMA
from tests.utils import MEDS_transforms_pipeline_tester, parse_shards_yaml

WANT_OUTPUTS = parse_shards_yaml(
    """
metadata/codes: |-2
  code,description,parent_codes,code/vocab_index
  EYE_COLOR//BLUE,"Blue Eyes. Less common than brown.",,1
  EYE_COLOR//BROWN,"Brown Eyes. The most common eye color.",,2
  EYE_COLOR//HAZEL,"Hazel eyes. These are uncommon",,3
  HR,"Heart Rate",LOINC/8867-4,4
  TEMP,"Body Temperature",LOINC/8310-5,5
""",
    **MEDS_CODE_METADATA_SCHEMA,
)


def test_fit_vocabulary_indices_with_default_stage_config(simple_static_MEDS: Path):
    MEDS_transforms_pipeline_tester(
        script=FIT_VOCABULARY_INDICES_SCRIPT,
        stage_name="fit_vocabulary_indices",
        stage_kwargs={},
        want_outputs=WANT_OUTPUTS,
        input_dir=simple_static_MEDS,
    )

    MEDS_transforms_pipeline_tester(
        script=FIT_VOCABULARY_INDICES_SCRIPT,
        stage_name="fit_vocabulary_indices",
        stage_kwargs={"ordering_method": "file"},
        want_outputs=WANT_OUTPUTS,
        input_dir=simple_static_MEDS,
        should_error=True,
    )
