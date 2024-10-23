#!/usr/bin/env python
"""TODO."""
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.utils import hydra_loguru_init
from MEDS_transforms.vocabulary_mapping import OmopConceptRelationshipMapping, Vocabulary, VocabularyMapping

ICD9 = Vocabulary("ICD9", ["ICD9CM", "ICD9sPCS"])
ICD10 = Vocabulary("ICD10", ["ICD10CM", "ICD10PCS"])

SUPPORTED_VOCABULARIES = [ICD9, ICD10]
# TODO: might change this to Dict[Tuple[str, str], List[VocabularyMapping]] as there could be multiple mapping
#  strategies to convert from the source to target vocabularies
SUPPORTED_TRANSLATIONS = {(ICD9.vocabulary_name, ICD10.vocabulary_name): OmopConceptRelationshipMapping}


def get_vocabulary(vocabulary_name: str) -> Vocabulary:
    """Retrieve a vocabulary object by its name from the list of supported vocabularies.

    This function searches through the `SUPPORTED_VOCABULARIES` to find a vocabulary
    that matches the provided `vocabulary_name`. If a match is found, the corresponding
    `Vocabulary` object is returned. If no match is found, a `ValueError` is raised
    detailing the unsupported vocabulary name and listing all supported vocabularies
    and translations.

    Args:
        vocabulary_name (str):
            The name of the vocabulary to retrieve.

    Returns:
        Vocabulary:
            The corresponding `Vocabulary` object if the name is found.

    Raises:
        ValueError:
            If `vocabulary_name` is not found in the `SUPPORTED_VOCABULARIES`,
        this exception is raised, along with details about the available supported
        vocabularies and translations.

    Examples:
        >>> vocab = get_vocabulary("ICD9")
        >>> print(vocab)
        Vocabulary(vocabulary_name='ICD9', omop_vocabularies=['ICD9CM', 'ICD9sPCS'])
        >>> vocab = get_vocabulary("invalid_vocabulary")
        Traceback (most recent call last):
        ...
        ValueError: invalid_vocabulary is not a supported vocabulary
        Supported vocabularies: ['ICD9', 'ICD10']
        Supported translations: from ICD9 to ICD10
        <BLANKLINE>
    """
    for vocabulary in SUPPORTED_VOCABULARIES:
        if vocabulary_name.casefold() == vocabulary.vocabulary_name.casefold():
            return vocabulary
    raise ValueError(
        f"{vocabulary_name} is not a supported vocabulary\n"
        f"Supported vocabularies: {[_.vocabulary_name for _ in SUPPORTED_VOCABULARIES]}\n"
        f"Supported translations: "
        f"{'\n'.join([f'from {s} to {t}' for s, t in SUPPORTED_TRANSLATIONS.keys()])}\n"
    )


def get_vocabulary_mapping(
    vocabulary_cache_dir: Path,
    source_vocabulary: Vocabulary,
    target_vocabulary: Vocabulary,
) -> VocabularyMapping:
    """Retrieves a vocabulary mapping between a source and target vocabulary.

    This function checks if a translation between the specified source and target
    vocabularies is supported. If supported, it returns the appropriate vocabulary
    mapping from the cache or generates it if necessary. If the translation is not
    supported, a ValueError is raised, listing the available supported translations.

    Args:
        vocabulary_cache_dir (Path):
            Directory where vocabulary mappings are cached.
        source_vocabulary (Vocabulary):
            The source vocabulary for translation.
        target_vocabulary (Vocabulary):
            The target vocabulary for translation.

    Returns:
        VocabularyMapping:
            A mapping object that provides translation between
        the source and target vocabulary.

    Raises:
        ValueError: If there is no supported translation between the source and
        target vocabularies.

     Examples:
        >>> from pathlib import Path
        >>> from MEDS_transforms.vocabulary_mapping import Vocabulary, VocabularyMapping
        >>> from MEDS_transforms.vocabulary_mapping import OmopConceptRelationshipMapping
        >>> source_vocab = Vocabulary("ICD9", ["ICD9CM"])
        >>> target_vocab = Vocabulary("ICD10", ["ICD10CM"])
        >>> vocab_cache_dir = Path("/mock/cache/dir")
        >>> mapping = get_vocabulary_mapping(vocab_cache_dir, source_vocab, target_vocab)
        >>> isinstance(mapping, OmopConceptRelationshipMapping)
        True
        >>> unsupported_vocab = Vocabulary("SNOMED", ["SNOMEDCT"])
        >>> get_vocabulary_mapping(vocab_cache_dir, unsupported_vocab, target_vocab)
        Traceback (most recent call last):
        ...
        ValueError: Supported translations: from ICD9 to ICD10
        <BLANKLINE>
    """
    vocabulary_tuple = (
        source_vocabulary.vocabulary_name,
        target_vocabulary.vocabulary_name,
    )
    if vocabulary_tuple in SUPPORTED_TRANSLATIONS:
        return SUPPORTED_TRANSLATIONS[vocabulary_tuple](
            vocabulary_cache_dir, source_vocabulary, target_vocabulary
        )
    raise ValueError(
        f"Supported translations: "
        f"{'\n'.join([f'from {s} to {t}' for s, t in SUPPORTED_TRANSLATIONS.keys()])}\n"
    )


def add_metadata_translation(
    code_metadata: pl.DataFrame,
    source_vocabulary: Vocabulary,
    target_vocabulary: Vocabulary,
    translation_col: str,
    vocabulary_cache_dir: Path,
) -> pl.DataFrame:
    """Validate the code metadata has the requisite columns and is unique.

    Args:
        code_metadata (pl.DataFrame): Metadata about the codes in the MEDS dataset,
            with a column `code` and a collection of code modifier columns.
        source_vocabulary (Vocabulary): The source vocabulary to be translated.
        target_vocabulary (Vocabulary): The preferred target vocabulary to be translated to.
        translation_col (str): The name of the column that stores the translated `code`.
        vocabulary_cache_dir (Path): The cache folder that contains the OMOP concept_relationship data.

    Returns:
        pl.DataFrame: A DataFrame with the translated codes.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> import polars as pl
        >>> from MEDS_transforms.vocabulary_mapping import Vocabulary
        >>> source_vocab = Vocabulary("ICD9", ["ICD9CM"])
        >>> target_vocab = Vocabulary("ICD10", ["ICD10CM"])
        >>> code_metadata = pl.DataFrame({
        ...     "code": ["ICD9CM//V9.60", "ICD9CM//V15", "None_ICD9_code"],
        ... })
        >>> with tempfile.TemporaryDirectory() as tmpdirname:
        ...     temp_dir = Path(tmpdirname)
        ...     concept_df = pl.DataFrame({
        ...         "concept_id": [1, 2, 3],
        ...         "concept_code": ["V9.60", "V10.60", "SNOMED_code"],
        ...         "vocabulary_id": ["ICD9CM", "ICD10CM", "SNOMED"]
        ...     })
        ...     concept_relationship_df = pl.DataFrame({
        ...         "concept_id_1": [1, 2, 3],
        ...         "concept_id_2": [3, 3, 3],
        ...         "relationship_id": ["Maps to", "Maps to", "Maps to"]
        ...     })
        ...     (temp_dir / "concept").mkdir(exist_ok=True)
        ...     (temp_dir / "concept_relationship").mkdir(exist_ok=True)
        ...     concept_df.write_parquet(str(temp_dir / "concept" / "data.parquet"))
        ...     concept_relationship_df.write_parquet(
        ...         str(temp_dir / "concept_relationship" / "data.parquet")
        ...     )
        ...     updated_code_metadata = add_metadata_translation(
        ...         code_metadata, source_vocab, target_vocab, "translated", temp_dir
        ...     )
        >>> updated_code_metadata
        shape: (3, 2)
        ┌────────────────┬─────────────────┐
        │ code           ┆ translated      │
        │ ---            ┆ ---             │
        │ str            ┆ str             │
        ╞════════════════╪═════════════════╡
        │ ICD9CM//V9.60  ┆ ICD10CM//V10.60 │
        │ ICD9CM//V15    ┆ ICD9CM//V15     │
        │ None_ICD9_code ┆ None_ICD9_code  │
        └────────────────┴─────────────────┘
    """
    vocabulary_mapping = get_vocabulary_mapping(vocabulary_cache_dir, source_vocabulary, target_vocabulary)
    mapping = vocabulary_mapping.get_code_mappings()
    translated_col_expr = pl.col("code").replace_strict(mapping, return_dtype=pl.String, default=None)
    return code_metadata.with_columns(
        pl.coalesce(translated_col_expr, pl.col("code")).cast(pl.String).alias(translation_col)
    )


@hydra.main(
    version_base=None,
    config_path=str(PREPROCESS_CONFIG_YAML.parent),
    config_name=PREPROCESS_CONFIG_YAML.stem,
)
def main(cfg: DictConfig):
    """TODO."""

    hydra_loguru_init()

    logger.info(
        f"Running with config:\n{OmegaConf.to_yaml(cfg)}\n"
        f"Stage: {cfg.stage}\n\n"
        f"Stage config:\n{OmegaConf.to_yaml(cfg.stage_cfg)}"
    )

    metadata_input_dir = Path(cfg.stage_cfg.metadata_input_dir)
    vocabulary_cache_dir = Path(cfg.stage_cfg.get("vocabulary_cache_dir", "vocabulary_cache"))
    # If the vocabulary cache dir is an absolute path, we will use it as is otherwise we assume
    # it's a relative path w.r.t metadata_input_dir
    if not vocabulary_cache_dir.is_absolute():
        vocabulary_cache_dir = metadata_input_dir / vocabulary_cache_dir

    output_dir = Path(cfg.stage_cfg.reducer_output_dir)
    code_metadata = pl.read_parquet(metadata_input_dir / "codes.parquet", use_pyarrow=True)
    source_vocabulary = get_vocabulary(cfg.stage_cfg.get("source_vocabulary", "ICD9"))
    target_vocabulary = get_vocabulary(cfg.stage_cfg.get("source_vocabulary", "ICD10"))
    logger.info("Adding code translation.")
    code_metadata = add_metadata_translation(
        code_metadata,
        source_vocabulary,
        target_vocabulary,
        cfg.stage_cfg.get("translation_col", "translated"),
        vocabulary_cache_dir,
    )

    output_fp = output_dir / "codes.parquet"
    output_fp.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Indices assigned. Writing to {output_fp}")

    code_metadata.write_parquet(output_fp, use_pyarrow=True)

    logger.info(f"Done with {cfg.stage}")


if __name__ == "__main__":  # pragma: no cover
    main()
