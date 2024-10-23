#!/usr/bin/env python
"""TODO."""
from abc import ABC, abstractmethod
from typing import List, Dict
from dataclasses import dataclass
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.utils import hydra_loguru_init


@dataclass
class Vocabulary:
    vocabulary_name: str
    omop_vocabularies: List[str]


class VocabularyMapping(ABC):
    """
    Abstract base class for defining mappings between source and target vocabularies.

    This class provides a structure for mapping concept codes from one vocabulary to
    another, with the actual logic for generating the mappings being implemented
    in subclasses. It initializes with a source vocabulary and a target vocabulary,
    both of which are necessary to create the mappings.

    Attributes:
        vocabulary_cache_dir (Path): The directory where vocabulary data (e.g., concept
        tables, mappings) is cached or stored.

        source_vocabulary (Vocabulary): The vocabulary object representing the source
        vocabulary that contains the concepts to be mapped.

        target_vocabulary (Vocabulary): The vocabulary object representing the target
        vocabulary that the source concepts are mapped to.

    Methods:
        get_code_mappings() -> Dict[str, str]:
            Abstract method that must be implemented by subclasses. It should return a
            dictionary mapping source vocabulary codes (str) to target vocabulary codes (str).
    """

    def __init__(
        self,
        vocabulary_cache_dir: Path,
        source_vocabulary: Vocabulary,
        target_vocabulary: Vocabulary,
    ):
        """
        Initializes the VocabularyMapping class with a source vocabulary, target
        vocabulary, and a directory for caching vocabulary data.

        Args:
            vocabulary_cache_dir (Path): The directory path where vocabulary-related
            data is cached.

            source_vocabulary (Vocabulary): The source vocabulary object that contains
            the concepts to be mapped.

            target_vocabulary (Vocabulary): The target vocabulary object where the
            source concepts will be mapped to.
        """
        self.vocabulary_cache_dir = vocabulary_cache_dir
        self.source_vocabulary = source_vocabulary
        self.target_vocabulary = target_vocabulary

    @abstractmethod
    def get_code_mappings(self) -> Dict[str, str]:
        """
        Abstract method to retrieve the mapping between source and target vocabulary codes.

        Subclasses must implement this method to return a dictionary mapping source
        vocabulary codes (str) to target vocabulary codes (str).

        Returns:
            Dict[str, str]: A dictionary where keys are source vocabulary codes and values
            are the corresponding target vocabulary codes.
        """
        pass


class OmopConceptRelationshipMapping(VocabularyMapping):
    """
    This class defines the mapping between source and target OMOP concept vocabularies
    using relationships defined in the OMOP 'concept' and 'concept_relationship' tables.
    It extends the VocabularyMapping class and provides a method to retrieve mappings
    of source concept codes to target concept codes based on OMOP vocabularies.

    Methods:
        create_hash(concept_ids: List[str]) -> str:
            Creates a unique hash by sorting and concatenating concept IDs into a string.

        get_code_mappings() -> Dict[str, str]:
            Returns a dictionary mapping source OMOP concept codes to target OMOP
            concept codes using the "Maps to" relationships from the OMOP vocabulary.
            The function joins source and target mappings via hashed concept ID lists.
    """

    @staticmethod
    def create_hash(concept_ids: List[str]) -> str:
        """
        Creates a unique hash string from a list of concept IDs by sorting and concatenating
        them with a hyphen ("-").

        Args:
            concept_ids (List[str]): A list of concept IDs to hash.

        Returns:
            str: A hashed string representing the sorted concept IDs concatenated with a hyphen.
        """
        return "-".join(map(str, sorted(concept_ids)))

    def get_code_mappings(self) -> Dict[str, str]:
        """
        Retrieves mappings between source concept codes and target concept codes based on the
        OMOP 'Maps to' relationships. The function first loads concept and concept relationship
        tables, generates mappings for both the source and target OMOP vocabularies, and then
        joins these mappings on hashed concept IDs to produce the final dictionary mapping
        source codes to target codes.

        Returns:
            Dict[str, str]: A dictionary mapping source concept codes (str) to target concept
            codes (str), based on the OMOP "Maps to" relationship.

        Raises:
            Exception: If the vocabulary tables cannot be loaded or the mappings cannot be created.
        """
        source_omop_vocabularies = self.source_vocabulary.omop_vocabularies
        target_omop_vocabularies = self.target_vocabulary.omop_vocabularies

        concept = try_loading_vocabulary_table(self.vocabulary_cache_dir, "concept")
        concept_relationship = try_loading_vocabulary_table(
            self.vocabulary_cache_dir, "concept_relationship"
        )
        concept_id_to_code_mapping = (
            concept.filter(
                pl.col("vocabulary_id").is_in(
                    self.source_vocabulary.omop_vocabularies
                    + self.target_vocabulary.omop_vocabularies
                )
            )
            .with_columns(
                pl.concat_str(["vocabulary_id", "concept_code"], separator="//").alias(
                    "code"
                )
            )
            .select("concept_id", "code")
        )

        source_concept_id_to_mapped_concepts = (
            create_concept_to_mapped_concepts_dict(
                concept, concept_relationship, source_omop_vocabularies
            )
            .with_columns(
                pl.col("mapped_concept_ids")
                .map_elements(self.create_hash, return_dtype=pl.String)
                .alias("hash")
            )
            .join(concept_id_to_code_mapping, on="concept_id")
            .drop(["mapped_concept_ids", "concept_id"])
            .rename({"code": "source_code"})
        )

        target_concept_id_to_mapped_concepts = (
            create_concept_to_mapped_concepts_dict(
                concept, concept_relationship, target_omop_vocabularies
            )
            .with_columns(
                pl.col("mapped_concept_ids")
                .map_elements(self.create_hash, return_dtype=pl.String)
                .alias("hash")
            )
            .join(concept_id_to_code_mapping, on="concept_id")
            .drop(["mapped_concept_ids", "concept_id"])
            .rename({"code": "target_code"})
        )

        return {
            row["source_code"]: row["target_code"]
            for row in source_concept_id_to_mapped_concepts.join(
                target_concept_id_to_mapped_concepts, on="hash"
            )
            .collect()
            .to_dicts()
        }


ICD9 = Vocabulary("ICD9", ["ICD9CM", "ICD9sPCS"])
ICD10 = Vocabulary("ICD10", ["ICD10CM", "ICD10PCS"])

SUPPORTED_VOCABULARIES = [ICD9, ICD10]
SUPPORTED_TRANSLATIONS = {
    (ICD9.vocabulary_name, ICD10.vocabulary_name): OmopConceptRelationshipMapping
}


def try_loading_vocabulary_table(
    vocabulary_cache_dir: Path, vocabulary_table: str
) -> pl.LazyFrame:
    """
    Try loading a vocabulary polars dataframe from vocabulary_cache_dir

    Args:
        vocabulary_cache_dir:
        vocabulary_table:
    Raises:
        FileNotFoundError: If the `vocabulary_cache_dir` does not exist
        pl.exceptions.ComputeError: the concept_relationship dataframe cannot be loaded

    Returns: the concept_relationship dataframe

    """
    if not vocabulary_cache_dir.exists():
        raise FileNotFoundError(
            f"{vocabulary_cache_dir} does not exist, the OMOP concept_relationship table must exist in {vocabulary_cache_dir}"
        )
    try:
        wildcard_path = vocabulary_cache_dir / vocabulary_table / "*.parquet"
        return pl.scan_parquet(wildcard_path)
    except pl.exceptions.ComputeError as e:
        # Handle the error
        raise f"Error loading the concept_relationship parquet files from: {vocabulary_cache_dir}" from e


def create_concept_to_parent_dict(
    concept: pl.LazyFrame,
    concept_relationship: pl.LazyFrame,
    omop_vocabularies: List[str],
) -> pl.LazyFrame:
    """
    Creates a mapping of concept IDs to their parent concept IDs based on the "Is a"
    relationship in the OMOP vocabulary.

    This function filters the provided concept data to include only those belonging to
    the specified OMOP vocabularies. It then joins the concept data with the concept
    relationships to identify parent-child relationships based on the "Is a" relationship.
    The result is a LazyFrame that contains concept IDs and their corresponding parent concept IDs.

    Args:
        concept (pl.LazyFrame): The concept table in lazy format, containing details such as
        concept_id and vocabulary_id.
        concept_relationship (pl.LazyFrame): The concept relationship table in lazy format,
        which contains the relationships between different concepts.
        omop_vocabularies (List[str]): A list of vocabulary IDs representing the OMOP vocabularies
        to be included in the mapping.

    Returns:
        pl.LazyFrame: A LazyFrame containing the mapping of each concept ID to its corresponding
        parent concept ID. The output will have two columns: "concept_id" and "parent_concept_id".
    """
    concept_to_parent_mapping_df = (
        concept.filter(pl.col("vocabulary_id").is_in(omop_vocabularies))
        .select("concept_id")
        .join(concept_relationship, left_on="concept_id", right_on="concept_id_1")
        .filter(pl.col("relationship_id") == "Is a")
        .with_columns(pl.col("concept_id_2").alias("parent_concept_id"))
        .select("concept_id", "parent_concept_id")
    )
    return concept_to_parent_mapping_df


def create_concept_to_mapped_concepts_dict(
    concept: pl.LazyFrame,
    concept_relationship: pl.LazyFrame,
    omop_vocabularies: List[str],
) -> pl.LazyFrame:
    """
    Creates a mapping of concept IDs to their corresponding mapped concept IDs
    based on the "Maps to" relationship in the OMOP vocabulary.

    This function filters the provided concept data to only include those
    belonging to the specified OMOP vocabularies. It then joins the concept
    data with the concept relationships to identify mappings and returns a
    LazyFrame with each concept ID mapped to its unique corresponding
    concept IDs based on the "Maps to" relationship.

    Args:
        concept (pl.LazyFrame): The concept table in lazy format, containing the
        concept details like concept_id and vocabulary_id.
        concept_relationship (pl.LazyFrame): The concept relationship table in
        lazy format, which contains the relationships between different concepts.
        omop_vocabularies (List[str]): A list of vocabulary IDs representing the
        OMOP vocabularies to be included in the mapping.

    Returns:
        pl.LazyFrame: A LazyFrame containing the mapping of each concept ID to its
        unique mapped concept IDs. The output will have two columns: "concept_id"
        and "mapped_concept_ids".
    """
    concept_to_mapped_concept_ids = (
        concept.filter(pl.col("vocabulary_id").is_in(omop_vocabularies))
        .select("concept_id")
        .join(
            concept_relationship,
            how="left",
            left_on="concept_id",
            right_on="concept_id_1",
        )
        .filter(pl.col("relationship_id") == "Maps to")
        .with_columns(
            pl.coalesce(pl.col("concept_id_2"), pl.col("concept_id")).alias(
                "mapped_concept_id"
            )
        )
        .group_by("concept_id")
        .agg(pl.col("mapped_concept_id").unique().alias("mapped_concept_ids"))
    )
    return concept_to_mapped_concept_ids


def get_vocabulary(vocabulary_name: str) -> Vocabulary:
    """
    Retrieve a vocabulary object by its name from the list of supported vocabularies.

    This function searches through the `SUPPORTED_VOCABULARIES` to find a vocabulary
    that matches the provided `vocabulary_name`. If a match is found, the corresponding
    `Vocabulary` object is returned. If no match is found, a `ValueError` is raised
    detailing the unsupported vocabulary name and listing all supported vocabularies
    and translations.

    Args:
        vocabulary_name (str): The name of the vocabulary to retrieve.

    Returns:
        Vocabulary: The corresponding `Vocabulary` object if the name is found.

    Raises:
        ValueError: If `vocabulary_name` is not found in the `SUPPORTED_VOCABULARIES`,
        this exception is raised, along with details about the available supported
        vocabularies and translations.

    Examples:
        >>> vocab = get_vocabulary("ICD9")
        >>> print(vocab)

        >>> get_vocabulary("invalid_vocab")
        Traceback (most recent call last):
        ...
        ValueError: invalid_vocab is not a supported vocabulary
        Supported vocabularies: ['vocab1', 'vocab2']
        Supported translations: ['translation1', 'translation2']
    """
    for vocabulary in SUPPORTED_VOCABULARIES:
        if vocabulary_name.casefold() == vocabulary.vocabulary_name.casefold():
            return vocabulary
    raise ValueError(
        f"{vocabulary_name} is not a supported vocabulary\n"
        f"Supported vocabularies: {SUPPORTED_VOCABULARIES}\n"
        f"Supported translations: {SUPPORTED_TRANSLATIONS}\n"
    )


def get_vocabulary_mapping(
    vocabulary_cache_dir: Path,
    source_vocabulary: Vocabulary,
    target_vocabulary: Vocabulary,
) -> VocabularyMapping:
    """
    Retrieves a vocabulary mapping between a source and target vocabulary.

    This function checks if a translation between the specified source and target
    vocabularies is supported. If supported, it returns the appropriate vocabulary
    mapping from the cache or generates it if necessary. If the translation is not
    supported, a ValueError is raised, listing the available supported translations.

    Args:
        vocabulary_cache_dir (Path): Directory where vocabulary mappings are cached.
        source_vocabulary (Vocabulary): The source vocabulary for translation.
        target_vocabulary (Vocabulary): The target vocabulary for translation.

    Returns:
        VocabularyMapping: A mapping object that provides translation between
        the source and target vocabulary.

    Raises:
        ValueError: If there is no supported translation between the source and
        target vocabularies.
    """
    vocabulary_tuple = (
        source_vocabulary.vocabulary_name,
        target_vocabulary.vocabulary_name,
    )
    if vocabulary_tuple in SUPPORTED_TRANSLATIONS:
        return SUPPORTED_TRANSLATIONS[vocabulary_tuple](
            vocabulary_cache_dir, source_vocabulary, target_vocabulary
        )
    raise ValueError(f"Supported translations: {SUPPORTED_TRANSLATIONS}\n")


def add_metadata_translation(
    code_metadata: pl.DataFrame,
    source_vocabulary: Vocabulary,
    target_vocabulary: Vocabulary,
    translation_col: str,
    vocabulary_cache_dir: Path,
) -> pl.DataFrame:
    """Validate the code metadata has the requisite columns and is unique.

    Args:
        code_metadata: Metadata about the codes in the MEDS dataset, with a column `code` and a collection
            of code modifier columns.
        source_vocabulary: the source vocabulary to be translated.
        target_vocabulary: the preferred target vocabulary to be translated to.
        translation_col: The names of column that stores the translated `code`.
        vocabulary_cache_dir: the cache folder that contains the OMOP concept_relationship data.

    Raises:
        KeyError: If the `code_metadata` dataset does not contain the specified `code_modifiers` or `code`
            columns.
        ValueError: If the `code_metadata` dataset is not unique on the `code` and `code_modifiers` columns.

    Examples:
        >>> code_metadata = pl.DataFrame({
        ...     "code": ["ICD9CM//V10.60", "ICD9CM//V15", "None_ICD9_code"],
        ... })
        >>> source_vocabulary = get_vocabulary("ICD9")
        >>> target_vocabulary = get_vocabulary("ICD10")
        >>> vocabulary_cache_dir = Path("vocabulary_cache_dir")
        >>> add_metadata_translation(
        ...     code_metadata,
        ...     source_vocabulary,
        ...     target_vocabulary,
        ...     "translated_col",
        ...     vocabulary_cache_dir
        ...)
        shape: (3, 2)
        ┌────────────────┬─────────────────┐
        │ code           ┆ translated_col  │
        │ ---            ┆ ---             │
        │ str            ┆ str             │
        ╞════════════════╪═════════════════╡
        │ ICD9CM//V10.60 ┆ ICD10CM//Z97.15 │
        │ ICD9CM//V15    ┆ ICD10CM//Z97.15 │
        │ test           ┆ test            │
        └────────────────┴─────────────────┘
    """

    vocabulary_mapping = get_vocabulary_mapping(
        vocabulary_cache_dir, source_vocabulary, target_vocabulary
    )
    mapping = vocabulary_mapping.get_code_mappings()
    translated_col_expr = pl.col("code").replace_strict(
        mapping, return_dtype=pl.String, default=None
    )
    return code_metadata.with_columns(
        pl.coalesce(translated_col_expr, pl.col("code"))
        .cast(pl.String)
        .alias(translation_col)
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
    vocabulary_cache_dir = Path(
        cfg.stage_cfg.get("vocabulary_cache_dir", "vocabulary_cache")
    )
    # If the vocabulary cache dir is an absolute path, we will use it as is otherwise we assume it's a relative path w.r.t metadata_input_dir
    if not vocabulary_cache_dir.is_absolute():
        vocabulary_cache_dir = metadata_input_dir / vocabulary_cache_dir

    output_dir = Path(cfg.stage_cfg.reducer_output_dir)
    code_metadata = pl.read_parquet(
        metadata_input_dir / "codes.parquet", use_pyarrow=True
    )
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
