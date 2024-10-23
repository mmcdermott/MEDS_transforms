from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass
class Vocabulary:
    vocabulary_name: str
    omop_vocabularies: list[str]


class VocabularyMapping(ABC):
    """Abstract base class for defining mappings between source and target vocabularies.

    This class provides a structure for mapping concept codes from one vocabulary to
    another, with the actual logic for generating the mappings being implemented
    in subclasses. It initializes with a source vocabulary and a target vocabulary,
    both of which are necessary to create the mappings.

    Attributes:
        vocabulary_cache_dir (Path):
            The directory where vocabulary data (e.g., concept tables, mappings) is cached or stored.

        source_vocabulary (Vocabulary):
            The vocabulary object representing the source vocabulary that contains the concepts to be mapped.

        target_vocabulary (Vocabulary):
            The vocabulary object representing the target vocabulary that the source concepts are mapped to.

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
        """Initializes the VocabularyMapping class with a source vocabulary, target vocabulary, and a
        directory for caching vocabulary data.

        Args:
            vocabulary_cache_dir (Path):
                The directory path where vocabulary-related data is cached.

            source_vocabulary (Vocabulary):
                The source vocabulary object that contains the concepts to be mapped.

            target_vocabulary (Vocabulary):
                The target vocabulary object where the source concepts will be mapped to.
        """
        self.vocabulary_cache_dir = vocabulary_cache_dir
        self.source_vocabulary = source_vocabulary
        self.target_vocabulary = target_vocabulary

    @abstractmethod
    def get_code_mappings(self) -> dict[str, str]:
        """Abstract method to retrieve the mapping between source and target vocabulary codes.

        Subclasses must implement this method to return a dictionary mapping source
        vocabulary codes (str) to target vocabulary codes (str).

        Returns:
            Dict[str, str]: A dictionary where keys are source vocabulary codes and values
            are the corresponding target vocabulary codes.
        """


class OmopConceptRelationshipMapping(VocabularyMapping):
    """This class defines the mapping between source and target OMOP concept vocabularies using relationships
    defined in the OMOP 'concept' and 'concept_relationship' tables. It extends the VocabularyMapping class
    and provides a method to retrieve mappings of source concept codes to target concept codes based on OMOP
    vocabularies.

    Methods:
        create_hash(concept_ids: List[str]) -> str:
            Creates a unique hash by sorting and concatenating concept IDs into a string.

        get_code_mappings() -> Dict[str, str]:
            Returns a dictionary mapping source OMOP concept codes to target OMOP
            concept codes using the "Maps to" relationships from the OMOP vocabulary.
            The function joins source and target mappings via hashed concept ID lists.
    """

    @staticmethod
    def create_hash(concept_ids: list[str]) -> str:
        """Creates a unique hash string from a list of concept IDs by sorting and concatenating them with a
        hyphen ("-").

        Args:
            concept_ids (List[str]): A list of concept IDs to hash.

        Returns:
            str: A hashed string representing the sorted concept IDs concatenated with a hyphen.
        """
        return "-".join(map(str, sorted(concept_ids)))

    def get_code_mappings(self) -> dict[str, str]:
        """Retrieves mappings between source concept codes and target concept codes based on the OMOP 'Maps
        to' relationships. The function first loads concept and concept relationship tables, generates
        mappings for both the source and target OMOP vocabularies, and then joins these mappings on hashed
        concept IDs to produce the final dictionary mapping source codes to target codes.

        Returns:
            Dict[str, str]: A dictionary mapping source concept codes (str) to target concept
            codes (str), based on the OMOP "Maps to" relationship.

        Raises:
            Exception: If the vocabulary tables cannot be loaded or the mappings cannot be created.
        """
        source_omop_vocabularies = self.source_vocabulary.omop_vocabularies
        target_omop_vocabularies = self.target_vocabulary.omop_vocabularies

        concept = try_loading_vocabulary_table(self.vocabulary_cache_dir, "concept")
        concept_relationship = try_loading_vocabulary_table(self.vocabulary_cache_dir, "concept_relationship")
        concept_id_to_code_mapping = (
            concept.filter(
                pl.col("vocabulary_id").is_in(
                    self.source_vocabulary.omop_vocabularies + self.target_vocabulary.omop_vocabularies
                )
            )
            .with_columns(pl.concat_str(["vocabulary_id", "concept_code"], separator="//").alias("code"))
            .select("concept_id", "code")
        )

        source_concept_id_to_mapped_concepts = (
            create_concept_to_mapped_concepts_dict(concept, concept_relationship, source_omop_vocabularies)
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
            create_concept_to_mapped_concepts_dict(concept, concept_relationship, target_omop_vocabularies)
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


def try_loading_vocabulary_table(vocabulary_cache_dir: Path, vocabulary_table: str) -> pl.LazyFrame:
    """Try loading a vocabulary polars dataframe from vocabulary_cache_dir.

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
            f"{vocabulary_cache_dir} does not exist, "
            f"the OMOP concept_relationship table must exist in {vocabulary_cache_dir}"
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
    omop_vocabularies: list[str],
) -> pl.LazyFrame:
    """Creates a mapping of concept IDs to their parent concept IDs based on the "Is a" relationship in the
    OMOP vocabulary.

    This function filters the provided concept data to include only those belonging to
    the specified OMOP vocabularies. It then joins the concept data with the concept
    relationships to identify parent-child relationships based on the "Is a" relationship.
    The result is a LazyFrame that contains concept IDs and their corresponding parent concept IDs.

    Args:
        concept (pl.LazyFrame):
            The concept table in lazy format, containing details such as concept_id
            and vocabulary_id.

        concept_relationship (pl.LazyFrame):
            The concept relationship table in lazy format, which contains the relationships
            between different concepts.

        omop_vocabularies (List[str]):
            A list of vocabulary IDs representing the OMOP vocabularies to be included
            in the mapping.

    Returns:
        pl.LazyFrame:
            A LazyFrame containing the mapping of each concept ID to its corresponding
            parent concept ID. The output will have two columns: "concept_id" and
            "parent_concept_id".
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
    omop_vocabularies: list[str],
) -> pl.LazyFrame:
    """Creates a mapping of concept IDs to their corresponding mapped concept IDs based on the "Maps to"
    relationship in the OMOP vocabulary.

    This function filters the provided concept data to include only those
    belonging to the specified OMOP vocabularies. It then joins the concept
    data with the concept relationships to identify mappings and returns a
    LazyFrame with each concept ID mapped to its unique corresponding
    concept IDs based on the "Maps to" relationship.

    Args:
        concept (pl.LazyFrame):
            The concept table in lazy format, containing details such as concept_id
            and vocabulary_id.

        concept_relationship (pl.LazyFrame):
            The concept relationship table in lazy format, which contains the
            relationships between different concepts.

        omop_vocabularies (List[str]):
            A list of vocabulary IDs representing the OMOP vocabularies to be included
            in the mapping.

    Returns:
        pl.LazyFrame:
            A LazyFrame containing the mapping of each concept ID to its unique mapped
            concept IDs. The output will have two columns: "concept_id" and
            "mapped_concept_ids".
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
        .with_columns(pl.coalesce(pl.col("concept_id_2"), pl.col("concept_id")).alias("mapped_concept_id"))
        .group_by("concept_id")
        .agg(pl.col("mapped_concept_id").unique().alias("mapped_concept_ids"))
    )
    return concept_to_mapped_concept_ids
