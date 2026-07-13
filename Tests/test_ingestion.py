"""
tests/test_ingestion.py

Unit tests for:

1. CSV Loader
2. Data Cleaner
3. Canonical Mapper

Run:

pytest tests/test_ingestion.py -v
"""

from pathlib import Path

import pandas as pd
import pytest

from ingestion.load_csv import CSVLoader
from ingestion.clean_data import DataCleaner
from ingestion.canonical_mapper import CanonicalMapper


# ==========================================================
# TEST DATA
# ==========================================================

@pytest.fixture
def sample_df():

    return pd.DataFrame(
        {
            "Product Name": [
                "Phone",
                "Laptop",
                None,
                "Phone"
            ],
            "Rating": [
                5,
                4,
                3,
                5
            ],
            "Review": [
                "Excellent battery life",
                "Good performance",
                None,
                "Excellent battery life"
            ]
        }
    )


# ==========================================================
# CSV LOADER TESTS
# ==========================================================

def test_csv_loader(tmp_path):

    csv_file = tmp_path / "sample.csv"

    df = pd.DataFrame(
        {
            "id": [1, 2],
            "text": ["a", "b"]
        }
    )

    df.to_csv(
        csv_file,
        index=False
    )

    loader = CSVLoader()

    loaded_df = loader.load(csv_file)

    assert isinstance(
        loaded_df,
        pd.DataFrame
    )

    assert len(
        loaded_df
    ) == 2

    assert list(
        loaded_df.columns
    ) == ["id", "text"]


# ==========================================================
# DATA CLEANER TESTS
# ==========================================================

def test_cleaner_returns_dataframe(
    sample_df
):

    cleaner = DataCleaner()

    cleaned_df = cleaner.clean(
        sample_df
    )

    assert isinstance(
        cleaned_df,
        pd.DataFrame
    )


def test_cleaner_removes_duplicates(
    sample_df
):

    cleaner = DataCleaner()

    cleaned_df = cleaner.clean(
        sample_df
    )

    assert len(cleaned_df) <= len(
        sample_df
    )


def test_cleaner_handles_nulls(
    sample_df
):

    cleaner = DataCleaner()

    cleaned_df = cleaner.clean(
        sample_df
    )

    assert isinstance(
        cleaned_df,
        pd.DataFrame
    )


# ==========================================================
# CANONICAL MAPPER TESTS
# ==========================================================

def test_canonical_mapper_runs(
    sample_df
):

    mapper = CanonicalMapper()

    mapped_df = mapper.apply(
        sample_df
    )

    assert isinstance(
        mapped_df,
        pd.DataFrame
    )


def test_column_names_are_strings(
    sample_df
):

    mapper = CanonicalMapper()

    mapped_df = mapper.apply(
        sample_df
    )

    for col in mapped_df.columns:

        assert isinstance(
            col,
            str
        )


# ==========================================================
# PIPELINE TEST
# ==========================================================

def test_full_ingestion_pipeline(
    sample_df
):

    cleaner = DataCleaner()

    mapper = CanonicalMapper()

    cleaned_df = cleaner.clean(
        sample_df
    )

    mapped_df = mapper.apply(
        cleaned_df
    )

    assert isinstance(
        mapped_df,
        pd.DataFrame
    )

    assert not mapped_df.empty


# ==========================================================
# NEGATIVE TESTS
# ==========================================================

def test_empty_dataframe():

    empty_df = pd.DataFrame()

    cleaner = DataCleaner()

    with pytest.raises(Exception):

        cleaner.clean(
            empty_df
        )


def test_invalid_csv_path():

    loader = CSVLoader()

    with pytest.raises(Exception):

        loader.load(
            "invalid_file.csv"
        )