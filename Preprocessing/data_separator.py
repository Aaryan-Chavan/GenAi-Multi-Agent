# Preprocessing/data_separator.py

from pathlib import Path
import os
import pandas as pd


class DataSeparator:

    def __init__(
        self,
        output_dir: str = "data/processed"
    ):

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    # ==========================================
    # SEPARATE DATA
    # ==========================================

    def separate(
        self,
        df: pd.DataFrame,
        schema_result: dict
    ) -> dict:

        structured_cols = schema_result.get(
            "structured_columns",
            []
        )

        unstructured_cols = schema_result.get(
            "unstructured_columns",
            []
        )

        metadata_cols = schema_result.get(
            "metadata_columns",
            []
        )

        # ----------------------------------
        # Ensure record_id exists everywhere
        # ----------------------------------

        if "record_id" in df.columns:

            if "record_id" not in unstructured_cols:
                unstructured_cols.insert(
                    0,
                    "record_id"
                )

            if "record_id" not in metadata_cols:
                metadata_cols.insert(
                    0,
                    "record_id"
                )

            if "record_id" not in structured_cols:
                structured_cols.insert(
                    0,
                    "record_id"
                )

        structured_df = self._safe_select(
            df,
            structured_cols
        )

        unstructured_df = self._safe_select(
            df,
            unstructured_cols
        )

        metadata_df = self._safe_select(
            df,
            metadata_cols
        )

        return {
            "structured_df": structured_df,
            "unstructured_df": unstructured_df,
            "metadata_df": metadata_df
        }

    # ==========================================
    # SAVE
    # ==========================================

    def save(
        self,
        separated_data: dict
    ):

        structured_df = separated_data[
            "structured_df"
        ]

        unstructured_df = separated_data[
            "unstructured_df"
        ]

        metadata_df = separated_data[
            "metadata_df"
        ]

        structured_path = (
            self.output_dir /
            "structured.csv"
        )

        unstructured_path = (
            self.output_dir /
            "unstructured.csv"
        )

        metadata_path = (
            self.output_dir /
            "metadata.csv"
        )

        self._safe_write(
            structured_df,
            structured_path
        )

        self._safe_write(
            unstructured_df,
            unstructured_path
        )

        self._safe_write(
            metadata_df,
            metadata_path
        )

        print(
            "\nSeparated datasets saved:"
        )

        print(
            f"Structured  -> {structured_path}"
        )

        print(
            f"Unstructured -> {unstructured_path}"
        )

        print(
            f"Metadata -> {metadata_path}"
        )

    # ==========================================
    # SUMMARY
    # ==========================================

    def summary(
        self,
        separated_data: dict
    ):

        structured_df = separated_data[
            "structured_df"
        ]

        unstructured_df = separated_data[
            "unstructured_df"
        ]

        metadata_df = separated_data[
            "metadata_df"
        ]

        print("\n" + "=" * 60)
        print("DATA SEPARATION SUMMARY")
        print("=" * 60)

        print(
            f"\nStructured Columns ({len(structured_df.columns)}):"
        )

        for col in structured_df.columns:
            print(f"  • {col}")

        print(
            f"\nUnstructured Columns ({len(unstructured_df.columns)}):"
        )

        for col in unstructured_df.columns:
            print(f"  • {col}")

        print(
            f"\nMetadata Columns ({len(metadata_df.columns)}):"
        )

        for col in metadata_df.columns:
            print(f"  • {col}")

        print("\n" + "=" * 60)

    # ==========================================
    # SAFE COLUMN SELECTION
    # ==========================================

    @staticmethod
    def _safe_select(
        df: pd.DataFrame,
        columns: list
    ) -> pd.DataFrame:

        existing_columns = [

            col

            for col in columns

            if col in df.columns

        ]

        return df[
            existing_columns
        ].copy()

    # ==========================================
    # SAFE CSV WRITE
    # ==========================================

    @staticmethod
    def _safe_write(
        df: pd.DataFrame,
        path: Path
    ):

        try:

            temp_path = str(path) + ".tmp"

            df.to_csv(
                temp_path,
                index=False
            )

            os.replace(
                temp_path,
                path
            )

        except PermissionError:

            raise PermissionError(
                f"\nCannot write file:\n{path}\n\n"
                f"Close Excel or any program using the file."
            )