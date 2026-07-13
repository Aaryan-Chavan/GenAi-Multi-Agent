# ingestion/clean_data.py

import pandas as pd
import numpy as np


class DataCleaner:

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:

        print("\nStarting data cleaning...")

        # --------------------------------
        # Standardize column names
        # --------------------------------

        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace(r"[^\w]", "", regex=True)
        )

        # --------------------------------
        # Remove duplicate rows
        # --------------------------------

        before = len(df)

        df = df.drop_duplicates()

        print(
            f"Duplicates removed: {before - len(df)}"
        )

        # --------------------------------
        # Remove fully empty rows
        # --------------------------------

        df = df.dropna(how="all")

        # --------------------------------
        # Remove fully empty columns
        # --------------------------------

        df = df.dropna(axis=1, how="all")

        # --------------------------------
        # Strip text columns
        # --------------------------------

        object_cols = df.select_dtypes(
            include=["object"]
        ).columns

        for col in object_cols:

            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .replace("nan", np.nan)
            )

        # --------------------------------
        # Missing value handling
        # --------------------------------

        for col in df.columns:

            if pd.api.types.is_numeric_dtype(
                df[col]
            ):

                median = df[col].median()

                df[col] = df[col].fillna(
                    median
                )

            else:

                mode = (
                    df[col].mode()[0]
                    if not df[col].mode().empty
                    else "unknown"
                )

                df[col] = df[col].fillna(
                    mode
                )

        # --------------------------------
        # Remove non-printable chars
        # --------------------------------

        for col in object_cols:

            df[col] = df[col].str.replace(
                r"[^\x20-\x7E]",
                "",
                regex=True
            )

        print(
            f"Cleaned Shape: {df.shape}"
        )

        return df