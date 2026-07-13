# ingestion/load_csv.py

import pandas as pd
from pathlib import Path


class CSVLoader:

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

    def load(self) -> pd.DataFrame:

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"File not found: {self.file_path}"
            )

        print(f"\nLoading: {self.file_path}")

        df = pd.read_csv(
            self.file_path,
            low_memory=False
        )

        print(
            f"Rows: {len(df)} | Columns: {len(df.columns)}"
        )

        return df