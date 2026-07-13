import json
import pandas as pd


class CanonicalMapper:

    def __init__(
        self,
        mapping_file="config/canonical_mapping.json"
    ):

        with open(
            mapping_file,
            "r",
            encoding="utf-8"
        ) as f:

            self.mapping_rules = json.load(f)

    def apply(self, df):

        rename_dict = {}

        for canonical_name, aliases in (
            self.mapping_rules.items()
        ):

            aliases = [
                a.lower()
                for a in aliases
            ]

            for col in df.columns:

                if col.lower() in aliases:

                    rename_dict[col] = (
                        canonical_name
                    )

        df = df.rename(
            columns=rename_dict
        )

        print(
            f"Mapped {len(rename_dict)} columns"
        )

        return df