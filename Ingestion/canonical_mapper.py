import json
import pandas as pd

from Config import settings as _settings

# Sourced from Config.settings like every other path in the pipeline
# (CLEANED_DATA_FILE, RAW_DATA_FILE, etc.), instead of a hardcoded
# relative path that only resolved when the process happened to be
# launched from the project root.
DEFAULT_CANONICAL_MAPPING_FILE = getattr(
    _settings,
    "CANONICAL_MAPPING_FILE",
    None,
)


class CanonicalMapper:

    def __init__(
        self,
        mapping_file=None
    ):

        mapping_file = mapping_file or DEFAULT_CANONICAL_MAPPING_FILE

        if mapping_file is None:
            raise ValueError(
                "No mapping_file provided and Config.settings has no "
                "CANONICAL_MAPPING_FILE defined. Add "
                "CANONICAL_MAPPING_FILE = BASE_DIR / 'Config' / "
                "'canonical_mapper.json' to Config/settings.py."
            )

        with open(
            mapping_file,
            "r",
            encoding="utf-8"
        ) as f:

            self.mapping_rules = json.load(f)

    def map(self, df):

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