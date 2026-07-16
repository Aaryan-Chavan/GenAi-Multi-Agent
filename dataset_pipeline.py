import time
import pandas as pd

# ==========================================================
# CONFIG
# ==========================================================

from Config.settings import (
    RAW_DATA_FILE,
    CLEANED_DATA_FILE,
    MAPPED_DATA_FILE,
    ENABLE_QUANTIZATION,
    EMBEDDINGS_DIR,
    EMBEDDINGS_FILE,
    QDRANT_BATCH_SIZE,
    STRUCTURED_TABLE,
    METADATA_TABLE,
    INTELLIGENCE_TABLE,
)

# ==========================================================
# INGESTION
# ==========================================================

from Ingestion.load_csv import CSVLoader
from Ingestion.clean_data import DataCleaner
from Ingestion.canonical_mapper import CanonicalMapper

# ==========================================================
# PREPROCESSING
# ==========================================================

from Preprocessing.schema_analyzer import SchemaAnalyzer
from Preprocessing.data_separator import DataSeparator
from Preprocessing.chunking import TextChunker
from Preprocessing.precomputed_intelligence import PrecomputedIntelligence

# ==========================================================
# EMBEDDINGS
# ==========================================================

from Embeddings.embedding_generator import EmbeddingGenerator
from Embeddings.quantization import EmbeddingQuantizer

# ==========================================================
# STORAGE
# ==========================================================

from Storage.duckdb_loader import DuckDBLoader
from Storage.qdrant_loader import QdrantLoader, detect_primary_key


# ==========================================================
# TIMER
# ==========================================================

def stage(name):

    class Timer:

        def __enter__(self):
            self.start = time.time()
            print("\n" + "=" * 80)
            print(name)
            print("=" * 80)
            return self

        def __exit__(self, *args):
            elapsed = time.time() - self.start
            print(f"\nCompleted in {elapsed:.2f} sec")

    return Timer()


# ==========================================================
# HELPERS
# ==========================================================

def find_embedding_files():
    """Return (embedding_path, metadata_path) or (None, None)."""

    embedding_file = EMBEDDINGS_DIR / "embeddings.npy"

    metadata_file = None
    for file in EMBEDDINGS_DIR.glob("*"):
        lower = file.name.lower()
        if "metadata" in lower and lower.endswith((".csv", ".parquet", ".pkl")):
            metadata_file = file
            break

    if embedding_file.exists() and metadata_file is not None:
        return embedding_file, metadata_file

    return None, None


def load_metadata_file(path):
    """Load a metadata file based on its extension."""

    suffix = str(path).lower()

    if suffix.endswith(".csv"):
        return pd.read_csv(path, low_memory=False)
    elif suffix.endswith(".parquet"):
        return pd.read_parquet(path)
    else:
        return pd.read_pickle(path)


def duckdb_tables_valid(duckdb_loader, expected_structured, expected_metadata):
    """
    Returns True if structured and metadata tables exist
    and their row counts match the expected values.
    """

    for table, expected in [
        (STRUCTURED_TABLE,  expected_structured),
        (METADATA_TABLE,    expected_metadata),
    ]:

        if not duckdb_loader.table_exists(table):
            return False

        if duckdb_loader.row_count(table) != expected:
            return False

    return True


def duckdb_intelligence_valid(duckdb_loader, expected_rows):
    """
    Returns True if the intelligence table exists
    and its row count matches the expected value.
    """

    if not duckdb_loader.table_exists(INTELLIGENCE_TABLE):
        return False

    return duckdb_loader.row_count(INTELLIGENCE_TABLE) == expected_rows


def qdrant_valid(qdrant_loader, expected_vectors):
    """
    Returns True if the Qdrant collection exists
    and its vector count matches the expected value.
    """

    if not qdrant_loader.collection_exists():
        return False

    return qdrant_loader.count() == expected_vectors


def compute_expected_vectors(df: pd.DataFrame) -> int:
    """
    The number of points Qdrant will actually end up holding is the
    number of UNIQUE rows by whatever primary key QdrantLoader itself
    would detect for this dataframe -- never the raw row count, since
    QdrantLoader deduplicates on upload. This mirrors qdrant_loader.py's
    own detect_primary_key() so the two never drift out of sync, and
    works for any dataset regardless of column names.
    """
    key_columns, strategy = detect_primary_key(df)

    if strategy == "none" or not key_columns:
        # No reliable id column(s) -- QdrantLoader falls back to
        # text-hash/UUID-based IDs, at which point every row is
        # its own point and the raw row count IS correct.
        return len(df)

    return df[key_columns].drop_duplicates().shape[0]


def merge_intelligence(metadata_df_embeddings, intelligence_df):
    """
    Left-join intelligence columns onto metadata_df_embeddings.

    - Identifies shared join key (chunk_id preferred, record_id fallback)
    - Drops overlapping columns from intelligence_df to prevent _x/_y splits
    - Flattens any cells that ended up as pandas Series after the merge
    - Re-casts numeric columns to their correct dtype after flattening
    """

    # ── find join key ─────────────────────────────────────────────────────────
    join_key = None
    for candidate in ("chunk_id", "record_id"):
        if (
            candidate in metadata_df_embeddings.columns
            and candidate in intelligence_df.columns
        ):
            join_key = candidate
            break

    if join_key is None:
        raise ValueError(
            "Cannot merge intelligence: no shared key (chunk_id / record_id) "
            "found between metadata_df_embeddings and intelligence_df."
        )

    # ── drop overlapping columns from intelligence side (keep join key) ───────
    overlap = [
        c for c in intelligence_df.columns
        if c in metadata_df_embeddings.columns and c != join_key
    ]
    if overlap:
        print(f"  Dropping overlapping columns from intelligence_df : {overlap}")

    intel_to_merge = intelligence_df.drop(columns=overlap)

    # ── merge ─────────────────────────────────────────────────────────────────
    enriched = metadata_df_embeddings.merge(intel_to_merge, on=join_key, how="left")

    # ── flatten any Series objects produced by the merge ─────────────────────
    series_cols = []
    for col in enriched.columns:
        sample = enriched[col].iloc[0]
        if isinstance(sample, pd.Series):
            series_cols.append(col)
            enriched[col] = enriched[col].apply(
                lambda x: x.iloc[0] if isinstance(x, pd.Series) else x
            )
            # re-cast to numeric where possible
            enriched[col] = pd.to_numeric(enriched[col], errors="ignore")

    if series_cols:
        print(f"  WARNING — flattened Series objects in : {series_cols}")

    new_cols = [c for c in enriched.columns if c not in metadata_df_embeddings.columns]
    print(f"  Intelligence columns merged : {new_cols}")
    print(f"  Enriched dataframe shape    : {enriched.shape}")

    return enriched


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("\n" + "=" * 80)
    print("STARTING PIPELINE")
    print("=" * 80)

    # ──────────────────────────────────────────────────────
    # 1-2. LOAD RAW CSV -> CLEANING
    # (skipped only if CLEANED_DATA_FILE already exists)
    # ──────────────────────────────────────────────────────

    if CLEANED_DATA_FILE.exists():

        with stage("1-2. LOAD/CLEAN — SKIPPED (cleaned file already exists)"):

            print(f"  Found : {CLEANED_DATA_FILE} — skipping load_csv, cleaning.")

            cleaned_df = pd.read_csv(CLEANED_DATA_FILE, low_memory=False)

            print(f"  Rows    : {len(cleaned_df):,}")
            print(f"  Columns : {len(cleaned_df.columns)}")

    else:

        # ── 1. LOAD RAW CSV ──────────────────────────────────────────
        with stage("1. LOADING RAW CSV"):

            loader = CSVLoader(str(RAW_DATA_FILE))
            raw_df = loader.load()

            if raw_df is None or raw_df.empty:
                raise ValueError("Loaded CSV is empty or None.")

            print(f"  Rows    : {len(raw_df):,}")
            print(f"  Columns : {len(raw_df.columns)}")

        # ── 2. CLEANING ──────────────────────────────────────────────
        with stage("2. CLEANING"):

            cleaner = DataCleaner()
            rows_before = len(raw_df)
            cleaned_df = cleaner.clean(raw_df)

            CLEANED_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
            if hasattr(cleaner, "save"):
                cleaner.save(cleaned_df, str(CLEANED_DATA_FILE))
            else:
                cleaned_df.to_csv(CLEANED_DATA_FILE, index=False)

            print(f"  Rows before : {rows_before:,}")
            print(f"  Rows after  : {len(cleaned_df):,}")
            print(f"  Rows dropped: {rows_before - len(cleaned_df):,}")

            del raw_df

    # ──────────────────────────────────────────────────────
    # 3. CANONICAL MAPPING
    # (always runs, regardless of whether stages 1-2 were skipped —
    # not gated by CLEANED_DATA_FILE.exists())
    # ──────────────────────────────────────────────────────

    with stage("3. CANONICAL MAPPING"):

        mapper = CanonicalMapper()
        mapped_df = mapper.map(cleaned_df)

        if "record_id" not in mapped_df.columns:
            mapped_df.insert(0, "record_id", range(len(mapped_df)))

        MAPPED_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(mapper, "save"):
            mapper.save(mapped_df, str(MAPPED_DATA_FILE))
        else:
            mapped_df.to_csv(MAPPED_DATA_FILE, index=False)

        print(f"  Rows    : {len(mapped_df):,}")
        print(f"  Columns : {len(mapped_df.columns)}")

        del cleaned_df

    df = mapped_df

    # ──────────────────────────────────────────────────────
    # 4. SCHEMA ANALYSIS
    # ──────────────────────────────────────────────────────

    with stage("4. SCHEMA ANALYSIS"):

        analyzer = SchemaAnalyzer()
        schema_result = analyzer.analyze(df)
        analyzer.print_summary(schema_result)

    # ──────────────────────────────────────────────────────
    # 5. DATA SEPARATION
    # ──────────────────────────────────────────────────────

    with stage("5. DATA SEPARATION"):

        separator = DataSeparator()
        separated_data = separator.separate(df, schema_result)
        separator.summary(separated_data)
        separator.save(separated_data)

    structured_df   = separated_data["structured_df"]
    metadata_df     = separated_data["metadata_df"]
    unstructured_df = separated_data["unstructured_df"]

    # ──────────────────────────────────────────────────────
    # 6. DUCKDB — structured + metadata tables
    # ──────────────────────────────────────────────────────

    with stage("6. LOADING DUCKDB"):

        duckdb_loader = DuckDBLoader()

        if duckdb_tables_valid(
            duckdb_loader,
            expected_structured=len(structured_df),
            expected_metadata=len(metadata_df),
        ):
            print(
                f"  structured_data : {len(structured_df):,} rows — already loaded, skipping."
            )
            print(
                f"  metadata_data   : {len(metadata_df):,} rows — already loaded, skipping."
            )

        else:
            print("  Tables missing or row count mismatch — reloading.")
            duckdb_loader.load_structured(structured_df)
            duckdb_loader.load_metadata(metadata_df)

        duckdb_loader.show_tables()

    # ──────────────────────────────────────────────────────
    # 7. CHECK EMBEDDINGS
    # ──────────────────────────────────────────────────────

    with stage("7. CHECK EMBEDDINGS"):

        embedding_file, metadata_file = find_embedding_files()
        embeddings_exist = embedding_file is not None

        if embeddings_exist:
            print(f"  Found : {embedding_file}")
            print(f"  Found : {metadata_file}")
        else:
            print("  No pre-computed embeddings found — will generate fresh.")

    # ──────────────────────────────────────────────────────
    # BRANCH A — embeddings already exist
    # ──────────────────────────────────────────────────────

    if embeddings_exist:

        with stage("8A. LOAD EXISTING EMBEDDINGS"):

            metadata_df_embeddings = load_metadata_file(metadata_file)

            print(f"  Loaded metadata : {len(metadata_df_embeddings):,} rows")
            print(f"  Embeddings npy  : {embedding_file}")

    # ──────────────────────────────────────────────────────
    # BRANCH B — generate embeddings from scratch
    # ──────────────────────────────────────────────────────

    else:

        embedder = EmbeddingGenerator()

        # 8B-i. CHUNKING
        with stage("8B-i. CHUNK CREATION"):

            chunker  = TextChunker()
            chunk_df = chunker.chunk_dataframe(unstructured_df)
            chunker.summary(chunk_df)
            chunker.save(chunk_df)

        # 8B-ii. EMBEDDING GENERATION
        with stage("8B-ii. EMBEDDING GENERATION"):

            metadata_df_embeddings, unique_df = embedder.generate(chunk_df)

        # 8B-iii. OPTIONAL QUANTIZATION
        if ENABLE_QUANTIZATION:

            with stage("8B-iii. QUANTIZATION"):

                embeddings = embedder.load_embeddings()

                quantizer = EmbeddingQuantizer()
                quantized_embeddings, scale = quantizer.quantize_int8(embeddings)

                error_metrics = quantizer.reconstruction_error(
                    embeddings, quantized_embeddings, scale
                )

                quantizer.summary(embeddings, quantized_embeddings, error_metrics)
                quantizer.save_quantized(quantized_embeddings, scale, error_metrics=error_metrics)

                del embeddings

    # ──────────────────────────────────────────────────────
    # 9. PRECOMPUTED INTELLIGENCE
    # (must run BEFORE Qdrant so payloads include intel fields)
    # ──────────────────────────────────────────────────────

    with stage("9. PRECOMPUTED INTELLIGENCE"):

        intelligence_engine = PrecomputedIntelligence()
        expected_intelligence = len(metadata_df_embeddings)

        if not embeddings_exist:
            # Fresh run — always generate from chunk_df
            intelligence_df = intelligence_engine.generate(chunk_df)
            intelligence_engine.summary(intelligence_df)
            intelligence_engine.save(intelligence_df)

            duckdb_loader.load_intelligence(intelligence_df)
            duckdb_loader.show_tables()

        else:
            # Pre-computed embeddings path
            if duckdb_intelligence_valid(duckdb_loader, expected_intelligence):
                print(
                    f"  Intelligence table already has "
                    f"{expected_intelligence:,} rows — loading from DuckDB."
                )
                # Read back from DuckDB so we can merge below
                intelligence_df = duckdb_loader.read_table(INTELLIGENCE_TABLE)

            else:
                print(
                    "  Intelligence table missing or row count mismatch — regenerating."
                )
                intelligence_df = intelligence_engine.generate(
                    load_metadata_file(metadata_file)
                )
                intelligence_engine.summary(intelligence_df)
                intelligence_engine.save(intelligence_df)

                duckdb_loader.load_intelligence(intelligence_df)
                duckdb_loader.show_tables()

        # Merge intelligence columns into the embeddings dataframe so that
        # Qdrant payloads are fully enriched before upload.
        with stage("9b. MERGE INTELLIGENCE INTO EMBEDDINGS DATAFRAME"):
            metadata_df_embeddings = merge_intelligence(
                metadata_df_embeddings, intelligence_df
            )

    # ──────────────────────────────────────────────────────
    # 9c. PAYLOAD INTEGRITY FIX
    # Ensure all scalar fields are proper Python scalars,
    # not accidentally serialized pandas Series objects.
    # ──────────────────────────────────────────────────────

    with stage("9c. PAYLOAD INTEGRITY FIX"):

        fixes = {}

        for col in metadata_df_embeddings.columns:

            sample = metadata_df_embeddings[col].iloc[0]

            if isinstance(sample, pd.Series):

                print(f"  Fixing  : {col}  (Series → scalar)")

                metadata_df_embeddings[col] = metadata_df_embeddings[col].apply(
                    lambda x: x.iloc[0] if isinstance(x, pd.Series) else x
                )

                # Re-cast to correct dtype after flattening
                try:
                    metadata_df_embeddings[col] = pd.to_numeric(
                        metadata_df_embeddings[col], errors="ignore"
                    )
                except Exception:
                    pass

                fixes[col] = str(metadata_df_embeddings[col].dtype)

        if fixes:
            print(f"\n  Fixed {len(fixes)} column(s) : {list(fixes.keys())}")
        else:
            print("  All columns are clean — no Series objects found.")

    # ──────────────────────────────────────────────────────
    # 10. LOAD QDRANT
    # (uses intelligence-enriched metadata_df_embeddings)
    # ──────────────────────────────────────────────────────

    with stage("10. LOADING QDRANT"):

        qdrant_loader = QdrantLoader()

        expected_vectors = compute_expected_vectors(metadata_df_embeddings)

        if qdrant_valid(qdrant_loader, expected_vectors):
            print(
                f"  Collection already has {expected_vectors:,} vectors — skipping."
            )

        else:
            if qdrant_loader.collection_exists():
                print(
                    "  Vector count mismatch or empty — recreating collection..."
                )
                qdrant_loader.delete_collection()

            qdrant_loader.load_dataframe(
                metadata_df_embeddings,
                str(EMBEDDINGS_FILE),
                batch_size=QDRANT_BATCH_SIZE,
                parallel=8,
            )

        print(f"\n  Vectors Stored : {qdrant_loader.count():,}")

    # ──────────────────────────────────────────────────────
    # DONE
    # ──────────────────────────────────────────────────────

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()