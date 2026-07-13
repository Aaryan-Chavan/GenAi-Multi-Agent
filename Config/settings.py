# config/settings.py
import os
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# ============================================================
# DATA DIRECTORIES
# ============================================================

DATA_DIR = BASE_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
CLEANED_DATA_DIR = DATA_DIR / "cleaned"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# ============================================================
# DATABASE DIRECTORIES
# ============================================================

DATABASE_DIR = BASE_DIR / "database"

DUCKDB_DIR = DATABASE_DIR / "duckdb"
QDRANT_DIR = DATABASE_DIR / "qdrant"

# ============================================================
# LOGS
# ============================================================

LOGS_DIR = BASE_DIR / "logs"

# ============================================================
# INPUT FILES
# ============================================================

RAW_DATA_FILE = (
    RAW_DATA_DIR / "dataset.csv"
)

CLEANED_DATA_FILE = (
    CLEANED_DATA_DIR / "cleaned_dataset.csv"
)

# ============================================================
# PREPROCESSING OUTPUTS
# ============================================================

STRUCTURED_FILE = (
    PROCESSED_DATA_DIR / "structured.csv"
)

UNSTRUCTURED_FILE = (
    PROCESSED_DATA_DIR / "unstructured.csv"
)

METADATA_FILE = (
    PROCESSED_DATA_DIR / "metadata.csv"
)

CHUNKS_FILE = (
    PROCESSED_DATA_DIR / "chunks.csv"
)

INTELLIGENCE_FILE = (
    PROCESSED_DATA_DIR /
    "precomputed_intelligence.csv"
)

# ============================================================
# EMBEDDINGS
# ============================================================

EMBEDDING_MODEL = (
    "BAAI/bge-small-en-v1.5"
)

# auto / cpu / cpu
EMBEDDING_DEVICE = "cuda"

# 24 GB GPU
EMBEDDING_BATCH_SIZE = 2048

# save storage
EMBEDDING_DTYPE = "float16"

EMBEDDINGS_FILE = (
    EMBEDDINGS_DIR /
    "embeddings.npy"
)

# ============================================================
# QUANTIZATION
# ============================================================

ENABLE_QUANTIZATION = False

QUANTIZED_EMBEDDINGS_FILE = (
    EMBEDDINGS_DIR /
    "quantized_embeddings.npz"
)

# ============================================================
# CHUNKING
# ============================================================

# Recursive chunking settings

CHUNK_SIZE = 512

CHUNK_OVERLAP = 30

MIN_TEXT_LENGTH = 20

MAX_CHUNK_LENGTH = 512

# ============================================================
# SCHEMA ANALYZER
# ============================================================

TEXT_LENGTH_THRESHOLD = 80

TEXT_WORD_THRESHOLD = 10

UNIQUE_RATIO_THRESHOLD = 0.95

# ============================================================
# PRECOMPUTED INTELLIGENCE
# ============================================================

# Fast vectorized implementation

MAX_KEYWORDS = 0

POSITIVE_SENTIMENT_THRESHOLD = 0.10

NEGATIVE_SENTIMENT_THRESHOLD = -0.10

# ============================================================
# DUCKDB
# ============================================================

DUCKDB_FILE = (
    DUCKDB_DIR /
    "analytics.duckdb"
)

STRUCTURED_TABLE = (
    "structured_data"
)

METADATA_TABLE = (
    "metadata_data"
)

INTELLIGENCE_TABLE = (
    "precomputed_intelligence"
)

# ============================================================
# QDRANT
# ============================================================

QDRANT_HOST = "localhost"

QDRANT_PORT = 6333

QDRANT_COLLECTION = (
    "documents"
)

QDRANT_DISTANCE = "Cosine"

# Batch insertion

QDRANT_BATCH_SIZE = 50000

# ============================================================
# REDIS CACHE
# ============================================================

REDIS_HOST = os.getenv("REDIS_HOST")

REDIS_PORT = 6379

REDIS_DB = 0

CACHE_TTL_SECONDS = 3600

# ============================================================
# LLM
# ============================================================
LLM_MODEL = "Qwen/Qwen2.5-14B-Instruct"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_RESPONSE_TOKENS = 2048

TEMPERATURE = 0.1

TOP_P = 0.9

MAX_CONTEXT_CHUNKS = 30

# ============================================================
# RETRIEVAL
# ============================================================

TOP_K_QDRANT = 50

TOP_K_DUCKDB = 500

MAX_RETRIEVAL_RESULTS = 1000

# ============================================================
# LOGGING
# ============================================================

LOG_LEVEL = "INFO"

LOG_FILE = (
    LOGS_DIR /
    "application.log"
)

# ============================================================
# EVALUATION
# ============================================================

ENABLE_EVALUATION = True

EVALUATION_SAMPLE_SIZE = 100

MIN_ACCEPTABLE_ACCURACY = 0.80

MAX_ACCEPTABLE_LATENCY = 10

# ============================================================
# PERFORMANCE
# ============================================================

ENABLE_PARALLEL_PIPELINE = True

MAX_WORKERS = 2

# ============================================================
# DIRECTORIES
# ============================================================

DIRECTORIES = [

    DATA_DIR,

    RAW_DATA_DIR,
    CLEANED_DATA_DIR,
    PROCESSED_DATA_DIR,
    EMBEDDINGS_DIR,

    DATABASE_DIR,

    DUCKDB_DIR,
    QDRANT_DIR,

    LOGS_DIR
]

for directory in DIRECTORIES:

    directory.mkdir(
        parents=True,
        exist_ok=True
    )