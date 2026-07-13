# Embeddings/embedding_generator.py

import gc
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from sentence_transformers import SentenceTransformer

from Config.settings import (
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    EMBEDDINGS_DIR,
)

# ============================================================
# DEVICE RESOLUTION
# ============================================================

def _resolve_device(requested: str) -> str:

    if _TORCH_AVAILABLE and torch.cuda.is_available():

        props = torch.cuda.get_device_properties(0)

        vram_gb = (
            props.total_memory /
            1024 ** 3
        )

        print(
            f"GPU : {props.name}"
        )

        print(
            f"VRAM: {vram_gb:.1f} GB"
        )

        return "cuda"

    print(
        f"CUDA not available -> {requested}"
    )

    return requested


# ============================================================
# EMBEDDING GENERATOR
# ============================================================

class EmbeddingGenerator:

    MEMMAP_THRESHOLD = 500_000

    def __init__(
        self,
        load_model: bool = True
    ):

        self.output_dir = Path(
            EMBEDDINGS_DIR
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.batch_size = (
            EMBEDDING_BATCH_SIZE
        )

        self.device = (
            EMBEDDING_DEVICE
        )

        self.model = None

        if load_model:

            self.device = (
                _resolve_device(
                    EMBEDDING_DEVICE
                )
            )

            print(
                f"\nLoading model: "
                f"{EMBEDDING_MODEL}"
            )

            self.model = (
                SentenceTransformer(
                    EMBEDDING_MODEL,
                    device=self.device
                )
            )

    # ========================================================
    # VALIDATION
    # ========================================================

    @staticmethod
    def validate_input(
        df: pd.DataFrame
    ) -> None:

        if df is None:

            raise ValueError(
                "Input dataframe is None."
            )

        if df.empty:

            raise ValueError(
                "Input dataframe is empty."
            )

        if "chunk_text" not in df.columns:

            raise ValueError(
                f"chunk_text column missing. "
                f"Available columns: "
                f"{df.columns.tolist()}"
            )

    # ========================================================
    # CLEANING
    # ========================================================

    @staticmethod
    def _clean_text(
        text
    ) -> str:

        if pd.isna(text):

            return ""

        text = (
            str(text)
            .replace("\n", " ")
            .replace("\t", " ")
        )

        return " ".join(
            text.split()
        ).strip()

    @staticmethod
    def _sha256(
        text: str
    ) -> str:

        return hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()

    # ========================================================
    # PREPARE DATAFRAME
    # ========================================================

    def prepare_dataframe(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        self.validate_input(df)

        result = df.copy()

        if "chunk_id" not in result.columns:

            result["chunk_id"] = (
                range(len(result))
            )

        result["chunk_text"] = (
            result["chunk_text"]
            .map(self._clean_text)
        )

        result = result[
            result["chunk_text"]
            .str.len() > 0
        ].copy()

        if result.empty:

            raise ValueError(
                "No valid chunk text found."
            )

        result["text_hash"] = (
            result["chunk_text"]
            .map(self._sha256)
        )

        if "record_id" not in result.columns:

            result.insert(
                0,
                "record_id",
                range(len(result))
            )

        result.reset_index(
            drop=True,
            inplace=True
        )

        return result

    # ========================================================
    # UNIQUE TEXT TABLE
    # ========================================================

    @staticmethod
    def build_unique_text_df(
        prepared_df: pd.DataFrame
    ) -> pd.DataFrame:

        unique_df = (

            prepared_df[
                [
                    "text_hash",
                    "chunk_text"
                ]
            ]

            .drop_duplicates(
                subset=["text_hash"]
            )

            .reset_index(
                drop=True
            )

            .copy()
        )

        unique_df[
            "embedding_index"
        ] = range(
            len(unique_df)
        )

        return unique_df

    # ========================================================
    # ATTACH EMBEDDING INDEX
    # ========================================================

    @staticmethod
    def attach_embedding_index(
        prepared_df: pd.DataFrame,
        unique_df: pd.DataFrame
    ) -> pd.DataFrame:

        mapping = unique_df[
            [
                "text_hash",
                "embedding_index"
            ]
        ]

        result = prepared_df.merge(
            mapping,
            on="text_hash",
            how="left"
        )

        if result[
            "embedding_index"
        ].isna().any():

            raise ValueError(
                "Embedding index mapping failed."
            )

        result[
            "embedding_index"
        ] = result[
            "embedding_index"
        ].astype(int)

        return result

    # ========================================================
    # MODEL CHECK
    # ========================================================

    def _assert_model(
        self
    ):

        if self.model is None:

            raise RuntimeError(
                "Model not loaded."
            )

    # ========================================================
    # RAM MODE
    # ========================================================

    def _encode_standard(
        self,
        texts: List[str]
    ) -> np.ndarray:

        print(
            f"\nEncoding "
            f"{len(texts):,} "
            f"unique texts..."
        )

        vectors = self.model.encode(

            texts,

            batch_size=
            self.batch_size,

            convert_to_numpy=True,

            normalize_embeddings=True,

            show_progress_bar=True
        )

        return np.asarray(
            vectors,
            dtype=np.float32
        )

    # ========================================================
    # MEMMAP MODE
    # ========================================================

    def _encode_to_memmap(
        self,
        texts: List[str],
        mmap_path: Path,
        dim: int
    ) -> np.ndarray:

        total = len(texts)

        mmap = np.memmap(
            mmap_path,
            dtype="float32",
            mode="w+",
            shape=(total, dim)
        )

        written = 0

        for start in range(
            0,
            total,
            self.batch_size
        ):

            end = min(
                start +
                self.batch_size,
                total
            )

            vectors = self.model.encode(

                texts[start:end],

                batch_size=
                self.batch_size,

                convert_to_numpy=True,

                normalize_embeddings=True,

                show_progress_bar=False
            )

            mmap[start:end] = (
                vectors.astype(
                    np.float32
                )
            )

            written += (
                end - start
            )

            if (
                written %
                (
                    self.batch_size * 10
                ) == 0
            ) or end == total:

                print(
                    f"Encoded "
                    f"{written:,} / "
                    f"{total:,}"
                )

        mmap.flush()

        result = np.array(
            mmap,
            dtype=np.float32
        )

        del mmap

        try:

            mmap_path.unlink(
                missing_ok=True
            )

        except Exception:
            pass

        return result

    # ========================================================
    # GPU CLEANUP
    # ========================================================

    def _release_gpu(
        self
    ):

        if (
            _TORCH_AVAILABLE
            and
            self.device == "cuda"
            and
            self.model is not None
        ):

            del self.model

            self.model = None

            gc.collect()

            torch.cuda.empty_cache()

            print(
                "\nGPU cache cleared."
            )
                # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    def generate(
        self,
        df: pd.DataFrame,
        return_unique_df: bool = True,
        base_name: str = "embeddings",
    ):
        """
        prepare → deduplicate → encode → save immediately

        Returns
        -------
        metadata_df
        unique_df
        """

        self._assert_model()

        prepared_df = self.prepare_dataframe(df)

        unique_df = self.build_unique_text_df(
            prepared_df
        )

        texts = unique_df[
            "chunk_text"
        ].tolist()

        total_rows = len(
            prepared_df
        )

        total_unique = len(
            unique_df
        )

        print(
            f"  Rows after cleaning          : "
            f"{total_rows:,}"
        )

        print(
            f"  Unique texts to encode       : "
            f"{total_unique:,}"
        )

        # ====================================================
        # DETERMINE EMBEDDING DIMENSION
        # ====================================================

        probe = self.model.encode(

            texts[: min(2, total_unique)],

            batch_size=2,

            convert_to_numpy=True,

            normalize_embeddings=True,

            show_progress_bar=False

        )

        dim = probe.shape[1]

        print(
            f"  Embedding dimension          : "
            f"{dim}"
        )

        # ====================================================
        # GENERATE EMBEDDINGS
        # ====================================================

        if total_unique >= self.MEMMAP_THRESHOLD:

            print(
                "  Large dataset — memmap mode."
            )

            mmap_path = (
                self.output_dir /
                f"{base_name}_tmp.mmap"
            )

            embeddings = self._encode_to_memmap(
                texts,
                mmap_path,
                dim
            )

        else:

            embeddings = self._encode_standard(
                texts
            )

        # ====================================================
        # RELEASE GPU
        # ====================================================

        self._release_gpu()

        # ====================================================
        # ATTACH EMBEDDING INDEX
        # ====================================================

        metadata_df = (
            self.attach_embedding_index(
                prepared_df,
                unique_df
            )
        )

        print(
            f"  Embedding matrix shape       : "
            f"{embeddings.shape}"
        )

        # ====================================================
        # SAVE IMMEDIATELY
        # ====================================================

        self.save(

            metadata_df=metadata_df,

            embeddings=embeddings,

            unique_df=unique_df,

            base_name=base_name

        )

        print(
            "\nEmbeddings saved successfully."
        )

        # ====================================================
        # SUMMARY BEFORE MEMORY CLEANUP
        # ====================================================

        self.summary(

            metadata_df=metadata_df,

            embeddings=embeddings,

            unique_df=unique_df

        )

        # ====================================================
        # FREE MEMORY
        # ====================================================

        del embeddings

        gc.collect()

        if _TORCH_AVAILABLE:

            try:
                torch.cuda.empty_cache()

            except Exception:
                pass

        # ====================================================
        # RETURN ONLY DATAFRAMES
        # ====================================================

        if return_unique_df:

            return (
                metadata_df,
                unique_df
            )

        return metadata_df
        # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    def save(
        self,
        metadata_df: pd.DataFrame,
        embeddings: np.ndarray,
        unique_df: Optional[pd.DataFrame] = None,
        base_name: str = "embeddings",
    ) -> None:

        if metadata_df is None:

            raise ValueError(
                "metadata_df cannot be None"
            )

        if embeddings is None:

            raise ValueError(
                "embeddings cannot be None"
            )

        emb_file = (
            self.output_dir /
            f"{base_name}.npy"
        )

        meta_file = (
            self.output_dir /
            f"{base_name}_metadata.csv"
        )

        unique_file = (
            self.output_dir /
            f"{base_name}_unique_texts.csv"
        )

        manifest_file = (
            self.output_dir /
            f"{base_name}_manifest.json"
        )

        np.save(
            emb_file,
            embeddings
        )

        metadata_df.to_csv(
            meta_file,
            index=False
        )

        if unique_df is not None:

            unique_df.to_csv(
                unique_file,
                index=False
            )

        manifest = {

            "embedding_model":
            EMBEDDING_MODEL,

            "embedding_device":
            self.device,

            "embedding_batch_size":
            self.batch_size,

            "row_count":
            int(len(metadata_df)),

            "unique_embedding_count":
            int(embeddings.shape[0]),

            "dimensions":
            int(embeddings.shape[1]),

            "dtype":
            str(embeddings.dtype),

            "files": {

                "embeddings":
                emb_file.name,

                "metadata":
                meta_file.name,

                "unique_texts":
                (
                    unique_file.name
                    if unique_df is not None
                    else None
                )

            }

        }

        with open(
            manifest_file,
            "w",
            encoding="utf-8"
        ) as fh:

            json.dump(
                manifest,
                fh,
                indent=2
            )

        print(
            f"\nEmbeddings   -> {emb_file}"
        )

        print(
            f"Metadata     -> {meta_file}"
        )

        if unique_df is not None:

            print(
                f"Unique texts -> {unique_file}"
            )

        print(
            f"Manifest     -> {manifest_file}"
        )

    # --------------------------------------------------------
    # LOAD FULL PACKAGE
    # --------------------------------------------------------

    def load(
        self,
        base_name: str = "embeddings"
    ):

        emb_file = (
            self.output_dir /
            f"{base_name}.npy"
        )

        meta_file = (
            self.output_dir /
            f"{base_name}_metadata.csv"
        )

        unique_file = (
            self.output_dir /
            f"{base_name}_unique_texts.csv"
        )

        manifest_file = (
            self.output_dir /
            f"{base_name}_manifest.json"
        )

        if not emb_file.exists():

            raise FileNotFoundError(
                emb_file
            )

        if not meta_file.exists():

            raise FileNotFoundError(
                meta_file
            )

        embeddings = np.load(
            emb_file,
            mmap_mode="r"
        )

        metadata_df = pd.read_csv(
            meta_file
        )

        unique_df = None

        if unique_file.exists():

            unique_df = pd.read_csv(
                unique_file
            )

        manifest = None

        if manifest_file.exists():

            with open(
                manifest_file,
                "r",
                encoding="utf-8"
            ) as fh:

                manifest = json.load(
                    fh
                )

        return (
            metadata_df,
            embeddings,
            unique_df,
            manifest
        )

    # --------------------------------------------------------
    # LOAD EMBEDDINGS ONLY
    # --------------------------------------------------------

    def load_embeddings(
        self,
        base_name="embeddings"
    ):

        embedding_file = (
            self.output_dir /
            f"{base_name}.npy"
        )

        if not embedding_file.exists():

            raise FileNotFoundError(
                embedding_file
            )

        print(
            f"\nLoading embeddings:"
        )

        print(
            embedding_file
        )

        return np.load(
            embedding_file,
            mmap_mode="r"
        )

    # --------------------------------------------------------
    # QDRANT PAYLOAD BUILDER
    # --------------------------------------------------------

    @staticmethod
    def to_qdrant_payload(
        metadata_df: pd.DataFrame
    ) -> List[Dict]:

        if metadata_df is None:

            return []

        if metadata_df.empty:

            return []

        preferred = [

            "record_id",
            "chunk_id",
            "chunk_index",
            "embedding_index",

            "chunk_text",

            "clean_chunk_text",

            "source_columns",
            "source_column",

            "source_type",
            "document_type",

            "topic",
            "topics",

            "aspect",
            "aspects",

            "keywords",

            "sentiment",
            "sentiment_label",

            "rating",

            "timestamp",
            "date",

            "brand",

            "main_category",

            "product_id"

        ]

        available = [

            col

            for col in preferred

            if col in metadata_df.columns

        ]

        return metadata_df[
            available
        ].to_dict(
            orient="records"
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    def summary(
        self,
        metadata_df: pd.DataFrame,
        embeddings: Optional[np.ndarray] = None,
        unique_df: Optional[pd.DataFrame] = None,
    ):

        print("\n" + "=" * 60)

        print(
            "EMBEDDING SUMMARY"
        )

        print("=" * 60)

        print(
            f"Metadata rows          : "
            f"{len(metadata_df):,}"
        )

        if unique_df is not None:

            dedup_pct = (

                1
                -
                len(unique_df)
                /
                len(metadata_df)

            ) * 100

            print(
                f"Unique embedded texts  : "
                f"{len(unique_df):,}"
            )

            print(
                f"Deduplication          : "
                f"{dedup_pct:.2f}%"
            )

        if embeddings is not None:

            print(
                f"Embedding vectors      : "
                f"{embeddings.shape[0]:,}"
            )

            print(
                f"Embedding dimensions   : "
                f"{embeddings.shape[1]}"
            )

            print(
                f"Embedding dtype        : "
                f"{embeddings.dtype}"
            )

            print(
                f"Matrix size            : "
                f"{embeddings.nbytes / 1024**2:.2f} MB"
            )

        print("=" * 60)