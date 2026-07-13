# Preprocessing/chunking.py

import re
from pathlib import Path
from typing import List

import pandas as pd

from Config.settings import CHUNKS_FILE

# ============================================================
# CONFIGURABLE DEFAULTS
# ============================================================

CHUNK_SIZE         = 200   # target max words per chunk
CHUNK_OVERLAP      = 30    # overlap words carried into next chunk
MIN_CHUNK_WORDS    = 15    # drop chunks shorter than this
MAX_CHUNKS_PER_DOC = 25    # hard cap — prevents runaway long documents

# Module-level compiled regex (paid once at import)
_PARA_SPLIT   = re.compile(r"\n{2,}|\r\n{2,}")
_SENT_SPLIT   = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE   = re.compile(r"[\n\t\r ]+")


class TextChunker:

    def __init__(
        self,
        chunk_size:         int = CHUNK_SIZE,
        chunk_overlap:      int = CHUNK_OVERLAP,
        min_chunk_words:    int = MIN_CHUNK_WORDS,
        max_chunks_per_doc: int = MAX_CHUNKS_PER_DOC,
        field_separator:    str = " | ",
    ):
        self.chunk_size         = chunk_size
        self.chunk_overlap      = chunk_overlap
        self.min_chunk_words    = min_chunk_words
        self.max_chunks_per_doc = max_chunks_per_doc
        self.field_separator    = field_separator

    # --------------------------------------------------------
    # COLUMN DETECTION
    # --------------------------------------------------------

    def get_text_columns(self, df: pd.DataFrame) -> List[str]:
        excluded = {"record_id", "product_id"}
        return [
            col for col in df.columns
            if col not in excluded
            and pd.api.types.is_string_dtype(df[col])
        ]

    # --------------------------------------------------------
    # ROW → DOCUMENT  (merge all text columns into one string)
    # --------------------------------------------------------

    def _merge_row(self, row_dict: dict, text_columns: List[str]) -> str:
        seen  = set()
        parts = []
        for col in text_columns:
            val = row_dict.get(col)
            if val is None or (isinstance(val, float) and val != val):
                # val != val is the fastest NaN check without pandas
                continue
            val = _WHITESPACE.sub(" ", str(val)).strip()
            if not val or val in seen:
                continue
            seen.add(val)
            parts.append(val)
        return self.field_separator.join(parts)

    # --------------------------------------------------------
    # SPLITTING PRIMITIVES
    # --------------------------------------------------------

    @staticmethod
    def _split_paragraphs(text: str) -> List[str]:
        return [p.strip() for p in _PARA_SPLIT.split(text) if p.strip()]

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]

    def _word_windows(self, text: str) -> List[str]:
        """Hard fallback: fixed-size word windows with overlap."""
        words = text.split()
        step  = max(1, self.chunk_size - self.chunk_overlap)
        return [
            " ".join(words[i: i + self.chunk_size])
            for i in range(0, len(words), step)
            if words[i: i + self.chunk_size]
        ]

    # --------------------------------------------------------
    # PACK UNITS INTO CHUNKS  (shared by paragraph + sentence paths)
    # --------------------------------------------------------

    def _pack(self, units: List[str]) -> List[str]:
        """
        Greedily pack text units into chunks ≤ chunk_size words,
        carrying an overlap window into each new chunk.
        Units that exceed chunk_size are sub-split via _word_windows.
        """
        chunks: List[str] = []
        buf:    List[str] = []
        buf_wc = 0

        for unit in units:
            uwc = len(unit.split())

            if uwc > self.chunk_size:
                # Flush current buffer first
                if buf:
                    chunks.append(" ".join(buf))
                    buf, buf_wc = [], 0
                chunks.extend(self._word_windows(unit))
                continue

            if buf_wc + uwc <= self.chunk_size:
                buf.append(unit)
                buf_wc += uwc
            else:
                if buf:
                    chunks.append(" ".join(buf))

                # Build overlap from tail of current buffer
                overlap, ov_wc = [], 0
                for s in reversed(buf):
                    sw = len(s.split())
                    if ov_wc + sw > self.chunk_overlap:
                        break
                    overlap.insert(0, s)
                    ov_wc += sw

                buf    = overlap + [unit]
                buf_wc = ov_wc + uwc

        if buf:
            chunks.append(" ".join(buf))

        return chunks

    # --------------------------------------------------------
    # HIERARCHICAL CHUNK  (paragraph → sentence → word)
    # --------------------------------------------------------

    def chunk_document(self, document: str) -> List[str]:
        if not document:
            return []

        doc = document.strip()
        wc  = len(doc.split())

        if wc == 0:
            return []

        # Short enough to return as-is
        if wc <= self.chunk_size:
            return [doc] if wc >= self.min_chunk_words else []

        # Try paragraph split first
        paragraphs = self._split_paragraphs(doc)

        if len(paragraphs) > 1:
            raw: List[str] = []
            for pc in self._pack(paragraphs):
                if len(pc.split()) > self.chunk_size:
                    sents = self._split_sentences(pc)
                    raw.extend(self._pack(sents) if len(sents) > 1 else self._word_windows(pc))
                else:
                    raw.append(pc)
        else:
            # Single paragraph — go straight to sentence split
            sents = self._split_sentences(doc)
            raw   = self._pack(sents) if len(sents) > 1 else self._word_windows(doc)

        # Quality filter + dedup + cap
        seen:     set       = set()
        filtered: List[str] = []

        for chunk in raw:
            chunk = chunk.strip()
            if len(chunk.split()) < self.min_chunk_words:
                continue
            h = hash(chunk)
            if h in seen:
                continue
            seen.add(h)
            filtered.append(chunk)
            if len(filtered) >= self.max_chunks_per_doc:
                break

        return filtered

    # --------------------------------------------------------
    # MAIN ENTRY POINT
    # --------------------------------------------------------

    def chunk_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge all unstructured text columns per row into a single document,
        then chunk that document.  One document per row means no chunk explosion.

        Output columns
        --------------
        chunk_id         globally unique int  (Qdrant / DuckDB key)
        record_id        original row identifier
        product_id       passed through if present
        source_columns   comma-joined names of contributing columns
        chunk_index      position within the document (0-based)
        chunk_text       chunk content
        chunk_word_count
        chunk_char_count
        """
        empty_cols = [
            "chunk_id", "record_id", "source_columns",
            "chunk_index", "chunk_text", "chunk_word_count", "chunk_char_count",
        ]

        if df.empty:
            return pd.DataFrame(columns=empty_cols)

        text_columns = self.get_text_columns(df)
        if not text_columns:
            return pd.DataFrame(columns=empty_cols)

        has_product  = "product_id" in df.columns
        source_label = ", ".join(text_columns)
        total_rows   = len(df)
        records      = []

        for idx, row in enumerate(df.itertuples(index=False), start=1):
            row_dict   = row._asdict()
            record_id  = row_dict.get("record_id", idx)
            product_id = row_dict.get("product_id") if has_product else None

            document = self._merge_row(row_dict, text_columns)
            chunks   = self.chunk_document(document)

            for ci, chunk in enumerate(chunks):
                rec = {
                    "record_id":        record_id,
                    "source_columns":   source_label,
                    "chunk_index":      ci,
                    "chunk_text":       chunk,
                    "chunk_word_count": len(chunk.split()),
                    "chunk_char_count": len(chunk),
                }
                if has_product:
                    rec["product_id"] = product_id
                records.append(rec)

            if idx % 10_000 == 0:
                print(
                    f"  Chunked {idx:,} / {total_rows:,} rows "
                    f"| chunks so far: {len(records):,}"
                )

        if not records:
            return pd.DataFrame(columns=empty_cols)

        chunk_df = pd.DataFrame(records)
        chunk_df.insert(0, "chunk_id", range(len(chunk_df)))

        return chunk_df

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    def save(self, chunk_df: pd.DataFrame) -> None:
        out = Path(CHUNKS_FILE)
        out.parent.mkdir(parents=True, exist_ok=True)
        chunk_df.to_csv(out, index=False)
        print(f"\n  Chunks saved -> {out}")

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    def summary(self, chunk_df: pd.DataFrame) -> None:
        print("\n" + "=" * 60)
        print("CHUNKING SUMMARY")
        print("=" * 60)
        print(f"Total Chunks     : {len(chunk_df):,}")

        if chunk_df.empty:
            print("=" * 60)
            return

        if "chunk_word_count" in chunk_df.columns:
            wc = chunk_df["chunk_word_count"]
            print(f"Avg Words/Chunk  : {wc.mean():.1f}")
            print(f"Min Words/Chunk  : {wc.min()}")
            print(f"Max Words/Chunk  : {wc.max()}")

        if "record_id" in chunk_df.columns:
            cpd = chunk_df.groupby("record_id").size()
            print(f"\nChunks per document — mean: {cpd.mean():.1f}  "
                  f"max: {cpd.max()}  p95: {cpd.quantile(0.95):.0f}")

        print("=" * 60)