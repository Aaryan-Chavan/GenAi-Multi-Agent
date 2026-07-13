from __future__ import annotations

import re
import math
import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict

import numpy as np


class ContextCompressor:
    """
    ==========================================================
                Enterprise Context Compressor
    ==========================================================

    Responsibilities
    ----------------
    ✓ Remove duplicate chunks
    ✓ Remove near duplicate chunks
    ✓ Rank retrieved chunks
    ✓ Increase diversity
    ✓ Apply token budget
    ✓ Produce LLM-ready context

    Pipeline

        Retriever
             ↓
      Exact Deduplication
             ↓
      Near Duplicate Removal
             ↓
      Hybrid Reranking
             ↓
      Diversity Selection
             ↓
      Token Budget
             ↓
      Prompt Context

    Expected Input
    --------------

    [
        {
            "chunk_id": "...",
            "chunk_text": "...",
            "semantic_score":0.92,
            "topic":"battery",
            "record_id":123
        }
    ]
    """

    ###############################################################
    # Initialization
    ###############################################################

    def __init__(

        self,

        max_context_chunks: int = 20,

        max_context_tokens: int = 3000,

        similarity_threshold: float = 0.92,

        diversity: bool = True,

        enable_logging: bool = False

    ):

        self.max_context_chunks = max_context_chunks

        self.max_context_tokens = max_context_tokens

        self.similarity_threshold = similarity_threshold

        self.diversity = diversity

        self.logger = logging.getLogger(__name__)

        if enable_logging:

            logging.basicConfig(

                level=logging.INFO,

                format="%(message)s"

            )

        self.reset_statistics()

    ###############################################################
    # Statistics
    ###############################################################

    def reset_statistics(self):

        self.stats = {

            "input_chunks": 0,

            "after_exact_duplicate": 0,

            "after_similarity": 0,

            "after_reranking": 0,

            "after_diversity": 0,

            "final_chunks": 0,

            "tokens": 0

        }

    ###############################################################
    # Utilities
    ###############################################################

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text before comparison.
        """

        if text is None:
            return ""

        text = str(text).lower()

        text = re.sub(r"\s+", " ", text)

        text = re.sub(r"[^\w\s]", "", text)

        return text.strip()

    ###############################################################

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Approximate token count.

        GPT
        Llama
        Qwen

        ~1 token ≈ 0.75 words
        """

        if not text:
            return 0

        words = len(text.split())

        return math.ceil(words * 1.35)

    ###############################################################

    @staticmethod
    def get_semantic_score(
        chunk: Dict[str, Any]
    ) -> float:

        return float(

            chunk.get(

                "semantic_score",

                chunk.get(

                    "score",

                    0.0

                )

            )

        )

    ###############################################################

    @staticmethod
    def chunk_length(
        chunk: Dict
    ) -> int:

        return len(

            chunk.get(

                "chunk_text",

                ""

            )

        )

    ###############################################################

    @staticmethod
    def get_document_id(
        chunk: Dict
    ):

        """
        Used for diversity.

        Prefer record_id

        fallback document_id

        fallback product_id
        """

        return (

            chunk.get("record_id")

            or chunk.get("document_id")

            or chunk.get("product_id")

            or chunk.get("parent_asin")

        )

    ###############################################################
    # Exact Duplicate Removal
    ###############################################################

    def remove_exact_duplicates(

        self,

        chunks: List[Dict]

    ) -> List[Dict]:

        """
        Removes duplicate chunk_ids.

        Then removes duplicate text.

        Keeps highest semantic score.
        """

        if not chunks:

            return []

        best_chunks = {}

        text_lookup = {}

        for chunk in chunks:

            chunk_id = chunk.get("chunk_id")

            text = self.normalize_text(

                chunk.get(

                    "chunk_text",

                    ""

                )

            )

            score = self.get_semantic_score(chunk)

            ###################################################

            if chunk_id not in best_chunks:

                best_chunks[chunk_id] = chunk

            else:

                old_score = self.get_semantic_score(

                    best_chunks[chunk_id]

                )

                if score > old_score:

                    best_chunks[chunk_id] = chunk

            ###################################################

            if text not in text_lookup:

                text_lookup[text] = chunk

            else:

                previous = text_lookup[text]

                if score > self.get_semantic_score(previous):

                    text_lookup[text] = chunk

        unique = list(text_lookup.values())

        unique.sort(

            key=self.get_semantic_score,

            reverse=True

        )

        self.stats["after_exact_duplicate"] = len(unique)

        return unique

    ###############################################################
    # Basic Summary
    ###############################################################

    def summary(self):

        print()

        print("=" * 60)

        print("CONTEXT COMPRESSION SUMMARY")

        print("=" * 60)

        for key, value in self.stats.items():

            print(f"{key:30}: {value}")

        print("=" * 60)
        ###############################################################
# Similarity Utilities
###############################################################

from difflib import SequenceMatcher


def text_similarity(
    self,
    text1: str,
    text2: str
) -> float:
    """
    Fast lexical similarity.

    Much faster than computing embeddings again.
    """

    if not text1 or not text2:
        return 0.0

    text1 = self.normalize_text(text1)
    text2 = self.normalize_text(text2)

    return SequenceMatcher(
        None,
        text1,
        text2
    ).ratio()


###############################################################


def is_similar(
    self,
    chunk_a: Dict,
    chunk_b: Dict
) -> bool:
    """
    Determines whether two chunks are essentially
    saying the same thing.
    """

    score = self.text_similarity(

        chunk_a.get("chunk_text", ""),

        chunk_b.get("chunk_text", "")

    )

    return score >= self.similarity_threshold


###############################################################


def metadata_similarity(
    self,
    chunk_a: Dict,
    chunk_b: Dict
) -> float:
    """
    Metadata overlap score.

    Helps keep chunks from different
    products/documents.
    """

    score = 0.0

    keys = [

        "topic",

        "document_type",

        "source_type",

        "product_id",

        "parent_asin"

    ]

    for key in keys:

        if (

            chunk_a.get(key)

            and

            chunk_a.get(key)

            ==

            chunk_b.get(key)

        ):

            score += 0.20

    return min(score, 1.0)


###############################################################
# Near Duplicate Removal
###############################################################

def remove_similar_chunks(

    self,

    chunks: List[Dict]

) -> List[Dict]:

    """
    Removes chunks that are almost identical.

    Keeps the highest semantic score.

    Complexity:

        O(n²)

    But n is typically

        30-80 chunks

    which is negligible.
    """

    if len(chunks) <= 1:

        self.stats["after_similarity"] = len(chunks)

        return chunks

    kept = []

    removed = set()

    ordered = sorted(

        chunks,

        key=self.get_semantic_score,

        reverse=True

    )

    for i, chunk in enumerate(ordered):

        if i in removed:
            continue

        kept.append(chunk)

        for j in range(i + 1, len(ordered)):

            if j in removed:
                continue

            candidate = ordered[j]

            ###################################################

            similar = self.is_similar(

                chunk,

                candidate

            )

            ###################################################

            if not similar:

                continue

            ###################################################

            meta = self.metadata_similarity(

                chunk,

                candidate

            )

            ###################################################

            if meta >= 0.40:

                removed.add(j)

            else:

                similarity = self.text_similarity(

                    chunk["chunk_text"],

                    candidate["chunk_text"]

                )

                if similarity > 0.97:

                    removed.add(j)

    self.stats["after_similarity"] = len(kept)

    return kept


###############################################################
# Sorting
###############################################################

def sort_by_score(

    self,

    chunks: List[Dict]

) -> List[Dict]:

    """
    Highest semantic score first.
    """

    chunks.sort(

        key=self.get_semantic_score,

        reverse=True

    )

    return chunks


###############################################################
# Debug Similarity
###############################################################

def print_similarity(

    self,

    chunks: List[Dict]

):

    """
    Debug utility.

    Prints highly similar chunks.
    """

    n = len(chunks)

    print("\nSimilarity Matrix\n")

    for i in range(n):

        for j in range(i + 1, n):

            sim = self.text_similarity(

                chunks[i]["chunk_text"],

                chunks[j]["chunk_text"]

            )

            if sim > self.similarity_threshold:

                print(

                    f"{chunks[i].get('chunk_id')}"

                    " <-> "

                    f"{chunks[j].get('chunk_id')}"

                    f"  {sim:.3f}"

                )
                ###############################################################
# Review Quality Score
###############################################################

def quality_score(
    self,
    chunk: Dict
) -> float:
    """
    Score how informative a chunk is.
    """

    score = 0.0

    text = chunk.get("chunk_text", "")

    word_count = len(text.split())

    if word_count >= 40:
        score += 0.25

    elif word_count >= 20:
        score += 0.15

    sentiment = str(
        chunk.get(
            "sentiment_label",
            ""
        )
    ).lower()

    if sentiment in [
        "positive",
        "negative"
    ]:
        score += 0.15

    topic = chunk.get("topic")

    if topic:
        score += 0.15

    keywords = chunk.get("keywords")

    if keywords:
        score += 0.10

    rating = chunk.get("rating")

    try:
        rating = float(rating)

        if rating >= 4:
            score += 0.10

    except:
        pass

    return min(score, 1.0)
###############################################################
# Recency Score
###############################################################

from datetime import datetime


def recency_score(
    self,
    chunk: Dict
) -> float:

    review_date = chunk.get(
        "review_date"
    )

    if not review_date:
        return 0.0

    try:

        date = datetime.fromisoformat(
            str(review_date)
        )

    except Exception:

        return 0.0

    days = (

        datetime.now()

        - date

    ).days

    if days < 30:
        return 1.0

    if days < 180:
        return 0.8

    if days < 365:
        return 0.5

    return 0.2

###############################################################
# Final Hybrid Score
###############################################################

def hybrid_score(
    self,
    chunk: Dict
) -> float:
    """
    Final score used for reranking.
    """

    semantic = self.get_semantic_score(chunk)

    quality = self.quality_score(chunk)

    recency = self.recency_score(chunk)

    final_score = (

        semantic * 0.70 +

        quality * 0.20 +

        recency * 0.10

    )

    return round(
        final_score,
        5
    )

###############################################################
# Hybrid Ranking
###############################################################

def rerank_chunks(
    self,
    chunks: List[Dict]
) -> List[Dict]:

    for chunk in chunks:

        chunk["hybrid_score"] = self.hybrid_score(
            chunk
        )

    chunks.sort(

        key=lambda x:

        x["hybrid_score"],

        reverse=True

    )

    return chunks

###############################################################
# Topic Diversity
###############################################################

def diversify_topics(
    self,
    chunks: List[Dict]
) -> List[Dict]:

    seen = set()

    results = []

    for chunk in chunks:

        topic = chunk.get(
            "topic"
        )

        if topic:

            if topic in seen:
                continue

            seen.add(topic)

        results.append(chunk)

    return results

###############################################################
# Document Diversity
###############################################################

def diversify_documents(
    self,
    chunks: List[Dict]
) -> List[Dict]:

    seen = set()

    output = []

    for chunk in chunks:

        doc = (

            chunk.get("parent_asin")

            or

            chunk.get("record_id")

        )

        if doc in seen:
            continue

        seen.add(doc)

        output.append(chunk)

    return output

###############################################################
# Context Limiter
###############################################################

def limit_context(
    self,
    chunks: List[Dict]
) -> List[Dict]:

    return chunks[
        :self.max_context_chunks
    ]

###############################################################
# Main Compression Pipeline
###############################################################

def compress(
    self,
    chunks: List[Dict]
) -> List[Dict]:
    """
    Production-ready compression pipeline.
    """

    self.stats = {}

    self.stats["input"] = len(chunks)

    chunks = self.remove_exact_duplicates(
        chunks
    )

    chunks = self.remove_similar_chunks(
        chunks
    )

    chunks = self.rerank_chunks(
        chunks
    )

    chunks = self.diversify_documents(
        chunks
    )

    chunks = self.diversify_topics(
        chunks
    )

    chunks = self.limit_context(
        chunks
    )

    self.stats["final"] = len(chunks)

    return chunks

###############################################################
# Statistics
###############################################################

def summary(self):

    print("\n" + "=" * 60)

    print("CONTEXT COMPRESSOR")

    print("=" * 60)

    for key, value in self.stats.items():

        print(

            f"{key:20}"

            f"{value}"

        )

    print("=" * 60)