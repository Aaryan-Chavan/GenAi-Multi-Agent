"""
===============================================================================
Semantic Agent Test Suite
===============================================================================

Tests the SemanticAgent against a variety of semantic search queries.

Author : OpenAI
===============================================================================
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from typing import List

from Agents.semantic_agent import (
    SemanticAgent,
    QdrantConfig,
    EmbedConfig,
    RetrievalConfig,
    SchemaConfig,
    LLMConfig,
)

###############################################################################
# AGENT CONFIGURATION
###############################################################################

QDRANT_CFG = QdrantConfig(
    host="http://localhost:6333",
    collection="documents",
)

EMBED_CFG = EmbedConfig(
    model_name="BAAI/bge-small-en-v1.5",
)

RETRIEVAL_CFG = RetrievalConfig()

SCHEMA_CFG = SchemaConfig()

LLM_CFG = LLMConfig(
    enabled=True,
    base_url="http://localhost:11434",
    model="qwen3:14b",
)

###############################################################################
# SEMANTIC TEST QUERIES
###############################################################################

TEST_QUERIES: List[str] = [

    ###########################################################################
    # BRAND REVIEWS
    ###########################################################################

    "Blink security camera reviews",
    "Ring doorbell reviews",
    "Sony headphone reviews",
    "Logitech mouse reviews",
    "Samsung SSD reviews",
    "Apple AirPods reviews",

    ###########################################################################
    # PRODUCT SEARCH
    ###########################################################################

    "Wireless routers",
    "Gaming keyboards",
    "Bluetooth speakers",
    "Home security cameras",
    "External hard drives",
    "Noise cancelling headphones",

    ###########################################################################
    # FEATURE SEARCH
    ###########################################################################

    "Products with long battery life",
    "Easy installation products",
    "Motion detection cameras",
    "Cloud storage cameras",
    "Alexa compatible products",
    "Waterproof speakers",

    ###########################################################################
    # SENTIMENT
    ###########################################################################

    "Very positive reviews",
    "Very negative reviews",
    "Happy customers",
    "Angry customers",
    "Highly recommended products",
    "Products customers regret buying",

    ###########################################################################
    # COMPLAINTS
    ###########################################################################

    "Delivery complaints",
    "Refund complaints",
    "Return issues",
    "Poor customer support",
    "Broken products",
    "Battery problems",
    "Connectivity issues",

    ###########################################################################
    # QUALITY
    ###########################################################################

    "Excellent quality products",
    "Poor build quality",
    "Reliable electronics",
    "Durable products",
    "Cheap plastic products",

    ###########################################################################
    # VALUE
    ###########################################################################

    "Good value for money",
    "Expensive products worth buying",
    "Budget security cameras",
    "Affordable electronics",

    ###########################################################################
    # COMPARISON
    ###########################################################################

    "Blink vs Ring",
    "Apple vs Samsung",
    "Sony vs Bose headphones",
    "Best wireless mouse",

    ###########################################################################
    # NATURAL LANGUAGE
    ###########################################################################

    "I need a security camera with excellent battery life.",

    "I want a router with stable WiFi connection.",

    "Recommend an affordable Bluetooth speaker.",

    "I need a gaming keyboard with great build quality.",

    "Suggest products customers love.",

    ###########################################################################
    # META QUERIES
    ###########################################################################

    "What products receive the most praise?",

    "What products receive the most complaints?",

    "What features do customers like?",

    "What are the biggest complaints about Blink cameras?",

    "What do customers think about NETGEAR?",

]
###############################################################################
# HELPER FUNCTIONS
###############################################################################

MAX_RESULTS_TO_PRINT = 10


def line():
    print("=" * 110)


def header(title: str):
    line()
    print(title)
    line()


def section(title: str):
    print("\n" + "-" * 110)
    print(title)
    print("-" * 110)


###############################################################################
# PRINT FUNCTIONS
###############################################################################

def print_success(success: bool):

    if success:
        print("\n✅ SUCCESS")
    else:
        print("\n❌ FAILED")


def print_query(query: str):

    header("QUERY")
    print(query)


def print_execution_time(ms: float):

    section("Execution Time")
    print(f"{ms:.2f} ms")


def print_entities(result):

    section("Entities")

    entities = getattr(result, "entities", None)

    if not entities:
        print("No entities detected.")
        return

    if isinstance(entities, dict):

        print(json.dumps(
            entities,
            indent=2,
            default=str
        ))

    else:

        for entity in entities:
            print(entity)


def print_metadata(result):

    section("Metadata")

    metadata = getattr(result, "metadata", None)

    if not metadata:
        print("No metadata.")
        return

    if isinstance(metadata, dict):

        for key, value in metadata.items():

            if isinstance(value, dict):

                print(f"{key}:")

                for k2, v2 in value.items():
                    print(f"    {k2}: {v2}")

            elif isinstance(value, list):

                print(f"{key}: {len(value)} items")

            else:

                print(f"{key}: {value}")

    else:

        print(metadata)


###############################################################################
# CHUNK PRINTING
###############################################################################

def print_chunk(chunk, index):

    print("\n" + "-" * 110)
    print(f"Result #{index}")

    if not isinstance(chunk, dict):
        print(chunk)
        return

    if "score" in chunk:
        print(f"Score      : {chunk['score']:.4f}")

    if "document_id" in chunk:
        print(f"Document   : {chunk['document_id']}")

    if "chunk_id" in chunk:
        print(f"Chunk ID   : {chunk['chunk_id']}")

    if "brand" in chunk:
        print(f"Brand      : {chunk['brand']}")

    if "category" in chunk:
        print(f"Category   : {chunk['category']}")

    if "product_id" in chunk:
        print(f"Product ID : {chunk['product_id']}")

    if "product_title" in chunk:
        print(f"Title      : {chunk['product_title']}")

    if "rating" in chunk:
        print(f"Rating     : {chunk['rating']}")

    if "verified_purchase" in chunk:
        print(f"Verified   : {chunk['verified_purchase']}")

    text = (
        chunk.get("chunk_text")
        or chunk.get("text")
        or chunk.get("content")
        or ""
    )

    if text:

        print("\nText")
        print("-" * 80)

        if len(text) > 1000:
            print(text[:1000] + " ...")
        else:
            print(text)


###############################################################################
# RETRIEVED CHUNKS
###############################################################################

def print_chunks(result):

    section("Retrieved Chunks")

    chunks = getattr(result, "chunks", None)

    if not chunks:
        print("No chunks retrieved.")
        return

    print(f"Total Retrieved : {len(chunks)}")

    scores = [
        c["score"]
        for c in chunks
        if isinstance(c, dict) and "score" in c
    ]

    if scores:

        print(f"Highest Score : {max(scores):.4f}")
        print(f"Lowest Score  : {min(scores):.4f}")
        print(f"Average Score : {statistics.mean(scores):.4f}")

    print()

    limit = min(MAX_RESULTS_TO_PRINT, len(chunks))

    for i in range(limit):
        print_chunk(chunks[i], i + 1)

    if len(chunks) > limit:
        print(f"\n... {len(chunks)-limit} more chunks")


###############################################################################
# CONFIDENCE
###############################################################################

def print_confidence(result):

    section("Confidence")

    found = False

    attrs = [

        "retrieval_confidence",

        "semantic_confidence",

        "embedding_confidence",

        "overall_confidence",

    ]

    for attr in attrs:

        if hasattr(result, attr):

            found = True

            print(f"{attr:<30}: {getattr(result, attr)}")

    if not found:
        print("Not available.")


###############################################################################
# ERRORS
###############################################################################

def print_error(result):

    if getattr(result, "success", True):
        return

    section("Error")

    if hasattr(result, "error"):
        print(result.error)

    elif hasattr(result, "errors"):
        print(result.errors)

    else:
        print("Unknown error.")


###############################################################################
# COMPLETE RESULT
###############################################################################

def print_result(result):

    print_success(getattr(result, "success", False))

    latency = getattr(result, "latency_ms", None)

    if latency is not None:
        print_execution_time(latency)

    print_confidence(result)

    print_entities(result)

    print_metadata(result)

    print_chunks(result)

    print_error(result)
###############################################################################
# BENCHMARK
###############################################################################

class Benchmark:

    def __init__(self):

        self.total = 0
        self.success = 0
        self.failed = 0

        self.total_latency = 0.0

        self.total_chunks = 0

        self.all_scores = []

        self.confidences = []

    ###########################################################################
    # Update benchmark after every query
    ###########################################################################

    def update(self, result):

        self.total += 1

        if getattr(result, "success", False):
            self.success += 1
        else:
            self.failed += 1

        #######################################################################
        # Latency
        #######################################################################

        latency = getattr(result, "latency_ms", None)

        if latency is not None:
            self.total_latency += latency

        #######################################################################
        # Retrieved chunks
        #######################################################################

        chunks = getattr(result, "chunks", [])

        self.total_chunks += len(chunks)

        #######################################################################
        # Similarity scores
        #######################################################################

        for chunk in chunks:

            if isinstance(chunk, dict):

                score = chunk.get("score")

                if score is not None:

                    self.all_scores.append(score)

        #######################################################################
        # Confidence
        #######################################################################

        if hasattr(result, "overall_confidence"):

            confidence = getattr(result, "overall_confidence")

            if confidence is not None:

                self.confidences.append(confidence)

    ###########################################################################
    # Print benchmark summary
    ###########################################################################

    def summary(self):

        header("SEMANTIC BENCHMARK SUMMARY")

        print(f"Total Tests              : {self.total}")

        print(f"Successful               : {self.success}")

        print(f"Failed                   : {self.failed}")

        print()

        #######################################################################
        # Success Rate
        #######################################################################

        if self.total:

            success_rate = (self.success / self.total) * 100

            print(f"Success Rate             : {success_rate:.2f}%")

            print(
                f"Average Latency          : "
                f"{self.total_latency/self.total:.2f} ms"
            )

            print(
                f"Average Retrieved Chunks : "
                f"{self.total_chunks/self.total:.2f}"
            )

        #######################################################################
        # Similarity statistics
        #######################################################################

        if self.all_scores:

            print()

            print("Similarity Scores")

            print("------------------------------")

            print(
                f"Average Score            : "
                f"{statistics.mean(self.all_scores):.4f}"
            )

            print(
                f"Highest Score            : "
                f"{max(self.all_scores):.4f}"
            )

            print(
                f"Lowest Score             : "
                f"{min(self.all_scores):.4f}"
            )

        #######################################################################
        # Confidence statistics
        #######################################################################

        if self.confidences:

            print()

            print(
                f"Average Confidence       : "
                f"{statistics.mean(self.confidences):.4f}"
            )

            print(
                f"Maximum Confidence       : "
                f"{max(self.confidences):.4f}"
            )

            print(
                f"Minimum Confidence       : "
                f"{min(self.confidences):.4f}"
            )

        print()

        line()

###############################################################################
# CORE TEST EXECUTION
###############################################################################

async def run_single_test(
    agent: SemanticAgent,
    query: str,
    benchmark: Benchmark,
):

    print("\n" + "=" * 110)
    print("RUNNING TEST")
    print("=" * 110)

    print_query(query)

    start = time.perf_counter()

    try:

        #######################################################################
        # Run Semantic Agent
        #######################################################################

        result = await agent.run(query)

        end = time.perf_counter()

        #######################################################################
        # Attach latency if agent didn't
        #######################################################################

        if not hasattr(result, "latency_ms"):

            result.latency_ms = (end - start) * 1000

        #######################################################################
        # Print everything
        #######################################################################

        print_result(result)

        #######################################################################
        # Update benchmark
        #######################################################################

        benchmark.update(result)

        return result

    except Exception as e:

        end = time.perf_counter()

        print("\n❌ EXCEPTION OCCURRED")
        print(str(e))

        #######################################################################
        # Dummy Result
        #######################################################################

        class DummyResult:

            success = False

            latency_ms = (end - start) * 1000

            chunks = []

            metadata = {}

            entities = {}

            error = str(e)

            overall_confidence = 0.0

        dummy = DummyResult()

        benchmark.update(dummy)

        print_result(dummy)

        return dummy


###############################################################################
# RUN ALL TESTS
###############################################################################

async def run_all_tests(agent: SemanticAgent):

    benchmark = Benchmark()

    total = len(TEST_QUERIES)

    for index, query in enumerate(TEST_QUERIES, start=1):

        print("\n\n")
        print(f"TEST {index}/{total}")

        await run_single_test(
            agent,
            query,
            benchmark,
        )

    ###########################################################################
    # Final Summary
    ###########################################################################

    benchmark.summary()


###############################################################################
# RUN CUSTOM QUERY
###############################################################################

async def run_custom_query(agent: SemanticAgent):

    benchmark = Benchmark()

    while True:

        print()

        query = input(
            "\nEnter Semantic Query (or 'back'): "
        ).strip()

        if query.lower() == "back":
            break

        if not query:
            continue

        await run_single_test(
            agent,
            query,
            benchmark,
        )

    ###########################################################################
    # Session Summary
    ###########################################################################

    if benchmark.total:

        benchmark.summary()
###############################################################################
# INTERACTIVE MENU
###############################################################################

def print_menu():

    print()

    header("SEMANTIC AGENT TEST MENU")

    print("1. Run single benchmark query")
    print("2. Run all benchmark queries")
    print("3. Run custom semantic queries")
    print("4. Exit")


async def run_interactive(agent: SemanticAgent):

    benchmark = Benchmark()

    while True:

        print_menu()

        choice = input("\nEnter choice: ").strip()

        #######################################################################
        # Run one benchmark query
        #######################################################################

        if choice == "1":

            print()

            print("Available Queries\n")

            for i, q in enumerate(TEST_QUERIES, start=1):
                print(f"{i}. {q}")

            try:

                idx = int(
                    input("\nEnter query number: ")
                ) - 1

            except ValueError:

                print("Invalid input.")
                continue

            if 0 <= idx < len(TEST_QUERIES):

                await run_single_test(
                    agent,
                    TEST_QUERIES[idx],
                    benchmark,
                )

            else:

                print("Invalid query number.")

        #######################################################################
        # Run complete benchmark
        #######################################################################

        elif choice == "2":

            await run_all_tests(agent)

        #######################################################################
        # Custom queries
        #######################################################################

        elif choice == "3":

            await run_custom_query(agent)

        #######################################################################
        # Exit
        #######################################################################

        elif choice == "4":

            print("\nClosing Semantic Agent...")
            break

        else:

            print("\nInvalid choice.")


###############################################################################
# MAIN
###############################################################################

async def main():

    print("\n🚀 Initializing Semantic Agent...\n")

    agent = SemanticAgent(

        qdrant_cfg=QDRANT_CFG,

        embed_cfg=EMBED_CFG,

        retrieval_cfg=RETRIEVAL_CFG,

        schema_cfg=SCHEMA_CFG,

        llm_cfg=LLM_CFG,

    )

    print("✅ Semantic Agent initialized successfully.")

    mode = input(

        "\nChoose mode:\n"
        "1. Interactive\n"
        "2. Run all benchmark queries\n\n"
        "Enter choice: "

    ).strip()

    try:

        if mode == "1":

            await run_interactive(agent)

        else:

            await run_all_tests(agent)

    finally:

        print("\nClosing agent...")

        await agent.close()

        print("Done.")


###############################################################################
# ENTRY POINT
###############################################################################

if __name__ == "__main__":

    asyncio.run(main())