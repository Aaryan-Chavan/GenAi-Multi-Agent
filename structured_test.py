"""
===============================================================================
Structured Agent Test Suite
===============================================================================

Tests the StructuredAgent against a variety of analytical SQL queries.

Author : OpenAI
===============================================================================
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import List

from Agents.structured_agent import (
    StructuredAgent,
    AgentConfig,
)

###############################################################################
# DATABASE CONFIGURATION
###############################################################################

DATABASE_PATH = r"F:\PROJECT\Database\Duckdb\analytics.duckdb"

CONFIG = AgentConfig()
###############################################################################
# TEST QUERIES
###############################################################################

TEST_QUERIES: List[str] = [

    ###########################################################################
    # BASIC
    ###########################################################################

    "Show all brands.",

    "Show all categories.",

    "Count total reviews.",

    "Count verified purchases.",

    "Count non verified purchases.",

    ###########################################################################
    # AGGREGATION
    ###########################################################################

    "Average rating by brand.",

    "Average price by category.",

    "Average helpful votes by brand.",

    "Maximum rating by category.",

    "Minimum price by brand.",

    ###########################################################################
    # GROUP BY
    ###########################################################################

    "Show review count by brand.",

    "Show review count by category.",

    "Show average rating grouped by brand.",

    "Show average rating grouped by category.",

    ###########################################################################
    # FILTERS
    ###########################################################################

    "Show products with rating greater than 4.",

    "Show verified purchases with rating equal to 5.",

    "Show Samsung products.",

    "Show Apple products with rating above 4.",

    ###########################################################################
    # SORTING
    ###########################################################################

    "Top 10 brands by average rating.",

    "Top 10 categories by review count.",

    "Top 10 products by helpful votes.",

    "Highest rated brands.",

    ###########################################################################
    # HAVING
    ###########################################################################

    "Brands having more than 100 reviews.",

    "Brands having at least 500 reviews ordered by average rating.",

    "Categories having average rating above 4.",

    ###########################################################################
    # MULTI METRIC
    ###########################################################################

    (
        "For each brand show average rating, "
        "average helpful votes and total reviews."
    ),

    (
        "For each category show average price, "
        "average rating and review count."
    ),

    ###########################################################################
    # TIME
    ###########################################################################

    "Monthly review count.",

    "Yearly review count.",

    "Monthly average rating.",

    ###########################################################################
    # COMPLEX
    ###########################################################################

    (
        "For each brand show average rating, average helpful votes, "
        "verified purchase percentage and review count."
    ),

    (
        "Return brands having at least 500 reviews ordered by "
        "average rating descending."
    ),

    (
        "Find the top five brands by review count in every category."
    ),

    (
        "Show categories whose average price is greater than "
        "the overall average price."
    ),

    (
        "Find brands whose average rating is above the global average."
    ),

    (
        "Compare Apple and Samsung average ratings."
    ),

    (
        "Show review trend by month for each brand."
    ),

    (
        "Rank brands by average helpful votes."
    ),

    (
        "Find brands with the highest verified purchase percentage."
    ),

]
###############################################################################
# HELPER FUNCTIONS
###############################################################################

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
# RESULT PRINTING
###############################################################################

def print_success(success: bool):
    if success:
        print("\n✅ SUCCESS")
    else:
        print("\n❌ FAILED")


def print_query(query: str):
    header("QUERY")
    print(query)


def print_execution_time(ms):
    section("Execution Time")
    print(f"{ms:.2f} ms")


def print_sql(sql: str):

    section("Generated SQL")

    if not sql:
        print("<No SQL Generated>")
        return

    print(sql)


def print_rows(rows):

    section("Rows")

    if not rows:
        print("No rows returned.")
        return

    max_rows = min(len(rows), 10)

    for i in range(max_rows):
        print(rows[i])

    if len(rows) > max_rows:
        print(f"\n... {len(rows)-max_rows} more rows")


def print_columns(columns):

    section("Columns")

    if not columns:
        print("No columns.")
        return

    for c in columns:
        print("-", c)


def print_tables(tables):

    section("Tables Used")

    if not tables:
        print("Unknown")
        return

    for t in tables:
        print("-", t)


def print_confidence(result):

    section("Confidence")

    attrs = [
        "planner_confidence",
        "schema_confidence",
        "sql_confidence",
        "repair_confidence",
        "execution_confidence",
        "overall_confidence",
    ]

    for attr in attrs:
        if hasattr(result, attr):
            value = getattr(result, attr)
            print(f"{attr:<30}: {value}")


def print_metadata(metadata):

    section("Metadata")

    if not metadata:
        print("No metadata.")
        return

    if isinstance(metadata, dict):

        for k, v in metadata.items():

            if isinstance(v, dict):

                print(f"{k}:")

                for k2, v2 in v.items():
                    print(f"    {k2}: {v2}")

            elif isinstance(v, list):

                print(f"{k}: {len(v)} items")

            else:
                print(f"{k}: {v}")

    else:
        print(metadata)


def print_error(result):

    if getattr(result, "success", True):
        return

    section("Error")

    if hasattr(result, "error"):
        print(result.error)

    elif hasattr(result, "errors"):
        print(result.errors)

    else:
        print("Unknown error")


###############################################################################
# RESULT SUMMARY
###############################################################################

def print_result(result):

    print_success(getattr(result, "success", False))

    if hasattr(result, "latency_ms"):
        print_execution_time(result.latency_ms)

    elif hasattr(result, "execution_time_ms"):
        print_execution_time(result.execution_time_ms)

    if hasattr(result, "sql"):
        print_sql(result.sql)

    if hasattr(result, "tables_used"):
        print_tables(result.tables_used)

    elif hasattr(result, "join_path"):
        print_tables(result.join_path)

    if hasattr(result, "columns"):
        print_columns(result.columns)

    elif hasattr(result, "returned_fields"):
        print_columns(result.returned_fields)

    if hasattr(result, "rows"):
        print_rows(result.rows)

    elif hasattr(result, "data"):
        print_rows(result.data)

    print_confidence(result)

    if hasattr(result, "metadata"):
        print_metadata(result.metadata)

    print_error(result)


###############################################################################
# BENCHMARK HELPERS
###############################################################################

class Benchmark:

    def __init__(self):

        self.total = 0
        self.success = 0
        self.failed = 0

        self.total_time = 0.0

        self.confidences = []

    def update(self, result):

        self.total += 1

        if getattr(result, "success", False):
            self.success += 1
        else:
            self.failed += 1

        if hasattr(result, "latency_ms"):
            self.total_time += result.latency_ms

        elif hasattr(result, "execution_time_ms"):
            self.total_time += result.execution_time_ms

        if hasattr(result, "overall_confidence"):
            self.confidences.append(result.overall_confidence)

    def summary(self):

        header("BENCHMARK SUMMARY")

        print(f"Tests Run        : {self.total}")
        print(f"Passed          : {self.success}")
        print(f"Failed          : {self.failed}")

        if self.total:

            print(
                f"Success Rate    : "
                f"{100*self.success/self.total:.2f}%"
            )

            print(
                f"Average Time    : "
                f"{self.total_time/self.total:.2f} ms"
            )

        if self.confidences:

            avg = sum(self.confidences)/len(self.confidences)

            print(
                f"Avg Confidence  : "
                f"{avg:.3f}"
            )

###############################################################################
# CORE TEST EXECUTION
###############################################################################

async def run_single_test(agent, query: str, benchmark: Benchmark):

    print("\n" + "=" * 110)
    print("RUNNING TEST")
    print("=" * 110)

    print_query(query)

    start = time.time()

    try:

        result = await agent.run(query, debug=True)

        end = time.time()

        # attach timing if not already present
        if not hasattr(result, "latency_ms"):
            result.latency_ms = (end - start) * 1000

        print_result(result)

        benchmark.update(result)

        return result

    except Exception as e:

        end = time.time()

        print("\n❌ EXCEPTION OCCURRED")
        print(str(e))

        class DummyResult:
            success = False
            sql = None
            rows = []
            tables_used = []
            overall_confidence = 0.0
            latency_ms = (end - start) * 1000

        benchmark.update(DummyResult())

        return None


###############################################################################
# RUN ALL TESTS
###############################################################################

async def run_all_tests(agent):

    benchmark = Benchmark()

    for i, query in enumerate(TEST_QUERIES, 1):

        print("\n\n")
        print(f"TEST {i}/{len(TEST_QUERIES)}")

        await run_single_test(agent, query, benchmark)

    benchmark.summary()


###############################################################################
# INTERACTIVE MENU
###############################################################################

def print_menu():

    print("\n")
    header("STRUCTURED AGENT TEST MENU")

    print("1. Run single test")
    print("2. Run all tests")
    print("3. Exit")


async def run_interactive(agent):

    benchmark = Benchmark()

    while True:

        print_menu()

        choice = input("\nEnter choice: ").strip()

        if choice == "1":

            print("\nSelect query index:\n")

            for i, q in enumerate(TEST_QUERIES):
                print(f"{i+1}. {q}")

            idx = int(input("\nEnter query number: ")) - 1

            if 0 <= idx < len(TEST_QUERIES):

                await run_single_test(agent, TEST_QUERIES[idx], benchmark)

            else:
                print("Invalid index")

        elif choice == "2":

            await run_all_tests(agent)

        elif choice == "3":

            print("Exiting...")
            break

        else:

            print("Invalid choice")


###############################################################################
# MAIN ENTRY POINT
###############################################################################

async def main():

    print("\n🚀 Initializing Structured Agent...\n")

    agent = StructuredAgent(
        db_path=DATABASE_PATH,
        config = AgentConfig()
    )

    print("✅ Agent initialized successfully")

    mode = input(
        "\nChoose mode:\n"
        "1. Interactive\n"
        "2. Run all tests\n\n"
        "Enter choice: "
    ).strip()

    if mode == "1":
        await run_interactive(agent)

    else:
        await run_all_tests(agent)


###############################################################################
# ENTRY POINT
###############################################################################

if __name__ == "__main__":
    asyncio.run(main())