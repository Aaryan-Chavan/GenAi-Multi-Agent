"""
Standalone Redis Cache Tester
=============================

Completely independent from the project.

Tests

✓ Connection
✓ Set / Get
✓ Exists
✓ Delete
✓ Batch Set
✓ Batch Get
✓ Prefix Delete
✓ TTL
✓ Serialization
✓ Cached Function
✓ Timed Cached Function
✓ Performance
✓ Statistics
✓ Cleanup

"""

import random
import string
import time
from pprint import pprint

from Storage.redis_cache import RedisCache


# -------------------------------------------------------
# Utilities
# -------------------------------------------------------

def banner(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def success(msg):
    print(f"✅ {msg}")


def failure(msg):
    print(f"❌ {msg}")


def random_string(length=12):
    return "".join(random.choice(string.ascii_letters) for _ in range(length))


# -------------------------------------------------------
# Sample Complex Object
# -------------------------------------------------------

sample_object = {
    "user": {
        "id": 101,
        "name": "Alice",
        "age": 27,
        "active": True,
    },
    "orders": [
        {
            "id": 1,
            "price": 99.5,
            "items": ["A", "B", "C"]
        },
        {
            "id": 2,
            "price": 145.2,
            "items": ["D"]
        }
    ],
    "metadata": {
        "country": "India",
        "city": "Mumbai"
    }
}


# -------------------------------------------------------
# Fake expensive function
# -------------------------------------------------------

def expensive_function():

    time.sleep(1)

    return {
        "result": random.randint(1000, 9999),
        "timestamp": time.time()
    }


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():

    banner("REDIS CACHE TEST")

    cache = RedisCache(ttl=5)

    # ---------------------------------------------------

    banner("1. CONNECTION")

    if cache.ping():
        success("Redis Connected")
    else:
        failure("Redis NOT running")
        return

    # ---------------------------------------------------

    banner("2. SIMPLE SET / GET")

    key = cache.make_key("simple", "hello")

    cache.set(key, sample_object)

    value = cache.get(key)

    pprint(value)

    # ---------------------------------------------------

    banner("3. EXISTS")

    print(cache.exists(key))

    # ---------------------------------------------------

    banner("4. DELETE")

    cache.delete(key)

    print(cache.exists(key))

    # ---------------------------------------------------

    banner("5. BATCH SET")

    batch = {}

    for i in range(20):

        batch[
            cache.make_key("batch", str(i))
        ] = {

            "index": i,

            "value": random_string()

        }

    cache.batch_set(batch)

    success("Batch Stored")

    # ---------------------------------------------------

    banner("6. BATCH GET")

    values = cache.batch_get(list(batch.keys()))

    print("Objects Retrieved :", len(values))

    # ---------------------------------------------------

    banner("7. PREFIX DELETE")

    cache.delete_prefix("batch")

    remaining = cache.batch_get(list(batch.keys()))

    print("Remaining Objects :", sum(v is not None for v in remaining))

    # ---------------------------------------------------

    banner("8. TTL TEST")

    ttl_key = cache.make_key("ttl", "demo")

    cache.set(ttl_key, {"hello": "world"}, ttl=3)

    print("Immediately :", cache.get(ttl_key))

    print("Waiting 4 seconds...")

    time.sleep(4)

    print("After TTL :", cache.get(ttl_key))

    # ---------------------------------------------------

    banner("9. LARGE OBJECT")

    large = {

        "numbers": list(range(50000)),

        "text": random_string(100000)

    }

    large_key = cache.make_key("large", "1")

    start = time.perf_counter()

    cache.set(large_key, large)

    write_time = time.perf_counter() - start

    start = time.perf_counter()

    cache.get(large_key)

    read_time = time.perf_counter() - start

    print(f"Write Time : {write_time:.4f} sec")

    print(f"Read Time  : {read_time:.4f} sec")

    # ---------------------------------------------------

    banner("10. CACHED FUNCTION")

    start = time.perf_counter()

    result1 = cache.cached(

        "expensive",

        "query1",

        expensive_function

    )

    t1 = time.perf_counter() - start

    start = time.perf_counter()

    result2 = cache.cached(

        "expensive",

        "query1",

        expensive_function

    )

    t2 = time.perf_counter() - start

    print()

    print("First Call")

    pprint(result1)

    print("Time :", round(t1, 4), "sec")

    print()

    print("Second Call (Cached)")

    pprint(result2)

    print("Time :", round(t2, 6), "sec")

    # ---------------------------------------------------

    banner("11. TIMED CACHE")

    result, elapsed = cache.timed_cached(

        "timed",

        "query",

        expensive_function

    )

    pprint(result)

    print("Elapsed :", elapsed)

    # ---------------------------------------------------

    banner("12. PERFORMANCE TEST")

    N = 10000

    start = time.perf_counter()

    for i in range(N):

        cache.set(

            cache.make_key("perf", str(i)),

            {

                "i": i,

                "value": random.random()

            }

        )

    write = time.perf_counter() - start

    start = time.perf_counter()

    for i in range(N):

        cache.get(

            cache.make_key("perf", str(i))

        )

    read = time.perf_counter() - start

    print(f"SET {N:,} keys : {write:.3f} sec")

    print(f"GET {N:,} keys : {read:.3f} sec")

    print(f"Average SET : {(write/N)*1000:.4f} ms")

    print(f"Average GET : {(read/N)*1000:.4f} ms")

    # ---------------------------------------------------

    banner("13. MEMORY USAGE")

    info = cache.client.info("memory")

    print("Used Memory      :", info["used_memory_human"])

    print("Peak Memory      :", info["used_memory_peak_human"])

    print("Dataset Memory   :", info["used_memory_dataset_human"])

    # ---------------------------------------------------

    banner("14. CACHE STATISTICS")

    cache.summary()

    # ---------------------------------------------------

    banner("15. CLEANUP")

    cache.delete_prefix("perf")
    cache.delete_prefix("large")
    cache.delete_prefix("timed")
    cache.delete_prefix("expensive")
    cache.delete_prefix("ttl")
    cache.delete_prefix("simple")

    success("Cleanup Finished")

    cache.close()

    banner("ALL TESTS COMPLETED")


# -------------------------------------------------------

if __name__ == "__main__":

    main()