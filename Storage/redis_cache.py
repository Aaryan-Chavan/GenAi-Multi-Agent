#redis_cache.py
from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional

import orjson
import redis

from Config.settings import (
    REDIS_HOST,
    REDIS_PORT,
    REDIS_DB,
    CACHE_TTL_SECONDS
)


class RedisCache:

    """
    Production Redis Cache

    Caches

    • Intent classification
    • Retrieval plans
    • DuckDB results
    • Qdrant results
    • Hybrid retrieval
    • Context compression
    • Final LLM answers

    """

    ##########################################################

    def __init__(

        self,

        host=REDIS_HOST,

        port=REDIS_PORT,

        db=REDIS_DB,

        ttl=CACHE_TTL_SECONDS

    ):

        self.ttl = ttl

        self.hits = 0

        self.misses = 0

        self.client = redis.Redis(

            host=host,

            port=port,

            db=db,

            decode_responses=False,

            socket_timeout=5,

            socket_connect_timeout=5,

            retry_on_timeout=True

        )

    ##########################################################
    # Utilities
    ##########################################################

    @staticmethod
    def make_key(

        namespace: str,

        value: str

    ) -> str:

        hashed = hashlib.sha256(

            value.encode()

        ).hexdigest()

        return f"{namespace}:{hashed}"

    ##########################################################

    @staticmethod
    def serialize(obj):

        return orjson.dumps(obj)

    ##########################################################

    @staticmethod
    def deserialize(data):

        if data is None:

            return None

        return orjson.loads(data)

    ##########################################################
    # Basic Operations
    ##########################################################

    def set(

        self,

        key: str,

        value: Any,

        ttl: Optional[int] = None

    ):

        ttl = ttl or self.ttl

        self.client.setex(

            key,

            ttl,

            self.serialize(value)

        )

    ##########################################################

    def get(

        self,

        key: str

    ):

        value = self.client.get(key)

        if value is None:

            self.misses += 1

            return None

        self.hits += 1

        return self.deserialize(value)

    ##########################################################

    def exists(

        self,

        key

    ):

        return bool(

            self.client.exists(key)

        )

    ##########################################################

    def delete(

        self,

        key

    ):

        self.client.delete(key)

    ##########################################################
    # Batch
    ##########################################################

    def batch_set(

        self,

        data: Dict[str, Any],

        ttl=None

    ):

        ttl = ttl or self.ttl

        pipe = self.client.pipeline()

        for key, value in data.items():

            pipe.setex(

                key,

                ttl,

                self.serialize(value)

            )

        pipe.execute()

    ##########################################################

    def batch_get(

        self,

        keys: List[str]

    ):

        values = self.client.mget(keys)

        results = []

        for value in values:

            if value:

                self.hits += 1

                results.append(

                    self.deserialize(value)

                )

            else:

                self.misses += 1

                results.append(None)

        return results

    ##########################################################
    # Prefix Delete
    ##########################################################

    def delete_prefix(

        self,

        prefix

    ):

        cursor = 0

        while True:

            cursor, keys = self.client.scan(

                cursor,

                match=f"{prefix}:*",

                count=1000

            )

            if keys:

                self.client.delete(*keys)

            if cursor == 0:

                break

    ##########################################################
    # Flush
    ##########################################################

    def flush(self):

        self.client.flushdb()

    ##########################################################
    # Statistics
    ##########################################################

    @property
    def hit_rate(self):

        total = self.hits + self.misses

        if total == 0:

            return 0.0

        return self.hits / total

    ##########################################################

    def summary(self):

        print("\n" + "=" * 60)

        print("REDIS CACHE")

        print("=" * 60)

        print(f"Hits          : {self.hits}")

        print(f"Misses        : {self.misses}")

        print(f"Hit Rate      : {self.hit_rate:.2%}")

        print(f"TTL           : {self.ttl}s")

        print("=" * 60)

    ##########################################################
    # Health
    ##########################################################

    def ping(self):

        try:

            return self.client.ping()

        except Exception:

            return False

    ##########################################################
    # Cached Function
    ##########################################################

    def cached(

        self,

        namespace: str,

        query: str,

        fn

    ):

        key = self.make_key(

            namespace,

            query

        )

        cached = self.get(key)

        if cached is not None:

            return cached

        result = fn()

        self.set(

            key,

            result

        )

        return result

    ##########################################################
    # Timing
    ##########################################################

    def timed_cached(

        self,

        namespace,

        query,

        fn

    ):

        start = time.perf_counter()

        result = self.cached(

            namespace,

            query,

            fn

        )

        elapsed = time.perf_counter() - start

        return result, elapsed

    ##########################################################
    # Context Manager
    ##########################################################

    def close(self):

        self.client.close()

    def __enter__(self):

        return self

    def __exit__(

        self,

        exc_type,

        exc,

        tb

    ):

        self.close()