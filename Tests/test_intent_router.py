"""
tests/test_intent_router.py
============================================================
Standalone unit tests for intent_router.py

Run with:
    pip install pytest --break-system-packages
    pytest tests/test_intent_router.py -v

No external config file is required — tests use in-memory dicts
via IntentRouter(config=...) so the suite is fully self-contained
and does not depend on intent_config.yaml being present on disk.
============================================================
"""

import json
import logging
import time

import pytest

from Agents.intent_router import (
    ClassifierPlugin,
    ConfigValidationError,
    DatasetMetadata,
    IntentResult,
    IntentRouter,
    RoutingPlugin,
    RouterConfig,
    RuleEnginePlugin,
    SchemaContext,
    TextNormalizer,
    _LRUCache,
    _PluginVerdict,
    _RoutingContext,
    get_router,
    route,
    set_log_level,
)


# ============================================================
# SHARED FIXTURES
# ============================================================

GENERIC_CONFIG = {
    "settings": {
        "dominant_threshold": 1.0,
        "gap_threshold": 0.3,
        "low_confidence_threshold": 0.35,
        "cache_size": 128,
        "normalizer": {
            "strip_accents": True,
            "casefold": True,
            "remove_punctuation": True,
            "collapse_whitespace": True,
        },
        "schema_boosts": {
            "structured": {"numeric_columns": 0.5},
            "trend": {"time_columns": 0.6},
        },
    },
    "intents": {
        "structured": {
            "patterns": [
                [r"\b(count|total|how many|average)\b", 1.0],
            ]
        },
        "trend": {
            "patterns": [
                [r"\b(trend|over time|growth|change)\b", 1.0],
            ]
        },
        "semantic": {
            "patterns": [
                [r"\b(opinion|feedback|summarize|feel about)\b", 1.0],
            ]
        },
    },
    "routing": {
        "structured": {
            "needs_sql": True, "needs_vector": False,
            "needs_time_series": False, "routing_path": "SQL",
        },
        "trend": {
            "needs_sql": True, "needs_vector": False,
            "needs_time_series": True, "routing_path": "SQL",
        },
        "semantic": {
            "needs_sql": False, "needs_vector": True,
            "routing_path": "Qdrant",
            "extra_flags": {"needs_summary": True},
        },
        "hybrid": {
            "needs_sql": True, "needs_vector": True, "routing_path": "Hybrid",
        },
    },
    "training_examples": {},
}


@pytest.fixture
def router():
    return IntentRouter(config=GENERIC_CONFIG, use_cache=True)


@pytest.fixture
def router_no_cache():
    return IntentRouter(config=GENERIC_CONFIG, use_cache=False)


# ============================================================
# CONFIG VALIDATION (RouterConfig)
# ============================================================

class TestRouterConfigValidation:
    def test_valid_config_loads(self):
        cfg = RouterConfig.from_dict(GENERIC_CONFIG)
        assert "structured" in cfg.intents
        assert "hybrid" in cfg.routing

    def test_non_dict_top_level_raises(self):
        with pytest.raises(ConfigValidationError, match="must be a mapping"):
            RouterConfig.from_dict(["not", "a", "dict"])

    def test_missing_intents_raises(self):
        bad = {"routing": {"hybrid": {}}}
        with pytest.raises(ConfigValidationError, match="intents"):
            RouterConfig.from_dict(bad)

    def test_empty_intents_raises(self):
        bad = {"intents": {}, "routing": {"hybrid": {}}}
        with pytest.raises(ConfigValidationError, match="non-empty 'intents'"):
            RouterConfig.from_dict(bad)

    def test_missing_routing_raises(self):
        bad = {"intents": {"x": {"patterns": [["a", 1.0]]}}}
        with pytest.raises(ConfigValidationError, match="routing"):
            RouterConfig.from_dict(bad)

    def test_missing_hybrid_fallback_raises(self):
        bad = {
            "intents": {"x": {"patterns": [["a", 1.0]]}},
            "routing": {"x": {"routing_path": "SQL"}},
        }
        with pytest.raises(ConfigValidationError, match="hybrid"):
            RouterConfig.from_dict(bad)

    def test_intent_without_routing_entry_raises(self):
        bad = {
            "intents": {
                "x": {"patterns": [["a", 1.0]]},
                "y": {"patterns": [["b", 1.0]]},
            },
            "routing": {"hybrid": {}, "x": {"routing_path": "SQL"}},
        }
        with pytest.raises(ConfigValidationError, match="intents.y"):
            RouterConfig.from_dict(bad)

    def test_invalid_regex_raises(self):
        bad = {
            "intents": {"x": {"patterns": [["(unclosed", 1.0]]}},
            "routing": {"hybrid": {}, "x": {"routing_path": "SQL"}},
        }
        with pytest.raises(ConfigValidationError, match="invalid regex"):
            RouterConfig.from_dict(bad)

    def test_malformed_pattern_entry_raises(self):
        bad = {
            "intents": {"x": {"patterns": [["only_one_element"]]}},
            "routing": {"hybrid": {}, "x": {"routing_path": "SQL"}},
        }
        with pytest.raises(ConfigValidationError, match="invalid pattern entry"):
            RouterConfig.from_dict(bad)

    def test_negative_threshold_raises(self):
        bad = dict(GENERIC_CONFIG)
        bad_settings = dict(GENERIC_CONFIG["settings"])
        bad_settings["dominant_threshold"] = -1
        bad = {**GENERIC_CONFIG, "settings": bad_settings}
        with pytest.raises(ConfigValidationError, match="dominant_threshold"):
            RouterConfig.from_dict(bad)

    def test_invalid_cache_size_raises(self):
        bad_settings = dict(GENERIC_CONFIG["settings"])
        bad_settings["cache_size"] = 0
        bad = {**GENERIC_CONFIG, "settings": bad_settings}
        with pytest.raises(ConfigValidationError, match="cache_size"):
            RouterConfig.from_dict(bad)

    def test_training_examples_wrong_type_raises(self):
        bad = {**GENERIC_CONFIG, "training_examples": {"structured": "not a list"}}
        with pytest.raises(ConfigValidationError, match="training_examples"):
            RouterConfig.from_dict(bad)

    def test_defaults_applied_when_settings_omitted(self):
        minimal = {
            "intents": {"x": {"patterns": [["a", 1.0]]}},
            "routing": {"hybrid": {}, "x": {"routing_path": "SQL"}},
        }
        cfg = RouterConfig.from_dict(minimal)
        assert cfg.settings["cache_size"] == 4096
        assert cfg.settings["dominant_threshold"] == 1.2

    def test_load_with_none_path_uses_builtin_default(self):
        cfg = RouterConfig.load(None)
        assert "structured" in cfg.intents
        assert "semantic" in cfg.intents

    def test_load_with_nonexistent_path_falls_back(self, tmp_path):
        missing = tmp_path / "does_not_exist.yaml"
        cfg = RouterConfig.load(missing)
        assert "structured" in cfg.intents  # built-in default

    def test_load_json_file(self, tmp_path):
        path = tmp_path / "cfg.json"
        path.write_text(json.dumps(GENERIC_CONFIG), encoding="utf-8")
        cfg = RouterConfig.load(path)
        assert "structured" in cfg.intents

    def test_load_invalid_json_raises(self, tmp_path):
        path = tmp_path / "cfg.json"
        path.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(ConfigValidationError, match="not valid JSON"):
            RouterConfig.load(path)

    def test_load_unsupported_extension_raises(self, tmp_path):
        path = tmp_path / "cfg.txt"
        path.write_text("irrelevant", encoding="utf-8")
        with pytest.raises(ConfigValidationError, match="Unsupported config file extension"):
            RouterConfig.load(path)


# ============================================================
# TEXT NORMALIZER
# ============================================================

class TestTextNormalizer:
    def test_lowercase_and_punctuation_removed(self):
        norm = TextNormalizer({
            "strip_accents": True, "casefold": True,
            "remove_punctuation": True, "collapse_whitespace": True,
        })
        result = norm.normalize("Hello, World!!!")
        assert result == "hello world"

    def test_accent_stripping(self):
        norm = TextNormalizer({"strip_accents": True, "casefold": True,
                                "remove_punctuation": False, "collapse_whitespace": True})
        result = norm.normalize("café résumé")
        assert "é" not in result
        assert "cafe" in result

    def test_whitespace_collapsing(self):
        norm = TextNormalizer({"strip_accents": False, "casefold": False,
                                "remove_punctuation": False, "collapse_whitespace": True})
        result = norm.normalize("hello    world\n\tfoo")
        assert result == "hello world foo"

    def test_options_disabled(self):
        norm = TextNormalizer({"strip_accents": False, "casefold": False,
                                "remove_punctuation": False, "collapse_whitespace": False})
        text = "Hello,  World!"
        assert norm.normalize(text) == text

    def test_empty_string(self):
        norm = TextNormalizer({})
        assert norm.normalize("") == ""

    def test_none_input_handled(self):
        norm = TextNormalizer({})
        assert norm.normalize(None) == ""


# ============================================================
# LRU CACHE
# ============================================================

class TestLRUCache:
    def test_put_and_get(self):
        cache = _LRUCache(capacity=10)
        result = IntentResult(query_type="structured")
        cache.put("query text", result)
        got = cache.get("query text")
        assert got is not None
        assert got.query_type == "structured"

    def test_miss_returns_none(self):
        cache = _LRUCache(capacity=10)
        assert cache.get("nonexistent") is None

    def test_lru_eviction(self):
        cache = _LRUCache(capacity=2)
        cache.put("a", IntentResult(query_type="a"))
        cache.put("b", IntentResult(query_type="b"))
        cache.put("c", IntentResult(query_type="c"))
        assert cache.get("a") is None
        assert cache.get("b") is not None
        assert cache.get("c") is not None

    def test_get_promotes_to_most_recently_used(self):
        cache = _LRUCache(capacity=2)
        cache.put("a", IntentResult(query_type="a"))
        cache.put("b", IntentResult(query_type="b"))
        cache.get("a")  # promote a
        cache.put("c", IntentResult(query_type="c"))  # should evict b, not a
        assert cache.get("a") is not None
        assert cache.get("b") is None

    def test_clear(self):
        cache = _LRUCache(capacity=10)
        cache.put("a", IntentResult(query_type="a"))
        cache.clear()
        assert cache.get("a") is None


# ============================================================
# SCHEMA / METADATA DATACLASSES
# ============================================================

class TestSchemaContext:
    def test_defaults_are_empty(self):
        s = SchemaContext()
        assert s.tables == ()
        assert s.numeric_columns == ()

    def test_to_dict(self):
        s = SchemaContext(time_columns=("created_at",))
        d = s.to_dict()
        assert d["time_columns"] == ("created_at",)


class TestDatasetMetadata:
    def test_defaults(self):
        m = DatasetMetadata()
        assert m.domain == ""
        assert m.available_retrievers == ()

    def test_to_dict(self):
        m = DatasetMetadata(domain="finance", available_retrievers=("sql",))
        d = m.to_dict()
        assert d["domain"] == "finance"
        assert d["available_retrievers"] == ("sql",)


# ============================================================
# RULE ENGINE PLUGIN
# ============================================================

class TestRuleEnginePlugin:
    def test_dominant_intent_wins(self):
        plugin = RuleEnginePlugin()
        cfg = RouterConfig.from_dict(GENERIC_CONFIG)
        context = _RoutingContext(
            normalized_text="what is the total count of items",
            raw_text="What is the total count of items?",
            schema=None, dataset_metadata=None, config=cfg,
        )
        verdict = plugin.evaluate(context)
        assert verdict.intent == "structured"
        assert verdict.confidence > 0

    def test_no_match_returns_hybrid_zero_confidence(self):
        plugin = RuleEnginePlugin()
        cfg = RouterConfig.from_dict(GENERIC_CONFIG)
        context = _RoutingContext(
            normalized_text="asdkjfh qwerty zzz",
            raw_text="asdkjfh qwerty zzz",
            schema=None, dataset_metadata=None, config=cfg,
        )
        verdict = plugin.evaluate(context)
        assert verdict.intent == "hybrid"
        assert verdict.confidence == 0.0

    def test_schema_boost_increases_score(self):
        plugin = RuleEnginePlugin()
        cfg = RouterConfig.from_dict(GENERIC_CONFIG)
        schema = SchemaContext(time_columns=("created_at",))

        ctx_without = _RoutingContext(
            normalized_text="how has this changed",
            raw_text="how has this changed",
            schema=None, dataset_metadata=None, config=cfg,
        )
        ctx_with = _RoutingContext(
            normalized_text="how has this changed",
            raw_text="how has this changed",
            schema=schema, dataset_metadata=None, config=cfg,
        )
        v_without = plugin.evaluate(ctx_without)
        v_with = plugin.evaluate(ctx_with)
        # trend pattern doesn't match "how has this changed" directly
        # (no "trend"/"over time"/"growth"/"change" as whole word... "change" IS there)
        assert v_with.diagnostics["scores"]["trend"] >= v_without.diagnostics["scores"]["trend"]

    def test_plugin_name(self):
        assert RuleEnginePlugin().name == "rule"


# ============================================================
# CLASSIFIER PLUGIN
# ============================================================

class TestClassifierPlugin:
    def test_no_training_examples_is_noop(self):
        plugin = ClassifierPlugin(training_examples={})
        cfg = RouterConfig.from_dict(GENERIC_CONFIG)
        context = _RoutingContext(
            normalized_text="anything at all",
            raw_text="anything at all",
            schema=None, dataset_metadata=None, config=cfg,
        )
        assert plugin.evaluate(context) is None

    def test_trains_and_predicts_with_examples(self):
        examples = {
            "structured": ["how many rows exist", "total count of entries"],
            "semantic": ["what do people think", "summarize opinions here"],
        }
        plugin = ClassifierPlugin(training_examples=examples, prefer_embeddings=False)
        cfg = RouterConfig.from_dict(GENERIC_CONFIG)
        plugin.ensure_trained()

        context = _RoutingContext(
            normalized_text="how many rows exist",
            raw_text="how many rows exist",
            schema=None, dataset_metadata=None, config=cfg,
        )
        verdict = plugin.evaluate(context)
        # Either a real verdict (sklearn installed) or None (sklearn missing)
        if verdict is not None:
            assert verdict.method == "classifier"
            assert 0.0 <= verdict.confidence <= 1.0

    def test_retrain_adds_examples(self):
        plugin = ClassifierPlugin(
            training_examples={"structured": ["count of x"], "semantic": ["opinion on y"]},
            prefer_embeddings=False,
        )
        plugin.ensure_trained()
        plugin.retrain({"structured": ["new example about totals"]})
        # Should not raise; internal state updated (or remains no-op if sklearn absent)

    def test_plugin_name(self):
        plugin = ClassifierPlugin(training_examples={})
        assert plugin.name == "classifier"


# ============================================================
# INTENT ROUTER — CORE ROUTING BEHAVIOR
# ============================================================

class TestIntentRouterRouting:
    def test_empty_query_returns_hybrid(self, router):
        result = router.route("")
        assert result.query_type == "hybrid"
        assert result.confidence == 0.0

    def test_whitespace_only_query_returns_hybrid(self, router):
        result = router.route("   ")
        assert result.query_type == "hybrid"

    def test_structured_query_routes_correctly(self, router):
        result = router.route("What is the total count of items?")
        assert result.query_type == "structured"
        assert result.needs_sql is True
        assert result.needs_vector is False
        assert result.routing_path == "SQL"

    def test_trend_query_sets_time_series_flag(self, router):
        result = router.route("Show me the growth trend over time")
        assert result.query_type == "trend"
        assert result.needs_time_series is True

    def test_semantic_query_sets_extra_flags(self, router):
        result = router.route("Summarize the general feedback and opinion")
        assert result.query_type == "semantic"
        assert result.needs_vector is True
        assert result.extra_flags.get("needs_summary") is True

    def test_unmatched_query_falls_back_to_hybrid(self, router):
        result = router.route("zzz qqq xxx yyy")
        assert result.query_type == "hybrid"
        assert result.needs_sql is True
        assert result.needs_vector is True

    def test_result_has_latency_recorded(self, router):
        result = router.route("total count of items")
        assert result.latency_ms >= 0.0

    def test_result_is_json_serializable(self, router):
        result = router.route("total count of items")
        json.dumps(result.to_dict())  # should not raise


class TestIntentRouterCaching:
    def test_second_identical_call_hits_cache(self, router):
        q = "total count of items please"
        r1 = router.route(q)
        r2 = router.route(q)
        assert r1.method != "cache"
        assert r2.method == "cache"

    def test_cache_disabled_never_hits(self, router_no_cache):
        q = "total count of items please"
        r1 = router_no_cache.route(q)
        r2 = router_no_cache.route(q)
        assert r1.method != "cache"
        assert r2.method != "cache"

    def test_clear_cache_forces_recompute(self, router):
        q = "total count of items please"
        router.route(q)
        router.clear_cache()
        r2 = router.route(q)
        assert r2.method != "cache"

    def test_normalization_makes_variants_share_cache(self, router):
        r1 = router.route("Total   Count of ITEMS!!!")
        r2 = router.route("total count of items")
        assert r2.method == "cache"
        assert r1.query_type == r2.query_type

    def test_per_call_schema_override_bypasses_cache(self, router):
        q = "total count of items please"
        router.route(q)
        schema = SchemaContext(numeric_columns=("amount",))
        r2 = router.route(q, schema=schema)
        # schema override forces recompute, not a cache hit
        assert r2.method != "cache"


class TestIntentRouterSchemaAndMetadata:
    def test_default_schema_applied_when_no_override(self):
        schema = SchemaContext(time_columns=("created_at",))
        r = IntentRouter(config=GENERIC_CONFIG, schema=schema)
        result = r.route("how has this changed")
        assert result.query_type in ("trend", "hybrid")  # boosted toward trend

    def test_retriever_downgrade_removes_unavailable_flags(self, router):
        metadata = DatasetMetadata(available_retrievers=("vector",))
        result = router.route("total count of items", dataset_metadata=metadata)
        assert result.needs_sql is False
        assert result.extra_flags.get("retriever_downgraded") is True

    def test_no_metadata_leaves_flags_untouched(self, router):
        result = router.route("total count of items")
        assert result.needs_sql is True
        assert "retriever_downgraded" not in result.extra_flags

    def test_both_retrievers_unavailable_marks_unavailable(self, router):
        metadata = DatasetMetadata(available_retrievers=("knowledge_graph",))
        result = router.route("total count of items", dataset_metadata=metadata)
        assert result.routing_path == "Unavailable"


class TestIntentRouterBatch:
    def test_route_batch_returns_matching_count(self, router):
        queries = ["total count", "growth trend", "opinion feedback"]
        results = router.route_batch(queries)
        assert len(results) == 3
        assert all(isinstance(r, IntentResult) for r in results)

    def test_route_batch_empty_list(self, router):
        assert router.route_batch([]) == []


class TestIntentRouterUnknownIntentFallback:
    def test_plugin_returning_unknown_intent_falls_back_to_hybrid(self, router):
        class RogueEvilPlugin(RoutingPlugin):
            @property
            def name(self):
                return "rogue"

            def evaluate(self, context):
                return _PluginVerdict(intent="not_a_real_intent", confidence=0.99, method="rogue")

        r = IntentRouter(config=GENERIC_CONFIG, plugins=[RogueEvilPlugin()])
        result = r.route("anything")
        assert result.query_type == "not_a_real_intent"
        assert result.routing_path == "Hybrid"  # falls back to hybrid's routing entry

    def test_plugin_exception_is_caught_and_skipped(self, router):
        class ExplodingPlugin(RoutingPlugin):
            @property
            def name(self):
                return "exploding"

            def evaluate(self, context):
                raise RuntimeError("boom")

        r = IntentRouter(config=GENERIC_CONFIG, plugins=[ExplodingPlugin()])
        result = r.route("total count of items")  # should not raise
        assert result.query_type == "structured"


# ============================================================
# CONFIG RELOAD
# ============================================================

class TestConfigReload:
    def test_reload_with_new_dict(self, router):
        result_before = router.route("total count of items")
        assert result_before.query_type == "structured"

        new_config = {
            "intents": {
                "only_intent": {"patterns": [[r"\btotal\b", 1.0]]},
            },
            "routing": {
                "hybrid": {"needs_sql": True, "needs_vector": True, "routing_path": "Hybrid"},
                "only_intent": {"needs_sql": True, "routing_path": "SQL"},
            },
        }
        router.reload_config(config=new_config)
        result_after = router.route("total count of items")
        assert result_after.query_type == "only_intent"

    def test_reload_clears_cache(self, router):
        router.route("total count of items")
        new_config = dict(GENERIC_CONFIG)
        router.reload_config(config=new_config)
        result = router.route("total count of items")
        assert result.method != "cache"

    def test_reload_with_invalid_config_raises_and_keeps_old(self, router):
        bad_config = {"intents": {}}
        with pytest.raises(ConfigValidationError):
            router.reload_config(config=bad_config)
        # router should still work with the old config
        result = router.route("total count of items")
        assert result.query_type == "structured"


# ============================================================
# RETRAIN CLASSIFIER
# ============================================================

class TestRetrainClassifier:
    def test_retrain_does_not_raise(self, router):
        samples = [
            ("how many total records", "structured"),
            ("what do people think about it", "semantic"),
        ]
        router.retrain_classifier(samples)  # should not raise

    def test_retrain_clears_cache(self, router):
        router.route("total count of items")
        router.retrain_classifier([("total sum please", "structured")])
        result = router.route("total count of items")
        assert result.method != "cache"


# ============================================================
# WARMUP
# ============================================================

class TestWarmup:
    def test_warmup_does_not_raise(self, router):
        router.warmup()

    def test_warmup_is_idempotent(self, router):
        router.warmup()
        router.warmup()  # should not raise or duplicate state

    def test_warmup_clears_cache(self, router):
        router.route("total count of items")
        router.warmup()
        result = router.route("total count of items")
        assert result.method != "cache"


# ============================================================
# BACKWARD-COMPATIBLE CONSTRUCTOR ARGS
# ============================================================

class TestBackwardCompatibleConstructor:
    def test_legacy_cache_size_arg_applied(self):
        r = IntentRouter(config=GENERIC_CONFIG, cache_size=10)
        assert r._config.settings["cache_size"] == 10

    def test_legacy_classifier_threshold_arg_applied(self):
        r = IntentRouter(config=GENERIC_CONFIG, classifier_confidence_threshold=0.9)
        assert r._config.settings["low_confidence_threshold"] == 0.9

    def test_use_cache_false_disables_cache(self):
        r = IntentRouter(config=GENERIC_CONFIG, use_cache=False)
        assert r._cache is None

    def test_default_construction_uses_builtin_config(self):
        r = IntentRouter()
        result = r.route("how many records exist")
        assert isinstance(result, IntentResult)


# ============================================================
# INTENT RESULT DATACLASS
# ============================================================

class TestIntentResult:
    def test_defaults(self):
        r = IntentResult()
        assert r.query_type == "hybrid"
        assert r.confidence == 0.0
        assert r.extra_flags == {}
        assert r.plugin_scores == {}

    def test_to_dict_roundtrip(self):
        r = IntentResult(query_type="structured", needs_sql=True, confidence=0.8)
        d = r.to_dict()
        assert d["query_type"] == "structured"
        assert d["needs_sql"] is True
        assert d["confidence"] == 0.8


# ============================================================
# MODULE-LEVEL SINGLETON
# ============================================================

class TestSingleton:
    def test_get_router_returns_same_instance(self, monkeypatch):
        import Agents.intent_router as mod
        monkeypatch.setattr(mod, "_default_router", None)
        r1 = get_router()
        r2 = get_router()
        assert r1 is r2

    def test_module_level_route_function_works(self, monkeypatch):
        import Agents.intent_router as mod
        monkeypatch.setattr(mod, "_default_router", None)
        result = route("how many records are there")
        assert isinstance(result, IntentResult)

    def test_get_router_ignores_kwargs_after_first_init(self, monkeypatch):
        import Agents.intent_router as mod
        monkeypatch.setattr(mod, "_default_router", None)
        r1 = get_router(cache_size=999)
        r2 = get_router(cache_size=1)  # ignored, singleton already exists
        assert r1 is r2
        assert r1._config.settings["cache_size"] == 999


# ============================================================
# LOGGING
# ============================================================

class TestLogging:
    def test_set_log_level_accepts_string(self):
        set_log_level("DEBUG")
        import Agents.intent_router as mod
        assert mod.LOGGER.level == logging.DEBUG

    def test_set_log_level_accepts_int(self):
        set_log_level(logging.WARNING)
        import Agents.intent_router as mod
        assert mod.LOGGER.level == logging.WARNING

    def test_router_accepts_log_level_kwarg(self):
        IntentRouter(config=GENERIC_CONFIG, log_level="ERROR")
        import Agents.intent_router as mod
        assert mod.LOGGER.level == logging.ERROR
        set_log_level(logging.INFO)  # reset for other tests


# ============================================================
# UNICODE / MULTI-LANGUAGE NORMALIZATION (integration)
# ============================================================

class TestUnicodeQueries:
    def test_accented_query_still_routes(self, router):
        # "café" style accents shouldn't break routing on ASCII patterns
        result = router.route("Résumé of total count of items")
        assert isinstance(result, IntentResult)

    def test_non_latin_script_does_not_crash(self, router):
        result = router.route("これはテストです")  # Japanese, no pattern match expected
        assert result.query_type == "hybrid"


# ============================================================
# THREAD SAFETY (light smoke test)
# ============================================================

class TestThreadSafety:
    def test_concurrent_routing_does_not_crash(self, router):
        import threading

        errors = []

        def worker():
            try:
                for _ in range(20):
                    router.route("total count of items")
                    router.route("growth trend over time")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors