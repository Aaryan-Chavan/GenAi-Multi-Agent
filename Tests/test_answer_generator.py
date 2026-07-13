"""
tests/test_answer_generator.py
============================================================
Unit tests for LLM.answer_generator.AnswerGenerator (schema-agnostic)
============================================================
"""

import pytest
from unittest.mock import MagicMock

from LLM.answer_generator import (
    AnswerGenerator,
    AnswerResult,
    AnswerStatus,
    AnswerType,
    SchemaConfig,
    get_answer_generator,
)


def make_generation_result(
    text="This is the answer.",
    model="qwen3-14b",
    prompt_tokens=10,
    generated_tokens=5,
    total_tokens=15,
    finish_reason="stop",
    tokens_per_second=42.0,
    request_id="req-123",
    cache_hit=False,
    latency=0.25,
):
    result = MagicMock()
    result.text = text
    result.model = model
    result.prompt_tokens = prompt_tokens
    result.generated_tokens = generated_tokens
    result.total_tokens = total_tokens
    result.finish_reason = finish_reason
    result.tokens_per_second = tokens_per_second
    result.request_id = request_id
    result.cache_hit = cache_hit
    result.latency = latency
    return result


@pytest.fixture
def mock_qwen_client():
    client = MagicMock()
    client.generate.return_value = make_generation_result()
    client.health_check.return_value = {"status": "ok"}
    client.hash_prompt = MagicMock(side_effect=lambda q: f"hash({q})")
    return client


@pytest.fixture
def mock_prompt_manager():
    manager = MagicMock()
    manager.build_prompt.return_value = "GENERAL_PROMPT"
    manager.build_structured_prompt.return_value = "STRUCTURED_PROMPT"
    manager.build_semantic_prompt.return_value = "SEMANTIC_PROMPT"
    manager.build_hybrid_prompt.return_value = "HYBRID_PROMPT"
    return manager


@pytest.fixture
def generator(mock_qwen_client, mock_prompt_manager):
    return AnswerGenerator(
        qwen_client=mock_qwen_client,
        prompt_manager=mock_prompt_manager,
    )


@pytest.fixture
def review_context():
    """Reviews-style dataset."""
    return [
        {"chunk_id": 1, "text": "Great battery.", "score": 0.9, "document": "doc1"},
        {"chunk_id": 2, "content": "Screen is dim.", "score": 0.7, "table": "tbl1"},
    ]


@pytest.fixture
def legal_context():
    """Different domain entirely, no 'score'/'chunk_id' style fields."""
    return [
        {"passage": "The court ruled in favor of plaintiff.", "similarity": 0.88, "case_id": "X-1"},
    ]


# ============================================================
# SCHEMA DISCOVERY
# ============================================================

class TestFieldDiscovery:
    def test_finds_text_field_from_candidates(self, generator):
        assert generator._find_text_field({"text": "hi"}) == "text"
        assert generator._find_text_field({"content": "hi"}) == "content"
        assert generator._find_text_field({"passage": "hi"}) == "passage"

    def test_returns_none_when_no_text_field(self, generator):
        assert generator._find_text_field({"foo": "bar"}) is None

    def test_skips_empty_text_candidates(self, generator):
        assert generator._find_text_field({"text": "   ", "content": "real"}) == "content"

    def test_finds_score_field_from_candidates(self, generator):
        assert generator._find_score_field({"score": 0.5}) == "score"
        assert generator._find_score_field({"similarity": 0.5}) == "similarity"
        assert generator._find_score_field({"relevance": 0.9}) == "relevance"

    def test_score_field_must_be_numeric(self, generator):
        assert generator._find_score_field({"score": "high"}) is None

    def test_custom_schema_config(self, mock_qwen_client, mock_prompt_manager):
        cfg = SchemaConfig(text_field_candidates=("body",))
        gen = AnswerGenerator(
            qwen_client=mock_qwen_client,
            prompt_manager=mock_prompt_manager,
            schema_config=cfg,
        )
        assert gen._find_text_field({"body": "hi", "text": "should not match"}) == "body"


# ============================================================
# VALIDATION
# ============================================================

class TestValidateQuery:
    def test_valid_query_is_stripped(self, generator):
        assert generator._validate_query("  hello  ") == "hello"

    def test_non_string_raises_type_error(self, generator):
        with pytest.raises(TypeError):
            generator._validate_query(123)

    def test_empty_string_raises_value_error(self, generator):
        with pytest.raises(ValueError):
            generator._validate_query("   ")


class TestValidateContext:
    def test_none_returns_empty_list(self, generator):
        assert generator._validate_context(None) == []

    def test_non_list_raises_type_error(self, generator):
        with pytest.raises(TypeError):
            generator._validate_context("not a list")

    def test_filters_non_dict_items(self, generator):
        context = [{"a": 1}, "bad", 42, {"b": 2}]
        assert generator._validate_context(context) == [{"a": 1}, {"b": 2}]


# ============================================================
# CONTEXT FORMATTING (schema-agnostic)
# ============================================================

class TestFormatContext:
    def test_empty_context_returns_placeholder(self, generator):
        assert generator._format_context([]) == "No external context available."

    def test_reviews_dataset(self, generator, review_context):
        result = generator._format_context(review_context)
        assert "[Context 1]" in result
        assert "Great battery." in result
        assert "Screen is dim." in result
        # metadata fields surfaced without being hardcoded
        assert "Document:" in result or "Document :" in result or "document" in result.lower()

    def test_legal_dataset_with_no_known_fields(self, generator, legal_context):
        result = generator._format_context(legal_context)
        assert "court ruled in favor" in result
        assert "Case Id" in result or "case_id" in result.lower()

    def test_skips_records_with_no_text(self, generator):
        context = [{"foo": "bar"}, {"text": "valid text"}]
        result = generator._format_context(context)
        assert "valid text" in result
        assert result.count("[Context") == 1

    def test_all_empty_records_returns_placeholder(self, generator):
        context = [{"foo": "bar"}]
        assert generator._format_context(context) == "No external context available."


class TestFormatSources:
    def test_extracts_non_text_fields(self, generator, review_context):
        sources = generator._format_sources(review_context)
        assert len(sources) == 2
        assert sources[0]["chunk_id"] == 1
        assert "text" not in sources[0]  # text field itself excluded

    def test_excludes_configured_fields(self, mock_qwen_client, mock_prompt_manager):
        cfg = SchemaConfig(excluded_source_fields=("embedding",))
        gen = AnswerGenerator(
            qwen_client=mock_qwen_client,
            prompt_manager=mock_prompt_manager,
            schema_config=cfg,
        )
        context = [{"text": "hi", "embedding": [0.1], "chunk_id": 1}]
        sources = gen._format_sources(context)
        assert "embedding" not in sources[0]
        assert sources[0]["chunk_id"] == 1

    def test_empty_context_returns_empty_list(self, generator):
        assert generator._format_sources([]) == []


class TestEstimateConfidence:
    def test_empty_context_returns_zero(self, generator):
        assert generator._estimate_confidence([]) == 0.0

    def test_uses_discovered_score_field(self, generator, legal_context):
        # legal_context uses "similarity" not "score"
        confidence = generator._estimate_confidence(legal_context)
        assert confidence == 0.88

    def test_average_across_mixed_score_field_names(self, generator):
        context = [{"score": 0.8}, {"similarity": 0.6}]
        confidence = generator._estimate_confidence(context)
        assert confidence == 0.7

    def test_clamped_to_range(self, generator):
        assert generator._estimate_confidence([{"score": 5.0}]) == 1.0
        assert generator._estimate_confidence([{"score": -1.0}]) == 0.0

    def test_no_numeric_score_field_returns_zero(self, generator):
        assert generator._estimate_confidence([{"score": "high"}]) == 0.0


class TestCleanAnswer:
    @pytest.mark.parametrize("prefix", ["Answer:", "Final Answer:", "Response:", "Assistant:", "AI:"])
    def test_strips_known_prefixes(self, prefix):
        raw = f"{prefix} The result is 42."
        assert AnswerGenerator._clean_answer(raw) == "The result is 42."

    def test_no_prefix_untouched(self):
        assert AnswerGenerator._clean_answer("Just an answer.") == "Just an answer."

    def test_empty_returns_empty(self):
        assert AnswerGenerator._clean_answer("") == ""


# ============================================================
# PROMPT BUILDING
# ============================================================

class TestBuildPrompt:
    @pytest.mark.parametrize(
        "answer_type,expected_call,expected_prompt",
        [
            (AnswerType.STRUCTURED, "build_structured_prompt", "STRUCTURED_PROMPT"),
            (AnswerType.SEMANTIC, "build_semantic_prompt", "SEMANTIC_PROMPT"),
            (AnswerType.HYBRID, "build_hybrid_prompt", "HYBRID_PROMPT"),
            (AnswerType.GENERAL, "build_prompt", "GENERAL_PROMPT"),
        ],
    )
    def test_dispatches_correct_template(
        self, generator, mock_prompt_manager, answer_type, expected_call, expected_prompt
    ):
        prompt = generator._build_prompt("query", [], answer_type)
        assert prompt == expected_prompt
        getattr(mock_prompt_manager, expected_call).assert_called_once()


# ============================================================
# PUBLIC generate()
# ============================================================

class TestGenerate:
    def test_successful_generation_with_review_dataset(self, generator, review_context):
        result = generator.generate("What about battery?", review_context, AnswerType.HYBRID)
        assert isinstance(result, AnswerResult)
        assert result.status == AnswerStatus.SUCCESS.value
        assert result.confidence == pytest.approx(0.8, rel=1e-2)
        assert len(result.sources) == 2

    def test_successful_generation_with_legal_dataset(self, generator, legal_context):
        result = generator.generate("What was the ruling?", legal_context, AnswerType.GENERAL)
        assert result.status == AnswerStatus.SUCCESS.value
        assert result.confidence == pytest.approx(0.88, rel=1e-2)

    def test_default_answer_type_is_hybrid(self, generator, mock_prompt_manager):
        generator.generate("query")
        mock_prompt_manager.build_hybrid_prompt.assert_called_once()

    def test_no_context_still_succeeds(self, generator):
        result = generator.generate("query with no context")
        assert result.status == AnswerStatus.SUCCESS.value
        assert result.confidence == 0.0
        assert result.sources == []

    def test_invalid_query_returns_failed(self, generator):
        result = generator.generate("")
        assert result.status == AnswerStatus.FAILED.value

    def test_invalid_context_type_returns_failed(self, generator):
        result = generator.generate("query", context="not-a-list")
        assert result.status == AnswerStatus.FAILED.value

    def test_qwen_exception_is_caught(self, generator, mock_qwen_client):
        mock_qwen_client.generate.side_effect = RuntimeError("LLM exploded")
        result = generator.generate("query")
        assert result.status == AnswerStatus.FAILED.value
        assert "LLM exploded" in result.answer

    def test_total_requests_increments(self, generator):
        generator.generate("q1")
        generator.generate("q2")
        assert generator.total_requests == 2

    def test_qwen_called_with_built_prompt(self, generator, mock_qwen_client):
        generator.generate("query", answer_type=AnswerType.STRUCTURED)
        _, kwargs = mock_qwen_client.generate.call_args
        assert kwargs["prompt"] == "STRUCTURED_PROMPT"


# ============================================================
# STATISTICS / HEALTH / RESET / DIAGNOSTICS
# ============================================================

class TestStatistics:
    def test_zero_requests(self, generator):
        stats = generator.get_statistics()
        assert stats["total_requests"] == 0
        assert stats["average_latency"] == 0.0

    def test_after_requests(self, generator):
        generator.generate("q1")
        generator.generate("q2")
        stats = generator.get_statistics()
        assert stats["total_requests"] == 2


class TestHealthCheck:
    def test_healthy_when_qwen_healthy(self, generator):
        health = generator.health_check()
        assert health["status"] == "healthy"

    def test_degraded_when_qwen_raises(self, generator, mock_qwen_client):
        mock_qwen_client.health_check.side_effect = RuntimeError("boom")
        health = generator.health_check()
        assert health["status"] == "degraded"


class TestReset:
    def test_reset_clears_stats(self, generator):
        generator.generate("q1")
        generator.reset()
        assert generator.total_requests == 0
        assert generator.total_latency == 0.0


class TestDiagnostics:
    def test_includes_generator_and_health(self, generator):
        diag = generator.diagnostics()
        assert "generator" in diag
        assert "health" in diag


class TestSingleton:
    def test_returns_same_instance(self, monkeypatch):
        import LLM.answer_generator as mod
        monkeypatch.setattr(mod, "_GENERATOR", None)
        gen1 = get_answer_generator()
        gen2 = get_answer_generator()
        assert gen1 is gen2