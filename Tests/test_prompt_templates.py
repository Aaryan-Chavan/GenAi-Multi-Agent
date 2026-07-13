"""
tests/test_prompt_templates.py
============================================================
Unit tests for llm.prompt_templates (schema-agnostic version)
============================================================
"""

import pytest

from LLM.prompt_templates import (
    PromptConfig,
    PromptTemplateManager,
    PromptType,
    ResponseStyle,
    RetrievalMode,
)


@pytest.fixture
def manager():
    return PromptTemplateManager()


@pytest.fixture
def review_records():
    """Dataset shape #1: review/sentiment style."""
    return [
        {
            "chunk_id": "r1",
            "chunk_text": "The battery life is excellent.",
            "score": 0.912345,
            "sentiment": "positive",
            "aspect": "battery",
        },
        {
            "chunk_id": "r2",
            "chunk_text": "The screen is too dim outdoors.",
            "score": 0.77,
            "sentiment": "negative",
            "aspect": "display",
        },
    ]


@pytest.fixture
def legal_records():
    """Dataset shape #2: completely different domain, no sentiment/aspect."""
    return [
        {
            "case_number": "2024-CV-001",
            "jurisdiction": "N.D. Cal.",
            "text": "The court held that the defendant breached contract.",
            "filing_date": "2024-01-15",
        }
    ]


@pytest.fixture
def alt_text_key_records():
    """Dataset shape #3: uses 'content' instead of 'chunk_text'/'text'."""
    return [{"content": "Some passage body.", "author": "Jane Doe"}]


# ============================================================
# CONFIG
# ============================================================

class TestPromptConfig:
    def test_defaults(self):
        cfg = PromptConfig()
        assert cfg.language == "English"
        assert cfg.max_context_chunks == 30
        assert "chunk_text" in cfg.text_field_candidates
        assert cfg.excluded_metadata_fields == ()

    def test_custom_text_field_candidates(self):
        cfg = PromptConfig(text_field_candidates=("body_text",))
        assert cfg.text_field_candidates == ("body_text",)

    def test_immutable_via_slots(self):
        cfg = PromptConfig()
        with pytest.raises(AttributeError):
            cfg.nonexistent_field = 1


# ============================================================
# INTERNAL HELPERS
# ============================================================

class TestHelpers:
    def test_clean_text_strips_and_removes_cr(self):
        assert PromptTemplateManager.clean_text("  hello\r\n\tworld  ") == "hello\n world"

    def test_clean_text_none_returns_empty(self):
        assert PromptTemplateManager.clean_text(None) == ""

    def test_clean_lines_removes_blank_lines(self):
        text = "line1\n\n  \nline2\n"
        assert PromptTemplateManager.clean_lines(text) == "line1\nline2"

    def test_format_list_empty(self):
        assert PromptTemplateManager.format_list([]) == "None"

    def test_format_list_values(self):
        assert PromptTemplateManager.format_list(["a", "b"]) == "- a\n- b"

    def test_normalize_query_collapses_whitespace(self):
        assert PromptTemplateManager.normalize_query("  what   is\n\tX?  ") == "what is X?"

    def test_join_sections_skips_empty(self):
        result = PromptTemplateManager.join_sections("a", "", "  ", "b")
        assert result == "a\n\nb"

    def test_header_format(self):
        assert PromptTemplateManager.header("TITLE") == "TITLE\n-----"


# ============================================================
# DYNAMIC TEXT FIELD DISCOVERY
# ============================================================

class TestFindTextField:
    def test_finds_chunk_text_first(self, manager):
        record = {"chunk_text": "a", "text": "b"}
        assert manager._find_text_field(record) == "chunk_text"

    def test_falls_back_to_content_style(self, manager, alt_text_key_records):
        assert manager._find_text_field(alt_text_key_records[0]) == "content"

    def test_returns_none_when_no_candidate_present(self, manager):
        assert manager._find_text_field({"foo": "bar"}) is None

    def test_skips_empty_candidate_values(self, manager):
        record = {"chunk_text": "   ", "text": "real text"}
        assert manager._find_text_field(record) == "text"


# ============================================================
# SCHEMA-AGNOSTIC format_context()
# ============================================================

class TestFormatContext:
    def test_empty_records_returns_placeholder(self, manager):
        assert manager.format_context([]) == "No relevant context available."

    def test_review_dataset_renders_all_fields(self, manager, review_records):
        result = manager.format_context(review_records)
        assert "[Context 1]" in result
        assert "Sentiment : positive" in result
        assert "Aspect : battery" in result
        assert "Score : 0.9123" in result  # float formatting
        assert "The battery life is excellent." in result

    def test_legal_dataset_renders_without_hardcoded_fields(self, manager, legal_records):
        """Fields never named anywhere in the class (case_number,
        jurisdiction, filing_date) must still show up automatically."""
        result = manager.format_context(legal_records)
        assert "Case Number : 2024-CV-001" in result
        assert "Jurisdiction : N.D. Cal." in result
        assert "Filing Date : 2024-01-15" in result
        assert "breached contract" in result

    def test_alt_text_key_dataset(self, manager, alt_text_key_records):
        result = manager.format_context(alt_text_key_records)
        assert "Some passage body." in result
        assert "Author : Jane Doe" in result
        # 'content' itself should not be duplicated as a metadata line
        assert "Content :" not in result

    def test_record_with_no_text_field_still_shows_metadata(self, manager):
        records = [{"foo": "bar", "baz": 1}]
        result = manager.format_context(records)
        assert "(no primary text field found on this record)" in result
        assert "Foo : bar" in result
        assert "Baz : 1" in result

    def test_priority_fields_appear_before_others(self, manager):
        records = [{
            "chunk_text": "body",
            "zzz_custom": "last",
            "chunk_id": "id1",
            "score": 0.5,
        }]
        result = manager.format_context(records)
        id_pos = result.index("Chunk Id")
        score_pos = result.index("Score")
        custom_pos = result.index("Zzz Custom")
        assert id_pos < custom_pos
        assert score_pos < custom_pos

    def test_excluded_metadata_fields_are_hidden(self):
        cfg = PromptConfig(excluded_metadata_fields=("internal_id", "embedding"))
        mgr = PromptTemplateManager(config=cfg)
        records = [{"chunk_text": "body", "internal_id": "secret", "embedding": [0.1, 0.2]}]
        result = mgr.format_context(records)
        assert "Internal Id" not in result
        assert "Embedding" not in result
        assert "body" in result

    def test_none_and_empty_values_are_skipped(self, manager):
        records = [{"chunk_text": "body", "topic": None, "aspect": ""}]
        result = manager.format_context(records)
        assert "Topic" not in result
        assert "Aspect" not in result

    def test_respects_max_context_chunks(self):
        cfg = PromptConfig(max_context_chunks=1)
        mgr = PromptTemplateManager(config=cfg)
        records = [
            {"chunk_text": "first"},
            {"chunk_text": "second"},
        ]
        result = mgr.format_context(records)
        assert "first" in result
        assert "second" not in result

    def test_non_dict_records_are_skipped(self, manager):
        records = ["not-a-dict", {"chunk_text": "valid"}]
        result = manager.format_context(records)
        assert "[Context" in result
        assert "valid" in result

    def test_custom_text_field_candidates_take_effect(self):
        cfg = PromptConfig(text_field_candidates=("body_text",))
        mgr = PromptTemplateManager(config=cfg)
        records = [{"body_text": "custom field body", "chunk_text": "should not be used as text"}]
        result = mgr.format_context(records)
        assert "custom field body" in result
        # chunk_text should show as metadata since it's no longer the text key
        assert "Chunk Text : should not be used as text" in result


# ============================================================
# CITATIONS
# ============================================================

class TestFormatCitations:
    def test_disabled_returns_empty(self):
        cfg = PromptConfig(include_citations=False)
        mgr = PromptTemplateManager(config=cfg)
        assert mgr.format_citations([{"chunk_id": "1"}]) == ""

    def test_empty_records_returns_empty(self, manager):
        assert manager.format_citations([]) == ""

    def test_dedupes_and_sorts(self, manager):
        records = [{"chunk_id": "b"}, {"chunk_id": "a"}, {"chunk_id": "b"}]
        result = manager.format_citations(records)
        assert result == "\n\nSources:\n- a\n- b"

    def test_records_without_chunk_id_ignored(self, manager):
        records = [{"chunk_id": None}, {"foo": "bar"}]
        assert manager.format_citations(records) == ""


# ============================================================
# SYSTEM / USER PROMPTS
# ============================================================

class TestSystemAndUserPrompt:
    def test_system_prompt_contains_core_rules(self, manager):
        sp = manager.system_prompt()
        assert "Never fabricate facts." in sp
        assert "Answer ONLY using the provided context." in sp

    def test_user_prompt_normalizes_query(self, manager):
        result = manager.user_prompt("  what   is X?  ")
        assert "what is X?" in result
        assert "User Question" in result


# ============================================================
# BUILD_PROMPT AND VARIANTS
# ============================================================

class TestBuildPromptVariants:
    def test_build_prompt_includes_all_sections(self, manager, review_records):
        prompt = manager.build_prompt("What about battery?", review_records)
        assert "SYSTEM" in prompt
        assert "RETRIEVED CONTEXT" in prompt
        assert "USER QUESTION" in prompt
        assert "INSTRUCTIONS" in prompt
        assert "battery life is excellent" in prompt

    def test_build_hybrid_prompt_wraps_base_prompt(self, manager, review_records):
        prompt = manager.build_hybrid_prompt("query", review_records)
        assert "HYBRID RETRIEVAL" in prompt
        assert "Combine all relevant evidence" in prompt

    def test_build_semantic_prompt(self, manager, review_records):
        prompt = manager.build_semantic_prompt("query", review_records)
        assert "SEMANTIC SEARCH" in prompt

    def test_build_structured_prompt_mentions_structured_data(self, manager, review_records):
        prompt = manager.build_structured_prompt("query", review_records)
        assert "STRUCTURED RETRIEVAL" in prompt
        assert "structured analytical data" in prompt

    def test_build_json_prompt_contains_schema(self, manager, review_records):
        prompt = manager.build_json_prompt("query", review_records)
        assert '"answer"' in prompt
        assert "Return ONLY valid JSON." in prompt

    def test_build_comparison_prompt(self, manager, review_records):
        prompt = manager.build_comparison_prompt("compare these", review_records)
        assert "COMPARISON TASK" in prompt

    def test_build_memory_prompt(self, manager):
        history = ["User: hi", "Assistant: hello"]
        prompt = manager.build_memory_prompt(history, "follow up question")
        assert "CONVERSATION HISTORY" in prompt
        assert "follow up question" in prompt

    def test_build_query_rewrite_prompt(self, manager):
        prompt = manager.build_query_rewrite_prompt("original query")
        assert "QUERY REWRITE" in prompt
        assert "original query" in prompt


# ============================================================
# PROMPT REGISTRY (get_prompt)
# ============================================================

class TestPromptRegistry:
    def test_available_prompts_matches_registry_keys(self, manager):
        names = manager.available_prompts()
        for name in names:
            # should not raise for a known type dispatch existing
            assert callable(
                {
                    "default": manager.build_prompt,
                    "rag": manager.build_prompt,
                    "summary": manager.build_summary_prompt,
                    "comparison": manager.build_comparison_prompt,
                    "sentiment": manager.build_sentiment_prompt,
                    "analytics": manager.build_analytics_prompt,
                    "json": manager.build_json_prompt,
                    "api": manager.build_api_prompt,
                    "sql": manager.build_sql_reasoning_prompt,
                    "memory": manager.build_memory_prompt,
                    "multi_document": manager.build_multi_document_prompt,
                    "confidence": manager.build_confidence_prompt,
                    "safe": manager.build_safe_reasoning_prompt,
                    "rewrite": manager.build_query_rewrite_prompt,
                    "expand": manager.build_query_expansion_prompt,
                    "keywords": manager.build_keyword_prompt,
                }[name]
            )

    def test_get_prompt_dispatches_default(self, manager, review_records):
        result = manager.get_prompt("default", query="q", context=review_records)
        assert "SYSTEM" in result

    def test_get_prompt_case_insensitive_and_trims(self, manager):
        result = manager.get_prompt("  REWRITE  ", query="hello")
        assert "QUERY REWRITE" in result

    def test_get_prompt_unsupported_type_raises(self, manager):
        with pytest.raises(ValueError, match="Unsupported prompt type"):
            manager.get_prompt("not_a_real_type", query="q")


# ============================================================
# VALIDATION
# ============================================================

class TestValidation:
    def test_validate_context_none_raises(self):
        with pytest.raises(ValueError):
            PromptTemplateManager.validate_context(None)

    def test_validate_context_non_list_raises(self):
        with pytest.raises(TypeError):
            PromptTemplateManager.validate_context("not a list")

    def test_validate_context_valid_list_ok(self):
        PromptTemplateManager.validate_context([{"a": 1}])  # should not raise

    def test_validate_query_non_string_raises(self):
        with pytest.raises(TypeError):
            PromptTemplateManager.validate_query(123)

    def test_validate_query_empty_raises(self):
        with pytest.raises(ValueError):
            PromptTemplateManager.validate_query("   ")


# ============================================================
# METADATA / ENUMS
# ============================================================

class TestMetadataAndEnums:
    def test_metadata_flags(self):
        meta = PromptTemplateManager.metadata()
        assert meta["supports_json"] is True
        assert meta["version"] == "1.0"

    def test_prompt_type_enum_values(self):
        assert PromptType.HYBRID.value == "hybrid"
        assert PromptType.JSON.value == "json"

    def test_response_style_enum_values(self):
        assert ResponseStyle.BULLET.value == "bullet"

    def test_retrieval_mode_enum_values(self):
        assert RetrievalMode.SEMANTIC.value == "semantic"