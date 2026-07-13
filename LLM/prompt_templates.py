from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence


# ============================================================
# PROMPT CONSTANTS
# ============================================================

DEFAULT_SYSTEM_ROLE = (
    "You are an expert AI assistant specialized in "
    "retrieval augmented generation."
)

DEFAULT_LANGUAGE = "English"

DEFAULT_TONE = "Professional"

DEFAULT_RESPONSE_STYLE = "Detailed"

DEFAULT_MAX_CONTEXT = 30

DEFAULT_MAX_BULLETS = 10

DEFAULT_MAX_SENTENCES = 12


# ============================================================
# ENUMS
# ============================================================

class PromptType(Enum):

    SYSTEM = "system"

    HYBRID = "hybrid"

    SEMANTIC = "semantic"

    STRUCTURED = "structured"

    SUMMARIZATION = "summarization"

    COMPARISON = "comparison"

    ANALYTICS = "analytics"

    QUESTION_ANSWER = "question_answer"

    JSON = "json"

    CITATION = "citation"

    FOLLOW_UP = "follow_up"

    MEMORY = "memory"

    EVALUATION = "evaluation"


class ResponseStyle(Enum):

    SHORT = "short"

    NORMAL = "normal"

    DETAILED = "detailed"

    BULLET = "bullet"

    TABLE = "table"

    JSON = "json"


class RetrievalMode(Enum):

    HYBRID = "hybrid"

    SEMANTIC = "semantic"

    STRUCTURED = "structured"


# ============================================================
# PROMPT CONFIGURATION
# ============================================================

@dataclass(slots=True)
class PromptConfig:

    language: str = DEFAULT_LANGUAGE

    tone: str = DEFAULT_TONE

    style: str = DEFAULT_RESPONSE_STYLE

    max_context_chunks: int = DEFAULT_MAX_CONTEXT

    include_citations: bool = True

    include_reasoning: bool = False

    include_sources: bool = True

    allow_assumptions: bool = False

    temperature: float = 0.1

    # ------------------------------------------------------
    # Schema-agnostic context formatting controls.
    # These do NOT define a schema — they only tune *how*
    # whatever fields happen to exist get rendered/ordered.
    # ------------------------------------------------------

    # Fields checked (in order) as candidates for a record's main
    # body text. First non-empty match wins. Extend this list to
    # support new datasets without touching any method body.
    text_field_candidates: Sequence[str] = field(
        default_factory=lambda: (
            "chunk_text", "text", "content", "chunk", "body", "passage",
        )
    )

    # Fields shown first, in this order, if present on a record.
    # Purely cosmetic ordering — any other field discovered on the
    # record is still shown, just appended afterward alphabetically.
    priority_metadata_fields: Sequence[str] = field(
        default_factory=lambda: (
            "chunk_id", "score", "document_type", "source_type",
        )
    )

    # Fields that should never be surfaced as metadata (e.g. internal
    # ids, raw embeddings, noisy/irrelevant keys). Empty by default —
    # nothing is hidden unless explicitly configured.
    excluded_metadata_fields: Sequence[str] = field(
        default_factory=lambda: ()
    )


# ============================================================
# TEMPLATE MANAGER
# ============================================================

class PromptTemplateManager:
    """
    Central prompt manager.

    All prompt builders inherit from this class.

    Responsibilities

    • Build prompts
    • Format retrieved context
    • Apply safety rules
    • Generate instructions
    • Standardize prompts
    """

    def __init__(
        self,
        config: Optional[PromptConfig] = None
    ) -> None:

        self.config = config or PromptConfig()

    # ========================================================
    # INTERNAL HELPERS
    # ========================================================

    @staticmethod
    def clean_text(
        text: Any
    ) -> str:

        if text is None:
            return ""

        return (
            str(text)
            .replace("\r", "")
            .replace("\t", " ")
            .strip()
        )

    @staticmethod
    def clean_lines(
        text: str
    ) -> str:

        lines = []

        for line in text.splitlines():

            line = line.strip()

            if line:
                lines.append(line)

        return "\n".join(lines)

    @staticmethod
    def format_list(
        values: Sequence[Any]
    ) -> str:

        if not values:

            return "None"

        return "\n".join(

            f"- {value}"

            for value in values
        )

    @staticmethod
    def normalize_query(
        query: str
    ) -> str:

        return " ".join(

            str(query).split()

        ).strip()

    @staticmethod
    def join_sections(
        *sections: str
    ) -> str:

        cleaned = [

            section.strip()

            for section in sections

            if section and section.strip()

        ]

        return "\n\n".join(cleaned)

    @staticmethod
    def separator() -> str:

        return "-" * 80

    @staticmethod
    def header(
        title: str
    ) -> str:

        return (
            f"{title}\n"
            f"{'-' * len(title)}"
        )

    # ========================================================
    # DYNAMIC FIELD DISCOVERY
    # ========================================================

    def _find_text_field(self, record: Dict[str, Any]) -> Optional[str]:
        """
        Return the key holding this record's main body text, checked
        against config.text_field_candidates in order. Returns None
        if no candidate field has a non-empty value.
        """
        for key in self.config.text_field_candidates:
            value = record.get(key)
            if value and str(value).strip():
                return key
        return None

    # ========================================================
    # CONTEXT SERIALIZATION (schema-agnostic)
    # ========================================================

    def format_context(
        self,
        records: List[Dict[str, Any]]
    ) -> str:
        """
        Convert retrieved records into a clean, deterministic prompt
        context WITHOUT assuming any fixed dataset schema.

        The main body text is located via config.text_field_candidates
        (first match wins). Every other non-empty field present on a
        record is automatically surfaced as metadata — nothing is
        silently dropped, and nothing is hardcoded to a specific
        dataset shape (reviews, legal docs, tickets, DB rows, etc. all
        work the same way).
        """

        if not records:
            return "No relevant context available."

        formatted_records = []

        max_records = min(
            len(records),
            self.config.max_context_chunks
        )

        for index, record in enumerate(records[:max_records], start=1):

            if not isinstance(record, dict):
                continue

            text_key = self._find_text_field(record)
            text = self.clean_text(record.get(text_key, "")) if text_key else ""

            entry = [f"[Context {index}]"]

            if not text_key:
                entry.append("(no primary text field found on this record)")

            # Discover metadata dynamically instead of naming fields.
            meta_keys = [
                k for k in record.keys()
                if k != text_key
                and k not in self.config.excluded_metadata_fields
                and record.get(k) not in (None, "")
            ]

            priority = self.config.priority_metadata_fields
            ordered_keys = (
                [k for k in priority if k in meta_keys]
                + sorted(k for k in meta_keys if k not in priority)
            )

            for key in ordered_keys:
                value = record[key]
                label = key.replace("_", " ").title()
                if isinstance(value, float):
                    entry.append(f"{label} : {value:.4f}")
                else:
                    entry.append(f"{label} : {self.clean_text(value)}")

            if text:
                entry.append("")
                entry.append(text)

            formatted_records.append("\n".join(entry))

        return "\n\n".join(formatted_records) if formatted_records else "No relevant context available."

    # ========================================================
    # CITATION FORMATTER
    # ========================================================

    def format_citations(
        self,
        records: List[Dict[str, Any]]
    ) -> str:

        if (
            not self.config.include_citations
            or not records
        ):
            return ""

        citations = []

        for record in records:

            chunk_id = record.get(
                "chunk_id"
            )

            if chunk_id:

                citations.append(
                    str(chunk_id)
                )

        citations = sorted(
            set(citations)
        )

        if not citations:

            return ""

        return (
            "\n\nSources:\n"
            + "\n".join(
                f"- {cid}"
                for cid in citations
            )
        )

    # ========================================================
    # SYSTEM PROMPT
    # ========================================================

    def system_prompt(
        self
    ) -> str:

        rules = [

            DEFAULT_SYSTEM_ROLE,

            "Answer ONLY using the provided context.",

            "Never fabricate facts.",

            "Never guess information.",

            "If the answer is unavailable, explicitly state that.",

            "Use concise and accurate language.",

            "Prefer factual statements over assumptions.",

            "Maintain professional tone.",

            "Use citations whenever available.",

            "Do not expose internal reasoning."
        ]

        return "\n".join(
            rules
        )

    # ========================================================
    # USER PROMPT
    # ========================================================

    def user_prompt(
        self,
        query: str
    ) -> str:

        query = self.normalize_query(
            query
        )

        return (
            "User Question\n"
            "-------------\n"
            f"{query}"
        )

    # ========================================================
    # INSTRUCTIONS
    # ========================================================

    def answer_instructions(
        self
    ) -> str:

        instructions = [

            "Answer the user's question accurately.",

            "Only use retrieved context.",

            "If multiple contexts agree, synthesize them.",

            "If contexts disagree, explain the conflict.",

            "Do not invent missing information.",

            "Mention uncertainty when appropriate.",

            "Organize the answer clearly.",

            "End with citations if available."
        ]

        return "\n".join(

            f"- {item}"

            for item in instructions
        )

    # ========================================================
    # BASE PROMPT BUILDER
    # ========================================================

    def build_prompt(
        self,
        query: str,
        context: List[Dict[str, Any]]
    ) -> str:

        context_text = self.format_context(
            context
        )

        prompt = self.join_sections(

            self.header(
                "SYSTEM"
            ),

            self.system_prompt(),

            self.header(
                "RETRIEVED CONTEXT"
            ),

            context_text,

            self.header(
                "USER QUESTION"
            ),

            self.user_prompt(
                query
            ),

            self.header(
                "INSTRUCTIONS"
            ),

            self.answer_instructions()
        )

        return prompt
    # ========================================================
    # HYBRID RAG PROMPT
    # ========================================================

    def build_hybrid_prompt(
        self,
        query: str,
        context: List[Dict[str, Any]]
    ) -> str:
        """
        Prompt used by HybridRetriever.
        """

        prompt = self.join_sections(

            self.header(
                "HYBRID RETRIEVAL"
            ),

            (
                "The following information has been "
                "retrieved using semantic search and "
                "structured metadata filtering."
            ),

            self.build_prompt(
                query=query,
                context=context
            ),

            (
                "Combine all relevant evidence into a "
                "single accurate response."
            )
        )

        return prompt

    # ========================================================
    # SEMANTIC SEARCH PROMPT
    # ========================================================

    def build_semantic_prompt(
        self,
        query: str,
        context: List[Dict[str, Any]]
    ) -> str:
        """
        Prompt for semantic retrieval.
        """

        return self.join_sections(

            self.header(
                "SEMANTIC SEARCH"
            ),

            (
                "Use semantic similarity to answer "
                "the user's question."
            ),

            self.build_prompt(
                query=query,
                context=context
            )
        )

    # ========================================================
    # STRUCTURED RETRIEVAL PROMPT
    # ========================================================

    def build_structured_prompt(
        self,
        query: str,
        context: List[Dict[str, Any]]
    ) -> str:
        """
        Prompt used when retrieved data comes
        primarily from DuckDB.
        """

        return self.join_sections(

            self.header(
                "STRUCTURED RETRIEVAL"
            ),

            (
                "The following information comes "
                "from structured analytical data."
            ),

            self.build_prompt(
                query=query,
                context=context
            )
        )

    # ========================================================
    # SUMMARIZATION PROMPT
    # ========================================================

    def build_summary_prompt(
        self,
        context: List[Dict[str, Any]]
    ) -> str:
        """
        Generate concise summary.
        """

        return self.join_sections(

            self.header(
                "SUMMARIZATION"
            ),

            (
                "Summarize the retrieved information "
                "without changing factual meaning."
            ),

            self.format_context(
                context
            )
        )

    # ========================================================
    # DOCUMENT SUMMARY
    # ========================================================

    def build_document_summary_prompt(
        self,
        title: str,
        context: List[Dict[str, Any]]
    ) -> str:

        return self.join_sections(

            self.header(
                title
            ),

            (
                "Produce an executive summary."
            ),

            self.format_context(
                context
            )
        )

    # ========================================================
    # SHORT ANSWER
    # ========================================================

    def build_short_answer_prompt(
        self,
        query: str,
        context: List[Dict[str, Any]]
    ) -> str:

        return self.join_sections(

            self.build_prompt(
                query,
                context
            ),

            (
                "Respond in no more than "
                "three sentences."
            )
        )

    # ========================================================
    # DETAILED ANSWER
    # ========================================================

    def build_detailed_prompt(
        self,
        query: str,
        context: List[Dict[str, Any]]
    ) -> str:

        return self.join_sections(

            self.build_prompt(
                query,
                context
            ),

            (
                "Provide a comprehensive answer "
                "covering every relevant detail."
            )
        )

    # ========================================================
    # BULLET FORMAT
    # ========================================================

    def build_bullet_prompt(
        self,
        query: str,
        context: List[Dict[str, Any]]
    ) -> str:

        return self.join_sections(

            self.build_prompt(
                query,
                context
            ),

            (
                f"Answer using at most "
                f"{DEFAULT_MAX_BULLETS} bullet points."
            )
        )

    # ========================================================
    # TABLE FORMAT
    # ========================================================

    def build_table_prompt(
        self,
        query: str,
        context: List[Dict[str, Any]]
    ) -> str:

        return self.join_sections(

            self.build_prompt(
                query,
                context
            ),

            (
                "Return the answer as a markdown table "
                "whenever possible."
            )
        )
    # ========================================================
    # COMPARISON PROMPT
    # ========================================================

    def build_comparison_prompt(
        self,
        query: str,
        context: List[Dict[str, Any]]
    ) -> str:
        """
        Compare multiple products, reviews, entities,
        or documents using retrieved evidence.
        """

        return self.join_sections(

            self.header(
                "COMPARISON TASK"
            ),

            self.build_prompt(
                query,
                context
            ),

            (
                "Compare all relevant items objectively.\n"
                "Highlight similarities, differences,\n"
                "advantages and disadvantages.\n"
                "Do not invent missing information."
            )
        )

    # ========================================================
    # ANALYTICS PROMPT
    # ========================================================

    def build_analytics_prompt(
        self,
        query: str,
        context: List[Dict[str, Any]]
    ) -> str:

        return self.join_sections(

            self.header(
                "ANALYTICS"
            ),

            self.build_prompt(
                query,
                context
            ),

            (
                "Perform analytical reasoning.\n"
                "Identify important statistics,\n"
                "patterns, trends and insights.\n"
                "Explain observations clearly."
            )
        )

    # ========================================================
    # SENTIMENT ANALYSIS
    # ========================================================

    def build_sentiment_prompt(
        self,
        context: List[Dict[str, Any]]
    ) -> str:

        return self.join_sections(

            self.header(
                "SENTIMENT ANALYSIS"
            ),

            self.format_context(
                context
            ),

            (
                "Analyze customer sentiment.\n"
                "Identify positive, neutral and negative opinions.\n"
                "Summarize the overall sentiment."
            )
        )

    # ========================================================
    # ASPECT ANALYSIS
    # ========================================================

    def build_aspect_prompt(
        self,
        context: List[Dict[str, Any]]
    ) -> str:

        return self.join_sections(

            self.header(
                "ASPECT ANALYSIS"
            ),

            self.format_context(
                context
            ),

            (
                "Identify major product aspects.\n"
                "Explain strengths and weaknesses\n"
                "for every aspect."
            )
        )

    # ========================================================
    # TOPIC ANALYSIS
    # ========================================================

    def build_topic_prompt(
        self,
        context: List[Dict[str, Any]]
    ) -> str:

        return self.join_sections(

            self.header(
                "TOPIC ANALYSIS"
            ),

            self.format_context(
                context
            ),

            (
                "Identify major discussion topics.\n"
                "Group related ideas together.\n"
                "Provide concise explanations."
            )
        )

    # ========================================================
    # TREND ANALYSIS
    # ========================================================

    def build_trend_prompt(
        self,
        context: List[Dict[str, Any]]
    ) -> str:

        return self.join_sections(

            self.header(
                "TREND ANALYSIS"
            ),

            self.format_context(
                context
            ),

            (
                "Identify recurring trends.\n"
                "Mention increasing or decreasing patterns.\n"
                "Support every conclusion using context."
            )
        )

    # ========================================================
    # FOLLOW-UP QUESTION GENERATION
    # ========================================================

    def build_followup_prompt(
        self,
        query: str,
        answer: str
    ) -> str:

        return self.join_sections(

            self.header(
                "FOLLOW-UP QUESTIONS"
            ),

            f"Question:\n{query}",

            f"Answer:\n{answer}",

            (
                "Generate five intelligent follow-up\n"
                "questions that the user may ask next.\n"
                "Questions must remain relevant."
            )
        )

    # ========================================================
    # HALLUCINATION PREVENTION
    # ========================================================

    def build_grounding_prompt(
        self,
        query: str,
        context: List[Dict[str, Any]]
    ) -> str:

        return self.join_sections(

            self.header(
                "FACT GROUNDING"
            ),

            self.build_prompt(
                query,
                context
            ),

            (
                "Every statement must be grounded\n"
                "in retrieved evidence.\n"
                "Never fabricate facts.\n"
                "If evidence is insufficient,\n"
                "explicitly state that."
            )
        )

    # ========================================================
    # CITATION PROMPT
    # ========================================================

    def build_citation_prompt(
        self,
        query: str,
        context: List[Dict[str, Any]]
    ) -> str:

        citations = self.format_citations(
            context
        )

        return self.join_sections(

            self.build_prompt(
                query,
                context
            ),

            (
                "Every important statement should\n"
                "be traceable to retrieved context."
            ),

            citations
        )
    # ========================================================
    # JSON RESPONSE PROMPT
    # ========================================================

    def build_json_prompt(
        self,
        query: str,
        context: List[Dict[str, Any]]
    ) -> str:
        """
        Force the model to return a valid JSON object.
        """

        schema = """
{
    "answer": "",
    "summary": "",
    "confidence": "",
    "citations": [],
    "reason": ""
}
"""

        return self.join_sections(

            self.build_prompt(
                query,
                context
            ),

            self.header(
                "JSON FORMAT"
            ),

            (
                "Return ONLY valid JSON.\n"
                "Do not include markdown.\n"
                "Do not include explanations."
            ),

            schema.strip()
        )

    # ========================================================
    # API RESPONSE PROMPT
    # ========================================================

    def build_api_prompt(
        self,
        query: str,
        context: List[Dict[str, Any]]
    ) -> str:

        return self.join_sections(

            self.build_prompt(
                query,
                context
            ),

            (
                "Generate an API-friendly response.\n"
                "Keep formatting deterministic.\n"
                "Avoid conversational filler."
            )
        )

    # ========================================================
    # SQL ANALYTICS PROMPT
    # ========================================================

    def build_sql_reasoning_prompt(
        self,
        query: str,
        context: List[Dict[str, Any]]
    ) -> str:

        return self.join_sections(

            self.header(
                "SQL ANALYTICS"
            ),

            self.build_prompt(
                query,
                context
            ),

            (
                "Interpret numerical information.\n"
                "Explain calculations.\n"
                "Do not invent statistics."
            )
        )

    # ========================================================
    # MEMORY PROMPT
    # ========================================================

    def build_memory_prompt(
        self,
        conversation_history: List[str],
        query: str
    ) -> str:

        history = "\n".join(
            conversation_history
        )

        return self.join_sections(

            self.header(
                "CONVERSATION HISTORY"
            ),

            history,

            self.header(
                "CURRENT QUESTION"
            ),

            query,

            (
                "Use previous conversation only\n"
                "when it is directly relevant."
            )
        )

    # ========================================================
    # MULTI DOCUMENT REASONING
    # ========================================================

    def build_multi_document_prompt(
        self,
        query: str,
        context: List[Dict[str, Any]]
    ) -> str:

        return self.join_sections(

            self.header(
                "MULTI DOCUMENT REASONING"
            ),

            self.build_prompt(
                query,
                context
            ),

            (
                "Combine information from all\n"
                "retrieved documents.\n"
                "Resolve contradictions carefully.\n"
                "Mention conflicting evidence."
            )
        )

    # ========================================================
    # CONFIDENCE ESTIMATION
    # ========================================================

    def build_confidence_prompt(
        self,
        query: str,
        context: List[Dict[str, Any]]
    ) -> str:

        return self.join_sections(

            self.build_prompt(
                query,
                context
            ),

            (
                "Estimate confidence as\n"
                "High, Medium or Low.\n"
                "Confidence should depend only\n"
                "on retrieved evidence."
            )
        )

    # ========================================================
    # SAFE REASONING PROMPT
    # ========================================================

    def build_safe_reasoning_prompt(
        self,
        query: str,
        context: List[Dict[str, Any]]
    ) -> str:

        return self.join_sections(

            self.build_prompt(
                query,
                context
            ),

            (
                "Reason internally using the\n"
                "retrieved context.\n"
                "Return only the final answer.\n"
                "Never expose internal reasoning."
            )
        )

    # ========================================================
    # QUESTION REWRITING
    # ========================================================

    def build_query_rewrite_prompt(
        self,
        query: str
    ) -> str:

        return self.join_sections(

            self.header(
                "QUERY REWRITE"
            ),

            (
                "Rewrite the question for\n"
                "better semantic retrieval.\n"
                "Preserve original intent.\n"
                "Return only one rewritten query."
            ),

            query
        )

    # ========================================================
    # QUERY EXPANSION
    # ========================================================

    def build_query_expansion_prompt(
        self,
        query: str
    ) -> str:

        return self.join_sections(

            self.header(
                "QUERY EXPANSION"
            ),

            (
                "Generate five semantically\n"
                "equivalent search queries.\n"
                "Each query should preserve\n"
                "the original meaning."
            ),

            query
        )

    # ========================================================
    # KEYWORD EXTRACTION
    # ========================================================

    def build_keyword_prompt(
        self,
        query: str
    ) -> str:

        return self.join_sections(

            self.header(
                "KEYWORD EXTRACTION"
            ),

            (
                "Extract important keywords.\n"
                "Remove stop words.\n"
                "Return only keywords."
            ),

            query
        )
    # ========================================================
    # PROMPT REGISTRY
    # ========================================================

    def get_prompt(
        self,
        prompt_type: str,
        **kwargs
    ) -> str:
        """
        Generic prompt dispatcher.
        """

        prompt_type = (
            prompt_type.lower()
            .strip()
        )

        registry = {

            "default": self.build_prompt,
            "rag": self.build_prompt,

            "summary": self.build_summary_prompt,

            "comparison": self.build_comparison_prompt,

            "sentiment": self.build_sentiment_prompt,

            "analytics": self.build_analytics_prompt,

            "json": self.build_json_prompt,

            "api": self.build_api_prompt,

            "sql": self.build_sql_reasoning_prompt,

            "memory": self.build_memory_prompt,

            "multi_document": self.build_multi_document_prompt,

            "confidence": self.build_confidence_prompt,

            "safe": self.build_safe_reasoning_prompt,

            "rewrite": self.build_query_rewrite_prompt,

            "expand": self.build_query_expansion_prompt,

            "keywords": self.build_keyword_prompt
        }

        if prompt_type not in registry:

            raise ValueError(
                f"Unsupported prompt type: {prompt_type}"
            )

        return registry[
            prompt_type
        ](**kwargs)

    # ========================================================
    # AVAILABLE PROMPTS
    # ========================================================

    @staticmethod
    def available_prompts() -> List[str]:

        return [

            "default",
            "rag",
            "summary",
            "comparison",
            "sentiment",
            "analytics",
            "json",
            "api",
            "sql",
            "memory",
            "multi_document",
            "confidence",
            "safe",
            "rewrite",
            "expand",
            "keywords"
        ]

    # ========================================================
    # VALIDATION
    # ========================================================

    @staticmethod
    def validate_context(
        context: List[Dict[str, Any]]
    ) -> None:

        if context is None:

            raise ValueError(
                "Context cannot be None."
            )

        if not isinstance(
            context,
            list
        ):

            raise TypeError(
                "Context must be a list."
            )

    @staticmethod
    def validate_query(
        query: str
    ) -> None:

        if not isinstance(
            query,
            str
        ):

            raise TypeError(
                "Query must be string."
            )

        if not query.strip():

            raise ValueError(
                "Query cannot be empty."
            )

    # ========================================================
    # PROMPT METADATA
    # ========================================================

    @staticmethod
    def metadata() -> Dict[str, Any]:

        return {

            "version": "1.0",

            "supports_json": True,

            "supports_rag": True,

            "supports_memory": True,

            "supports_query_rewrite": True,

            "supports_query_expansion": True,

            "supports_multi_document": True,

            "supports_confidence": True,

            "supports_sql": True
        }

    # ========================================================
    # SUMMARY
    # ========================================================

    def summary(self) -> None:

        print("\n" + "=" * 60)

        print(
            "PROMPT TEMPLATE MANAGER"
        )

        print("=" * 60)

        print(
            f"Templates : {len(self.available_prompts())}"
        )

        print(
            "\nSupported Templates:\n"
        )

        for template in self.available_prompts():

            print(
                f" • {template}"
            )

        print("\nMetadata:\n")

        for key, value in self.metadata().items():

            print(
                f"{key}: {value}"
            )

        print("=" * 60)