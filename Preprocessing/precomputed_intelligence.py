# Preprocessing/precomputed_intelligence.py

import re
from pathlib import Path

import numpy as np
import pandas as pd

from Config.settings import PROCESSED_DATA_DIR

# ============================================================
# MODULE-LEVEL COMPILED REGEX
# ============================================================

_NLP_SPLIT_RE = re.compile(r"[,|;]\s*")


class PrecomputedIntelligence:
    """
    Fast, fully-vectorised chunk-level intelligence generator.

    All operations use pandas/numpy vectorisation — no per-row loops,
    no heavy NLP models. Scales to millions of rows.

    Produces:
    - Sentiment Analysis       (score, label, intensity)
    - Aspect Extraction        (10 universal aspects)
    - Topic Modeling           (TF-IDF keyword extraction)
    - Complaint Tagging        (typed complaint categories)
    - Severity / Priority      (LOW / MEDIUM / HIGH / CRITICAL)
    - Urgency Detection        (time-sensitive language)
    - Intent Detection         (complaint / inquiry / feedback / comparison / recommendation)
    - Readability Score        (Flesch-like, zero extra cost)
    - Domain Signals           (price / delivery / quality / return / support)
    - Basic Stats              (char, word, sentence counts etc.)
    """

    # ============================================================
    # SENTIMENT PATTERNS
    # ============================================================

    _POS_STRONG = re.compile(
        r"excellent|amazing|awesome|fantastic|outstanding|"
        r"exceptional|superb|brilliant|perfect|love|loved|"
        r"best|incredible|wonderful|magnificent|flawless"
    )
    _POS_MILD = re.compile(
        r"good|great|nice|satisfied|happy|pleased|decent|"
        r"fine|solid|useful|helpful|impressed|comfortable|"
        r"recommend|recommended|worth|smooth|fast|premium|beautiful"
    )
    _NEG_STRONG = re.compile(
        r"terrible|awful|horrible|worst|hate|disgusting|"
        r"atrocious|abysmal|dreadful|pathetic|unacceptable|"
        r"outrageous|furious|fraud|scam|useless|broken|defective"
    )
    _NEG_MILD = re.compile(
        r"bad|poor|slow|cheap|disappointed|disappointing|"
        r"issue|issues|problem|problems|waste|refund|return|"
        r"damaged|fake|missing|late|delay|wrong|stopped|"
        r"unhappy|mediocre|lacking|subpar|inferior"
    )
    _NEGATION = re.compile(
        r"\b(?:not|no|never|neither|nor|barely|hardly|"
        r"scarcely|doesn't|don't|didn't|isn't|wasn't|"
        r"won't|wouldn't|can't|cannot)\b",
        re.IGNORECASE
    )

    # ============================================================
    # ASPECT PATTERNS  (10 universal aspects)
    # ============================================================

    _ASPECTS = {
        "price":       re.compile(r"price|cost|expensive|cheap|value|worth|money|pricing|budget|affordable|overpriced"),
        "quality":     re.compile(r"quality|build|material|durable|durability|premium|sturdy|finish|craftsmanship|texture|feel"),
        "delivery":    re.compile(r"delivery|shipping|shipment|arrived|arrival|late|courier|packaging|packed|dispatch|transit"),
        "support":     re.compile(r"service|support|seller|customer care|customer service|response|resolved|resolution|helpdesk|agent"),
        "usability":   re.compile(r"easy|difficult|simple|complicated|user.friendly|intuitive|confusing|interface|setup|install|use"),
        "performance": re.compile(r"fast|slow|speed|performance|efficient|lag|loading|responsive|quick|smooth|battery|power"),
        "design":      re.compile(r"design|look|appearance|style|color|colour|aesthetic|attractive|sleek|ugly|size|weight|compact"),
        "reliability": re.compile(r"reliable|unreliable|consistent|inconsistent|stable|unstable|crash|bug|error|glitch|stopped working|not working"),
        "content":     re.compile(r"content|information|description|accurate|inaccurate|misleading|details|specs|features|documentation"),
        "comparison":  re.compile(r"better than|worse than|compared to|versus|vs|alternative|competitor|similar|unlike|than other"),
    }

    # ============================================================
    # COMPLAINT TYPE PATTERNS
    # ============================================================

    _COMPLAINT_TYPES = {
        "billing":    re.compile(r"charge|charged|overcharged|billing|invoice|payment|refund|money|price|cost|fee|extra charge"),
        "delivery":   re.compile(r"late|delay|not delivered|missing|lost|wrong item|shipment|shipping|courier|arrived damaged"),
        "quality":    re.compile(r"broken|defective|damaged|poor quality|fake|counterfeit|not working|stopped working|malfunction"),
        "technical":  re.compile(r"error|bug|crash|glitch|not loading|slow|freeze|frozen|not opening|failed|technical issue"),
        "support":    re.compile(r"no response|rude|unhelpful|ignored|not resolved|bad service|poor support|no help|waiting"),
        "policy":     re.compile(r"return policy|refund policy|warranty|guarantee|terms|conditions|policy|not accepted|rejected"),
    }

    # ============================================================
    # URGENCY PATTERNS
    # ============================================================

    _URGENCY = re.compile(
        r"urgent|urgently|immediately|asap|as soon as possible|"
        r"right now|still waiting|been waiting|no response yet|"
        r"days? ago|weeks? ago|escalate|escalation|still not|"
        r"unresolved|follow.?up|reminder|second time|third time|"
        r"not yet|haven't received|have not received"
    )

    # ============================================================
    # INTENT PATTERNS
    # ============================================================

    _INTENT = {
        "complaint":       re.compile(r"complaint|complain|issue|problem|broken|defective|not working|terrible|worst|awful|bad experience|disappointed"),
        "inquiry":         re.compile(r"how to|how do|what is|where is|when will|can i|is it|does it|do you|please help|need help|want to know"),
        "feedback":        re.compile(r"feedback|suggest|suggestion|improve|improvement|would be better|should have|wish|hope|feature request"),
        "comparison":      re.compile(r"better than|worse than|compared to|versus|vs|which one|alternative|recommend between|difference between"),
        "recommendation":  re.compile(r"recommend|worth buying|should i buy|is it good|good product|love this|excellent product|highly recommend|must buy"),
    }

    # ============================================================
    # DOMAIN SIGNALS
    # ============================================================

    _PRICE_SIG    = re.compile(r"price|cost|expensive|cheap|value|worth|money|pricing|budget")
    _DELIVERY_SIG = re.compile(r"delivery|shipping|shipment|arrived|arrival|late|courier|packaging|packed|box")
    _QUALITY_SIG  = re.compile(r"quality|build|material|durable|durability|premium|cheap|sturdy|finish|broken|defective")
    _RETURN_SIG   = re.compile(r"return|refund|replacement|replaced|exchange")
    _SUPPORT_SIG  = re.compile(r"service|support|seller|customer care|customer service|response|resolved|resolution")
    _REVIEW_SIG   = re.compile(r"pros|cons|review|rating|stars|recommend")
    _DIGIT        = re.compile(r"\d")
    _CURRENCY     = re.compile(r"\$|₹|rs\.?|usd|inr|eur", re.IGNORECASE)

    def __init__(
        self,
        output_dir: str = None,
        keep_original_chunk_text: bool = True,
    ):
        self.output_dir = Path(output_dir) if output_dir else Path(PROCESSED_DATA_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_original_chunk_text = keep_original_chunk_text

    # ============================================================
    # VALIDATION
    # ============================================================

    @staticmethod
    def validate_input(chunk_df: pd.DataFrame) -> None:
        if chunk_df is None:
            raise ValueError("Input dataframe is None.")
        if chunk_df.empty:
            raise ValueError("Input dataframe is empty.")
        if "chunk_text" not in chunk_df.columns:
            raise ValueError("'chunk_text' column not found.")

    # ============================================================
    # CLEAN TEXT
    # ============================================================

    @staticmethod
    def _clean_series(series: pd.Series) -> pd.Series:
        return (
            series
            .fillna("")
            .astype(str)
            .str.replace(r"[\n\t\r]+", " ", regex=True)
            .str.replace(r" {2,}", " ", regex=True)
            .str.strip()
        )

    # ============================================================
    # BASIC STATS
    # ============================================================

    @staticmethod
    def _word_count(s: pd.Series) -> pd.Series:
        return s.str.split().str.len().fillna(0).astype("int32")

    @staticmethod
    def _sentence_count(s: pd.Series) -> pd.Series:
        punct = s.str.count(r"[.!?]+").fillna(0).astype("int32")
        non_empty = s.str.len().gt(0)
        result = np.where(non_empty & (punct == 0), 1, punct)
        return pd.Series(result, index=s.index, dtype="int32")

    @staticmethod
    def _avg_word_length(char_count: pd.Series, word_count: pd.Series) -> pd.Series:
        return (
            char_count
            .div(word_count.replace(0, np.nan))
            .fillna(0.0)
            .round(2)
        )

    # ============================================================
    # READABILITY  (Flesch-like, vectorised)
    # Flesch Reading Ease ≈ 206.835
    #                       - 1.015  * (words / sentences)
    #                       - 84.6   * (avg_word_length / 5)
    # Clamped to [0, 100]; higher = easier to read.
    # ============================================================

    @staticmethod
    def _readability(
        word_count: pd.Series,
        sentence_count: pd.Series,
        avg_word_length: pd.Series,
    ) -> pd.Series:

        safe_sentences = sentence_count.replace(0, 1)
        asl = word_count.div(safe_sentences)          # avg sentence length
        asw = avg_word_length.div(5).clip(lower=0.1)  # proxy for syllables

        score = (206.835
                 - 1.015  * asl
                 - 84.6   * asw).clip(0, 100).round(2)

        return score.astype("float32")

    @staticmethod
    def _readability_label(score: pd.Series) -> pd.Series:
        return pd.Series(
            np.select(
                [score >= 70, score >= 50, score >= 30],
                ["easy", "moderate", "difficult"],
                default="very_difficult",
            ),
            index=score.index,
            dtype="object",
        )

    # ============================================================
    # SENTIMENT ANALYSIS
    # ============================================================

    def _sentiment_score(self, lower: pd.Series) -> pd.Series:
        """
        Float score in [-1, +1].
        Strong hits count 2×, mild hits count 1×.
        Negation in the chunk halves the positive score.
        """

        pos = (lower.str.count(self._POS_STRONG) * 2
               + lower.str.count(self._POS_MILD))
        neg = (lower.str.count(self._NEG_STRONG) * 2
               + lower.str.count(self._NEG_MILD))

        has_negation = lower.str.contains(self._NEGATION, na=False)
        pos = pos.where(~has_negation, pos * 0.5)

        total = pos + neg
        score = pos.sub(neg).div(total.replace(0, np.nan)).fillna(0.0)
        return score.clip(-1, 1).round(3).astype("float32")

    def _sentiment_label(self, score: pd.Series) -> pd.Series:
        return pd.Series(
            np.select(
                [score >= 0.3, score <= -0.3, score.between(-0.1, 0.1)],
                ["positive",   "negative",    "neutral"],
                default="mixed",
            ),
            index=score.index,
            dtype="object",
        )

    def _sentiment_intensity(self, score: pd.Series) -> pd.Series:
        abs_score = score.abs()
        return pd.Series(
            np.select(
                [abs_score >= 0.7, abs_score >= 0.4],
                ["strong",         "moderate"],
                default="mild",
            ),
            index=score.index,
            dtype="object",
        )

    # ============================================================
    # ASPECT EXTRACTION  (10 universal aspects)
    # ============================================================

    def _extract_aspects(self, lower: pd.Series) -> pd.Series:
        """
        Returns a comma-separated string of detected aspects per chunk.
        e.g. "price,quality,delivery"
        """

        aspect_hits = pd.DataFrame(index=lower.index)

        for aspect, pattern in self._ASPECTS.items():
            aspect_hits[aspect] = lower.str.contains(pattern, na=False)

        def _join(row):
            return ",".join(col for col in aspect_hits.columns if row[col]) or "none"

        return aspect_hits.apply(_join, axis=1)

    def _aspect_count(self, aspects: pd.Series) -> pd.Series:
        return aspects.apply(
            lambda x: 0 if x == "none" else len(x.split(","))
        ).astype("int16")

    # ============================================================
    # TOPIC MODELING  (TF-IDF keyword extraction, vectorised)
    # ============================================================

    _STOPWORDS = frozenset({
        "the","a","an","and","or","but","in","on","at","to","for",
        "of","with","by","from","is","it","its","this","that","these",
        "those","are","was","were","be","been","being","have","has",
        "had","do","does","did","will","would","could","should","may",
        "might","shall","can","not","no","nor","so","yet","both",
        "either","neither","i","we","you","he","she","they","me","us",
        "him","her","them","my","our","your","his","their","what",
        "which","who","whom","when","where","why","how","all","each",
        "every","both","few","more","most","other","some","such",
        "than","too","very","just","also","as","if","then","into",
        "about","up","out","there","here","get","got","like","just",
    })

    def _extract_topics(self, clean: pd.Series, top_n: int = 5) -> pd.Series:
        """
        Per-chunk top-N keywords by TF-IDF score.
        Returns comma-separated keyword string.
        """

        # Tokenize
        token_series = (
            clean.str.lower()
            .str.replace(r"[^a-z\s]", " ", regex=True)
            .str.split()
        )

        # Filter stopwords and short tokens
        filtered = token_series.apply(
            lambda tokens: [
                t for t in (tokens or [])
                if t not in self._STOPWORDS and len(t) > 3
            ]
        )

        # Document frequency across all chunks
        from collections import Counter
        df_counts: Counter = Counter()
        for tokens in filtered:
            df_counts.update(set(tokens))

        total_docs = max(len(clean), 1)

        def _top_keywords(tokens):
            if not tokens:
                return "none"
            tf = Counter(tokens)
            total = len(tokens)
            scored = {
                t: (tf[t] / total) * np.log((total_docs + 1) / (df_counts.get(t, 0) + 1))
                for t in tf
            }
            top = sorted(scored, key=scored.get, reverse=True)[:top_n]
            return ",".join(top) if top else "none"

        return filtered.apply(_top_keywords)

    # ============================================================
    # COMPLAINT TAGGING
    # ============================================================

    def _complaint_type(self, lower: pd.Series) -> pd.Series:
        """
        Returns comma-separated complaint categories detected.
        e.g. "billing,delivery"
        """

        type_hits = pd.DataFrame(index=lower.index)

        for ctype, pattern in self._COMPLAINT_TYPES.items():
            type_hits[ctype] = lower.str.contains(pattern, na=False)

        def _join(row):
            return ",".join(col for col in type_hits.columns if row[col]) or "none"

        return type_hits.apply(_join, axis=1)

    def _complaint_score(self, lower: pd.Series) -> pd.Series:
        neg_strong = lower.str.count(self._NEG_STRONG)
        neg_mild   = lower.str.count(self._NEG_MILD)
        return (neg_strong * 2 + neg_mild).fillna(0).astype("int16")

    # ============================================================
    # SEVERITY / PRIORITY
    # ============================================================

    def _severity(
        self,
        complaint_score: pd.Series,
        sentiment_score: pd.Series,
        complaint_type:  pd.Series,
        urgency:         pd.Series,
    ) -> pd.Series:
        """
        CRITICAL : complaint_score >= 4 AND strongly negative OR urgent
        HIGH     : complaint_score >= 3 OR (score <= -0.6 AND is_complaint)
        MEDIUM   : complaint_score >= 1 OR mildly negative
        LOW      : everything else
        """

        is_urgent        = urgency
        is_strong_neg    = sentiment_score <= -0.6
        has_complaint    = complaint_type.ne("none")
        high_complaint   = complaint_score >= 3
        critical_complaint = complaint_score >= 4

        return pd.Series(
            np.select(
                [
                    (critical_complaint & is_strong_neg) | (critical_complaint & is_urgent),
                    high_complaint | (is_strong_neg & has_complaint),
                    (complaint_score >= 1) | (sentiment_score <= -0.2),
                ],
                ["CRITICAL", "HIGH", "MEDIUM"],
                default="LOW",
            ),
            index=complaint_score.index,
            dtype="object",
        )

    # ============================================================
    # URGENCY DETECTION
    # ============================================================

    def _urgency(self, lower: pd.Series) -> pd.Series:
        return lower.str.contains(self._URGENCY, na=False)

    # ============================================================
    # INTENT DETECTION
    # ============================================================

    def _intent(self, lower: pd.Series) -> pd.Series:
        """
        Returns the dominant intent. Priority: complaint > inquiry >
        feedback > comparison > recommendation > general
        """

        intent_hits = pd.DataFrame(index=lower.index)

        for intent, pattern in self._INTENT.items():
            intent_hits[intent] = lower.str.contains(pattern, na=False)

        priority = ["complaint", "inquiry", "feedback", "comparison", "recommendation"]

        def _dominant(row):
            for intent in priority:
                if row[intent]:
                    return intent
            return "general"

        return intent_hits.apply(_dominant, axis=1)

    # ============================================================
    # SAFE LENGTH
    # ============================================================

    @classmethod
    def _safe_len_series(cls, series: pd.Series) -> pd.Series:
        def _len(val) -> int:
            if val is None or (isinstance(val, float) and val != val):
                return 0
            if isinstance(val, (list, tuple, set)):
                return len(val)
            s = str(val).strip()
            return len([x for x in _NLP_SPLIT_RE.split(s) if x.strip()]) if s else 0
        return series.map(_len).astype("int16")

    # ============================================================
    # MAIN GENERATION
    # ============================================================

    def generate(self, chunk_df: pd.DataFrame) -> pd.DataFrame:

        self.validate_input(chunk_df)
        result = chunk_df.copy()

        # ── CLEAN + LOWERCASE ───────────────────────────────────
        clean                      = self._clean_series(result["chunk_text"])
        result["clean_chunk_text"] = clean
        lower                      = clean.str.lower()

        # ── BASIC STATS ─────────────────────────────────────────
        char_count    = clean.str.len().astype("int32")
        word_count    = self._word_count(clean)
        sentence_count = self._sentence_count(clean)
        avg_word_len  = self._avg_word_length(char_count, word_count)

        result["char_count"]      = char_count
        result["word_count"]      = word_count
        result["sentence_count"]  = sentence_count
        result["avg_word_length"] = avg_word_len
        result["token_estimate"]  = word_count.mul(1.3).round().astype("int32")

        # ── FLAGS ───────────────────────────────────────────────
        result["is_blank"]        = char_count.eq(0)
        result["has_numbers"]     = clean.str.contains(self._DIGIT,    na=False)
        result["has_question"]    = clean.str.contains(r"\?",          na=False, regex=True)
        result["has_exclamation"] = clean.str.contains(r"!",           na=False, regex=True)
        result["has_currency"]    = clean.str.contains(self._CURRENCY, na=False)
        result["has_percent"]     = clean.str.contains(r"%",           na=False, regex=True)

        # ── LENGTH BUCKET ────────────────────────────────────────
        result["length_bucket"] = pd.Series(
            np.select(
                [word_count < 20, word_count < 80, word_count < 150],
                ["short",         "medium",         "long"],
                default="very_long",
            ),
            index=result.index,
        )

        # ── READABILITY ──────────────────────────────────────────
        result["readability_score"] = self._readability(word_count, sentence_count, avg_word_len)
        result["readability_label"] = self._readability_label(result["readability_score"])

        # ── SENTIMENT ANALYSIS ───────────────────────────────────
        sentiment_score             = self._sentiment_score(lower)
        result["sentiment_score"]   = sentiment_score
        result["sentiment_label"]   = self._sentiment_label(sentiment_score)
        result["sentiment_intensity"] = self._sentiment_intensity(sentiment_score)

        # ── ASPECT EXTRACTION ────────────────────────────────────
        aspects                  = self._extract_aspects(lower)
        result["aspects_detected"] = aspects
        result["aspect_count"]   = self._aspect_count(aspects)

        # ── TOPIC MODELING ───────────────────────────────────────
        result["topics_keywords"] = self._extract_topics(clean)

        # ── COMPLAINT TAGGING ────────────────────────────────────
        complaint_score              = self._complaint_score(lower)
        complaint_type               = self._complaint_type(lower)
        result["complaint_score"]    = complaint_score
        result["complaint_type"]     = complaint_type
        result["is_complaint"]       = complaint_type.ne("none")

        # ── URGENCY DETECTION ────────────────────────────────────
        urgency                  = self._urgency(lower)
        result["is_urgent"]      = urgency

        # ── INTENT DETECTION ─────────────────────────────────────
        result["intent"]         = self._intent(lower)

        # ── SEVERITY / PRIORITY ──────────────────────────────────
        result["severity"]       = self._severity(
            complaint_score,
            sentiment_score,
            complaint_type,
            urgency,
        )

        # ── DOMAIN SIGNALS ───────────────────────────────────────
        result["price_signal"]          = lower.str.contains(self._PRICE_SIG,    na=False)
        result["delivery_signal"]       = lower.str.contains(self._DELIVERY_SIG, na=False)
        result["quality_signal"]        = lower.str.contains(self._QUALITY_SIG,  na=False)
        result["return_refund_signal"]  = lower.str.contains(self._RETURN_SIG,   na=False)
        result["support_signal"]        = lower.str.contains(self._SUPPORT_SIG,  na=False)
        result["review_pattern_signal"] = lower.str.contains(self._REVIEW_SIG,   na=False)

        # ── OPTIONAL NLP COLUMNS (preserve if exist) ─────────────
        for col in ("entities", "topics", "aspects", "sentiment",
                    "source_type", "document_type"):
            if col not in result.columns:
                result[col] = np.nan

        result["entity_count"] = self._safe_len_series(result["entities"])
        result["topic_count"]  = self._safe_len_series(result["topics"])

        # ── CHUNK_ID SAFETY NET ──────────────────────────────────
        if "chunk_id" not in result.columns:
            result["chunk_id"] = range(len(result))

        # ── COLUMN ORDER ─────────────────────────────────────────
        preferred = [
            # identifiers
            "record_id", "chunk_id", "source_columns",
            # text
            "chunk_text", "clean_chunk_text",
            # basic stats
            "char_count", "word_count", "sentence_count",
            "avg_word_length", "token_estimate", "length_bucket",
            # readability
            "readability_score", "readability_label",
            # flags
            "is_blank", "has_numbers", "has_question",
            "has_exclamation", "has_currency", "has_percent",
            # sentiment
            "sentiment_score", "sentiment_label", "sentiment_intensity",
            # aspects
            "aspects_detected", "aspect_count",
            # topics
            "topics_keywords",
            # complaints
            "complaint_score", "complaint_type", "is_complaint",
            # urgency + intent
            "is_urgent", "intent",
            # severity
            "severity",
            # domain signals
            "price_signal", "delivery_signal", "quality_signal",
            "return_refund_signal", "support_signal", "review_pattern_signal",
            # optional NLP pass-through
            "entity_count", "topic_count", "aspect_count",
            "entities", "topics", "aspects", "sentiment",
            "source_type", "document_type",
        ]

        if not self.keep_original_chunk_text:
            result = result.drop(columns=["chunk_text"], errors="ignore")

        ordered   = [c for c in preferred if c in result.columns]
        remaining = [c for c in result.columns if c not in ordered]
        return result[ordered + remaining]

    # ============================================================
    # SAVE
    # ============================================================

    def save(
        self,
        intelligence_df: pd.DataFrame,
        file_name: str = "precomputed_intelligence.csv",
    ) -> None:
        out = self.output_dir / file_name
        intelligence_df.to_csv(out, index=False)
        print(f"\n  Intelligence saved -> {out}")

    # ============================================================
    # SUMMARY
    # ============================================================

    def summary(self, intelligence_df: pd.DataFrame) -> None:

        print("\n" + "=" * 60)
        print("PRECOMPUTED INTELLIGENCE SUMMARY")
        print("=" * 60)
        print(f"  Records : {len(intelligence_df):,}")

        if intelligence_df.empty:
            print("=" * 60)
            return

        for col, label in [
            ("length_bucket",      "Length Bucket"),
            ("readability_label",  "Readability"),
            ("sentiment_label",    "Sentiment"),
            ("sentiment_intensity","Sentiment Intensity"),
            ("aspects_detected",   "Top Aspects"),
            ("complaint_type",     "Complaint Types"),
            ("is_complaint",       "Is Complaint"),
            ("is_urgent",          "Is Urgent"),
            ("intent",             "Intent"),
            ("severity",           "Severity / Priority"),
        ]:
            if col in intelligence_df.columns:
                print(f"\n  {label}:")
                print(
                    intelligence_df[col]
                    .value_counts(dropna=False)
                    .head(10)
                    .to_string()
                )

        print("=" * 60)