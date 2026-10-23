"""
feature_extraction.py
---------------------
Transform raw texts into a numerical vector space for Authorship Attribution
and Generative AI Detection, implementing the feature families identified
in the literature review (Points 6, 7).

Feature Families Extracted:
1. Stylometric & Lexical Richness Features:
   - Type-Token Ratio (TTR: V / N)
   - Root TTR / Guiraud's R (V / sqrt(N))
   - Yule's Characteristic K (statistical repetition invariant to text length)
   - Simpson's D Index (probability of two random words being identical)
   - Hapax Legomena Ratio (V_1 / V: vocabulary frequency of once-occurring words)
   - Dislegomena Ratio (V_2 / V: twice-occurring words)
   - Average Word Length & Standard Deviation
   - Average Sentence Length & Sentence Count
   - Digit Density & Uppercase Character Ratio
   - Punctuation Density (total and individual mark frequencies: comma, period,
     semicolon, colon, exclamation, question mark, dash/hyphen, quotation, parentheses)
2. Topic-Invariant Function Words (Stop-Words Frequencies):
   - Relative frequency distribution of 70 closed-class syntactic words
     (prepositions, conjunctions, pronouns, auxiliary verbs) based on Mosteller & Wallace
     and Stamatatos (2009).
3. Sub-Word Character N-grams (TF-IDF):
   - Boundary-aware character 3-to-4-grams (analyzer='char_wb') capturing morphological
     affixes, sub-word stems, and punctuation n-grams.
4. Word N-grams (TF-IDF):
   - Word unigrams and bigrams preserving function words to capture short syntactic structures.
5. Unified Baseline Feature Pipeline:
   - Scikit-learn compliant composite transformer with feature name tracking,
     producing sparse (scipy.sparse.csr_matrix) or dense feature matrices.
"""

from collections import Counter
import math
import re
from typing import List, Dict, Any, Optional, Sequence, Union, Tuple
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


# -------------------------------------------------------------------------
# 1. Closed-class Topic-Invariant Function Words
# -------------------------------------------------------------------------
TOP_FUNCTION_WORDS = [
    "the", "of", "and", "to", "a", "in", "that", "is", "was", "he",
    "for", "it", "with", "as", "his", "on", "be", "at", "by", "i",
    "this", "had", "not", "are", "but", "from", "or", "have", "an", "they",
    "which", "one", "you", "were", "her", "all", "she", "there", "would", "their",
    "we", "him", "been", "has", "when", "who", "will", "more", "no", "if",
    "out", "so", "said", "what", "up", "its", "about", "into", "than", "them",
    "can", "only", "other", "some", "could", "time", "these", "two", "may", "then"
]


# -------------------------------------------------------------------------
# 2. Stylometric & Lexical Richness Feature Extractor
# -------------------------------------------------------------------------
class StylometricFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts multi-level stylometric, structural, and lexical richness features
    from raw texts into a dense numerical matrix.
    """

    FEATURE_NAMES = [
        "ttr",
        "root_ttr",
        "yule_k",
        "simpson_d",
        "hapax_ratio",
        "dislegomena_ratio",
        "avg_word_len",
        "word_len_std",
        "avg_sentence_len",
        "sentence_count",
        "char_count",
        "word_count",
        "digit_density",
        "uppercase_ratio",
        "punct_density",
        "comma_density",
        "period_density",
        "semicolon_density",
        "colon_density",
        "exclamation_density",
        "question_density",
        "dash_density",
        "quote_density",
        "parenthesis_density",
        # Point 13 Advanced Custom Derived Features:
        "ratio_word_to_sentence_len",
        "syntactic_pause_ratio",
        "expressive_punct_ratio",
        "functional_lexical_ratio",
        "brunet_w",
        "honore_r",
    ]

    def __init__(self):
        self._word_re = re.compile(r"\b[a-zA-Z]+\b")
        self._sentence_re = re.compile(r"[.!?]+(?:\s+|$)")
        self._punct_re = re.compile(r"[^\w\s]")

    def fit(self, X: Sequence[str], y: Optional[Any] = None) -> "StylometricFeatureExtractor":
        return self

    def transform(self, X: Sequence[str]) -> np.ndarray:
        """
        Transforms a sequence of raw text strings into a 2D numpy array of stylometric metrics.
        Shape: (len(X), len(FEATURE_NAMES)).
        """
        features = np.zeros((len(X), len(self.FEATURE_NAMES)), dtype=np.float64)

        for i, text in enumerate(X):
            features[i, :] = self._extract_single(text)

        return features

    def _extract_single(self, text: str) -> np.ndarray:
        if not text or not isinstance(text, str):
            return np.zeros(len(self.FEATURE_NAMES), dtype=np.float64)

        total_chars = max(len(text), 1)
        words = self._word_re.findall(text.lower())
        n_words = len(words)

        if n_words == 0:
            return np.zeros(len(self.FEATURE_NAMES), dtype=np.float64)

        word_counts = Counter(words)
        v = len(word_counts)  # Vocabulary size (unique tokens)
        freq_spectrum = Counter(word_counts.values())

        # 1. Lexical Richness Metrics
        ttr = v / n_words
        root_ttr = v / math.sqrt(n_words)

        # Yule's Characteristic K: 10^4 * (sum(m^2 * V(m, N)) - N) / N^2
        m2_sum = sum((m ** 2) * vm for m, vm in freq_spectrum.items())
        yule_k = 10000.0 * (m2_sum - n_words) / max(n_words ** 2, 1)

        # Simpson's D: sum(n_i * (n_i - 1)) / (N * (N - 1))
        if n_words > 1:
            simpson_d = sum(cnt * (cnt - 1) for cnt in word_counts.values()) / (n_words * (n_words - 1))
        else:
            simpson_d = 0.0

        hapax_count = freq_spectrum.get(1, 0)
        dislegomena_count = freq_spectrum.get(2, 0)
        hapax_ratio = hapax_count / max(v, 1)
        dislegomena_ratio = dislegomena_count / max(v, 1)

        # 2. Structural & Morphological Features
        word_lengths = [len(w) for w in words]
        avg_word_len = float(np.mean(word_lengths))
        word_len_std = float(np.std(word_lengths))

        sentences = [s.strip() for s in self._sentence_re.split(text) if s.strip()]
        sentence_count = max(len(sentences), 1)
        avg_sentence_len = n_words / sentence_count

        digits = sum(c.isdigit() for c in text)
        digit_density = digits / total_chars

        uppercases = sum(c.isupper() for c in text)
        uppercase_ratio = uppercases / total_chars

        # 3. Punctuation Densities
        puncts = len(self._punct_re.findall(text))
        punct_density = puncts / total_chars

        comma_density = text.count(",") / total_chars
        period_density = text.count(".") / total_chars
        semicolon_density = text.count(";") / total_chars
        colon_density = text.count(":") / total_chars
        exclamation_density = text.count("!") / total_chars
        question_density = text.count("?") / total_chars
        dash_density = (text.count("-") + text.count("—") + text.count("–")) / total_chars
        quote_density = (text.count('"') + text.count("'") + text.count("“") + text.count("”")) / total_chars
        paren_density = (text.count("(") + text.count(")") + text.count("[") + text.count("]")) / total_chars

        # 4. Point 13: Advanced Custom Derived Features
        ratio_word_to_sentence_len = avg_word_len / max(avg_sentence_len, 0.001)
        syntactic_pause_ratio = text.count(",") / max(sentence_count, 1)
        expressive_punct = (
            text.count("!") + text.count("?") + text.count("-") +
            text.count("—") + text.count("–") + text.count('"') + text.count("'")
        )
        expressive_punct_ratio = expressive_punct / max(puncts, 1)

        n_func = sum(word_counts.get(fw, 0) for fw in TOP_FUNCTION_WORDS)
        n_content = max(n_words - n_func, 1)
        functional_lexical_ratio = n_func / n_content

        # Brunet's W: N^(V^-0.172)
        brunet_w = (n_words ** (v ** -0.172)) if (v > 0 and n_words > 0) else 0.0

        # Honoré's R: 100 * ln(N) / (1 - V_1/V)
        honore_r = (100.0 * math.log(max(n_words, 2))) / max(1.0 - (hapax_count / max(v, 1)), 0.01)

        return np.array([
            ttr,
            root_ttr,
            yule_k,
            simpson_d,
            hapax_ratio,
            dislegomena_ratio,
            avg_word_len,
            word_len_std,
            avg_sentence_len,
            float(sentence_count),
            float(total_chars),
            float(n_words),
            digit_density,
            uppercase_ratio,
            punct_density,
            comma_density,
            period_density,
            semicolon_density,
            colon_density,
            exclamation_density,
            question_density,
            dash_density,
            quote_density,
            paren_density,
            # Point 13 Derived:
            ratio_word_to_sentence_len,
            syntactic_pause_ratio,
            expressive_punct_ratio,
            functional_lexical_ratio,
            brunet_w,
            honore_r,
        ], dtype=np.float64)

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> List[str]:
        return [f"stylometry__{name}" for name in self.FEATURE_NAMES]


# -------------------------------------------------------------------------
# 3. Function Words (Stop-words) Extractor
# -------------------------------------------------------------------------
class FunctionWordExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts relative frequencies of closed-class function words (stop-words).
    Normalizes frequencies by text word length to ensure topic-invariance.
    """

    def __init__(self, function_words: Optional[List[str]] = None):
        self.function_words = function_words or TOP_FUNCTION_WORDS
        self._word_re = re.compile(r"\b[a-zA-Z]+\b")

    def fit(self, X: Sequence[str], y: Optional[Any] = None) -> "FunctionWordExtractor":
        return self

    def transform(self, X: Sequence[str]) -> np.ndarray:
        features = np.zeros((len(X), len(self.function_words)), dtype=np.float64)
        word_to_idx = {w: idx for idx, w in enumerate(self.function_words)}

        for i, text in enumerate(X):
            if not text or not isinstance(text, str):
                continue
            words = self._word_re.findall(text.lower())
            n_words = len(words)
            if n_words == 0:
                continue

            counts = Counter(words)
            for w, idx in word_to_idx.items():
                features[i, idx] = counts.get(w, 0) / n_words

        return features

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> List[str]:
        return [f"func_word__{w}" for w in self.function_words]


# -------------------------------------------------------------------------
# 4. Composite Baseline Feature Pipeline
# -------------------------------------------------------------------------
class BaselineFeaturePipeline:
    """
    Unified numerical vectorizer combining:
    1. Stylometric features (StandardScaler applied)
    2. Function words frequencies (StandardScaler applied)
    3. Character N-grams TF-IDF (analyzer='char_wb', n=(3, 4))
    4. Optional Word N-grams TF-IDF (analyzer='word', n=(1, 2))

    Outputs a sparse scipy.sparse.csr_matrix or dense numpy array,
    preserving full feature names across all dimensions.
    """

    def __init__(
        self,
        char_ngram_range: Tuple[int, int] = (3, 4),
        char_max_features: int = 3000,
        include_word_ngrams: bool = True,
        word_ngram_range: Tuple[int, int] = (1, 2),
        word_max_features: int = 1500,
        scale_dense_features: bool = True,
    ):
        self.char_ngram_range = char_ngram_range
        self.char_max_features = char_max_features
        self.include_word_ngrams = include_word_ngrams
        self.word_ngram_range = word_ngram_range
        self.word_max_features = word_max_features
        self.scale_dense_features = scale_dense_features

        # Sub-components
        self.stylometric_extractor = StylometricFeatureExtractor()
        self.function_word_extractor = FunctionWordExtractor()

        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.char_ngram_range,
            max_features=self.char_max_features,
            sublinear_tf=True,
        )

        if self.include_word_ngrams:
            self.word_vectorizer = TfidfVectorizer(
                analyzer="word",
                ngram_range=self.word_ngram_range,
                max_features=self.word_max_features,
                sublinear_tf=True,
            )
        else:
            self.word_vectorizer = None

        self.dense_scaler = StandardScaler() if self.scale_dense_features else None
        self.feature_names_: List[str] = []
        self.is_fitted: bool = False

    def fit(self, texts: Sequence[str], y: Optional[Any] = None) -> "BaselineFeaturePipeline":
        """Fit vectorizers and scalers on the training text corpus."""
        # Fit stylometric & function extractors (stateless)
        self.stylometric_extractor.fit(texts)
        self.function_word_extractor.fit(texts)

        # Fit TF-IDF character n-grams
        self.char_vectorizer.fit(texts)

        # Fit TF-IDF word n-grams if enabled
        if self.word_vectorizer is not None:
            self.word_vectorizer.fit(texts)

        # Compute dense features to fit scaler
        sty_feat = self.stylometric_extractor.transform(texts)
        func_feat = self.function_word_extractor.transform(texts)
        dense_feat = np.hstack([sty_feat, func_feat])

        if self.dense_scaler is not None:
            self.dense_scaler.fit(dense_feat)

        # Build feature names
        names = []
        names.extend(self.stylometric_extractor.get_feature_names_out())
        names.extend(self.function_word_extractor.get_feature_names_out())
        names.extend([f"char_ngram__{t}" for t in self.char_vectorizer.get_feature_names_out()])
        if self.word_vectorizer is not None:
            names.extend([f"word_ngram__{t}" for t in self.word_vectorizer.get_feature_names_out()])

        self.feature_names_ = names
        self.is_fitted = True
        return self

    def transform(self, texts: Sequence[str]) -> sp.csr_matrix:
        """
        Transforms texts into a combined sparse CSR feature matrix.
        Shape: (len(texts), total_features).
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before calling transform().")

        # 1. Dense stylometric & function features
        sty_feat = self.stylometric_extractor.transform(texts)
        func_feat = self.function_word_extractor.transform(texts)
        dense_feat = np.hstack([sty_feat, func_feat])

        if self.dense_scaler is not None:
            dense_feat = self.dense_scaler.transform(dense_feat)

        dense_sparse = sp.csr_matrix(dense_feat)

        # 2. Sparse TF-IDF matrices
        char_sparse = self.char_vectorizer.transform(texts)

        matrices = [dense_sparse, char_sparse]
        if self.word_vectorizer is not None:
            word_sparse = self.word_vectorizer.transform(texts)
            matrices.append(word_sparse)

        return sp.hstack(matrices, format="csr")

    def fit_transform(self, texts: Sequence[str], y: Optional[Any] = None) -> sp.csr_matrix:
        """Fits the pipeline on texts and immediately returns the feature matrix."""
        self.fit(texts, y)
        return self.transform(texts)

    def get_feature_names_out(self) -> List[str]:
        """Returns the list of all feature names in exact column order."""
        return self.feature_names_

    @property
    def total_features(self) -> int:
        return len(self.feature_names_)


# -------------------------------------------------------------------------
# 5. Helper Factory Function
# -------------------------------------------------------------------------
def create_baseline_pipeline(
    char_max_features: int = 3000,
    word_max_features: int = 1500,
    include_word_ngrams: bool = True,
) -> BaselineFeaturePipeline:
    """Convenience factory for standard baseline feature extraction pipeline."""
    return BaselineFeaturePipeline(
        char_ngram_range=(3, 4),
        char_max_features=char_max_features,
        include_word_ngrams=include_word_ngrams,
        word_ngram_range=(1, 2),
        word_max_features=word_max_features,
        scale_dense_features=True,
    )
