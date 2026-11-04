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
        sty_feat, func_feat = self._extract_dense_raw(texts)
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

    def _extract_dense_raw(self, texts: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts unscaled stylometric and function-word matrices with NaN/Inf protection."""
        sty_feat = np.nan_to_num(
            self.stylometric_extractor.transform(texts), nan=0.0, posinf=0.0, neginf=0.0
        )
        func_feat = np.nan_to_num(
            self.function_word_extractor.transform(texts), nan=0.0, posinf=0.0, neginf=0.0
        )
        return sty_feat, func_feat

    def fit_transform(self, texts: Sequence[str], y: Optional[Any] = None) -> sp.csr_matrix:
        """Fits the pipeline on texts and returns the feature matrix in a single pass."""
        X, _ = self.fit_transform_with_raw_dense(texts, y)
        return X

    def fit_transform_with_raw_dense(
        self, texts: Sequence[str], y: Optional[Any] = None
    ) -> Tuple[sp.csr_matrix, np.ndarray]:
        """
        Fits the pipeline on training texts in a single pass and returns:
        - X_sparse: Combined CSR matrix (with scaled dense features + TF-IDF n-grams)
        - dense_raw: Unscaled dense matrix (stylometry + function words)
        """
        self.stylometric_extractor.fit(texts)
        self.function_word_extractor.fit(texts)

        sty_feat, func_feat = self._extract_dense_raw(texts)
        dense_raw = np.hstack([sty_feat, func_feat])

        if self.dense_scaler is not None:
            dense_scaled = self.dense_scaler.fit_transform(dense_raw)
        else:
            dense_scaled = dense_raw

        dense_sparse = sp.csr_matrix(dense_scaled)
        char_sparse = self.char_vectorizer.fit_transform(texts)

        matrices = [dense_sparse, char_sparse]
        if self.word_vectorizer is not None:
            word_sparse = self.word_vectorizer.fit_transform(texts)
            matrices.append(word_sparse)

        # Build feature names
        names = []
        names.extend(self.stylometric_extractor.get_feature_names_out())
        names.extend(self.function_word_extractor.get_feature_names_out())
        names.extend([f"char_ngram__{t}" for t in self.char_vectorizer.get_feature_names_out()])
        if self.word_vectorizer is not None:
            names.extend([f"word_ngram__{t}" for t in self.word_vectorizer.get_feature_names_out()])

        self.feature_names_ = names
        self.is_fitted = True

        return sp.hstack(matrices, format="csr"), dense_raw

    def transform_with_raw_dense(self, texts: Sequence[str]) -> Tuple[sp.csr_matrix, np.ndarray]:
        """
        Transforms texts using the already-fitted pipeline and returns:
        - X_sparse: Combined CSR matrix
        - dense_raw: Unscaled dense matrix (stylometry + function words)
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before calling transform_with_raw_dense().")

        sty_feat, func_feat = self._extract_dense_raw(texts)
        dense_raw = np.hstack([sty_feat, func_feat])

        if self.dense_scaler is not None:
            dense_scaled = self.dense_scaler.transform(dense_raw)
        else:
            dense_scaled = dense_raw

        dense_sparse = sp.csr_matrix(dense_scaled)
        char_sparse = self.char_vectorizer.transform(texts)

        matrices = [dense_sparse, char_sparse]
        if self.word_vectorizer is not None:
            word_sparse = self.word_vectorizer.transform(texts)
            matrices.append(word_sparse)

        return sp.hstack(matrices, format="csr"), dense_raw

    def get_family_slices(self) -> Dict[str, Tuple[int, int]]:
        """Returns column index ranges (start, end) for each feature family."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before querying family slices.")
        n_sty = len(self.stylometric_extractor.get_feature_names_out())
        n_func = len(self.function_word_extractor.get_feature_names_out())
        n_char = len(self.char_vectorizer.get_feature_names_out())
        n_word = len(self.word_vectorizer.get_feature_names_out()) if self.word_vectorizer else 0

        s_sty = (0, n_sty)
        s_func = (n_sty, n_sty + n_func)
        s_char = (n_sty + n_func, n_sty + n_func + n_char)
        s_word = (n_sty + n_func + n_char, n_sty + n_func + n_char + n_word)

        return {
            "stylometry": s_sty,
            "function_words": s_func,
            "char_ngrams": s_char,
            "word_ngrams": s_word,
            "dense_all": (0, n_sty + n_func),
            "ngrams_all": (n_sty + n_func, n_sty + n_func + n_char + n_word),
            "all": (0, n_sty + n_func + n_char + n_word),
        }

    def get_feature_names_out(self) -> List[str]:
        """Returns the list of all feature names in exact column order."""
        return self.feature_names_

    @property
    def total_features(self) -> int:
        return len(self.feature_names_)


# -------------------------------------------------------------------------
# 5. Helper Factory & Full Corpus Caching Functions
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


def extract_and_cache_full_corpus(
    train_path,
    val_path,
    output_dir,
    char_max_features: int = 3000,
    word_max_features: int = 1500,
    max_records: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Executes full-scale feature extraction on Train and Validation sets (strictly fitting
    only on Train to prevent data leakage) and serializes compressed matrices and metadata
    to `output_dir` (e.g., data/processed/).
    """
    import json
    from pathlib import Path
    import time
    import pandas as pd
    from src.data_utils import iter_jsonl

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Train records
    t0_load = time.time()
    train_texts, train_labels, train_models, train_genres, train_ids = [], [], [], [], []
    for i, rec in enumerate(iter_jsonl(Path(train_path))):
        if max_records and i >= max_records:
            break
        train_ids.append(rec.get("id", f"train_{i}"))
        train_texts.append(rec.get("text", ""))
        train_labels.append(int(rec.get("label", 0)))
        train_models.append(rec.get("model", "unknown"))
        train_genres.append(rec.get("genre", "unknown"))

    # 2. Load Validation records
    val_texts, val_labels, val_models, val_genres, val_ids = [], [], [], [], []
    if val_path and Path(val_path).exists():
        for i, rec in enumerate(iter_jsonl(Path(val_path))):
            if max_records and i >= max_records:
                break
            val_ids.append(rec.get("id", f"val_{i}"))
            val_texts.append(rec.get("text", ""))
            val_labels.append(int(rec.get("label", 0)))
            val_models.append(rec.get("model", "unknown"))
            val_genres.append(rec.get("genre", "unknown"))

    load_time = time.time() - t0_load

    # 3. Fit & Transform Train (Strict no-leakage)
    pipeline = create_baseline_pipeline(
        char_max_features=char_max_features,
        word_max_features=word_max_features,
        include_word_ngrams=True,
    )

    t0_train = time.time()
    X_train, X_dense_raw_train = pipeline.fit_transform_with_raw_dense(train_texts)
    train_extract_time = time.time() - t0_train

    # 4. Transform Validation
    t0_val = time.time()
    if val_texts:
        X_val, X_dense_raw_val = pipeline.transform_with_raw_dense(val_texts)
    else:
        X_val = sp.csr_matrix((0, X_train.shape[1]), dtype=np.float64)
        X_dense_raw_val = np.zeros((0, X_dense_raw_train.shape[1]), dtype=np.float64)
    val_extract_time = time.time() - t0_val

    # 5. Encode targets (Binary + Multiclass Model Attribution)
    # Put 'human' at index 0, followed by sorted AI models
    unique_ai_models = sorted({m for m in set(train_models) | set(val_models) if m != "human"})
    model_classes = ["human"] + unique_ai_models
    model_to_idx = {m: idx for idx, m in enumerate(model_classes)}

    y_bin_train = np.array(train_labels, dtype=np.int64)
    y_bin_val = np.array(val_labels, dtype=np.int64)
    y_model_train = np.array([model_to_idx[m] for m in train_models], dtype=np.int64)
    y_model_val = np.array([model_to_idx[m] for m in val_models], dtype=np.int64)

    # 6. Integrity Audit
    train_nan_count = int(np.isnan(X_train.data).sum())
    train_inf_count = int(np.isinf(X_train.data).sum())
    val_nan_count = int(np.isnan(X_val.data).sum()) if X_val.shape[0] > 0 else 0
    val_inf_count = int(np.isinf(X_val.data).sum()) if X_val.shape[0] > 0 else 0

    train_sparsity = 100.0 * (1.0 - (X_train.nnz / max(X_train.shape[0] * X_train.shape[1], 1)))
    val_sparsity = 100.0 * (1.0 - (X_val.nnz / max(X_val.shape[0] * X_val.shape[1], 1))) if X_val.shape[0] > 0 else 0.0
    ram_mb = (
        X_train.data.nbytes + X_train.indices.nbytes + X_train.indptr.nbytes +
        X_val.data.nbytes + X_val.indices.nbytes + X_val.indptr.nbytes +
        X_dense_raw_train.nbytes + X_dense_raw_val.nbytes
    ) / (1024.0 * 1024.0)

    # 7. Save artifacts to disk
    sp.save_npz(out_path / "X_train.npz", X_train)
    sp.save_npz(out_path / "X_val.npz", X_val)
    np.save(out_path / "X_dense_raw_train.npy", X_dense_raw_train)
    np.save(out_path / "X_dense_raw_val.npy", X_dense_raw_val)
    np.save(out_path / "y_bin_train.npy", y_bin_train)
    np.save(out_path / "y_bin_val.npy", y_bin_val)
    np.save(out_path / "y_model_train.npy", y_model_train)
    np.save(out_path / "y_model_val.npy", y_model_val)

    df_meta_train = pd.DataFrame({
        "id": train_ids,
        "label": train_labels,
        "model": train_models,
        "model_idx": y_model_train,
        "genre": train_genres,
    })
    df_meta_train.to_csv(out_path / "meta_train.csv", index=False)

    df_meta_val = pd.DataFrame({
        "id": val_ids,
        "label": val_labels,
        "model": val_models,
        "model_idx": y_model_val,
        "genre": val_genres,
    })
    df_meta_val.to_csv(out_path / "meta_val.csv", index=False)

    feature_names = pipeline.get_feature_names_out()
    dense_feature_names = (
        pipeline.stylometric_extractor.get_feature_names_out() +
        pipeline.function_word_extractor.get_feature_names_out()
    )
    family_slices = {k: list(v) for k, v in pipeline.get_family_slices().items()}

    metadata = {
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_features": int(X_train.shape[1]),
        "n_dense_features": len(dense_feature_names),
        "family_slices": family_slices,
        "model_classes": model_classes,
        "model_to_idx": model_to_idx,
        "load_time_sec": round(load_time, 3),
        "train_extract_time_sec": round(train_extract_time, 3),
        "val_extract_time_sec": round(val_extract_time, 3),
        "total_extract_time_sec": round(train_extract_time + val_extract_time, 3),
        "train_throughput_docs_per_sec": round(X_train.shape[0] / max(train_extract_time, 1e-6), 1),
        "val_throughput_docs_per_sec": round(X_val.shape[0] / max(val_extract_time, 1e-6), 1) if X_val.shape[0] > 0 else 0.0,
        "train_sparsity_pct": round(train_sparsity, 3),
        "val_sparsity_pct": round(val_sparsity, 3),
        "train_nnz": int(X_train.nnz),
        "val_nnz": int(X_val.nnz),
        "memory_footprint_mb": round(ram_mb, 2),
        "integrity_checks": {
            "train_nan_count": train_nan_count,
            "train_inf_count": train_inf_count,
            "val_nan_count": val_nan_count,
            "val_inf_count": val_inf_count,
            "status": "PASSED" if (train_nan_count + train_inf_count + val_nan_count + val_inf_count) == 0 else "FAILED",
        },
        "dense_scaler_mean": pipeline.dense_scaler.mean_.tolist() if pipeline.dense_scaler is not None else [],
        "dense_scaler_scale": pipeline.dense_scaler.scale_.tolist() if pipeline.dense_scaler is not None else [],
        "dense_feature_names": dense_feature_names,
        "feature_names": feature_names,
    }

    with open(out_path / "feature_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_dense_raw_train": X_dense_raw_train,
        "X_dense_raw_val": X_dense_raw_val,
        "y_bin_train": y_bin_train,
        "y_bin_val": y_bin_val,
        "y_model_train": y_model_train,
        "y_model_val": y_model_val,
        "meta_train": df_meta_train,
        "meta_val": df_meta_val,
        "metadata": metadata,
        "pipeline": pipeline,
    }


def load_cached_corpus(processed_dir) -> Dict[str, Any]:
    """Loads previously extracted and cached feature matrices and metadata from `processed_dir`."""
    import json
    from pathlib import Path
    import pandas as pd

    p_dir = Path(processed_dir)
    if not (p_dir / "X_train.npz").exists():
        raise FileNotFoundError(f"Cached features not found in {p_dir}. Run extraction first.")

    with open(p_dir / "feature_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return {
        "X_train": sp.load_npz(p_dir / "X_train.npz"),
        "X_val": sp.load_npz(p_dir / "X_val.npz"),
        "X_dense_raw_train": np.load(p_dir / "X_dense_raw_train.npy"),
        "X_dense_raw_val": np.load(p_dir / "X_dense_raw_val.npy"),
        "y_bin_train": np.load(p_dir / "y_bin_train.npy"),
        "y_bin_val": np.load(p_dir / "y_bin_val.npy"),
        "y_model_train": np.load(p_dir / "y_model_train.npy"),
        "y_model_val": np.load(p_dir / "y_model_val.npy"),
        "meta_train": pd.read_csv(p_dir / "meta_train.csv"),
        "meta_val": pd.read_csv(p_dir / "meta_val.csv"),
        "metadata": metadata,
    }


def slice_feature_family(
    X: sp.csr_matrix, metadata: Dict[str, Any], family: str
) -> Tuple[sp.csr_matrix, List[str]]:
    """
    Slices a combined CSR matrix to extract a specific feature family:
    'stylometry', 'function_words', 'char_ngrams', 'word_ngrams', 'dense_all', 'ngrams_all', or 'all'.
    """
    slices = metadata["family_slices"]
    if family not in slices:
        raise ValueError(f"Unknown family '{family}'. Valid options: {list(slices.keys())}")
    start, end = slices[family]
    feature_names = metadata["feature_names"][start:end]
    return X[:, start:end], feature_names

