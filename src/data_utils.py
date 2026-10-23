"""
data_utils.py
-------------
Core utilities for loading, inspecting, evaluating data quality, and extracting
stylometric features for PAN authorship attribution and generative AI detection.

Designed to fulfill:
- Point 1, 2, 8: Initial inspection (N, k, sample rows inspection)
- Point 3, 4, 12: Data quality & integrity checks, missing value audit, class balance
- Point 10, 11: Stylometric feature extraction foundations (TTR, punctuation, n-grams)
"""

import json
from json import JSONDecodeError
from pathlib import Path
import re
from typing import Iterable, Tuple, Optional, Dict, Any, List, Union
from collections import Counter

# Standard common stopwords for lexical/n-gram filtering
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd",
    "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


def iter_jsonl(path: Union[str, Path]) -> Iterable[Dict[str, Any]]:
    """
    Stream records from a JSON Lines (.jsonl) file.
    Skips empty lines. Raises JSONDecodeError if a line is corrupted.
    """
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def dataset_stats(
    path: Union[str, Path],
    label_field_candidates: Tuple[str, ...] = ("label", "class", "y"),
    author_field_candidates: Tuple[str, ...] = ("model", "author", "author_id"),
) -> Tuple[int, int, Optional[str]]:
    """
    Compute dataset statistics from a JSONL file in a single streaming pass.

    Returns a tuple: (N, k, used_label_field)
    - N: total number of records
    - k: number of unique classes based on the first label field found
    - used_label_field: the name of the label field used
    """
    path = Path(path)
    labels = set()
    used_field: Optional[str] = None
    count = 0

    for rec in iter_jsonl(path):
        count += 1
        if used_field is None:
            for f in label_field_candidates:
                if f in rec:
                    used_field = f
                    break
        if used_field and used_field in rec:
            val = rec[used_field]
            labels.add(val if isinstance(val, (int, str, float, bool)) else json.dumps(val))

    N = count
    k = len(labels)
    return N, k, used_field


def inspect_dataset_structure(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Thoroughly inspect the dataset structure, returning counts for:
    - N (total records)
    - Binary classes k_binary (e.g. Human vs AI, 'label')
    - Candidate authors / generative models k_authors ('model')
    - Genre domains k_genres ('genre')
    - Counters for all targets
    """
    path = Path(path)
    total_records = 0
    label_counts: Counter = Counter()
    model_counts: Counter = Counter()
    genre_counts: Counter = Counter()
    keys_found = set()

    for rec in iter_jsonl(path):
        total_records += 1
        if total_records == 1:
            keys_found = set(rec.keys())

        if "label" in rec:
            label_counts[rec["label"]] += 1
        if "model" in rec:
            model_counts[rec["model"]] += 1
        if "genre" in rec:
            genre_counts[rec["genre"]] += 1

    return {
        "path": str(path),
        "total_records_N": total_records,
        "k_labels": len(label_counts),
        "k_models_authors": len(model_counts),
        "k_genres": len(genre_counts),
        "keys": sorted(list(keys_found)),
        "label_distribution": dict(label_counts),
        "model_distribution": dict(model_counts.most_common()),
        "genre_distribution": dict(genre_counts.most_common()),
    }


def present_sample_records(
    path: Union[str, Path],
    n_samples: int = 3,
    max_text_chars: int = 240,
) -> List[Dict[str, Any]]:
    """
    Extract a clean, formatted sample of records showing author/model ID,
    label, genre, and text preview (Points 1, 2, 8).
    """
    path = Path(path)
    samples: List[Dict[str, Any]] = []

    for i, rec in enumerate(iter_jsonl(path)):
        if i >= n_samples:
            break
        text = rec.get("text", "")
        preview = text[:max_text_chars] + ("..." if len(text) > max_text_chars else "")
        samples.append({
            "sample_index": i + 1,
            "id": rec.get("id"),
            "author_model": rec.get("model", "unknown"),
            "label": rec.get("label"),
            "label_name": "Human (0)" if rec.get("label") == 0 else "AI-Generated (1)",
            "genre": rec.get("genre", "unknown"),
            "total_chars": len(text),
            "total_words": len(text.split()),
            "text_preview": preview,
        })
    return samples


def evaluate_data_quality(
    path: Union[str, Path],
    text_field: str = "text",
    expected_fields: Tuple[str, ...] = ("id", "text", "model", "label", "genre"),
) -> Dict[str, Any]:
    """
    Evaluate overall data quality in a single streaming pass (Points 3, 4, 12).

    Checks:
    - Corrupted JSON lines (syntax decode errors)
    - Missing required fields
    - Empty, whitespace-only, or non-string texts
    - Duplicate record IDs
    - Duplicate text bodies
    - Extreme length outliers (e.g. texts < 10 words)
    """
    path = Path(path)
    total_lines = 0
    parsed_records = 0
    corrupted_lines = 0

    missing_fields_counts: Dict[str, int] = {f: 0 for f in expected_fields}
    empty_text_count = 0
    non_string_text_count = 0
    short_text_count = 0  # < 10 words

    seen_ids = set()
    duplicate_id_count = 0

    seen_texts = set()
    duplicate_text_count = 0

    word_counts: List[int] = []

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            total_lines += 1
            line = raw.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
                parsed_records += 1
            except JSONDecodeError:
                corrupted_lines += 1
                continue

            # Check expected schema fields
            for ef in expected_fields:
                if ef not in rec or rec[ef] is None:
                    missing_fields_counts[ef] += 1

            # Check duplicate ID
            rec_id = rec.get("id")
            if rec_id:
                if rec_id in seen_ids:
                    duplicate_id_count += 1
                else:
                    seen_ids.add(rec_id)

            # Check text validity
            text_val = rec.get(text_field)
            if text_val is None:
                empty_text_count += 1
            elif not isinstance(text_val, str):
                non_string_text_count += 1
            else:
                stripped = text_val.strip()
                if not stripped:
                    empty_text_count += 1
                else:
                    words = stripped.split()
                    w_len = len(words)
                    word_counts.append(w_len)
                    if w_len < 10:
                        short_text_count += 1

                    # Hash text to check duplicate contents efficiently
                    thash = hash(stripped)
                    if thash in seen_texts:
                        duplicate_text_count += 1
                    else:
                        seen_texts.add(thash)

    min_words = min(word_counts) if word_counts else 0
    max_words = max(word_counts) if word_counts else 0
    mean_words = sum(word_counts) / len(word_counts) if word_counts else 0

    return {
        "path": str(path),
        "total_lines_read": total_lines,
        "total_records_parsed": parsed_records,
        "corrupted_lines": corrupted_lines,
        "missing_fields": missing_fields_counts,
        "empty_text_count": empty_text_count,
        "non_string_text_count": non_string_text_count,
        "duplicate_id_count": duplicate_id_count,
        "duplicate_text_count": duplicate_text_count,
        "short_text_outliers": short_text_count,
        "min_word_count": min_words,
        "max_word_count": max_words,
        "mean_word_count": round(mean_words, 2),
        "data_integrity_status": "EXCELLENT" if (corrupted_lines == 0 and empty_text_count == 0 and duplicate_id_count == 0) else "NEEDS_CLEANING",
    }


def extract_exploratory_features(
    path: Union[str, Path],
    max_records: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Extract key exploratory stylometric features and n-grams across classes (Point 10, 11).

    Computes:
    - Text lengths (words and characters) separated by Human vs AI
    - Type-Token Ratio (TTR: lexical richness)
    - Punctuation density (punctuation marks / total chars)
    - Average word length
    - Function word / stopword ratio
    - Most frequent unigrams and bigrams per class (top 20)
    """
    path = Path(path)
    word_pattern = re.compile(r"\b[a-zA-Z]+\b")
    punct_pattern = re.compile(r"[^\w\s]")

    data: Dict[str, Dict[str, List[float]]] = {
        "Human": {"word_counts": [], "char_counts": [], "ttr": [], "punct_density": [], "avg_word_len": []},
        "AI": {"word_counts": [], "char_counts": [], "ttr": [], "punct_density": [], "avg_word_len": []},
    }

    unigrams: Dict[str, Counter] = {"Human": Counter(), "AI": Counter()}
    bigrams: Dict[str, Counter] = {"Human": Counter(), "AI": Counter()}

    count = 0
    for rec in iter_jsonl(path):
        count += 1
        if max_records and count > max_records:
            break

        text = rec.get("text", "")
        if not text:
            continue

        label_key = "Human" if rec.get("label") == 0 else "AI"
        chars = len(text)
        words = word_pattern.findall(text.lower())
        n_words = len(words)
        if n_words == 0:
            continue

        # Stylometric metrics
        ttr = len(set(words)) / n_words
        puncts = len(punct_pattern.findall(text))
        punct_density = puncts / max(1, chars)
        avg_word_len = sum(len(w) for w in words) / n_words

        target = data[label_key]
        target["word_counts"].append(n_words)
        target["char_counts"].append(chars)
        target["ttr"].append(ttr)
        target["punct_density"].append(punct_density)
        target["avg_word_len"].append(avg_word_len)

        # N-grams (filter pure stopwords for unigrams to show content/stylistic words, or keep function words)
        unigrams[label_key].update([w for w in words if w not in STOPWORDS])
        bigram_pairs = [f"{w1} {w2}" for w1, w2 in zip(words[:-1], words[1:])]
        bigrams[label_key].update(bigram_pairs)

    summary_stats = {}
    for group in ["Human", "AI"]:
        group_data = data[group]
        n_grp = len(group_data["word_counts"])
        summary_stats[group] = {
            "record_count": n_grp,
            "mean_word_count": round(sum(group_data["word_counts"]) / max(1, n_grp), 2),
            "mean_char_count": round(sum(group_data["char_counts"]) / max(1, n_grp), 2),
            "mean_ttr": round(sum(group_data["ttr"]) / max(1, n_grp), 4),
            "mean_punct_density": round(sum(group_data["punct_density"]) / max(1, n_grp), 4),
            "mean_avg_word_len": round(sum(group_data["avg_word_len"]) / max(1, n_grp), 3),
        }

    return {
        "raw_distributions": data,
        "summary_statistics": summary_stats,
        "top_unigrams": {
            "Human": unigrams["Human"].most_common(20),
            "AI": unigrams["AI"].most_common(20),
        },
        "top_bigrams": {
            "Human": bigrams["Human"].most_common(20),
            "AI": bigrams["AI"].most_common(20),
        },
    }


def find_default_train_path() -> Optional[Path]:
    """Locate the training JSONL file under data/raw/*/train.jsonl."""
    root = Path(__file__).resolve().parents[1]
    raw = root / "data" / "raw"
    if not raw.exists():
        return None
    for sub in raw.iterdir():
        if sub.is_dir():
            candidate = sub / "train.jsonl"
            if candidate.exists():
                return candidate
    return None


def find_default_val_path() -> Optional[Path]:
    """Locate the validation JSONL file under data/raw/*/val.jsonl."""
    root = Path(__file__).resolve().parents[1]
    raw = root / "data" / "raw"
    if not raw.exists():
        return None
    for sub in raw.iterdir():
        if sub.is_dir():
            candidate = sub / "val.jsonl"
            if candidate.exists():
                return candidate
    return None


if __name__ == "__main__":
    train_path = find_default_train_path()
    if not train_path:
        print("Default train.jsonl not found.")
    else:
        print(f"=== INITIAL INSPECTION (Points 1, 2, 8) ===")
        stats = inspect_dataset_structure(train_path)
        print(f"Dataset path: {stats['path']}")
        print(f"Total records (N): {stats['total_records_N']:,}")
        print(f"Binary classes (k): {stats['k_labels']} -> {stats['label_distribution']}")
        print(f"Candidate authors / models (k): {stats['k_models_authors']}")
        print(f"Genres (k): {stats['k_genres']} -> {stats['genre_distribution']}")

        print(f"\n=== SAMPLE ROWS (Points 1, 2, 8) ===")
        samples = present_sample_records(train_path, n_samples=2)
        for s in samples:
            print(f"[{s['sample_index']}] ID: {s['id']} | Author/Model: {s['author_model']} | Label: {s['label_name']} | Genre: {s['genre']}")
            print(f"    Words: {s['total_words']} | Chars: {s['total_chars']}")
            print(f"    Preview: \"{s['text_preview']}\"")

        print(f"\n=== DATA QUALITY & INTEGRITY (Points 3, 4, 12) ===")
        dq = evaluate_data_quality(train_path)
        print(f"Parsed records: {dq['total_records_parsed']} / {dq['total_lines_read']}")
        print(f"Corrupted lines: {dq['corrupted_lines']}")
        print(f"Empty texts: {dq['empty_text_count']}, Non-string: {dq['non_string_text_count']}")
        print(f"Duplicate IDs: {dq['duplicate_id_count']}, Duplicate texts: {dq['duplicate_text_count']}")
        print(f"Word length range: [{dq['min_word_count']} - {dq['max_word_count']}], Mean: {dq['mean_word_count']}")
        print(f"Integrity Status: {dq['data_integrity_status']}")
