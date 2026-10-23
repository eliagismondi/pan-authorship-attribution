import json
from json import JSONDecodeError
from pathlib import Path
from typing import Iterable, Tuple, Optional, Dict, Any, List
from collections import Counter

# Optional import; only needed for plotting
try:
    import matplotlib.pyplot as plt
except Exception:  # matplotlib may not be installed in some environments
    plt = None  # type: ignore


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """
    Stream records from a JSON Lines (.jsonl) file.
    Skips empty lines. Raises if a line is not valid JSON.
    """
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def dataset_stats(
    path: Path,
    label_field_candidates: Tuple[str, ...] = ("author", "label", "class", "y"),
) -> Tuple[int, int, Optional[str]]:
    """
    Compute dataset statistics from a JSONL file.

    Returns a tuple: (N, k, used_label_field)
    - N: total number of records
    - k: number of unique classes based on the first label field found among candidates
    - used_label_field: the name of the label field used (None if none found)
    """
    labels: List[Any] = []
    used_field: Optional[str] = None

    i = -1
    for i, rec in enumerate(iter_jsonl(path)):
        if used_field is None:
            # detect label field on the first record that has any of the candidates
            for f in label_field_candidates:
                if f in rec:
                    used_field = f
                    break
        if used_field is not None and used_field in rec:
            labels.append(rec[used_field])

    N = i + 1 if i >= 0 else 0
    if used_field is None:
        # No label field found; k is undefined → 0
        return N, 0, None

    # Compute unique classes; robustly handle hashable/non-hashable labels
    try:
        k = len(set(labels))
    except TypeError:
        # Fallback for unhashable labels (e.g., lists): convert to JSON strings
        k = len(set(json.dumps(l, sort_keys=True) for l in labels))

    return N, k, used_field


def evaluate_data_quality(
    path: Path,
    text_field_candidates: Tuple[str, ...] = ("text", "content", "body", "x", "document"),
) -> Dict[str, Any]:
    """
    Evaluate overall data quality focusing on text fields.

    Checks:
    - Corrupted JSON lines (cannot be parsed)
    - Missing text field
    - Empty / whitespace-only / None texts
    - Non-string text fields

    Returns a dictionary with counts and the chosen text field name.
    """
    path = Path(path)
    total_lines = 0
    parsed_records = 0
    corrupted_lines = 0
    missing_text = 0
    empty_text = 0
    non_string_text = 0

    used_text_field: Optional[str] = None

    with path.open('r', encoding='utf-8') as f:
        for raw in f:
            total_lines += 1
            line = raw.strip()
            if not line:
                # treat as parsed empty line; skip
                continue
            try:
                rec = json.loads(line)
                parsed_records += 1
            except JSONDecodeError:
                corrupted_lines += 1
                continue

            # determine text field on the first suitable record
            if used_text_field is None:
                for tf in text_field_candidates:
                    if tf in rec:
                        used_text_field = tf
                        break

            if used_text_field is None or used_text_field not in rec:
                missing_text += 1
                continue

            val = rec[used_text_field]
            if val is None:
                empty_text += 1
            elif isinstance(val, str):
                if val.strip() == "":
                    empty_text += 1
            else:
                non_string_text += 1

    return {
        "path": str(path),
        "total_lines": total_lines,
        "total_records_parsed": parsed_records,
        "corrupted_lines": corrupted_lines,
        "used_text_field": used_text_field,
        "missing_text_count": missing_text,
        "empty_text_count": empty_text,
        "non_string_text_count": non_string_text,
    }


def compute_label_counts(
    path: Path,
    label_field_candidates: Tuple[str, ...] = ("author", "label", "class", "y"),
) -> Tuple[Counter, Optional[str]]:
    """
    Compute counts per label (e.g., per author). Returns (Counter, used_label_field).
    Skips records without the chosen label field once it is determined.
    """
    counts: Counter = Counter()
    used_field: Optional[str] = None

    for rec in iter_jsonl(path):
        if used_field is None:
            for f in label_field_candidates:
                if f in rec:
                    used_field = f
                    break
        if used_field is None:
            continue
        if used_field in rec:
            counts[rec[used_field]] += 1

    return counts, used_field


def plot_class_balance(
    label_counts: Counter,
    title: str,
    output_path: Path,
    rotate_labels: bool = False,
) -> Optional[Path]:
    """
    Generate and save a bar chart for class distribution.

    Returns the path to the saved figure, or None if plotting is unavailable.
    """
    if plt is None:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels, values = zip(*label_counts.most_common()) if label_counts else ([], [])

    plt.figure(figsize=(max(6, min(16, 0.35 * max(5, len(labels)))), 6))
    bars = plt.bar(range(len(labels)), values, color="#4C78A8")
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(range(len(labels)), [str(l) for l in labels], rotation=45 if (rotate_labels or len(labels) > 10) else 0, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def find_default_train_path() -> Optional[Path]:
    """Try to locate a default training JSONL file under data/raw/*/train.jsonl."""
    root = Path(__file__).resolve().parents[1]  # project root
    raw = root / "data" / "raw"
    if not raw.exists():
        return None
    for sub in raw.iterdir():
        candidate = sub / "train.jsonl"
        if candidate.exists():
            return candidate
    return None


if __name__ == "__main__":
    train_path = find_default_train_path()
    if train_path is None:
        print("No default train.jsonl found under data/raw/*/")
    else:
        print(f"Dataset: {train_path}")

        # Basic stats
        N, k, field = dataset_stats(train_path)
        print(f"Total records (N): {N}")
        if field:
            print(f"Number of classes (k) based on '{field}': {k}")
        else:
            print("No label field found; k=0")

        # Data quality evaluation
        dq = evaluate_data_quality(train_path)
        print("Data quality checks:")
        print(f"  Parsed records: {dq['total_records_parsed']} of {dq['total_lines']} lines; corrupted lines: {dq['corrupted_lines']}")
        print(f"  Text field used: {dq['used_text_field']}")
        print(f"  Missing text field: {dq['missing_text_count']}")
        print(f"  Empty/None texts: {dq['empty_text_count']}")
        print(f"  Non-string texts: {dq['non_string_text_count']}")

        # Class balance plot
        counts, used_label = compute_label_counts(train_path)
        if counts and used_label is not None:
            fig_path = Path(__file__).resolve().parents[1] / "reports" / "figures" / "class_balance_train.png"
            saved = plot_class_balance(counts, title=f"Class balance by '{used_label}' (train)", output_path=fig_path)
            if saved is not None:
                print(f"Saved class balance figure to: {saved}")
            else:
                print("matplotlib not available; skipping class balance plot.")
        else:
            print("No label field found for class balance plot; skipping.")
