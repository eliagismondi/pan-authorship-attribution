"""
main.py
-------
Main execution script for 'Punto 1: Data Understanding & Literature (Days 1-2)'.

Executes:
1. Literature review summary & task setup
2. Initial Inspection: N records, k classes (binary, candidate authors, genres), sample preview
3. Data Quality & Balance Audit: programmatic checks for corruption, missing fields, duplicates
4. Stylometric Exploratory Extraction: length variance, TTR, punctuation density, n-grams
5. Export of all presentation-ready charts to reports/figures/
"""

import sys
import time
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_utils import (
    find_default_train_path,
    find_default_val_path,
    inspect_dataset_structure,
    present_sample_records,
    evaluate_data_quality,
    extract_exploratory_features,
)
from src.visualization import (
    plot_binary_class_balance,
    plot_author_model_distribution,
    plot_genre_distribution,
    plot_text_length_distribution,
    plot_stylometric_distributions,
    plot_top_ngrams,
)


def run_punto_uno_pipeline():
    start_time = time.time()
    print("=" * 75)
    print("   PAN AUTHORSHIP ATTRIBUTION & GENERATIVE AI DETECTION")
    print("   STAGE 1: DATA UNDERSTANDING & EXPLORATORY ANALYSIS (DAYS 1-2)")
    print("=" * 75)

    figures_dir = ROOT_DIR / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Locate Datasets
    train_path = find_default_train_path()
    val_path = find_default_val_path()

    if not train_path or not train_path.exists():
        print(f"[ERROR] Training dataset not found in data/raw/*/")
        return

    print(f"\n[1] INITIAL INSPECTION (Points 1, 2, 8)")
    print(f" -> Train Dataset: {train_path.name}")
    train_stats = inspect_dataset_structure(train_path)
    N_train = train_stats["total_records_N"]
    k_labels = train_stats["k_labels"]
    k_models = train_stats["k_models_authors"]
    k_genres = train_stats["k_genres"]

    print(f"    * Total Training Records (N): {N_train:,}")
    print(f"    * Binary Classes (k): {k_labels} {train_stats['label_distribution']}")
    print(f"    * Candidate Authors / Generative Models (k): {k_models}")
    print(f"    * Genres / Domains (k): {k_genres} {train_stats['genre_distribution']}")

    if val_path and val_path.exists():
        val_stats = inspect_dataset_structure(val_path)
        N_val = val_stats["total_records_N"]
        print(f" -> Validation Dataset: {val_path.name}")
        print(f"    * Total Validation Records (N): {N_val:,}")
        print(f"    * Grand Total Records (Train + Val): {N_train + N_val:,}")

    # 2. Present Sample Rows (Point 1, 2, 8)
    print(f"\n[2] SAMPLE ROW PRESENTATION (Points 1, 2, 8)")
    samples = present_sample_records(train_path, n_samples=3)
    for s in samples:
        print(f" --- Sample #{s['sample_index']} ---")
        print(f"  ID: {s['id']}")
        print(f"  Author/Model: {s['author_model']} | Label: {s['label_name']} | Genre: {s['genre']}")
        print(f"  Length: {s['total_words']} words ({s['total_chars']} chars)")
        print(f"  Raw Text Preview: \"{s['text_preview']}\"\n")

    # 3. Data Quality & Integrity Audit (Points 3, 4, 12)
    print(f"[3] DATA QUALITY & INTEGRITY AUDIT (Points 3, 4, 12)")
    dq = evaluate_data_quality(train_path)
    print(f"  * Lines Processed: {dq['total_lines_read']:,} (Parsed: {dq['total_records_parsed']:,})")
    print(f"  * Corrupted Lines (JSON errors): {dq['corrupted_lines']}")
    print(f"  * Missing Fields: {dq['missing_fields']}")
    print(f"  * Empty Texts: {dq['empty_text_count']} | Non-String Texts: {dq['non_string_text_count']}")
    print(f"  * Duplicate Record IDs: {dq['duplicate_id_count']}")
    print(f"  * Duplicate Text Contents: {dq['duplicate_text_count']}")
    print(f"  * Word Count Extremes: Min={dq['min_word_count']}, Max={dq['max_word_count']}, Mean={dq['mean_word_count']}")
    print(f"  * Quality Status: {dq['data_integrity_status']}")

    # 4. Stylometric & Exploratory Feature Extraction (Points 10, 11)
    print(f"\n[4] STYLOMETRIC & EXPLORATORY FEATURE EXTRACTION (Points 10, 11)")
    print("  -> Computing Type-Token Ratio, Punctuation Density, and N-grams...")
    feature_results = extract_exploratory_features(train_path)
    summary = feature_results["summary_statistics"]

    print("\n  Summary Stylometric Comparison:")
    print("  " + "-" * 65)
    print(f"  {'Metric':<25} | {'Human (0)':<18} | {'AI Generated (1)':<18}")
    print("  " + "-" * 65)
    print(f"  {'Record Count (N)':<25} | {summary['Human']['record_count']:<18} | {summary['AI']['record_count']:<18}")
    print(f"  {'Mean Word Count':<25} | {summary['Human']['mean_word_count']:<18} | {summary['AI']['mean_word_count']:<18}")
    print(f"  {'Mean Char Count':<25} | {summary['Human']['mean_char_count']:<18} | {summary['AI']['mean_char_count']:<18}")
    print(f"  {'Type-Token Ratio (TTR)':<25} | {summary['Human']['mean_ttr']:<18} | {summary['AI']['mean_ttr']:<18}")
    print(f"  {'Punctuation Density':<25} | {summary['Human']['mean_punct_density']:<18} | {summary['AI']['mean_punct_density']:<18}")
    print(f"  {'Avg Word Length (chars)':<25} | {summary['Human']['mean_avg_word_len']:<18} | {summary['AI']['mean_avg_word_len']:<18}")
    print("  " + "-" * 65)

    # 5. Visualizations Generation (Points 4, 10, 12, 13)
    print(f"\n[5] GENERATING CHARTS & DELIVERABLES (reports/figures/)")

    p1 = plot_binary_class_balance(train_stats["label_distribution"], figures_dir / "class_balance_binary.png")
    print(f"  [+] Saved: {p1.name}")

    p2 = plot_author_model_distribution(train_stats["model_distribution"], figures_dir / "class_balance_authors.png")
    print(f"  [+] Saved: {p2.name}")

    p3 = plot_genre_distribution(train_stats["genre_distribution"], figures_dir / "class_balance_genres.png")
    print(f"  [+] Saved: {p3.name}")

    raw_dist = feature_results["raw_distributions"]
    p4 = plot_text_length_distribution(
        raw_dist["Human"]["word_counts"],
        raw_dist["AI"]["word_counts"],
        figures_dir / "text_length_distribution.png",
    )
    print(f"  [+] Saved: {p4.name}")

    p5 = plot_stylometric_distributions(raw_dist, figures_dir / "stylometric_distributions.png")
    print(f"  [+] Saved: {p5.name}")

    p6 = plot_top_ngrams(
        feature_results["top_unigrams"],
        feature_results["top_bigrams"],
        figures_dir / "top_ngrams_comparison.png",
        top_n=15,
    )
    print(f"  [+] Saved: {p6.name}")

    total_elapsed = time.time() - start_time
    print(f"\nPipeline successfully completed in {total_elapsed:.2f}s!")
    print(f"All figures exported to: {figures_dir}")
    print("=" * 75)


if __name__ == "__main__":
    run_punto_uno_pipeline()
