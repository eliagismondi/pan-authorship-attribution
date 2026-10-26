"""
main.py
-------
Execution pipeline for:
- Stage 1: Data Understanding & Exploratory Analysis (Points 1, 2, 3, 4, 8, 10, 11, 12)
- Stage 2: Baseline Feature Extraction into Vector Space (Points 6, 7)
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Optional, List, Dict, Tuple

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
    iter_jsonl,
)
from src.feature_extraction import (
    create_baseline_pipeline,
    StylometricFeatureExtractor,
    FunctionWordExtractor,
)


def run_punto_uno_pipeline():
    from src.visualization import (
        plot_binary_class_balance,
        plot_author_model_distribution,
        plot_genre_distribution,
        plot_text_length_distribution,
        plot_stylometric_distributions,
        plot_top_ngrams,
    )
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
    print(f"\nStage 1 successfully completed in {total_elapsed:.2f}s!")
    print(f"All figures exported to: {figures_dir}")
    print("=" * 75)


def run_baseline_extraction_pipeline(max_records: int = 1000):
    """
    Executes Baseline Extraction (Points 6, 7):
    Transforms raw texts into a numerical vector space based on features
    identified in the literature review.
    """
    start_time = time.time()
    print("=" * 75)
    print("   PAN AUTHORSHIP ATTRIBUTION & GENERATIVE AI DETECTION")
    print("   STAGE 2: BASELINE VECTOR SPACE EXTRACTION (POINTS 6, 7)")
    print("=" * 75)

    train_path = find_default_train_path()
    if not train_path or not train_path.exists():
        print(f"[ERROR] Training dataset not found in data/raw/*/")
        return

    print(f"\n[1] STREAMING RECORDS FROM CORPUS")
    texts = []
    labels = []
    models = []
    for i, rec in enumerate(iter_jsonl(train_path)):
        if max_records and i >= max_records:
            break
        texts.append(rec["text"])
        labels.append(rec["label"])
        models.append(rec.get("model", "unknown"))

    print(f"  -> Loaded {len(texts):,} records for numerical vectorization")

    # 2. Initialize and Fit Pipeline
    print(f"\n[2] INITIALIZING MULTI-LEVEL FEATURE EXTRACTION PIPELINE")
    print("  Feature Spaces Configured:")
    print("    * 1. Stylometric & Lexical Richness: TTR, Guiraud's R, Yule's K, Simpson's D,")
    print("         Hapax/Dislegomena ratios, Punctuation density, Sentence/Word lengths (24 dims)")
    print("    * 2. Topic-Invariant Function Words: Relative frequencies of 70 closed-class terms")
    print("    * 3. Character N-Grams: 3-to-4-grams boundary-aware TF-IDF (3,000 max dims)")
    print("    * 4. Word N-Grams: 1-to-2-grams lexical TF-IDF with function words (1,500 max dims)")

    pipeline = create_baseline_pipeline(
        char_max_features=3000,
        word_max_features=1500,
        include_word_ngrams=True,
    )

    t0 = time.time()
    X = pipeline.fit_transform(texts)
    fit_transform_time = time.time() - t0

    feature_names = pipeline.get_feature_names_out()
    total_dims = len(feature_names)
    sparsity = 100.0 * (1.0 - (X.nnz / (X.shape[0] * X.shape[1])))

    print(f"\n[3] NUMERICAL VECTOR SPACE SUMMARY (Points 6, 7)")
    print(f"  * Resulting Feature Matrix Shape: {X.shape[0]:,} rows x {X.shape[1]:,} dimensions")
    print(f"  * Matrix Representation: {type(X).__name__} (Compressed Sparse Row)")
    print(f"  * Sparsity: {sparsity:.2f}% (Dense values: {X.nnz:,})")
    print(f"  * Extraction Throughput: {len(texts) / fit_transform_time:.1f} texts/second")
    print(f"  * Elapsed Time: {fit_transform_time:.2f} seconds")

    # Breakdown by family
    sty_count = sum(1 for f in feature_names if f.startswith("stylometry__"))
    func_count = sum(1 for f in feature_names if f.startswith("func_word__"))
    char_count = sum(1 for f in feature_names if f.startswith("char_ngram__"))
    word_count = sum(1 for f in feature_names if f.startswith("word_ngram__"))

    print(f"\n  Breakdown by Feature Family:")
    print(f"    - Stylometric Dimensions: {sty_count}")
    print(f"    - Function Words Dimensions: {func_count}")
    print(f"    - Character N-Gram Dimensions: {char_count}")
    print(f"    - Word N-Gram Dimensions: {word_count}")
    print(f"    - Total Feature Dimensions: {total_dims}")

    print(f"\n[4] SAMPLE DIMENSION INSPECTION")
    print(f"  First 8 Stylometric Dimensions:")
    for fn in feature_names[:8]:
        print(f"    • {fn}")
    print(f"  Sample Function Word Dimensions:")
    for fn in [f for f in feature_names if f.startswith("func_word__")][:5]:
        print(f"    • {fn}")
    print(f"  Sample Character N-gram Dimensions:")
    for fn in [f for f in feature_names if f.startswith("char_ngram__")][:5]:
        print(f"    • {fn}")

    total_elapsed = time.time() - start_time
    print(f"\nStage 2 completed in {total_elapsed:.2f}s!")
    print("=" * 75)
    return X, pipeline


def run_correlation_matrix_pipeline(n_samples: Optional[int] = None):
    """
    Executes Correlation Matrix & Predictive Power Analysis (Points 9, 13):
    1. Derives custom features (word-to-sentence ratio, punctuation density, expressive ratios, etc.)
    2. Calculates empirical Pearson correlation with target output y (Human=0, AI=1) to demonstrate predictive power.
    3. Analyzes inter-feature collinearity (|r| >= 0.60) to detect variables measuring the same underlying construct.
    4. Generates publication-ready figures:
       - reports/figures/correlation_heatmap.png (Heatmap of collinearity and target correlation)
       - reports/figures/feature_predictive_power_ranking.png (Predictive power ranking regarding output y)
    """
    start_time = time.time()
    print("=" * 75)
    print("   PAN AUTHORSHIP ATTRIBUTION & GENERATIVE AI DETECTION")
    print("   CORRELATION MATRIX, COLLINEARITY & PREDICTIVE POWER (POINTS 9, 13)")
    print("=" * 75)

    import pandas as pd
    from src.visualization import (
        plot_correlation_heatmap,
        plot_feature_predictive_power,
    )

    figures_dir = ROOT_DIR / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    train_path = find_default_train_path()
    if not train_path or not train_path.exists():
        print(f"[ERROR] Training dataset not found in data/raw/*/")
        return

    print(f"\n[1] STREAMING RECORDS FOR CORRELATION MATRIX (POINTS 9, 13)")
    texts = []
    targets = []
    authors = []
    for i, rec in enumerate(iter_jsonl(train_path)):
        if n_samples and i >= n_samples:
            break
        texts.append(rec["text"])
        targets.append(rec["label"])
        authors.append(rec.get("model", "unknown"))

    print(f"  -> Extracted {len(texts):,} records for correlation analysis")

    # 2. Extract Stylometric & Advanced Derived Features
    print(f"\n[2] COMPUTING FULL STYLOMETRIC & DERIVED FEATURE SPACE")
    t0 = time.time()
    extractor = StylometricFeatureExtractor()
    X_dense = extractor.transform(texts)
    feature_names = extractor.FEATURE_NAMES
    print(f"  -> Computed {X_dense.shape[1]} features across {X_dense.shape[0]:,} samples in {time.time()-t0:.2f}s")

    df = pd.DataFrame(X_dense, columns=feature_names)
    df["target_is_ai"] = targets

    # Focus subset containing all newly derived custom features and primary stylometric anchors
    key_features = [
        # Points 9, 13 custom derived:
        "ratio_word_to_sentence_len",
        "punct_density",
        "syntactic_pause_ratio",
        "expressive_punct_ratio",
        "functional_lexical_ratio",
        "brunet_w",
        "honore_r",
        # Core stylometric anchors:
        "ttr",
        "yule_k",
        "avg_word_len",
        "avg_sentence_len",
        "comma_density",
        "dash_density",
        "quote_density",
        "digit_density",
        "target_is_ai",
    ]

    df_focus = df[key_features]
    corr_matrix = df_focus.corr(method="pearson")

    # 3. Target Correlation Ranking (Predictive Power regarding Output y)
    print(f"\n[3] PREDICTIVE POWER REGARDING TARGET OUTPUT y (Author Identity / AI Class)")
    print("  (Positive r = higher in AI; Negative r = higher in Human)")
    print("  " + "-" * 70)
    target_corr = corr_matrix["target_is_ai"].drop("target_is_ai").sort_values(ascending=False)
    for feat, r in target_corr.items():
        direction = "Predictive of AI (r > 0)" if r > 0 else "Predictive of Human (r < 0)"
        strength = "Strong" if abs(r) >= 0.40 else "Moderate" if abs(r) >= 0.20 else "Mild"
        print(f"    • {feat:<30} : r = {r:+.4f}  [{strength} - {direction}]")
    print("  " + "-" * 70)

    # 4. Collinearity Inspection (|r| >= 0.60: variables measuring the same metric)
    print(f"\n[4] INTER-FEATURE COLLINEARITY INSPECTION (|r| >= 0.60)")
    high_corr_pairs = []
    features_only = [f for f in key_features if f != "target_is_ai"]
    for i in range(len(features_only)):
        for j in range(i + 1, len(features_only)):
            f1, f2 = features_only[i], features_only[j]
            r_val = corr_matrix.loc[f1, f2]
            if abs(r_val) >= 0.60:
                high_corr_pairs.append((f1, f2, r_val))

    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    if high_corr_pairs:
        for f1, f2, r_val in high_corr_pairs:
            print(f"    ⚠️ Collinear Pair: {f1:<28} <-> {f2:<24} (r = {r_val:+.4f})")
    else:
        print("    No severe inter-feature collinearity detected among selected features.")

    # 5. Export Heatmap & Predictive Power Figures
    print(f"\n[5] EXPORTING CORRELATION DELIVERABLES (reports/figures/)")
    heatmap_path = figures_dir / "correlation_heatmap.png"
    plot_correlation_heatmap(
        corr_matrix,
        heatmap_path,
        title="Stylometric Feature Collinearity & Target Correlation Matrix (Points 9, 13)",
    )
    print(f"  [+] Saved: {heatmap_path.name}")

    ranking_path = figures_dir / "feature_predictive_power_ranking.png"
    plot_feature_predictive_power(
        target_corr,
        ranking_path,
        title="Feature Predictive Power Regarding Target Output y (Points 9, 13)",
    )
    print(f"  [+] Saved: {ranking_path.name}")

    total_elapsed = time.time() - start_time
    print(f"\nPoints 9 & 13 successfully completed in {total_elapsed:.2f}s!")
    print("=" * 75)
    return corr_matrix


# Backward-compatible alias
run_point_13_advanced_derivation = run_correlation_matrix_pipeline



def run_point_10_feature_distributions(n_samples: Optional[int] = None):
    """
    Executes Feature Distributions (Point 10):
    1. Generates histograms and KDE distribution plots showing key features (X) across the entire dataset.
    2. Generates comparative boxplots showing key features (X) broken down among different candidate authors (Human vs AI models).
    3. Exports publication-quality figures:
       - reports/figures/feature_distributions_entire_dataset.png
       - reports/figures/feature_distributions_by_author.png
    """
    start_time = time.time()
    print("=" * 75)
    print("   PAN AUTHORSHIP ATTRIBUTION & GENERATIVE AI DETECTION")
    print("   FEATURE DISTRIBUTIONS ACROSS DATASET & AUTHORS (POINT 10)")
    print("=" * 75)

    import pandas as pd
    from src.visualization import (
        plot_feature_distributions_entire_dataset,
        plot_feature_distributions_by_author,
    )

    figures_dir = ROOT_DIR / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    train_path = find_default_train_path()
    if not train_path or not train_path.exists():
        print(f"[ERROR] Training dataset not found in data/raw/*/")
        return

    print(f"\n[1] STREAMING RECORDS FOR POINT 10 FEATURE DISTRIBUTIONS")
    texts = []
    models = []
    labels = []
    genres = []
    for i, rec in enumerate(iter_jsonl(train_path)):
        if n_samples and i >= n_samples:
            break
        texts.append(rec.get("text", ""))
        models.append(rec.get("model", "unknown"))
        labels.append(rec.get("label", 0))
        genres.append(rec.get("genre", "unknown"))

    print(f"  -> Extracted {len(texts):,} texts for distribution analysis")

    print(f"\n[2] COMPUTING FULL STYLOMETRIC FEATURE MATRIX (X)")
    t0 = time.time()
    extractor = StylometricFeatureExtractor()
    X_dense = extractor.transform(texts)
    feature_names = extractor.FEATURE_NAMES
    elapsed_feat = time.time() - t0
    print(f"  -> Extracted {X_dense.shape[1]} features across {X_dense.shape[0]:,} samples in {elapsed_feat:.2f}s")

    df = pd.DataFrame(X_dense, columns=feature_names)
    df["model"] = models
    df["label"] = labels
    df["genre"] = genres

    key_features = [
        ("word_count", "Text Length (Word Count)"),
        ("ttr", "Type-Token Ratio (TTR)"),
        ("avg_word_len", "Average Word Length"),
        ("punct_density", "Punctuation Density"),
        ("avg_sentence_len", "Average Sentence Length"),
        ("functional_lexical_ratio", "Functional-to-Lexical Ratio"),
    ]

    print(f"\n[3] DESCRIPTIVE STATISTICS ACROSS THE ENTIRE DATASET (X)")
    print("  " + "-" * 75)
    print(f"  {'Feature':<28} | {'Mean':<8} | {'Std':<8} | {'Median':<8} | {'IQR':<8} | {'Skewness':<8}")
    print("  " + "-" * 75)
    for fkey, fname in key_features:
        s = df[fkey].dropna()
        q25, q75 = s.quantile(0.25), s.quantile(0.75)
        iqr = q75 - q25
        print(f"  {fname:<28} | {s.mean():<8.3f} | {s.std():<8.3f} | {s.median():<8.3f} | {iqr:<8.3f} | {s.skew():<8.3f}")
    print("  " + "-" * 75)

    print(f"\n[4] COMPARATIVE BREAKDOWN AMONG KEY AUTHORS (Human vs Top LLMs)")
    top_models = ["human"] + [m for m in df["model"].value_counts().index if m != "human"][:5]
    print("  " + "-" * 85)
    print(f"  {'Author / Model':<20} | {'TTR':<8} | {'AvgWordLen':<10} | {'PunctDens':<10} | {'FuncLexRatio':<12} | {'AvgSentLen':<10}")
    print("  " + "-" * 85)
    for m in top_models:
        sub = df[df["model"] == m]
        print(f"  {m:<20} | {sub['ttr'].mean():<8.3f} | {sub['avg_word_len'].mean():<10.3f} | {sub['punct_density'].mean():<10.4f} | {sub['functional_lexical_ratio'].mean():<12.3f} | {sub['avg_sentence_len'].mean():<10.2f}")
    print("  " + "-" * 85)

    print(f"\n[5] GENERATING POINT 10 DELIVERABLES (reports/figures/)")
    p1 = plot_feature_distributions_entire_dataset(
        df,
        figures_dir / "feature_distributions_entire_dataset.png",
        title="Key Feature Distributions Across Entire Dataset (Point 10)",
    )
    print(f"  [+] Saved: {p1.name}")

    p2 = plot_feature_distributions_by_author(
        df,
        figures_dir / "feature_distributions_by_author.png",
        top_k_authors=11,
        title="Key Feature Distributions Among Candidate Authors & Models (Point 10)",
    )
    print(f"  [+] Saved: {p2.name}")

    total_elapsed = time.time() - start_time
    print(f"\nPoint 10 successfully completed in {total_elapsed:.2f}s!")
    print(f"Figures exported to: {figures_dir}")
    print("=" * 75)
    return df


def main():
    parser = argparse.ArgumentParser(description="PAN Authorship Attribution & GenAI Pipeline")
    parser.add_argument("--stage", choices=["1", "2", "3", "9", "10", "13", "all"], default="all",
                        help="Pipeline stage: '1' (EDA), '2' (Baseline Vector Extraction), '3'/'9'/'13' (Correlation Matrix & Collinearity), '10' (Point 10 Feature Distributions), 'all'")
    parser.add_argument("--records", type=int, default=0,
                        help="Number of records to vectorize (default: 0 for full train set)")
    args = parser.parse_args()

    max_rec = None if args.records == 0 else args.records

    if args.stage in ["1", "all"]:
        run_punto_uno_pipeline()

    if args.stage in ["10", "all"]:
        run_point_10_feature_distributions(n_samples=max_rec)

    if args.stage in ["3", "9", "13", "all"]:
        run_correlation_matrix_pipeline(n_samples=max_rec)

    if args.stage in ["2", "all"]:
        run_baseline_extraction_pipeline(max_records=max_rec or 1000)


if __name__ == "__main__":
    main()
