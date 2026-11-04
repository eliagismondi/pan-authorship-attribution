"""
feature_importance.py
---------------------
Stage 3: Measure Feature Influence for Output (Dual-Task: Binary Detection & Model Attribution).

Quantifies how individual features and feature families influence the target output y:
1. Univariate Statistical Metrics (Binary Human vs AI):
   - Pearson linear correlation (r) and Spearman rank correlation (rho)
   - ANOVA F-statistic (f_classif) and Chi-squared (chi2) for sparse N-grams
   - Mutual Information (Information Gain via mutual_info_classif)
2. Multiclass / Model Attribution Discriminative Power (22 AI Models + Human):
   - Multiclass One-Way ANOVA F-statistic (between-model vs within-model variance)
   - Architectural family stylometric fingerprinting (OpenAI, Gemini, LLaMA, Mistral, DeepSeek/Qwen, Falcon)
3. Multivariate Model-Based Feature Importance:
   - Regularized Logistic Regression coefficients (beta_j) and Odds Ratios (exp(beta_j))
   - Validation-set Permutation Importance (drop in Macro F1 when shuffling features)
4. Feature Family Ablation Study:
   - Individual family performance & Leave-One-Family-Out (LOFO) impact on both
     Binary Detection (Human vs AI) and Multiclass Model Attribution (23 classes).
"""

import json
from pathlib import Path
import time
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from sklearn.feature_selection import f_classif, chi2, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.svm import LinearSVC

from src.feature_extraction import slice_feature_family


MODEL_FAMILY_MAP = {
    "human": "Human",
    "gpt-3.5-turbo": "OpenAI (GPT/o3)",
    "gpt-4o": "OpenAI (GPT/o3)",
    "gpt-4o-mini": "OpenAI (GPT/o3)",
    "o3-mini": "OpenAI (GPT/o3)",
    "gpt-4.5-preview": "OpenAI (GPT/o3)",
    "gpt-4-turbo": "OpenAI (GPT/o3)",
    "gpt-4-turbo-paraphrase": "OpenAI (GPT/o3)",
    "gemini-2.0-flash": "Google (Gemini/PaLM)",
    "gemini-1.5-pro": "Google (Gemini/PaLM)",
    "gemini-pro": "Google (Gemini/PaLM)",
    "gemini-pro-paraphrase": "Google (Gemini/PaLM)",
    "text-bison-002": "Google (Gemini/PaLM)",
    "llama-3.1-8b-instruct": "Meta (LLaMA)",
    "llama-3.3-70b-instruct": "Meta (LLaMA)",
    "llama-2-70b-chat": "Meta (LLaMA)",
    "llama-2-7b-chat": "Meta (LLaMA)",
    "ministral-8b-instruct-2410": "Mistral AI",
    "mistral-7b-instruct-v0.2": "Mistral AI",
    "mixtral-8x7b-instruct-v0.1": "Mistral AI",
    "deepseek-r1-distill-qwen-32b": "DeepSeek / Qwen",
    "qwen1.5-72b-chat-8bit": "DeepSeek / Qwen",
    "falcon3-10b-instruct": "TII Falcon",
}


def compute_dense_univariate_influence(
    X_dense_raw: np.ndarray,
    y_bin: np.ndarray,
    dense_feature_names: List[str],
    mi_subsample: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Computes comprehensive univariate statistical influence metrics for all dense features
    (30 stylometric + 70 function words) against binary target y (0=Human, 1=AI).
    """
    n_samples, n_features = X_dense_raw.shape
    f_vals, f_pvals = f_classif(X_dense_raw, y_bin)

    # Subsample for KNN-based Mutual Information for computational efficiency
    rng = np.random.RandomState(random_state)
    if n_samples > mi_subsample:
        idx = rng.choice(n_samples, size=mi_subsample, replace=False)
        mi_scores = mutual_info_classif(
            X_dense_raw[idx], y_bin[idx], discrete_features=False, random_state=random_state
        )
    else:
        mi_scores = mutual_info_classif(
            X_dense_raw, y_bin, discrete_features=False, random_state=random_state
        )

    human_mask = y_bin == 0
    ai_mask = y_bin == 1

    records = []
    for j, fname in enumerate(dense_feature_names):
        col = X_dense_raw[:, j]
        r_val, r_pval = stats.pearsonr(col, y_bin)
        rho_val, rho_pval = stats.spearmanr(col, y_bin)

        mean_h = float(np.mean(col[human_mask]))
        mean_ai = float(np.mean(col[ai_mask]))
        std_pooled = float(np.std(col)) + 1e-12
        cohen_d = (mean_ai - mean_h) / std_pooled

        family = "Stylometry" if fname.startswith("stylometry__") else "Function Words"
        short_name = fname.split("__", 1)[-1]

        records.append({
            "feature": fname,
            "short_name": short_name,
            "family": family,
            "pearson_r": float(r_val),
            "abs_pearson_r": float(abs(r_val)),
            "pearson_pval": float(r_pval),
            "spearman_rho": float(rho_val),
            "abs_spearman_rho": float(abs(rho_val)),
            "anova_f": float(f_vals[j]),
            "anova_pval": float(f_pvals[j]),
            "mutual_info": float(mi_scores[j]),
            "mean_human": mean_h,
            "mean_ai": mean_ai,
            "cohen_d": float(cohen_d),
            "direction": "AI (+)" if r_val > 0 else "Human (-)",
        })

    df_res = pd.DataFrame(records)
    df_res.sort_values(by="abs_pearson_r", ascending=False, inplace=True)
    df_res.reset_index(drop=True, inplace=True)
    return df_res


def compute_sparse_ngram_influence(
    X_train: sp.csr_matrix,
    y_bin: np.ndarray,
    metadata: Dict[str, Any],
    top_k: int = 30,
) -> pd.DataFrame:
    """
    Ranks sparse TF-IDF character and word n-grams by ANOVA F-score and Chi-squared statistic.
    """
    X_ngrams, ngram_names = slice_feature_family(X_train, metadata, "ngrams_all")
    f_vals, _ = f_classif(X_ngrams, y_bin)
    chi2_vals, _ = chi2(X_ngrams, y_bin)

    human_mask = y_bin == 0
    ai_mask = y_bin == 1
    mean_h = np.asarray(X_ngrams[human_mask].mean(axis=0)).ravel()
    mean_ai = np.asarray(X_ngrams[ai_mask].mean(axis=0)).ravel()
    diff = mean_ai - mean_h

    df_ng = pd.DataFrame({
        "feature": ngram_names,
        "short_name": [n.split("__", 1)[-1] for n in ngram_names],
        "family": ["Char N-Gram" if n.startswith("char_ngram__") else "Word N-Gram" for n in ngram_names],
        "anova_f": f_vals,
        "chi2": chi2_vals,
        "mean_tfidf_human": mean_h,
        "mean_tfidf_ai": mean_ai,
        "tfidf_diff_ai_minus_human": diff,
        "direction": ["AI (+)" if d > 0 else "Human (-)" for d in diff],
    })

    df_ng.sort_values(by="anova_f", ascending=False, inplace=True)
    df_ng.reset_index(drop=True, inplace=True)
    return df_ng.head(top_k * 2)


def compute_multiclass_model_attribution_influence(
    X_dense_raw: np.ndarray,
    y_model: np.ndarray,
    meta_df: pd.DataFrame,
    dense_feature_names: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Measures feature influence for distinguishing between the 23 candidate classes
    (Human + 22 AI Models) and across the 22 AI models specifically (LLM Fingerprinting).
    Also computes standardized feature signatures across LLM families.
    """
    # 1. Multiclass ANOVA across all 23 classes (Human + 22 AI)
    f_all23, p_all23 = f_classif(X_dense_raw, y_model)

    # 2. Multiclass ANOVA strictly among the 22 AI models (excluding Human, y_model > 0)
    ai_mask = y_model > 0
    f_ai22, p_ai22 = f_classif(X_dense_raw[ai_mask], y_model[ai_mask])

    records = []
    for j, fname in enumerate(dense_feature_names):
        family = "Stylometry" if fname.startswith("stylometry__") else "Function Words"
        short_name = fname.split("__", 1)[-1]
        records.append({
            "feature": fname,
            "short_name": short_name,
            "family": family,
            "anova_f_23classes": float(f_all23[j]),
            "anova_p_23classes": float(p_all23[j]),
            "anova_f_22ai_models": float(f_ai22[j]),
            "anova_p_22ai_models": float(p_ai22[j]),
        })

    df_multiclass = pd.DataFrame(records).sort_values(
        by="anova_f_22ai_models", ascending=False
    ).reset_index(drop=True)

    # 3. Compute standardized z-score profiles per architectural family for Stylometric features
    sty_mask = [i for i, f in enumerate(dense_feature_names) if f.startswith("stylometry__")]
    sty_names = [dense_feature_names[i].split("__", 1)[-1] for i in sty_mask]
    X_sty = X_dense_raw[:, sty_mask]

    # Z-score normalize across corpus for visual comparability
    sty_mean = np.mean(X_sty, axis=0)
    sty_std = np.std(X_sty, axis=0) + 1e-12
    X_sty_z = (X_sty - sty_mean) / sty_std

    df_sty_z = pd.DataFrame(X_sty_z, columns=sty_names)
    df_sty_z["arch_family"] = meta_df["model"].map(lambda m: MODEL_FAMILY_MAP.get(m, "Other AI"))

    family_order = [
        "Human",
        "OpenAI (GPT/o3)",
        "Google (Gemini/PaLM)",
        "Meta (LLaMA)",
        "Mistral AI",
        "DeepSeek / Qwen",
        "TII Falcon",
    ]
    existing_families = [f for f in family_order if f in df_sty_z["arch_family"].values]
    df_family_profiles = df_sty_z.groupby("arch_family")[sty_names].mean().loc[existing_families]

    return df_multiclass, df_family_profiles


def compute_model_based_influence(
    X_train: sp.csr_matrix,
    X_val: sp.csr_matrix,
    y_bin_train: np.ndarray,
    y_bin_val: np.ndarray,
    metadata: Dict[str, Any],
    random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Computes multivariate model-based feature importance:
    1. Dense features (100 dims): L2 Logistic Regression coefficients (beta), Odds Ratios,
       and Validation Set Permutation Importance (delta Macro F1).
    2. Global feature space (4,600 dims): L2 Logistic Regression top positive & negative weights.
    """
    # Part A: Dense features (Stylometry + Function Words)
    X_dense_tr, dense_names = slice_feature_family(X_train, metadata, "dense_all")
    X_dense_va, _ = slice_feature_family(X_val, metadata, "dense_all")

    X_dense_tr_arr = X_dense_tr.toarray()
    X_dense_va_arr = X_dense_va.toarray()

    clf_dense = LogisticRegression(C=1.0, max_iter=1000, random_state=random_state)
    clf_dense.fit(X_dense_tr_arr, y_bin_train)

    betas_dense = clf_dense.coef_.ravel()
    odds_ratios = np.exp(np.clip(betas_dense, -10, 10))

    # Permutation importance on Validation set
    perm = permutation_importance(
        clf_dense,
        X_dense_va_arr,
        y_bin_val,
        scoring="f1_macro",
        n_repeats=5,
        random_state=random_state,
        n_jobs=1,
    )

    df_dense_model = pd.DataFrame({
        "feature": dense_names,
        "short_name": [n.split("__", 1)[-1] for n in dense_names],
        "family": ["Stylometry" if n.startswith("stylometry__") else "Function Words" for n in dense_names],
        "beta_weight": betas_dense,
        "abs_beta": np.abs(betas_dense),
        "odds_ratio": odds_ratios,
        "perm_importance_f1_drop": perm.importances_mean,
        "perm_importance_std": perm.importances_std,
        "direction": ["AI (+)" if b > 0 else "Human (-)" for b in betas_dense],
    }).sort_values(by="abs_beta", ascending=False).reset_index(drop=True)

    # Part B: Full 4,600-dim feature space
    clf_full = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=random_state)
    clf_full.fit(X_train, y_bin_train)
    betas_full = clf_full.coef_.ravel()
    all_names = metadata["feature_names"]

    families_full = []
    for n in all_names:
        if n.startswith("stylometry__"):
            families_full.append("Stylometry")
        elif n.startswith("func_word__"):
            families_full.append("Function Words")
        elif n.startswith("char_ngram__"):
            families_full.append("Char N-Gram")
        else:
            families_full.append("Word N-Gram")

    df_full_weights = pd.DataFrame({
        "feature": all_names,
        "short_name": [n.split("__", 1)[-1] for n in all_names],
        "family": families_full,
        "beta_weight": betas_full,
        "abs_beta": np.abs(betas_full),
        "direction": ["AI (+)" if b > 0 else "Human (-)" for b in betas_full],
    }).sort_values(by="abs_beta", ascending=False).reset_index(drop=True)

    return {
        "dense_model_importance": df_dense_model,
        "global_model_weights": df_full_weights,
    }


def run_feature_family_ablation_study(
    X_train: sp.csr_matrix,
    X_val: sp.csr_matrix,
    y_bin_train: np.ndarray,
    y_bin_val: np.ndarray,
    y_model_train: np.ndarray,
    y_model_val: np.ndarray,
    metadata: Dict[str, Any],
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Evaluates individual feature families, combinations, and Leave-One-Family-Out (LOFO)
    configurations on both:
    - Task 1: Binary Detection (Human vs AI) -> Macro F1, Accuracy, ROC-AUC
    - Task 2: Multiclass Model Attribution (23 classes) -> Macro F1, Accuracy
    """
    slices = metadata["family_slices"]

    def _get_cols(families_list: List[str]) -> sp.csr_matrix:
        blocks_tr, blocks_va = [], []
        for fam in families_list:
            s, e = slices[fam]
            blocks_tr.append(X_train[:, s:e])
            blocks_va.append(X_val[:, s:e])
        return sp.hstack(blocks_tr, format="csr"), sp.hstack(blocks_va, format="csr")

    configs = [
        ("1. Stylometry Only", ["stylometry"], "Single Family"),
        ("2. Function Words Only", ["function_words"], "Single Family"),
        ("3. Dense Combined (Sty + Func)", ["stylometry", "function_words"], "Sub-Group"),
        ("4. Word N-Grams Only", ["word_ngrams"], "Single Family"),
        ("5. Char N-Grams Only", ["char_ngrams"], "Single Family"),
        ("6. N-Grams Combined (Char + Word)", ["char_ngrams", "word_ngrams"], "Sub-Group"),
        ("7. Full Pipeline (All Families)", ["stylometry", "function_words", "char_ngrams", "word_ngrams"], "Full System"),
        ("LOFO: w/o Stylometry", ["function_words", "char_ngrams", "word_ngrams"], "Leave-One-Out"),
        ("LOFO: w/o Function Words", ["stylometry", "char_ngrams", "word_ngrams"], "Leave-One-Out"),
        ("LOFO: w/o Word N-Grams", ["stylometry", "function_words", "char_ngrams"], "Leave-One-Out"),
        ("LOFO: w/o Char N-Grams", ["stylometry", "function_words", "word_ngrams"], "Leave-One-Out"),
    ]

    results = []
    for label_name, fam_list, group_type in configs:
        t0 = time.time()
        X_tr_sub, X_va_sub = _get_cols(fam_list)
        n_dims = X_tr_sub.shape[1]

        # Binary Classifier (Logistic Regression L2)
        clf_bin = LogisticRegression(C=1.0, max_iter=600, solver="lbfgs", random_state=random_state)
        clf_bin.fit(X_tr_sub, y_bin_train)
        y_pred_bin = clf_bin.predict(X_va_sub)
        y_prob_bin = clf_bin.predict_proba(X_va_sub)[:, 1]

        bin_f1 = f1_score(y_bin_val, y_pred_bin, average="macro")
        bin_acc = accuracy_score(y_bin_val, y_pred_bin)
        bin_auc = roc_auc_score(y_bin_val, y_prob_bin)

        # Multiclass Model Attribution Classifier (LinearSVC is fast & accurate on 23 classes)
        clf_multi = LinearSVC(C=0.5, max_iter=1500, random_state=random_state)
        clf_multi.fit(X_tr_sub, y_model_train)
        y_pred_multi = clf_multi.predict(X_va_sub)

        multi_f1 = f1_score(y_model_val, y_pred_multi, average="macro")
        multi_acc = accuracy_score(y_model_val, y_pred_multi)

        # Strictly on AI samples (22 AI models attribution accuracy)
        ai_mask_val = y_bin_val == 1
        ai_only_acc = accuracy_score(y_model_val[ai_mask_val], y_pred_multi[ai_mask_val])

        elapsed = time.time() - t0
        results.append({
            "config_name": label_name,
            "group_type": group_type,
            "n_features": int(n_dims),
            "binary_macro_f1": round(float(bin_f1), 4),
            "binary_accuracy": round(float(bin_acc), 4),
            "binary_roc_auc": round(float(bin_auc), 4),
            "multiclass23_macro_f1": round(float(multi_f1), 4),
            "multiclass23_accuracy": round(float(multi_acc), 4),
            "ai22_attribution_acc": round(float(ai_only_acc), 4),
            "elapsed_sec": round(elapsed, 2),
        })

    df_ablation = pd.DataFrame(results)

    # Calculate delta vs Full Pipeline
    full_row = df_ablation[df_ablation["config_name"] == "7. Full Pipeline (All Families)"].iloc[0]
    df_ablation["delta_binary_f1"] = (df_ablation["binary_macro_f1"] - full_row["binary_macro_f1"]).round(4)
    df_ablation["delta_multi_f1"] = (df_ablation["multiclass23_macro_f1"] - full_row["multiclass23_macro_f1"]).round(4)

    return df_ablation


def execute_full_feature_influence_analysis(
    cached_data: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Orchestrates the complete Stage 3 Feature Influence analysis and exports tables to CSV/JSON.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train = cached_data["X_train"]
    X_val = cached_data["X_val"]
    X_dense_raw_train = cached_data["X_dense_raw_train"]
    y_bin_train = cached_data["y_bin_train"]
    y_bin_val = cached_data["y_bin_val"]
    y_model_train = cached_data["y_model_train"]
    y_model_val = cached_data["y_model_val"]
    meta_train = cached_data["meta_train"]
    metadata = cached_data["metadata"]
    dense_names = metadata["dense_feature_names"]

    # 1. Dense Univariate Influence
    df_dense_uni = compute_dense_univariate_influence(
        X_dense_raw_train, y_bin_train, dense_names
    )
    df_dense_uni.to_csv(output_dir / "influence_dense_univariate.csv", index=False)

    # 2. Sparse N-gram Univariate Influence
    df_ngram_uni = compute_sparse_ngram_influence(
        X_train, y_bin_train, metadata, top_k=30
    )
    df_ngram_uni.to_csv(output_dir / "influence_sparse_ngrams_top60.csv", index=False)

    # 3. Multiclass Model Attribution Influence & LLM Family Signatures
    df_multiclass_anova, df_family_profiles = compute_multiclass_model_attribution_influence(
        X_dense_raw_train, y_model_train, meta_train, dense_names
    )
    df_multiclass_anova.to_csv(output_dir / "influence_multiclass_anova.csv", index=False)
    df_family_profiles.to_csv(output_dir / "llm_family_stylometric_profiles.csv")

    # 4. Multivariate Model-Based Influence (Logistic Regression Weights + Permutation)
    model_inf = compute_model_based_influence(
        X_train, X_val, y_bin_train, y_bin_val, metadata
    )
    df_dense_model = model_inf["dense_model_importance"]
    df_global_weights = model_inf["global_model_weights"]
    df_dense_model.to_csv(output_dir / "influence_dense_logistic_permutation.csv", index=False)
    df_global_weights.head(100).to_csv(output_dir / "influence_global_top100_weights.csv", index=False)

    # 5. Feature Family Ablation Study
    df_ablation = run_feature_family_ablation_study(
        X_train, X_val, y_bin_train, y_bin_val, y_model_train, y_model_val, metadata
    )
    df_ablation.to_csv(output_dir / "feature_family_ablation_results.csv", index=False)

    return {
        "dense_univariate": df_dense_uni,
        "sparse_ngrams": df_ngram_uni,
        "multiclass_anova": df_multiclass_anova,
        "family_profiles": df_family_profiles,
        "dense_model_importance": df_dense_model,
        "global_weights": df_global_weights,
        "ablation": df_ablation,
    }
