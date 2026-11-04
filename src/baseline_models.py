"""
baseline_models.py
------------------
Stage 4: Run the Baseline Models (Dual-Task: Binary GenAI Detection & Multiclass Model Attribution).

Implements, trains, cross-validates, and benchmarks the canonical baseline classifiers:
1. Dummy Baseline (Majority Class Reference)
2. Complement Naive Bayes (Probabilistic Text Baseline on non-negative feature space)
3. Logistic Regression (L2 Ridge & L1 Lasso Regularized Linear Baselines)
4. Linear Support Vector Machine (LinearSVC - Stamatatos Authorship Attribution Standard)

Evaluates two tasks:
- Task 1 (Primary): Binary Classification (0 = Human, 1 = AI-Generated)
- Task 2 (Extended): Authorship / LLM Model Attribution (23 classes: Human + 22 LLMs),
  comparing Direct 23-Class Multiclass vs Two-Stage Hierarchical Pipeline
  (Stage 1: Binary Human vs AI -> Stage 2: 22-LLM Specialist Classifier).

Includes full error analysis:
- Cross-Genre Generalization (Fiction, Essays, News)
- Per-LLM Detection Evasion Rate & Exact/Family Attribution Accuracy.
"""

from pathlib import Path
import time
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.special import softmax
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    top_k_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC

from src.feature_importance import MODEL_FAMILY_MAP


def build_non_negative_matrices(
    X_train: sp.csr_matrix,
    X_val: sp.csr_matrix,
    X_dense_raw_train: np.ndarray,
    X_dense_raw_val: np.ndarray,
    metadata: Dict[str, Any],
) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """
    Constructs non-negative feature matrices for Naive Bayes by scaling dense features
    to [0, 1] with MinMaxScaler (fitted strictly on Train) and concatenating with non-negative TF-IDF.
    """
    n_dense = metadata["n_dense_features"]
    mm_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    dense_tr_pos = np.clip(mm_scaler.fit_transform(X_dense_raw_train), 0.0, None)
    dense_va_pos = np.clip(mm_scaler.transform(X_dense_raw_val), 0.0, None)

    X_tr_ngrams = X_train[:, n_dense:]
    X_va_ngrams = X_val[:, n_dense:]

    X_tr_pos = sp.hstack([sp.csr_matrix(dense_tr_pos), X_tr_ngrams], format="csr")
    X_va_pos = sp.hstack([sp.csr_matrix(dense_va_pos), X_va_ngrams], format="csr")
    return X_tr_pos, X_va_pos


def evaluate_binary_baselines(
    X_train: sp.csr_matrix,
    X_val: sp.csr_matrix,
    X_tr_pos: sp.csr_matrix,
    X_va_pos: sp.csr_matrix,
    y_bin_train: np.ndarray,
    y_bin_val: np.ndarray,
    n_cv_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Trains, cross-validates (5-fold Stratified CV on Train), and evaluates all baseline models
    on the Validation set for Task 1: Binary Generative AI Detection (0=Human, 1=AI).
    """
    models_spec = [
        (
            "Dummy (Majority Class)",
            DummyClassifier(strategy="most_frequent"),
            False,
        ),
        (
            "Complement Naive Bayes",
            ComplementNB(alpha=0.5),
            True,  # Uses non-negative matrix
        ),
        (
            "Logistic Regression (L1 Lasso)",
            LogisticRegression(penalty="l1", C=1.0, solver="liblinear", max_iter=1000, random_state=random_state),
            False,
        ),
        (
            "Logistic Regression (L2 Ridge)",
            LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=random_state),
            False,
        ),
        (
            "Linear SVM (Stamatatos Standard)",
            LinearSVC(C=0.5, max_iter=2000, random_state=random_state),
            False,
        ),
    ]

    skf = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=random_state)

    summary_rows = []
    roc_curves = {}
    confusion_matrices = {}
    predictions_val = {}
    probabilities_val = {}
    fitted_models = {}

    for name, clf_proto, use_pos in models_spec:
        X_tr_use = X_tr_pos if use_pos else X_train
        X_va_use = X_va_pos if use_pos else X_val

        # 1. 5-Fold Stratified Cross-Validation on Train
        cv_f1s = []
        cv_accs = []
        t0_cv = time.time()
        for tr_idx, te_idx in skf.split(X_tr_use, y_bin_train):
            fold_clf = clone(clf_proto)
            fold_clf.fit(X_tr_use[tr_idx], y_bin_train[tr_idx])
            fold_pred = fold_clf.predict(X_tr_use[te_idx])
            cv_f1s.append(f1_score(y_bin_train[te_idx], fold_pred, average="macro"))
            cv_accs.append(accuracy_score(y_bin_train[te_idx], fold_pred))
        cv_time = time.time() - t0_cv

        # 2. Fit on full Train & Evaluate on official Validation
        t0_fit = time.time()
        clf = clone(clf_proto)
        clf.fit(X_tr_use, y_bin_train)
        fit_time = time.time() - t0_fit

        y_pred = clf.predict(X_va_use)
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X_va_use)[:, 1]
        elif hasattr(clf, "decision_function"):
            dec = clf.decision_function(X_va_use)
            y_score = 1.0 / (1.0 + np.exp(-np.clip(dec, -20, 20)))
        else:
            y_score = y_pred.astype(float)

        macro_f1 = f1_score(y_bin_val, y_pred, average="macro")
        acc = accuracy_score(y_bin_val, y_pred)
        bal_acc = balanced_accuracy_score(y_bin_val, y_pred)
        prec = precision_score(y_bin_val, y_pred, zero_division=0)
        rec = recall_score(y_bin_val, y_pred, zero_division=0)
        f1_human = f1_score(y_bin_val, y_pred, pos_label=0, zero_division=0)
        f1_ai = f1_score(y_bin_val, y_pred, pos_label=1, zero_division=0)
        try:
            auc = roc_auc_score(y_bin_val, y_score)
        except Exception:
            auc = 0.5

        cm = confusion_matrix(y_bin_val, y_pred, labels=[0, 1])
        fpr, tpr, _ = roc_curve(y_bin_val, y_score)

        summary_rows.append({
            "model_name": name,
            "cv5_macro_f1_mean": round(float(np.mean(cv_f1s)), 4),
            "cv5_macro_f1_std": round(float(np.std(cv_f1s)), 4),
            "cv5_acc_mean": round(float(np.mean(cv_accs)), 4),
            "val_macro_f1": round(float(macro_f1), 4),
            "val_accuracy": round(float(acc), 4),
            "val_balanced_acc": round(float(bal_acc), 4),
            "val_precision_ai": round(float(prec), 4),
            "val_recall_ai": round(float(rec), 4),
            "val_f1_human": round(float(f1_human), 4),
            "val_f1_ai": round(float(f1_ai), 4),
            "val_roc_auc": round(float(auc), 4),
            "tn_human": int(cm[0, 0]),
            "fp_human_as_ai": int(cm[0, 1]),
            "fn_ai_as_human": int(cm[1, 0]),
            "tp_ai": int(cm[1, 1]),
            "train_fit_sec": round(fit_time, 2),
            "cv_total_sec": round(cv_time, 2),
        })

        roc_curves[name] = (fpr, tpr, auc)
        confusion_matrices[name] = cm
        predictions_val[name] = y_pred
        probabilities_val[name] = y_score
        fitted_models[name] = clf

    df_summary = pd.DataFrame(summary_rows)
    return {
        "summary_df": df_summary,
        "roc_curves": roc_curves,
        "confusion_matrices": confusion_matrices,
        "predictions_val": predictions_val,
        "probabilities_val": probabilities_val,
        "fitted_models": fitted_models,
    }


def evaluate_multiclass_and_two_stage_attribution(
    X_train: sp.csr_matrix,
    X_val: sp.csr_matrix,
    X_tr_pos: sp.csr_matrix,
    X_va_pos: sp.csr_matrix,
    y_bin_train: np.ndarray,
    y_bin_val: np.ndarray,
    y_model_train: np.ndarray,
    y_model_val: np.ndarray,
    model_classes: List[str],
    best_binary_preds_val: np.ndarray,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Evaluates Task 2: Model Attribution / LLM Fingerprinting (predicting which of the
    23 classes [Human + 22 AI models] wrote each text).
    Compares:
    1. Direct 23-class Multiclass Baselines (Dummy, ComplementNB, Logistic Regression, LinearSVC)
    2. Two-Stage Hierarchical Pipeline:
       - Stage 1: Binary Detector (Human vs AI)
       - Stage 2: 22-LLM Specialist Classifier (trained strictly on AI texts y_bin == 1)
    """
    ai_mask_tr = y_bin_train == 1
    ai_mask_va = y_bin_val == 1

    # Map each of the 23 classes to its 7-family architectural index
    family_list = [
        "Human",
        "OpenAI (GPT/o3)",
        "Google (Gemini/PaLM)",
        "Meta (LLaMA)",
        "Mistral AI",
        "DeepSeek / Qwen",
        "TII Falcon",
    ]
    fam_to_idx = {f: i for i, f in enumerate(family_list)}
    class_idx_to_fam_idx = np.array([
        fam_to_idx[MODEL_FAMILY_MAP.get(c, "OpenAI (GPT/o3)")] for c in model_classes
    ])
    y_fam_val = class_idx_to_fam_idx[y_model_val]

    direct_specs = [
        ("Dummy (Majority Class - 23c)", DummyClassifier(strategy="most_frequent"), False),
        ("Complement Naive Bayes (23c)", ComplementNB(alpha=0.5), True),
        (
            "Logistic Regression (Multinomial L2 - 23c)",
            LogisticRegression(C=1.0, solver="lbfgs", max_iter=600, random_state=random_state),
            False,
        ),
        (
            "Linear SVM (One-vs-Rest - 23c)",
            LinearSVC(C=0.5, max_iter=2000, random_state=random_state),
            False,
        ),
    ]

    rows = []
    multiclass_preds = {}

    for name, clf_proto, use_pos in direct_specs:
        X_tr_u = X_tr_pos if use_pos else X_train
        X_va_u = X_va_pos if use_pos else X_val

        t0 = time.time()
        clf = clone(clf_proto)
        clf.fit(X_tr_u, y_model_train)
        fit_sec = time.time() - t0

        y_pred = clf.predict(X_va_u)
        if hasattr(clf, "predict_proba"):
            y_scores = clf.predict_proba(X_va_u)
        elif hasattr(clf, "decision_function"):
            y_scores = softmax(clf.decision_function(X_va_u), axis=1)
        else:
            y_scores = np.eye(len(model_classes))[y_pred]

        macro_f1_23 = f1_score(y_model_val, y_pred, average="macro", zero_division=0)
        weighted_f1_23 = f1_score(y_model_val, y_pred, average="weighted", zero_division=0)
        acc_23 = accuracy_score(y_model_val, y_pred)
        top3_acc_23 = top_k_accuracy_score(
            y_model_val, y_scores, k=3, labels=np.arange(len(model_classes))
        )

        # AI-only subset evaluation (how well it identifies the exact LLM on true AI texts)
        ai_exact_acc = accuracy_score(y_model_val[ai_mask_va], y_pred[ai_mask_va])
        ai_macro_f1 = f1_score(
            y_model_val[ai_mask_va], y_pred[ai_mask_va],
            labels=np.arange(1, len(model_classes)), average="macro", zero_division=0
        )

        # Family-level accuracy (7 architectural families)
        y_fam_pred = class_idx_to_fam_idx[y_pred]
        fam_acc = accuracy_score(y_fam_val, y_fam_pred)
        fam_macro_f1 = f1_score(y_fam_val, y_fam_pred, average="macro", zero_division=0)

        # Implied Binary Accuracy (predicting class 0 vs >0)
        implied_bin_pred = (y_pred > 0).astype(int)
        implied_bin_f1 = f1_score(y_bin_val, implied_bin_pred, average="macro")

        rows.append({
            "model_name": name,
            "architecture": "Direct 23-Class",
            "val_23class_macro_f1": round(float(macro_f1_23), 4),
            "val_23class_weighted_f1": round(float(weighted_f1_23), 4),
            "val_23class_top1_acc": round(float(acc_23), 4),
            "val_23class_top3_acc": round(float(top3_acc_23), 4),
            "val_22ai_exact_acc": round(float(ai_exact_acc), 4),
            "val_22ai_macro_f1": round(float(ai_macro_f1), 4),
            "val_7family_acc": round(float(fam_acc), 4),
            "val_7family_macro_f1": round(float(fam_macro_f1), 4),
            "implied_binary_macro_f1": round(float(implied_bin_f1), 4),
            "fit_sec": round(fit_sec, 2),
        })
        multiclass_preds[name] = y_pred

    # ------------------------------------------------------------------
    # Two-Stage Hierarchical Pipeline:
    # Stage 1: Binary Detector (Human=0 vs AI=1)
    # Stage 2: 22-LLM Specialist Classifier trained strictly on AI texts
    # ------------------------------------------------------------------
    t0_2s = time.time()
    stage2_svm = LinearSVC(C=0.5, max_iter=2000, random_state=random_state)
    stage2_svm.fit(X_train[ai_mask_tr], y_model_train[ai_mask_tr])

    stage2_lr = LogisticRegression(C=1.5, solver="lbfgs", max_iter=600, random_state=random_state)
    stage2_lr.fit(X_train[ai_mask_tr], y_model_train[ai_mask_tr])
    fit_2s_sec = time.time() - t0_2s

    for s2_name, s2_clf in [
        ("Two-Stage Hierarchical (Stage1 Bin + Stage2 Linear SVM)", stage2_svm),
        ("Two-Stage Hierarchical (Stage1 Bin + Stage2 Logistic Reg)", stage2_lr),
    ]:
        y_pred_2stage = np.zeros_like(y_model_val)
        pred_ai_mask = best_binary_preds_val == 1
        if np.any(pred_ai_mask):
            y_pred_2stage[pred_ai_mask] = s2_clf.predict(X_val[pred_ai_mask])

        # Construct 23-class score matrix for Top-3 accuracy
        if hasattr(s2_clf, "decision_function"):
            s2_scores_all = softmax(s2_clf.decision_function(X_val), axis=1)
        else:
            s2_scores_all = s2_clf.predict_proba(X_val)

        scores_23 = np.zeros((X_val.shape[0], len(model_classes)), dtype=np.float64)
        # If binary predicts human, put high weight on class 0, else distribute across 1..22
        scores_23[~pred_ai_mask, 0] = 0.95
        scores_23[~pred_ai_mask, 1:] = 0.05 / 22.0
        scores_23[pred_ai_mask, 0] = 0.02
        scores_23[pred_ai_mask, 1:] = 0.98 * s2_scores_all[pred_ai_mask]

        macro_f1_23 = f1_score(y_model_val, y_pred_2stage, average="macro", zero_division=0)
        weighted_f1_23 = f1_score(y_model_val, y_pred_2stage, average="weighted", zero_division=0)
        acc_23 = accuracy_score(y_model_val, y_pred_2stage)
        top3_acc_23 = top_k_accuracy_score(
            y_model_val, scores_23, k=3, labels=np.arange(len(model_classes))
        )

        ai_exact_acc = accuracy_score(y_model_val[ai_mask_va], y_pred_2stage[ai_mask_va])
        ai_macro_f1 = f1_score(
            y_model_val[ai_mask_va], y_pred_2stage[ai_mask_va],
            labels=np.arange(1, len(model_classes)), average="macro", zero_division=0
        )

        y_fam_pred = class_idx_to_fam_idx[y_pred_2stage]
        fam_acc = accuracy_score(y_fam_val, y_fam_pred)
        fam_macro_f1 = f1_score(y_fam_val, y_fam_pred, average="macro", zero_division=0)
        implied_bin_f1 = f1_score(y_bin_val, (y_pred_2stage > 0).astype(int), average="macro")

        rows.append({
            "model_name": s2_name,
            "architecture": "Two-Stage Hierarchical",
            "val_23class_macro_f1": round(float(macro_f1_23), 4),
            "val_23class_weighted_f1": round(float(weighted_f1_23), 4),
            "val_23class_top1_acc": round(float(acc_23), 4),
            "val_23class_top3_acc": round(float(top3_acc_23), 4),
            "val_22ai_exact_acc": round(float(ai_exact_acc), 4),
            "val_22ai_macro_f1": round(float(ai_macro_f1), 4),
            "val_7family_acc": round(float(fam_acc), 4),
            "val_7family_macro_f1": round(float(fam_macro_f1), 4),
            "implied_binary_macro_f1": round(float(implied_bin_f1), 4),
            "fit_sec": round(fit_2s_sec, 2),
        })
        multiclass_preds[s2_name] = y_pred_2stage

    df_multiclass_summary = pd.DataFrame(rows)

    # Family Confusion Matrix (7x7) for the best Two-Stage model
    best_2s_preds = multiclass_preds["Two-Stage Hierarchical (Stage1 Bin + Stage2 Linear SVM)"]
    best_fam_preds = class_idx_to_fam_idx[best_2s_preds]
    cm_family = confusion_matrix(y_fam_val, best_fam_preds, labels=np.arange(len(family_list)))
    df_cm_family = pd.DataFrame(cm_family, index=family_list, columns=family_list)

    return {
        "summary_df": df_multiclass_summary,
        "predictions_val": multiclass_preds,
        "family_confusion_matrix": df_cm_family,
        "family_list": family_list,
        "class_idx_to_fam_idx": class_idx_to_fam_idx,
    }


def compute_cross_genre_and_per_model_diagnostics(
    meta_val: pd.DataFrame,
    y_bin_val: np.ndarray,
    y_model_val: np.ndarray,
    binary_preds_dict: Dict[str, np.ndarray],
    best_multiclass_pred: np.ndarray,
    model_classes: List[str],
    class_idx_to_fam_idx: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes detailed error diagnostics on Validation Set:
    1. Cross-Domain / Cross-Genre Generalization (Fiction, Essays, News) across all baseline models.
    2. Per-Author / Per-LLM Breakdown (23 classes): Binary Detection Rate, Evasion Rate (FN),
       Exact Model Attribution Accuracy, and Architectural Family Accuracy.
    """
    genres = sorted(meta_val["genre"].unique())
    genre_rows = []

    for model_name, y_pred_bin in binary_preds_dict.items():
        for g in genres:
            mask = (meta_val["genre"] == g).values
            y_true_g = y_bin_val[mask]
            y_pred_g = y_pred_bin[mask]

            f1_g = f1_score(y_true_g, y_pred_g, average="macro", zero_division=0)
            acc_g = accuracy_score(y_true_g, y_pred_g)
            cm_g = confusion_matrix(y_true_g, y_pred_g, labels=[0, 1])
            tn, fp, fn, tp = cm_g.ravel()
            fpr = fp / max(tn + fp, 1)
            fnr = fn / max(fn + tp, 1)

            # Also check 23-class attribution accuracy within this genre for the best model
            attr_acc_g = accuracy_score(y_model_val[mask], best_multiclass_pred[mask])

            genre_rows.append({
                "baseline_model": model_name,
                "genre": g,
                "n_samples": int(mask.sum()),
                "n_human": int((y_true_g == 0).sum()),
                "n_ai": int((y_true_g == 1).sum()),
                "binary_macro_f1": round(float(f1_g), 4),
                "binary_accuracy": round(float(acc_g), 4),
                "false_positive_rate_human": round(float(fpr), 4),
                "false_negative_rate_ai": round(float(fnr), 4),
                "multiclass23_acc_best": round(float(attr_acc_g), 4),
            })

    df_genre = pd.DataFrame(genre_rows)

    # 2. Per-Model / Per-Author Breakdown using the best binary detector + Two-Stage Attribution
    best_bin_pred = binary_preds_dict["Linear SVM (Stamatatos Standard)"]
    y_fam_true = class_idx_to_fam_idx[y_model_val]
    y_fam_pred = class_idx_to_fam_idx[best_multiclass_pred]

    per_model_rows = []
    for cls_idx, cls_name in enumerate(model_classes):
        mask = y_model_val == cls_idx
        n_cls = int(mask.sum())
        if n_cls == 0:
            continue

        is_human = cls_idx == 0
        true_bin_label = 0 if is_human else 1
        bin_correct = float(np.mean(best_bin_pred[mask] == true_bin_label))
        evasion_or_fp_rate = 1.0 - bin_correct

        exact_attr_acc = float(np.mean(best_multiclass_pred[mask] == cls_idx))
        family_attr_acc = float(np.mean(y_fam_pred[mask] == y_fam_true[mask]))

        # Find most common misattribution when wrong
        preds_for_cls = best_multiclass_pred[mask]
        wrong_preds = preds_for_cls[preds_for_cls != cls_idx]
        if len(wrong_preds) > 0:
            vals, counts = np.unique(wrong_preds, return_counts=True)
            top_confused_cls = model_classes[vals[np.argmax(counts)]]
        else:
            top_confused_cls = "None (100%)"

        per_model_rows.append({
            "model_name": cls_name,
            "arch_family": MODEL_FAMILY_MAP.get(cls_name, "Other"),
            "true_binary_class": "Human (0)" if is_human else "AI (1)",
            "n_val_samples": n_cls,
            "binary_detection_acc": round(bin_correct, 4),
            "error_or_evasion_rate": round(evasion_or_fp_rate, 4),
            "exact_model_attribution_acc": round(exact_attr_acc, 4),
            "family_attribution_acc": round(family_attr_acc, 4),
            "most_confused_with": top_confused_cls,
        })

    df_per_model = pd.DataFrame(per_model_rows).sort_values(
        by=["true_binary_class", "exact_model_attribution_acc"], ascending=[False, False]
    ).reset_index(drop=True)

    return df_genre, df_per_model


def execute_full_baseline_benchmark(
    cached_data: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Runs the complete Stage 4 Dual-Task Baseline Benchmark (Binary Detection + Model Attribution)
    and exports summary tables to `output_dir`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train = cached_data["X_train"]
    X_val = cached_data["X_val"]
    X_dense_raw_train = cached_data["X_dense_raw_train"]
    X_dense_raw_val = cached_data["X_dense_raw_val"]
    y_bin_train = cached_data["y_bin_train"]
    y_bin_val = cached_data["y_bin_val"]
    y_model_train = cached_data["y_model_train"]
    y_model_val = cached_data["y_model_val"]
    meta_val = cached_data["meta_val"]
    metadata = cached_data["metadata"]
    model_classes = metadata["model_classes"]

    # 1. Build non-negative feature matrices for Complement Naive Bayes
    X_tr_pos, X_va_pos = build_non_negative_matrices(
        X_train, X_val, X_dense_raw_train, X_dense_raw_val, metadata
    )

    # 2. Task 1: Binary Classification Baselines (Human vs AI)
    bin_results = evaluate_binary_baselines(
        X_train, X_val, X_tr_pos, X_va_pos, y_bin_train, y_bin_val
    )
    bin_results["summary_df"].to_csv(output_dir / "baseline_binary_results.csv", index=False)

    # Select best binary predictions for Stage 1 of the Two-Stage Hierarchical Pipeline
    best_bin_preds = bin_results["predictions_val"]["Linear SVM (Stamatatos Standard)"]

    # 3. Task 2: Model Attribution Baselines (23 classes & Two-Stage Hierarchical)
    multi_results = evaluate_multiclass_and_two_stage_attribution(
        X_train, X_val, X_tr_pos, X_va_pos,
        y_bin_train, y_bin_val, y_model_train, y_model_val,
        model_classes, best_bin_preds,
    )
    multi_results["summary_df"].to_csv(output_dir / "baseline_multiclass_attribution_results.csv", index=False)
    multi_results["family_confusion_matrix"].to_csv(output_dir / "baseline_family_confusion_matrix.csv")

    # 4. Cross-Genre & Per-Model Diagnostics
    best_multi_preds = multi_results["predictions_val"][
        "Two-Stage Hierarchical (Stage1 Bin + Stage2 Linear SVM)"
    ]
    df_genre, df_per_model = compute_cross_genre_and_per_model_diagnostics(
        meta_val, y_bin_val, y_model_val,
        bin_results["predictions_val"],
        best_multi_preds,
        model_classes,
        multi_results["class_idx_to_fam_idx"],
    )
    df_genre.to_csv(output_dir / "baseline_cross_genre_diagnostics.csv", index=False)
    df_per_model.to_csv(output_dir / "baseline_per_llm_attribution_diagnostics.csv", index=False)

    return {
        "binary": bin_results,
        "multiclass": multi_results,
        "cross_genre_df": df_genre,
        "per_model_df": df_per_model,
    }
