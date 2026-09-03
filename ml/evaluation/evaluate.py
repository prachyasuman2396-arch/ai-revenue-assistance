import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Ensure project root in sys.path
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from ml.training.train import load_and_split_data

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_PATH = BASE_DIR / "models" / "champion_model.joblib"
EVAL_OUT_PATH = BASE_DIR / "models" / "evaluation_report.json"


def evaluate_champion_model(
    model_path: Path = MODEL_PATH,
    output_path: Path = EVAL_OUT_PATH,
) -> Dict[str, Any]:
    """Performs rigorous out-of-sample evaluation on the 15% held-out test split."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at: {model_path}. Run Phase 8 first."
        )

    logger.info(f"Loading champion model from: {model_path}")
    model = joblib.load(model_path)

    # Obtain exact test split
    _, _, X_test, _, _, y_test = load_and_split_data()
    test_count = len(y_test)
    churn_count = int(y_test.sum())

    # Predict probabilities and binary outcomes
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred_default = (y_prob >= 0.5).astype(int)

    # 1. Primary Metrics
    roc_auc = float(roc_auc_score(y_test, y_prob))
    pr_auc = float(average_precision_score(y_test, y_prob))
    brier = float(brier_score_loss(y_test, y_prob))
    f1 = float(f1_score(y_test, y_pred_default))
    precision = float(precision_score(y_test, y_pred_default))
    recall = float(recall_score(y_test, y_pred_default))

    # 2. Confusion Matrix at Threshold 0.5
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_default).ravel()

    # 3. Decision Threshold Analysis (Precision-Recall-Cost Trade-offs)
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    threshold_analysis = []
    for thresh in thresholds:
        preds = (y_prob >= thresh).astype(int)
        c_matrix = confusion_matrix(y_test, preds)
        t_tn, t_fp, t_fn, t_tp = c_matrix.ravel()
        t_prec = precision_score(y_test, preds, zero_division=0)
        t_rec = recall_score(y_test, preds, zero_division=0)
        t_f1 = f1_score(y_test, preds, zero_division=0)
        threshold_analysis.append(
            {
                "threshold": thresh,
                "precision": round(float(t_prec), 4),
                "recall": round(float(t_rec), 4),
                "f1": round(float(t_f1), 4),
                "true_positives": int(t_tp),
                "false_positives": int(t_fp),
                "false_negatives": int(t_fn),
                "true_negatives": int(t_tn),
            }
        )

    # 4. Calibration Curve (Reliability Diagram)
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=5)
    calibration_data = {
        "bin_mean_predicted_prob": [round(float(p), 4) for p in prob_pred],
        "bin_empirical_fraction_positives": [round(float(p), 4) for p in prob_true],
    }

    # 5. Segment Error Analysis (False Positives vs False Negatives)
    test_analysis_df = X_test.copy()
    test_analysis_df["true_churn"] = y_test
    test_analysis_df["pred_prob"] = np.round(y_prob, 4)
    test_analysis_df["pred_label"] = y_pred_default

    fp_cases = test_analysis_df[
        (test_analysis_df["true_churn"] == 0) & (test_analysis_df["pred_label"] == 1)
    ]
    fn_cases = test_analysis_df[
        (test_analysis_df["true_churn"] == 1) & (test_analysis_df["pred_label"] == 0)
    ]

    error_analysis = {
        "false_positive_count": len(fp_cases),
        "false_positive_characteristics": {
            "avg_monthly_charges": round(float(fp_cases["monthly_charges"].mean()), 2)
            if not fp_cases.empty
            else 0.0,
            "avg_tenure": round(float(fp_cases["tenure"].mean()), 1)
            if not fp_cases.empty
            else 0.0,
            "pct_month_to_month": round(
                float((fp_cases["contract"] == "Month-to-month").mean() * 100), 2
            )
            if not fp_cases.empty
            else 0.0,
        },
        "false_negative_count": len(fn_cases),
        "false_negative_characteristics": {
            "avg_monthly_charges": round(float(fn_cases["monthly_charges"].mean()), 2)
            if not fn_cases.empty
            else 0.0,
            "avg_tenure": round(float(fn_cases["tenure"].mean()), 1)
            if not fn_cases.empty
            else 0.0,
            "pct_month_to_month": round(
                float((fn_cases["contract"] == "Month-to-month").mean() * 100), 2
            )
            if not fn_cases.empty
            else 0.0,
        },
    }

    report = {
        "test_population": test_count,
        "test_churn_count": churn_count,
        "test_churn_rate_pct": round((churn_count / test_count) * 100, 2),
        "overall_metrics": {
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "brier_score": round(brier, 4),
            "f1_score": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        },
        "confusion_matrix_at_0_5": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "threshold_tradeoffs": threshold_analysis,
        "calibration_curve": calibration_data,
        "error_analysis": error_analysis,
    }

    # Print Formatted Evaluation Report
    print("\n" + "=" * 70)
    print("PHASE 9: RIGOROUS MODEL EVALUATION & ERROR ANALYSIS")
    print("=" * 70)
    print(
        f"Test Set Size: {test_count} | Actual Churners: {churn_count} ({report['test_churn_rate_pct']}%)"
    )
    print("\n[Primary Discrimination & Calibration Metrics]")
    print(f"  ROC-AUC:     {roc_auc:.4f}")
    print(
        f"  PR-AUC:      {pr_auc:.4f} (Baseline No-Skill: {churn_count / test_count:.4f})"
    )
    print(f"  Brier Score: {brier:.4f} (Lower = Better Calibration)")
    print(f"  F1-Score:    {f1:.4f}")
    print(f"  Precision:   {precision:.4f}")
    print(f"  Recall:      {recall:.4f}")

    print("\n[Confusion Matrix (Threshold = 0.50)]")
    print(f"  TN: {tn:<5} | FP (False Alarms): {fp:<5}")
    print(f"  FN: {fn:<5} | TP (Captured):     {tp:<5}")

    print("\n[Threshold Sensitivity Table]")
    print(
        f"{'Threshold':<11} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'TP':<6} | {'FP':<6}"
    )
    print("-" * 65)
    for t in threshold_analysis:
        print(
            f"{t['threshold']:<11.2f} | {t['precision']:<10.4f} | {t['recall']:<10.4f} | {t['f1']:<10.4f} | {t['true_positives']:<6} | {t['false_positives']:<6}"
        )

    print("\n[Error Analysis]")
    print(
        f"  False Positives (N={len(fp_cases)}): Avg Bill=${error_analysis['false_positives']['avg_monthly_charges'] if 'false_positives' in error_analysis else error_analysis['false_positive_characteristics']['avg_monthly_charges']}, Avg Tenure={error_analysis['false_positive_characteristics']['avg_tenure']} mos, {error_analysis['false_positive_characteristics']['pct_month_to_month']}% Month-to-month"
    )
    print(
        f"  False Negatives (N={len(fn_cases)}): Avg Bill=${error_analysis['false_negative_characteristics']['avg_monthly_charges']}, Avg Tenure={error_analysis['false_negative_characteristics']['avg_tenure']} mos, {error_analysis['false_negative_characteristics']['pct_month_to_month']}% Month-to-month"
    )

    # Persist JSON report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Evaluation report saved to: {output_path}")

    return report


if __name__ == "__main__":
    evaluate_champion_model()
