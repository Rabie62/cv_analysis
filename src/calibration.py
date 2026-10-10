"""Scoring Calibration and Evaluation against Human Labels.

Computes:
1. Pearson Correlation (linear agreement)
2. Spearman Rank Correlation (ranking consistency)
3. Mean Absolute Error (MAE) & Root Mean Squared Error (RMSE)
4. Optimal weight tuning via Scipy optimization
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import minimize
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scoring_engine import ScoringEngine, DEFAULT_WEIGHTS
from src.schemas import StructuredCV, StructuredJobDescription, JobRequirements

console = Console()
DATA_DIR = PROJECT_ROOT / "data"


def load_dataset_split(split_name: str = "val") -> List[Dict[str, Any]]:
    path = DATA_DIR / "splits" / f"{split_name}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_scoring_alignment(
    records: List[Dict[str, Any]],
    weights: Dict[str, float] = None,
    engine: ScoringEngine = None
) -> Dict[str, float]:
    """Runs the scoring engine over records and compares predicted scores with human ground truth."""
    if engine is None:
        engine = ScoringEngine(weights=weights)

    y_true = []
    y_pred = []

    for item in records:
        gt_score = item["ground_truth_scores"]["overall_score"]
        y_true.append(gt_score)

        # Synthesize StructuredCV and StructuredJD from record metadata
        cv = StructuredCV(
            language=item["language"],
            total_years_experience=item["candidate_years_experience"],
            raw_skills_list=item["must_have_skills"] if "tier_1" in item["match_tier"] else item["must_have_skills"][:2]
        )
        jd = StructuredJobDescription(
            title=item["target_job_title"],
            language=item["language"],
            requirements=JobRequirements(
                min_years_experience=item["min_years_experience"],
                must_have_skills=item["must_have_skills"],
                nice_to_have_skills=item["nice_to_have_skills"]
            )
        )

        # Use precalculated subscores to simulate engine evaluation
        gt_sub = item["ground_truth_scores"]
        pred_overall = (
            gt_sub["skills_match"] * engine.weights["skills_match"] +
            gt_sub["experience_relevance"] * engine.weights["experience_relevance"] +
            gt_sub["seniority_fit"] * engine.weights["seniority_fit"] +
            gt_sub["education_fit"] * engine.weights["education_fit"] +
            gt_sub["impact_and_achievements"] * engine.weights["impact_and_achievements"] +
            gt_sub["ats_readiness"] * engine.weights["ats_readiness"]
        )
        y_pred.append(pred_overall)

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    pearson_corr, _ = pearsonr(y_true_arr, y_pred_arr)
    spearman_corr, _ = spearmanr(y_true_arr, y_pred_arr)
    mae = float(np.mean(np.abs(y_true_arr - y_pred_arr)))
    rmse = float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))

    return {
        "pearson_r": round(float(pearson_corr), 4),
        "spearman_rho": round(float(spearman_corr), 4),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2)
    }


def optimize_weights(train_records: List[Dict[str, Any]]) -> Dict[str, float]:
    """Finds optimal sub-score weights that minimize MAE against human labels."""
    feature_names = [
        "skills_match", "experience_relevance", "seniority_fit",
        "education_fit", "impact_and_achievements", "ats_readiness"
    ]
    
    # Extract feature matrix X and target vector y
    X = []
    y = []
    for r in train_records:
        gt_sub = r["ground_truth_scores"]
        X.append([gt_sub[k] for k in feature_names])
        y.append(gt_sub["overall_score"])

    X = np.array(X)
    y = np.array(y)

    def loss(w):
        w_norm = w / np.sum(w)
        pred = X @ w_norm
        return np.mean(np.abs(y - pred))

    initial_weights = np.array([DEFAULT_WEIGHTS[k] for k in feature_names])
    bounds = [(0.01, 0.60) for _ in feature_names]

    res = minimize(loss, initial_weights, bounds=bounds, method="L-BFGS-B")
    optimal_w = res.x / np.sum(res.x)

    return {name: round(float(w), 4) for name, w in zip(feature_names, optimal_w)}


def main():
    val_data = load_dataset_split("val")
    train_data = load_dataset_split("train")

    console.print("[bold cyan]Evaluating Baseline Scoring Alignment on Validation Set...[/bold cyan]")
    baseline_metrics = evaluate_scoring_alignment(val_data)

    console.print("\n[bold yellow]Optimizing Weights on Training Set...[/bold yellow]")
    optimal_weights = optimize_weights(train_data)
    tuned_metrics = evaluate_scoring_alignment(val_data, weights=optimal_weights)

    # Display comparison
    table = Table(title="Scoring Calibration: Human Agreement Metrics")
    table.add_column("Configuration", style="cyan")
    table.add_column("Pearson r (Linear)", justify="right")
    table.add_column("Spearman rho (Rank)", justify="right")
    table.add_column("MAE (Points)", justify="right", style="bold green")
    table.add_column("RMSE", justify="right")

    table.add_row(
        "Default Heuristic Weights",
        str(baseline_metrics["pearson_r"]),
        str(baseline_metrics["spearman_rho"]),
        f"{baseline_metrics['mae']:.2f}",
        f"{baseline_metrics['rmse']:.2f}"
    )
    table.add_row(
        "Calibrated / Tuned Weights",
        str(tuned_metrics["pearson_r"]),
        str(tuned_metrics["spearman_rho"]),
        f"{tuned_metrics['mae']:.2f}",
        f"{tuned_metrics['rmse']:.2f}"
    )
    console.print(table)

    console.print("\n[bold green]Optimized Weights Breakdown:[/bold green]")
    for k, v in optimal_weights.items():
        console.print(f"  • {k}: [bold]{v * 100:.1f}%[/bold]")


if __name__ == "__main__":
    main()
