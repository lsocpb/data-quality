from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.keystroke_leave_one_out import LeaveOneOutEvaluation
from src.keystroke_knn import available_metrics


@dataclass(frozen=True)
class KeystrokeResearchArtifacts:
    raw_results: pd.DataFrame
    aggregated_results: pd.DataFrame
    accuracy_curve: pd.DataFrame
    best_by_metric: pd.DataFrame
    best_overall: pd.DataFrame


def _sorted_aggregated_results(summary_df: pd.DataFrame) -> pd.DataFrame:
    metric_order = available_metrics()
    sorted_df = summary_df.copy()
    metric_rank = {metric_name: index for index, metric_name in enumerate(metric_order)}
    sorted_df.loc[:, "_metric_rank"] = sorted_df["metric"].map(metric_rank).fillna(len(metric_rank))
    sorted_df = sorted_df.sort_values(["_metric_rank", "k"]).reset_index(drop=True)
    sorted_df = sorted_df.drop(columns="_metric_rank")
    return sorted_df


def _build_accuracy_curve(aggregated_df: pd.DataFrame) -> pd.DataFrame:
    return (
        aggregated_df.loc[:, ["metric", "k", "accuracy", "iterations", "correct_predictions"]]
        .sort_values(["metric", "k"])
        .reset_index(drop=True)
    )


def _build_best_by_metric(aggregated_df: pd.DataFrame) -> pd.DataFrame:
    ranked = aggregated_df.sort_values(
        ["metric", "accuracy", "f1_macro", "precision_macro", "recall_macro", "k"],
        ascending=[True, False, False, False, False, True],
    )
    best_by_metric = ranked.groupby("metric", as_index=False).first()
    return best_by_metric.sort_values("metric").reset_index(drop=True)


def _build_best_overall(aggregated_df: pd.DataFrame) -> pd.DataFrame:
    return (
        aggregated_df.sort_values(
            ["accuracy", "f1_macro", "precision_macro", "recall_macro", "k", "metric"],
            ascending=[False, False, False, False, True, True],
        )
        .reset_index(drop=True)
        .head(1)
    )


def build_keystroke_research_artifacts(
    evaluation: LeaveOneOutEvaluation,
) -> KeystrokeResearchArtifacts:
    metric_order = available_metrics()
    raw_results = evaluation.iterations.copy()
    metric_rank = {metric_name: index for index, metric_name in enumerate(metric_order)}
    raw_results.loc[:, "_metric_rank"] = raw_results["metric"].map(metric_rank).fillna(len(metric_rank))
    raw_results = raw_results.sort_values(["_metric_rank", "k", "actual_user", "actual_sample"]).reset_index(
        drop=True
    )
    raw_results = raw_results.drop(columns="_metric_rank")
    aggregated_results = _sorted_aggregated_results(evaluation.summary)
    accuracy_curve = _build_accuracy_curve(aggregated_results)
    best_by_metric = _build_best_by_metric(aggregated_results)
    best_overall = _build_best_overall(aggregated_results)

    return KeystrokeResearchArtifacts(
        raw_results=raw_results,
        aggregated_results=aggregated_results,
        accuracy_curve=accuracy_curve,
        best_by_metric=best_by_metric,
        best_overall=best_overall,
    )


def plot_accuracy_vs_k(accuracy_curve: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for metric_name, metric_rows in accuracy_curve.groupby("metric"):
        metric_rows = metric_rows.sort_values("k")
        plt.plot(metric_rows["k"], metric_rows["accuracy"], marker="o", label=metric_name)

    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs k dla leave-one-out")
    plt.xticks(sorted(accuracy_curve["k"].unique()))
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Metryka")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
