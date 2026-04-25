from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Iterable

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.keystroke_knn import (
    available_metrics,
    canonical_metric_name,
    classify_sample,
    feature_columns,
)


@dataclass(frozen=True)
class LeaveOneOutEvaluation:
    iterations: pd.DataFrame
    summary: pd.DataFrame


def leave_one_out_rows(
    features: pd.DataFrame,
    *,
    user_col: str = "UserId",
    sample_col: str = "SampleNumber",
) -> Iterable[pd.Series]:
    sorted_features = features.sort_values([user_col, sample_col]).reset_index(drop=True)
    for _, row in sorted_features.iterrows():
        yield row


def _normalize_ks(ks: Iterable[int]) -> list[int]:
    normalized_ks: list[int] = []
    for value in ks:
        k = int(value)
        if k <= 0:
            raise ValueError("All k values must be positive")
        if k not in normalized_ks:
            normalized_ks.append(k)
    if not normalized_ks:
        raise ValueError("At least one k value is required")
    return normalized_ks


def _normalize_metrics(metrics: Iterable[str] | None) -> list[str]:
    if metrics is None:
        return available_metrics()

    normalized_metrics: list[str] = []
    for metric in metrics:
        canonical_name = canonical_metric_name(metric)
        if canonical_name not in normalized_metrics:
            normalized_metrics.append(canonical_name)
    if not normalized_metrics:
        raise ValueError("At least one metric is required")
    return normalized_metrics


def _serialize_value(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)


def evaluate_leave_one_out(
    features: pd.DataFrame,
    *,
    ks: Iterable[int],
    metrics: Iterable[str] | None = None,
    user_col: str = "UserId",
    sample_col: str = "SampleNumber",
) -> LeaveOneOutEvaluation:
    if features.empty:
        raise ValueError("features cannot be empty")

    if not feature_columns(features):
        raise ValueError("No feature columns found")

    normalized_ks = _normalize_ks(ks)
    normalized_metrics = _normalize_metrics(metrics)
    sorted_features = features.sort_values([user_col, sample_col]).reset_index(drop=True)
    ordered_labels = sorted_features[user_col].drop_duplicates().tolist()

    iteration_rows: list[dict] = []
    summary_rows: list[dict] = []

    for metric_name in normalized_metrics:
        for k in normalized_ks:
            actual_users: list[object] = []
            predicted_users: list[object] = []

            for row in leave_one_out_rows(
                sorted_features,
                user_col=user_col,
                sample_col=sample_col,
            ):
                prediction = classify_sample(
                    sorted_features,
                    row,
                    k=k,
                    metric=metric_name,
                    exclude_same_sample=True,
                    user_col=user_col,
                    sample_col=sample_col,
                )
                actual_user = row[user_col]
                predicted_user = prediction.predicted_user
                is_correct = actual_user == predicted_user

                actual_users.append(actual_user)
                predicted_users.append(predicted_user)

                iteration_rows.append(
                    {
                        "metric": metric_name,
                        "k": k,
                        "actual_user": actual_user,
                        "actual_sample": row[sample_col],
                        "predicted_user": predicted_user,
                        "is_correct": is_correct,
                        "score": round(float(prediction.score), 6),
                        "neighbor_count": len(prediction.neighbors),
                        "neighbors_json": _serialize_value(prediction.neighbors),
                        "distances_by_user_json": _serialize_value(prediction.distances_by_user),
                    }
                )

            accuracy = accuracy_score(actual_users, predicted_users)
            precision, recall, f1, _ = precision_recall_fscore_support(
                actual_users,
                predicted_users,
                labels=ordered_labels,
                average="macro",
                zero_division=0,
            )

            summary_rows.append(
                {
                    "metric": metric_name,
                    "k": k,
                    "iterations": len(actual_users),
                    "correct_predictions": int(
                        sum(actual == predicted for actual, predicted in zip(actual_users, predicted_users))
                    ),
                    "accuracy": round(float(accuracy), 6),
                    "precision_macro": round(float(precision), 6),
                    "recall_macro": round(float(recall), 6),
                    "f1_macro": round(float(f1), 6),
                }
            )

    iterations_df = pd.DataFrame(iteration_rows)
    summary_df = pd.DataFrame(summary_rows)
    return LeaveOneOutEvaluation(iterations=iterations_df, summary=summary_df)
