from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import pandas as pd

MetricFn = Callable[[Iterable[float], Iterable[float]], float]


def _as_float_lists(a: Iterable[float], b: Iterable[float]) -> tuple[list[float], list[float]]:
    left = [float(value) for value in a]
    right = [float(value) for value in b]
    if len(left) != len(right):
        raise ValueError(f"Vectors must have equal length: {len(left)} != {len(right)}")
    return left, right


def euclidean_distance(a: Iterable[float], b: Iterable[float]) -> float:
    left, right = _as_float_lists(a, b)
    return sum((x - y) ** 2 for x, y in zip(left, right)) ** 0.5


def chebyshev_distance(a: Iterable[float], b: Iterable[float]) -> float:
    left, right = _as_float_lists(a, b)
    if not left:
        return 0.0
    return max(abs(x - y) for x, y in zip(left, right))


def bray_curtis_distance(a: Iterable[float], b: Iterable[float]) -> float:
    left, right = _as_float_lists(a, b)
    numerator = sum(abs(x - y) for x, y in zip(left, right))
    denominator = sum(abs(x + y) for x, y in zip(left, right))
    if denominator == 0:
        return 0.0
    return numerator / denominator


METRICS: dict[str, MetricFn] = {
    "euclidean": euclidean_distance,
    "chebyshev": chebyshev_distance,
    "bray_curtis": bray_curtis_distance,
    "bray-curtis": bray_curtis_distance,
    "braycurtis": bray_curtis_distance,
}


def canonical_metric_name(metric: str | MetricFn) -> str:
    if callable(metric):
        return getattr(metric, "__name__", "callable_metric")

    key = metric.lower()
    if key in {"bray-curtis", "braycurtis"}:
        return "bray_curtis"
    return key


def available_metrics() -> list[str]:
    ordered_metrics: list[str] = []
    for metric_name in METRICS:
        canonical_name = canonical_metric_name(metric_name)
        if canonical_name not in ordered_metrics:
            ordered_metrics.append(canonical_name)
    return ordered_metrics


@dataclass(frozen=True)
class KNNPrediction:
    predicted_user: object
    neighbors: list[dict]
    distances_by_user: dict[object, float]
    score: float


def feature_columns(features: pd.DataFrame) -> list[str]:
    return [column for column in features.columns if column.startswith("hold_")]


def get_metric(metric: str | MetricFn) -> MetricFn:
    if callable(metric):
        return metric
    key = canonical_metric_name(metric)
    if key not in METRICS:
        raise ValueError(f"Unsupported metric '{metric}'. Choose one of: {sorted(METRICS)}")
    return METRICS[key]


def knn_predict(
    train_features: pd.DataFrame,
    sample_vector: Iterable[float],
    *,
    k: int = 3,
    metric: str | MetricFn = "euclidean",
    user_col: str = "UserId",
    sample_col: str = "SampleNumber",
    feature_cols: list[str] | None = None,
) -> KNNPrediction:
    if k <= 0:
        raise ValueError("k must be positive")
    if train_features.empty:
        raise ValueError("train_features cannot be empty")

    feature_cols = feature_cols or feature_columns(train_features)
    if not feature_cols:
        raise ValueError("No feature columns found")

    metric_fn = get_metric(metric)
    sample_vector = [float(value) for value in sample_vector]

    neighbors = []
    for _, row in train_features.iterrows():
        train_vector = [row[column] for column in feature_cols]
        distance = metric_fn(sample_vector, train_vector)
        neighbors.append({"user": row[user_col], "sample": row[sample_col], "distance": distance})

    neighbors = sorted(neighbors, key=lambda item: item["distance"])[: min(k, len(neighbors))]

    grouped_distances: dict[object, list[float]] = {}
    for neighbor in neighbors:
        grouped_distances.setdefault(neighbor["user"], []).append(neighbor["distance"])

    distances_by_user = {
        user: sum(distances) / len(distances)
        for user, distances in grouped_distances.items()
    }
    vote_counts = {user: len(distances) for user, distances in grouped_distances.items()}
    predicted_user = min(vote_counts, key=lambda user: (-vote_counts[user], distances_by_user[user]))

    return KNNPrediction(
        predicted_user=predicted_user,
        neighbors=neighbors,
        distances_by_user=distances_by_user,
        score=distances_by_user[predicted_user],
    )


def classify_sample(
    features: pd.DataFrame,
    sample_row: pd.Series,
    *,
    k: int = 3,
    metric: str | MetricFn = "euclidean",
    exclude_same_sample: bool = False,
    user_col: str = "UserId",
    sample_col: str = "SampleNumber",
) -> KNNPrediction:
    feature_cols = feature_columns(features)
    train_features = features

    if exclude_same_sample:
        mask = ~((features[user_col] == sample_row[user_col]) & (features[sample_col] == sample_row[sample_col]))
        train_features = features[mask]

    sample_vector = [sample_row[column] for column in feature_cols]
    return knn_predict(
        train_features,
        sample_vector,
        k=k,
        metric=metric,
        user_col=user_col,
        sample_col=sample_col,
        feature_cols=feature_cols,
    )
