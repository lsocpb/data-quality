from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from src.keystroke_knn import (
    KNNPrediction,
    canonical_metric_name,
    classify_sample,
    distances_to_training_rows,
    feature_columns,
)


@dataclass(frozen=True)
class ThresholdRecommendation:
    metric: str
    k: int
    threshold: float
    decision_accuracy: float
    false_accept_rate: float
    false_reject_rate: float
    strategy: str


@dataclass(frozen=True)
class IdentificationDecision:
    predicted_user: object
    accepted_user: object | None
    score: float
    threshold: float
    rejected: bool
    prediction: KNNPrediction


@dataclass(frozen=True)
class VerificationDecision:
    claimed_user: object
    matched: bool
    score: float
    threshold: float
    predicted_user: object
    claim_neighbors: list[dict]
    prediction: KNNPrediction


def _candidate_thresholds(scores: list[float]) -> list[float]:
    if not scores:
        return [0.0]

    ordered = sorted(set(float(score) for score in scores))
    candidates = [0.0]
    candidates.extend((left + right) / 2 for left, right in zip(ordered, ordered[1:]))
    candidates.extend(ordered)
    return sorted(set(candidates))


def suggest_identification_threshold(
    iterations: pd.DataFrame,
    *,
    metric: str,
    k: int,
) -> ThresholdRecommendation:
    metric_name = canonical_metric_name(metric)
    subset = iterations[
        (iterations["metric"] == metric_name) & (iterations["k"] == int(k))
    ].copy()
    if subset.empty:
        raise ValueError(f"No leave-one-out rows found for metric={metric_name}, k={k}")

    scores = subset["score"].astype(float).tolist()
    correct_mask = subset["is_correct"].astype(bool)
    correct_scores = subset.loc[correct_mask, "score"].astype(float).tolist()
    incorrect_scores = subset.loc[~correct_mask, "score"].astype(float).tolist()

    strategy = "decision_accuracy"
    if correct_scores and incorrect_scores and max(correct_scores) < min(incorrect_scores):
        threshold = (max(correct_scores) + min(incorrect_scores)) / 2
        strategy = "midpoint_gap"
    else:
        ranked_candidates = []
        total = len(subset)
        total_incorrect = len(incorrect_scores)
        total_correct = len(correct_scores)

        for candidate in _candidate_thresholds(scores):
            accepted = subset["score"].astype(float) <= candidate
            accepted_correct = int((correct_mask & accepted).sum())
            rejected_incorrect = int((~correct_mask & ~accepted).sum())
            false_accepts = int((~correct_mask & accepted).sum())
            false_rejects = int((correct_mask & ~accepted).sum())
            decision_accuracy = (accepted_correct + rejected_incorrect) / total
            false_accept_rate = false_accepts / total_incorrect if total_incorrect else 0.0
            false_reject_rate = false_rejects / total_correct if total_correct else 0.0

            ranked_candidates.append(
                (
                    -decision_accuracy,
                    false_accept_rate,
                    false_reject_rate,
                    candidate,
                )
            )

        threshold = min(ranked_candidates)[3]

    accepted = subset["score"].astype(float) <= threshold
    accepted_correct = int((correct_mask & accepted).sum())
    rejected_incorrect = int((~correct_mask & ~accepted).sum())
    false_accepts = int((~correct_mask & accepted).sum())
    false_rejects = int((correct_mask & ~accepted).sum())
    total = len(subset)

    return ThresholdRecommendation(
        metric=metric_name,
        k=int(k),
        threshold=round(float(threshold), 6),
        decision_accuracy=round(float((accepted_correct + rejected_incorrect) / total), 6),
        false_accept_rate=round(float(false_accepts / len(incorrect_scores)), 6) if incorrect_scores else 0.0,
        false_reject_rate=round(float(false_rejects / len(correct_scores)), 6) if correct_scores else 0.0,
        strategy=strategy,
    )


def identify_sample(
    features: pd.DataFrame,
    sample_row: pd.Series,
    *,
    k: int = 3,
    metric: str = "euclidean",
    threshold: float,
    exclude_same_sample: bool = False,
    user_col: str = "UserId",
    sample_col: str = "SampleNumber",
) -> IdentificationDecision:
    prediction = classify_sample(
        features,
        sample_row,
        k=k,
        metric=metric,
        exclude_same_sample=exclude_same_sample,
        user_col=user_col,
        sample_col=sample_col,
    )
    rejected = float(prediction.score) > float(threshold)
    return IdentificationDecision(
        predicted_user=prediction.predicted_user,
        accepted_user=None if rejected else prediction.predicted_user,
        score=round(float(prediction.score), 6),
        threshold=round(float(threshold), 6),
        rejected=rejected,
        prediction=prediction,
    )


def verification_score(
    features: pd.DataFrame,
    sample_row: pd.Series,
    *,
    claimed_user: object,
    k: int = 3,
    metric: str = "euclidean",
    exclude_same_sample: bool = False,
    user_col: str = "UserId",
    sample_col: str = "SampleNumber",
) -> tuple[float, list[dict]]:
    if k <= 0:
        raise ValueError("k must be positive")

    feature_cols = feature_columns(features)
    if not feature_cols:
        raise ValueError("No feature columns found")

    candidate_rows = features[features[user_col] == claimed_user]
    if exclude_same_sample:
        candidate_rows = candidate_rows[
            ~(
                (candidate_rows[user_col] == sample_row[user_col])
                & (candidate_rows[sample_col] == sample_row[sample_col])
            )
        ]
    if candidate_rows.empty:
        raise ValueError(f"Claimed user '{claimed_user}' has no training samples")

    sample_vector = [sample_row[column] for column in feature_cols]
    claim_neighbors = distances_to_training_rows(
        candidate_rows,
        sample_vector,
        metric=metric,
        user_col=user_col,
        sample_col=sample_col,
        feature_cols=feature_cols,
    )[: min(k, len(candidate_rows))]
    score = sum(neighbor["distance"] for neighbor in claim_neighbors) / len(claim_neighbors)
    return round(float(score), 6), claim_neighbors


def verify_claimed_identity(
    features: pd.DataFrame,
    sample_row: pd.Series,
    *,
    claimed_user: object,
    k: int = 3,
    metric: str = "euclidean",
    threshold: float,
    exclude_same_sample: bool = False,
    user_col: str = "UserId",
    sample_col: str = "SampleNumber",
) -> VerificationDecision:
    prediction = classify_sample(
        features,
        sample_row,
        k=k,
        metric=metric,
        exclude_same_sample=exclude_same_sample,
        user_col=user_col,
        sample_col=sample_col,
    )
    score, claim_neighbors = verification_score(
        features,
        sample_row,
        claimed_user=claimed_user,
        k=k,
        metric=metric,
        exclude_same_sample=exclude_same_sample,
        user_col=user_col,
        sample_col=sample_col,
    )
    return VerificationDecision(
        claimed_user=claimed_user,
        matched=score <= float(threshold),
        score=score,
        threshold=round(float(threshold), 6),
        predicted_user=prediction.predicted_user,
        claim_neighbors=claim_neighbors,
        prediction=prediction,
    )
