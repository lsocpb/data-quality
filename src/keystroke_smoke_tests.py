# np. w src/keystroke_smoke_tests.py

import pandas as pd

from src.keystroke_identity import (
    identify_sample,
    suggest_identification_threshold,
    verify_claimed_identity,
)
from src.keystroke_leave_one_out import evaluate_leave_one_out
from src.keystroke_pipeline import clean_data, sort_data, build_features
from src.keystroke_knn import classify_sample
from src.keystroke_research_artifacts import build_keystroke_research_artifacts
from src.keystroke_smoke_fixtures import (
    build_smoke_raw_events,
    expected_smoke_features,
    expected_leave_one_out_summary,
    smoke_test_sample,
    EXPECTED_SMOKE_PREDICTION,
    EXPECTED_SMOKE_K,
    EXPECTED_SMOKE_METRIC,
    EXPECTED_IDENTIFICATION_THRESHOLD,
    EXPECTED_IDENTIFICATION_REJECT_THRESHOLD,
    EXPECTED_VERIFICATION_ACCEPT_CLAIM,
    EXPECTED_VERIFICATION_REJECT_CLAIM,
    EXPECTED_LOO_KS,
    EXPECTED_LOO_METRICS,
)


def smoke_test_keystroke_pipeline_and_knn():
    raw_df = build_smoke_raw_events()

    actual_features = build_features(sort_data(clean_data(raw_df)))
    expected_features = expected_smoke_features()

    actual_features = actual_features[expected_features.columns]

    pd.testing.assert_frame_equal(
        actual_features.reset_index(drop=True),
        expected_features.reset_index(drop=True),
        check_dtype=False,
        atol=0.01,
    )

    train_df = expected_features.copy()
    sample = smoke_test_sample()

    prediction = classify_sample(
        train_df,
        sample,
        k=EXPECTED_SMOKE_K,
        metric=EXPECTED_SMOKE_METRIC,
        exclude_same_sample=False,
    )

    assert prediction.predicted_user == EXPECTED_SMOKE_PREDICTION, (
        f"Expected {EXPECTED_SMOKE_PREDICTION}, got {prediction.predicted_user}"
    )

    print("✅ Smoke test pipeline + kNN passed")
    print(f"Expected prediction: {EXPECTED_SMOKE_PREDICTION}")
    print(f"Actual prediction: {prediction.predicted_user}")
    print(f"Metric: {EXPECTED_SMOKE_METRIC}, k={EXPECTED_SMOKE_K}")
    print(f"Score: {prediction.score:.6f}")

    loo_evaluation = evaluate_leave_one_out(
        expected_features,
        ks=EXPECTED_LOO_KS,
        metrics=EXPECTED_LOO_METRICS,
    )
    expected_summary = expected_leave_one_out_summary()

    actual_summary = loo_evaluation.summary[expected_summary.columns]
    pd.testing.assert_frame_equal(
        actual_summary.reset_index(drop=True),
        expected_summary.reset_index(drop=True),
        check_dtype=False,
        atol=0.000001,
    )

    expected_iteration_count = len(expected_features) * len(EXPECTED_LOO_KS) * len(EXPECTED_LOO_METRICS)
    assert len(loo_evaluation.iterations) == expected_iteration_count, (
        f"Expected {expected_iteration_count} LOO rows, got {len(loo_evaluation.iterations)}"
    )

    k1_predictions = loo_evaluation.iterations[
        (loo_evaluation.iterations["metric"] == "bray_curtis")
        & (loo_evaluation.iterations["k"] == 1)
    ]["predicted_user"].tolist()
    assert k1_predictions == ["Anita", "Anita", "Anita", "Olo", "Olo", "Olo"], (
        f"Unexpected LOO predictions for bray_curtis, k=1: {k1_predictions}"
    )

    threshold_recommendation = suggest_identification_threshold(
        loo_evaluation.iterations,
        metric=EXPECTED_SMOKE_METRIC,
        k=EXPECTED_SMOKE_K,
    )
    assert abs(threshold_recommendation.threshold - EXPECTED_IDENTIFICATION_THRESHOLD) < 0.000001, (
        f"Expected threshold {EXPECTED_IDENTIFICATION_THRESHOLD}, "
        f"got {threshold_recommendation.threshold}"
    )

    accepted_identification = identify_sample(
        train_df,
        sample,
        k=EXPECTED_SMOKE_K,
        metric=EXPECTED_SMOKE_METRIC,
        threshold=threshold_recommendation.threshold,
        exclude_same_sample=False,
    )
    assert accepted_identification.accepted_user == EXPECTED_SMOKE_PREDICTION, (
        f"Expected accepted user {EXPECTED_SMOKE_PREDICTION}, "
        f"got {accepted_identification.accepted_user}"
    )
    assert not accepted_identification.rejected, "Expected identification to be accepted"

    rejected_identification = identify_sample(
        train_df,
        sample,
        k=EXPECTED_SMOKE_K,
        metric=EXPECTED_SMOKE_METRIC,
        threshold=EXPECTED_IDENTIFICATION_REJECT_THRESHOLD,
        exclude_same_sample=False,
    )
    assert rejected_identification.rejected, "Expected identification to be rejected by strict threshold"
    assert rejected_identification.accepted_user is None, "Rejected identification should not accept a user"

    accepted_verification = verify_claimed_identity(
        train_df,
        sample,
        claimed_user=EXPECTED_VERIFICATION_ACCEPT_CLAIM,
        k=EXPECTED_SMOKE_K,
        metric=EXPECTED_SMOKE_METRIC,
        threshold=threshold_recommendation.threshold,
        exclude_same_sample=False,
    )
    assert accepted_verification.matched, "Expected claimed identity to be accepted"

    rejected_verification = verify_claimed_identity(
        train_df,
        sample,
        claimed_user=EXPECTED_VERIFICATION_REJECT_CLAIM,
        k=EXPECTED_SMOKE_K,
        metric=EXPECTED_SMOKE_METRIC,
        threshold=threshold_recommendation.threshold,
        exclude_same_sample=False,
    )
    assert not rejected_verification.matched, "Expected wrong claimed identity to be rejected"

    print("✅ Leave-one-out smoke test passed")
    print(f"LOO iterations: {len(loo_evaluation.iterations)}")
    print("LOO metrics checked: accuracy, precision_macro, recall_macro, f1_macro")
    print("LOO threshold, identification and verification checks passed")

    research_artifacts = build_keystroke_research_artifacts(loo_evaluation)
    pd.testing.assert_frame_equal(
        research_artifacts.aggregated_results.reset_index(drop=True),
        expected_summary.reset_index(drop=True),
        check_dtype=False,
        atol=0.000001,
    )

    assert len(research_artifacts.raw_results) == expected_iteration_count, (
        f"Expected {expected_iteration_count} raw artifact rows, got {len(research_artifacts.raw_results)}"
    )
    assert len(research_artifacts.best_by_metric) == len(EXPECTED_LOO_METRICS), (
        f"Expected {len(EXPECTED_LOO_METRICS)} best-by-metric rows, got {len(research_artifacts.best_by_metric)}"
    )
    assert len(research_artifacts.best_overall) == 1, (
        f"Expected a single best-overall row, got {len(research_artifacts.best_overall)}"
    )

    print("✅ Research artifacts smoke test passed")
