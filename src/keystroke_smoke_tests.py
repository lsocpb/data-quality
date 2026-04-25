# np. w src/keystroke_smoke_tests.py

import pandas as pd

from src.keystroke_pipeline import clean_data, sort_data, build_features
from src.keystroke_knn import classify_sample
from src.keystroke_smoke_fixtures import (
    build_smoke_raw_events,
    expected_smoke_features,
    smoke_test_sample,
    EXPECTED_SMOKE_PREDICTION,
    EXPECTED_SMOKE_K,
    EXPECTED_SMOKE_METRIC,
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