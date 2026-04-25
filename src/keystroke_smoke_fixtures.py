# src/keystroke_smoke_fixtures.py

import pandas as pd


SMOKE_FEATURE_COLUMNS = [
    "hold_a",
    "hold_e",
    "hold_o",
    "hold_r",
    "hold_d",
    "hold_w",
    "hold_Shift",
    "hold_Backspace",
]


def build_smoke_raw_events() -> pd.DataFrame:
    """
    Stały zestaw surowych zdarzeń klawiatury.

    Dane są przygotowane tak, żeby przypominały wartości z realnego pliku
    keystrokes_features.csv, ale nie wymagają połączenia z bazą danych.
    """
    rows = []

    templates = {
        "Anita": {
            1: {"a": 107.12, "e": 113.43, "o": 73.28, "r": 97.19, "d": 106.40, "w": 87.60, "Backspace": 88.40},
            2: {"a": 105.33, "e": 85.65, "o": 103.06, "r": 105.32, "d": 71.37, "w": 86.63, "Backspace": 85.06},
        },
        "Majkel": {
            1: {"a": 105.85, "e": 119.04, "o": 129.35, "r": 114.07, "d": 127.75, "w": 108.20},
            2: {"a": 110.00, "e": 118.68, "o": 131.57, "r": 125.00, "d": 119.15, "w": 112.97, "Shift": 346.10, "Backspace": 97.27},
        },
        "Olo": {
            1: {"a": 100.95, "e": 96.66, "o": 79.76, "r": 105.84, "d": 95.15, "w": 144.80, "Shift": 284.60, "Backspace": 86.57},
            2: {"a": 99.06, "e": 99.22, "o": 91.86, "r": 95.70, "d": 112.15, "w": 129.95, "Shift": 235.30, "Backspace": 129.10},
        },
    }

    event_id = 0

    for user_id, samples in templates.items():
        for sample_number, keys in samples.items():
            press_time = 1000 * sample_number

            for key, hold_time in keys.items():
                rows.append(
                    {
                        "UserId": user_id,
                        "SampleNumber": sample_number,
                        "KeyPressed": key,
                        "PressTime": press_time,
                        "ReleaseTime": press_time + hold_time,
                    }
                )

                event_id += 1
                press_time += 200

    return pd.DataFrame(rows)


def expected_smoke_features() -> pd.DataFrame:
    """
    Oczekiwany rezultat feature engineering.

    To jest wynik, który powinien powstać po:
    clean_data -> sort_data -> build_features
    dla danych z build_smoke_raw_events().
    """
    rows = [
        {
            "UserId": "Anita",
            "SampleNumber": 1,
            "hold_a": 107.12,
            "hold_e": 113.43,
            "hold_o": 73.28,
            "hold_r": 97.19,
            "hold_d": 106.40,
            "hold_w": 87.60,
            "hold_Shift": 0.00,
            "hold_Backspace": 88.40,
        },
        {
            "UserId": "Anita",
            "SampleNumber": 2,
            "hold_a": 105.33,
            "hold_e": 85.65,
            "hold_o": 103.06,
            "hold_r": 105.32,
            "hold_d": 71.37,
            "hold_w": 86.63,
            "hold_Shift": 0.00,
            "hold_Backspace": 85.06,
        },
        {
            "UserId": "Majkel",
            "SampleNumber": 1,
            "hold_a": 105.85,
            "hold_e": 119.04,
            "hold_o": 129.35,
            "hold_r": 114.07,
            "hold_d": 127.75,
            "hold_w": 108.20,
            "hold_Shift": 0.00,
            "hold_Backspace": 0.00,
        },
        {
            "UserId": "Majkel",
            "SampleNumber": 2,
            "hold_a": 110.00,
            "hold_e": 118.68,
            "hold_o": 131.57,
            "hold_r": 125.00,
            "hold_d": 119.15,
            "hold_w": 112.97,
            "hold_Shift": 346.10,
            "hold_Backspace": 97.27,
        },
        {
            "UserId": "Olo",
            "SampleNumber": 1,
            "hold_a": 100.95,
            "hold_e": 96.66,
            "hold_o": 79.76,
            "hold_r": 105.84,
            "hold_d": 95.15,
            "hold_w": 144.80,
            "hold_Shift": 284.60,
            "hold_Backspace": 86.57,
        },
        {
            "UserId": "Olo",
            "SampleNumber": 2,
            "hold_a": 99.06,
            "hold_e": 99.22,
            "hold_o": 91.86,
            "hold_r": 95.70,
            "hold_d": 112.15,
            "hold_w": 129.95,
            "hold_Shift": 235.30,
            "hold_Backspace": 129.10,
        },
    ]

    return pd.DataFrame(rows)


def smoke_test_sample() -> pd.Series:
    
    return pd.Series(
        {
            "UserId": "UNKNOWN",
            "SampleNumber": 99,
            "hold_a": 108.00,
            "hold_e": 119.00,
            "hold_o": 130.00,
            "hold_r": 120.00,
            "hold_d": 123.00,
            "hold_w": 111.00,
            "hold_Shift": 0.00,
            "hold_Backspace": 0.00,
        }
    )


EXPECTED_SMOKE_PREDICTION = "Majkel"
EXPECTED_SMOKE_METRIC = "bray_curtis"
EXPECTED_SMOKE_K = 1