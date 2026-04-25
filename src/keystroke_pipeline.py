from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
NOTEBOOKS_ENV_PATH = PROJECT_ROOT / "notebooks" / ".env"


def _load_database_url_from_env_file(env_path: Path) -> str | None:
    if not env_path.exists():
        return None

    load_dotenv(env_path)
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return database_url

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        prefix = line.split("=", 1)[0]
        if "://" in prefix:
            return line

    return None


# CONFIG / ENGINE
def get_engine():
    database_url = _load_database_url_from_env_file(ENV_PATH)
    if not database_url:
        database_url = _load_database_url_from_env_file(NOTEBOOKS_ENV_PATH)
    if not database_url:
        load_dotenv()
        database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL not found. Put it in "
            f"{ENV_PATH} or {NOTEBOOKS_ENV_PATH}"
        )

    return create_engine(database_url)

#LOAD DATA
def load_keystrokes(engine):
    query = 'SELECT * FROM "Keystrokes"'
    df = pd.read_sql(query, engine)
    return df

#CLEAN DATA
def clean_data(df):
    df = df.copy()

    # usuwamy brakujące wartości
    df = df.dropna()

    # poprawność czasu
    df = df[df["ReleaseTime"] >= df["PressTime"]]

    # liczymy hold time
    df["hold_time"] = df["ReleaseTime"] - df["PressTime"]

    # usuwamy anomalia (np. > 1 sekundy)
    df = df[df["hold_time"] < 1000]

    return df

#SORT DATA
def sort_data(df):
    return df.sort_values(["UserId", "SampleNumber", "PressTime"])

#FEATURE ENGINEERING
def build_features(df):
    df = df.copy()

    # agregacja: średni hold_time per klawisz
    features = (
        df.groupby(["UserId", "SampleNumber", "KeyPressed"])["hold_time"]
        .mean()
        .unstack()
    )

    # brakujące klawisze → 0
    features = features.fillna(0)

    # nazwy kolumn
    features.columns = [f"hold_{str(col)}" for col in features.columns]

    features = features.reset_index()

    return features

#PIPELINE
def run_pipeline(engine):
    df = load_keystrokes(engine)
    df = clean_data(df)
    df = sort_data(df)
    features = build_features(df)

    return features

#TEST
def test_pipeline(engine):
    df = run_pipeline(engine)

    # podstawowe kolumny
    assert "UserId" in df.columns
    assert "SampleNumber" in df.columns

    # czy są jakieś cechy
    feature_cols = [col for col in df.columns if col.startswith("hold_")]
    assert len(feature_cols) > 0, "No feature columns found"

    # czy coś jest w danych
    assert len(df) > 0, "Empty dataframe"

    print("✅ Pipeline działa poprawnie!")
