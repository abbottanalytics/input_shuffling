"""Synthetic donor-response data with a KNOWN data-generating model.

Because the true response probability is known, input shuffling can be run on
the true model itself ("oracle"), giving a ground-truth importance ranking to
compare fitted models against.

True model (logit of P(TARGET_B = 1)):

    logit = B0
            - 0.08 * (RECENCY - 12)            strong, negative
            + 0.70 * log(FREQUENCY)            strong, positive
            + 0.45 * log(LASTGIFT / 10)        moderate, positive
            + 0.18 * HOMEOWNER * WEALTH        interaction only
            + 0.012 * (AGE - 55)               weak, positive

Variables with NO true effect:
    AVGGIFT   correlated with LASTGIFT (r ~ 0.85) — fitted models may borrow
              LASTGIFT's influence through it
    NOISE_A   pure noise
    NOISE_B   pure noise

WEALTH and HOMEOWNER have no main effect; they matter only together.

Run as a script to regenerate the CSV:

    python examples/sample_donor/in_shuff_sample_data.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE_SEED = 2026
SAMPLE_N = 4000
TEST_FRACTION = 0.40
B0 = -3.1

FEATURE_COLUMNS = [
    "RECENCY", "FREQUENCY", "LASTGIFT", "AVGGIFT", "WEALTH",
    "HOMEOWNER", "AGE", "NOISE_A", "NOISE_B",
]

DATA_PATH = Path(__file__).resolve().parent / "sample_donor_data.csv"


def true_logit(df: pd.DataFrame) -> np.ndarray:
    return (
        B0
        - 0.08 * (df["RECENCY"].to_numpy() - 12)
        + 0.70 * np.log(df["FREQUENCY"].to_numpy())
        + 0.45 * np.log(df["LASTGIFT"].to_numpy() / 10)
        + 0.18 * df["HOMEOWNER"].to_numpy() * df["WEALTH"].to_numpy()
        + 0.012 * (df["AGE"].to_numpy() - 55)
    )


def true_response_probability(df: pd.DataFrame) -> np.ndarray:
    """P(TARGET_B = 1) under the data-generating model."""
    return 1.0 / (1.0 + np.exp(-true_logit(df)))


def generate_sample_data(n: int = SAMPLE_N, seed: int = SAMPLE_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    lastgift = np.round(np.exp(rng.normal(np.log(15), 0.6, n)), 0).clip(1, 200)
    df = pd.DataFrame({
        "RECORD_ID": np.arange(1, n + 1),
        "RECENCY": rng.integers(1, 37, n),
        "FREQUENCY": (rng.poisson(4, n) + 1).clip(1, 30),
        "LASTGIFT": lastgift,
        "AVGGIFT": np.round(lastgift * np.exp(rng.normal(0, 0.3, n)), 2),
        "WEALTH": rng.integers(0, 10, n),
        "HOMEOWNER": (rng.random(n) < 0.6).astype(int),
        "AGE": np.round(rng.normal(55, 15, n)).clip(18, 95).astype(int),
        "NOISE_A": np.round(rng.normal(0, 1, n), 4),
        "NOISE_B": rng.integers(0, 5, n),
    })
    df["TRUE_PROB"] = np.round(true_response_probability(df), 6)
    df["TARGET_B"] = (rng.random(n) < df["TRUE_PROB"]).astype(int)
    df["split_flag"] = np.where(rng.random(n) < TEST_FRACTION, "TEST", "TRAIN")
    return df


def load_sample_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


if __name__ == "__main__":
    df = generate_sample_data()
    df.to_csv(DATA_PATH, index=False)
    print(f"Wrote {DATA_PATH} ({len(df)} rows)")
    print(f"Response rate: {df['TARGET_B'].mean():.3f}")
    print(df["split_flag"].value_counts().to_string())
    print(f"corr(LASTGIFT, AVGGIFT) = {df['LASTGIFT'].corr(df['AVGGIFT']):.3f}")
