# Copyright (C) 2026 Dean Abbott / Abbott Analytics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The sample donor example reproduces its stored answers."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "sample_donor"
sys.path.insert(0, str(EXAMPLE_DIR))

import in_shuff_run_sample as sample  # noqa: E402
from in_shuff_sample_data import generate_sample_data, load_sample_data  # noqa: E402


def test_committed_csv_matches_generator():
    pd.testing.assert_frame_equal(load_sample_data(), generate_sample_data(), check_dtype=False)


def test_oracle_matches_expected_answers():
    df = load_sample_data()
    test = df[df["split_flag"] == "TEST"]
    result = sample.run_input_shuffling("oracle", sample.true_response_probability, test)
    ok, msg = sample.compare_with_expected("oracle", result)
    assert ok, msg

    # Variables outside the true model never move the true score
    zero = result.set_index("feature").loc[["AVGGIFT", "NOISE_A", "NOISE_B"], "mean_record_iqr"]
    assert (zero == 0).all()
    assert result.loc[result["rank"] == 1, "feature"].item() == "RECENCY"


def test_fitted_models_agree_with_oracle_on_top_variable():
    df = load_sample_data()
    train, test = df[df["split_flag"] == "TRAIN"], df[df["split_flag"] == "TEST"]
    scorers = sample.build_models(train)
    result = sample.run_input_shuffling("logistic", scorers["logistic"], test)
    assert result.loc[result["rank"] == 1, "feature"].item() == "RECENCY"
    noise_ranks = result.set_index("feature").loc[["NOISE_A", "NOISE_B"], "rank"]
    assert (noise_ranks >= 7).all()
