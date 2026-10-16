"""Tests for dashboard helpers (auth + telemetry audit); no Streamlit needed."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dashboard"))

import auth  # noqa: E402
from common import FEATURE_COLUMNS  # noqa: E402
from data_audit import normalize_and_align_features, trace_file_bugs_and_anomalies  # noqa: E402


def test_password_hash_roundtrip_and_unique_salt():
    h1, h2 = auth.hash_password("s3cret!"), auth.hash_password("s3cret!")
    assert h1 != h2  # per-user random salt
    assert auth.verify_password("s3cret!", h1)
    assert not auth.verify_password("wrong", h1)


def test_legacy_sha256_hash_is_rejected_not_crashing():
    legacy = "c2815acee3ea8726a163304bfa3302b10bfd6eada6a1dbf95d5ba94a1650bb84"
    assert auth.verify_password("TrustFL@2026", legacy) is False


def test_load_users_creates_demo_account(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "USERS_FILE", tmp_path / "users.json")
    users = auth.load_users()
    assert "demo@trustfl.org" in users
    assert auth.verify_password("TrustFL@2026", users["demo@trustfl.org"]["password_hash"])


def test_alias_mapping_and_imputation():
    df = pd.DataFrame({"tot_fwd_pkts": [1, 2], "Flow Duration": [0.1, 0.2], "class": ["normal", "dos"]})
    out, imputed = normalize_and_align_features(df)
    assert {"src_pkts", "duration", "label"} <= set(out.columns)
    assert out["label"].tolist() == [0, 1]
    assert set(imputed) == set(FEATURE_COLUMNS) - {"src_pkts", "duration"}


def test_trace_flags_and_sanitizes_corrupt_telemetry():
    df = pd.DataFrame(np.ones((4, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    df.loc[0, "duration"] = -5.0
    df.loc[1, "src_bytes"] = -1
    df.loc[2, "dst_bytes"] = np.nan
    result = trace_file_bugs_and_anomalies(df, [])
    bugs = {f["bug"] for f in result["findings"]}
    assert "Clock Skew / Negative Duration" in bugs
    assert "Integer Underflow / Negative Counter" in bugs
    san = result["sanitized_df"]
    assert (san["duration"] >= 0).all() and (san["src_bytes"] >= 0).all()
    assert not san.isnull().any().any()


def test_password_over_72_bytes_is_rejected_cleanly():
    import pytest
    with pytest.raises(ValueError):
        auth.hash_password("x" * 73)
    assert auth.verify_password("x" * 73, auth.hash_password("ok")) is False


def test_corrupt_user_file_is_preserved_not_overwritten(tmp_path, monkeypatch):
    f = tmp_path / "users.json"
    f.write_text("{not json")
    monkeypatch.setattr(auth, "USERS_FILE", f)
    users = auth.load_users()
    assert "demo@trustfl.org" in users
    assert (tmp_path / "users.corrupt").read_text() == "{not json"


def test_score_device_handles_odd_labels_and_text_cells():
    import joblib
    from scoring import score_device

    results = ROOT / "results"
    scaler = joblib.load(results / "scaler.joblib")
    npz = np.load(results / "global_model_weights.npz")
    weights = [npz[k] for k in npz.files]
    consensus = np.load(results / "reference_consensus.npy")

    rng = np.random.RandomState(0)
    df = pd.DataFrame(rng.rand(30, len(FEATURE_COLUMNS)), columns=FEATURE_COLUMNS)
    df["duration"] = df["duration"].astype(object)
    df.loc[0, "duration"] = "oops"          # non-numeric cell
    df["label"] = rng.choice([0, 1, 2, -1], 30)  # non-binary labels
    out = score_device(df, scaler, weights, consensus)
    assert out["trust_score"] is not None and 0.0 <= out["trust_score"] <= 1.0
    assert len(out["preds"]) == 30
    assert score_device(df.drop(columns="label"), scaler, weights, consensus)["trust_score"] is None


def test_text_cells_in_numeric_columns_are_reported_not_crashing():
    df = pd.DataFrame({c: [1.0, 2.0] for c in FEATURE_COLUMNS})
    df["duration"] = ["abc", "2"]
    aligned, imputed = normalize_and_align_features(df)
    result = trace_file_bugs_and_anomalies(aligned, imputed)
    assert any(f["bug"] == "Missing / NaN Sensor Values" for f in result["findings"])
    assert not result["sanitized_df"]["duration"].isnull().any()
