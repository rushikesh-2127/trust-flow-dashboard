"""Unit tests for partitioning, poisoning, and the trust-weighted strategy."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from common import (  # noqa: E402
    FEATURE_COLUMNS, LABEL_COLUMN, partition_iid, partition_non_iid, poison_node,
)
import model_utils as mu  # noqa: E402


def _toy_df(n=400, seed=0):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(rng.randn(n, len(FEATURE_COLUMNS)), columns=FEATURE_COLUMNS)
    df[LABEL_COLUMN] = rng.randint(0, 2, n)
    types = np.array(["normal", "dos", "scanning", "ddos", "mitm"])
    df["attack_type"] = np.where(df[LABEL_COLUMN] == 0, "normal", rng.choice(types[1:], n))
    return df


def test_iid_partition_covers_all_rows():
    df = _toy_df()
    parts = partition_iid(df, 4)
    assert len(parts) == 4
    assert sum(len(p) for p in parts) == len(df)


def test_non_iid_nodes_differ():
    parts = partition_non_iid(_toy_df(), 3)
    sigs = [tuple(sorted(p["attack_type"].value_counts().items())) for p in parts]
    assert len(set(sigs)) == 3


def test_poison_flips_expected_fraction():
    df = _toy_df()
    out = poison_node(df, 0.5)
    flipped = (out[LABEL_COLUMN] != df[LABEL_COLUMN]).mean()
    assert flipped == pytest.approx(0.5, abs=0.01)
    assert (poison_node(df, 0.0)[LABEL_COLUMN] == df[LABEL_COLUMN]).all()


def test_poison_rejects_bad_fraction():
    with pytest.raises(ValueError):
        poison_node(_toy_df(), 1.5)


def test_get_set_weights_roundtrip():
    m = mu.build_model()
    X = np.random.randn(8, len(FEATURE_COLUMNS))
    mu.init_architecture(m, X, np.array([0, 1] * 4))
    w = mu.get_weights(m)
    mu.set_weights(m, [x + 1 for x in w])
    assert all(np.allclose(a + 1, b) for a, b in zip(w, mu.get_weights(m)))


def _fit_res(weights, n, nid):
    from flwr.common import ndarrays_to_parameters
    return (None, SimpleNamespace(
        parameters=ndarrays_to_parameters(weights), num_examples=n, metrics={"node_id": nid}
    ))


def test_trust_strategy_downweights_boosted_attacker():
    pytest.importorskip("flwr")
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    from trust_strategy import TrustWeightedFedAvg

    base = [np.zeros((3, 3)), np.zeros(3)]
    honest = [np.ones((3, 3)), np.ones(3)]
    attacker = [-10 * np.ones((3, 3)), -10 * np.ones(3)]
    strat = TrustWeightedFedAvg(initial_parameters=ndarrays_to_parameters(base))
    results = [
        _fit_res(honest, 100, "0"), _fit_res(honest, 100, "1"),
        _fit_res(honest, 100, "2"), _fit_res(attacker, 100, "3"),
    ]
    params, metrics = strat.aggregate_fit(1, results, [])
    agg = parameters_to_ndarrays(params)
    assert metrics["trust/3"] < metrics["trust/0"]
    # plain FedAvg would give (3*1 - 10)/4 = -1.75; trust should land far above that
    assert agg[0].mean() > 0


def test_trust_strategy_rejects_duplicate_node_ids():
    pytest.importorskip("flwr")
    from trust_strategy import TrustWeightedFedAvg

    w = [np.ones((2, 2))]
    strat = TrustWeightedFedAvg()
    with pytest.raises(ValueError):
        strat.aggregate_fit(1, [_fit_res(w, 10, "0"), _fit_res(w, 10, "0")], [])


def test_trust_strategy_excludes_nan_client_and_logs_zero_trust():
    pytest.importorskip("flwr")
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    from trust_strategy import TrustWeightedFedAvg

    base = [np.zeros((2, 2))]
    good = [np.ones((2, 2))]
    bad = [np.full((2, 2), np.nan)]
    strat = TrustWeightedFedAvg(initial_parameters=ndarrays_to_parameters(base))
    params, metrics = strat.aggregate_fit(
        1, [_fit_res(good, 10, "0"), _fit_res(good, 10, "1"), _fit_res(bad, 10, "2")], []
    )
    assert np.isfinite(parameters_to_ndarrays(params)[0]).all()
    assert "trust/2" not in metrics
    assert any(e["node_id"] == "2" and e["trust"] == 0.0 for e in strat.trust_log)


def test_trust_strategy_handles_zero_examples_and_no_initial_parameters():
    pytest.importorskip("flwr")
    from flwr.common import parameters_to_ndarrays
    from trust_strategy import TrustWeightedFedAvg

    w = [np.ones((2, 2))]
    strat = TrustWeightedFedAvg()  # no initial_parameters
    params, _ = strat.aggregate_fit(1, [_fit_res(w, 0, "0"), _fit_res(w, 0, "1")], [])
    assert np.allclose(parameters_to_ndarrays(params)[0], 1.0)
    assert all(e["trust"] == pytest.approx(1.0) for e in strat.trust_log)
