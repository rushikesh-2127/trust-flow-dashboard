"""Score an uploaded device's traffic against the trained federated model."""

import numpy as np
import pandas as pd

import model_utils as mu
from common import FEATURE_COLUMNS, LABEL_COLUMN


def _numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce feature columns to numbers; unparseable cells become 0."""
    return df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _model_from(weights, X: np.ndarray):
    model = mu.build_model()
    n = min(4, len(X))
    mu.init_architecture(model, X[:n], np.array([0, 1, 0, 1])[:n])
    mu.set_weights(model, weights)
    return model


def score_device(df: pd.DataFrame, scaler, global_weights, consensus, epochs: int = 2) -> dict:
    """Classify every flow and, if labels are present, compute a trust score.

    The trust score is the cosine similarity between the weight update this
    device would send (local training on its labelled flows) and the saved
    healthy-consensus direction, clipped at 0 -- the same signal
    ``TrustWeightedFedAvg`` uses during training.

    :param df: Sanitized frame containing all FEATURE_COLUMNS (and optionally a label).
    :param scaler: Fitted scaler from training.
    :param global_weights: Trained global model weights.
    :param consensus: Saved reference consensus direction.
    :param epochs: Local training epochs used to produce the update.
    :returns: ``preds``, ``probs``, ``attack_rate`` and ``trust_score``
        (None when the frame has no label column).
    """
    X = scaler.transform(_numeric_features(df))
    model = _model_from(global_weights, X)
    preds = model.predict(X)
    probs = model.predict_proba(X)

    trust_score = None
    if LABEL_COLUMN in df.columns:
        # Any positive value counts as "attack"; missing/garbage counts as normal.
        y = (pd.to_numeric(df[LABEL_COLUMN], errors="coerce").fillna(0).to_numpy() > 0).astype(np.int64)
        local = _model_from(global_weights, X)
        mu.local_train(local, X, y, epochs=epochs)
        delta = mu.flatten_weights(mu.get_weights(local)) - mu.flatten_weights(global_weights)
        dn, cn = np.linalg.norm(delta), np.linalg.norm(consensus)
        trust_score = max(float(delta @ consensus / (dn * cn)), 0.0) if dn > 0 and cn > 0 else 0.0

    return {
        "preds": preds,
        "probs": probs,
        "attack_rate": float(preds.mean()),
        "trust_score": trust_score,
    }
