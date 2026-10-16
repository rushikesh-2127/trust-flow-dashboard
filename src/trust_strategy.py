"""
trust_strategy.py

Phase 4 (core contribution): TrustWeightedFedAvg.

Standard FedAvg weights each client's contribution purely by how many
examples it has. That means a malicious or corrupted node with a lot of
(poisoned) data gets a big vote. TrustWeightedFedAvg additionally scores
each client's update by:

  1. Similarity  -- cosine similarity between this client's weight update
     and the (num-examples-weighted) mean update across all participating
     clients this round. An update that points in a very different
     direction from the consensus is suspicious.
  2. Historical consistency -- an exponential moving average (EMA) of a
     client's similarity score across rounds, so a node that is
     *consistently* a bit off is penalised more than one with a single bad
     round, and a previously-flagged node can slowly regain trust if it
     starts behaving normally again.

The final aggregation weight for each client combines its trust score with
its example count (so trust down-weights bad actors without ignoring
dataset size entirely), then does a weighted average of the raw weights.
"""

from typing import Optional
import numpy as np
# pyrefly: ignore [missing-import]
import flwr as fl
# pyrefly: ignore [missing-import]
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
# pyrefly: ignore [missing-import]
from flwr.server.client_proxy import ClientProxy

import model_utils as mu


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the cosine similarity between two flat vectors.

    :param a: First vector.
    :param b: Second vector.
    :returns: Cosine similarity in [-1, 1], or 0.0 if either vector is zero.
    """
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class TrustWeightedFedAvg(fl.server.strategy.FedAvg):
    """Federated Averaging strategy with dynamic trust-weighted aggregation.

    Calculates client update reliability by measuring cosine similarity against
    the healthy consensus update direction, tracks historical trust using an
    Exponential Moving Average (EMA), and weights client updates accordingly to
    mitigate model poisoning and Byzantine attacks.
    """

    def __init__(self, *args, trust_ema_alpha: float = 0.5, trust_floor: float = 0.05, **kwargs):
        """Initialise TrustWeightedFedAvg with EMA trust tracking.

        :param trust_ema_alpha: EMA smoothing factor for trust scores.
            Higher values weight the current round more heavily.
        :param trust_floor: Minimum per-node aggregation weight so no
            node is ever fully zeroed out.
        """
        super().__init__(*args, **kwargs)
        self.trust_ema_alpha = trust_ema_alpha
        self.trust_floor = trust_floor  # minimum weight so no node is ever fully zeroed out
        self.trust_history: dict[str, float] = {}
        self.trust_log: list[dict] = []  # for later plotting/analysis
        self.current_weights: Optional[list[np.ndarray]] = (
            parameters_to_ndarrays(kwargs["initial_parameters"])
            if kwargs.get("initial_parameters") is not None
            else None
        )
        self.last_consensus: Optional[np.ndarray] = None  # unit-normalized "healthy" update direction

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list,
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate client updates using trust-weighted averaging.

        Computes per-client trust scores via cosine similarity to the consensus
        direction, applies an EMA for historical consistency, then performs a
        trust-and-example-count weighted average of raw client weights.

        :param server_round: Current federated learning round number.
        :param results: Client proxies and their fit results from this round.
        :param failures: List of failed client updates (unused).
        :returns: The aggregated global parameters and a metrics dict of
            per-node trust scores.
        """
        if not results:
            return None, {}

        node_ids, weight_list, n_examples, deltas = [], [], [], []
        prev_flat = (
            mu.flatten_weights(self.current_weights) if self.current_weights is not None else None
        )

        for _, fit_res in results:
            nid = str(fit_res.metrics.get("node_id", "unknown"))
            if nid in node_ids:
                raise ValueError(
                    f"Duplicate node_id {nid!r} in round {server_round}. "
                    "Each client must map to a unique partition."
                )
            w = parameters_to_ndarrays(fit_res.parameters)
            flat = mu.flatten_weights(w)
            if not np.all(np.isfinite(flat)):
                # A NaN/inf update would poison the whole average: exclude it
                # and record zero trust so it shows up in the log.
                self.trust_history[nid] = 0.0
                self.trust_log.append({"round": server_round, "node_id": nid, "trust": 0.0, "raw_similarity": float("nan")})
                continue
            node_ids.append(nid)
            weight_list.append(w)
            n_examples.append(fit_res.num_examples)
            deltas.append(flat - prev_flat if prev_flat is not None else flat)

        if not node_ids:
            return None, {}

        # Consensus direction: mean of UNIT-NORMALIZED deltas (not raw deltas).
        # Averaging raw deltas would let a magnitude-boosted malicious update
        # (a "model replacement" style attack) drag the consensus itself
        # toward it, defeating the whole point of comparing against it.
        # Normalizing first means every client gets an equal-magnitude "vote"
        # on the consensus direction, so as long as honest clients are the
        # majority, the consensus stays anchored to their direction
        # regardless of how large the malicious update's magnitude is.
        unit_deltas = []
        for d in deltas:
            norm = np.linalg.norm(d)
            unit_deltas.append(d / norm if norm > 0 else d)
        consensus = np.mean(np.stack(unit_deltas), axis=0)
        if prev_flat is not None:
            self.last_consensus = consensus  # reused to score new/unseen data later

        round_scores = {}
        for nid, delta in zip(node_ids, deltas):
            # Without the previous global weights there is no "update" to
            # compare (round 1 with no initial_parameters): no information,
            # so every client gets full trust this round.
            sim = _cosine_sim(delta, consensus) if prev_flat is not None else 1.0
            sim_clipped = max(sim, 0.0)  # negative similarity -> treat as 0 trust this round
            prev = self.trust_history.get(nid, sim_clipped)
            ema = self.trust_ema_alpha * sim_clipped + (1 - self.trust_ema_alpha) * prev
            self.trust_history[nid] = ema
            round_scores[nid] = ema
            self.trust_log.append({"round": server_round, "node_id": nid, "trust": ema, "raw_similarity": sim})

        # Combine trust with example count, apply a floor so no client is fully zeroed
        raw_weights = np.array(
            [max(round_scores[nid], self.trust_floor) * n for nid, n in zip(node_ids, n_examples)],
            dtype=float,
        )
        total = raw_weights.sum()
        norm_weights = raw_weights / total if total > 0 else np.full(len(raw_weights), 1.0 / len(raw_weights))

        # Weighted average of the RAW client weights (not deltas)
        new_global = [np.zeros_like(layer) for layer in weight_list[0]]
        for w, alpha in zip(weight_list, norm_weights):
            for layer_idx, layer in enumerate(w):
                new_global[layer_idx] = new_global[layer_idx] + alpha * layer

        self.current_weights = new_global
        metrics = {f"trust/{nid}": round_scores[nid] for nid in node_ids}
        return ndarrays_to_parameters(new_global), metrics
