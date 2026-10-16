# Methodology

This document describes the problem, the threat model, the trust-weighted aggregation rule, the
experimental setup and the results — in the order a paper's Methodology and Results sections would
present them. Every number comes from `results/comparison.json` and `results/baseline_results.json`
(produced by `python src/run_experiment.py --suite --rounds 8` and `python src/baseline_model.py`).

Notation: *n* = number of nodes (4), *T* = rounds (8), *w_t* = global weights after round *t*,
*w_k* = node *k*'s weights after local training, *n_k* = node *k*'s number of training examples.

---

## 1. Problem

Several IoT networks want a shared intrusion-detection model (classify a network flow as
**normal** or **attack**). Pooling their traffic in one place is undesirable for privacy and
bandwidth reasons, so they use **federated learning**: each network trains on its own data and
only sends model weights to a coordinating server, which averages them into a new global model.

Standard averaging (**FedAvg**) weights each node by its data size only. It therefore trusts every
participant equally, so a single malicious or faulty participant can steer the global model.

## 2. Threat model

| Aspect | Assumption in this project |
|---|---|
| Attacker | One of the *n* nodes (node 0) is compromised; the server is honest |
| Goal | Degrade the global model's accuracy (untargeted poisoning) |
| Capability 1 — **label flipping** | The attacker flips a fraction `poison_fraction` of its local labels (`common.poison_node`). Experiments use 1.0, i.e. every label is inverted |
| Capability 2 — **update boosting** | The attacker scales its update: `w_sent = g + b·(w_local − g)` with `g` the received global weights and `b = boost_factor = 4` (a "model replacement"-style attack, `fl_client.NodeClient.fit`) |
| Knowledge | The attacker does not adapt to the defence (no adaptive/colluding attacks) |
| Honest majority | Honest nodes are the majority (3 of 4). The defence relies on this |

Out of scope: eavesdropping, backdoors, Sybil attacks, colluding attackers, attacks on the server,
and cryptographic protection of updates. The simulation has no transport layer at all.

## 3. Model

A scikit-learn `MLPClassifier` is the model on every node and on the server
(`src/model_utils.py`):

- Architecture: 16 inputs → 32 → 16 → 1 output (logistic), ReLU hidden units
- Optimiser: Adam, learning rate 1e-3, L2 `alpha` 1e-4, scikit-learn's default mini-batch size 200
- 1,089 trainable parameters, exchanged as a list of six NumPy arrays (3 weight matrices, 3 bias vectors)

Local training is `partial_fit` for `local_epochs` passes (default 2) over the node's local data.
Using scikit-learn keeps the weights as plain arrays that Flower can transmit directly, with no
deep-learning framework dependency.

## 4. Federated procedure

For each round *t = 1 … T*:

1. The server sends the global weights *w_{t−1}* to all nodes (all nodes participate every round).
2. Each node loads the weights, trains for `local_epochs` on its local training slice, and returns
   its weights *w_k* (boosted if it is the attacker) with *n_k* and its `node_id`.
3. The server aggregates the returned weights with the chosen strategy (§5 or plain FedAvg) into *w_t*.
4. The server evaluates *w_t* on a **held-out global test set** that no node ever trains on, and
   records accuracy, F1, precision, recall and log-loss.

Round 0 in the results is the evaluation of the untrained initial model.

## 5. Trust-weighted aggregation (`TrustWeightedFedAvg`)

Implemented in `src/trust_strategy.py`. Let *g* be the previous global weights.

**Step 1 — update vectors.** For each node *k*, flatten its returned weights and subtract *g*:
`Δ_k = flat(w_k) − flat(g)`.

**Step 2 — consensus direction.** Normalise every update to unit length and average:

```
u_k = Δ_k / ‖Δ_k‖          c = (1/m) · Σ_k u_k
```

Normalising first is the key design choice. If raw updates were averaged, an attacker who scales
its update by 4 would dominate *c* and the "consensus" would simply be the attacker's direction.
With unit vectors every node gets one vote of equal size, so with an honest majority *c* stays
anchored to the honest direction no matter how large the attacker's update is.

**Step 3 — similarity.** `s_k = cos(Δ_k, c)`, clipped at zero: `s_k⁺ = max(s_k, 0)`.
A negative similarity (pointing against the consensus) counts as zero trust for the round.

**Step 4 — history (EMA).** With smoothing factor α = 0.5:

```
R_k(t) = α · s_k⁺(t) + (1 − α) · R_k(t−1)         R_k(1) = s_k⁺(1)
```

A node that is only occasionally off recovers; a node that is consistently off decays toward 0.

**Step 5 — aggregation weights.** With floor τ = 0.05:

```
a_k = max(R_k, τ) · n_k          w_t = Σ_k (a_k / Σ_j a_j) · w_k
```

Note the average is over the **raw weights**, not the updates. The floor keeps every node's weight
strictly positive so no participant is ever silenced completely and the weights never sum to zero.

**Worked example (real numbers, round 1 of the attack run).** Similarities: nodes 1, 2, 3 ≈ 0.92–0.93,
node 0 (attacker) = −0.217. Trust: 0.93, 0.92, 0.92, **0.00**. All nodes hold about the same number
of examples, so the attacker's weight is `0.05 / (0.93 + 0.92 + 0.92 + 0.05) ≈ 1.8 %`, versus
25 % under FedAvg.

**Robustness details** (added during the final review): a client returning non-finite weights
(NaN/inf) is excluded from the average and logged with trust 0; duplicate `node_id`s raise an
error; if no previous global weights are available (round 1 without `initial_parameters`) every
client receives full trust that round; if all example counts are zero the weights fall back to uniform.

**Cost.** One flatten, one normalisation and one dot product per client per round — O(model size ×
clients). No extra communication.

## 6. Why plain FedAvg collapses

With FedAvg the attacker holds 25 % of the weight. Its update is `4·(w_local − g)` pointing in the
*wrong* direction (flipped labels), so its contribution to the average is roughly one full honest
update in magnitude, against the honest nodes' combined three-quarters. The honest signal is
cancelled and the model drifts below chance level: accuracy falls from 37 % at initialisation to
**26.3 %** — worse than predicting the majority class (≈ 65 %), because the model is being pushed
to predict the inverse of the truth.

## 7. Experimental setup

| Item | Value |
|---|---|
| Dataset | Synthetic, 60,000 flows × 16 numeric features (`data/generate_synthetic_data.py`); 35.3 % attacks |
| Global test set | 15 % stratified hold-out = 9,000 flows; never seen by any node |
| Training pool | 51,000 flows → 4 IID partitions of 12,750 |
| Per-node split | first 85 % (10,837 flows) for training, last 15 % for the client's own validation |
| Scaler | `StandardScaler` fitted on the training pool only, shipped to nodes |
| Nodes / rounds / local epochs | 4 / 8 / 2 |
| Attack (when on) | Node 0, label-flip fraction 1.0, boost factor 4 |
| Trust parameters | α = 0.5 (EMA), τ = 0.05 (floor) |
| Random seed | 42 (data split, partitioning, model initialisation) |
| Strategies compared | FedAvg and TrustWeightedFedAvg |
| Metrics | Accuracy, F1, precision, recall, log-loss on the global test set |

Four runs form the **suite**: `clean_fedavg`, `fedavg_under_attack`, `clean_trust`,
`trust_under_attack`. Two centralized baselines give reference points: a Random Forest and the
**same MLP** trained on all training data for 16 epochs (the same number of passes as 8 rounds × 2
local epochs).

## 8. Results

### 8.1 Final accuracy (round 8, global test set)

| Run | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|
| Centralized Random Forest (200 trees) | 88.64 % | 0.842 | — | — |
| Centralized MLP (16 epochs) | 91.28 % | 0.872 | — | — |
| FedAvg, clean | 86.16 % | 0.791 | 0.846 | 0.743 |
| FedAvg, under attack | 26.26 % | 0.248 | 0.194 | 0.344 |
| Trust-weighted, clean | 86.18 % | 0.792 | 0.846 | 0.744 |
| Trust-weighted, under attack | 86.28 % | 0.796 | 0.839 | 0.757 |

### 8.2 Accuracy by round

| Round | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| FedAvg clean | .374 | .729 | .777 | .801 | .821 | .833 | .846 | .855 | .862 |
| FedAvg attacked | .374 | .379 | .315 | .315 | .292 | .283 | .265 | .271 | .263 |
| Trust clean | .374 | .729 | .777 | .801 | .821 | .834 | .846 | .855 | .862 |
| Trust attacked | .374 | .727 | .774 | .801 | .820 | .839 | .850 | .858 | .863 |

### 8.3 Trust trajectory in the attack run

| Round | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Attacker (node 0) trust | 0.000 | 0.000 | 0.057 | 0.042 | 0.021 | 0.010 | 0.005 | 0.003 |
| Honest nodes, mean trust | 0.923 | 0.912 | 0.904 | 0.895 | 0.880 | 0.860 | 0.838 | 0.816 |

The attacker's raw cosine similarity is −0.217 in round 1 and stays within ±0.12 of zero afterwards.
The small bump in round 3 (raw similarity +0.115) is the EMA plus one round where the attacker's
update happened to point weakly along the consensus; it never exceeds 0.06.

### 8.4 Observations

1. **The defence works in this setting.** Under attack, trust-weighted aggregation reaches the same
   accuracy as the clean runs (86.3 % vs 86.2 %); FedAvg falls to 26.3 %.
2. **No cost when nobody attacks.** Clean trust-weighted and clean FedAvg are indistinguishable
   (86.18 % vs 86.16 %). The 0.1-point gap between "trust under attack" and "trust clean" is within
   noise, not evidence that attacks help.
3. **Federation costs accuracy relative to centralized training**: 86.2 % vs 91.3 % for the same
   MLP. The gap comes from federated averaging itself (local drift, a few rounds), not from the
   trust mechanism.
4. **Honest trust decays over rounds** (0.92 → 0.82; 0.98 → 0.86 even with no attacker). As the
   model converges, honest updates become small and noisier, so their cosine similarity falls.
   Trust is relative, so the ranking stays correct, but absolute trust values should not be read as
   probabilities, and fixed thresholds on them (as the dashboard uses) would need re-calibration for
   other settings.

## 9. Evaluation protocol notes

- All strategies see identical data, partitions, initial weights and seed, so differences come only
  from aggregation.
- The global test set is used only for evaluation, never for any training decision.
- Trust scores are computed from updates only; the server never sees raw data or labels.
- The dashboard's **device trust score** reuses the same signal: it trains a copy of the global
  model on the uploaded device's labelled flows for 2 epochs and measures the cosine similarity of
  that update to the saved consensus direction (`dashboard/scoring.py`).

## 10. Reproducing the numbers

```powershell
python -m venv .venv ; .venv\Scripts\Activate.ps1
pip install -r requirements.lock.txt         # exact tested versions
python data/generate_synthetic_data.py       # only if data/raw is missing
python src/baseline_model.py
python src/run_experiment.py --suite --rounds 8
```

Results are deterministic given the seed for data and model initialisation, but scikit-learn's
mini-batch shuffling and Ray scheduling can cause small last-digit differences between machines.
A single seed is used; for publication, repeat with several seeds and report mean ± standard
deviation (see [LIMITATIONS_AND_FUTURE_WORK.md](LIMITATIONS_AND_FUTURE_WORK.md)).
