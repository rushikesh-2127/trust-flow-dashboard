# Limitations and Future Work

What the results do and do not show, the known weaknesses of the code, and a prioritised list of
next steps. Read this before quoting any number from the project.

## What the results support

- On this synthetic dataset, with 4 nodes and one attacker that flips every label and boosts its
  update ×4, trust-weighted aggregation keeps accuracy at the clean level (86.3 %) while plain
  FedAvg falls to 26.3 %.
- Trust-weighting does not hurt when nobody attacks (86.18 % vs 86.16 %).
- The attacker's trust stays at or below 0.06 in every round.

## Scientific limitations

1. **One seed, one run per configuration.** Differences such as 86.28 % vs 86.18 % are noise, and no
   confidence intervals exist. *Fix:* repeat the suite over ≥ 5 seeds and report mean ± std; test
   significance of the FedAvg-vs-trust gap under attack.
2. **Synthetic data.** The data come from `make_classification`. Nothing guarantees real IoT traffic
   behaves similarly, and `attack_type` is random with respect to the features, so the seven attack
   types are indistinguishable and the "non-IID by attack type" partition does not create real
   feature-space heterogeneity. *Fix:* run on TON_IoT / CIC-IoT2023 (the loader is designed for it).
3. **Only IID data was evaluated.** The non-IID partitioner exists and is unit-tested, but no
   experiment was run with it. Non-IID data is exactly where cosine-similarity trust is most at
   risk of penalising honest-but-different nodes. *Fix:* run the suite with `--partition noniid`
   and measure false-positive down-weighting.
4. **Small federation.** 4 nodes, all participating every round; one attacker (25 %). *Fix:* vary
   node count (10–100), participation rate and attacker fraction.
5. **Non-adaptive attacker.** The attack does not know about the defence. A smarter attacker can
   craft updates that are aligned with the consensus but still harmful (small-step or
   "a little is enough" style attacks), collude with other attackers, or behave honestly for several
   rounds to build trust first. *Fix:* implement adaptive and colluding attacks.
6. **Needs an honest majority.** The consensus is the mean of unit updates; with more than half the
   nodes malicious it follows the attackers. This is inherent to the method and should be stated in
   the paper. *Fix:* sweep the attacker fraction (1 of 4, 2 of 4, 3 of 4) and report where it breaks.
7. **Only two attack types.** Label flipping and update boosting. No backdoors, targeted
   poisoning, or free-riding/noisy updates.
8. **No baseline defences.** FedAvg is the only comparison. Standard robust aggregators (Krum,
   median, trimmed mean, FLTrust, norm clipping) are not compared. *Fix:* add at least coordinate-wise
   median, trimmed mean and Krum.
9. **Trust decays for honest nodes** (0.92 → 0.82 in 8 rounds; 0.98 → 0.86 with no attacker),
   because honest updates become noisy and small as the model converges. The ranking stays right, but
   it shows the signal is "relative agreement", not a calibrated probability.
10. **Federation gap.** Centralized MLP reaches 91.3 %; federated 86.2 %. The project does not try
    to close this (more rounds/epochs, learning-rate schedules, FedProx).
11. **Fixed hyperparameters** (α = 0.5, floor 0.05, boost 4, 8 rounds, 2 local epochs) were not tuned
    or ablated. *Fix:* ablate α, the floor and boost; try more rounds.
12. **The dashboard verdict bands (0.60 / 0.20) are hand-picked** from eight sample files, not
    calibrated (no ROC analysis, no false-positive rate). The trust score of an uploaded device
    also depends on its *size* and class balance, and on training for 2 epochs from the final model.

## Technical

1. **`flwr.simulation.run_simulation` is deprecated** in Flower 1.40 (prints a warning). It works
   today and the exact version is pinned in `requirements.lock.txt`; migrating to the `flwr run`
   project structure will eventually be required.
2. **Simulation only.** All nodes run in one process tree via Ray. There is no network, TLS,
   authentication, update signing or secure aggregation, so the dashboard text cannot claim
   cryptographic protection (and no longer does).
3. **Ray on Windows is slow to start**; a suite takes several minutes. One full-suite run produced
   no output for about 30 minutes and had to be stopped; the identical command then completed
   normally. The cause was not diagnosed, so if a run stalls, stop it, run `ray stop --force` and retry.
4. **Model and artefact fragility.** `scaler.joblib` is a pickle tied to the scikit-learn version
   that wrote it (it produced an `InconsistentVersionWarning` once). `results/*` binaries are
   committed, so every suite run shows as modified in git. *Fix:* store the scaler as JSON
   (mean/scale) and stop committing generated weights, or tag them with the version used.
5. **Sample device CSVs have no generating script**, so they cannot be regenerated or audited.
   *Fix:* add `data/generate_sample_devices.py`.
6. **Weak test coverage at the edges:** no automated test runs the Flower simulation or the
   Streamlit UI; both were checked manually. *Fix:* a short (1-round, 2-node) simulation smoke test and
   Streamlit's `AppTest` for the tabs.
7. **`dashboard/app.py` is still ~1,800 lines**, mostly CSS and HTML strings. Further splitting
   into per-tab modules and a stylesheet file would help maintenance.
8. **Clients are rebuilt every round** (data scaling and model construction), which is wasteful but
   harmless at this scale.
9. **`src/` is imported by path manipulation**, not as an installed package.
10. **Metrics-in-process:** `run_once` accumulates metrics through a closure shared with the
    strategy, because `run_simulation` no longer returns a history object.

## Security and product limitations (dashboard)

- Open sign-up, no e-mail verification, no password reset, no rate limiting, and error messages
  reveal whether an e-mail has an account. The demo credentials are printed on the login page.
- Users are stored in a local JSON file; the app is single-machine and not designed for concurrent
  production use.
- User-supplied text is HTML-escaped before display, but the app still renders HTML via
  `unsafe_allow_html` in many places and should be reviewed again if more user-controlled text is added.
- Uploaded files are processed in memory and not persisted; a very large upload can exhaust memory.
- The Device Analyzer's trust score requires labels and only makes sense for data in the training
  schema; missing features are zero-filled.
- The live feed replays held-out test rows; it is a demonstration, not monitoring of a real network.

## Prioritised next steps

**For a publishable result**
1. Multi-seed runs with confidence intervals (limitation 1).
2. A real dataset (limitation 2).
3. Non-IID experiments (limitation 3) and attacker-fraction sweep (limitation 6).
4. Comparison with robust aggregators (limitation 8) and ablations of α, τ and boost (limitation 11).
5. At least one adaptive attack (limitation 5).

**For the codebase**
1. Replace `run_simulation` with the current Flower entry point (technical 1).
2. JSON scaler and a version stamp on `results/` (technical 4).
3. Sample-device generator and simulation/UI smoke tests (technical 5–6).
4. Split `app.py` and move CSS to a file (technical 7).

**For the dashboard**
1. Calibrate verdict bands with held-out labelled devices and show a confidence interval.
2. Refuse to give a verdict when more than half of the features had to be imputed.
3. Proper authentication if it is ever exposed beyond a demo.
