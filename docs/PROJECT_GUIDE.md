# TrustFL — Complete Project Guide

This is the one document that explains the whole project: the idea, the data, every module and
function, how the pieces connect, how to run and extend it, and how to fix common problems. You
should be able to read it top to bottom and then work on any file in the repository.

**Contents**

1. [The project in plain language](#1-the-project-in-plain-language)
2. [Glossary](#2-glossary)
3. [Architecture and data flow](#3-architecture-and-data-flow)
4. [Repository layout](#4-repository-layout)
5. [The data](#5-the-data)
6. [The experiment engine (`src/`)](#6-the-experiment-engine-src)
7. [The dashboard (`dashboard/`)](#7-the-dashboard-dashboard)
8. [Result files and their formats](#8-result-files-and-their-formats)
9. [Tests](#9-tests)
10. [Configuration and dependencies](#10-configuration-and-dependencies)
11. [Running things — command reference](#11-running-things--command-reference)
12. [How to extend the project](#12-how-to-extend-the-project)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. The project in plain language

**The scenario.** Imagine four small companies that each run an IoT network (cameras, sensors,
smart meters). Each wants a machine-learning model that spots attacks in network traffic. None
wants to hand their raw traffic to the others. **Federated learning** solves this: each company
trains on its own data and sends only *model weights* to a server, which combines them into one
shared model.

**The problem.** The usual way of combining ("FedAvg") counts every company equally. If one company
is hacked — or deliberately lies — its poisoned weights drag the shared model down for everyone.

**The idea in this project.** Before combining, the server asks of each company: *"is your update
pointing the same way as everybody else's?"* Updates that disagree strongly with the group get a
low **trust** score and are given almost no say. Trust is remembered over rounds, so repeated bad
behaviour is punished and one-off glitches are forgiven.

**What the repository contains.**

- A **simulator** that runs this whole story on a laptop (four "companies" are four slices of one
  dataset) and compares plain FedAvg with the trust-weighted method, with and without an attacker.
- A **dashboard** that shows the results and lets you upload a device's traffic to see whether it
  looks like a healthy or a poisoned participant.
- **Tests and documentation.**

**What it is not.** It is not a real network deployment (no transport, encryption or
authentication) and it runs on *synthetic* data. See
[LIMITATIONS_AND_FUTURE_WORK.md](LIMITATIONS_AND_FUTURE_WORK.md).

## 2. Glossary

| Term | Meaning here |
|---|---|
| **Node / client** | One simulated IoT network. Holds one slice of the data and trains locally |
| **Server / strategy** | Combines the nodes' weights each round. Here: FedAvg or `TrustWeightedFedAvg` |
| **Round** | One cycle: send global weights → local training → aggregation → evaluation |
| **Weights** | The model's numbers (1,089 of them), exchanged as a list of six NumPy arrays |
| **Update (Δ)** | A node's new weights minus the global weights it started from |
| **FedAvg** | Federated averaging: weighted average of node weights by their example counts |
| **Cosine similarity** | How aligned two vectors' directions are: 1 = same, 0 = unrelated, −1 = opposite |
| **Consensus direction** | Average of all nodes' unit-length updates; the "group direction" |
| **Trust score** | Smoothed (EMA) cosine similarity of a node to the consensus, in [0, 1] |
| **EMA** | Exponential moving average: `new = α·current + (1−α)·previous` |
| **Trust floor** | Minimum trust (0.05) used in weighting so nobody gets exactly zero |
| **IID / non-IID** | Data split evenly at random / skewed so nodes see different kinds of traffic |
| **Label flipping** | Poisoning by swapping 0↔1 in the labels |
| **Boost factor** | Multiplier an attacker applies to its update to amplify its influence |
| **Model replacement** | An attack that scales an update so large it overwrites the global model |
| **Byzantine** | A participant that may behave arbitrarily badly (malicious or faulty) |
| **Flower (flwr)** | The federated-learning framework used to simulate nodes and the server |
| **Ray** | The engine Flower uses to run simulated nodes as parallel processes |
| **Suite** | The four-run comparison produced by `run_experiment.py --suite` |
| **Global test set** | 15 % of the data held out; never used for training by anyone |

## 3. Architecture and data flow

```
                    data/raw/synthetic_iot_traffic.csv  (60,000 flows)
                                   │  src/common.py: load → stratified split
                    ┌──────────────┴───────────────┐
              train pool 51,000                 test set 9,000  ──────────────┐
                    │ fit scaler (train only)                                  │
          partition_iid → 4 slices of 12,750                                   │
                    │ (optional) poison_node() on node 0                       │
        ┌───────────┼───────────┬───────────┐                                  │
      node 0      node 1      node 2      node 3        ← fl_client.NodeClient  │
        │ local training (MLP, 2 epochs); node 0 may boost its update          │
        └───────────┴─────┬─────┴───────────┘                                  │
                          ▼  weights + n_examples + node_id                    │
            strategy.aggregate_fit  (FedAvg  or  TrustWeightedFedAvg)          │
                          ▼  new global weights                                │
              evaluate_fn  ◄────────────────────────────────────────────────────┘
                          ▼  accuracy / F1 / precision / recall per round
   results/comparison.json   results/global_model_weights.npz   results/scaler.joblib
   results/reference_consensus.npy        results/baseline_results.json
                          │
                          ▼
                dashboard/app.py  (reads results, scores uploaded CSVs, live feed)
```

**Two phases, one handoff.** Phase 1 (offline) is `baseline_model.py` and `run_experiment.py`: they
train and write files into `results/`. Phase 2 (online) is the Streamlit dashboard, which only
*reads* `results/` and the sample CSVs. The dashboard never trains the global model.

## 4. Repository layout

```
trust-flow-dashboard/
├── README.md                      project overview and quick start
├── CONTRIBUTING.md                workflow, interface contract, style
├── TEAM_WORKFLOW.md               original two-person work split (historical)
├── requirements.txt               dependencies (minimum versions)
├── requirements.lock.txt          exact versions the results were produced with
├── .streamlit/config.toml         dashboard theme and settings
├── .gitignore
│
├── src/                           ── experiment engine ──
│   ├── common.py                  data loading, scaling, partitioning, poisoning
│   ├── model_utils.py             MLP construction and weight get/set
│   ├── fl_client.py               NodeClient: one simulated node
│   ├── trust_strategy.py          TrustWeightedFedAvg (the core contribution)
│   ├── run_experiment.py          orchestrator and command-line interface
│   └── baseline_model.py          centralized baselines (Random Forest + MLP)
│
├── dashboard/                     ── Streamlit app ──
│   ├── app.py                     the app: UI, CSS, tabs
│   ├── auth.py                    users, bcrypt passwords
│   ├── data_audit.py              column alignment and telemetry bug tracer
│   ├── scoring.py                 classify an uploaded device and compute its trust score
│   ├── assets/                    background and illustration images
│   └── users.json                 created on first run; git-ignored
│
├── data/
│   ├── generate_synthetic_data.py creates data/raw/synthetic_iot_traffic.csv
│   ├── partition_data.py          writes IID / non-IID node CSVs (optional utility)
│   ├── raw/synthetic_iot_traffic.csv
│   ├── nodes/                     output folder of partition_data.py
│   └── sample_device_*.csv        eight example device captures for the dashboard
│
├── results/                       ── generated outputs (committed for convenience) ──
│   ├── baseline_results.json
│   ├── comparison.json
│   ├── global_model_weights.npz
│   ├── reference_consensus.npy
│   └── scaler.joblib
│
├── tests/                         test_core.py, test_dashboard_helpers.py
└── docs/                          documentation, images/
```

## 5. The data

### 5.1 Schema

Every row is one network flow. The generator mimics the *structure* of public IoT-IDS datasets
(TON_IoT / CIC-IoT) using the same column names, but the values are **synthetic**.

| Column | Type | Meaning |
|---|---|---|
| `duration` | float | Flow duration (seconds) |
| `src_bytes`, `dst_bytes` | float | Payload bytes sent by source / destination |
| `src_pkts`, `dst_pkts` | int | Packets sent by source / destination |
| `src_ip_bytes`, `dst_ip_bytes` | float | IP-level bytes from source / destination |
| `missed_bytes` | float | Bytes missed by the capture |
| `conn_state_enc` | int 0–5 | Encoded connection state |
| `proto_enc` | int 0–2 | Encoded protocol (TCP / UDP / ICMP) |
| `service_enc` | int 0–7 | Encoded service (HTTP, DNS, SSL, …) |
| `http_request_body_len`, `http_response_body_len` | float | HTTP body sizes |
| `dns_qclass` | float | DNS query class |
| `ssl_version_enc` | float | Encoded TLS version |
| `weird_notice_enc` | float | Encoded "weird"/notice events |
| **`label`** | int | **0 = normal, 1 = attack** — the target the model predicts |
| `attack_type` | str | `normal`, or one of `scanning, dos, ddos, backdoor, injection, password, xss` |

The 16 feature names are in `common.FEATURE_COLUMNS`. The model only sees these 16 columns;
`label` is the target and `attack_type` is used only by the partitioner and the dashboard's audit.

### 5.2 How the synthetic data is made (`data/generate_synthetic_data.py`)

`generate()` uses scikit-learn's `make_classification` (60,000 rows, 16 features, 10 informative,
3 redundant, 3 clusters per class, 65/35 class weights, 2 % label noise, separation 1.2, seed 42).
It then rescales some columns to look like flow statistics (e.g. durations ×5, bytes ×500),
discretises `proto_enc`, `service_enc` and `conn_state_enc`, assigns a random `attack_type` to every
attack row, shuffles, and writes `data/raw/synthetic_iot_traffic.csv`. Result: 35.3 % attacks.

> **Important:** `attack_type` is assigned *independently* of the features. The seven attack types
> are therefore not distinguishable from each other by the model — they exist only as labels for
> partitioning and display.

### 5.3 Splits

`common.get_global_test_split()` produces a stratified 85 / 15 split → **51,000 train, 9,000 test**.
The test set is never seen by any node. The training pool is split across nodes (§6.1).

### 5.4 Sample device files (`data/sample_device_*.csv`)

Used by the dashboard's Device Analyzer. They use the same schema as the synthetic dataset. No
script in this repository generates them (they were prepared separately and committed), so treat
them as fixed example inputs. "Measured trust" is the dashboard score with the model currently saved in `results/`
(it changes if you re-run the suite).

| File | Rows | Attack share | What it represents | Measured trust |
|---|---|---|---|---|
| `sample_device_trusted.csv` | 10,000 | 35 % | Healthy device | 0.71 |
| `sample_device_clean.csv` | 8,000 | 35 % | Benchmark clean slice | 0.67 |
| `sample_device_recon_probe.csv` | 7,000 | 36 % | "Low-and-slow" recon | 0.22 |
| `sample_device_ddos_surge.csv` | 5,000 | 70 % | Volumetric surge, many attacks | 0.26 |
| `sample_device_corrupted_telemetry.csv` | 3,020 | 34 % | Sensor bugs: negative durations, NaNs, duplicates, underflow | 0.35 |
| `sample_device_poisoned.csv` | 3,000 | 64 % | Benchmark poisoned slice (labels corrupted) | 0.00 |
| `sample_device_adversary.csv` | 4,000 | 57 % | Label-inversion poisoning | 0.00 |
| `sample_device_unlabeled.csv` | 2,500 | — | Raw capture with **no label/attack_type columns** | not computable |

Because the synthetic attack types carry no feature signal, "recon" and "ddos" scenarios are
scenario *names* for different label/volume mixes, not realistic traffic differences.

## 6. The experiment engine (`src/`)

All modules in `src/` import each other with plain `import common` / `import model_utils`, so
scripts must be run so that `src/` is on the path — `python src/run_experiment.py` does that
automatically; tests and the dashboard add it with `sys.path.insert`.

### `common.py`

Shared data utilities.

**Constants:** `ROOT` (repo root), `RAW_DATA_PATH` (`data/raw/synthetic_iot_traffic.csv`),
`NODES_DIR` (`data/nodes`, created at import), `FEATURE_COLUMNS` (the 16 features),
`LABEL_COLUMN = "label"`, `RANDOM_STATE = 42`.

| Function | What it does |
|---|---|
| `load_dataset(path=RAW_DATA_PATH)` | Reads the CSV. If the default file is missing it runs the generator first; if it is still missing raises `FileNotFoundError` with a helpful message. Generator errors are **not** swallowed. To use a real dataset, point `path` at a CSV with the same column names |
| `get_global_test_split(test_size=0.15, random_state=42)` | Loads the data and returns `(train_df, test_df)`, stratified on `label` |
| `fit_scaler(train_df)` | Fits a `StandardScaler` on the 16 feature columns of the **training** data only (never the test set — avoids leakage) |
| `to_xy(df, scaler)` | Returns `(X, y)`: scaled feature matrix and integer labels |
| `_split_dataframe(df, n_chunks)` | Splits a frame into `n_chunks` non-empty, near-equal pieces; validates `1 ≤ n_chunks ≤ len(df)` |
| `partition_iid(df, n_nodes, random_state)` | Shuffles then splits into equal chunks — every node sees the same distribution |
| `partition_non_iid(df, n_nodes, primary_share=0.7, random_state)` | Skewed split: attack types (except `normal`) are shuffled and divided among nodes; node *i* gets `primary_share` of its assigned types' rows, plus `(1−primary_share)·0.3` of all other attack rows (sampled with a node-specific seed), plus 1/*n* of the normal rows. Rows can appear on more than one node, and some are unused |
| `save_node_partitions(nodes, prefix)` | Writes each node's frame to `data/nodes/<prefix>_<i>.csv` and returns the paths |
| `poison_node(node_df, flip_fraction, random_state)` | Returns a **copy** with `flip_fraction` of rows' labels flipped (`1 − label`), chosen without replacement. Raises `ValueError` for a fraction outside [0, 1] or a missing label column |

### `model_utils.py`

Wraps scikit-learn's `MLPClassifier` so its weights behave like plain arrays.

| Function | What it does |
|---|---|
| `build_model(random_state=42)` | Returns an `MLPClassifier`: hidden layers (32, 16), ReLU, Adam, lr 1e-3, `alpha=1e-4`, `max_iter=1`, `warm_start=True` (so training continues between calls) |
| `init_architecture(model, X_sample, y_sample)` | scikit-learn only allocates weight arrays on the first fit. This calls `partial_fit` once on a tiny sample (with `classes=[0,1]`) purely to create `coefs_` / `intercepts_` of the right shape |
| `get_weights(model)` | Returns copies of `coefs_` then `intercepts_` as a list of six arrays: shapes `(16,32) (32,16) (16,1) (32,) (16,) (1,)` |
| `set_weights(model, weights)` | Overwrites `coefs_` / `intercepts_` with copies of `weights` (same order). The split point is the number of layers, fixed by the architecture |
| `local_train(model, X, y, epochs=1)` | Calls `partial_fit` `epochs` times — each call is one pass over the data in mini-batches of 200 |
| `flatten_weights(weights)` | Concatenates all arrays into one 1-D vector (length 1,089) — used for cosine similarity |

### `fl_client.py`

**`class NodeClient(fl.client.NumPyClient)`** — one simulated node.

- `__init__(node_id, node_df, scaler, boost_factor=1.0)`: scales the node's features with the shared
  scaler; uses the first 85 % of rows for training and the last 15 % as a local validation slice
  (if the slice would be empty, validation reuses the training data); builds a local model and
  initialises its architecture. `boost_factor > 1` marks the node as an attacker.
- `get_parameters(config)`: returns the local model's weights.
- `fit(parameters, config)`: loads the global weights, trains for `config["local_epochs"]`
  (default 2), gets the new weights. **If `boost_factor != 1`** it replaces them with
  `g + boost·(w_local − g)` where `g` is the received global weights — the attacker amplifies its
  *update*, not its raw weights. Returns `(weights, len(X_train), {"node_id": node_id})`.
- `evaluate(parameters, config)`: loss, accuracy, F1 on the local validation slice. Defined for
  Flower's API completeness; the experiments set `fraction_evaluate=0` and evaluate centrally.

The client is re-created every round by the simulation runtime, so it holds no state between rounds.

### `trust_strategy.py`

**`_cosine_sim(a, b)`**: cosine similarity of two vectors, 0.0 if either has zero length.

**`class TrustWeightedFedAvg(fl.server.strategy.FedAvg)`** — the core contribution.

- `__init__(*args, trust_ema_alpha=0.5, trust_floor=0.05, **kwargs)` — passes everything else to
  `FedAvg`. State: `trust_history` (node → current EMA), `trust_log` (list of per-round records for
  plotting), `current_weights` (the previous global weights, taken from `initial_parameters`),
  `last_consensus` (latest consensus direction).
- `aggregate_fit(server_round, results, failures)` — called by Flower once per round. In order:
  1. Decode each client's weights and flatten them. A duplicate `node_id` raises `ValueError`; a
     client with NaN/inf weights is dropped and logged with trust 0.
  2. Compute each update `Δ = flat(weights) − flat(previous global)`.
  3. Normalise every `Δ` to unit length; consensus `c` = mean of those unit vectors. (Normalising
     first stops a large attacker update from hijacking the consensus.)
  4. For each node: `sim = cos(Δ, c)`, clip at 0, update the EMA
     `R = α·sim + (1−α)·R_prev` (first round uses `sim` itself), append a record to `trust_log`.
  5. Weight = `max(R, floor) · n_examples`, normalised to sum to 1 (uniform if the sum is 0).
  6. New global weights = weighted average of the **raw weights**. Store as `current_weights`.
  7. Return the parameters and a metrics dict `{"trust/<node_id>": R}`.

A step-by-step derivation with real numbers is in [METHODOLOGY.md §5](METHODOLOGY.md#5-trust-weighted-aggregation-trustweightedfedavg).

### `run_experiment.py`

The orchestrator. Everything you run goes through here.

| Function | What it does |
|---|---|
| `build_node_partitions(train_df, n_nodes, partition_type, poison_node_idx, poison_fraction, seed)` | Validates arguments, partitions (`"iid"` or non-IID), resets indices and applies `poison_node` to the chosen node |
| `make_client_fn(node_dfs, scaler, poison_node_idx, boost_factor)` | Returns the `client_fn(context)` Flower calls to build a client. It reads `context.node_config["partition-id"]` (0…n−1, assigned by the simulation) to pick the node's data slice; the attacker index gets `boost_factor`. *This is the fix for the earlier bug where random node ids caused duplicated/missing partitions* |
| `make_evaluate_fn(X_test, y_test, template_model)` | Returns `evaluate_fn(round, weights, config)` → `(log-loss, {accuracy, f1, precision, recall})` on the global test set |
| `run_once(strategy_name, partition_type, n_nodes, n_rounds, local_epochs, poison_node_idx, poison_fraction, boost_factor, seed, verbose, return_strategy)` | One full experiment (below). Returns a result dict, or `(result, strategy, scaler)` |
| `save_dashboard_artifacts(strategy, scaler)` | Writes `global_model_weights.npz`, `scaler.joblib`, and `reference_consensus.npy` to `results/` |
| `run_suite(n_nodes=4, n_rounds=10, local_epochs=2, poison_fraction=1.0, boost_factor=4.0, seed=42)` | Runs the four standard configurations, saves `comparison.json`, prints a summary. The `clean_trust` run also saves the dashboard artefacts. (The CLI passes `--rounds` default 8) |
| `main()` | Parses the command line |

**What `run_once` does, step by step**

1. Load and split data; fit the scaler on the training pool; scale the test set.
2. Partition the training pool across nodes; optionally poison one.
3. Build a template MLP and take its initial weights — the same starting point for every strategy.
4. Create the strategy (`FedAvg` or `TrustWeightedFedAvg`) with: all nodes every round
   (`fraction_fit=1.0`), no client-side evaluation, `local_epochs` sent to clients each round, a
   central `evaluate_fn` that records metrics into a shared dict, and the initial parameters.
5. Wrap clients in a Flower `ClientApp` and the strategy in a `ServerApp`, then call
   `run_simulation(...)`, which starts Ray, spawns `n_nodes` simulated nodes and runs the rounds.
6. Collect per-round metrics (index 0 is the untrained model), keep the last as `final_metrics`,
   attach the `trust_log` for the trust strategy, and raise a clear error if no metrics were recorded.

**Command-line flags**

| Flag | Default | Meaning |
|---|---|---|
| `--strategy {fedavg,trust}` | `fedavg` | Aggregation rule |
| `--partition {iid,noniid}` | `iid` | Data split |
| `--n-nodes N` | 4 | Number of simulated nodes |
| `--rounds N` | 8 | Federated rounds |
| `--local-epochs N` | 2 | Local training passes per round |
| `--poison-node I` | none | 0-based index of the attacker (omit for a clean run) |
| `--poison-fraction F` | 1.0 | Fraction of that node's labels to flip |
| `--boost-factor B` | 4.0 | Update multiplier for the attacker |
| `--suite` | off | Run all four standard configurations and save `comparison.json` |

Note: `--boost-factor` only matters for the node chosen with `--poison-node`.

### `baseline_model.py`

`main()` trains two **centralized** reference models on the whole training pool and evaluates them
on the same test set: a 200-tree Random Forest, and the same MLP used in the federated runs for 16
epochs (`MLP_EPOCHS`, equal to 8 rounds × 2 local epochs). Writes `results/baseline_results.json`:
the Random Forest under the top-level `accuracy` / `f1` / `report` keys and the MLP under `mlp`.

### `data/partition_data.py` (optional utility)

`report(nodes, label)` prints per-node size, attack rate and dominant attack types. `main()` writes
the held-out test set and IID / non-IID node CSVs under `data/nodes/`. The experiments do **not**
read these files — they partition in memory — so this script is for inspection and for external tools.

## 7. The dashboard (`dashboard/`)

Run with `streamlit run dashboard/app.py`. Streamlit re-executes `app.py` from top to bottom on
every interaction; the structure below is that top-to-bottom order. A user-facing description is in
[DASHBOARD_GUIDE.md](DASHBOARD_GUIDE.md).

### 7.1 `app.py` — order of execution

| Section | What happens |
|---|---|
| Imports and paths | Adds `src/` and `dashboard/` to `sys.path`; imports `auth`, `data_audit`, `scoring` |
| `st.set_page_config` | Title, icon, wide layout |
| `load_bg_img_b64()` + `CUSTOM_CSS` | Base64-encodes the background image and injects a large stylesheet (fonts, cards `.tf-card`, badges `.tf-badge`, scanner animation, sidebar, inputs) |
| Readability layer | A second small stylesheet: frosted content panel, solid tab bar, metric tiles, `.tf-statrow` / `.tf-stat` stat tiles, responsive rule |
| Auth | Initialises `st.session_state.auth_user`, loads users, defines `render_auth_portal()` (sign-in and create-account forms); if nobody is signed in it renders the portal and calls `st.stop()` |
| Header + sidebar | Slim header with node-count and strategy badges; sidebar with the signed-in user, a **"Latest experiment"** summary computed by `summarize_defense`, and Sign Out |
| Data loading | `load_json` (cached by file path **and modification time**) loads `baseline_results.json` and `comparison.json` |
| `summarize_defense(comparison)` | Derives node count, attacker id, final-round attacker/honest mean trust and accuracy gain from the saved results |
| `md_html(html)` | Renders an HTML snippet without Markdown breaking it (see below) |
| Tabs | Four tabs described in §7.2 |

**Why `md_html` exists.** `st.markdown` treats a blank line as the end of an HTML block and a
4-space indent as a code block, so indented multi-line HTML turned into visible code. `md_html`
strips indentation and blank lines before rendering.

**Security notes.** Anything that originates from a user (display name, e-mail, uploaded file name,
uploaded column names that end up in audit findings) is passed through `html.escape` before being
placed in HTML. Passwords never leave `auth.py` unhashed.

**Session-state keys:** `auth_user` (signed-in user dict or `None`), `active_scenario_key` (selected
preset), `custom_csv_df` / `custom_csv_name` (the uploaded file), `live_feed_log`,
`live_feed_running`, `force_attack_ticks` (live feed state).

**Caches:** `_load_json_cached` (`st.cache_data`) and `get_demo_model` (`st.cache_resource`, loads
the federated model once per server process).

### 7.2 The tabs

**Device Analyzer.** Pick a preset (`PRESET_SCENARIOS` maps keys to sample CSVs) or upload a CSV.
The pipeline for the chosen frame:

1. `normalize_and_align_features` → standard column names, all 16 features present.
2. `trace_file_bugs_and_anomalies` → findings, a health score and a sanitised copy.
3. Four stat tiles (health, flows, critical bugs, remediation) and an expandable findings list.
4. `score_device` → per-flow predictions and, if labels exist, the trust score.
5. A gauge (0–1), a verdict badge, the attack share and the first 50 classified flows.

**Verdict thresholds** (`app.py`): trust **> 0.60 → TRUSTED**, **> 0.20 → SUSPICIOUS**, otherwise
**UNTRUSTED / ADVERSARY**; no labels → **LABELS MISSING**. The thresholds are fixed numbers chosen
by inspection of the sample files, not calibrated statistically (see limitations).

**Live Traffic Feed.** Loads the saved federated model (`get_demo_model`) and the held-out test
set. A Streamlit *fragment* refreshes every 2 s while running: it samples one flow (a normal one,
or an attack one while `force_attack_ticks > 0`), classifies it, prepends it to the log (newest 25
kept) and shows recent-flow / attacks-blocked / attack-rate tiles with the table. "Simulate Attack
Burst" forces the next 5 samples to be attacks.

**Model Performance.** KPI tiles (centralized MLP, FedAvg clean, FedAvg attacked, Trust attacked),
a grouped bar chart of accuracy/F1 for the four runs, a per-round accuracy line for any run, and
the **per-node trust trajectory** (attacker highlighted using the real attacker id from the results).
If the results file has duplicate node rows (old output) it warns and de-duplicates.

**How It Works.** Static explanation: invariants tiles, the three-step story, a FedAvg vs TrustFL
comparison filled from the real results, the 4-stage pipeline and links. No hard-coded accuracies.

### 7.3 `auth.py`

| Function | What it does |
|---|---|
| `hash_password(password)` | bcrypt with a fresh random salt; raises `ValueError` above 72 bytes (bcrypt's limit) |
| `verify_password(password, stored_hash)` | bcrypt check; returns `False` (never raises) for wrong, over-long or non-bcrypt (legacy) hashes |
| `load_users()` | Reads `dashboard/users.json`. If absent, creates it with the demo account `demo@trustfl.org`. If the file is corrupt it is renamed to `users.corrupt` (preserved) and a fresh file is created |
| `save_users(users)` | Atomic write: writes `users.tmp` then replaces `users.json`, so a crash cannot leave a half-written file |

Users are stored as `{email: {password_hash, name, created_at}}`. The file is git-ignored.

### 7.4 `data_audit.py`

**`ALIAS_MAP`** maps common column names to the model's: e.g. `flow_duration`/`dur` → `duration`;
`tot_fwd_pkts`/`packets_sent` → `src_pkts`; `tot_bwd_pkts`/`packets_rcvd` → `dst_pkts`;
`totlen_fwd_pkts`/`bytes_sent` → `src_bytes`; `totlen_bwd_pkts`/`bytes_rcvd` → `dst_bytes`;
`protocol`/`proto` → `proto_enc`; `service` → `service_enc`; `class`/`target`/`attack`/`is_attack` → `label`.

**`normalize_and_align_features(df)`** → `(aligned_df, imputed_cols)`:
1. Lower-cases and underscores header names, then renames via the alias map or exact feature names.
2. Adds every missing feature column filled with **0.0** and lists them in `imputed_cols`.
3. Coerces feature columns to numbers; unparseable cells become NaN (reported later as missing).
4. If `label` is text, maps `0, 0.0, normal, benign, safe, false, clean` → 0 and anything else → 1.

**`trace_file_bugs_and_anomalies(df, imputed_cols)`** → `{findings, health_score, sanitized_df}`.
Checks, in order:

| Check | Severity | Remediation applied to `sanitized_df` |
|---|---|---|
| Imputed (missing) feature columns | Info | already zero-filled |
| Duplicate rows | Warning | `drop_duplicates` |
| Missing / NaN cells | Critical if > 50 cells, else Warning | fill with column median (0 if all NaN) |
| Negative `duration` (clock skew) | Critical | clip to 0 |
| Negative byte/packet counters (underflow) | Critical | clip to 0 |
| `src_pkts` > 1,000,000 (flood/overflow) | Critical | flagged only |
| `attack_type` dos/ddos > 40 % of rows | Critical | flagged only |
| `attack_type` scanning > 15 % of rows | Warning | flagged only |

**Health score** = `max(100 − 25·critical − 10·warning − 5·info, 5)`.

### 7.5 `scoring.py`

**`score_device(df, scaler, global_weights, consensus, epochs=2)`** returns
`{preds, probs, attack_rate, trust_score}`:

1. Coerce features to numbers (bad cells → 0), scale with the **training** scaler.
2. Build a model with the saved global weights; predict every flow.
3. If a `label` column exists: coerce it (any positive value = attack, anything else = normal), make a
   second copy of the global model, train it for 2 epochs on the device's labelled flows, compute
   `Δ = new − global`, and set `trust = max(cos(Δ, consensus), 0)` — the same signal the
   aggregation strategy uses. Without labels `trust_score` is `None`.

## 8. Result files and their formats

| File | Producer | Format / contents |
|---|---|---|
| `results/comparison.json` | `run_suite` | `{tag: run_result}` for `clean_fedavg`, `fedavg_under_attack`, `clean_trust`, `trust_under_attack` |
| `results/baseline_results.json` | `baseline_model.py` | `{accuracy, f1, report, mlp:{accuracy, f1, epochs}}` |
| `results/global_model_weights.npz` | `save_dashboard_artifacts` | NumPy archive `arr_0 … arr_5`: the final global MLP weights (order of `get_weights`) |
| `results/reference_consensus.npy` | `save_dashboard_artifacts` | 1,089-vector: last-round consensus direction of the clean trust run |
| `results/scaler.joblib` | `save_dashboard_artifacts` | Pickled fitted `StandardScaler` (version-sensitive — regenerate after upgrading scikit-learn) |

**`run_result` structure** (value of each tag in `comparison.json`):

```json
{
  "strategy": "trust", "partition": "iid", "n_nodes": 4, "n_rounds": 8,
  "poison_node": 0, "poison_fraction": 1.0,
  "final_metrics": {"accuracy": 0.8628, "f1": 0.7956, "precision": 0.8388, "recall": 0.7567},
  "metrics_by_round": {"accuracy": [0.374, 0.727, …], "f1": […], "precision": […], "recall": […]},
  "trust_log": [{"round": 1, "node_id": "0", "trust": 0.0, "raw_similarity": -0.217}, …]
}
```

`metrics_by_round` has `n_rounds + 1` entries (index 0 = untrained model). `trust_log` appears only
for the trust strategy and has one entry per node per round. `raw_similarity` is the unclipped cosine
similarity; `trust` is the EMA of its clipped value.

## 9. Tests

Run `python -m pytest tests` (about 10 seconds, 18 tests). They need no browser and no Ray.

**`tests/test_core.py`**

| Test | Checks |
|---|---|
| `test_iid_partition_covers_all_rows` | IID split keeps every row exactly once |
| `test_non_iid_nodes_differ` | Non-IID nodes get distinct attack-type mixes |
| `test_poison_flips_expected_fraction` | `poison_node` flips the requested fraction; 0.0 changes nothing |
| `test_poison_rejects_bad_fraction` | A fraction > 1 raises `ValueError` |
| `test_get_set_weights_roundtrip` | `set_weights` then `get_weights` returns what was set |
| `test_trust_strategy_downweights_boosted_attacker` | A 10× opposite update gets lower trust and the aggregate stays on the honest side |
| `test_trust_strategy_rejects_duplicate_node_ids` | Duplicate ids raise |
| `test_trust_strategy_excludes_nan_client_and_logs_zero_trust` | A NaN client cannot poison the average and is logged with trust 0 |
| `test_trust_strategy_handles_zero_examples_and_no_initial_parameters` | No division by zero; round 1 without a baseline gives full trust |

**`tests/test_dashboard_helpers.py`**: password hashing (unique salts, verify, legacy hash rejected,
72-byte limit), demo-account creation, corrupt-file preservation, alias mapping and imputation,
bug tracing and sanitising, text-in-numeric-column handling, and `score_device` with odd labels and
text cells (requires the files in `results/`).

**Not covered by automated tests:** the Flower/Ray simulation end to end and the Streamlit UI —
both were verified by hand (a real suite run and headless-browser checks of every tab).

## 10. Configuration and dependencies

**`requirements.txt`** lists minimum versions: `streamlit`, `pandas`, `numpy`, `plotly`,
`scikit-learn`, `matplotlib`, `joblib`, `flwr[simulation]` (≥ 1.20, includes Ray), `bcrypt`, `pytest`.
**`requirements.lock.txt`** pins the exact versions used for the reported results
(flwr 1.40.0, ray 2.55.1, streamlit 1.66.0, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.4,
plotly 7.1.0, bcrypt 5.0.0).

**`.streamlit/config.toml`**: dark theme colours, usage statistics off, static serving off.

**`.gitignore`**: ignores environments, caches, and `dashboard/users.json`; whitelists the synthetic
CSV, `results/scaler.joblib` and the `.gitkeep` placeholders.

Flower's `run_simulation` prints a deprecation warning in version 1.40; it works, but see
[limitations](LIMITATIONS_AND_FUTURE_WORK.md#technical).

## 11. Running things — command reference

```powershell
# environment
python -m venv .venv ; .venv\Scripts\Activate.ps1 ; pip install -r requirements.txt

# data (only needed if data/raw is missing)
python data/generate_synthetic_data.py
python data/partition_data.py                       # optional inspection utility

# training / experiments
python src/baseline_model.py
python src/run_experiment.py --strategy fedavg --partition iid
python src/run_experiment.py --strategy trust  --partition iid --poison-node 0 --poison-fraction 1.0
python src/run_experiment.py --strategy trust  --partition noniid --n-nodes 4 --rounds 10
python src/run_experiment.py --suite --rounds 8     # four runs + dashboard artefacts (~5–10 min)

# app and tests
streamlit run dashboard/app.py
python -m pytest tests
```

Timing (laptop, CPU only): the suite takes several minutes, mostly Ray start-up and per-round
client set-up; a 2-round single run takes about 30 seconds.

## 12. How to extend the project

**Use a real dataset (TON_IoT, CIC-IoT).** Prepare a CSV with a binary `label` column and the 16
feature columns (or change `FEATURE_COLUMNS` and the generator's names together), put it at
`data/raw/` and point `common.RAW_DATA_PATH` at it. Re-run `baseline_model.py` and
`run_experiment.py --suite`. Re-check the dashboard's verdict thresholds, and extend `ALIAS_MAP` for
your column names.

**Add a new aggregation strategy.** Subclass `fl.server.strategy.FedAvg`, implement
`aggregate_fit(server_round, results, failures)` returning `(Parameters, metrics)`, add a name to the
`--strategy` choices and to `run_once`'s `if/else`. If it keeps a `trust_log`, the dashboard charts
pick it up automatically.

**Add a new attack.** Data-level attacks belong in `common.py` next to `poison_node` and are applied
in `build_node_partitions`; update-level attacks belong in `NodeClient.fit` next to the boost. Add a
configuration to `run_suite` and a test.

**Add a new data-audit check.** Append a finding in `trace_file_bugs_and_anomalies` (category,
severity, bug, details, action), apply the fix to `sanitized_df`, and add a test in
`tests/test_dashboard_helpers.py`.

**Multiple seeds.** Wrap `run_suite(seed=…)` in a loop, save each result under a different name, and
aggregate mean ± std for the paper tables.

**Change the model.** Edit `build_model` / `HIDDEN_LAYER_SIZES` in `model_utils.py`. Everything else
works with any model exposing weights as a list of arrays; the saved artefacts in `results/` must be
regenerated.

## 13. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ERROR: Exit Code: 701 … Extra dependencies required for simulation are missing` | Flower is installed but **Ray is not** in the Python you ran. Typical on a fresh clone (the `.venv` is not in git) or when `pip install flwr` was used without the extra. Create/activate a venv and run `pip install -r requirements.txt` (it contains `flwr[simulation]`), then check `python -c "import ray, flwr; print(ray.__version__, flwr.__version__)"`. Use Python 3.12; very new Python versions may lack Ray wheels |
| Warnings `run_simulation is deprecated` and `Ray support on Windows is experimental` | Harmless. Flower prefers WSL2 for simulations on Windows, but the suite also runs natively (just slowly to start) |
| `pip install` takes very long | `flwr[simulation]` installs Ray and many dependencies; wait for it |
| First suite run seems stuck | Ray start-up on Windows is slow (it spawns many processes). A 2-round suite took ≈ 2 minutes. If nothing finishes after ~15 minutes, stop it, run `ray stop --force`, and retry |
| Many stray `python` processes after a run | Leftover Ray workers: `ray stop --force` |
| Dashboard says "Model artifacts not found" / "No experiment results" | Run `python src/run_experiment.py --suite` |
| Dashboard shows old numbers after a new run | Results are cached by file modification time; reload the page. If a helper module change is not picked up, restart `streamlit` |
| `ImportError: cannot import name …` from `auth`/`scoring` after editing | Streamlit kept an old module in memory — restart the server |
| Cannot log in with an old account | Accounts created before the bcrypt change used SHA-256 and no longer work; sign up again |
| `InconsistentVersionWarning` loading the scaler | `scaler.joblib` was written by a different scikit-learn version; regenerate with `--suite` |
| `ValueError: Duplicate node_id …` | A client mapped to a partition twice; ensure `num_supernodes` equals `--n-nodes` and use partition ids, not random ids |
| `Simulation finished without any evaluation metrics` | The Flower/Ray run failed; read the log above the error |
| Port 8501 already in use | `streamlit run dashboard/app.py --server.port 8502` |
| Password rejected as too long | bcrypt supports at most 72 bytes; choose a shorter password |
| Windows console shows `?` for symbols | Console code page; set `PYTHONUTF8=1` |
