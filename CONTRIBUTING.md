# Contributing

Thanks for working on TrustFL. This guide covers setup, the rules that keep the code consistent,
the interface contract between modules, and how to submit changes. Background reading:
[docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md).

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt          # or requirements.lock.txt for the exact tested versions
python -m pytest tests                   # should report 18 passed
```

## Workflow

1. Branch from `main`: `git checkout -b <your-name>/<topic>`.
2. Make a focused change. Add or update a test for any behaviour you change.
3. Run `python -m pytest tests` and, if you touched `src/`, a quick experiment
   (`python src/run_experiment.py --strategy trust --rounds 2 --poison-node 0`).
4. If you changed training or aggregation, re-run `python src/run_experiment.py --suite` and
   include the new accuracy/F1 numbers in your pull request. Regenerated files in `results/` change
   the dashboard, so mention them.
5. Open a pull request into `main`. Another person reviews before merging — results that go into a
   paper deserve two sets of eyes.
6. Rebase on `main` regularly (`git fetch origin && git rebase origin/main`).

Commit messages: a short imperative summary line, then a body that says *why*.

## Interface contract

These interfaces are shared between modules; change them only with a heads-up to the other
contributors and matching test and doc updates.

| Interface | Contract |
|---|---|
| `model_utils.get_weights(model)` / `set_weights(model, weights)` | Weights travel as a **list of NumPy arrays** (coefs then intercepts); round-trip must be lossless |
| `fl_client.NodeClient(node_id, node_df, scaler, boost_factor=1.0)` | One client per simulated node. `fit()` returns `(weights, num_examples, {"node_id": str})` |
| Strategy classes | Subclass `fl.server.strategy.FedAvg`; implement `aggregate_fit(server_round, results, failures) -> (Parameters, metrics)`. Each client's `fit` metrics contain a **unique** `node_id` |
| Trust strategies | May expose `trust_log` (list of `{round, node_id, trust, raw_similarity}`), `current_weights`, `last_consensus` — the dashboard and `save_dashboard_artifacts` read them |
| `results/comparison.json` | Schema in [PROJECT_GUIDE §8](docs/PROJECT_GUIDE.md#8-result-files-and-their-formats); the dashboard depends on the four tag names |
| `common.FEATURE_COLUMNS` / `LABEL_COLUMN` | The model's input schema; also used by the dashboard's audit and scoring |

## Code style

- Python 3.12, type hints on public functions, docstrings in the existing reST style
  (`:param x:` / `:returns:`).
- Match the surrounding code; keep comments for *why*, not *what*.
- Lint before pushing: `pip install ruff` then `ruff check src dashboard data tests` — the repo
  passes with no warnings.
- Never put secrets in the repo. `dashboard/users.json` is git-ignored; keep it that way.
- Anything user-supplied that is shown through `unsafe_allow_html` must be wrapped in `html.escape`.

## Adding things

Step-by-step recipes (new strategy, attack, audit check, dataset, seeds) are in
[PROJECT_GUIDE §12](docs/PROJECT_GUIDE.md#12-how-to-extend-the-project).

## Reporting problems

Open an issue with: what you ran (command), what you expected, what happened (full error text), your
Python version, and the output of `pip freeze | findstr "flwr ray streamlit scikit-learn pandas"`.
Common problems and fixes are listed in
[PROJECT_GUIDE §13](docs/PROJECT_GUIDE.md#13-troubleshooting).

## Project history

The original two-person work division (federated core vs trust and robustness) is kept in
[TEAM_WORKFLOW.md](TEAM_WORKFLOW.md) for reference.
