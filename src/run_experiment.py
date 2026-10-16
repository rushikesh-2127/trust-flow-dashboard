"""
run_experiment.py

Phase 3 + 4 + 5 orchestrator. This is the script you actually run.

Examples
--------
Clean run, plain FedAvg, IID partition (control condition):
    python src/run_experiment.py --strategy fedavg --partition iid

Clean run, trust-weighted, IID (sanity check -- should be close to FedAvg):
    python src/run_experiment.py --strategy trust --partition iid

Poisoning attack, plain FedAvg (expect degraded accuracy):
    python src/run_experiment.py --strategy fedavg --partition iid --poison-node 0 --poison-fraction 0.8

Poisoning attack, trust-weighted (expect accuracy to hold up -- THE key result):
    python src/run_experiment.py --strategy trust --partition iid --poison-node 0 --poison-fraction 0.8

Run the full comparison suite in one go and save results/comparison.json:
    python src/run_experiment.py --suite
"""

import argparse
import json
from pathlib import Path

import numpy as np
# pyrefly: ignore [missing-import]
import flwr as fl
# pyrefly: ignore [missing-import]
from flwr.app import Context
# pyrefly: ignore [missing-import]
from flwr.clientapp import ClientApp
# pyrefly: ignore [missing-import]
from flwr.server import ServerConfig, ServerAppComponents
# pyrefly: ignore [missing-import]
from flwr.serverapp import ServerApp
# pyrefly: ignore [missing-import]
from flwr.simulation import run_simulation
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss

from common import (
    get_global_test_split, fit_scaler, to_xy,
    partition_iid, partition_non_iid, poison_node, RANDOM_STATE,
)
import model_utils as mu
from fl_client import NodeClient
from trust_strategy import TrustWeightedFedAvg

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def build_node_partitions(train_df, n_nodes, partition_type, poison_node_idx, poison_fraction, seed=RANDOM_STATE):
    """Partition *train_df* across *n_nodes* virtual nodes and optionally poison one.

    :param train_df: Full training DataFrame before splitting.
    :param n_nodes: Number of federated nodes to create.
    :param partition_type: ``"iid"`` for equal random splits, ``"noniid"``
        for label-skewed splits.
    :param poison_node_idx: 0-based index of the node to poison,
        or ``None`` for a clean run.
    :param poison_fraction: Fraction of the poisoned node's labels to
        flip (0.0–1.0).
    :param seed: Random seed for reproducibility.
    :returns: Per-node DataFrames, with the selected node
        poisoned if *poison_node_idx* is not ``None``.
    """
    if not isinstance(n_nodes, int) or n_nodes < 1:
        raise ValueError("--n-nodes must be at least 1.")
    if n_nodes > len(train_df):
        raise ValueError(f"--n-nodes ({n_nodes}) cannot exceed training rows ({len(train_df)}).")
    if poison_node_idx is not None and not 0 <= poison_node_idx < n_nodes:
        raise ValueError(f"--poison-node must be between 0 and {n_nodes - 1}.")
    if not 0.0 <= poison_fraction <= 1.0:
        raise ValueError("--poison-fraction must be between 0.0 and 1.0.")
    if partition_type == "iid":
        nodes = partition_iid(train_df, n_nodes, random_state=seed)
    else:
        nodes = partition_non_iid(train_df, n_nodes, random_state=seed)
    nodes = [n.reset_index(drop=True) for n in nodes]
    if poison_node_idx is not None:
        nodes[poison_node_idx] = poison_node(nodes[poison_node_idx], poison_fraction, random_state=seed)
    return nodes


def make_client_fn(node_dfs, scaler, poison_node_idx=None, boost_factor=1.0):
    """Build a modern Flower ClientApp callback.

    The simulation runtime gives each virtual node a stable integer ``node_id``.
    We use that directly instead of relying on the old ``node_config["partition-id"]``
    convention, which is not automatically populated by ``run_simulation``.

    :param node_dfs: Per-node DataFrames produced by
        :func:`build_node_partitions`.
    :param scaler: Fitted scaler used to normalise features.
    :param poison_node_idx: 0-based index of the node that should
        apply the *boost_factor* weight scaling, or ``None`` for a clean run.
    :param boost_factor: Multiplicative scaling applied to the poisoned
        node's model-update magnitude to amplify its influence on aggregation.
    :returns: A ``client_fn(context)`` callable suitable for
        :class:`ClientApp`.
    """
    def client_fn(context: Context):
        """Instantiate the :class:`NodeClient` for the virtual node identified by *context*.

        :param context: Flower simulation context supplying ``node_id``.
        """
        # run_simulation populates node_config["partition-id"] (0..num_supernodes-1)
        # when launched with num_supernodes. Fall back to node_id only if absent.
        cfg = getattr(context, "node_config", None) or {}
        if "partition-id" in cfg:
            pid = int(cfg["partition-id"]) % len(node_dfs)
        else:
            pid = int(context.node_id) % len(node_dfs)
        node_df = node_dfs[pid]
        boost = boost_factor if (poison_node_idx is not None and pid == poison_node_idx) else 1.0
        return NodeClient(
            node_id=str(pid),
            node_df=node_df,
            scaler=scaler,
            boost_factor=boost,
        ).to_client()

    return client_fn


def make_evaluate_fn(X_test, y_test, template_model):
    """Return a centralised evaluation callable for Flower strategy ``evaluate_fn``.

    :param X_test: Held-out feature matrix (normalised).
    :param y_test: Corresponding binary labels.
    :param template_model: An initialised model whose weights will be updated
        in-place before evaluation.
    :returns: ``evaluate_fn(server_round, parameters_ndarrays, config)``
        that computes log-loss plus accuracy, F1, precision and recall
        on the global test set.
    """
    def evaluate_fn(server_round, parameters_ndarrays, config):
        """Evaluate the global model on the held-out test set for *server_round*.

        :param server_round: Current federated learning round number.
        :param parameters_ndarrays: Global model weights.
        :param config: Configuration dict passed by the Flower strategy.
        """
        mu.set_weights(template_model, parameters_ndarrays)
        probs = template_model.predict_proba(X_test)
        preds = template_model.predict(X_test)
        loss = log_loss(y_test, probs, labels=[0, 1])
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1": f1_score(y_test, preds, zero_division=0),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
        }
        return loss, metrics
    return evaluate_fn


def run_once(strategy_name, partition_type, n_nodes, n_rounds, local_epochs,
             poison_node_idx, poison_fraction, boost_factor=1.0, seed=RANDOM_STATE, verbose=True,
             return_strategy=False):
    """Run a single federated learning experiment and return its metrics.

    :param strategy_name: Aggregation strategy: ``"fedavg"`` or ``"trust"``.
    :param partition_type: How to split data: ``"iid"`` or ``"noniid"``.
    :param n_nodes: Number of simulated federated nodes.
    :param n_rounds: Number of federated communication rounds.
    :param local_epochs: Local training epochs per node per round.
    :param poison_node_idx: 0-based index of the node to poison,
        or ``None`` for a clean run.
    :param poison_fraction: Fraction of labels to flip on the poisoned
        node (0.0–1.0).
    :param boost_factor: Update-magnitude multiplier for the poisoned node.
    :param seed: Random seed for data splitting, partitioning, and model
        initialisation.
    :param verbose: If ``True``, print a summary line after the run.
    :param return_strategy: If ``True``, also return the strategy instance
        and fitted scaler alongside the result dict.
    :returns: Result dict, or ``(result, strategy, scaler)`` if
        *return_strategy* is ``True``.
    """
    if boost_factor < 0:
        raise ValueError("boost_factor must be non-negative.")

    train_df, test_df = get_global_test_split(random_state=seed)
    scaler = fit_scaler(train_df)
    X_test, y_test = to_xy(test_df, scaler)

    node_dfs = build_node_partitions(train_df, n_nodes, partition_type, poison_node_idx, poison_fraction, seed)

    # Build + initialize a template model to get architecture-correct initial weights
    template_model = mu.build_model(random_state=seed)
    init_X, init_y = np.zeros((4, X_test.shape[1])), np.array([0, 1, 0, 1])
    mu.init_architecture(template_model, init_X, init_y)
    initial_weights = mu.get_weights(template_model)
    initial_parameters = fl.common.ndarrays_to_parameters(initial_weights)

    common_kwargs = dict(
        fraction_fit=1.0,
        fraction_evaluate=0.0,  # we use centralized evaluate_fn instead of client-side eval
        min_fit_clients=n_nodes,
        min_available_clients=n_nodes,
        on_fit_config_fn=lambda rnd: {"local_epochs": local_epochs},
        evaluate_fn=make_evaluate_fn(X_test, y_test, template_model),
        initial_parameters=initial_parameters,
    )

    if strategy_name not in {"fedavg", "trust"}:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    # Flower's current simulation API uses ClientApp/ServerApp + run_simulation.
    # Keep the centralized metrics in a shared dict because run_simulation no
    # longer returns the old History object.
    rounds_metrics = {}

    def evaluate_fn_with_history(server_round, parameters_ndarrays, config):
        """Evaluate the global model and accumulate per-round metrics into *rounds_metrics*.

        :param server_round: Current federated learning round number.
        :param parameters_ndarrays: Global model weights.
        :param config: Configuration dict passed by the Flower strategy.
        """
        loss, metrics = make_evaluate_fn(X_test, y_test, template_model)(
            server_round, parameters_ndarrays, config
        )
        for metric_name, value in metrics.items():
            rounds_metrics.setdefault(metric_name, []).append(float(value))
        return loss, metrics

    if strategy_name == "fedavg":
        # Re-create FedAvg so its evaluation callback writes into rounds_metrics.
        strategy = fl.server.strategy.FedAvg(
            **{**common_kwargs, "evaluate_fn": evaluate_fn_with_history}
        )
    else:
        strategy = TrustWeightedFedAvg(
            **{**common_kwargs, "evaluate_fn": evaluate_fn_with_history}
        )

    client_app = ClientApp(
        client_fn=make_client_fn(
            node_dfs,
            scaler,
            poison_node_idx=poison_node_idx,
            boost_factor=boost_factor,
        )
    )

    def server_fn(context: Context):
        """Configure the :class:`ServerApp` with the chosen strategy and round count.

        :param context: Flower simulation context for the server.
        """
        return ServerAppComponents(
            strategy=strategy,
            config=ServerConfig(num_rounds=n_rounds),
        )

    server_app = ServerApp(server_fn=server_fn)

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=n_nodes,
        backend_config={
            "client_resources": {"num_cpus": 1, "num_gpus": 0},
        },
    )

    final = {k: v[-1] for k, v in rounds_metrics.items() if v}
    if not final:
        raise RuntimeError(
            "Simulation finished without any evaluation metrics -- the Flower/Ray run "
            "most likely failed. Check the log above for the underlying error."
        )
    result = {
        "strategy": strategy_name,
        "partition": partition_type,
        "n_nodes": n_nodes,
        "n_rounds": n_rounds,
        "poison_node": poison_node_idx,
        "poison_fraction": poison_fraction if poison_node_idx is not None else 0.0,
        "final_metrics": final,
        "metrics_by_round": rounds_metrics,
    }
    trust_log = getattr(strategy, "trust_log", None)
    if trust_log:
        result["trust_log"] = trust_log

    if verbose:
        tag = f"[{strategy_name} | {partition_type} | poison={poison_node_idx}]"
        print(f"{tag} final accuracy={final.get('accuracy'):.4f}  f1={final.get('f1'):.4f}")

    if return_strategy:
        return result, strategy, scaler
    return result


def save_dashboard_artifacts(strategy, scaler):
    """Persist artefacts consumed by the Streamlit dashboard's device-analysis tool.

    Saves the trained global model weights, the fitted scaler, and the healthy
    consensus direction so that arbitrary new traffic files can be scored for
    trust without re-running the full federated simulation.

    :param strategy: Finished strategy instance exposing
        ``current_weights`` and ``last_consensus``.
    :param scaler: Fitted scaler used during training; persisted
        so the dashboard can normalise uploaded feature vectors consistently.
    """
    # pyrefly: ignore [missing-import]
    import joblib
    weights = strategy.current_weights
    np.savez(RESULTS_DIR / "global_model_weights.npz", *weights)
    joblib.dump(scaler, RESULTS_DIR / "scaler.joblib")
    if strategy.last_consensus is not None:
        np.save(RESULTS_DIR / "reference_consensus.npy", strategy.last_consensus)
    print(f"Saved dashboard artifacts to {RESULTS_DIR} (global_model_weights.npz, scaler.joblib, reference_consensus.npy)")


def run_suite(n_nodes=4, n_rounds=10, local_epochs=2, poison_fraction=1.0, boost_factor=4.0, seed=RANDOM_STATE):
    """Run the Phase 5 comparison suite: clean baseline, FedAvg+attack, Trust+attack.

    Executes four configurations in sequence and writes a ``comparison.json``
    summary to the results directory.  The clean-trust run also saves dashboard
    artefacts via :func:`save_dashboard_artifacts`.

    :param n_nodes: Number of federated nodes for every run in the suite.
    :param n_rounds: Number of federated learning rounds per run.
    :param local_epochs: Number of local SGD epochs each node trains per round.
    :param poison_fraction: Fraction of labels flipped on the poisoned
        node (0.0–1.0).
    :param boost_factor: Weight-update scaling applied to the poisoned
        node's gradient to amplify its influence on aggregation.
    :param seed: Random seed used for data splitting, partitioning and model
        initialisation across all runs.
    :returns: Dict mapping run tag → result dict (mirrors the saved JSON).
    """
    configs = [
        dict(strategy_name="fedavg", poison_node_idx=None, boost_factor=1.0, tag="clean_fedavg"),
        dict(strategy_name="fedavg", poison_node_idx=0, boost_factor=boost_factor, tag="fedavg_under_attack"),
        dict(strategy_name="trust", poison_node_idx=None, boost_factor=1.0, tag="clean_trust"),
        dict(strategy_name="trust", poison_node_idx=0, boost_factor=boost_factor, tag="trust_under_attack"),
    ]
    all_results = {}
    for cfg in configs:
        tag = cfg.pop("tag")
        if tag == "clean_trust":
            # This run's final global model + consensus direction becomes the
            # reference used by the dashboard's "upload & analyze a device" tool
            # to score arbitrary new/uploaded traffic later.
            res, strategy, scaler = run_once(
                partition_type="iid",
                n_nodes=n_nodes,
                n_rounds=n_rounds,
                local_epochs=local_epochs,
                poison_fraction=poison_fraction,
                seed=seed,
                return_strategy=True,
                **cfg,
            )
            save_dashboard_artifacts(strategy, scaler)
        else:
            res = run_once(
                partition_type="iid",
                n_nodes=n_nodes,
                n_rounds=n_rounds,
                local_epochs=local_epochs,
                poison_fraction=poison_fraction,
                seed=seed,
                **cfg,
            )
        all_results[tag] = res

    out_path = RESULTS_DIR / "comparison.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== Phase 5 comparison summary ===")
    for tag, res in all_results.items():
        fm = res["final_metrics"]
        print(f"{tag:22s} acc={fm.get('accuracy', float('nan')):.4f}  f1={fm.get('f1', float('nan')):.4f}")
    print(f"\nSaved full results to {out_path}")
    return all_results


def main():
    """Parse CLI arguments and dispatch to :func:`run_once` or :func:`run_suite`."""
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", choices=["fedavg", "trust"], default="fedavg")
    p.add_argument("--partition", choices=["iid", "noniid"], default="iid")
    p.add_argument("--n-nodes", type=int, default=4)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--local-epochs", type=int, default=2)
    p.add_argument("--poison-node", type=int, default=None, help="index of node to poison (0-based)")
    p.add_argument("--poison-fraction", type=float, default=1.0)
    p.add_argument("--boost-factor", type=float, default=4.0, help="update-scaling factor for the poisoned node")
    p.add_argument("--suite", action="store_true", help="run the full Phase 5 comparison suite")
    args = p.parse_args()

    if args.suite:
        run_suite(n_nodes=args.n_nodes, n_rounds=args.rounds, local_epochs=args.local_epochs,
                  poison_fraction=args.poison_fraction, boost_factor=args.boost_factor)
    else:
        run_once(
            strategy_name=args.strategy,
            partition_type=args.partition,
            n_nodes=args.n_nodes,
            n_rounds=args.rounds,
            local_epochs=args.local_epochs,
            poison_node_idx=args.poison_node,
            poison_fraction=args.poison_fraction,
            boost_factor=args.boost_factor,
        )


if __name__ == "__main__":
    main()
