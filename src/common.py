"""
common.py

Shared utilities: dataset loading, preprocessing, and node partitioning
(IID and non-IID) used by the baseline model, the federated clients, and the
attack simulation.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT / "data" / "raw" / "synthetic_iot_traffic.csv"
NODES_DIR = ROOT / "data" / "nodes"
NODES_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = [
    "duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts",
    "src_ip_bytes", "dst_ip_bytes", "missed_bytes", "conn_state_enc",
    "proto_enc", "service_enc", "http_request_body_len",
    "http_response_body_len", "dns_qclass", "ssl_version_enc", "weird_notice_enc",
]
LABEL_COLUMN = "label"
RANDOM_STATE = 42


def load_dataset(path: Path = RAW_DATA_PATH) -> DataFrame:
    """Load the dataset. Swap `path` to point at a real TON_IoT/CIC-IoT2023
    CSV once available -- as long as it exposes the same `label` column and
    numeric feature columns (see FEATURE_COLUMNS), nothing else needs to change.

    :param path: Path to the CSV dataset file.
    :returns: The loaded dataset as a DataFrame.
    """
    if not path.exists():
        if path == RAW_DATA_PATH:
            gen_script = ROOT / "data" / "generate_synthetic_data.py"
            if gen_script.exists():
                import sys
                sys.path.insert(0, str(ROOT / "data"))
                import generate_synthetic_data
                generate_synthetic_data.generate()
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python data/generate_synthetic_data.py` first "
                "(or point RAW_DATA_PATH at your real dataset)."
            )
    return pd.read_csv(path)


def get_global_test_split(test_size=0.15, random_state=RANDOM_STATE):
    """Held-out global test set, never seen by any federated client.
    Used only for final, apples-to-apples evaluation of the global model.

    :param test_size: Fraction of data to reserve as the global test set.
    :param random_state: Seed for reproducibility.
    """
    df = load_dataset()
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[LABEL_COLUMN]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def fit_scaler(train_df: DataFrame) -> StandardScaler:
    """Fit a StandardScaler on the feature columns of the training DataFrame.

    :param train_df: Training data used to fit the scaler.
    :returns: A scaler fitted on FEATURE_COLUMNS.
    """
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLUMNS])
    return scaler


def to_xy(df: DataFrame, scaler: StandardScaler):
    """Transform a DataFrame into scaled feature matrix X and label array y.

    :param df: DataFrame containing feature and label columns.
    :param scaler: A fitted scaler to apply to the features.
    :returns: ``(X, y)`` where X is the scaled numpy feature array and
        y is the integer label array.
    """
    X = scaler.transform(df[FEATURE_COLUMNS])
    y = df[LABEL_COLUMN].values.astype(np.int64)
    return X, y


def _split_dataframe(df: DataFrame, n_chunks: int):
    """Split into non-empty DataFrames with predictable validation.

    :param df: The DataFrame to split.
    :param n_chunks: Number of roughly-equal chunks to produce.
    """
    if not isinstance(n_chunks, int) or n_chunks < 1:
        raise ValueError("n_chunks/n_nodes must be at least 1.")
    if n_chunks > len(df):
        raise ValueError(
            f"Cannot create {n_chunks} non-empty partitions from only {len(df)} rows."
        )
    idx_chunks = np.array_split(np.arange(len(df)), n_chunks)
    return [df.iloc[idx].reset_index(drop=True) for idx in idx_chunks]


def partition_iid(df: DataFrame, n_nodes: int, random_state=RANDOM_STATE):
    """Randomly shuffle and split into n_nodes roughly-equal, class-balanced chunks.

    :param df: The full dataset to partition.
    :param n_nodes: Number of federated nodes to split the data into.
    :param random_state: Seed for reproducibility.
    """
    shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return _split_dataframe(shuffled, n_nodes)


def partition_non_iid(df: DataFrame, n_nodes: int, primary_share=0.7, random_state=RANDOM_STATE):
    """Each node gets a disproportionate share of one or two attack_types,
    simulating networks that see different kinds of traffic/attacks.
    ``primary_share`` controls how skewed each node's data is toward its
    assigned attack type(s) (higher = more non-IID).

    :param df: The full dataset to partition across nodes.
    :param n_nodes: Number of federated nodes to split the data into.
    :param primary_share: Fraction of each node's data that comes from
        its assigned attack type(s). Higher values = more non-IID.
    :param random_state: Seed for reproducibility.
    """
    rng = np.random.RandomState(random_state)
    attack_types = [t for t in df["attack_type"].unique() if t != "normal"]
    rng.shuffle(attack_types)
    assignments = np.array_split(attack_types, n_nodes)

    normal_df = df[df["attack_type"] == "normal"]
    normal_chunks = _split_dataframe(
        normal_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True), n_nodes
    )

    nodes = []
    for i in range(n_nodes):
        primary_types = list(assignments[i])
        primary_df = df[df["attack_type"].isin(primary_types)]
        # Sample primary_share of this node's "primary" attack rows,
        # plus a small amount of everything else to avoid zero-shot classes.
        n_primary = int(len(primary_df) * primary_share)
        primary_sample = primary_df.sample(
            n=min(n_primary, len(primary_df)), random_state=random_state + i
        )
        other_df = df[~df["attack_type"].isin(primary_types) & (df["attack_type"] != "normal")]
        other_sample = other_df.sample(
            frac=(1 - primary_share) * 0.3, random_state=random_state + 1000 + i
        )
        node_df = pd.concat([primary_sample, other_sample, normal_chunks[i]])
        node_df = node_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        nodes.append(node_df)
    return nodes


def save_node_partitions(nodes, prefix="node"):
    """Save each node's DataFrame partition to a CSV file in NODES_DIR.

    :param nodes: List of per-node DataFrames to save.
    :param prefix: Filename prefix for each partition file.
    :returns: List of file paths where partitions were saved.
    """
    paths = []
    for i, node_df in enumerate(nodes):
        p = NODES_DIR / f"{prefix}_{i}.csv"
        node_df.to_csv(p, index=False)
        paths.append(p)
    return paths


def poison_node(node_df: DataFrame, flip_fraction: float, random_state=RANDOM_STATE) -> DataFrame:
    """Simulate label-flip poisoning and return a new dataframe.

    :param node_df: The node's dataset to poison.
    :param flip_fraction: Fraction of labels to randomly flip (0.0–1.0).
    :param random_state: Seed for reproducibility.
    :returns: A copy of node_df with a fraction of labels flipped.
    """
    if not 0.0 <= flip_fraction <= 1.0:
        raise ValueError("flip_fraction must be between 0.0 and 1.0.")
    if LABEL_COLUMN not in node_df.columns:
        raise ValueError(f"Missing required label column: {LABEL_COLUMN}")
    poisoned = node_df.copy()
    rng = np.random.RandomState(random_state)
    n_flip = int(len(poisoned) * flip_fraction)
    if n_flip == 0:
        return poisoned
    flip_idx = rng.choice(poisoned.index, size=n_flip, replace=False)
    poisoned.loc[flip_idx, LABEL_COLUMN] = 1 - poisoned.loc[flip_idx, LABEL_COLUMN]
    return poisoned
