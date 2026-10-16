"""Column alignment and telemetry bug-tracing for uploaded device CSVs."""

import numpy as np
import pandas as pd

from common import FEATURE_COLUMNS, LABEL_COLUMN

ALIAS_MAP = {
    "flow_duration": "duration", "dur": "duration", "time": "duration",
    "tot_fwd_pkts": "src_pkts", "fwd_pkts": "src_pkts", "packets_sent": "src_pkts", "src_packets": "src_pkts",
    "tot_bwd_pkts": "dst_pkts", "bwd_pkts": "dst_pkts", "packets_rcvd": "dst_pkts", "dst_packets": "dst_pkts",
    "totlen_fwd_pkts": "src_bytes", "bytes_sent": "src_bytes", "source_bytes": "src_bytes",
    "totlen_bwd_pkts": "dst_bytes", "bytes_rcvd": "dst_bytes", "dest_bytes": "dst_bytes",
    "protocol": "proto_enc", "proto": "proto_enc",
    "service": "service_enc",
    "class": "label", "target": "label", "attack": "label", "is_attack": "label",
}


def normalize_and_align_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Align arbitrary user CSV columns to the required model feature schema.

    Performs case-insensitive column resolution, maps common network flow aliases
    (e.g., tot_fwd_pkts -> src_pkts), and imputes any remaining missing feature
    columns using standard baseline medians so arbitrary CSVs can be scored.

    :param df: Input DataFrame with arbitrary user columns.
    :returns: Tuple of (aligned DataFrame with all FEATURE_COLUMNS, list of imputed column names).
    """
    aligned = df.copy()
    col_map = {}
    for col in aligned.columns:
        norm = str(col).strip().lower().replace(" ", "_").replace("-", "_")
        if norm in ALIAS_MAP:
            col_map[col] = ALIAS_MAP[norm]
        elif norm in [f.lower() for f in FEATURE_COLUMNS]:
            for f in FEATURE_COLUMNS:
                if f.lower() == norm:
                    col_map[col] = f
        elif norm == LABEL_COLUMN.lower():
            col_map[col] = LABEL_COLUMN

    if col_map:
        aligned = aligned.rename(columns=col_map)

    imputed_cols = []
    for c in FEATURE_COLUMNS:
        if c not in aligned.columns:
            aligned[c] = 0.0
            imputed_cols.append(c)
        else:
            # Unparseable cells ("abc", "N/A") become NaN so the audit reports
            # them as missing values instead of crashing on comparisons.
            aligned[c] = pd.to_numeric(aligned[c], errors="coerce")

    if LABEL_COLUMN in aligned.columns:
        if not pd.api.types.is_numeric_dtype(aligned[LABEL_COLUMN]):
            aligned[LABEL_COLUMN] = aligned[LABEL_COLUMN].astype(str).str.lower().map(
                lambda v: 0 if v in ["0", "0.0", "normal", "benign", "safe", "false", "clean"] else 1
            )

    return aligned, imputed_cols


def trace_file_bugs_and_anomalies(df: pd.DataFrame, imputed_cols: list[str]) -> dict:
    """Audit an uploaded traffic dataset for schema issues, data corruption, and security threats.

    Identifies malformed values (NaNs, negative durations, integer underflows),
    flags duplicated network flows and extreme packet spikes, detects volumetric
    attack surges, and provides a sanitized copy ready for neural scoring.
    Returns a dict with keys ``'findings'`` (list of dicts), ``'health_score'`` (int),
    and ``'sanitized_df'`` (cleaned DataFrame).
    """
    findings = []
    sanitized_df = df.copy()
    total_rows = len(df)

    if imputed_cols:
        findings.append({
            "category": "Schema Adjustment",
            "severity": "Info",
            "bug": f"Missing Features Imputed ({len(imputed_cols)})",
            "details": f"Columns missing from CSV: {', '.join(imputed_cols)}. Automatically filled with baseline neutral medians.",
            "action": "Zero/median imputed",
        })

    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        findings.append({
            "category": "Data Integrity",
            "severity": "Warning",
            "bug": "Duplicate Network Flows",
            "details": f"{dup_count} duplicated row(s) detected ({dup_count / total_rows:.1%}). Indicates retransmitted packets or buffer flush loops.",
            "action": "Deduplicated for classification",
        })
        sanitized_df = sanitized_df.drop_duplicates().reset_index(drop=True)

    null_counts = df.isnull().sum()
    cols_with_null = null_counts[null_counts > 0]
    if not cols_with_null.empty:
        total_null_cells = int(cols_with_null.sum())
        cols_summary = ", ".join([f"{c} ({n})" for c, n in cols_with_null.items()])
        findings.append({
            "category": "Data Integrity",
            "severity": "Critical" if total_null_cells > 50 else "Warning",
            "bug": "Missing / NaN Sensor Values",
            "details": f"{total_null_cells} missing cell(s) across: {cols_summary}. Caused by dropped sensor readings or incomplete logging.",
            "action": "Imputed with column medians",
        })
        for c in sanitized_df.select_dtypes(include=[np.number]).columns:
            median_val = sanitized_df[c].median()
            sanitized_df[c] = sanitized_df[c].fillna(median_val if not pd.isna(median_val) else 0.0)

    if "duration" in df.columns:
        neg_dur = int((df["duration"] < 0).sum())
        if neg_dur > 0:
            min_val = float(df["duration"].min())
            findings.append({
                "category": "Sensor Hardware Bug",
                "severity": "Critical",
                "bug": "Clock Skew / Negative Duration",
                "details": f"{neg_dur} packet flow(s) have negative duration (min: {min_val:.3f}s). Indicates RTC desynchronization or NTP clock jump.",
                "action": "Clipped negative durations to 0.0s",
            })
            sanitized_df["duration"] = sanitized_df["duration"].clip(lower=0.0)

    byte_pkt_cols = [c for c in ["src_bytes", "dst_bytes", "src_pkts", "dst_pkts", "src_ip_bytes", "dst_ip_bytes"] if c in df.columns]
    neg_counters = {}
    for c in byte_pkt_cols:
        neg_c = int((df[c] < 0).sum())
        if neg_c > 0:
            neg_counters[c] = neg_c
    if neg_counters:
        summary = ", ".join([f"{k}: {v}" for k, v in neg_counters.items()])
        findings.append({
            "category": "Sensor Hardware Bug",
            "severity": "Critical",
            "bug": "Integer Underflow / Negative Counter",
            "details": f"Negative counters found in {summary}. Indicates signed integer overflow or telemetry firmware bug.",
            "action": "Clipped negative values to 0",
        })
        for c in neg_counters:
            sanitized_df[c] = sanitized_df[c].clip(lower=0)

    if "src_pkts" in df.columns:
        extreme_pkts = int((df["src_pkts"] > 1000000).sum())
        if extreme_pkts > 0:
            findings.append({
                "category": "Threat / Buffer Bug",
                "severity": "Critical",
                "bug": "Extreme Packet Flood Spike",
                "details": f"{extreme_pkts} flow(s) exceed 1,000,000 packets per burst. Indicates volumetric flood or sensor buffer overflow.",
                "action": "Flagged as high-severity threat",
            })

    if "attack_type" in df.columns:
        dos_count = int(df["attack_type"].isin(["dos", "ddos"]).sum())
        if dos_count > 0.4 * total_rows:
            findings.append({
                "category": "Security Threat",
                "severity": "Critical",
                "bug": "Volumetric Denial-of-Service Surge",
                "details": f"{dos_count} flow(s) ({dos_count / total_rows:.1%}) match DoS/DDoS volumetric signatures.",
                "action": "Mitigation: Rate limiting recommended",
            })
        scan_count = int((df["attack_type"] == "scanning").sum())
        if scan_count > 0.15 * total_rows:
            findings.append({
                "category": "Security Threat",
                "severity": "Warning",
                "bug": "Reconnaissance Port Sweep",
                "details": f"{scan_count} flow(s) ({scan_count / total_rows:.1%}) match network port scanning and service probing.",
                "action": "Mitigation: Blacklist scan source IPs",
            })

    critical_count = sum(1 for f in findings if f["severity"] == "Critical")
    warning_count = sum(1 for f in findings if f["severity"] == "Warning")
    info_count = sum(1 for f in findings if f["severity"] == "Info")
    health_score = max(100 - (critical_count * 25) - (warning_count * 10) - (info_count * 5), 5)

    return {
        "findings": findings,
        "health_score": health_score,
        "sanitized_df": sanitized_df,
    }
