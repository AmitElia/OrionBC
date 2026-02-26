from __future__ import print_function

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

DATASET_CONFIGS = {
    "tissue": {
        "path": "data/tissue_bc/GSE270497_All.txt",
        "feature_col": "smallRNAName",
        "drop_columns": ["smallRNApreName", "smallRNASequence"],
    },
    "circulating": {
        "path": "data/circulating_bc/GSE197020.txt",
        "feature_col": "smallRNASequence",
        "drop_columns": [],
    },
}

# Labels derived from GEO GSE197020 metadata (disease state: breast cancer/normal).
CIRCULATING_CANCER_SAMPLES = {
    "TM64", "TM212", "TM221", "TM55", "TM222", "TM56", "TM223", "TM57",
    "TM238", "TM239", "TM240", "TM241", "TM242", "TM36", "TM37", "TM38",
    "TM58", "TM39", "TM59", "TM40", "TM61", "TM41", "TM117", "TM118",
    "TM119", "TM62", "TM120", "TM63", "TM107", "TM108", "TM109", "TM110",
    "TM51", "TM44", "TM45", "TM46", "TM47", "TM48", "TM49", "TM50",
    "TM42", "TM43", "TM15", "TM16", "TM17", "TM18", "TM210", "TM211",
    "TM213", "TM214", "TM215", "TM216", "TM217", "TM218", "TM219", "TM220",
}

CIRCULATING_CONTROL_SAMPLES = {
    "TM159", "TM160", "TM161", "TM162", "TM163", "TM175", "TM176", "TM177",
    "TM178", "TM19", "TM20", "TM21", "TM22", "TM23", "TM24", "TM25",
    "TM26", "TM27", "TM28", "TM29", "TM30", "TM147", "TM148", "TM149",
    "TM150", "TM151", "TM152", "TM153", "TM154", "TM155", "TM156", "TM157",
    "TM158", "TM229", "TM233", "TM331", "TM332", "TM333", "TM334", "TM335",
}


def _resolve_dataset_path(dataset_name, data_path=None):
    if data_path is not None:
        path = Path(data_path)
    else:
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(
                f"Unknown dataset_name='{dataset_name}'. "
                f"Valid options: {sorted(DATASET_CONFIGS)}"
            )
        path = Path(DATASET_CONFIGS[dataset_name]["path"])

    if path.is_absolute():
        return path

    root = Path.cwd()
    candidate = root / path
    if candidate.exists():
        return candidate

    return path


def _numeric_sample_suffix(sample_id):
    match = re.search(r"(\d+)$", str(sample_id))
    if match is None:
        raise ValueError(f"Could not parse numeric suffix from sample '{sample_id}'")
    return int(match.group(1))


def _normalize_feature_name(feature_name):
    return str(feature_name).strip().lower()


def infer_binary_labels(sample_names, dataset_name):
    sample_names = [str(sample).strip() for sample in sample_names]

    if dataset_name == "tissue":
        # GSE270497 sample naming in this repository uses BCE IDs where
        # ID < 300 are cancer (Positive-*) and >= 300 are healthy (Negative-*).
        labels = [1.0 if _numeric_sample_suffix(sample) < 300 else 0.0 for sample in sample_names]
        return np.asarray(labels, dtype=np.float32)

    if dataset_name == "circulating":
        labels = []
        for sample in sample_names:
            if sample in CIRCULATING_CANCER_SAMPLES:
                labels.append(1.0)
            elif sample in CIRCULATING_CONTROL_SAMPLES:
                labels.append(0.0)
            else:
                raise ValueError(
                    f"Sample '{sample}' not found in known GSE197020 class mapping."
                )
        return np.asarray(labels, dtype=np.float32)

    raise ValueError(
        f"Unknown dataset_name='{dataset_name}'. "
        f"Valid options: {sorted(DATASET_CONFIGS)}"
    )


def load_expression_dataset(dataset_name, data_path=None):
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset_name='{dataset_name}'. "
            f"Valid options: {sorted(DATASET_CONFIGS)}"
        )

    config = DATASET_CONFIGS[dataset_name]
    path = _resolve_dataset_path(dataset_name, data_path=data_path)
    raw_df = pd.read_csv(path, sep="\t")

    feature_col = config["feature_col"]
    if feature_col not in raw_df.columns:
        raise ValueError(
            f"Expected feature column '{feature_col}' in {path}, "
            f"found columns: {raw_df.columns.tolist()[:10]}..."
        )

    counts_df = raw_df.set_index(feature_col)

    for column in config["drop_columns"]:
        if column in counts_df.columns:
            counts_df = counts_df.drop(columns=column)

    counts_df.index = counts_df.index.map(_normalize_feature_name)
    counts_df = counts_df.groupby(level=0).sum()

    counts_df = counts_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    counts_df = counts_df.T
    counts_df.index = counts_df.index.astype(str)
    counts_df.columns = counts_df.columns.astype(str)

    labels = infer_binary_labels(counts_df.index.tolist(), dataset_name)

    return counts_df, labels


def align_feature_space(counts_df, required_features, fill_value=0.0):
    required_features = [_normalize_feature_name(feature) for feature in required_features]
    return counts_df.reindex(columns=required_features, fill_value=fill_value)


def get_feature_overlap(counts_df, required_features):
    available = set(counts_df.columns)
    required = [_normalize_feature_name(feature) for feature in required_features]
    present = [feature for feature in required if feature in available]
    missing = [feature for feature in required if feature not in available]
    return present, missing


def get_X_features(counts_df, labels, method="mannwhitneyu", n_features=500):
    print(f"Selecting top {n_features} signal features...")
    p_values = []

    cancer_samples = counts_df[labels == 1]
    healthy_samples = counts_df[labels == 0]

    if method == "mannwhitneyu":
        from scipy.stats import mannwhitneyu

        for column in counts_df.columns:
            try:
                _, p = mannwhitneyu(cancer_samples[column], healthy_samples[column])
            except ValueError:
                p = 1.0
            p_values.append(p)

        p_series = pd.Series(p_values, index=counts_df.columns)
        top_features = p_series.nsmallest(n_features).index.tolist()
        return top_features

    if method == "ttest":
        from scipy.stats import ttest_ind

        for column in counts_df.columns:
            _, p = ttest_ind(
                cancer_samples[column],
                healthy_samples[column],
                equal_var=False,
                nan_policy="omit",
            )
            p_values.append(p)

        p_series = pd.Series(p_values, index=counts_df.columns).dropna()
        top_features = p_series.nsmallest(n_features).index.tolist()
        return top_features

    if method == "pydeseq2":
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats

        print("Running PyDESeq2 (this may take a moment)...")
        clinical_df = pd.DataFrame({"condition": labels}, index=counts_df.index)
        dds = DeseqDataSet(
            counts=counts_df.astype(int),
            clinical=clinical_df,
            design_factors="condition",
            refit_cooks=True,
            n_cpus=4,
        )
        dds.deseq2()
        stat_res = DeseqStats(dds, contrast=["condition", "1", "0"])
        stat_res.summary()
        res_df = stat_res.results_df
        top_features = (
            res_df.sort_values("padj")
            .dropna()
            .head(n_features)
            .index.tolist()
        )
        return top_features

    raise ValueError(f"Unknown method: {method}")


def get_reference_features(counts_df, n_features=100):
    means = counts_df.mean(axis=0)
    stds = counts_df.std(axis=0)
    cv = stds / (means + 1e-6)

    abundant_genes = means[means > 10].index
    stable_features = cv[abundant_genes].nsmallest(n_features).index.tolist()
    return stable_features


class ZINBDataset(Dataset):
    def __init__(self, counts_df, labels, signal_features, ref_features):
        labels = np.asarray(labels, dtype=np.float32).reshape(-1)
        if len(labels) != len(counts_df):
            raise ValueError(
                f"Label length ({len(labels)}) must match sample count ({len(counts_df)})"
            )

        self.labels = torch.FloatTensor(labels).unsqueeze(1)

        aligned_signal = align_feature_space(counts_df, signal_features)
        aligned_reference = align_feature_space(counts_df, ref_features)

        x_data = aligned_signal.values.astype(np.float32)
        self.x = torch.from_numpy(x_data)

        r_data = aligned_reference.values.astype(np.float32)
        self.r = torch.from_numpy(r_data)

        library_size = counts_df.sum(axis=1).values.astype(np.float32)
        self.library_size = torch.from_numpy(library_size).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.x[idx], self.r[idx], self.library_size[idx], self.labels[idx]
