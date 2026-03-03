from __future__ import print_function

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# configuable dataset parameters for loading and processing
# Each dataset specifies 
# 	file path, 
# 	feature column name, 
# 	columns to drop, 
# 	and data mode (miRNA count vs log normalized data).
DATASET_CONFIGS = {
    "tissue": {
        "path": "data/tissue_bc/GSE270497_All.txt",
        "feature_col": "smallRNAName",
        "drop_columns": ["smallRNApreName", "smallRNASequence"],
        "data_mode": "count",
    },
    "circulating": {
        "path": "data/circulating_bc/GSE197020.txt",
        "feature_col": "smallRNASequence",
        "drop_columns": [],
        "data_mode": "count",
    },
}

# Pancancer data config
PANCANCER_MATRIX_PATH = "data/serum_pancancer/GSE211692_processed_data_matrix.txt"
PANCANCER_METADATA_PATH = "data/serum_pancancer/GSE211692_metadata.csv"
PANCANCER_FEATURE_COL = "ID_REF"
PANCANCER_SAMPLE_ID_COL = "Title"
PANCANCER_DISEASE_COL = "Disease State"

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

# Function for pancancer dataset loading and processing, including label inference and dataset summarization
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

# Helper functions for parsing sample IDs, normalizing feature and disease names, and summarizing dataset properties
def _numeric_sample_suffix(sample_id):
    match = re.search(r"(\d+)$", str(sample_id))
    if match is None:
        raise ValueError(f"Could not parse numeric suffix from sample '{sample_id}'")
    return int(match.group(1))

def _normalize_feature_name(feature_name):
    return str(feature_name).strip().lower()

def _resolve_path(path_like):
    path = Path(path_like)
    if path.is_absolute():
        return path
    candidate = Path.cwd() / path
    if candidate.exists():
        return candidate
    return path

def _normalize_disease_name(disease_name):
    return str(disease_name).strip().lower()

# infer data mode (count vs normalized) based on value properties
def infer_data_mode_from_values(values):
    values = np.asarray(values, dtype=np.float64)
    finite_mask = np.isfinite(values)
    if not finite_mask.all():
        return "unknown"

    if np.any(values < 0):
        return "normalized"

    rounded = np.round(values)
    integer_like_ratio = np.isclose(values, rounded, atol=1e-6).mean()
    if integer_like_ratio > 0.95:
        return "count"
    return "normalized"

# summarize dataset properties such as data mode, presence of negative values and dimensions
def summarize_expression_matrix(counts_df, data_mode):
    values = counts_df.to_numpy(dtype=np.float64, copy=False)
    finite_mask = np.isfinite(values)
    has_negative = bool((values < 0).any())

    if data_mode not in {"count", "normalized"}:
        raise ValueError("data_mode must be 'count' or 'normalized'")

    info = {
        "data_mode": data_mode,
        "inferred_data_mode": infer_data_mode_from_values(values),
        "has_negative_values": has_negative,
        "is_finite": bool(finite_mask.all()),
        "n_samples": int(counts_df.shape[0]),
        "n_features": int(counts_df.shape[1]),
    }
    return info


def add_serum_pancancer_stage_labels(metadata_df, disease_col=PANCANCER_DISEASE_COL):
    """
    Adds stage 1 binary labels and stage 2 multi-class labels to the serum pancancer metadata.
    Stage 1: Binary classification of 'cancer' vs 'non_cancer' 
    Stage 2: Multi-class classification of specific cancer types vs 'non_cancer'.
    """
    if disease_col not in metadata_df.columns:
        raise ValueError(
            f"Expected disease column '{disease_col}' in metadata. "
            f"Available columns: {metadata_df.columns.tolist()}"
        )

    metadata_out = metadata_df.copy()
    disease_norm = metadata_out[disease_col].astype(str).map(_normalize_disease_name)

    benign_mask = disease_norm.str.contains("benign", regex=False)
    control_mask = disease_norm.eq("no cancer")
    non_cancer_mask = benign_mask | control_mask

    metadata_out["disease_state_normalized"] = disease_norm
    metadata_out["stage1_label"] = (~non_cancer_mask).astype(np.float32)
    metadata_out["stage1_group"] = np.where(
        control_mask,
        "no_cancer_control",
        np.where(benign_mask, "non_cancer_condition", "cancer"),
    )

    cancer_states = sorted(disease_norm[~non_cancer_mask].unique().tolist())
    stage2_class_to_index = {class_name: idx for idx, class_name in enumerate(cancer_states)}
    metadata_out["stage2_class_name"] = np.where(
        ~non_cancer_mask,
        disease_norm,
        "non_cancer",
    )
    metadata_out["stage2_label"] = (
        metadata_out["stage2_class_name"].map(stage2_class_to_index).fillna(-1).astype(int)
    )

    return metadata_out, stage2_class_to_index


def load_serum_pancancer_dataset(
    matrix_path=PANCANCER_MATRIX_PATH,
    metadata_path=PANCANCER_METADATA_PATH,
    feature_col=PANCANCER_FEATURE_COL,
    sample_id_col=PANCANCER_SAMPLE_ID_COL,
    disease_col=PANCANCER_DISEASE_COL,
    normalize_feature_names=True,
    data_mode="normalized",
    return_info=False,
):
    """
    Loads the serum pancancer dataset, aligns metadata with the expression matrix, and adds stage 1 and stage 2 labels.
     - Stage 1: Binary classification of 'cancer' vs 'non_cancer'
     - Stage 2: Multi-class classification of specific cancer types vs 'non_cancer'
     
    """
    matrix_path = _resolve_path(matrix_path)
    metadata_path = _resolve_path(metadata_path)

    metadata_df = pd.read_csv(metadata_path)
    required_metadata_cols = {sample_id_col, disease_col}
    missing_metadata_cols = required_metadata_cols.difference(metadata_df.columns)
    if missing_metadata_cols:
        raise ValueError(
            f"Missing required metadata columns: {sorted(missing_metadata_cols)}. "
            f"Available columns: {metadata_df.columns.tolist()}"
        )

	# Ensure sample IDs are strings and check for duplicates
    metadata_df = metadata_df.copy()
    metadata_df[sample_id_col] = metadata_df[sample_id_col].astype(str)
    if metadata_df[sample_id_col].duplicated().any():
        dup_ids = metadata_df.loc[metadata_df[sample_id_col].duplicated(), sample_id_col].head(5).tolist()
        raise ValueError(
            f"Duplicated sample IDs found in metadata column '{sample_id_col}'. "
            f"Examples: {dup_ids}"
        )

    matrix_df = pd.read_csv(matrix_path, sep="\t")
    if feature_col not in matrix_df.columns:
        raise ValueError(
            f"Expected feature column '{feature_col}' in expression matrix. "
            f"Available columns (first 10): {matrix_df.columns.tolist()[:10]}"
        )

	# Set feature column as index and convert values to numeric, coercing errors to NaN and filling with 0.0
    expr_df = matrix_df.set_index(feature_col)
    expr_df = expr_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if normalize_feature_names:
        expr_df.index = expr_df.index.map(_normalize_feature_name)
        expr_df = expr_df.groupby(level=0).mean()

	# Transpose to have samples as rows and features as columns, ensuring indexes are strings for alignment
    counts_df = expr_df.T
    counts_df.index = counts_df.index.astype(str)
    counts_df.columns = counts_df.columns.astype(str)

    keep_mask = metadata_df[sample_id_col].isin(counts_df.index)
    metadata_aligned = metadata_df.loc[keep_mask].set_index(sample_id_col)
    if metadata_aligned.empty:
        raise ValueError(
            "No overlapping samples between metadata and expression matrix. "
            f"Metadata sample col='{sample_id_col}'."
        )

	# Align counts_df to the order of metadata_aligned
    counts_aligned = counts_df.loc[metadata_aligned.index]
    metadata_labeled, stage2_class_to_index = add_serum_pancancer_stage_labels(
        metadata_aligned.reset_index(),
        disease_col=disease_col,
    )
    metadata_labeled = metadata_labeled.set_index(sample_id_col)

    if not counts_aligned.index.equals(metadata_labeled.index):
        raise ValueError("Aligned counts and metadata indexes are not identical in order.")

    index_to_class = {idx: class_name for class_name, idx in stage2_class_to_index.items()}
    label_maps = {
        "stage2_class_to_index": stage2_class_to_index,
        "stage2_index_to_class": index_to_class,
    }

    dataset_info = summarize_expression_matrix(counts_aligned, data_mode=data_mode)

    if return_info:
        return counts_aligned, metadata_labeled, label_maps, dataset_info
    return counts_aligned, metadata_labeled, label_maps


def load_serum_pancancer_stage1_binary(
    matrix_path=PANCANCER_MATRIX_PATH,
    metadata_path=PANCANCER_METADATA_PATH,
    data_mode="normalized",
    return_info=False,
):
    """
    Loads the serum pancancer dataset and returns the expression matrix, stage 1 binary labels, and metadata.
    """
    loader_output = load_serum_pancancer_dataset(
        matrix_path=matrix_path,
        metadata_path=metadata_path,
        data_mode=data_mode,
        return_info=return_info,
    )
    if return_info:
        counts_df, metadata_df, label_maps, dataset_info = loader_output
    else:
        counts_df, metadata_df, label_maps = loader_output
    labels = metadata_df["stage1_label"].to_numpy(dtype=np.float32)
    if return_info:
        return counts_df, labels, metadata_df, label_maps, dataset_info
    return counts_df, labels, metadata_df, label_maps


def infer_binary_labels(sample_names, dataset_name):
    """
    Tissue vs Circulating
	Infers binary labels for samples based on their names and the dataset they belong to.
    """

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


def load_expression_dataset(dataset_name, data_path=None, data_mode=None, return_info=False):
    """
    Loads the specified expression dataset, processes it, and infers binary labels based on sample names.
	 - dataset_name: Name of the dataset to load (e.g., 'tissue' or 'circulating').
	 - data_path: Optional base path to look for the dataset file. If None, uses default paths from DATASET_CONFIGS.
	 - data_mode: Optional override for data mode ('count' or 'normalized'). If None, uses the mode specified in DATASET_CONFIGS.
	 - return_info: If True, also returns a dictionary summarizing dataset properties.
	 - Returns: counts_df (samples x features), labels (binary), and optionally dataset_info.
    """
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
    if data_mode is None:
        data_mode = config.get("data_mode", "count")
    dataset_info = summarize_expression_matrix(counts_df, data_mode=data_mode)

    if return_info:
        return counts_df, labels, dataset_info
    return counts_df, labels

# Aligns the feature space of the counts dataframe to a specified list of required features,
#  filling missing features with a specified value (default 0.0).
def align_feature_space(counts_df, required_features, fill_value=0.0):
    required_features = [_normalize_feature_name(feature) for feature in required_features]
    return counts_df.reindex(columns=required_features, fill_value=fill_value)

# Computes the overlap between the features in the counts dataframe and a list of required features, returning lists of present and missing features.
def get_feature_overlap(counts_df, required_features):
    available = set(counts_df.columns)
    required = [_normalize_feature_name(feature) for feature in required_features]
    present = [feature for feature in required if feature in available]
    missing = [feature for feature in required if feature not in available]
    return present, missing

# Selects the top N signal features based on statistical tests (Mann-Whitney U, t-test, or PyDESeq2) comparing cancer vs healthy samples.
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

# Selects reference features based on mean expression and coefficient of variation, 
# prioritizing stable features among abundant genes.
def get_reference_features(counts_df, n_features=100):
    means = counts_df.mean(axis=0)
    stds = counts_df.std(axis=0)
    cv = stds / (means + 1e-6)

    abundant_genes = means[means > 10].index
    stable_features = cv[abundant_genes].nsmallest(n_features).index.tolist()
    return stable_features


class ZINBDataset(Dataset):
    """
    PyTorch Dataset for zero-inflated negative binomial (ZINB) modeling of expression data.
     - counts_df: DataFrame of shape (samples x features) containing expression counts.
     - labels: Array-like of shape (samples,) containing binary or multi-class labels for each sample.
     - signal_features: List of feature names to use as input signals (X).
     - ref_features: List of feature names to use as reference features (R).
     - task_type: String indicating the type of classification task ('binary' or 'multiclass').
	 - The dataset returns tuples of (X, R, library_size, label) for each sample, where:
	   - X: Tensor of shape (num_signal_features,) containing the signal features for the sample
	   - R: Tensor of shape (num_ref_features,) containing the reference features for the sample
	   - library_size: Tensor of shape (1,) containing the total count (library size)
	   - label: Tensor containing the binary or multi-class label for the sample
    """
    def __init__(self, counts_df, labels, signal_features, ref_features, task_type="binary"):
        labels = np.asarray(labels).reshape(-1)
        if len(labels) != len(counts_df):
            raise ValueError(
                f"Label length ({len(labels)}) must match sample count ({len(counts_df)})"
            )

        self.task_type = str(task_type).strip().lower()
        if self.task_type == "binary":
            labels = labels.astype(np.float32)
            self.labels = torch.FloatTensor(labels).unsqueeze(1)
        elif self.task_type == "multiclass":
            labels = labels.astype(np.int64)
            self.labels = torch.LongTensor(labels)
        else:
            raise ValueError("task_type must be 'binary' or 'multiclass'")

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
