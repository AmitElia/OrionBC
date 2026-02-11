from __future__ import print_function

import numpy as np
import torch
import torch.utils
import pandas as pd
from torch.utils.data import Dataset

def get_X_features(counts_df, labels, method="mannwhitneyu", n_features=500):
	print(f"Selecting top {n_features} signal features...")
	p_values = []

	# Separate cancer and healthy populations
	cancer_samples = counts_df[labels == 1]
	healthy_samples = counts_df[labels == 0]

	if method == "mannwhitneyu":
		from scipy.stats import mannwhitneyu
		for column in counts_df.columns:
			stat, p = mannwhitneyu(cancer_samples[column], healthy_samples[column])
			p_values.append(p)
		
		# Sort by p-value
		p_series = pd.Series(p_values, index=counts_df.columns)

		# Get the genes with the smallest p-values
		top_features = p_series.nsmallest(n_features).index.tolist()
		return top_features
	
	elif method == "ttest":
		from scipy.stats import ttest_ind
		for column in counts_df.columns:
			stat, p = ttest_ind(cancer_samples[column], healthy_samples[column], 
                                equal_var=False, nan_policy='omit')
			p_values.append(p)
		p_series = pd.Series(p_values, index=counts_df.columns)
		p_series = p_series.dropna()
		# Get the genes with the smallest p-values
		top_features = p_series.nsmallest(n_features).index.tolist()
		return top_features
	
	elif method == "pydeseq2":
		from pydeseq2.dds import DeseqDataSet
		from pydeseq2.ds import DeseqStats
		print("Running PyDESeq2 (this may take a moment)...")
		# PyDESeq2 requires a DataFrame for metadata matching the count matrix index
		clinical_df = pd.DataFrame({'condition': labels}, index=counts_df.index)
		dds = DeseqDataSet(
            counts=counts_df.astype(int),
            clinical=clinical_df,
            design_factors="condition",
            refit_cooks=True,
            n_cpus=4  # Adjust based on available CPU cores
        )
		dds.deseq2()
		stat_res = DeseqStats(dds, contrast=["condition", "1", "0"])
		stat_res.summary()
		res_df = stat_res.results_df
		top_features = res_df.sort_values("padj").dropna().head(n_features).index.tolist()
		return top_features
	else:
		raise ValueError(f"Unknown method: {method}")
	

def get_reference_features(counts_df, n_features=100):
    means = counts_df.mean(axis=0)
    stds = counts_df.std(axis=0)
    # Coefficient of Variation (lower is more stable)
    cv = stds / (means + 1e-6)  # Add epsilon to avoid div by zero
    # Filter: Must have reasonable abundance (e.g., mean count > 10)
    abundant_genes = means[means > 10].index
    # Select the most stable among the abundant ones
    stable_features = cv[abundant_genes].nsmallest(n_features).index.tolist()
    return stable_features
	
class ZINBDataset(Dataset):
	def __init__(self, counts_df, labels, signal_features, ref_features):

		self.labels = torch.FloatTensor(labels).unsqueeze(1) # Shape: [Batch, 1]

		x_data = counts_df[signal_features].values.astype(np.float32)
		self.x = torch.from_numpy(x_data)

		r_data = counts_df[ref_features].values.astype(np.float32)
		self.r = torch.from_numpy(r_data)

		library_size = counts_df.sum(axis=1).values.astype(np.float32)
		self.library_size = torch.from_numpy(library_size).unsqueeze(1)

	def __len__(self):
		return len(self.labels)
	
	def __getitem__(self, idx):
		"""
        Returns:
            x_sample: The signal RNAs for this patient.
            r_sample: The reference RNAs for this patient.
            l_sample: The total library size (useful for scaling).
            label: 1 if Cancer, 0 if Healthy.
        """
		return self.x[idx], self.r[idx], self.library_size[idx], self.labels[idx]
	
	