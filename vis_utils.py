from __future__ import annotations

import copy
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:  # pragma: no cover - fallback when seaborn is unavailable
    sns = None
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    precision_recall_fscore_support,
    roc_auc_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Subset

from model.data_utils import ZINBDataset


def _fit_umap_embedding(z, n_neighbors=20, min_dist=0.1, random_state=42):
    try:
        from umap import UMAP

        emb = UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
        ).fit_transform(z)
        return emb, "UMAP"
    except ImportError:  # pragma: no cover - fallback when umap is unavailable
        emb = PCA(n_components=2, random_state=random_state).fit_transform(z)
        return emb, "PCA fallback"


def _barplot(ax, data, x, y, color):
    if sns is not None:
        sns.barplot(data=data, x=x, y=y, color=color, ax=ax)
        return

    x_vals = data[x].tolist()
    y_vals = data[y].to_numpy(dtype=float)
    ax.bar(x_vals, y_vals, color=color)


def _heatmap(matrix, xticklabels, yticklabels, title, cbar_label, annot=False, fmt=".2f", cmap="Blues"):
    if sns is not None:
        sns.heatmap(
            matrix,
            annot=annot,
            fmt=fmt,
            cmap=cmap,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            cbar_kws={"label": cbar_label},
        )
        plt.title(title)
        return

    im = plt.imshow(matrix, aspect="auto", cmap=cmap)
    plt.title(title)
    plt.xticks(ticks=np.arange(len(xticklabels)), labels=xticklabels, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(len(yticklabels)), labels=yticklabels)
    cbar = plt.colorbar(im)
    cbar.set_label(cbar_label)
    if annot:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                plt.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", color="black")


def _build_loader_from_indices(
    counts_df,
    labels_array,
    signal_features,
    ref_features,
    indices,
    batch_size,
):
    ds = ZINBDataset(counts_df, labels_array, signal_features, ref_features)
    return DataLoader(Subset(ds, indices), batch_size=batch_size, shuffle=False)


def _extract_latents(model_obj, loader_obj, split_name, device):
    model_obj.eval()
    z_chunks = []
    y_chunks = []
    split_chunks = []

    with torch.no_grad():
        for x, r, _, y in loader_obj:
            z, _, _, _, _, _ = model_obj.encoder(x.to(device), r.to(device))
            z_np = z.cpu().numpy()
            y_np = y.numpy().reshape(-1)
            z_chunks.append(z_np)
            y_chunks.append(y_np)
            split_chunks.append(np.array([split_name] * len(y_np)))

    return np.concatenate(z_chunks), np.concatenate(y_chunks), np.concatenate(split_chunks)


def _compute_latent_metrics(z_train, y_train, z_all, y_all, z_test, y_test):
    metrics = {}

    if len(np.unique(y_all)) > 1 and len(z_all) > 2:
        metrics["silhouette"] = float(silhouette_score(z_all, y_all))
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(z_all, y_all))
        metrics["davies_bouldin"] = float(davies_bouldin_score(z_all, y_all))
    else:
        metrics["silhouette"] = np.nan
        metrics["calinski_harabasz"] = np.nan
        metrics["davies_bouldin"] = np.nan

    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(z_train, y_train)
    test_probs = clf.predict_proba(z_test)[:, 1]
    test_pred = (test_probs > 0.5).astype(int)
    metrics["linear_auc_test"] = float(roc_auc_score(y_test, test_probs))
    metrics["linear_acc_test"] = float(accuracy_score(y_test, test_pred))

    return metrics


def _scatter_by_class_and_split(ax, emb, y, split_names, title):
    class_colors = {0: "royalblue", 1: "crimson"}
    split_markers = {"train": "o", "val": "^", "test": "X"}

    for cls in [0, 1]:
        for split_name, marker in split_markers.items():
            idx = (y == cls) & (split_names == split_name)
            if np.any(idx):
                ax.scatter(
                    emb[idx, 0],
                    emb[idx, 1],
                    c=class_colors[cls],
                    marker=marker,
                    s=60,
                    alpha=0.75,
                )

    class_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="crimson", label="Cancer", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="royalblue", label="Healthy", markersize=8),
    ]
    split_handles = [
        Line2D([0], [0], marker="o", color="black", linestyle="None", label="Train", markersize=7),
        Line2D([0], [0], marker="^", color="black", linestyle="None", label="Val", markersize=7),
        Line2D([0], [0], marker="X", color="black", linestyle="None", label="Test", markersize=7),
    ]

    leg1 = ax.legend(handles=class_handles, loc="upper right", title="Class")
    ax.add_artist(leg1)
    ax.legend(handles=split_handles, loc="lower right", title="Split")
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")


def plot_latent_diagnostics(
    model_obj,
    counts_df,
    labels_array,
    signal_features,
    ref_features,
    train_idx,
    val_idx,
    test_idx,
    title_prefix,
    umap_n_neighbors,
    umap_min_dist,
    batch_size=32,
    device="cpu",
):
    train_loader = _build_loader_from_indices(
        counts_df,
        labels_array,
        signal_features,
        ref_features,
        train_idx,
        batch_size=batch_size,
    )
    val_loader = _build_loader_from_indices(
        counts_df,
        labels_array,
        signal_features,
        ref_features,
        val_idx,
        batch_size=batch_size,
    )
    test_loader = _build_loader_from_indices(
        counts_df,
        labels_array,
        signal_features,
        ref_features,
        test_idx,
        batch_size=batch_size,
    )

    z_train, y_train, split_train = _extract_latents(model_obj, train_loader, "train", device=device)
    z_val, y_val, split_val = _extract_latents(model_obj, val_loader, "val", device=device)
    z_test, y_test, split_test = _extract_latents(model_obj, test_loader, "test", device=device)

    z_all = np.vstack([z_train, z_val, z_test])
    y_all = np.concatenate([y_train, y_val, y_test]).astype(int)
    split_all = np.concatenate([split_train, split_val, split_test])

    umap_emb, umap_label = _fit_umap_embedding(
        z_all,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        random_state=42,
    )
    pca_emb = PCA(n_components=2, random_state=42).fit_transform(z_all)

    metrics = _compute_latent_metrics(z_train, y_train, z_all, y_all, z_test, y_test)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _scatter_by_class_and_split(
        axes[0],
        umap_emb,
        y_all,
        split_all,
        f"{title_prefix} - {umap_label} (n={umap_n_neighbors}, min_dist={umap_min_dist})",
    )
    _scatter_by_class_and_split(
        axes[1],
        pca_emb,
        y_all,
        split_all,
        f"{title_prefix} - PCA",
    )
    plt.tight_layout()
    plt.show()

    print(f"\n{title_prefix} latent metrics")
    print("-" * 50)
    print(f"Silhouette score:           {metrics['silhouette']:.4f}")
    print(f"Calinski-Harabasz score:    {metrics['calinski_harabasz']:.4f}")
    print(f"Davies-Bouldin score:       {metrics['davies_bouldin']:.4f}")
    print(f"Linear separability AUC:    {metrics['linear_auc_test']:.4f}")
    print(f"Linear separability ACC:    {metrics['linear_acc_test']:.4f}")
    print("-" * 50)


def plot_history_with_validation(history, title, eval_every=5):
    train_loss = history.get("train_loss", [])
    val_auc = history.get("val_auc", [])
    val_acc = history.get("val_acc", [])

    epochs = np.arange(1, len(train_loss) + 1)
    val_epochs = np.arange(eval_every, eval_every * len(val_auc) + 1, eval_every)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, train_loss, color="navy", linewidth=2, label="Train loss")
    axes[0].set_title(f"{title} - Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    if len(val_auc) > 0:
        axes[1].plot(val_epochs, val_auc, marker="o", color="crimson", label="Val AUC")
        axes[1].plot(val_epochs, val_acc, marker="s", color="darkgreen", label="Val ACC")
    axes[1].set_title(f"{title} - Validation metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def extract_latents_by_indices(model_obj, dataset_obj, split_index_map, batch_size, device):
    model_obj.eval()

    all_z, all_y, all_split = [], [], []

    with torch.no_grad():
        for split_name, split_indices in split_index_map.items():
            loader = DataLoader(Subset(dataset_obj, split_indices), batch_size=batch_size, shuffle=False)
            for x, r, _, y in loader:
                z, _, _, _, _, _ = model_obj.encoder(x.to(device), r.to(device))
                all_z.append(z.detach().cpu().numpy())
                all_y.append(y.detach().cpu().numpy().reshape(-1))
                all_split.append(np.array([split_name] * len(y)))

    z_all = np.concatenate(all_z, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    split_all = np.concatenate(all_split, axis=0)
    return z_all, y_all, split_all


def _extract_latents_from_state(
    model_obj,
    state_dict,
    dataset_obj,
    split_index_map,
    batch_size,
    device,
):
    model_copy = copy.deepcopy(model_obj)
    model_copy.load_state_dict(copy.deepcopy(state_dict))
    model_copy.to(device)
    return extract_latents_by_indices(
        model_copy,
        dataset_obj,
        split_index_map,
        batch_size,
        device,
    )


def extract_latents_predictions_by_indices(model_obj, dataset_obj, split_index_map, batch_size, device):
    model_obj.eval()
    multiclass = int(getattr(model_obj, "n_classes", 1)) > 1

    all_z, all_y, all_split, all_pred, all_prob = [], [], [], [], []

    with torch.no_grad():
        for split_name, split_indices in split_index_map.items():
            loader = DataLoader(Subset(dataset_obj, split_indices), batch_size=batch_size, shuffle=False)
            for x, r, _, y in loader:
                x = x.to(device)
                r = r.to(device)
                z, _, _, _, _, _ = model_obj.encoder(x, r)
                out = model_obj(x, r)
                logits = out["pred_logit"]

                if multiclass:
                    prob = torch.softmax(logits, dim=1).detach().cpu().numpy()
                    pred = np.argmax(prob, axis=1)
                else:
                    prob = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
                    pred = (prob > 0.5).astype(int)

                all_z.append(z.detach().cpu().numpy())
                all_y.append(y.detach().cpu().numpy().reshape(-1))
                all_split.append(np.array([split_name] * len(y)))
                all_pred.append(pred)
                all_prob.append(prob)

    z_all = np.concatenate(all_z, axis=0)
    y_all = np.concatenate(all_y, axis=0).astype(int)
    split_all = np.concatenate(all_split, axis=0)
    pred_all = np.concatenate(all_pred, axis=0).astype(int)

    if multiclass:
        prob_all = np.vstack(all_prob)
    else:
        prob_all = np.concatenate(all_prob, axis=0)

    return z_all, y_all, split_all, pred_all, prob_all


def _scatter_binary_projection(ax, emb, y_all, split_all, title):
    class_colors = {0: "royalblue", 1: "crimson"}
    class_labels = {0: "Non-cancer", 1: "Cancer"}
    split_markers = {"train": "o", "val": "^", "test": "X"}

    for cls in [0, 1]:
        for split_name, marker in split_markers.items():
            idx = (y_all == cls) & (split_all == split_name)
            if np.any(idx):
                ax.scatter(
                    emb[idx, 0],
                    emb[idx, 1],
                    c=class_colors[cls],
                    marker=marker,
                    s=26,
                    alpha=0.75,
                )

    class_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=class_colors[1],
            label=class_labels[1],
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=class_colors[0],
            label=class_labels[0],
            markersize=8,
        ),
    ]
    split_handles = [
        Line2D([0], [0], marker="o", color="black", linestyle="None", label="Train", markersize=7),
        Line2D([0], [0], marker="^", color="black", linestyle="None", label="Val", markersize=7),
        Line2D([0], [0], marker="X", color="black", linestyle="None", label="Test", markersize=7),
    ]

    leg1 = ax.legend(handles=class_handles, loc="upper right", title="Class")
    ax.add_artist(leg1)
    ax.legend(handles=split_handles, loc="lower right", title="Split")
    ax.set_title(title)
    ax.set_xlabel("PCA-1")
    ax.set_ylabel("PCA-2")


def plot_binary_epoch_pca_comparison(
    model_obj,
    initial_state_dict,
    dataset_obj,
    split_index_map,
    batch_size,
    device,
    title,
):
    z_init, y_init, split_init = _extract_latents_from_state(
        model_obj=model_obj,
        state_dict=initial_state_dict,
        dataset_obj=dataset_obj,
        split_index_map=split_index_map,
        batch_size=batch_size,
        device=device,
    )
    z_last, y_last, split_last = extract_latents_by_indices(
        model_obj=model_obj,
        dataset_obj=dataset_obj,
        split_index_map=split_index_map,
        batch_size=batch_size,
        device=device,
    )

    y_init = y_init.astype(int)
    y_last = y_last.astype(int)

    shared_pca = PCA(n_components=2, random_state=42).fit(np.vstack([z_init, z_last]))
    emb_init = shared_pca.transform(z_init)
    emb_last = shared_pca.transform(z_last)

    x_min = float(min(emb_init[:, 0].min(), emb_last[:, 0].min()))
    x_max = float(max(emb_init[:, 0].max(), emb_last[:, 0].max()))
    y_min = float(min(emb_init[:, 1].min(), emb_last[:, 1].min()))
    y_max = float(max(emb_init[:, 1].max(), emb_last[:, 1].max()))
    x_pad = max((x_max - x_min) * 0.05, 1e-6)
    y_pad = max((y_max - y_min) * 0.05, 1e-6)
    xlim = (x_min - x_pad, x_max + x_pad)
    ylim = (y_min - y_pad, y_max + y_pad)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _scatter_binary_projection(
        axes[0],
        emb_init,
        y_init,
        split_init,
        f"{title} - PCA @ Epoch 0 (random init)",
    )
    _scatter_binary_projection(
        axes[1],
        emb_last,
        y_last,
        split_last,
        f"{title} - PCA @ Last Epoch",
    )
    for ax in axes:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
    plt.tight_layout()
    plt.show()


def plot_stage1_latent(model_obj, dataset_obj, split_index_map, batch_size, device, title):
    z_all, y_all, split_all = extract_latents_by_indices(
        model_obj, dataset_obj, split_index_map, batch_size, device
    )

    y_all = y_all.astype(int)
    umap_emb, umap_label = _fit_umap_embedding(z_all, n_neighbors=20, min_dist=0.1, random_state=42)
    pca_emb = PCA(n_components=2, random_state=42).fit_transform(z_all)

    class_colors = {0: "royalblue", 1: "crimson"}
    split_markers = {"train": "o", "val": "^", "test": "X"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, emb, name in [
        (axes[0], umap_emb, umap_label),
        (axes[1], pca_emb, "PCA"),
    ]:
        for cls in [0, 1]:
            for split_name, marker in split_markers.items():
                idx = (y_all == cls) & (split_all == split_name)
                if np.any(idx):
                    ax.scatter(
                        emb[idx, 0],
                        emb[idx, 1],
                        c=class_colors[cls],
                        marker=marker,
                        s=26,
                        alpha=0.75,
                    )

        class_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="crimson", label="Cancer", markersize=8),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="royalblue", label="Non-cancer", markersize=8),
        ]
        split_handles = [
            Line2D([0], [0], marker="o", color="black", linestyle="None", label="Train", markersize=7),
            Line2D([0], [0], marker="^", color="black", linestyle="None", label="Val", markersize=7),
            Line2D([0], [0], marker="X", color="black", linestyle="None", label="Test", markersize=7),
        ]
        leg1 = ax.legend(handles=class_handles, loc="upper right", title="Class")
        ax.add_artist(leg1)
        ax.legend(handles=split_handles, loc="lower right", title="Split")

        ax.set_title(f"{title} - {name}")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

    plt.tight_layout()
    plt.show()


def _safe_macro_auc(y_true, probs):
    try:
        return float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


def build_stage2_per_class_metrics(y_true, y_pred, y_prob, class_index_to_name):
    classes = sorted(int(k) for k in class_index_to_name.keys())

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=classes,
        zero_division=0,
    )

    y_true_bin = label_binarize(y_true, classes=classes)
    rows = []

    for i, cls in enumerate(classes):
        cls_name = class_index_to_name[cls]

        try:
            auc_ovr = float(roc_auc_score(y_true_bin[:, i], y_prob[:, i]))
        except ValueError:
            auc_ovr = np.nan

        try:
            pr_auc_ovr = float(average_precision_score(y_true_bin[:, i], y_prob[:, i]))
        except ValueError:
            pr_auc_ovr = np.nan

        rows.append(
            {
                "class_idx": cls,
                "class_name": cls_name,
                "support": int(support[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "roc_auc_ovr": auc_ovr,
                "pr_auc_ovr": pr_auc_ovr,
            }
        )

    return pd.DataFrame(rows)


def _build_stage2_class_color_map(classes):
    palette = plt.cm.get_cmap("tab20", max(len(classes), 13))
    return {int(cls): palette(i) for i, cls in enumerate(classes)}


def _fit_stage2_projection(z_all, y_all, method="lda_fallback"):
    method = str(method).strip().lower()
    if method in {"lda_fallback", "lda"}:
        try:
            lda = LinearDiscriminantAnalysis(n_components=2)
            emb = lda.fit_transform(z_all, y_all)
            return emb, "LDA"
        except Exception:
            if method == "lda":
                raise
            emb = PCA(n_components=2, random_state=42).fit_transform(z_all)
            return emb, "PCA fallback"

    if method == "pca":
        emb = PCA(n_components=2, random_state=42).fit_transform(z_all)
        return emb, "PCA"

    raise ValueError("projection_method must be one of {'lda_fallback', 'lda', 'pca'}")


def _plot_stage2_trueclass_projection(emb, y_all, class_index_to_name, class_color_map, title, projection_label):
    classes = sorted(np.unique(y_all).tolist())

    plt.figure(figsize=(10, 7))
    for cls in classes:
        idx = y_all == cls
        plt.scatter(
            emb[idx, 0],
            emb[idx, 1],
            s=16,
            alpha=0.75,
            color=class_color_map[int(cls)],
            label=class_index_to_name[int(cls)],
        )

    plt.title(f"{title} - {projection_label} by true class")
    plt.xlabel(f"{projection_label}-1")
    plt.ylabel(f"{projection_label}-2")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    plt.tight_layout()
    plt.show()


def _plot_stage2_correctness_projection(emb, y_all, pred_all, title, projection_label):
    correct = pred_all == y_all

    plt.figure(figsize=(9, 7))
    plt.scatter(
        emb[correct, 0],
        emb[correct, 1],
        s=16,
        alpha=0.35,
        color="seagreen",
        label="Correct",
    )
    plt.scatter(
        emb[~correct, 0],
        emb[~correct, 1],
        s=28,
        alpha=0.9,
        color="crimson",
        marker="x",
        label="Incorrect",
    )
    plt.title(f"{title} - {projection_label} prediction correctness")
    plt.xlabel(f"{projection_label}-1")
    plt.ylabel(f"{projection_label}-2")
    plt.legend()
    plt.tight_layout()
    plt.show()


def _plot_stage2_centroid_distances(z_all, y_all, class_index_to_name, title):
    classes = sorted(np.unique(y_all).tolist())
    names = [class_index_to_name[int(c)] for c in classes]

    centroids = np.vstack([z_all[y_all == c].mean(axis=0) for c in classes])
    dist = cdist(centroids, centroids, metric="euclidean")
    cmap_name = "mako" if sns is not None else "viridis"

    plt.figure(figsize=(11, 9))
    _heatmap(
        dist,
        xticklabels=names,
        yticklabels=names,
        title=f"{title} - Latent centroid distance heatmap",
        cbar_label="Euclidean distance",
        annot=False,
        cmap=cmap_name,
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def _plot_stage2_silhouette_by_class(z_all, y_all, class_index_to_name, title):
    if len(np.unique(y_all)) < 2:
        print("Not enough classes to compute silhouette scores.")
        return

    s = silhouette_samples(z_all, y_all)
    classes = sorted(np.unique(y_all).tolist())
    rows = []
    for cls in classes:
        rows.append(
            {
                "class_name": class_index_to_name[int(cls)],
                "silhouette_mean": float(np.mean(s[y_all == cls])),
            }
        )
    df = pd.DataFrame(rows).sort_values("silhouette_mean", ascending=False)

    plt.figure(figsize=(11, 5))
    _barplot(plt.gca(), data=df, x="class_name", y="silhouette_mean", color="steelblue")
    plt.title(f"{title} - Per-class silhouette score")
    plt.xlabel("Class")
    plt.ylabel("Mean silhouette")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def _top_confused_pairs(y_true, y_pred, classes, n_pairs=6):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    pairs = []
    for i, cls_i in enumerate(classes):
        for j, cls_j in enumerate(classes):
            if i == j:
                continue
            count = int(cm[i, j])
            if count > 0:
                pairs.append((count, int(cls_i), int(cls_j)))

    pairs.sort(reverse=True, key=lambda x: x[0])
    return [(a, b, c) for a, b, c in pairs[:n_pairs]]


def _plot_stage2_top_confused_pairs(
    emb,
    y_all,
    split_all,
    class_index_to_name,
    class_color_map,
    confused_pairs,
    title,
    projection_label,
    pair_point_scope,
):
    if len(confused_pairs) == 0:
        print("No off-diagonal confusion pairs to visualize.")
        return

    if pair_point_scope == "test":
        scope_mask = split_all == "test"
    elif pair_point_scope == "all":
        scope_mask = np.ones_like(y_all, dtype=bool)
    else:
        raise ValueError("pair_point_scope must be one of {'test', 'all'}")

    n_cols = 3
    n_rows = int(np.ceil(len(confused_pairs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.5 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i, (count, cls_true, cls_pred) in enumerate(confused_pairs):
        ax = axes[i]
        mask = scope_mask & ((y_all == cls_true) | (y_all == cls_pred))

        for cls in [cls_true, cls_pred]:
            cls_mask = mask & (y_all == cls)
            if np.any(cls_mask):
                ax.scatter(
                    emb[cls_mask, 0],
                    emb[cls_mask, 1],
                    color=class_color_map[int(cls)],
                    s=18,
                    alpha=0.75,
                    label=class_index_to_name[int(cls)],
                )
        ax.set_title(
            f"{class_index_to_name[cls_true]} vs {class_index_to_name[cls_pred]}\n"
            f"test confusions: {count}"
        )
        ax.set_xlabel(f"{projection_label}-1")
        ax.set_ylabel(f"{projection_label}-2")
        ax.legend(loc="best", fontsize=8, frameon=False)

    for j in range(len(confused_pairs), len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"{title} - Top confused class pairs", y=1.02)
    plt.tight_layout()
    plt.show()


def _top_separated_pairs(z_all, y_all, classes, n_pairs=6):
    centroids = []
    present_classes = []

    for cls in classes:
        mask = y_all == cls
        if np.any(mask):
            present_classes.append(int(cls))
            centroids.append(z_all[mask].mean(axis=0))

    if len(present_classes) < 2:
        return []

    centroids = np.vstack(centroids)
    dist = cdist(centroids, centroids, metric="euclidean")

    pairs = []
    for i in range(len(present_classes)):
        for j in range(i + 1, len(present_classes)):
            pairs.append((float(dist[i, j]), int(present_classes[i]), int(present_classes[j])))

    pairs.sort(reverse=True, key=lambda x: x[0])
    return pairs[:n_pairs]


def _plot_stage2_top_separated_pairs(
    emb,
    y_all,
    split_all,
    class_index_to_name,
    class_color_map,
    separated_pairs,
    title,
    projection_label,
    pair_point_scope,
):
    if len(separated_pairs) == 0:
        print("No class pairs available for separation visualization.")
        return

    if pair_point_scope == "test":
        scope_mask = split_all == "test"
    elif pair_point_scope == "all":
        scope_mask = np.ones_like(y_all, dtype=bool)
    else:
        raise ValueError("pair_point_scope must be one of {'test', 'all'}")

    n_cols = 3
    n_rows = int(np.ceil(len(separated_pairs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.5 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i, (dist_value, cls_a, cls_b) in enumerate(separated_pairs):
        ax = axes[i]
        mask = scope_mask & ((y_all == cls_a) | (y_all == cls_b))

        for cls in [cls_a, cls_b]:
            cls_mask = mask & (y_all == cls)
            if np.any(cls_mask):
                ax.scatter(
                    emb[cls_mask, 0],
                    emb[cls_mask, 1],
                    color=class_color_map[int(cls)],
                    s=18,
                    alpha=0.75,
                    label=class_index_to_name[int(cls)],
                )
        ax.set_title(
            f"{class_index_to_name[cls_a]} vs {class_index_to_name[cls_b]}\n"
            f"latent centroid dist: {dist_value:.3f}"
        )
        ax.set_xlabel(f"{projection_label}-1")
        ax.set_ylabel(f"{projection_label}-2")
        ax.legend(loc="best", fontsize=8, frameon=False)

    for j in range(len(separated_pairs), len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"{title} - Top separated class pairs", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_stage2_diagnostics(
    model_obj,
    dataset_obj,
    split_index_map,
    batch_size,
    device,
    class_index_to_name,
    y_true_test,
    y_pred_test,
    y_prob_test,
    title,
    n_top_confused_pairs=6,
    n_top_separated_pairs=6,
    projection_method="lda_fallback",
    pair_point_scope="test",
):
    classes = sorted(int(k) for k in class_index_to_name.keys())

    z_all, y_all, split_all, pred_all, _ = extract_latents_predictions_by_indices(
        model_obj,
        dataset_obj,
        split_index_map,
        batch_size,
        device,
    )

    emb_all, emb_label = _fit_stage2_projection(z_all, y_all, method=projection_method)
    class_color_map = _build_stage2_class_color_map(classes)

    per_class_df = build_stage2_per_class_metrics(
        y_true=y_true_test,
        y_pred=y_pred_test,
        y_prob=y_prob_test,
        class_index_to_name=class_index_to_name,
    )

    print("\nStage 2 per-class metrics (test split)")
    print("-" * 90)
    display_df = per_class_df.copy()
    for col in ["precision", "recall", "f1", "roc_auc_ovr", "pr_auc_ovr"]:
        display_df[col] = display_df[col].round(4)
    print(display_df.to_string(index=False))

    cm_counts = confusion_matrix(y_true_test, y_pred_test, labels=classes)
    row_sum = cm_counts.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_counts, row_sum, out=np.zeros_like(cm_counts, dtype=float), where=row_sum > 0)

    class_names = [class_index_to_name[c] for c in classes]

    plt.figure(figsize=(12, 9))
    _heatmap(
        cm_norm,
        xticklabels=class_names,
        yticklabels=class_names,
        title=f"{title} - Row-normalized confusion matrix (test)",
        cbar_label="Row-normalized rate",
        annot=True,
        fmt=".2f",
        cmap="Blues",
    )
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    _barplot(axes[0], data=per_class_df, x="class_name", y="roc_auc_ovr", color="teal")
    axes[0].set_title(f"{title} - Per-class ROC-AUC (OvR)")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("ROC-AUC")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=45)

    _barplot(axes[1], data=per_class_df, x="class_name", y="pr_auc_ovr", color="darkorange")
    axes[1].set_title(f"{title} - Per-class PR-AUC (OvR)")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("PR-AUC")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()

    _plot_stage2_trueclass_projection(
        emb_all,
        y_all,
        class_index_to_name,
        class_color_map,
        title,
        emb_label,
    )
    _plot_stage2_correctness_projection(emb_all, y_all, pred_all, title, emb_label)
    _plot_stage2_centroid_distances(z_all, y_all, class_index_to_name, title)
    _plot_stage2_silhouette_by_class(z_all, y_all, class_index_to_name, title)

    confused_pairs = _top_confused_pairs(y_true_test, y_pred_test, classes, n_pairs=n_top_confused_pairs)
    _plot_stage2_top_confused_pairs(
        emb_all,
        y_all,
        split_all,
        class_index_to_name,
        class_color_map,
        confused_pairs,
        title,
        emb_label,
        pair_point_scope,
    )

    if pair_point_scope == "test":
        sep_mask = split_all == "test"
    elif pair_point_scope == "all":
        sep_mask = np.ones_like(y_all, dtype=bool)
    else:
        raise ValueError("pair_point_scope must be one of {'test', 'all'}")

    separated_pairs = _top_separated_pairs(z_all[sep_mask], y_all[sep_mask], classes, n_pairs=n_top_separated_pairs)
    _plot_stage2_top_separated_pairs(
        emb_all,
        y_all,
        split_all,
        class_index_to_name,
        class_color_map,
        separated_pairs,
        title,
        emb_label,
        pair_point_scope,
    )

    return per_class_df
