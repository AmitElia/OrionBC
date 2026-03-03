from __future__ import annotations

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


def collect_latent_snapshot(model_obj, dataset_obj, split_index_map, batch_size, device):
    z_all, y_all, split_all = extract_latents_by_indices(
        model_obj=model_obj,
        dataset_obj=dataset_obj,
        split_index_map=split_index_map,
        batch_size=batch_size,
        device=device,
    )
    return {
        "z": z_all,
        "y": y_all.astype(int),
        "split": split_all,
    }


def plot_pca3d_latent_evolution(
    latent_snapshots,
    class_index_to_name,
    title,
    split_order=("train", "val", "test"),
):
    if latent_snapshots is None or len(latent_snapshots) == 0:
        print("No latent snapshots available for PCA 3D evolution.")
        return None

    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "plotly is required for 3D latent animation. Install with: pip install plotly"
        ) from exc

    snapshots_sorted = sorted(latent_snapshots, key=lambda s: int(s.get("epoch", 0)))
    required_keys = {"z", "y", "split"}
    all_latents = []

    for snapshot in snapshots_sorted:
        missing = required_keys - set(snapshot.keys())
        if missing:
            raise ValueError(f"Latent snapshot missing required keys: {sorted(missing)}")
        all_latents.append(np.asarray(snapshot["z"]))

    z_concat = np.vstack(all_latents)
    pca3 = PCA(n_components=3, random_state=42).fit(z_concat)

    frame_rows = []
    for fallback_idx, snapshot in enumerate(snapshots_sorted, start=1):
        epoch = int(snapshot.get("epoch", fallback_idx))
        z = np.asarray(snapshot["z"])
        y = np.asarray(snapshot["y"]).astype(int).reshape(-1)
        split = np.asarray(snapshot["split"]).astype(str).reshape(-1)

        if len(z) != len(y) or len(y) != len(split):
            raise ValueError("Each latent snapshot must align lengths for z, y, and split.")

        z_pca = pca3.transform(z)
        frame_df = pd.DataFrame(
            {
                "epoch": epoch,
                "pc1": z_pca[:, 0],
                "pc2": z_pca[:, 1],
                "pc3": z_pca[:, 2],
                "class_idx": y,
                "split": split,
            }
        )
        frame_rows.append(frame_df)

    plot_df = pd.concat(frame_rows, ignore_index=True)
    if plot_df.empty:
        print("No rows available after snapshot projection.")
        return None

    plot_df["class_name"] = plot_df["class_idx"].map(
        lambda c: class_index_to_name.get(int(c), str(int(c)))
    )

    split_order = tuple(split_order)
    available_splits = set(plot_df["split"].astype(str).unique().tolist())
    ordered_splits = [s for s in split_order if s in available_splits]
    ordered_splits += sorted(available_splits - set(ordered_splits))
    plot_df["split"] = pd.Categorical(plot_df["split"], categories=ordered_splits, ordered=True)

    axis_ranges = {}
    for col in ("pc1", "pc2", "pc3"):
        min_v = float(plot_df[col].min())
        max_v = float(plot_df[col].max())
        padding = (max_v - min_v) * 0.05 if max_v > min_v else 1.0
        axis_ranges[col] = [min_v - padding, max_v + padding]

    epochs_sorted = sorted(plot_df["epoch"].astype(int).unique().tolist())
    classes_sorted = sorted(plot_df["class_idx"].astype(int).unique().tolist())

    palette = px.colors.qualitative.Plotly + px.colors.qualitative.D3 + px.colors.qualitative.Safe
    color_map = {cls: palette[i % len(palette)] for i, cls in enumerate(classes_sorted)}

    centroid_df = (
        plot_df.groupby(["epoch", "class_idx", "class_name"], as_index=False)[["pc1", "pc2", "pc3"]]
        .mean()
        .sort_values(["class_idx", "epoch"])
    )

    fixed_scene = {
        "xaxis": {"title": "PC1", "range": axis_ranges["pc1"], "autorange": False},
        "yaxis": {"title": "PC2", "range": axis_ranges["pc2"], "autorange": False},
        "zaxis": {"title": "PC3", "range": axis_ranges["pc3"], "autorange": False},
        "aspectmode": "cube",
        "camera": {"eye": {"x": 1.25, "y": 1.25, "z": 1.1}},
    }

    def _build_cloud_trace(df_cls, cls):
        class_name = class_index_to_name.get(int(cls), str(int(cls)))
        customdata = (
            np.column_stack(
                [
                    df_cls["split"].astype(str).to_numpy(),
                    np.full(len(df_cls), class_name),
                    df_cls["epoch"].astype(int).to_numpy(),
                ]
            )
            if len(df_cls) > 0
            else np.empty((0, 3), dtype=object)
        )
        return go.Scatter3d(
            x=df_cls["pc1"],
            y=df_cls["pc2"],
            z=df_cls["pc3"],
            mode="markers",
            name=f"{class_name} cloud",
            legendgroup=f"class-{cls}",
            showlegend=False,
            marker={
                "size": 3,
                "opacity": 0.12,
                "color": color_map[cls],
            },
            customdata=customdata,
            hovertemplate=(
                "Class: %{customdata[1]}<br>"
                "Split: %{customdata[0]}<br>"
                "Epoch: %{customdata[2]}<br>"
                "PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra></extra>"
            ),
        )

    def _build_centroid_trace(df_cent, cls, showlegend):
        class_name = class_index_to_name.get(int(cls), str(int(cls)))
        return go.Scatter3d(
            x=df_cent["pc1"],
            y=df_cent["pc2"],
            z=df_cent["pc3"],
            mode="markers",
            name=class_name,
            legendgroup=f"class-{cls}",
            showlegend=showlegend,
            marker={
                "size": 9,
                "opacity": 1.0,
                "color": color_map[cls],
                "line": {"color": "black", "width": 1},
            },
            hovertemplate=(
                f"Centroid: {class_name}<br>"
                "PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra></extra>"
            ),
        )

    def _build_trail_trace(df_trail, cls):
        class_name = class_index_to_name.get(int(cls), str(int(cls)))
        return go.Scatter3d(
            x=df_trail["pc1"],
            y=df_trail["pc2"],
            z=df_trail["pc3"],
            mode="lines+markers",
            name=f"{class_name} trail",
            legendgroup=f"class-{cls}",
            showlegend=False,
            line={"color": color_map[cls], "width": 4},
            marker={"size": 2, "color": color_map[cls], "opacity": 0.9},
            hovertemplate=(
                f"Trail: {class_name}<br>"
                "PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra></extra>"
            ),
        )

    def _build_epoch_traces(epoch, show_centroid_legend=False):
        traces = []
        for cls in classes_sorted:
            cls_points = plot_df[(plot_df["epoch"] == epoch) & (plot_df["class_idx"] == cls)]
            traces.append(_build_cloud_trace(cls_points, cls))

        for cls in classes_sorted:
            cls_cent = centroid_df[(centroid_df["epoch"] == epoch) & (centroid_df["class_idx"] == cls)]
            traces.append(_build_centroid_trace(cls_cent, cls, showlegend=show_centroid_legend))

        for cls in classes_sorted:
            cls_trail = centroid_df[(centroid_df["epoch"] <= epoch) & (centroid_df["class_idx"] == cls)]
            traces.append(_build_trail_trace(cls_trail, cls))

        return traces

    initial_epoch = epochs_sorted[0]
    initial_traces = _build_epoch_traces(initial_epoch, show_centroid_legend=True)
    frames = [
        go.Frame(
            name=str(epoch),
            data=_build_epoch_traces(epoch, show_centroid_legend=False),
            layout=go.Layout(scene=fixed_scene),
        )
        for epoch in epochs_sorted
    ]

    fig = go.Figure(data=initial_traces, frames=frames)
    fig.update_layout(
        title=title,
        scene=fixed_scene,
        uirevision="pca3d-fixed-scene",
        legend={"title": {"text": "Class centroids"}},
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 1.02,
                "y": 1,
                "xanchor": "left",
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 500, "redraw": True},
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                                "mode": "immediate",
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Epoch: "},
                "pad": {"t": 40},
                "steps": [
                    {
                        "label": str(epoch),
                        "method": "animate",
                        "args": [
                            [str(epoch)],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                                "mode": "immediate",
                            },
                        ],
                    }
                    for epoch in epochs_sorted
                ],
            }
        ],
    )

    for frame in fig.frames:
        frame.layout = go.Layout(scene=fixed_scene)

    fig.show()
    return fig


def plot_stage1_pca3d_evolution(latent_snapshots, title="Stage 1 PCA 3D latent evolution"):
    class_map = {0: "Healthy", 1: "Cancer"}
    return plot_pca3d_latent_evolution(latent_snapshots, class_map, title)


def plot_stage2_pca3d_evolution(
    latent_snapshots,
    class_index_to_name,
    title="Stage 2 PCA 3D latent evolution",
):
    return plot_pca3d_latent_evolution(latent_snapshots, class_index_to_name, title)


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
                        alpha=0.25,
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


def _plot_stage2_trueclass_umap(umap_emb, y_all, class_index_to_name, title):
    classes = sorted(np.unique(y_all).tolist())
    palette = plt.cm.get_cmap("tab20", max(len(classes), 13))

    plt.figure(figsize=(10, 7))
    for i, cls in enumerate(classes):
        idx = y_all == cls
        plt.scatter(
            umap_emb[idx, 0],
            umap_emb[idx, 1],
            s=16,
            alpha=0.75,
            color=palette(i),
            label=class_index_to_name[int(cls)],
        )

    plt.title(f"{title} - UMAP by true class")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    plt.tight_layout()
    plt.show()


def _plot_stage2_correctness_umap(umap_emb, y_all, pred_all, title):
    correct = pred_all == y_all

    plt.figure(figsize=(9, 7))
    plt.scatter(
        umap_emb[correct, 0],
        umap_emb[correct, 1],
        s=16,
        alpha=0.35,
        color="seagreen",
        label="Correct",
    )
    plt.scatter(
        umap_emb[~correct, 0],
        umap_emb[~correct, 1],
        s=28,
        alpha=0.9,
        color="crimson",
        marker="x",
        label="Incorrect",
    )
    plt.title(f"{title} - UMAP prediction correctness")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend()
    plt.tight_layout()
    plt.show()


def _plot_stage2_lda(z_all, y_all, class_index_to_name, title):
    classes = sorted(np.unique(y_all).tolist())

    try:
        lda = LinearDiscriminantAnalysis(n_components=2)
        emb = lda.fit_transform(z_all, y_all)
        emb_title = "LDA"
    except Exception:
        emb = PCA(n_components=2, random_state=42).fit_transform(z_all)
        emb_title = "PCA fallback"

    palette = plt.cm.get_cmap("tab20", max(len(classes), 13))
    plt.figure(figsize=(10, 7))
    for i, cls in enumerate(classes):
        idx = y_all == cls
        plt.scatter(
            emb[idx, 0],
            emb[idx, 1],
            s=16,
            alpha=0.75,
            color=palette(i),
            label=class_index_to_name[int(cls)],
        )

    plt.title(f"{title} - {emb_title} by true class")
    plt.xlabel(f"{emb_title}-1")
    plt.ylabel(f"{emb_title}-2")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    plt.tight_layout()
    plt.show()


def _plot_stage2_centroid_distances(z_all, y_all, class_index_to_name, title):
    classes = sorted(np.unique(y_all).tolist())
    names = [class_index_to_name[int(c)] for c in classes]

    centroids = np.vstack([z_all[y_all == c].mean(axis=0) for c in classes])
    dist = cdist(centroids, centroids, metric="euclidean")

    plt.figure(figsize=(11, 9))
    _heatmap(
        dist,
        xticklabels=names,
        yticklabels=names,
        title=f"{title} - Latent centroid distance heatmap",
        cbar_label="Euclidean distance",
        annot=False,
        cmap="mako",
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


def _rank_confusion_pairs(y_true, y_pred, classes, n_pairs=6, mode="most"):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    support = cm.sum(axis=1)
    pairs = []

    for i, cls_true in enumerate(classes):
        for j, cls_pred in enumerate(classes):
            if i == j:
                continue
            pairs.append(
                {
                    "count": int(cm[i, j]),
                    "cls_true": int(cls_true),
                    "cls_pred": int(cls_pred),
                    "support_true": int(support[i]),
                }
            )

    if mode == "most":
        pairs.sort(
            key=lambda item: (
                -item["count"],
                -item["support_true"],
                item["cls_true"],
                item["cls_pred"],
            )
        )
    elif mode == "least":
        pairs.sort(
            key=lambda item: (
                item["count"],
                -item["support_true"],
                item["cls_true"],
                item["cls_pred"],
            )
        )
    else:
        raise ValueError("mode must be either 'most' or 'least'")

    return [(p["count"], p["cls_true"], p["cls_pred"]) for p in pairs[:n_pairs]]


def _top_confused_pairs(y_true, y_pred, classes, n_pairs=6):
    return _rank_confusion_pairs(y_true, y_pred, classes, n_pairs=n_pairs, mode="most")


def _least_confused_pairs(y_true, y_pred, classes, n_pairs=6):
    return _rank_confusion_pairs(y_true, y_pred, classes, n_pairs=n_pairs, mode="least")


def _plot_stage2_confusion_pairs(umap_emb, y_all, class_index_to_name, pair_rows, title, heading):
    if len(pair_rows) == 0:
        print("No class pairs available for confusion-based visualization.")
        return

    n_cols = 3
    n_rows = int(np.ceil(len(pair_rows) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.5 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i, (count, cls_true, cls_pred) in enumerate(pair_rows):
        ax = axes[i]
        mask = (y_all == cls_true) | (y_all == cls_pred)

        ax.scatter(
            umap_emb[mask, 0],
            umap_emb[mask, 1],
            c=np.where(y_all[mask] == cls_true, "royalblue", "darkorange"),
            s=18,
            alpha=0.75,
        )
        ax.set_title(
            f"{class_index_to_name[cls_true]} vs {class_index_to_name[cls_pred]}\n"
            f"test confusions: {count}"
        )
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")

    for j in range(len(pair_rows), len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"{title} - {heading}", y=1.02)
    plt.tight_layout()
    plt.show()


def _plot_stage2_top_confused_pairs(umap_emb, y_all, class_index_to_name, confused_pairs, title):
    _plot_stage2_confusion_pairs(
        umap_emb=umap_emb,
        y_all=y_all,
        class_index_to_name=class_index_to_name,
        pair_rows=confused_pairs,
        title=title,
        heading="Top confused class pairs",
    )


def _plot_stage2_least_confused_pairs(umap_emb, y_all, class_index_to_name, least_confused_pairs, title):
    _plot_stage2_confusion_pairs(
        umap_emb=umap_emb,
        y_all=y_all,
        class_index_to_name=class_index_to_name,
        pair_rows=least_confused_pairs,
        title=title,
        heading="Least-confused class pairs",
    )


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


def _plot_stage2_top_separated_pairs(umap_emb, y_all, class_index_to_name, separated_pairs, title):
    if len(separated_pairs) == 0:
        print("No class pairs available for separation visualization.")
        return

    n_cols = 3
    n_rows = int(np.ceil(len(separated_pairs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.5 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i, (dist_value, cls_a, cls_b) in enumerate(separated_pairs):
        ax = axes[i]
        mask = (y_all == cls_a) | (y_all == cls_b)

        ax.scatter(
            umap_emb[mask, 0],
            umap_emb[mask, 1],
            c=np.where(y_all[mask] == cls_a, "royalblue", "darkorange"),
            s=18,
            alpha=0.75,
        )
        ax.set_title(
            f"{class_index_to_name[cls_a]} vs {class_index_to_name[cls_b]}\n"
            f"latent centroid dist: {dist_value:.3f}"
        )
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")

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
    n_least_confused_pairs=6,
    n_top_separated_pairs=None,
):
    classes = sorted(int(k) for k in class_index_to_name.keys())

    z_all, y_all, _, pred_all, _ = extract_latents_predictions_by_indices(
        model_obj,
        dataset_obj,
        split_index_map,
        batch_size,
        device,
    )

    umap_emb, _ = _fit_umap_embedding(z_all, n_neighbors=20, min_dist=0.15, random_state=42)

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

    _plot_stage2_trueclass_umap(umap_emb, y_all, class_index_to_name, title)
    _plot_stage2_correctness_umap(umap_emb, y_all, pred_all, title)
    _plot_stage2_lda(z_all, y_all, class_index_to_name, title)
    _plot_stage2_centroid_distances(z_all, y_all, class_index_to_name, title)

    confused_pairs = _top_confused_pairs(y_true_test, y_pred_test, classes, n_pairs=n_top_confused_pairs)
    _plot_stage2_top_confused_pairs(umap_emb, y_all, class_index_to_name, confused_pairs, title)

    if n_top_separated_pairs is not None and n_least_confused_pairs == 6:
        print(
            "n_top_separated_pairs is deprecated and will be removed in a future version. "
            "Using it as n_least_confused_pairs."
        )
        n_least_confused_pairs = n_top_separated_pairs

    least_confused_pairs = _least_confused_pairs(
        y_true_test,
        y_pred_test,
        classes,
        n_pairs=n_least_confused_pairs,
    )
    _plot_stage2_least_confused_pairs(umap_emb, y_all, class_index_to_name, least_confused_pairs, title)

    return per_class_df
