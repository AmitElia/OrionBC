from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class OrionEncoder(nn.Module):
    def __init__(self, x_dim, r_dim, z_dim=32, l_dim=1):
        super(OrionEncoder, self).__init__()

        self.enc_x_hidden = nn.Linear(x_dim, 128)
        self.bn_x = nn.LayerNorm(128)

        self.z_mu = nn.Linear(128, z_dim)
        self.z_logvar = nn.Linear(128, z_dim)

        self.enc_r_hidden = nn.Linear(r_dim, 32)
        self.bn_r = nn.LayerNorm(32)

        self.l_mu = nn.Linear(32, l_dim)
        self.l_logvar = nn.Linear(32, l_dim)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x, r):
        h_x = F.relu(self.bn_x(self.enc_x_hidden(x)))
        z_mean = self.z_mu(h_x)
        z_logvar = self.z_logvar(h_x)
        z = self.reparameterize(z_mean, z_logvar)

        h_r = F.relu(self.bn_r(self.enc_r_hidden(r)))
        l_mean = self.l_mu(h_r)
        l_logvar = self.l_logvar(h_r)
        l = self.reparameterize(l_mean, l_logvar)

        return z, z_mean, z_logvar, l, l_mean, l_logvar


class ZINBDecoder(nn.Module):
    def __init__(self, z_dim, l_dim, output_dim):
        super(ZINBDecoder, self).__init__()

        self.decoder_hidden = nn.Linear(z_dim + l_dim, 128)
        self.bn_dec = nn.LayerNorm(128)

        self.dec_scale = nn.Linear(128, output_dim)
        self.dec_theta = nn.Linear(128, output_dim)
        self.dec_pi = nn.Linear(128, output_dim)

    def forward(self, z, l):
        combined = torch.cat([z, l], dim=1)
        h = F.relu(self.bn_dec(self.decoder_hidden(combined)))

        scale = F.softmax(self.dec_scale(h), dim=1)
        mean = torch.exp(l) * scale
        dispersion = torch.exp(self.dec_theta(h))
        dropout = torch.sigmoid(self.dec_pi(h))
        return mean, dispersion, dropout


class CancerClassifier(nn.Module):
    def __init__(self, z_dim, n_classes=1):
        super(CancerClassifier, self).__init__()
        if n_classes < 1:
            raise ValueError("n_classes must be >= 1")
        self.n_classes = int(n_classes)
        out_dim = 1 if self.n_classes == 1 else self.n_classes
        self.net = nn.Sequential(
            nn.Linear(z_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, out_dim),
        )

    def forward(self, z):
        return self.net(z)


class OrionVAE(nn.Module):
    def __init__(self, x_dim, r_dim, z_dim=32, l_dim=1, n_classes=1, task_type=None):
        super(OrionVAE, self).__init__()
        self.n_classes = int(n_classes)
        if task_type is None:
            task_type = "binary" if self.n_classes == 1 else "multiclass"
        self.task_type = str(task_type).strip().lower()
        if self.task_type not in {"binary", "multiclass"}:
            raise ValueError("task_type must be 'binary' or 'multiclass'")

        self.encoder = OrionEncoder(x_dim, r_dim, z_dim, l_dim)
        self.decoder = ZINBDecoder(z_dim, l_dim, output_dim=x_dim)
        self.classifier = CancerClassifier(z_dim, n_classes=self.n_classes)

    def forward(self, x, r):
        z, z_mu, z_logvar, l, l_mu, l_logvar = self.encoder(x, r)
        mean, dispersion, dropout = self.decoder(z, l)
        pred_logit = self.classifier(z)
        return {
            "mean": mean,
            "dispersion": dispersion,
            "dropout": dropout,
            "pred_logit": pred_logit,
            "z_mu": z_mu,
            "z_logvar": z_logvar,
            "l_mu": l_mu,
            "l_logvar": l_logvar,
        }


class OrionLoss(nn.Module):
    def __init__(
        self,
        beta=1.0,
        gamma=10.0,
        n_classes=1,
        task_type=None,
        recon_mode="zinb",
        class_weights=None,
        pos_weight=None,
    ):
        super(OrionLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.n_classes = int(n_classes)
        self.recon_mode = str(recon_mode).strip().lower()
        if task_type is None:
            task_type = "binary" if self.n_classes == 1 else "multiclass"
        self.task_type = str(task_type).strip().lower()
        if self.task_type not in {"binary", "multiclass"}:
            raise ValueError("task_type must be 'binary' or 'multiclass'")
        if self.recon_mode not in {"zinb", "mse"}:
            raise ValueError("recon_mode must be 'zinb' or 'mse'")

        if self.task_type == "binary":
            if pos_weight is None:
                self.class_criterion = nn.BCEWithLogitsLoss()
            else:
                self.class_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.class_criterion = nn.CrossEntropyLoss(weight=class_weights)

    def zinb_loss(self, x, mean, disp, pi, eps=1e-10):
        disp = torch.clamp(disp, min=eps, max=1e6)
        pi = torch.clamp(pi, min=eps, max=1.0 - eps)
        mean = torch.clamp(mean, min=eps, max=1e6)

        t1 = torch.lgamma(x + disp) - torch.lgamma(disp) - torch.lgamma(x + 1)
        t2 = (disp + x) * torch.log(1.0 + (mean / disp)) + (x * (torch.log(disp) - torch.log(mean)))
        nb_log_prob = t1 - t2

        zero_case = -torch.log(pi + ((1.0 - pi) * torch.exp(nb_log_prob)) + eps)
        nonzero_case = -torch.log(1.0 - pi + eps) - nb_log_prob

        result = torch.where(x < 1e-8, zero_case, nonzero_case)
        return result.sum()

    def kl_divergence(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def mse_recon_loss(self, x, mean):
        return F.mse_loss(mean, x, reduction="sum")

    def _classification_loss(self, logits, target_y):
        if self.task_type == "binary":
            target_y = target_y.float()
            if target_y.ndim == 1:
                target_y = target_y.unsqueeze(1)
            elif target_y.ndim > 2:
                target_y = target_y.reshape(-1, 1)
            return self.class_criterion(logits, target_y)

        target_y = target_y.long().reshape(-1)
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        return self.class_criterion(logits, target_y)

    def forward(self, outputs, target_x, target_y):
        if self.recon_mode == "zinb":
            recon_loss = self.zinb_loss(
                target_x,
                outputs["mean"],
                outputs["dispersion"],
                outputs["dropout"],
            )
        else:
            recon_loss = self.mse_recon_loss(
                target_x,
                outputs["mean"],
            )

        kl_z = self.kl_divergence(outputs["z_mu"], outputs["z_logvar"])
        kl_l = self.kl_divergence(outputs["l_mu"], outputs["l_logvar"])
        total_kl = kl_z + kl_l

        class_loss = self._classification_loss(outputs["pred_logit"], target_y)
        total_loss = recon_loss + (self.beta * total_kl) + (self.gamma * class_loss)

        return total_loss, recon_loss, total_kl, class_loss


def _is_binary_model(model):
    return int(getattr(model, "n_classes", 1)) == 1


def _safe_binary_auc(targets, preds):
    try:
        return roc_auc_score(targets, preds)
    except ValueError:
        return 0.5


def _safe_multiclass_auc(targets, probs):
    try:
        return roc_auc_score(targets, probs, multi_class="ovr", average="macro")
    except ValueError:
        return np.nan


def _assert_finite_tensor(tensor, name):
    if not torch.isfinite(tensor).all():
        raise ValueError(f"Non-finite values detected in tensor '{name}'")


def _assert_finite_array(array, name):
    if not np.isfinite(array).all():
        raise ValueError(f"Input contains NaN or Inf in '{name}'")


def train_orion(
    model,
    train_loader,
    val_loader,
    epochs=100,
    lr=1e-3,
    device="cuda",
    recon_mode="zinb",
):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = OrionLoss(
        beta=0.01,
        gamma=10.0,
        n_classes=getattr(model, "n_classes", 1),
        task_type=getattr(model, "task_type", None),
        recon_mode=recon_mode,
    ).to(device)
    history = {"train_loss": [], "val_auc": [], "val_acc": []}

    print(f"Starting training on {device} for {epochs} epochs (recon_mode={recon_mode})...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for x, r, _, y in train_loader:
            x, r, y = x.to(device), r.to(device), y.to(device)
            if x.dim() == 1:
                x = x.unsqueeze(0)
                r = r.unsqueeze(0)
                if y.ndim == 0:
                    y = y.unsqueeze(0)

            _assert_finite_tensor(x, "x")
            _assert_finite_tensor(r, "r")
            _assert_finite_tensor(y.float(), "y")

            optimizer.zero_grad()
            outputs = model(x, r)
            _assert_finite_tensor(outputs["pred_logit"], "pred_logit")
            loss, _, _, _ = criterion(outputs, x, y)
            if not torch.isfinite(loss):
                raise ValueError(
                    f"Non-finite loss encountered. "
                    f"recon_mode={recon_mode}, epoch={epoch + 1}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / max(len(train_loader), 1)
        history["train_loss"].append(avg_train_loss)

        if (epoch + 1) % 5 == 0:
            val_auc, val_acc = evaluate_model(model, val_loader, device)
            history["val_auc"].append(val_auc)
            history["val_acc"].append(val_acc)
            print(
                f"Epoch {epoch + 1:03d} | "
                f"Loss: {avg_train_loss:.4f} | "
                f"Val AUC: {val_auc:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

    return history


def predict_probabilities(model, loader, device, include_targets=True):
    model.eval()
    all_preds = []
    all_targets = []
    binary_mode = _is_binary_model(model)

    with torch.no_grad():
        for x, r, _, y in loader:
            x, r = x.to(device), r.to(device)
            if x.dim() == 1:
                x = x.unsqueeze(0)
                r = r.unsqueeze(0)

            outputs = model(x, r)
            logits = outputs["pred_logit"]

            if binary_mode:
                probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
            else:
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            _assert_finite_array(probs, "predicted_probabilities")
            all_preds.append(probs)

            if include_targets:
                if binary_mode:
                    target_np = y.detach().cpu().numpy().reshape(-1).astype(np.float32)
                else:
                    target_np = y.detach().cpu().numpy().reshape(-1).astype(np.int64)
                _assert_finite_array(target_np, "targets")
                all_targets.append(target_np)

    if binary_mode:
        preds = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.float32)
    else:
        preds = np.vstack(all_preds) if all_preds else np.zeros((0, int(getattr(model, "n_classes", 2))), dtype=np.float32)

    if not include_targets:
        return preds

    targets = np.concatenate(all_targets) if all_targets else np.array([], dtype=np.float32)
    return preds, targets


def evaluate_model(model, loader, device):
    all_preds, all_targets = predict_probabilities(model, loader, device, include_targets=True)
    _assert_finite_array(all_preds, "all_preds")
    _assert_finite_array(all_targets, "all_targets")

    if _is_binary_model(model):
        auc = _safe_binary_auc(all_targets, all_preds)
        preds_binary = (all_preds > 0.5).astype(int)
        acc = accuracy_score(all_targets, preds_binary)
        return auc, acc

    auc = _safe_multiclass_auc(all_targets, all_preds)
    preds_class = np.argmax(all_preds, axis=1)
    acc = accuracy_score(all_targets, preds_class)
    return auc, acc


def plot_training_results(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(epochs, history["train_loss"], "b-", label="Total Training Loss")
    ax1.set_title("Training Loss per Epoch")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    num_auc_records = len(history["val_auc"])
    if num_auc_records > 0:
        interval = len(history["train_loss"]) // max(num_auc_records, 1)
        auc_x_axis = [i * interval for i in range(1, num_auc_records + 1)]

        ax2.plot(auc_x_axis, history["val_auc"], "r-o", label="Validation ROC-AUC")
        ax2.set_title("Validation Performance (AUC)")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("AUC Score")
        ax2.legend()
        ax2.grid(True)
    else:
        ax2.set_title("Validation AUC (No data recorded)")

    plt.tight_layout()
    plt.show()


def evaluate_test_set(model, test_loader, loss_fn, device):
    model.eval()
    all_preds = []
    all_targets = []
    total_test_loss = 0.0
    binary_mode = _is_binary_model(model)

    with torch.no_grad():
        for x, r, _, y in test_loader:
            x, r, y = x.to(device), r.to(device), y.to(device)
            if x.dim() == 1:
                x = x.unsqueeze(0)
                r = r.unsqueeze(0)
                if y.ndim == 0:
                    y = y.unsqueeze(0)

            outputs = model(x, r)
            loss, _, _, _ = loss_fn(outputs, x, y)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite test loss encountered")
            total_test_loss += loss.item()

            if binary_mode:
                probs = torch.sigmoid(outputs["pred_logit"]).detach().cpu().numpy().reshape(-1)
                targets = y.detach().cpu().numpy().reshape(-1).astype(np.float32)
            else:
                probs = torch.softmax(outputs["pred_logit"], dim=1).detach().cpu().numpy()
                targets = y.detach().cpu().numpy().reshape(-1).astype(np.int64)

            _assert_finite_array(probs, "test_probabilities")
            _assert_finite_array(targets, "test_targets_batch")
            all_preds.append(probs)
            all_targets.append(targets)

    avg_loss = total_test_loss / max(len(test_loader), 1)

    if binary_mode:
        all_preds = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.float32)
        all_targets = np.concatenate(all_targets) if all_targets else np.array([], dtype=np.float32)
        _assert_finite_array(all_preds, "test_probabilities")
        _assert_finite_array(all_targets, "test_targets")
        binary_preds = (all_preds > 0.5).astype(int)

        auc = _safe_binary_auc(all_targets, all_preds)
        acc = accuracy_score(all_targets, binary_preds)
        prec = precision_score(all_targets, binary_preds, zero_division=0)
        rec = recall_score(all_targets, binary_preds, zero_division=0)
        f1 = f1_score(all_targets, binary_preds, zero_division=0)

        print("-" * 30)
        print(f"FINAL TEST RESULTS (N={len(all_targets)})")
        print("-" * 30)
        print(f"Test Loss:  {avg_loss:.4f}")
        print(f"ROC-AUC:    {auc:.4f}")
        print(f"Accuracy:   {acc:.4f}")
        print(f"Precision:  {prec:.4f} (Positive Predictive Value)")
        print(f"Recall:     {rec:.4f} (Sensitivity)")
        print(f"F1-Score:   {f1:.4f}")
        print("-" * 30)
        return all_preds, all_targets

    all_preds = np.vstack(all_preds) if all_preds else np.zeros((0, int(getattr(model, "n_classes", 2))), dtype=np.float32)
    all_targets = np.concatenate(all_targets) if all_targets else np.array([], dtype=np.int64)
    _assert_finite_array(all_preds, "test_probabilities")
    _assert_finite_array(all_targets, "test_targets")
    preds_class = np.argmax(all_preds, axis=1) if len(all_preds) > 0 else np.array([], dtype=np.int64)

    auc = _safe_multiclass_auc(all_targets, all_preds)
    acc = accuracy_score(all_targets, preds_class) if len(all_targets) > 0 else np.nan
    f1_macro = f1_score(all_targets, preds_class, average="macro", zero_division=0) if len(all_targets) > 0 else np.nan
    f1_weighted = f1_score(all_targets, preds_class, average="weighted", zero_division=0) if len(all_targets) > 0 else np.nan

    print("-" * 30)
    print(f"FINAL TEST RESULTS (N={len(all_targets)})")
    print("-" * 30)
    print(f"Test Loss:        {avg_loss:.4f}")
    print(f"ROC-AUC (macro):  {auc:.4f}")
    print(f"Accuracy:         {acc:.4f}")
    print(f"F1 macro:         {f1_macro:.4f}")
    print(f"F1 weighted:      {f1_weighted:.4f}")
    print("-" * 30)

    return all_preds, all_targets


def run_dataset_inference(
    model,
    counts_df,
    signal_features,
    ref_features,
    labels=None,
    batch_size=8,
    device="cpu",
    task_type=None,
):
    from torch.utils.data import DataLoader
    from model.data_utils import ZINBDataset

    if task_type is None:
        task_type = "binary" if _is_binary_model(model) else "multiclass"

    if labels is None:
        if task_type == "binary":
            labels = np.zeros(len(counts_df), dtype=np.float32)
        else:
            labels = np.zeros(len(counts_df), dtype=np.int64)

    dataset = ZINBDataset(
        counts_df=counts_df,
        labels=labels,
        signal_features=signal_features,
        ref_features=ref_features,
        task_type=task_type,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return predict_probabilities(model, loader, device, include_targets=True)
