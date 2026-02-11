from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score

class OrionEncoder(nn.Module):
    def __init__(self, x_dim, r_dim, z_dim=32, l_dim=1):
        super(OrionEncoder, self).__init__()
        
		# Signal encode (Arm 1) - tumor vs healthy
        self.enc_x_hidden = nn.Linear(x_dim, 128)
        self.bn_x = nn.LayerNorm(128)
		
		# Two outputs, Mu and log-variance
        self.z_mu = nn.Linear(128, z_dim)
        self.z_logvar = nn.Linear(128, z_dim)
        
		# Reference encode (Arm2) - Sequencing depth
        self.enc_r_hidden = nn.Linear(r_dim, 32)
        self.bn_r = nn.LayerNorm(32)
        
        self.l_mu = nn.Linear(32, l_dim)
        self.l_logvar = nn.Linear(32, l_dim)
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
        
    def forward(self, x, r):
        # Encode X
        h_x = F.relu(self.bn_x(self.enc_x_hidden(x)))
        z_mean = self.z_mu(h_x)
        z_logvar = self.z_logvar(h_x)
        z = self.reparameterize(z_mean, z_logvar)
        
		# Encode R
        h_r = F.relu(self.bn_r(self.enc_r_hidden(r)))
        l_mean = self.l_mu(h_r)
        l_logvar = self.l_logvar(h_r)
        l = self.reparameterize(l_mean, l_logvar)
        
        return z, z_mean, z_logvar, l, l_mean, l_logvar   

class ZINBDecoder(nn.Module):
    """
	A ZINB VAE must output three parameters for every gene to 
    model the Zero-Inflated Negative Binomial distribution:
    - Mean ($\mu$): The expected count.
    - Dispersion ($\theta$): The "shape" parameter (how spread 
      out the data is).
    - Dropout ($\pi$): The probability that the gene is a 
      "false zero" (technical dropout).
	"""
    def __init__(self, z_dim, l_dim, output_dim):
        super(ZINBDecoder, self).__init__()
        
        self.decoder_hidden = nn.Linear(z_dim + l_dim, 128)
        self.bn_dec = nn.LayerNorm(128)
        
		# Head 1: Mean (Mu) - Predicts the scaling factor for every gene
        self.dec_scale = nn.Linear(128, output_dim)
        # Head 2: Dispersion - Predicts the inverse dispersion
        self.dec_theta = nn.Linear(128, output_dim)
        # Head 3: Dropout (Pi) - Predicts the probability of zero-inflation
        self.dec_pi = nn.Linear(128, output_dim)
        
    def forward(self, z, l):
        # Concatenate biological latent (z) and library latent (l)
        combined = torch.cat([z, l], dim=1)
        h = F.relu(self.bn_dec(self.decoder_hidden(combined)))
        
		# Mean: Softmax enforces that these sum to 1 (proportions), 
        # then we multiply by the learned library size (l) to get counts.
        # Note: We exponentiate l to ensure it's positive.
        scale = F.softmax(self.dec_scale(h), dim=1)
        mean = torch.exp(l) * scale 				# 1
        dispersion = torch.exp(self.dec_theta(h)) 	# 2
        dropout = torch.sigmoid(self.dec_pi(h)) 	# 3
        return mean, dispersion, dropout

class CancerClassifier(nn.Module):
    def __init__(self, z_dim):
        super(CancerClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.5), # Essential to prevent overfitting on small N=180 dataset
            nn.Linear(16, 1) # Single output logit
        )
    def forward(self, z):
        return self.net(z)
    
class OrionVAE(nn.Module):
    def __init__(self, x_dim, r_dim, z_dim=32, l_dim=1):
        super(OrionVAE, self).__init__()
        self.encoder = OrionEncoder(x_dim, r_dim, z_dim, l_dim)
        self.decoder = ZINBDecoder(z_dim, l_dim, output_dim=x_dim)
        self.classifier = CancerClassifier(z_dim)
        
    def forward(self, x, r):
        # 1. Encode
        z, z_mu, z_logvar, l, l_mu, l_logvar = self.encoder(x, r)
        # 2. Decode (Reconstruct X, We do NOT reconstruct R)
        mean, dispersion, dropout = self.decoder(z, l)
        # 3. Classify based only on z (biology), ignoring l (technical noise)
        pred_logit = self.classifier(z)
        return {
            'mean': mean, 'dispersion': dispersion, 'dropout': dropout, # For ZINB Loss
            'pred_logit': pred_logit, # For Classification Loss
            'z_mu': z_mu, 'z_logvar': z_logvar, # For KL Loss (Z)
            'l_mu': l_mu, 'l_logvar': l_logvar  # For KL Loss (L)
        }
		
class OrionLoss(nn.Module):
    def __init__(self, beta=1.0, gamma=10.0):
        """
        Args:
            beta: Weight for KL Divergence (Regularization).
            gamma: Weight for Classification Loss (Prediction).
        """
        super(OrionLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss() # For cancer classification
        
    def zinb_loss(self, x, mean, disp, pi, eps=1e-10):
        """
        Computes the Negative Log Likelihood (NLL) of the ZINB distribution.
        
        Args:
            x: True counts (Input).
            mean: Predicted mean (mu).
            disp: Predicted dispersion (theta).
            pi: Predicted dropout probability.
        """
        # Clamp dispersion to avoid explosion
        disp = torch.clamp(disp, min=eps, max=1e6)
        pi = torch.clamp(pi, min=eps, max=1.0 - eps)
        mean = torch.clamp(mean, min=eps, max=1e6)
        
		# Log-Likelihood of the Negative Binomial (NB) part
        # Formula: log(Gamma(x+theta)) - log(Gamma(theta)) - log(x!) 
        #          + theta * log(theta / (theta+mu)) + x * log(mu / (theta+mu))
        t1 = torch.lgamma(x + disp) - torch.lgamma(disp) - torch.lgamma(x + 1)
        t2 = (disp + x) * torch.log(1.0 + (mean / disp)) + (x * (torch.log(disp) - torch.log(mean)))
        nb_log_prob = t1 - t2
        
        # Case A: x is zero. It could be dropout (pi) OR NB distribution generating a zero.
        zero_case = -torch.log(pi + ((1.0 - pi) * torch.exp(nb_log_prob)) + eps)
        # Case B: x is > 0. It cannot be dropout. It must be NB.
        nonzero_case = -torch.log(1.0 - pi + eps) - nb_log_prob
        
        result = torch.where(x < 1e-8, zero_case, nonzero_case)
        
        return result.sum()
    
    def kl_divergence(self, mu, logvar):
        # Standard VAE Regularization: KL(N(mu, sigma) || N(0, 1))
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    def forward(self, outputs, target_x, target_y):
        """
        outputs: Dictionary from OrionVAE.forward()
        target_x: The original signal matrix X.
        target_y: The cancer labels.
        """
        # 1. Reconstruction Loss (ZINB)
        recon_loss = self.zinb_loss(
            target_x, 
            outputs['mean'], 
            outputs['dispersion'], 
            outputs['dropout']
        )
        # 2. KL Divergence (Regularization for Z and L)
        kl_z = self.kl_divergence(outputs['z_mu'], outputs['z_logvar'])
        kl_l = self.kl_divergence(outputs['l_mu'], outputs['l_logvar'])
        total_kl = kl_z + kl_l
        
		# 3. Classification Loss (Supervised)
        class_loss = self.bce(outputs['pred_logit'], target_y)
        
        # Weighted Sum
        total_loss = recon_loss + (self.beta * total_kl) + (self.gamma * class_loss)
        
        return total_loss, recon_loss, total_kl, class_loss
    
def train_orion(model, train_loader, val_loader, epochs=100, lr=1e-3, device='cuda'):
    # 1. Setup
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = OrionLoss(beta=0.01, gamma=10.0).to(device)
    history = {'train_loss': [], 'val_auc': []}
    
    print(f"Starting training on {device} for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (x, r, l, y) in enumerate(train_loader):
            x, r, y = x.to(device), r.to(device), y.to(device)
            if x.dim() == 1:
                x = x.unsqueeze(0)
                r = r.unsqueeze(0)
                y = y.unsqueeze(0)
            optimizer.zero_grad()
            outputs = model(x, r)
            loss, recon, kl, cls_loss = criterion(outputs, x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_train_loss = epoch_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        if (epoch + 1) % 5 == 0:
            val_auc, val_acc = evaluate_model(model, val_loader, device)
            history['val_auc'].append(val_auc)
            print(f"Epoch {epoch+1:03d} | "
				f"Loss: {avg_train_loss:.4f} | "
				f"Val AUC: {val_auc:.4f} | "
				f"Val Acc: {val_acc:.4f}")
            
    return history

def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, r, l, y in loader:
            x, r, y = x.to(device), r.to(device), y.to(device)
            if x.dim() == 1:
                x = x.unsqueeze(0)
                r = r.unsqueeze(0)
                y = y.unsqueeze(0)
            outputs = model(x, r)
            # Apply Sigmoid to get probability (0-1) from logit
            probs = torch.sigmoid(outputs['pred_logit'])
            
            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(y.cpu().numpy())# Calculate Metrics
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5 # Handle edge case with single-class batch
        
	# Threshold at 0.5 for accuracy
    preds_binary = (np.array(all_preds) > 0.5).astype(int)
    acc = accuracy_score(all_targets, preds_binary)
    
    return auc, acc

def plot_training_results(history):
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Training Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Total Training Loss')
    ax1.set_title('Training Loss per Epoch')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Validation AUC
    num_auc_records = len(history['val_auc'])
    if num_auc_records > 0:
        interval = len(history['train_loss']) // num_auc_records
        auc_x_axis = [i * interval for i in range(1, num_auc_records + 1)]
        
        ax2.plot(auc_x_axis, history['val_auc'], 'r-o', label='Validation ROC-AUC')
        ax2.set_title('Validation Performance (AUC)')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('AUC Score')
        ax2.set_ylim([0.4, 1.05])
        ax2.legend()
        ax2.grid(True)
    else:
        ax2.set_title('Validation AUC (No data recorded)')

    plt.tight_layout()
    plt.show()
    
def evaluate_test_set(model, test_loader, loss_fn, device):
    model.eval()
    all_preds = []
    all_targets = []
    total_test_loss = 0
    
    with torch.no_grad():
        for x, r, l, y in test_loader:
            x, r, y = x.to(device), r.to(device), y.to(device)
            if x.dim() == 1:
                x = x.unsqueeze(0)
                r = r.unsqueeze(0)
                y = y.unsqueeze(0)
            
            # Forward pass
            outputs = model(x, r)
            
            # 1. Calculate Test Loss
            loss, _, _, _ = loss_fn(outputs, x, y)
            total_test_loss += loss.item()
            
            # 2. Collect probabilities and targets
            probs = torch.sigmoid(outputs['pred_logit'])
            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    # Convert to arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    binary_preds = (all_preds > 0.5).astype(int)
    
    # Calculate Statistics
    avg_loss = total_test_loss / len(test_loader)
    auc = roc_auc_score(all_targets, all_preds)
    acc = accuracy_score(all_targets, binary_preds)
    prec = precision_score(all_targets, binary_preds)
    rec = recall_score(all_targets, binary_preds)
    f1 = f1_score(all_targets, binary_preds)

    # Print Results
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
            