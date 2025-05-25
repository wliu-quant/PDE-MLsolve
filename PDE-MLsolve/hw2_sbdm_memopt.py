import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib # Import matplotlib
matplotlib.use('Agg') # Set non-interactive backend BEFORE importing pyplot
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import warnings
import pickle
import os
import gc
from scipy import ndimage

warnings.filterwarnings('ignore')

# Set device (CUDA/MPS/CPU)
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using CUDA acceleration")
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using MPS acceleration")
else:
    device = torch.device('cpu')
    print("Using CPU")

# Set default tensor type to float32 for compatibility
torch.set_default_dtype(torch.float32)
torch.manual_seed(42)
np.random.seed(42)

class PotentialEnergy:
    """Defines the potential energy function from equation (1)"""
    def __init__(self, d, L=2.5, lam=0.03, T=8.0):
        self.d = d
        self.L = L
        self.lam = lam
        self.T = T
        self.h = 1.0 / (1.0 + d)

    def __call__(self, x):
        """
        Compute potential energy V(x1, ..., xd)
        x: tensor of shape (..., d)
        """
        x_padded = torch.cat([x[..., -1:], x], dim=-1)
        diff = (x_padded[..., 1:] - x_padded[..., :-1]) / self.h
        kinetic_term = (self.lam / 2.0) * torch.sum(diff**2, dim=-1)
        potential_term = (1.0 / (4.0 * self.lam)) * torch.sum((1 - x**2)**2, dim=-1)
        return kinetic_term + potential_term

    def score_batched(self, x, batch_size=10000):
        """Compute score function in batches to save memory: -∇V(x)/T"""
        n_samples = x.shape[0]
        scores = []
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            x_batch = x[i:end_idx].requires_grad_(True)
            
            V_batch = self(x_batch)
            if V_batch.dim() == 0:
                V_batch = V_batch.unsqueeze(0)
            
            grad_V_batch = torch.autograd.grad(V_batch.sum(), x_batch, create_graph=False)[0]
            score_batch = -grad_V_batch / self.T
            scores.append(score_batch.detach())
            
            # Clear intermediate tensors
            del x_batch, V_batch, grad_V_batch, score_batch
            if device.type == 'mps':
                torch.mps.empty_cache()
            elif device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return torch.cat(scores, dim=0)

    def score(self, x):
        """Wrapper for score computation - uses batched version for large tensors"""
        if x.shape[0] > 5000:  # Use batched version for large sample sets
            return self.score_batched(x, batch_size=5000)
        
        # Original implementation for smaller batches
        x = x.requires_grad_(True)
        V = self(x)
        if V.dim() == 0:
            V = V.unsqueeze(0)
        grad_V = torch.autograd.grad(V.sum(), x, create_graph=False)[0]
        return -grad_V / self.T

def langevin_sampling(potential, d, n_samples=100000, n_steps=2000, dt=0.0005, batch_size=20000):
    """Sample using overdamped Langevin dynamics with memory-efficient batching"""
    print(f"Sampling {n_samples} samples in {d}D using Langevin dynamics (batch_size={batch_size})...")
    
    # Process samples in batches to reduce memory usage
    all_samples = []
    sqrt_2T_dt = np.sqrt(2 * potential.T * dt)
    burn_in_steps = n_steps // 4
    total_steps = n_steps + burn_in_steps
    
    for batch_start in range(0, n_samples, batch_size):
        batch_end = min(batch_start + batch_size, n_samples)
        current_batch_size = batch_end - batch_start
        
        print(f"Processing batch {batch_start//batch_size + 1}/{(n_samples-1)//batch_size + 1} "
              f"(samples {batch_start}-{batch_end-1})")
        
        # Initialize batch
        x = torch.rand(current_batch_size, d, device=device) * 2 * potential.L - potential.L
        
        # Run Langevin dynamics for this batch
        for step in tqdm(range(total_steps), desc=f"Langevin batch {batch_start//batch_size + 1}"):
            score = potential.score(x)
            noise = torch.randn_like(x) * sqrt_2T_dt
            x = x + potential.T * score * dt + noise
            x = torch.clamp(x, -potential.L, potential.L)
            
            # Periodic memory cleanup
            if step % 500 == 0:
                del score, noise
                if device.type == 'mps':
                    torch.mps.empty_cache()
                elif device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        # Store batch results
        all_samples.append(x.detach().cpu())
        del x
        
        # Clear memory after each batch
        if device.type == 'mps':
            torch.mps.empty_cache()
        elif device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
    # Concatenate all batches
    result = torch.cat(all_samples, dim=0).to(device)
    del all_samples
    gc.collect()
    
    return result

class FourierEmbedding(nn.Module):
    """Fourier feature embedding for time"""
    def __init__(self, embedding_size=256, scale=16):
        super().__init__()
        self.embedding_size = embedding_size
        self.scale = scale
    def forward(self, x):
        device = x.device
        dtype = x.dtype
        half_dim = self.embedding_size // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half_dim, dtype=dtype, device=device) / half_dim)
        scaled_time = x.unsqueeze(-1) * freqs.unsqueeze(0) * self.scale
        return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)

class ImprovedScoreNetwork(nn.Module):
    """Improved score network"""
    def __init__(self, d, embedding_dim=256, hidden_dims=[512, 512, 512, 512]):
        super().__init__()
        self.d = d
        self.embedding_dim = embedding_dim
        self.time_embedding = FourierEmbedding(embedding_dim)
        self.input_proj = nn.Linear(d, hidden_dims[0])
        self.time_proj = nn.Linear(embedding_dim, hidden_dims[0])
        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.layers.append(self._make_residual_block(hidden_dims[i], hidden_dims[i+1]))
        self.output_proj = nn.Linear(hidden_dims[-1], d)
        self._initialize_weights()
    def _make_residual_block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GroupNorm(min(32, out_dim//4) if out_dim >= 4 else 1, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
            nn.GroupNorm(min(32, out_dim//4) if out_dim >= 4 else 1, out_dim),
        )
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, x, t):
        t_emb = self.time_embedding(t)
        h = self.input_proj(x)
        h = h + self.time_proj(t_emb)
        h = F.silu(h)
        for layer in self.layers:
            residual = h
            h = layer(h)
            if h.shape == residual.shape: h = h + residual
            h = F.silu(h)
        return self.output_proj(h)

class VPSDEDiffusionModel:
    """VP-SDE diffusion model"""
    def __init__(self, d, T=1.0, beta_min=0.1, beta_max=20.0):
        self.d = d
        self.T = T
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.score_net = ImprovedScoreNetwork(d).to(device)
    def beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)
    def marginal_prob(self, t):
        log_mean_coeff = -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        mean_coeff = torch.exp(log_mean_coeff)
        std = torch.sqrt(1 - torch.exp(2 * log_mean_coeff).clamp(max=1.0-1e-5))
        return mean_coeff, std
    def sample_prior(self, x0, t):
        mean_coeff, std = self.marginal_prob(t)
        if x0.dim() == 2:
            mean_coeff = mean_coeff.unsqueeze(-1)
            std = std.unsqueeze(-1)
        mean = mean_coeff * x0
        noise = torch.randn_like(x0)
        return mean + std * noise, noise # Return both x_t and noise
    def train(self, data, n_epochs=600, batch_size=256, lr=2e-4):
        print(f"Training score network for d={self.d}...")
        optimizer = torch.optim.AdamW(self.score_net.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        dataset = torch.utils.data.TensorDataset(data)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True if device.type == 'cuda' else False)
        losses = []
        for epoch in tqdm(range(n_epochs), desc=f"Training d={self.d}"):
            epoch_losses = []
            for (batch_x,) in data_loader:
                batch_x = batch_x.to(device)
                t = torch.rand(batch_x.shape[0], device=device) * self.T
                t = torch.clamp(t, min=1e-5, max=1.0 - 1e-5)
                x_t, noise = self.sample_prior(batch_x, t) # Get noise
                predicted_score = self.score_net(x_t, t)
                _, std = self.marginal_prob(t)
                if x_t.dim() == 2: std = std.unsqueeze(-1)
                target_score = -noise / (std + 1e-5) # Use actual noise
                loss_weight = std.squeeze()**2
                loss = torch.mean(loss_weight.unsqueeze(-1) * (predicted_score - target_score)**2)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.score_net.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(loss.item())
                
                # Clear intermediate tensors
                del batch_x, t, x_t, noise, predicted_score, target_score, loss
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            scheduler.step()
            if epoch % 100 == 0 or epoch == n_epochs - 1:
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Periodic memory cleanup during training
            if epoch % 50 == 0:
                if device.type == 'mps':
                    torch.mps.empty_cache()
                elif device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()
        
        print("Training complete.")
        return losses
        
    def sample(self, n_samples=100000, n_steps=1000, batch_size=20000):
        """Generate samples using reverse SDE with memory-efficient batching"""
        print(f"Generating {n_samples} samples using reverse SDE (batch_size={batch_size})...")
        dt = self.T / n_steps
        sqrt_dt = np.sqrt(dt)
        
        all_samples = []
        self.score_net.eval()
        
        with torch.no_grad():
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)
                current_batch_size = batch_end - batch_start
                
                print(f"Generating batch {batch_start//batch_size + 1}/{(n_samples-1)//batch_size + 1} "
                      f"(samples {batch_start}-{batch_end-1})")
                
                # Initialize batch with noise
                x = torch.randn(current_batch_size, self.d, device=device)
                
                # Reverse SDE for this batch
                for i in tqdm(range(n_steps), desc=f"Reverse SDE batch {batch_start//batch_size + 1}"):
                    t_val = self.T - i * dt
                    t = torch.ones(current_batch_size, device=device) * t_val
                    t = torch.clamp(t, min=1e-5, max=1.0 - 1e-5)
                    score = self.score_net(x, t)
                    beta_t = self.beta(t).unsqueeze(-1)
                    drift = 0.5 * beta_t * x + beta_t * score
                    diffusion_coeff = torch.sqrt(beta_t)
                    x = x + drift * dt + diffusion_coeff * sqrt_dt * torch.randn_like(x)
                    
                    # Clear intermediate tensors periodically
                    if i % 200 == 0:
                        del score, beta_t, drift, diffusion_coeff
                        if device.type == 'mps':
                            torch.mps.empty_cache()
                        elif device.type == 'cuda':
                            torch.cuda.empty_cache()
                
                # Store batch results
                all_samples.append(x.detach().cpu())
                del x
                
                # Clear memory after each batch
                if device.type == 'mps':
                    torch.mps.empty_cache()
                elif device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()
        
        # Concatenate all batches
        result = torch.cat(all_samples, dim=0).to(device)
        del all_samples
        gc.collect()
        
        print("Diffusion sampling complete.")
        return result

def compute_statistics(samples, batch_size=50000):
    """Compute statistics in batches to handle large sample sets"""
    n_samples, d = samples.shape
    
    # Compute mean in batches
    mean_sum = torch.zeros(d, device=samples.device)
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch = samples[i:end_idx]
        mean_sum += torch.sum(batch, dim=0)
    mean = mean_sum / n_samples
    
    # Compute covariance in batches
    cov = torch.zeros(d, d, device=samples.device)
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch = samples[i:end_idx]
        batch_centered = batch - mean
        cov += torch.mm(batch_centered.T, batch_centered)
        del batch, batch_centered
        
        # Clear memory periodically
        if device.type == 'mps':
            torch.mps.empty_cache()
        elif device.type == 'cuda':
            torch.cuda.empty_cache()
    
    cov = cov / (n_samples - 1)
    return mean, cov

def plot_contour_comparison(samples1, samples2, d, title1="Langevin", title2="Diffusion", save_dir=None):
    """
    Plot contour comparison of X_7/X_8 marginal distributions for both methods
    """
    if d < 8: 
        print(f"Dimension {d} < 8, skipping contour plots.")
        return None
    
    print(f"Creating contour plots for d={d}...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Convert to numpy - use subsampling for very large datasets
    sample_size = min(50000, samples1.shape[0])
    indices = torch.randperm(samples1.shape[0])[:sample_size]
    
    s1_cpu = samples1[indices].cpu().numpy()
    s2_cpu = samples2[indices].cpu().numpy()
    
    # Extract X_7 and X_8
    x1_langevin, y1_langevin = s1_cpu[:, 6], s1_cpu[:, 7]  # X_7, X_8
    x2_diffusion, y2_diffusion = s2_cpu[:, 6], s2_cpu[:, 7]  # X_7, X_8
    
    # Determine common range for both plots
    all_x = np.concatenate([x1_langevin, x2_diffusion])
    all_y = np.concatenate([y1_langevin, y2_diffusion])
    x_min, x_max = np.percentile(all_x, [1, 99])
    y_min, y_max = np.percentile(all_y, [1, 99])
    
    # Number of bins for histogram-based density estimation
    n_bins = 60
    
    # Create density estimates using 2D histograms
    def create_density_estimate(x, y, x_range, y_range, n_bins):
        hist, x_edges, y_edges = np.histogram2d(x, y, bins=n_bins, 
                                                range=[[x_range[0], x_range[1]], 
                                                       [y_range[0], y_range[1]]], 
                                                density=True)
        # Smooth the histogram for better contours
        hist_smooth = ndimage.gaussian_filter(hist, sigma=1.0)
        
        # Create coordinate arrays for contour plotting
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        X, Y = np.meshgrid(x_centers, y_centers)
        
        return X, Y, hist_smooth.T  # Transpose for correct orientation
    
    # Create density estimates
    X1, Y1, Z1 = create_density_estimate(x1_langevin, y1_langevin, 
                                         (x_min, x_max), (y_min, y_max), n_bins)
    X2, Y2, Z2 = create_density_estimate(x2_diffusion, y2_diffusion, 
                                         (x_min, x_max), (y_min, y_max), n_bins)
    
    # Define contour levels (percentiles of the density)
    def get_contour_levels(Z, n_levels=8):
        # Get levels based on percentiles of non-zero density values
        nonzero_Z = Z[Z > 0]
        if len(nonzero_Z) > 0:
            levels = np.percentile(nonzero_Z, np.linspace(10, 90, n_levels))
            return sorted(levels)
        else:
            return np.linspace(Z.min(), Z.max(), n_levels)
    
    levels1 = get_contour_levels(Z1)
    levels2 = get_contour_levels(Z2)
    
    # Plot contours for Langevin
    cs1 = ax1.contour(X1, Y1, Z1, levels=levels1, colors='blue', alpha=0.8, linewidths=1.5)
    ax1.contourf(X1, Y1, Z1, levels=levels1, alpha=0.3, cmap='Blues')
    ax1.clabel(cs1, inline=True, fontsize=8, fmt='%.3f')
    ax1.set_title(f'{title1} - $X_7$ vs $X_8$ Contours (d={d})')
    ax1.set_xlabel('$X_7$')
    ax1.set_ylabel('$X_8$')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    
    # Plot contours for Diffusion
    cs2 = ax2.contour(X2, Y2, Z2, levels=levels2, colors='orange', alpha=0.8, linewidths=1.5)
    ax2.contourf(X2, Y2, Z2, levels=levels2, alpha=0.3, cmap='Oranges')
    ax2.clabel(cs2, inline=True, fontsize=8, fmt='%.3f')
    ax2.set_title(f'{title2} - $X_7$ vs $X_8$ Contours (d={d})')
    ax2.set_xlabel('$X_7$')
    ax2.set_ylabel('$X_8$')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = None
    if save_dir is not None:
        plot_file = os.path.join(save_dir, f"contour_plots_d{d}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Contour plots saved to: {plot_file}")
    
    plt.close()  # Close figure to free memory
    return plot_file

def plot_marginals(samples1, samples2, d, title1="Langevin", title2="Diffusion", save_dir=None):
    if d < 8: return None
    
    # Subsample for plotting if datasets are very large
    sample_size = min(20000, samples1.shape[0])
    indices = torch.randperm(samples1.shape[0])[:sample_size]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    s1_cpu = samples1[indices].cpu().numpy()
    s2_cpu = samples2[indices].cpu().numpy()
    axes[0, 0].hist(s1_cpu[:, 6], bins=50, alpha=0.7, density=True, color='blue', label=title1)
    axes[0, 0].hist(s2_cpu[:, 6], bins=50, alpha=0.7, density=True, color='orange', label=title2)
    axes[0, 0].set_title(f'X_7 Marginal Comparison (d={d})'); axes[0, 0].legend()
    axes[0, 1].hist(s1_cpu[:, 6], bins=50, alpha=0.7, density=True, color='blue')
    axes[0, 1].set_title(f'X_7 Marginal - {title1} (d={d})')
    axes[1, 0].hist(s1_cpu[:, 7], bins=50, alpha=0.7, density=True, color='blue', label=title1)
    axes[1, 0].hist(s2_cpu[:, 7], bins=50, alpha=0.7, density=True, color='orange', label=title2)
    axes[1, 0].set_title(f'X_8 Marginal Comparison (d={d})'); axes[1, 0].legend()
    axes[1, 1].hist(s2_cpu[:, 7], bins=50, alpha=0.7, density=True, color='orange')
    axes[1, 1].set_title(f'X_8 Marginal - {title2} (d={d})')
    for ax_row in axes:
        for ax in ax_row:
            ax.set_ylabel('Density'); ax.grid(True, alpha=0.3)
    axes[0,0].set_xlabel('$X_7$'); axes[0,1].set_xlabel('$X_7$')
    axes[1,0].set_xlabel('$X_8$'); axes[1,1].set_xlabel('$X_8$')
    plt.tight_layout()
    plot_file = None
    if save_dir is not None:
        plot_file = os.path.join(save_dir, f"marginal_plots_d{d}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Marginal plots saved to: {plot_file}")
    plt.close() # Close figure to free memory
    return plot_file

def plot_2d_comparison(samples1, samples2, d, title1="Langevin", title2="Diffusion", save_dir=None):
    if d < 8: return None
    
    # Subsample for plotting
    sample_size = min(20000, samples1.shape[0])
    indices = torch.randperm(samples1.shape[0])[:sample_size]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    s1_cpu = samples1[indices].cpu().numpy()
    s2_cpu = samples2[indices].cpu().numpy()
    ax1.hist(s1_cpu[:, 6], bins=60, alpha=0.8, density=True, color='blue', edgecolor='darkblue', linewidth=0.5)
    ax1.set_title(f'{title1} - $X_7$ Marginal (d={d})')
    ax2.hist(s2_cpu[:, 6], bins=60, alpha=0.8, density=True, color='orange', edgecolor='darkorange', linewidth=0.5)
    ax2.set_title(f'{title2} - $X_7$ Marginal (d={d})')
    all_data = np.concatenate([s1_cpu[:, 6], s2_cpu[:, 6]])
    x_min, x_max = np.percentile(all_data, [0.5, 99.5])
    y_max = 0
    for ax in [ax1, ax2]:
        ax.set_xlabel('$X_7$'); ax.set_ylabel('Density'); ax.grid(True, alpha=0.3)
        ax.set_xlim(x_min, x_max)
        y_max = max(y_max, ax.get_ylim()[1])
    for ax in [ax1, ax2]: ax.set_ylim(0, y_max * 1.05)
    plt.tight_layout()
    plot_file = None
    if save_dir is not None:
        plot_file = os.path.join(save_dir, f"x7_marginal_plots_d{d}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ X₇ marginal plots saved to: {plot_file}")
    plt.close() # Close figure to free memory
    return plot_file

def plot_2d_debug_marginal(samples, d, name_prefix, base_dir="debugplots", n_bins=50):
    """
    Plots the X_7 / X_8 2D marginal distribution and saves it.
    """
    if d < 8:
        print(f"Dimension {d} < 8, skipping 2D debug marginal plot.")
        return None

    os.makedirs(base_dir, exist_ok=True)
    
    # Subsample for plotting
    sample_size = min(30000, samples.shape[0])
    indices = torch.randperm(samples.shape[0])[:sample_size]
    samples_cpu = samples[indices].cpu().numpy()
    
    x = samples_cpu[:, 6] # X_7
    y = samples_cpu[:, 7] # X_8
    file_suffix = f"{name_prefix}_x7_x8_d{d}.png"

    plt.figure(figsize=(8, 7))
    plt.hist2d(x, y, bins=n_bins, cmap='viridis', density=True)
    plt.colorbar(label='Density')
    plt.title(f'{name_prefix.capitalize()} - $X_7$ vs $X_8$ 2D Marginal (d={d})')
    plt.xlabel('$X_7$')
    plt.ylabel('$X_8$')
    plt.grid(True, alpha=0.3, linestyle='--')
    x_min, x_max = np.percentile(x, [1, 99])
    y_min, y_max = np.percentile(y, [1, 99])
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.tight_layout()

    plot_file = os.path.join(base_dir, file_suffix)
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ 2D Debug plot saved to: {plot_file}")
    plt.close() # Close figure to free memory
    return plot_file

def clear_memory():
    """Enhanced memory clearing function"""
    gc.collect()
    if torch.cuda.is_available(): 
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        try: 
            torch.mps.empty_cache()
        except AttributeError: 
            pass

def process_single_dimension(d, n_samples, results_dir):
    print(f"\n{'='*60}\nProcessing dimension d = {d}\n{'='*60}")
    
    # Determine memory-efficient batch sizes based on dimension
    if d <= 16:
        langevin_batch_size = 50000
        diffusion_batch_size = 50000
    elif d <= 32:
        langevin_batch_size = 20000
        diffusion_batch_size = 20000
    else:  # d >= 64
        langevin_batch_size = 10000
        diffusion_batch_size = 10000
    
    try:
        plot_files = {'debug_plots': []}

        # Step 1: Langevin sampling & Plotting
        print("Step 1: Generating reference samples with Langevin dynamics...")
        potential = PotentialEnergy(d)
        langevin_samples = langevin_sampling(potential, d, n_samples, 
                                           batch_size=langevin_batch_size)
        langevin_file = os.path.join(results_dir, f"langevin_samples_d{d}.pkl")
        with open(langevin_file, 'wb') as f: 
            pickle.dump(langevin_samples.cpu().numpy(), f)
        print(f"✓ Langevin samples saved to: {langevin_file}")
        
        if d >= 8:
            print("Plotting Langevin 2D debug marginal...")
            plot_files['debug_plots'].append(plot_2d_debug_marginal(langevin_samples, d, "langevin"))

        # Clear memory before training
        clear_memory()

        # Step 2: Train score-based diffusion model
        print("Step 2: Training VP-SDE score-based diffusion model...")
        diffusion_model = VPSDEDiffusionModel(d)
        losses = diffusion_model.train(langevin_samples, n_epochs=600, batch_size=512, lr=2e-4)
        model_file = os.path.join(results_dir, f"score_model_d{d}.pt")
        torch.save({'model_state_dict': diffusion_model.score_net.state_dict(), 
                   'model_config': {'d': d, 'T': diffusion_model.T, 'beta_min': diffusion_model.beta_min, 'beta_max': diffusion_model.beta_max}, 
                   'training_losses': losses}, model_file)
        print(f"✓ Trained model saved to: {model_file}")

        # Clear memory before sampling
        clear_memory()

        # Step 3: Generate samples from diffusion model & Plotting
        print("Step 3: Generating samples from trained diffusion model...")
        diffusion_samples = diffusion_model.sample(n_samples, batch_size=diffusion_batch_size)
        diffusion_file = os.path.join(results_dir, f"diffusion_samples_d{d}.pkl")
        with open(diffusion_file, 'wb') as f: 
            pickle.dump(diffusion_samples.cpu().numpy(), f)
        print(f"✓ Diffusion samples saved to: {diffusion_file}")
        
        if d >= 8:
            print("Plotting Score Field 2D debug marginal...")
            plot_files['debug_plots'].append(plot_2d_debug_marginal(diffusion_samples, d, "score"))
        
        plot_files['debug_plots'] = [f for f in plot_files['debug_plots'] if f is not None]

        # Clear memory before statistics computation
        clear_memory()

        # Step 4: Compute statistics and comparisons
        print("Step 4: Computing statistics and comparisons...")
        mean_langevin, cov_langevin = compute_statistics(langevin_samples)
        mean_diffusion, cov_diffusion = compute_statistics(diffusion_samples)
        mean_l1_error = torch.norm(mean_langevin - mean_diffusion, p=1).item()
        mean_l2_error = torch.norm(mean_langevin - mean_diffusion, p=2).item()
        cov_frobenius_error = torch.norm(cov_langevin - cov_diffusion, p='fro').item()
        print(f"\nResults for d={d}:\nMean L1 error: {mean_l1_error:.6f}\nMean L2 error: {mean_l2_error:.6f}\nCovariance Frobenius error: {cov_frobenius_error:.6f}")

        # Step 5: Generate and save comparison plots
        if d >= 8:
            print("Plotting comparison marginal distributions...")
            marginal_plot_file = plot_marginals(langevin_samples, diffusion_samples, d, save_dir=results_dir)
            if marginal_plot_file: plot_files['marginal_plots'] = marginal_plot_file
            
            x7_plot_file = plot_2d_comparison(langevin_samples, diffusion_samples, d, save_dir=results_dir)
            if x7_plot_file: plot_files['x7_marginal_plots'] = x7_plot_file
            
            # NEW: Add contour plots
            contour_plot_file = plot_contour_comparison(langevin_samples, diffusion_samples, d, save_dir=results_dir)
            if contour_plot_file: plot_files['contour_plots'] = contour_plot_file
        else:
            print(f"Dimension {d} < 8, skipping comparison plots.")

        # Step 6: Save dimension-specific summary
        dimension_summary = {
            'd': d, 'n_samples': n_samples, 'mean_l1_error': mean_l1_error, 'mean_l2_error': mean_l2_error,
            'cov_frobenius_error': cov_frobenius_error, 'files': {'langevin_samples': langevin_file, 'model': model_file, 'diffusion_samples': diffusion_file, **plot_files},
            'training_losses': losses, 'final_loss': losses[-1] if losses else None
        }
        dim_results_file = os.path.join(results_dir, f"results_d{d}.pkl")
        with open(dim_results_file, 'wb') as f: pickle.dump(dimension_summary, f)
        print(f"✓ Dimension {d} results saved to: {dim_results_file}")

        # Step 7: Clear memory completely
        print("Clearing memory...")
        del langevin_samples, diffusion_samples, diffusion_model, potential, losses
        del mean_langevin, mean_diffusion, cov_langevin, cov_diffusion
        clear_memory()
        print(f"✓ Dimension {d} processing complete and memory cleared")
        return dimension_summary

    except Exception as e:
        print(f"Error processing dimension {d}: {str(e)}")
        import traceback
        traceback.print_exc()
        clear_memory()
        return None

def main():
    dimensions = [32, 64] # Use desired dimensions
    n_samples = 100000 # Use desired number of samples
    results_dir = "score_diffusion_results"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}/")

    summary_results = {}
    for d in dimensions:
        result = process_single_dimension(d, n_samples, results_dir)
        if result is not None:
            summary_results[d] = {
                'd': result['d'],
                'mean_l1_error': result['mean_l1_error'],
                'mean_l2_error': result['mean_l2_error'],
                'cov_frobenius_error': result['cov_frobenius_error'],
                'final_loss': result['final_loss'],
                'files': result['files']
            }

    print(f"\n{'='*80}\nCREATING FINAL SUMMARY\n{'='*80}")
    summary_file = os.path.join(results_dir, "summary_results.pkl")
    final_summary = {'dimensions': dimensions, 'n_samples': n_samples, 'results_summary': summary_results, 'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(summary_file, 'wb') as f: pickle.dump(final_summary, f)
    print(f"✓ Overall summary saved to: {summary_file}")

    print(f"\n{'='*80}\nFINAL SUMMARY RESULTS\n{'='*80}")
    print(f"{'Dimension':<10} {'Mean L1':<12} {'Mean L2':<12} {'Cov Frobenius':<15} {'Final Loss':<12} {'Samples':<10}")
    print(f"{'-'*80}")
    for d in dimensions:
        if d in summary_results:
            r = summary_results[d]
            final_loss = f"{r['final_loss']:.6f}" if r['final_loss'] is not None else 'N/A'
            print(f"{d:<10} {r['mean_l1_error']:<12.6f} {r['mean_l2_error']:<12.6f} {r['cov_frobenius_error']:<15.6f} {final_loss:<12} {n_samples:<10}")
        else:
            print(f"{d:<10} {'FAILED':<12} {'FAILED':<12} {'FAILED':<15} {'FAILED':<12} {n_samples:<10}")

    print(f"\n{'='*80}\nEXPORTED FILES INVENTORY:\n{'='*80}")
    for d in dimensions:
        if d in summary_results:
            print(f"\nDimension {d}:")
            files = summary_results[d]['files']
            print(f"  Langevin samples:  {files.get('langevin_samples', 'N/A')}")
            print(f"  Trained model:     {files.get('model', 'N/A')}")
            print(f"  Diffusion samples: {files.get('diffusion_samples', 'N/A')}")
            print(f"  Individual results: score_diffusion_results/results_d{d}.pkl")
            if 'x7_marginal_plots' in files: print(f"  X₇ marginal plots: {files['x7_marginal_plots']}")
            if 'marginal_plots' in files: print(f"  X₇,X₈ marginals:   {files['marginal_plots']}")
            if 'contour_plots' in files: print(f"  X₇,X₈ contours:    {files['contour_plots']}")
            if 'debug_plots' in files and files['debug_plots']:
                print(f"  2D Debug plots:")
                for dp_file in files['debug_plots']:
                    print(f"    - {dp_file}")
        else:
            print(f"\nDimension {d}: PROCESSING FAILED")

    print(f"\n✓ All processing complete! Check {results_dir}/ and debugplots/ for all files.")
    return summary_results

if __name__ == "__main__":
    results = main()