import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import warnings
import pickle
import os
import gc
from scipy import ndimage
import psutil

warnings.filterwarnings('ignore')

# Set device with memory optimization
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using CUDA acceleration")
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using MPS acceleration")
    # Set conservative memory settings for MPS - fix watermark ratios
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.7'  # Conservative high watermark
    os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.6'   # Conservative low watermark
    # Additional MPS optimizations
    os.environ['PYTORCH_MPS_ALLOCATOR_POLICY'] = 'garbage_collection'
else:
    device = torch.device('cpu')
    print("Using CPU")

torch.manual_seed(42)
np.random.seed(42)

def get_memory_info():
    """Get current memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"CUDA: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
    elif torch.backends.mps.is_available():
        try:
            allocated = torch.mps.current_allocated_memory() / 1024**3
            return f"MPS: {allocated:.2f}GB allocated"
        except:
            return "MPS: Memory info unavailable"
    else:
        ram = psutil.virtual_memory()
        return f"RAM: {ram.used/1024**3:.2f}GB used, {ram.available/1024**3:.2f}GB available"

def clear_memory():
    """Aggressive memory clearing"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except AttributeError:
            pass

def get_optimal_batch_size(base_size, d, max_memory_gb=16):
    """Dynamically adjust batch size based on dimension and available memory"""
    # Very conservative scaling for MPS
    if device.type == 'mps':
        max_memory_gb = min(max_memory_gb, 8)  # Very conservative for MPS
    
    # Scale down batch size based on dimension
    scale_factor = max(1, d / 8)  # Baseline for d=8
    optimal_size = int(base_size / scale_factor)
    
    # Much more aggressive reduction for MPS to avoid OOM
    if device.type == 'mps':
        optimal_size = min(optimal_size, 64)  # Reduced from 128 to 64
    
    return max(8, optimal_size)  # Reduced minimum batch size to 8

class PotentialEnergy:
    """Defines the potential energy function from equation (1)"""
    def __init__(self, d, L=2.5, lam=0.03, T=8.0):
        self.d = d
        self.L = L
        self.lam = lam
        self.T = T
        self.h = 1.0 / (1.0 + d)

    def __call__(self, x):
        """Compute potential energy V(x1, ..., xd)"""
        x_padded = torch.cat([x[..., -1:], x], dim=-1)
        diff = (x_padded[..., 1:] - x_padded[..., :-1]) / self.h
        kinetic_term = (self.lam / 2.0) * torch.sum(diff**2, dim=-1)
        potential_term = (1.0 / (4.0 * self.lam)) * torch.sum((1 - x**2)**2, dim=-1)
        return kinetic_term + potential_term

    def score(self, x):
        """Compute score function: -∇V(x)/T"""
        x = x.requires_grad_(True)
        V = self(x)
        if V.dim() == 0:
            V = V.unsqueeze(0)
        grad_V = torch.autograd.grad(V.sum(), x, create_graph=False)[0]
        return -grad_V / self.T

def langevin_sampling_chunked(potential, d, n_samples=100000, n_steps=2000, dt=0.0005, chunk_size=10000):
    """Memory-efficient Langevin sampling using chunks"""
    print(f"Sampling {n_samples} samples in {d}D using chunked Langevin dynamics...")
    
    # Much more aggressive chunk size reduction for MPS
    if device.type == 'mps':
        chunk_size = min(chunk_size, 2000)  # Reduced from 5000 to 2000
    elif d > 8:
        chunk_size = min(chunk_size, 3000)  # Also reduce for high dimensions
    
    all_samples = []
    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    
    burn_in_steps = n_steps // 4
    sqrt_2T_dt = np.sqrt(2 * potential.T * dt)
    
    for chunk_idx in range(n_chunks):
        current_chunk_size = min(chunk_size, n_samples - chunk_idx * chunk_size)
        print(f"Processing chunk {chunk_idx + 1}/{n_chunks} (size: {current_chunk_size})")
        
        x = torch.rand(current_chunk_size, d, device=device) * 2 * potential.L - potential.L
        
        for step in tqdm(range(n_steps + burn_in_steps), desc=f"Langevin chunk {chunk_idx+1}"):
            score = potential.score(x)
            noise = torch.randn_like(x) * sqrt_2T_dt
            x = x + potential.T * score * dt + noise
            x = torch.clamp(x, -potential.L, potential.L)
            
            # More frequent memory cleanup for MPS
            if step % 200 == 0:  # Reduced from 500 to 200
                clear_memory()
        
        all_samples.append(x.detach().cpu())
        del x
        clear_memory()
    
    # Concatenate all chunks
    result = torch.cat(all_samples, dim=0).to(device)
    del all_samples
    clear_memory()
    
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
    """Improved score network - architecture unchanged"""
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
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, t):
        t_emb = self.time_embedding(t)
        h = self.input_proj(x)
        h = h + self.time_proj(t_emb)
        h = F.silu(h)
        for layer in self.layers:
            residual = h
            h = layer(h)
            if h.shape == residual.shape:
                h = h + residual
            h = F.silu(h)
        return self.output_proj(h)

class VPSDEDiffusionModel:
    """Memory-optimized VP-SDE diffusion model"""
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
        return mean + std * noise, noise
    
    def train(self, data, n_epochs=600, base_batch_size=256, lr=2e-4):
        """Memory-optimized training with dynamic batch sizing"""
        print(f"Training score network for d={self.d} with {len(data)} samples...")
        print(f"Memory info: {get_memory_info()}")
        
        # Very conservative batch size for MPS
        if device.type == 'mps':
            base_batch_size = min(base_batch_size, 32)  # Much smaller for MPS
        
        # Dynamic batch size
        batch_size = get_optimal_batch_size(base_batch_size, self.d)
        print(f"Using batch size: {batch_size}")
        
        optimizer = torch.optim.AdamW(self.score_net.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        
        # Create dataset and dataloader with memory optimization
        dataset = torch.utils.data.TensorDataset(data)
        data_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            drop_last=True, 
            pin_memory=False,  # Disable pin_memory for MPS
            num_workers=0  # Disable multiprocessing
        )
        
        losses = []
        
        for epoch in tqdm(range(n_epochs), desc=f"Training N={len(data)}"):
            epoch_losses = []
            
            for batch_idx, (batch_x,) in enumerate(data_loader):
                try:
                    batch_x = batch_x.to(device)
                    
                    # Random time sampling
                    t = torch.rand(batch_x.shape[0], device=device) * self.T
                    t = torch.clamp(t, min=1e-5, max=1.0 - 1e-5)
                    
                    # Sample prior
                    x_t, noise = self.sample_prior(batch_x, t)
                    
                    # Predict score
                    predicted_score = self.score_net(x_t, t)
                    
                    # Compute target score
                    _, std = self.marginal_prob(t)
                    if x_t.dim() == 2:
                        std = std.unsqueeze(-1)
                    target_score = -noise / (std + 1e-5)
                    
                    # Compute loss
                    loss_weight = std.squeeze()**2
                    loss = torch.mean(loss_weight.unsqueeze(-1) * (predicted_score - target_score)**2)
                    
                    # Optimization step
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.score_net.parameters(), 1.0)
                    optimizer.step()
                    
                    epoch_losses.append(loss.item())
                    
                    # Clean up intermediate tensors
                    del batch_x, x_t, noise, predicted_score, target_score, loss
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"OOM error at batch {batch_idx}, clearing memory and continuing...")
                        clear_memory()
                        continue
                    else:
                        raise e
                
                # More frequent memory cleanup for MPS
                if batch_idx % 5 == 0:  # Very frequent cleanup
                    clear_memory()
            
            if epoch_losses:  # Only if we have valid losses
                avg_loss = np.mean(epoch_losses)
                losses.append(avg_loss)
                scheduler.step()
                
                if epoch % 150 == 0 or epoch == n_epochs - 1:
                    print(f"Epoch {epoch}, Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
                    print(f"Memory: {get_memory_info()}")
            
            # Memory cleanup between epochs
            clear_memory()
        
        print("Training complete.")
        return losses
    
    def sample(self, n_samples=100000, n_steps=1000, chunk_size=5000):
        """Memory-efficient sampling using chunks"""
        print(f"Generating {n_samples} samples using chunked reverse SDE...")
        
        # Much more aggressive chunk size reduction for MPS
        if device.type == 'mps':
            chunk_size = min(chunk_size, 1000)  # Reduced from 2000 to 1000
        elif self.d > 8:
            chunk_size = min(chunk_size, 1500)  # Also reduce for high dimensions
        
        dt = self.T / n_steps
        sqrt_dt = np.sqrt(dt)
        
        all_samples = []
        n_chunks = (n_samples + chunk_size - 1) // chunk_size
        
        self.score_net.eval()
        
        with torch.no_grad():
            for chunk_idx in range(n_chunks):
                current_chunk_size = min(chunk_size, n_samples - chunk_idx * chunk_size)
                print(f"Sampling chunk {chunk_idx + 1}/{n_chunks} (size: {current_chunk_size})")
                
                x = torch.randn(current_chunk_size, self.d, device=device)
                
                for i in tqdm(range(n_steps), desc=f"Reverse SDE chunk {chunk_idx+1}", leave=False):
                    t_val = self.T - i * dt
                    t = torch.ones(current_chunk_size, device=device) * t_val
                    t = torch.clamp(t, min=1e-5, max=1.0 - 1e-5)
                    
                    score = self.score_net(x, t)
                    beta_t = self.beta(t).unsqueeze(-1)
                    
                    drift = 0.5 * beta_t * x + beta_t * score
                    diffusion_coeff = torch.sqrt(beta_t)
                    x = x + drift * dt + diffusion_coeff * sqrt_dt * torch.randn_like(x)
                    
                    # More frequent cleanup for MPS
                    if i % 100 == 0:  # Reduced from 200 to 100
                        clear_memory()
                
                all_samples.append(x.detach().cpu())
                del x
                clear_memory()
        
        # Concatenate results
        result = torch.cat(all_samples, dim=0).to(device)
        del all_samples
        clear_memory()
        
        print("Diffusion sampling complete.")
        return result

def compute_statistics_chunked(samples, chunk_size=10000):
    """Memory-efficient statistics computation"""
    n_samples, d = samples.shape
    
    # Compute mean in chunks
    mean_sum = torch.zeros(d, device=samples.device)
    for i in range(0, n_samples, chunk_size):
        chunk = samples[i:i+chunk_size]
        mean_sum += torch.sum(chunk, dim=0)
    mean = mean_sum / n_samples
    
    # Compute covariance in chunks
    cov = torch.zeros(d, d, device=samples.device)
    for i in range(0, n_samples, chunk_size):
        chunk = samples[i:i+chunk_size]
        centered = chunk - mean.unsqueeze(0)
        cov += torch.mm(centered.T, centered)
        del chunk, centered
        clear_memory()
    
    cov = cov / (n_samples - 1)
    return mean, cov

def plot_convergence_metrics(convergence_data, save_dir):
    """Plot convergence metrics vs sample size"""
    print("Creating convergence plots...")
    
    sample_sizes = convergence_data['sample_sizes']
    mean_l1_errors = convergence_data['mean_l1_errors']
    mean_l2_errors = convergence_data['mean_l2_errors']
    cov_frobenius_errors = convergence_data['cov_frobenius_errors']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Mean L1 error vs sample size
    axes[0, 0].loglog(sample_sizes, mean_l1_errors, 'o-', color='blue', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Number of Training Samples (N)')
    axes[0, 0].set_ylabel('Mean L1 Error')
    axes[0, 0].set_title('Mean L1 Error vs Training Sample Size (d=8)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Mean L2 error vs sample size  
    axes[0, 1].loglog(sample_sizes, mean_l2_errors, 'o-', color='red', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Number of Training Samples (N)')
    axes[0, 1].set_ylabel('Mean L2 Error')
    axes[0, 1].set_title('Mean L2 Error vs Training Sample Size (d=8)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Covariance Frobenius error vs sample size
    axes[1, 0].loglog(sample_sizes, cov_frobenius_errors, 'o-', color='green', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Number of Training Samples (N)')
    axes[1, 0].set_ylabel('Covariance Frobenius Error')
    axes[1, 0].set_title('Covariance Frobenius Error vs Training Sample Size (d=8)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: All metrics on same plot (normalized)
    norm_l1 = np.array(mean_l1_errors) / max(mean_l1_errors)
    norm_l2 = np.array(mean_l2_errors) / max(mean_l2_errors)
    norm_cov = np.array(cov_frobenius_errors) / max(cov_frobenius_errors)
    
    axes[1, 1].semilogx(sample_sizes, norm_l1, 'o-', label='Mean L1 (normalized)', linewidth=2, markersize=6)
    axes[1, 1].semilogx(sample_sizes, norm_l2, 's-', label='Mean L2 (normalized)', linewidth=2, markersize=6)
    axes[1, 1].semilogx(sample_sizes, norm_cov, '^-', label='Cov Frobenius (normalized)', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('Number of Training Samples (N)')
    axes[1, 1].set_ylabel('Normalized Error')
    axes[1, 1].set_title('All Metrics vs Training Sample Size (Normalized, d=8)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = os.path.join(save_dir, "convergence_analysis_d8.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Convergence plots saved to: {plot_file}")
    
    return plot_file

def plot_2d_debug_marginal_chunked(samples, n_samples, name_prefix, base_dir="debugplots", n_bins=50, chunk_size=10000):
    """Memory-efficient 2D marginal plotting"""
    os.makedirs(base_dir, exist_ok=True)
    
    # Process in chunks to avoid memory issues
    x_values = []
    y_values = []
    
    n_total = samples.shape[0]
    for i in range(0, n_total, chunk_size):
        chunk = samples[i:i+chunk_size].cpu().numpy()
        x_values.extend(chunk[:, 6])  # X_7
        y_values.extend(chunk[:, 7])  # X_8
        del chunk
    
    x = np.array(x_values)
    y = np.array(y_values)
    del x_values, y_values
    
    file_suffix = f"{name_prefix}_x7_x8_N{n_samples}.png"

    plt.figure(figsize=(8, 7))
    plt.hist2d(x, y, bins=n_bins, cmap='viridis', density=True)
    plt.colorbar(label='Density')
    plt.title(f'{name_prefix.capitalize()} - $X_7$ vs $X_8$ 2D Marginal (N={n_samples})')
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
    plt.close()
    print(f"✓ 2D Debug plot saved to: {plot_file}")
    
    return plot_file

def process_single_sample_size(n_samples, d, reference_samples, results_dir):
    """Memory-optimized processing of single sample size"""
    print(f"\n{'='*60}\nProcessing N = {n_samples} samples (d={d})\n{'='*60}")
    print(f"Memory before processing: {get_memory_info()}")
    
    try:
        plot_files = {'debug_plots': []}

        # Step 1: Use subset of reference samples
        print(f"Step 1: Using {n_samples} samples from reference dataset...")
        training_samples = reference_samples[:n_samples].clone()
        
        # Save training samples and immediately clear reference
        training_file = os.path.join(results_dir, f"training_samples_N{n_samples}.pkl")
        with open(training_file, 'wb') as f:
            pickle.dump(training_samples.cpu().numpy(), f)
        print(f"✓ Training samples saved to: {training_file}")

        # Step 2: Train model
        print("Step 2: Training VP-SDE score-based diffusion model...")
        diffusion_model = VPSDEDiffusionModel(d)
        
        # Reduced epochs and optimized batch size for memory
        n_epochs = min(400, max(200, 600 * 10000 // n_samples))  # Scale epochs with sample size
        losses = diffusion_model.train(training_samples, n_epochs=n_epochs, base_batch_size=128, lr=2e-4)
        
        # Save model
        model_file = os.path.join(results_dir, f"score_model_N{n_samples}.pt")
        torch.save({
            'model_state_dict': diffusion_model.score_net.state_dict(),
            'model_config': {
                'd': d, 'T': diffusion_model.T, 
                'beta_min': diffusion_model.beta_min, 
                'beta_max': diffusion_model.beta_max
            },
            'training_losses': losses,
            'n_training_samples': n_samples
        }, model_file)
        print(f"✓ Trained model saved to: {model_file}")

        # Clear training data
        del training_samples
        clear_memory()

        # Step 3: Generate samples with reduced count for memory efficiency
        print("Step 3: Generating samples from trained diffusion model...")
        # Much smaller sample count for MPS
        if device.type == 'mps':
            sample_count = min(20000, n_samples)  # Very conservative for MPS
        else:
            sample_count = min(50000, n_samples)  # Conservative for other devices
        
        diffusion_samples = diffusion_model.sample(sample_count, chunk_size=1000 if device.type == 'mps' else 2000)
        
        # Save samples
        diffusion_file = os.path.join(results_dir, f"diffusion_samples_N{n_samples}.pkl")
        with open(diffusion_file, 'wb') as f:
            pickle.dump(diffusion_samples.cpu().numpy(), f)
        print(f"✓ Diffusion samples saved to: {diffusion_file}")
        
        # Generate debug plot
        print("Plotting 2D debug marginal...")
        plot_files['debug_plots'].append(
            plot_2d_debug_marginal_chunked(diffusion_samples, n_samples, "score")
        )
        plot_files['debug_plots'] = [f for f in plot_files['debug_plots'] if f is not None]

        # Step 4: Compute statistics efficiently
        print("Step 4: Computing statistics and comparisons...")
        mean_reference, cov_reference = compute_statistics_chunked(reference_samples, chunk_size=5000)
        mean_diffusion, cov_diffusion = compute_statistics_chunked(diffusion_samples, chunk_size=5000)
        
        # Compute errors
        mean_l1_error = torch.norm(mean_reference - mean_diffusion, p=1).item()
        mean_l2_error = torch.norm(mean_reference - mean_diffusion, p=2).item()
        cov_frobenius_error = torch.norm(cov_reference - cov_diffusion, p='fro').item()
        
        print(f"\nResults for N={n_samples}:")
        print(f"Mean L1 error: {mean_l1_error:.6f}")
        print(f"Mean L2 error: {mean_l2_error:.6f}")
        print(f"Covariance Frobenius error: {cov_frobenius_error:.6f}")

        # Step 5: Save results
        sample_summary = {
            'n_samples': n_samples,
            'd': d,
            'mean_l1_error': mean_l1_error,
            'mean_l2_error': mean_l2_error,
            'cov_frobenius_error': cov_frobenius_error,
            'files': {
                'training_samples': training_file,
                'model': model_file,
                'diffusion_samples': diffusion_file,
                **plot_files
            },
            'training_losses': losses,
            'final_loss': losses[-1] if losses else None,
            'n_epochs_used': n_epochs,
            'sample_count_generated': sample_count
        }
        
        sample_results_file = os.path.join(results_dir, f"results_N{n_samples}.pkl")
        with open(sample_results_file, 'wb') as f:
            pickle.dump(sample_summary, f)
        print(f"✓ Sample size {n_samples} results saved to: {sample_results_file}")

        # Clean up everything
        del diffusion_samples, diffusion_model, mean_reference, mean_diffusion
        del cov_reference, cov_diffusion, losses
        clear_memory()
        
        print(f"✓ Sample size {n_samples} processing complete")
        print(f"Memory after processing: {get_memory_info()}")
        
        return sample_summary

    except Exception as e:
        print(f"Error processing sample size {n_samples}: {str(e)}")
        import traceback
        traceback.print_exc()
        clear_memory()
        return None

def main():
    """Memory-optimized main function"""
    # Very conservative sample sizes for MPS
    d = 8
    if device.type == 'mps':
        sample_sizes = [10000, 20000, 50000, 100000, 200000, 500000, 1000000]  # Much smaller for MPS
    else:
        sample_sizes = [5000, 10000, 20000, 50000, 100000, 200000]  # Original sizes
    
    max_reference_samples = max(sample_sizes)
    
    results_dir = "score_diffusion_convergence_study"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}/")
    print(f"Fixed dimension: d = {d}")
    print(f"Sample sizes to study: {sample_sizes}")
    print(f"Device: {device}")
    print(f"Initial memory: {get_memory_info()}")

    # Step 1: Generate reference dataset using chunked sampling
    print(f"\n{'='*80}\nGenerating reference dataset with {max_reference_samples} samples\n{'='*80}")
    potential = PotentialEnergy(d)
    
    # Use smaller chunks for MPS
    chunk_size = 2000 if device.type == 'mps' else 5000
    reference_samples = langevin_sampling_chunked(
        potential, d, max_reference_samples, 
        chunk_size=chunk_size
    )
    
    reference_file = os.path.join(results_dir, f"reference_samples_d{d}_N{max_reference_samples}.pkl")
    with open(reference_file, 'wb') as f:
        pickle.dump(reference_samples.cpu().numpy(), f)
    print(f"✓ Reference samples saved to: {reference_file}")

    # Generate debug plot for reference
    plot_2d_debug_marginal_chunked(reference_samples, max_reference_samples, "reference")

    # Step 2: Process each sample size
    summary_results = {}
    convergence_data = {
        'sample_sizes': [], 
        'mean_l1_errors': [], 
        'mean_l2_errors': [], 
        'cov_frobenius_errors': []
    }
    
    for n_samples in sample_sizes:
        result = process_single_sample_size(n_samples, d, reference_samples, results_dir)
        
        if result is not None:
            summary_results[n_samples] = {
                'n_samples': result['n_samples'],
                'd': result['d'],
                'mean_l1_error': result['mean_l1_error'],
                'mean_l2_error': result['mean_l2_error'],
                'cov_frobenius_error': result['cov_frobenius_error'],
                'final_loss': result['final_loss'],
                'files': result['files']
            }
            
            convergence_data['sample_sizes'].append(n_samples)
            convergence_data['mean_l1_errors'].append(result['mean_l1_error'])
            convergence_data['mean_l2_errors'].append(result['mean_l2_error'])
            convergence_data['cov_frobenius_errors'].append(result['cov_frobenius_error'])

    # Clear reference samples after processing
    del reference_samples
    clear_memory()

    # Step 3: Create convergence plots
    print(f"\n{'='*80}\nCREATING CONVERGENCE PLOTS\n{'='*80}")
    convergence_plot_file = None
    if len(convergence_data['sample_sizes']) > 1:
        convergence_plot_file = plot_convergence_metrics(convergence_data, results_dir)
    else:
        print("Not enough data points for convergence plots")

    # Step 4: Save final summary
    print(f"\n{'='*80}\nCREATING FINAL SUMMARY\n{'='*80}")
    summary_file = os.path.join(results_dir, "convergence_summary_results.pkl")
    final_summary = {
        'd': d, 
        'sample_sizes': sample_sizes, 
        'max_reference_samples': max_reference_samples,
        'results_summary': summary_results, 
        'convergence_data': convergence_data,
        'convergence_plot': convergence_plot_file,
        'reference_file': reference_file,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'memory_optimized': True,
        'device_type': device.type
    }
    with open(summary_file, 'wb') as f:
        pickle.dump(final_summary, f)
    print(f"✓ Overall summary saved to: {summary_file}")

    # Step 5: Print results table
    print(f"\n{'='*80}\nMEMORY-OPTIMIZED CONVERGENCE STUDY RESULTS (d={d}, device={device.type})\n{'='*80}")
    print(f"{'N Samples':<12} {'Mean L1':<12} {'Mean L2':<12} {'Cov Frobenius':<15} {'Final Loss':<12}")
    print(f"{'-'*80}")
    
    for n_samples in sample_sizes:
        if n_samples in summary_results:
            r = summary_results[n_samples]
            final_loss = f"{r['final_loss']:.6f}" if r['final_loss'] is not None else 'N/A'
            print(f"{n_samples:<12} {r['mean_l1_error']:<12.6f} {r['mean_l2_error']:<12.6f} {r['cov_frobenius_error']:<15.6f} {final_loss:<12}")
        else:
            print(f"{n_samples:<12} {'FAILED':<12} {'FAILED':<12} {'FAILED':<15} {'FAILED':<12}")

    print(f"\n✓ Memory-optimized convergence study complete for {device.type}!")
    print(f"Final memory usage: {get_memory_info()}")
    
    return final_summary

if __name__ == "__main__":
    results = main()