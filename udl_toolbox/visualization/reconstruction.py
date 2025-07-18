"""
Reconstruction quality visualization utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from typing import Optional, List, Tuple, Dict, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error


class ReconstructionVisualizer:
    """
    Visualization tools for autoencoder reconstruction quality.
    
    Provides methods for comparing original and reconstructed data,
    analyzing reconstruction errors, and visualizing quality metrics.
    """
    
    def __init__(self, autoencoder, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize reconstruction visualizer.
        
        Args:
            autoencoder: Fitted autoencoder model
            figsize: Default figure size for matplotlib plots
        """
        self.autoencoder = autoencoder
        self.figsize = figsize
    
    def plot_reconstruction_comparison(
        self,
        data: np.ndarray,
        num_samples: int = 10,
        title: str = "Original vs Reconstructed",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot side-by-side comparison of original and reconstructed data.
        
        Args:
            data: Input data to reconstruct
            num_samples: Number of samples to display
            title: Plot title
            save_path: Optional path to save the plot
        """
        # Select random samples
        indices = np.random.choice(len(data), min(num_samples, len(data)), replace=False)
        selected_data = data[indices]
        
        # Get reconstructions
        reconstructions = self.autoencoder.reconstruct(selected_data).numpy()
        
        # Determine data type and plot accordingly
        if len(selected_data.shape) == 2:
            # 1D data (each row is a sample)
            self._plot_1d_reconstruction_comparison(
                selected_data, reconstructions, title, save_path
            )
        elif len(selected_data.shape) == 4:
            # Image data
            self._plot_image_reconstruction_comparison(
                selected_data, reconstructions, title, save_path
            )
        else:
            raise ValueError(f"Unsupported data shape: {selected_data.shape}")
    
    def _plot_1d_reconstruction_comparison(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        title: str,
        save_path: Optional[str]
    ) -> None:
        """Plot comparison for 1D data."""
        n_samples = len(original)
        n_cols = min(5, n_samples)
        n_rows = (n_samples + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_samples):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]
            
            x_axis = np.arange(len(original[i]))
            ax.plot(x_axis, original[i], label='Original', alpha=0.8)
            ax.plot(x_axis, reconstructed[i], label='Reconstructed', alpha=0.8, linestyle='--')
            
            # Compute MSE for this sample
            mse = mean_squared_error(original[i], reconstructed[i])
            ax.set_title(f'Sample {i} (MSE: {mse:.4f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_samples, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].set_visible(False)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_image_reconstruction_comparison(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        title: str,
        save_path: Optional[str]
    ) -> None:
        """Plot comparison for image data."""
        n_samples = len(original)
        
        fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 2, 4))
        if n_samples == 1:
            axes = axes.reshape(2, 1)
        
        for i in range(n_samples):
            # Original image
            ax_orig = axes[0, i]
            if original[i].shape[-1] == 1:
                ax_orig.imshow(original[i][:, :, 0], cmap='gray')
            else:
                ax_orig.imshow(original[i])
            ax_orig.set_title(f'Original {i}')
            ax_orig.axis('off')
            
            # Reconstructed image
            ax_recon = axes[1, i]
            if reconstructed[i].shape[-1] == 1:
                ax_recon.imshow(reconstructed[i][:, :, 0], cmap='gray')
            else:
                ax_recon.imshow(np.clip(reconstructed[i], 0, 1))
            
            # Compute MSE for this sample
            mse = mean_squared_error(original[i].flatten(), reconstructed[i].flatten())
            ax_recon.set_title(f'Recon {i} (MSE: {mse:.4f})')
            ax_recon.axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_reconstruction_error_distribution(
        self,
        data: np.ndarray,
        error_type: str = 'mse',
        bins: int = 50,
        title: str = "Reconstruction Error Distribution"
    ) -> None:
        """
        Plot distribution of reconstruction errors.
        
        Args:
            data: Input data
            error_type: Type of error ('mse', 'mae', 'pixel_wise')
            bins: Number of histogram bins
            title: Plot title
        """
        # Get reconstructions
        reconstructions = self.autoencoder.reconstruct(data).numpy()
        
        # Compute errors
        if error_type == 'mse':
            errors = [mean_squared_error(data[i].flatten(), reconstructions[i].flatten()) 
                     for i in range(len(data))]
            error_label = 'Mean Squared Error'
        elif error_type == 'mae':
            errors = [mean_absolute_error(data[i].flatten(), reconstructions[i].flatten()) 
                     for i in range(len(data))]
            error_label = 'Mean Absolute Error'
        elif error_type == 'pixel_wise':
            errors = np.mean((data - reconstructions) ** 2, axis=tuple(range(1, len(data.shape))))
            error_label = 'Pixel-wise MSE'
        else:
            raise ValueError(f"Unknown error type: {error_type}")
        
        # Plot distribution
        plt.figure(figsize=self.figsize)
        plt.hist(errors, bins=bins, alpha=0.7, edgecolor='black')
        plt.xlabel(error_label)
        plt.ylabel('Frequency')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        plt.axvline(np.mean(errors), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(errors):.4f}')
        plt.axvline(np.median(errors), color='orange', linestyle='--', 
                   label=f'Median: {np.median(errors):.4f}')
        plt.legend()
        plt.show()
    
    def plot_reconstruction_error_heatmap(
        self,
        data: np.ndarray,
        title: str = "Reconstruction Error Heatmap"
    ) -> None:
        """
        Plot heatmap of reconstruction errors (for image data).
        
        Args:
            data: Input image data
            title: Plot title
        """
        if len(data.shape) != 4:
            raise ValueError("Heatmap visualization only supported for image data")
        
        # Get reconstructions
        reconstructions = self.autoencoder.reconstruct(data).numpy()
        
        # Compute pixel-wise squared errors
        squared_errors = (data - reconstructions) ** 2
        
        # Average over samples and channels
        if squared_errors.shape[-1] > 1:
            avg_error = np.mean(squared_errors, axis=(0, 3))
        else:
            avg_error = np.mean(squared_errors, axis=(0, 3))[:, :, 0]
        
        # Plot heatmap
        plt.figure(figsize=self.figsize)
        sns.heatmap(avg_error, cmap='hot', cbar_kws={'label': 'Mean Squared Error'})
        plt.title(title)
        plt.xlabel('Pixel X')
        plt.ylabel('Pixel Y')
        plt.show()
    
    def plot_worst_reconstructions(
        self,
        data: np.ndarray,
        num_worst: int = 5,
        error_type: str = 'mse',
        title: str = "Worst Reconstructions"
    ) -> None:
        """
        Plot samples with worst reconstruction quality.
        
        Args:
            data: Input data
            num_worst: Number of worst samples to show
            error_type: Type of error to use for ranking
            title: Plot title
        """
        # Get reconstructions
        reconstructions = self.autoencoder.reconstruct(data).numpy()
        
        # Compute errors for each sample
        if error_type == 'mse':
            errors = [mean_squared_error(data[i].flatten(), reconstructions[i].flatten()) 
                     for i in range(len(data))]
        elif error_type == 'mae':
            errors = [mean_absolute_error(data[i].flatten(), reconstructions[i].flatten()) 
                     for i in range(len(data))]
        else:
            raise ValueError(f"Unknown error type: {error_type}")
        
        # Find worst samples
        worst_indices = np.argsort(errors)[-num_worst:][::-1]
        
        # Plot worst reconstructions
        if len(data.shape) == 2:
            self._plot_1d_reconstruction_comparison(
                data[worst_indices], reconstructions[worst_indices], title, None
            )
        elif len(data.shape) == 4:
            self._plot_image_reconstruction_comparison(
                data[worst_indices], reconstructions[worst_indices], title, None
            )
    
    def plot_best_reconstructions(
        self,
        data: np.ndarray,
        num_best: int = 5,
        error_type: str = 'mse',
        title: str = "Best Reconstructions"
    ) -> None:
        """
        Plot samples with best reconstruction quality.
        
        Args:
            data: Input data
            num_best: Number of best samples to show
            error_type: Type of error to use for ranking
            title: Plot title
        """
        # Get reconstructions
        reconstructions = self.autoencoder.reconstruct(data).numpy()
        
        # Compute errors for each sample
        if error_type == 'mse':
            errors = [mean_squared_error(data[i].flatten(), reconstructions[i].flatten()) 
                     for i in range(len(data))]
        elif error_type == 'mae':
            errors = [mean_absolute_error(data[i].flatten(), reconstructions[i].flatten()) 
                     for i in range(len(data))]
        else:
            raise ValueError(f"Unknown error type: {error_type}")
        
        # Find best samples
        best_indices = np.argsort(errors)[:num_best]
        
        # Plot best reconstructions
        if len(data.shape) == 2:
            self._plot_1d_reconstruction_comparison(
                data[best_indices], reconstructions[best_indices], title, None
            )
        elif len(data.shape) == 4:
            self._plot_image_reconstruction_comparison(
                data[best_indices], reconstructions[best_indices], title, None
            )
    
    def plot_reconstruction_quality_vs_latent_dim(
        self,
        data: np.ndarray,
        latent_dims: List[int],
        error_type: str = 'mse',
        title: str = "Reconstruction Quality vs Latent Dimension"
    ) -> None:
        """
        Plot reconstruction quality as a function of latent dimension.
        
        Args:
            data: Input data
            latent_dims: List of latent dimensions to test
            error_type: Type of error to measure
            title: Plot title
        """
        # This would require training multiple models - simplified version
        print("Note: This would require training autoencoders with different latent dimensions.")
        print("For demonstration, showing concept with current model:")
        
        # Get reconstructions with current model
        reconstructions = self.autoencoder.reconstruct(data).numpy()
        
        if error_type == 'mse':
            current_error = np.mean([mean_squared_error(data[i].flatten(), reconstructions[i].flatten()) 
                                   for i in range(len(data))])
        elif error_type == 'mae':
            current_error = np.mean([mean_absolute_error(data[i].flatten(), reconstructions[i].flatten()) 
                                   for i in range(len(data))])
        
        plt.figure(figsize=self.figsize)
        plt.plot([self.autoencoder.latent_dim], [current_error], 'ro', markersize=10, 
                label=f'Current Model (dim={self.autoencoder.latent_dim})')
        plt.xlabel('Latent Dimension')
        plt.ylabel(f'{error_type.upper()}')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()
    
    def compute_reconstruction_metrics(
        self,
        data: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute comprehensive reconstruction metrics.
        
        Args:
            data: Input data
            
        Returns:
            Dictionary of reconstruction metrics
        """
        # Get reconstructions
        reconstructions = self.autoencoder.reconstruct(data).numpy()
        
        # Compute various metrics
        metrics = {}
        
        # Mean squared error
        mse_per_sample = [mean_squared_error(data[i].flatten(), reconstructions[i].flatten()) 
                         for i in range(len(data))]
        metrics['mse_mean'] = np.mean(mse_per_sample)
        metrics['mse_std'] = np.std(mse_per_sample)
        
        # Mean absolute error
        mae_per_sample = [mean_absolute_error(data[i].flatten(), reconstructions[i].flatten()) 
                         for i in range(len(data))]
        metrics['mae_mean'] = np.mean(mae_per_sample)
        metrics['mae_std'] = np.std(mae_per_sample)
        
        # Structural similarity (for images)
        if len(data.shape) == 4:
            try:
                from skimage.metrics import structural_similarity as ssim
                ssim_scores = []
                for i in range(len(data)):
                    if data[i].shape[-1] == 1:
                        score = ssim(data[i][:, :, 0], reconstructions[i][:, :, 0])
                    else:
                        score = ssim(data[i], reconstructions[i], multichannel=True)
                    ssim_scores.append(score)
                metrics['ssim_mean'] = np.mean(ssim_scores)
                metrics['ssim_std'] = np.std(ssim_scores)
            except ImportError:
                print("scikit-image not available for SSIM computation")
        
        # Peak signal-to-noise ratio
        if len(data.shape) == 4:
            psnr_scores = []
            for i in range(len(data)):
                mse = mean_squared_error(data[i].flatten(), reconstructions[i].flatten())
                if mse == 0:
                    psnr = float('inf')
                else:
                    psnr = 20 * np.log10(1.0 / np.sqrt(mse))
                psnr_scores.append(psnr)
            metrics['psnr_mean'] = np.mean(psnr_scores)
            metrics['psnr_std'] = np.std(psnr_scores)
        
        return metrics
    
    def print_reconstruction_summary(
        self,
        data: np.ndarray
    ) -> None:
        """
        Print summary of reconstruction quality metrics.
        
        Args:
            data: Input data
        """
        metrics = self.compute_reconstruction_metrics(data)
        
        print("=== Reconstruction Quality Summary ===")
        print(f"Number of samples: {len(data)}")
        print(f"Data shape: {data.shape}")
        print()
        print(f"MSE: {metrics['mse_mean']:.6f} ± {metrics['mse_std']:.6f}")
        print(f"MAE: {metrics['mae_mean']:.6f} ± {metrics['mae_std']:.6f}")
        
        if 'ssim_mean' in metrics:
            print(f"SSIM: {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}")
        
        if 'psnr_mean' in metrics:
            print(f"PSNR: {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f} dB")
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary."""
        return {
            'autoencoder_type': type(self.autoencoder).__name__,
            'figsize': self.figsize
        }