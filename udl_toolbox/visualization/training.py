"""
Training progress and loss visualization utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any


class LossVisualizer:
    """
    Visualization tools for training progress and loss analysis.
    
    Provides methods for plotting training curves, loss components,
    and analyzing training dynamics.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize loss visualizer.
        
        Args:
            figsize: Default figure size for matplotlib plots
        """
        self.figsize = figsize
    
    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        title: str = "Training Curves",
        save_path: Optional[str] = None,
        interactive: bool = False
    ) -> None:
        """
        Plot training and validation loss curves.
        
        Args:
            history: Training history dictionary
            title: Plot title
            save_path: Optional path to save the plot
            interactive: Whether to create interactive plotly plot
        """
        if interactive:
            self._plot_training_curves_interactive(history, title)
        else:
            self._plot_training_curves_static(history, title, save_path)
    
    def _plot_training_curves_static(
        self,
        history: Dict[str, List[float]],
        title: str,
        save_path: Optional[str]
    ) -> None:
        """Create static matplotlib training curves."""
        # Determine number of subplots needed
        loss_types = [key for key in history.keys() if 'loss' in key.lower()]
        n_plots = len(loss_types)
        
        if n_plots == 1:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            axes = [ax]
        else:
            n_cols = min(2, n_plots)
            n_rows = (n_plots + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(self.figsize[0], self.figsize[1] * n_rows / 2))
            axes = axes.flatten() if n_plots > 1 else [axes]
        
        epochs = range(1, len(history[loss_types[0]]) + 1)
        
        for i, loss_type in enumerate(loss_types):
            ax = axes[i]
            
            # Plot training loss
            ax.plot(epochs, history[loss_type], label=f'Training {loss_type}', linewidth=2)
            
            # Plot validation loss if available
            val_key = f'val_{loss_type}'
            if val_key in history:
                ax.plot(epochs, history[val_key], label=f'Validation {loss_type}', linewidth=2)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(loss_type.replace('_', ' ').title())
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_training_curves_interactive(
        self,
        history: Dict[str, List[float]],
        title: str
    ) -> None:
        """Create interactive plotly training curves."""
        loss_types = [key for key in history.keys() if 'loss' in key.lower()]
        epochs = list(range(1, len(history[loss_types[0]]) + 1))
        
        fig = make_subplots(
            rows=len(loss_types),
            cols=1,
            subplot_titles=[loss_type.replace('_', ' ').title() for loss_type in loss_types],
            shared_xaxes=True
        )
        
        for i, loss_type in enumerate(loss_types):
            # Training loss
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history[loss_type],
                    mode='lines',
                    name=f'Training {loss_type}',
                    line=dict(width=2)
                ),
                row=i+1, col=1
            )
            
            # Validation loss if available
            val_key = f'val_{loss_type}'
            if val_key in history:
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=history[val_key],
                        mode='lines',
                        name=f'Validation {loss_type}',
                        line=dict(width=2)
                    ),
                    row=i+1, col=1
                )
        
        fig.update_layout(
            title=title,
            xaxis_title='Epoch',
            height=300 * len(loss_types)
        )
        
        fig.show()
    
    def plot_loss_components(
        self,
        history: Dict[str, List[float]],
        title: str = "Loss Components",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot individual loss components (reconstruction, regularization, etc.).
        
        Args:
            history: Training history dictionary
            title: Plot title
            save_path: Optional path to save the plot
        """
        epochs = range(1, len(history['loss']) + 1)
        
        plt.figure(figsize=self.figsize)
        
        # Plot total loss
        plt.plot(epochs, history['loss'], label='Total Loss', linewidth=3, alpha=0.8)
        
        # Plot components
        component_keys = [key for key in history.keys() 
                         if key in ['reconstruction_loss', 'regularization_loss', 'kl_loss', 'sparsity_loss']]
        
        for key in component_keys:
            if key in history:
                plt.plot(epochs, history[key], label=key.replace('_', ' ').title(), linewidth=2)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale often better for loss visualization
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_loss_smoothed(
        self,
        history: Dict[str, List[float]],
        window_size: int = 10,
        title: str = "Smoothed Training Curves"
    ) -> None:
        """
        Plot smoothed training curves using moving average.
        
        Args:
            history: Training history dictionary
            window_size: Size of moving average window
            title: Plot title
        """
        def smooth(data, window):
            return np.convolve(data, np.ones(window)/window, mode='valid')
        
        epochs = range(1, len(history['loss']) + 1)
        smoothed_epochs = range(window_size, len(history['loss']) + 1)
        
        plt.figure(figsize=self.figsize)
        
        # Plot original and smoothed total loss
        plt.plot(epochs, history['loss'], alpha=0.3, label='Total Loss (raw)')
        plt.plot(smoothed_epochs, smooth(history['loss'], window_size), 
                linewidth=2, label=f'Total Loss (smoothed, window={window_size})')
        
        # Plot validation loss if available
        if 'val_loss' in history:
            plt.plot(epochs, history['val_loss'], alpha=0.7, label='Validation Loss', linestyle='--')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.show()
    
    def plot_learning_rate_schedule(
        self,
        learning_rates: List[float],
        title: str = "Learning Rate Schedule"
    ) -> None:
        """
        Plot learning rate schedule over training.
        
        Args:
            learning_rates: List of learning rates per epoch
            title: Plot title
        """
        epochs = range(1, len(learning_rates) + 1)
        
        plt.figure(figsize=self.figsize)
        plt.plot(epochs, learning_rates, linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.show()
    
    def plot_gradient_norms(
        self,
        gradient_norms: List[float],
        title: str = "Gradient Norms During Training"
    ) -> None:
        """
        Plot gradient norms over training (for gradient explosion/vanishing analysis).
        
        Args:
            gradient_norms: List of gradient norms per step
            title: Plot title
        """
        steps = range(1, len(gradient_norms) + 1)
        
        plt.figure(figsize=self.figsize)
        plt.plot(steps, gradient_norms, alpha=0.7)
        plt.xlabel('Training Step')
        plt.ylabel('Gradient Norm')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Add reference lines
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Gradient norm = 1')
        plt.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Gradient norm = 0.1')
        plt.legend()
        plt.show()
    
    def plot_loss_landscape_1d(
        self,
        autoencoder,
        data_sample: np.ndarray,
        parameter_direction: np.ndarray,
        alpha_range: Tuple[float, float] = (-1.0, 1.0),
        num_points: int = 50,
        title: str = "1D Loss Landscape"
    ) -> None:
        """
        Plot 1D loss landscape along a specific direction.
        
        Args:
            autoencoder: Autoencoder model
            data_sample: Sample data for loss computation
            parameter_direction: Direction in parameter space
            alpha_range: Range of steps along direction
            num_points: Number of points to evaluate
            title: Plot title
        """
        alphas = np.linspace(alpha_range[0], alpha_range[1], num_points)
        losses = []
        
        # Store original parameters
        original_params = [param.numpy() for param in autoencoder.autoencoder.trainable_variables]
        
        for alpha in alphas:
            # Modify parameters
            for i, param in enumerate(autoencoder.autoencoder.trainable_variables):
                param.assign(original_params[i] + alpha * parameter_direction[i])
            
            # Compute loss
            reconstructed = autoencoder.reconstruct(data_sample)
            loss_dict = autoencoder._compute_loss(data_sample, reconstructed)
            losses.append(float(loss_dict['total_loss']))
        
        # Restore original parameters
        for i, param in enumerate(autoencoder.autoencoder.trainable_variables):
            param.assign(original_params[i])
        
        plt.figure(figsize=self.figsize)
        plt.plot(alphas, losses, linewidth=2)
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Current parameters')
        plt.xlabel('Step size (α)')
        plt.ylabel('Loss')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def analyze_training_stability(
        self,
        history: Dict[str, List[float]],
        window_size: int = 10
    ) -> Dict[str, float]:
        """
        Analyze training stability metrics.
        
        Args:
            history: Training history dictionary
            window_size: Window size for variance computation
            
        Returns:
            Dictionary of stability metrics
        """
        def local_variance(data, window):
            variances = []
            for i in range(window, len(data)):
                window_data = data[i-window:i]
                variances.append(np.var(window_data))
            return np.mean(variances)
        
        metrics = {}
        
        # Loss variance
        metrics['loss_variance'] = local_variance(history['loss'], window_size)
        
        # Final convergence (slope of last 20% of training)
        final_portion = int(0.2 * len(history['loss']))
        if final_portion > 1:
            final_losses = history['loss'][-final_portion:]
            epochs = np.arange(len(final_losses))
            slope = np.polyfit(epochs, final_losses, 1)[0]
            metrics['final_slope'] = slope
        
        # Early stopping metric (best validation loss epoch)
        if 'val_loss' in history:
            best_epoch = np.argmin(history['val_loss']) + 1
            metrics['best_val_epoch'] = best_epoch
            metrics['epochs_after_best'] = len(history['val_loss']) - best_epoch
        
        return metrics
    
    def plot_training_summary(
        self,
        history: Dict[str, List[float]],
        title: str = "Training Summary"
    ) -> None:
        """
        Create comprehensive training summary plot.
        
        Args:
            history: Training history dictionary
            title: Plot title
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(history['loss']) + 1)
        
        # Loss curves
        ax1 = axes[0, 0]
        ax1.plot(epochs, history['loss'], label='Training Loss')
        if 'val_loss' in history:
            ax1.plot(epochs, history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Loss components
        ax2 = axes[0, 1]
        component_keys = ['reconstruction_loss', 'regularization_loss']
        for key in component_keys:
            if key in history:
                ax2.plot(epochs, history[key], label=key.replace('_', ' ').title())
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Loss Components')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Loss distribution
        ax3 = axes[1, 0]
        ax3.hist(history['loss'], bins=30, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Loss Value')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Loss Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Training statistics
        ax4 = axes[1, 1]
        stats_text = f"""
Training Statistics:
• Total Epochs: {len(epochs)}
• Final Loss: {history['loss'][-1]:.6f}
• Min Loss: {min(history['loss']):.6f}
• Max Loss: {max(history['loss']):.6f}
• Loss Std: {np.std(history['loss']):.6f}
        """
        
        if 'val_loss' in history:
            best_val_epoch = np.argmin(history['val_loss']) + 1
            stats_text += f"""
• Best Val Epoch: {best_val_epoch}
• Best Val Loss: {min(history['val_loss']):.6f}
        """
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Training Statistics')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary."""
        return {
            'figsize': self.figsize
        }