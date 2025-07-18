"""
Latent space visualization utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tensorflow as tf
from typing import Optional, List, Tuple, Dict, Any
from ..projections import PCAProjection, TSNEProjection


class LatentSpaceVisualizer:
    """
    Comprehensive visualization tools for autoencoder latent spaces.
    
    Provides methods for visualizing latent representations, distributions,
    and relationships between data points in the encoded space.
    """
    
    def __init__(self, autoencoder, figsize: Tuple[int, int] = (10, 8)):
        """
        Initialize latent space visualizer.
        
        Args:
            autoencoder: Fitted autoencoder model
            figsize: Default figure size for matplotlib plots
        """
        self.autoencoder = autoencoder
        self.figsize = figsize
        
    def plot_2d_latent_space(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        method: str = 'direct',
        title: str = "2D Latent Space",
        save_path: Optional[str] = None,
        interactive: bool = False
    ) -> None:
        """
        Plot 2D visualization of latent space.
        
        Args:
            data: Input data to encode and visualize
            labels: Optional labels for coloring points
            method: Visualization method ('direct', 'pca', 'tsne')
            title: Plot title
            save_path: Optional path to save the plot
            interactive: Whether to create interactive plotly plot
        """
        # Encode data to latent space
        latent_data = self.autoencoder.encode(data).numpy()
        
        # Reduce to 2D if necessary
        if latent_data.shape[1] == 2:
            latent_2d = latent_data
        elif method == 'direct' and latent_data.shape[1] > 2:
            # Use first 2 dimensions
            latent_2d = latent_data[:, :2]
        elif method == 'pca':
            pca = PCAProjection(n_components=2)
            latent_2d = pca.fit_transform(latent_data)
        elif method == 'tsne':
            tsne = TSNEProjection(n_components=2)
            latent_2d = tsne.fit_transform(latent_data)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if interactive:
            self._plot_2d_interactive(latent_2d, labels, title)
        else:
            self._plot_2d_static(latent_2d, labels, title, save_path)
    
    def _plot_2d_static(
        self,
        latent_2d: np.ndarray,
        labels: Optional[np.ndarray],
        title: str,
        save_path: Optional[str]
    ) -> None:
        """Create static matplotlib 2D plot."""
        plt.figure(figsize=self.figsize)
        
        if labels is not None:
            scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
            plt.colorbar(scatter)
        else:
            plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.7)
        
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_2d_interactive(
        self,
        latent_2d: np.ndarray,
        labels: Optional[np.ndarray],
        title: str
    ) -> None:
        """Create interactive plotly 2D plot."""
        if labels is not None:
            fig = px.scatter(
                x=latent_2d[:, 0],
                y=latent_2d[:, 1],
                color=labels,
                title=title,
                labels={'x': 'Latent Dimension 1', 'y': 'Latent Dimension 2'}
            )
        else:
            fig = px.scatter(
                x=latent_2d[:, 0],
                y=latent_2d[:, 1],
                title=title,
                labels={'x': 'Latent Dimension 1', 'y': 'Latent Dimension 2'}
            )
        
        fig.show()
    
    def plot_3d_latent_space(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        method: str = 'direct',
        title: str = "3D Latent Space",
        interactive: bool = True
    ) -> None:
        """
        Plot 3D visualization of latent space.
        
        Args:
            data: Input data to encode and visualize
            labels: Optional labels for coloring points
            method: Visualization method ('direct', 'pca')
            title: Plot title
            interactive: Whether to create interactive plot
        """
        # Encode data to latent space
        latent_data = self.autoencoder.encode(data).numpy()
        
        # Reduce to 3D if necessary
        if latent_data.shape[1] == 3:
            latent_3d = latent_data
        elif method == 'direct' and latent_data.shape[1] > 3:
            # Use first 3 dimensions
            latent_3d = latent_data[:, :3]
        elif method == 'pca':
            pca = PCAProjection(n_components=3)
            latent_3d = pca.fit_transform(latent_data)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if interactive:
            self._plot_3d_interactive(latent_3d, labels, title)
        else:
            self._plot_3d_static(latent_3d, labels, title)
    
    def _plot_3d_static(
        self,
        latent_3d: np.ndarray,
        labels: Optional[np.ndarray],
        title: str
    ) -> None:
        """Create static matplotlib 3D plot."""
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        if labels is not None:
            scatter = ax.scatter(latent_3d[:, 0], latent_3d[:, 1], latent_3d[:, 2], 
                               c=labels, cmap='tab10', alpha=0.7)
            plt.colorbar(scatter)
        else:
            ax.scatter(latent_3d[:, 0], latent_3d[:, 1], latent_3d[:, 2], alpha=0.7)
        
        ax.set_xlabel('Latent Dimension 1')
        ax.set_ylabel('Latent Dimension 2')
        ax.set_zlabel('Latent Dimension 3')
        ax.set_title(title)
        
        plt.show()
    
    def _plot_3d_interactive(
        self,
        latent_3d: np.ndarray,
        labels: Optional[np.ndarray],
        title: str
    ) -> None:
        """Create interactive plotly 3D plot."""
        if labels is not None:
            fig = px.scatter_3d(
                x=latent_3d[:, 0],
                y=latent_3d[:, 1],
                z=latent_3d[:, 2],
                color=labels,
                title=title,
                labels={'x': 'Latent Dimension 1', 'y': 'Latent Dimension 2', 'z': 'Latent Dimension 3'}
            )
        else:
            fig = px.scatter_3d(
                x=latent_3d[:, 0],
                y=latent_3d[:, 1],
                z=latent_3d[:, 2],
                title=title,
                labels={'x': 'Latent Dimension 1', 'y': 'Latent Dimension 2', 'z': 'Latent Dimension 3'}
            )
        
        fig.show()
    
    def plot_latent_distributions(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        max_dims: int = 10,
        title: str = "Latent Dimension Distributions"
    ) -> None:
        """
        Plot distributions of latent dimensions.
        
        Args:
            data: Input data to encode
            labels: Optional labels for group comparisons
            max_dims: Maximum number of dimensions to plot
            title: Plot title
        """
        # Encode data
        latent_data = self.autoencoder.encode(data).numpy()
        n_dims = min(latent_data.shape[1], max_dims)
        
        # Create subplots
        n_cols = min(4, n_dims)
        n_rows = (n_dims + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_dims):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]
            
            if labels is not None:
                # Plot separate distributions for each label
                unique_labels = np.unique(labels)
                for label in unique_labels:
                    mask = labels == label
                    ax.hist(latent_data[mask, i], alpha=0.7, bins=30, 
                           label=f'Class {label}', density=True)
                ax.legend()
            else:
                ax.hist(latent_data[:, i], bins=30, alpha=0.7, density=True)
            
            ax.set_title(f'Latent Dimension {i}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_dims, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].set_visible(False)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def plot_latent_correlation_matrix(
        self,
        data: np.ndarray,
        title: str = "Latent Dimension Correlations"
    ) -> None:
        """
        Plot correlation matrix of latent dimensions.
        
        Args:
            data: Input data to encode
            title: Plot title
        """
        # Encode data
        latent_data = self.autoencoder.encode(data).numpy()
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(latent_data.T)
        
        # Plot heatmap
        plt.figure(figsize=self.figsize)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, fmt='.2f', cbar_kws={'label': 'Correlation'})
        plt.title(title)
        plt.xlabel('Latent Dimension')
        plt.ylabel('Latent Dimension')
        plt.show()
    
    def plot_latent_manifold_2d(
        self,
        grid_size: int = 20,
        latent_range: Tuple[float, float] = (-3, 3),
        title: str = "Latent Space Manifold"
    ) -> None:
        """
        Plot 2D latent space manifold (for 2D latent spaces).
        
        Args:
            grid_size: Size of the grid for sampling
            latent_range: Range for latent space sampling
            title: Plot title
        """
        if self.autoencoder.latent_dim != 2:
            raise ValueError("Manifold visualization only supported for 2D latent spaces")
        
        # Create grid of latent points
        x = np.linspace(latent_range[0], latent_range[1], grid_size)
        y = np.linspace(latent_range[0], latent_range[1], grid_size)
        xx, yy = np.meshgrid(x, y)
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        
        # Decode grid points
        reconstructions = self.autoencoder.decode(grid_points).numpy()
        
        # Plot grid of reconstructions
        n_cols = grid_size
        fig, axes = plt.subplots(grid_size, n_cols, figsize=(20, 20))
        
        for i in range(grid_size):
            for j in range(n_cols):
                idx = i * n_cols + j
                ax = axes[i, j]
                
                # Handle different data types
                if len(reconstructions[idx].shape) == 1:
                    # 1D data
                    ax.plot(reconstructions[idx])
                elif len(reconstructions[idx].shape) == 2:
                    # 2D data or grayscale image
                    if reconstructions[idx].shape[-1] == 1:
                        ax.imshow(reconstructions[idx][:, :, 0], cmap='gray')
                    else:
                        ax.imshow(reconstructions[idx], cmap='viridis')
                elif len(reconstructions[idx].shape) == 3:
                    # Color image
                    ax.imshow(reconstructions[idx])
                
                ax.axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def plot_latent_interpolation_path(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        num_steps: int = 10,
        title: str = "Latent Space Interpolation"
    ) -> None:
        """
        Visualize interpolation path in 2D latent space.
        
        Args:
            start_point: Starting point in input space
            end_point: Ending point in input space
            num_steps: Number of interpolation steps
            title: Plot title
        """
        from ..projections.interpolation import LatentSpaceInterpolation
        
        interpolator = LatentSpaceInterpolation(self.autoencoder)
        z_interp, x_interp = interpolator.interpolate_data_points(
            start_point, end_point, num_steps
        )
        
        # Plot interpolation path in latent space (if 2D)
        if self.autoencoder.latent_dim == 2:
            plt.figure(figsize=self.figsize)
            plt.plot(z_interp[:, 0], z_interp[:, 1], 'o-', linewidth=2, markersize=8)
            plt.scatter(z_interp[0, 0], z_interp[0, 1], color='green', s=100, label='Start')
            plt.scatter(z_interp[-1, 0], z_interp[-1, 1], color='red', s=100, label='End')
            plt.xlabel('Latent Dimension 1')
            plt.ylabel('Latent Dimension 2')
            plt.title(f'{title} - Latent Path')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
        
        # Plot reconstructed interpolation
        interpolator.visualize_interpolation(x_interp, title)
    
    def plot_cluster_analysis(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        method: str = 'pca',
        title: str = "Latent Space Clusters"
    ) -> None:
        """
        Analyze and visualize clusters in latent space.
        
        Args:
            data: Input data
            labels: Cluster labels
            method: Dimensionality reduction method for visualization
            title: Plot title
        """
        # Encode data
        latent_data = self.autoencoder.encode(data).numpy()
        
        # Reduce dimensionality for visualization
        if method == 'pca':
            reducer = PCAProjection(n_components=2)
            latent_2d = reducer.fit_transform(latent_data)
        elif method == 'tsne':
            reducer = TSNEProjection(n_components=2)
            latent_2d = reducer.fit_transform(latent_data)
        else:
            latent_2d = latent_data[:, :2]
        
        # Plot clusters
        plt.figure(figsize=self.figsize)
        scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
        
        # Add cluster centers
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            center = np.mean(latent_2d[mask], axis=0)
            plt.scatter(center[0], center[1], marker='x', s=200, color='black', linewidth=3)
        
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary."""
        return {
            'autoencoder_type': type(self.autoencoder).__name__,
            'latent_dim': self.autoencoder.latent_dim,
            'figsize': self.figsize
        }