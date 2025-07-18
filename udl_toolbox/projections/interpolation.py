"""
Latent space interpolation utilities for autoencoders.
"""

import tensorflow as tf
import numpy as np
from typing import List, Tuple, Optional, Callable
import matplotlib.pyplot as plt


class LatentSpaceInterpolation:
    """
    Utilities for interpolating in autoencoder latent spaces.
    
    Provides various interpolation methods and visualization tools
    for exploring the structure of learned latent representations.
    """
    
    def __init__(self, autoencoder):
        """
        Initialize latent space interpolation utilities.
        
        Args:
            autoencoder: Fitted autoencoder model with encode/decode methods
        """
        self.autoencoder = autoencoder
    
    def linear_interpolation(
        self,
        z1: tf.Tensor,
        z2: tf.Tensor,
        num_steps: int = 10,
        include_endpoints: bool = True
    ) -> tf.Tensor:
        """
        Perform linear interpolation between two latent points.
        
        Args:
            z1: First latent point
            z2: Second latent point
            num_steps: Number of interpolation steps
            include_endpoints: Whether to include start and end points
            
        Returns:
            Interpolated latent points
        """
        if include_endpoints:
            alphas = np.linspace(0, 1, num_steps)
        else:
            alphas = np.linspace(0, 1, num_steps + 2)[1:-1]
        
        interpolated = []
        for alpha in alphas:
            z_interp = (1 - alpha) * z1 + alpha * z2
            interpolated.append(z_interp)
        
        return tf.stack(interpolated)
    
    def spherical_interpolation(
        self,
        z1: tf.Tensor,
        z2: tf.Tensor,
        num_steps: int = 10,
        include_endpoints: bool = True
    ) -> tf.Tensor:
        """
        Perform spherical linear interpolation (SLERP) between two latent points.
        
        Useful when latent space has spherical structure (e.g., normalized embeddings).
        
        Args:
            z1: First latent point
            z2: Second latent point
            num_steps: Number of interpolation steps
            include_endpoints: Whether to include start and end points
            
        Returns:
            Spherically interpolated latent points
        """
        # Normalize vectors
        z1_norm = z1 / tf.norm(z1)
        z2_norm = z2 / tf.norm(z2)
        
        # Compute angle between vectors
        dot_product = tf.reduce_sum(z1_norm * z2_norm)
        omega = tf.acos(tf.clip_by_value(dot_product, -1.0, 1.0))
        
        if include_endpoints:
            alphas = np.linspace(0, 1, num_steps)
        else:
            alphas = np.linspace(0, 1, num_steps + 2)[1:-1]
        
        interpolated = []
        for alpha in alphas:
            if tf.abs(omega) < 1e-6:  # Vectors are parallel
                z_interp = (1 - alpha) * z1 + alpha * z2
            else:
                sin_omega = tf.sin(omega)
                z_interp = (tf.sin((1 - alpha) * omega) * z1 + tf.sin(alpha * omega) * z2) / sin_omega
            interpolated.append(z_interp)
        
        return tf.stack(interpolated)
    
    def interpolate_data_points(
        self,
        x1: tf.Tensor,
        x2: tf.Tensor,
        num_steps: int = 10,
        method: str = 'linear',
        include_endpoints: bool = True
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Interpolate between two data points via latent space.
        
        Args:
            x1: First data point
            x2: Second data point
            num_steps: Number of interpolation steps
            method: Interpolation method ('linear' or 'spherical')
            include_endpoints: Whether to include start and end points
            
        Returns:
            Tuple of (interpolated_latent, interpolated_reconstructions)
        """
        # Encode data points to latent space
        z1 = self.autoencoder.encode(tf.expand_dims(x1, 0))[0]
        z2 = self.autoencoder.encode(tf.expand_dims(x2, 0))[0]
        
        # Interpolate in latent space
        if method == 'linear':
            z_interpolated = self.linear_interpolation(z1, z2, num_steps, include_endpoints)
        elif method == 'spherical':
            z_interpolated = self.spherical_interpolation(z1, z2, num_steps, include_endpoints)
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
        
        # Decode interpolated latent points
        x_interpolated = self.autoencoder.decode(z_interpolated)
        
        return z_interpolated, x_interpolated
    
    def latent_arithmetic(
        self,
        base_point: tf.Tensor,
        direction_vector: tf.Tensor,
        scales: List[float] = [-2, -1, 0, 1, 2]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Perform arithmetic operations in latent space.
        
        Useful for exploring semantic directions (e.g., "smile" direction in face images).
        
        Args:
            base_point: Base latent point
            direction_vector: Direction vector to add/subtract
            scales: List of scaling factors for the direction
            
        Returns:
            Tuple of (modified_latent_points, reconstructions)
        """
        modified_points = []
        for scale in scales:
            modified_point = base_point + scale * direction_vector
            modified_points.append(modified_point)
        
        z_modified = tf.stack(modified_points)
        x_reconstructed = self.autoencoder.decode(z_modified)
        
        return z_modified, x_reconstructed
    
    def find_semantic_direction(
        self,
        positive_examples: tf.Tensor,
        negative_examples: tf.Tensor,
        method: str = 'mean_difference'
    ) -> tf.Tensor:
        """
        Find semantic direction in latent space from examples.
        
        Args:
            positive_examples: Data points with desired attribute
            negative_examples: Data points without desired attribute
            method: Method to compute direction ('mean_difference', 'svm')
            
        Returns:
            Semantic direction vector
        """
        # Encode examples to latent space
        z_positive = self.autoencoder.encode(positive_examples)
        z_negative = self.autoencoder.encode(negative_examples)
        
        if method == 'mean_difference':
            # Simple mean difference
            direction = tf.reduce_mean(z_positive, axis=0) - tf.reduce_mean(z_negative, axis=0)
        elif method == 'svm':
            # Use SVM to find separating hyperplane
            from sklearn.svm import SVC
            
            # Prepare data for SVM
            X = tf.concat([z_positive, z_negative], axis=0).numpy()
            y = np.concatenate([np.ones(len(z_positive)), np.zeros(len(z_negative))])
            
            # Fit SVM
            svm = SVC(kernel='linear')
            svm.fit(X, y)
            
            # Direction is the normal to the hyperplane
            direction = tf.constant(svm.coef_[0], dtype=tf.float32)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Normalize direction
        direction = direction / tf.norm(direction)
        
        return direction
    
    def random_walk(
        self,
        start_point: tf.Tensor,
        num_steps: int = 10,
        step_size: float = 0.1,
        random_seed: Optional[int] = None
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Perform random walk in latent space.
        
        Args:
            start_point: Starting latent point
            num_steps: Number of steps in the walk
            step_size: Size of each random step
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (walk_points, reconstructions)
        """
        if random_seed is not None:
            tf.random.set_seed(random_seed)
        
        current_point = start_point
        walk_points = [current_point]
        
        for _ in range(num_steps):
            # Generate random step
            step = tf.random.normal(shape=tf.shape(current_point), stddev=step_size)
            current_point = current_point + step
            walk_points.append(current_point)
        
        walk_points = tf.stack(walk_points)
        reconstructions = self.autoencoder.decode(walk_points)
        
        return walk_points, reconstructions
    
    def latent_space_neighbors(
        self,
        query_point: tf.Tensor,
        latent_dataset: tf.Tensor,
        k: int = 5,
        metric: str = 'euclidean'
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Find k nearest neighbors in latent space.
        
        Args:
            query_point: Query latent point
            latent_dataset: Dataset of latent points to search
            k: Number of neighbors to find
            metric: Distance metric ('euclidean', 'cosine')
            
        Returns:
            Tuple of (neighbor_indices, neighbor_points, distances)
        """
        if metric == 'euclidean':
            # Compute Euclidean distances
            distances = tf.norm(latent_dataset - query_point, axis=1)
        elif metric == 'cosine':
            # Compute cosine distances
            query_norm = tf.norm(query_point)
            dataset_norms = tf.norm(latent_dataset, axis=1)
            dot_products = tf.reduce_sum(latent_dataset * query_point, axis=1)
            cosine_similarities = dot_products / (query_norm * dataset_norms)
            distances = 1 - cosine_similarities
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Find k nearest neighbors
        _, neighbor_indices = tf.nn.top_k(-distances, k=k)
        neighbor_points = tf.gather(latent_dataset, neighbor_indices)
        neighbor_distances = tf.gather(distances, neighbor_indices)
        
        return neighbor_indices, neighbor_points, neighbor_distances
    
    def interpolation_quality_metric(
        self,
        z_interpolated: tf.Tensor,
        smoothness_weight: float = 1.0
    ) -> float:
        """
        Compute quality metric for interpolation smoothness.
        
        Args:
            z_interpolated: Interpolated latent points
            smoothness_weight: Weight for smoothness term
            
        Returns:
            Quality metric (lower is better)
        """
        # Compute second derivatives (curvature)
        if len(z_interpolated) < 3:
            return 0.0
        
        # Second differences approximation
        second_diffs = z_interpolated[2:] - 2 * z_interpolated[1:-1] + z_interpolated[:-2]
        curvature = tf.reduce_mean(tf.norm(second_diffs, axis=1))
        
        return float(smoothness_weight * curvature)
    
    def visualize_interpolation(
        self,
        x_interpolated: tf.Tensor,
        title: str = "Latent Space Interpolation",
        figsize: Tuple[int, int] = (15, 3)
    ) -> None:
        """
        Visualize interpolation results.
        
        Args:
            x_interpolated: Interpolated reconstructions
            title: Plot title
            figsize: Figure size
        """
        n_samples = len(x_interpolated)
        
        plt.figure(figsize=figsize)
        for i in range(n_samples):
            plt.subplot(1, n_samples, i + 1)
            
            # Handle different data shapes
            if len(x_interpolated[i].shape) == 1:
                # 1D data - plot as line
                plt.plot(x_interpolated[i])
            elif len(x_interpolated[i].shape) == 2:
                # 2D data - show as heatmap
                plt.imshow(x_interpolated[i], cmap='viridis')
            elif len(x_interpolated[i].shape) == 3:
                # 3D data (images) - show as image
                if x_interpolated[i].shape[-1] == 1:
                    plt.imshow(x_interpolated[i][:, :, 0], cmap='gray')
                else:
                    plt.imshow(x_interpolated[i])
            
            plt.axis('off')
            plt.title(f'Step {i}')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def get_config(self) -> dict:
        """Get configuration dictionary."""
        return {
            'autoencoder_type': type(self.autoencoder).__name__
        }