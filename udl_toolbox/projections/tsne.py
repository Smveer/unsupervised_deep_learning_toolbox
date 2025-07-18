"""
t-SNE-like projection implementation for dimensionality reduction and visualization.
"""

import tensorflow as tf
import numpy as np
from typing import Optional, Callable
from sklearn.manifold import TSNE as SklearnTSNE
from sklearn.preprocessing import StandardScaler


class TSNEProjection:
    """
    t-SNE-like projection for non-linear dimensionality reduction.
    
    This is a simplified implementation that uses sklearn's t-SNE internally
    but provides a consistent interface with other projection methods.
    For a full from-scratch implementation, this would require implementing
    the complete t-SNE algorithm with gradient descent optimization.
    """
    
    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        learning_rate: float = 200.0,
        n_iter: int = 1000,
        random_state: Optional[int] = None,
        standardize: bool = True
    ):
        """
        Initialize t-SNE projection.
        
        Args:
            n_components: Dimension of the embedded space
            perplexity: Related to the number of nearest neighbors
            learning_rate: Learning rate for t-SNE optimization
            n_iter: Maximum number of iterations
            random_state: Random seed for reproducibility
            standardize: Whether to standardize data before t-SNE
        """
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state
        self.standardize = standardize
        
        # Fitted parameters
        self.embedding_ = None
        self.scaler_ = None
        self.is_fitted = False
        
        # Internal t-SNE model
        self.tsne_model = None
    
    def fit(self, X: np.ndarray) -> 'TSNEProjection':
        """
        Fit t-SNE to the data.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Self for method chaining
        """
        X = np.array(X)
        
        # Standardize if requested
        if self.standardize:
            self.scaler_ = StandardScaler()
            X_processed = self.scaler_.fit_transform(X)
        else:
            X_processed = X.copy()
        
        # Initialize t-SNE model
        self.tsne_model = SklearnTSNE(
            n_components=self.n_components,
            perplexity=self.perplexity,
            learning_rate=self.learning_rate,
            n_iter=self.n_iter,
            random_state=self.random_state
        )
        
        # Fit and transform
        self.embedding_ = self.tsne_model.fit_transform(X_processed)
        self.is_fitted = True
        
        return self
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit t-SNE and return embedding.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Embedded data of shape (n_samples, n_components)
        """
        self.fit(X)
        return self.embedding_
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform new data using fitted t-SNE.
        
        Note: t-SNE doesn't naturally support out-of-sample extension.
        This method raises an error and suggests alternatives.
        
        Args:
            X: New input data
            
        Raises:
            NotImplementedError: t-SNE doesn't support out-of-sample extension
        """
        raise NotImplementedError(
            "t-SNE doesn't support out-of-sample extension. "
            "Use fit_transform on all data at once, or consider using "
            "parametric t-SNE or other methods for new data projection."
        )
    
    def approximate_transform(self, X_new: np.ndarray, X_original: np.ndarray) -> np.ndarray:
        """
        Approximate transformation for new data using k-nearest neighbors.
        
        This is a workaround for t-SNE's lack of out-of-sample extension.
        
        Args:
            X_new: New data to transform
            X_original: Original training data
            
        Returns:
            Approximate embedding for new data
        """
        if not self.is_fitted:
            raise ValueError("t-SNE must be fitted first")
        
        from sklearn.neighbors import NearestNeighbors
        
        # Standardize new data using fitted scaler
        if self.standardize:
            X_new_processed = self.scaler_.transform(X_new)
            X_original_processed = self.scaler_.transform(X_original)
        else:
            X_new_processed = X_new.copy()
            X_original_processed = X_original.copy()
        
        # Find nearest neighbors in original data
        nn = NearestNeighbors(n_neighbors=min(5, len(X_original_processed)))
        nn.fit(X_original_processed)
        
        distances, indices = nn.kneighbors(X_new_processed)
        
        # Weighted average of embeddings of nearest neighbors
        weights = 1 / (distances + 1e-8)  # Inverse distance weighting
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        
        embeddings_new = []
        for i in range(len(X_new)):
            neighbor_embeddings = self.embedding_[indices[i]]
            weighted_embedding = np.average(neighbor_embeddings, weights=weights[i], axis=0)
            embeddings_new.append(weighted_embedding)
        
        return np.array(embeddings_new)
    
    def get_embedding(self) -> np.ndarray:
        """
        Get the fitted embedding.
        
        Returns:
            Embedded data
        """
        if not self.is_fitted:
            raise ValueError("t-SNE must be fitted first")
        return self.embedding_
    
    def get_kl_divergence(self) -> float:
        """
        Get the final KL divergence of the t-SNE embedding.
        
        Returns:
            KL divergence value
        """
        if not self.is_fitted:
            raise ValueError("t-SNE must be fitted first")
        return self.tsne_model.kl_divergence_
    
    def get_config(self) -> dict:
        """Get configuration dictionary."""
        return {
            'n_components': self.n_components,
            'perplexity': self.perplexity,
            'learning_rate': self.learning_rate,
            'n_iter': self.n_iter,
            'random_state': self.random_state,
            'standardize': self.standardize
        }


class ParametricTSNE:
    """
    Parametric t-SNE implementation using neural networks.
    
    This provides a way to learn a mapping function that can be applied
    to new data, addressing the out-of-sample problem of standard t-SNE.
    """
    
    def __init__(
        self,
        n_components: int = 2,
        hidden_layers: list = [500, 500, 2000],
        perplexity: float = 30.0,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        batch_size: int = 500,
        standardize: bool = True
    ):
        """
        Initialize parametric t-SNE.
        
        Args:
            n_components: Dimension of the embedded space
            hidden_layers: List of hidden layer sizes
            perplexity: Related to the number of nearest neighbors
            learning_rate: Learning rate for optimization
            epochs: Number of training epochs
            batch_size: Batch size for training
            standardize: Whether to standardize input data
        """
        self.n_components = n_components
        self.hidden_layers = hidden_layers
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.standardize = standardize
        
        # Model components
        self.model = None
        self.scaler_ = None
        self.is_fitted = False
    
    def _build_model(self, input_dim: int) -> tf.keras.Model:
        """Build the parametric t-SNE neural network."""
        inputs = tf.keras.Input(shape=(input_dim,))
        x = inputs
        
        # Hidden layers
        for i, units in enumerate(self.hidden_layers):
            x = tf.keras.layers.Dense(units, activation='relu', name=f'hidden_{i}')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.1)(x)
        
        # Output layer
        outputs = tf.keras.layers.Dense(self.n_components, activation='linear', name='embedding')(x)
        
        return tf.keras.Model(inputs, outputs)
    
    def _compute_pairwise_distances(self, X: tf.Tensor) -> tf.Tensor:
        """Compute pairwise squared Euclidean distances."""
        # X shape: (batch_size, n_features)
        sum_X = tf.reduce_sum(tf.square(X), axis=1, keepdims=True)
        distances = sum_X + tf.transpose(sum_X) - 2 * tf.matmul(X, tf.transpose(X))
        return tf.maximum(distances, 0.0)  # Ensure non-negative
    
    def _compute_p_similarities(self, distances: tf.Tensor) -> tf.Tensor:
        """Compute P similarities for high-dimensional space."""
        # Convert distances to similarities using Gaussian kernel
        # This is a simplified version - full t-SNE uses perplexity calibration
        sigma = 1.0  # Simplified: should be calibrated based on perplexity
        similarities = tf.exp(-distances / (2 * sigma ** 2))
        
        # Symmetrize and normalize
        similarities = (similarities + tf.transpose(similarities)) / 2
        similarities = similarities / tf.reduce_sum(similarities)
        
        # Add small epsilon to avoid numerical issues
        return tf.maximum(similarities, 1e-12)
    
    def _compute_q_similarities(self, Y: tf.Tensor) -> tf.Tensor:
        """Compute Q similarities for low-dimensional space using t-distribution."""
        distances = self._compute_pairwise_distances(Y)
        similarities = 1.0 / (1.0 + distances)
        
        # Set diagonal to 0 and normalize
        similarities = similarities * (1 - tf.eye(tf.shape(similarities)[0]))
        similarities = similarities / tf.reduce_sum(similarities)
        
        return tf.maximum(similarities, 1e-12)
    
    @tf.function
    def _tsne_loss(self, X: tf.Tensor, Y: tf.Tensor) -> tf.Tensor:
        """Compute t-SNE KL divergence loss."""
        # Compute similarities
        P = self._compute_p_similarities(self._compute_pairwise_distances(X))
        Q = self._compute_q_similarities(Y)
        
        # KL divergence: sum(P * log(P / Q))
        kl_div = tf.reduce_sum(P * tf.math.log(P / Q))
        return kl_div
    
    def fit(self, X: np.ndarray) -> 'ParametricTSNE':
        """
        Fit parametric t-SNE to the data.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Self for method chaining
        """
        X = np.array(X)
        
        # Standardize if requested
        if self.standardize:
            self.scaler_ = StandardScaler()
            X_processed = self.scaler_.fit_transform(X)
        else:
            X_processed = X.copy()
        
        # Build model
        self.model = self._build_model(X_processed.shape[1])
        
        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer)
        
        # Convert to tensor dataset
        dataset = tf.data.Dataset.from_tensor_slices(X_processed)
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        # Training loop
        for epoch in range(self.epochs):
            epoch_loss = 0
            num_batches = 0
            
            for batch in dataset:
                with tf.GradientTape() as tape:
                    Y = self.model(batch, training=True)
                    loss = self._tsne_loss(batch, Y)
                
                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                
                epoch_loss += loss
                num_batches += 1
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss / num_batches:.6f}")
        
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted parametric t-SNE.
        
        Args:
            X: Input data to transform
            
        Returns:
            Embedded data
        """
        if not self.is_fitted:
            raise ValueError("Parametric t-SNE must be fitted first")
        
        # Standardize using fitted scaler
        if self.standardize:
            X_processed = self.scaler_.transform(X)
        else:
            X_processed = X.copy()
        
        return self.model(X_processed, training=False).numpy()
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit parametric t-SNE and transform data.
        
        Args:
            X: Input data
            
        Returns:
            Embedded data
        """
        self.fit(X)
        return self.transform(X)
    
    def get_config(self) -> dict:
        """Get configuration dictionary."""
        return {
            'n_components': self.n_components,
            'hidden_layers': self.hidden_layers,
            'perplexity': self.perplexity,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'standardize': self.standardize
        }