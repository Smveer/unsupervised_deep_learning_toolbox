"""
Sparse autoencoder implementation with sparsity regularization.
"""

import tensorflow as tf
from typing import Dict, List
from .vanilla import VanillaAutoencoder
from ..losses.regularization import SparsityRegularization


class SparseAutoencoder(VanillaAutoencoder):
    """
    Sparse autoencoder that encourages sparse activations in the hidden layer.
    
    Uses KL divergence-based sparsity regularization to learn sparse representations,
    which can lead to more meaningful and interpretable feature learning.
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder_layers: List[int] = None,
        decoder_layers: List[int] = None,
        activation: str = 'sigmoid',  # Sigmoid is typically used for sparsity
        output_activation: str = 'sigmoid',
        learning_rate: float = 0.001,
        loss_type: str = 'mse',
        sparsity_target: float = 0.05,
        sparsity_weight: float = 1.0,
        dropout_rate: float = 0.0,
        use_batch_norm: bool = False,
        name: str = "sparse_autoencoder"
    ):
        """
        Initialize sparse autoencoder.
        
        Args:
            input_dim: Dimension of input data
            latent_dim: Dimension of latent space
            encoder_layers: List of hidden layer sizes for encoder
            decoder_layers: List of hidden layer sizes for decoder
            activation: Activation function for hidden layers (sigmoid recommended)
            output_activation: Activation function for output layer
            learning_rate: Learning rate for optimizer
            loss_type: Type of reconstruction loss
            sparsity_target: Target average activation (rho)
            sparsity_weight: Weight for sparsity regularization (beta)
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
            name: Name of the model
        """
        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight
        
        # Initialize sparsity regularization
        self.sparsity_fn = SparsityRegularization(
            sparsity_target=sparsity_target,
            sparsity_weight=sparsity_weight
        )
        
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            activation=activation,
            output_activation=output_activation,
            learning_rate=learning_rate,
            loss_type=loss_type,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            name=name
        )
        
        # Store activations for sparsity computation
        self.hidden_activations = None
    
    def _build_encoder(self) -> tf.keras.Model:
        """Build the encoder network with sparsity monitoring."""
        inputs = tf.keras.Input(shape=(self.input_dim,), name=f"{self.name}_encoder_input")
        x = inputs
        
        # Hidden layers
        for i, units in enumerate(self.encoder_layers):
            x = tf.keras.layers.Dense(
                units,
                activation=self.activation,
                name=f"{self.name}_encoder_dense_{i}"
            )(x)
            
            if self.use_batch_norm:
                x = tf.keras.layers.BatchNormalization(
                    name=f"{self.name}_encoder_bn_{i}"
                )(x)
            
            if self.dropout_rate > 0:
                x = tf.keras.layers.Dropout(
                    self.dropout_rate,
                    name=f"{self.name}_encoder_dropout_{i}"
                )(x)
        
        # Latent layer with activation (important for sparsity)
        latent = tf.keras.layers.Dense(
            self.latent_dim,
            activation=self.activation,  # Use activation (typically sigmoid) for sparsity
            name=f"{self.name}_latent"
        )(x)
        
        encoder = tf.keras.Model(inputs, latent, name=f"{self.name}_encoder")
        
        # Create a model that also outputs the last hidden layer for sparsity computation
        if self.encoder_layers:
            # If we have hidden layers, use the last one for sparsity
            hidden_output = encoder.layers[-2].output  # Second to last layer (before latent)
        else:
            # If no hidden layers, use the latent layer itself
            hidden_output = latent
        
        self.encoder_with_hidden = tf.keras.Model(
            inputs, 
            [latent, hidden_output], 
            name=f"{self.name}_encoder_with_hidden"
        )
        
        return encoder
    
    @tf.function
    def _train_step(self, x: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        Perform a single training step with sparsity regularization.
        
        Args:
            x: Input data batch
            
        Returns:
            Dictionary of loss values
        """
        with tf.GradientTape() as tape:
            # Forward pass through encoder (get both latent and hidden activations)
            latent, hidden_activations = self.encoder_with_hidden(x, training=True)
            
            # Forward pass through decoder
            x_reconstructed = self.decoder(latent, training=True)
            
            # Compute reconstruction loss
            reconstruction_loss = self.loss_fn(x, x_reconstructed)
            
            # Compute sparsity loss
            sparsity_loss = self.sparsity_fn(hidden_activations)
            
            # Total loss
            total_loss = reconstruction_loss + sparsity_loss
        
        # Compute gradients
        gradients = tape.gradient(total_loss, self.autoencoder.trainable_variables)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.autoencoder.trainable_variables))
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'regularization_loss': sparsity_loss,
            'sparsity_loss': sparsity_loss
        }
    
    def _compute_loss(self, x: tf.Tensor, x_reconstructed: tf.Tensor, **kwargs) -> Dict[str, tf.Tensor]:
        """
        Compute the total loss including sparsity regularization.
        
        Args:
            x: Original input data
            x_reconstructed: Reconstructed data from autoencoder
            
        Returns:
            Dictionary containing loss components
        """
        # Reconstruction loss
        reconstruction_loss = self.loss_fn(x, x_reconstructed)
        
        # Get hidden activations for sparsity computation
        _, hidden_activations = self.encoder_with_hidden(x, training=False)
        sparsity_loss = self.sparsity_fn(hidden_activations)
        
        # Total loss
        total_loss = reconstruction_loss + sparsity_loss
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'regularization_loss': sparsity_loss,
            'sparsity_loss': sparsity_loss
        }
    
    def get_sparsity_statistics(self, x: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        Get sparsity statistics for the hidden layer.
        
        Args:
            x: Input data
            
        Returns:
            Dictionary with sparsity statistics
        """
        _, hidden_activations = self.encoder_with_hidden(x, training=False)
        
        # Compute average activation per neuron
        avg_activation = tf.reduce_mean(hidden_activations, axis=0)
        
        # Compute sparsity metrics
        sparsity_ratio = tf.reduce_mean(tf.cast(hidden_activations < 0.1, tf.float32))
        active_neurons = tf.reduce_sum(tf.cast(avg_activation > 0.1, tf.float32))
        
        return {
            'average_activation': avg_activation,
            'sparsity_ratio': sparsity_ratio,
            'active_neurons': active_neurons,
            'total_neurons': tf.constant(float(hidden_activations.shape[-1]))
        }
    
    def visualize_sparsity(self, x: tf.Tensor) -> None:
        """
        Print sparsity statistics.
        
        Args:
            x: Input data for analysis
        """
        stats = self.get_sparsity_statistics(x)
        
        print("=== Sparsity Statistics ===")
        print(f"Target sparsity: {self.sparsity_target:.3f}")
        print(f"Actual sparsity ratio: {float(stats['sparsity_ratio']):.3f}")
        print(f"Active neurons: {int(stats['active_neurons'])}/{int(stats['total_neurons'])}")
        print(f"Average activation: {float(tf.reduce_mean(stats['average_activation'])):.6f}")
        print(f"Min activation: {float(tf.reduce_min(stats['average_activation'])):.6f}")
        print(f"Max activation: {float(tf.reduce_max(stats['average_activation'])):.6f}")
    
    def get_config(self) -> Dict:
        """Get configuration dictionary."""
        config = super().get_config()
        config.update({
            'sparsity_target': self.sparsity_target,
            'sparsity_weight': self.sparsity_weight
        })
        return config