"""
Vanilla (basic) autoencoder implementation.
"""

import tensorflow as tf
from typing import Dict, List
from .base import BaseAutoencoder
from ..losses.reconstruction import MeanSquaredError, BinaryCrossentropy


class VanillaAutoencoder(BaseAutoencoder):
    """
    Basic autoencoder with fully connected encoder and decoder networks.
    
    This is the simplest form of autoencoder that learns to compress
    and reconstruct input data without any additional constraints.
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder_layers: List[int] = None,
        decoder_layers: List[int] = None,
        activation: str = 'relu',
        output_activation: str = 'sigmoid',
        learning_rate: float = 0.001,
        loss_type: str = 'mse',
        dropout_rate: float = 0.0,
        use_batch_norm: bool = False,
        name: str = "vanilla_autoencoder"
    ):
        """
        Initialize vanilla autoencoder.
        
        Args:
            input_dim: Dimension of input data
            latent_dim: Dimension of latent space
            encoder_layers: List of hidden layer sizes for encoder (default: [input_dim//2])
            decoder_layers: List of hidden layer sizes for decoder (default: symmetric to encoder)
            activation: Activation function for hidden layers
            output_activation: Activation function for output layer
            learning_rate: Learning rate for optimizer
            loss_type: Type of reconstruction loss ('mse' or 'binary_crossentropy')
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
            name: Name of the model
        """
        # Set default layer configurations
        if encoder_layers is None:
            encoder_layers = [input_dim // 2]
        if decoder_layers is None:
            # Mirror encoder layers
            decoder_layers = encoder_layers[::-1]
        
        self.loss_type = loss_type
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Initialize loss function
        if loss_type == 'mse':
            self.loss_fn = MeanSquaredError()
        elif loss_type == 'binary_crossentropy':
            self.loss_fn = BinaryCrossentropy()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            activation=activation,
            output_activation=output_activation,
            learning_rate=learning_rate,
            name=name
        )
    
    def _build_encoder(self) -> tf.keras.Model:
        """Build the encoder network."""
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
        
        # Latent layer
        latent = tf.keras.layers.Dense(
            self.latent_dim,
            activation='linear',  # Linear activation for latent space
            name=f"{self.name}_latent"
        )(x)
        
        return tf.keras.Model(inputs, latent, name=f"{self.name}_encoder")
    
    def _build_decoder(self) -> tf.keras.Model:
        """Build the decoder network."""
        inputs = tf.keras.Input(shape=(self.latent_dim,), name=f"{self.name}_decoder_input")
        x = inputs
        
        # Hidden layers
        for i, units in enumerate(self.decoder_layers):
            x = tf.keras.layers.Dense(
                units,
                activation=self.activation,
                name=f"{self.name}_decoder_dense_{i}"
            )(x)
            
            if self.use_batch_norm:
                x = tf.keras.layers.BatchNormalization(
                    name=f"{self.name}_decoder_bn_{i}"
                )(x)
            
            if self.dropout_rate > 0:
                x = tf.keras.layers.Dropout(
                    self.dropout_rate,
                    name=f"{self.name}_decoder_dropout_{i}"
                )(x)
        
        # Output layer
        outputs = tf.keras.layers.Dense(
            self.input_dim,
            activation=self.output_activation,
            name=f"{self.name}_output"
        )(x)
        
        return tf.keras.Model(inputs, outputs, name=f"{self.name}_decoder")
    
    def _compute_loss(self, x: tf.Tensor, x_reconstructed: tf.Tensor, **kwargs) -> Dict[str, tf.Tensor]:
        """
        Compute the reconstruction loss for vanilla autoencoder.
        
        Args:
            x: Original input data
            x_reconstructed: Reconstructed data from autoencoder
            
        Returns:
            Dictionary containing loss components
        """
        reconstruction_loss = self.loss_fn(x, x_reconstructed)
        
        return {
            'total_loss': reconstruction_loss,
            'reconstruction_loss': reconstruction_loss,
            'regularization_loss': tf.constant(0.0)
        }
    
    def get_config(self) -> Dict:
        """Get configuration dictionary."""
        config = super().get_config()
        config.update({
            'loss_type': self.loss_type,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm
        })
        return config
    
    def compress(self, x: tf.Tensor, compression_ratio: float = None) -> tf.Tensor:
        """
        Compress input data to latent representation.
        
        Args:
            x: Input data
            compression_ratio: If provided, calculates actual compression achieved
            
        Returns:
            Compressed latent representation
        """
        latent = self.encode(x)
        
        if compression_ratio is not None:
            actual_ratio = self.input_dim / self.latent_dim
            print(f"Compression ratio: {actual_ratio:.2f}:1")
            if compression_ratio != actual_ratio:
                print(f"Note: Requested ratio {compression_ratio:.2f}:1, actual ratio {actual_ratio:.2f}:1")
        
        return latent
    
    def decompress(self, latent: tf.Tensor) -> tf.Tensor:
        """
        Decompress latent representation to original space.
        
        Args:
            latent: Latent representation
            
        Returns:
            Reconstructed data
        """
        return self.decode(latent)