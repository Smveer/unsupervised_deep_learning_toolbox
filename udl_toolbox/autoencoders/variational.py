"""
Variational autoencoder implementation.
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple
from .base import BaseAutoencoder
from ..losses.vae_loss import VAELoss


class VariationalAutoencoder(BaseAutoencoder):
    """
    Variational Autoencoder (VAE) implementation.
    
    VAE learns a probabilistic latent representation by encoding inputs as
    distributions (mean and variance) rather than fixed points, and uses
    the reparameterization trick for training.
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
        beta: float = 1.0,
        reconstruction_loss: str = 'mse',
        dropout_rate: float = 0.0,
        use_batch_norm: bool = False,
        name: str = "variational_autoencoder"
    ):
        """
        Initialize variational autoencoder.
        
        Args:
            input_dim: Dimension of input data
            latent_dim: Dimension of latent space
            encoder_layers: List of hidden layer sizes for encoder
            decoder_layers: List of hidden layer sizes for decoder
            activation: Activation function for hidden layers
            output_activation: Activation function for output layer
            learning_rate: Learning rate for optimizer
            beta: Beta parameter for KL divergence weighting
            reconstruction_loss: Type of reconstruction loss ('mse' or 'binary_crossentropy')
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
            name: Name of the model
        """
        # Set default layer configurations
        if encoder_layers is None:
            encoder_layers = [input_dim // 2]
        if decoder_layers is None:
            decoder_layers = encoder_layers[::-1]
        
        self.beta = beta
        self.reconstruction_loss_type = reconstruction_loss
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Initialize VAE loss
        self.vae_loss_fn = VAELoss(
            reconstruction_loss=reconstruction_loss,
            beta=beta
        )
        
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
        """Build the encoder network that outputs mean and log variance."""
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
        
        # Mean and log variance layers
        z_mean = tf.keras.layers.Dense(
            self.latent_dim,
            activation='linear',
            name=f"{self.name}_z_mean"
        )(x)
        
        z_log_var = tf.keras.layers.Dense(
            self.latent_dim,
            activation='linear',
            name=f"{self.name}_z_log_var"
        )(x)
        
        # Sampling layer
        z = self._sampling_layer([z_mean, z_log_var])
        
        return tf.keras.Model(inputs, [z_mean, z_log_var, z], name=f"{self.name}_encoder")
    
    def _sampling_layer(self, args: List[tf.Tensor]) -> tf.Tensor:
        """
        Reparameterization trick: sample from latent distribution.
        
        Args:
            args: [z_mean, z_log_var]
            
        Returns:
            Sampled latent vector
        """
        z_mean, z_log_var = args
        batch_size = tf.shape(z_mean)[0]
        epsilon = tf.random.normal(shape=(batch_size, self.latent_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
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
    
    def _build_model(self):
        """Build the complete VAE model."""
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        
        # Build the full VAE
        inputs = tf.keras.Input(shape=(self.input_dim,), name=f"{self.name}_input")
        z_mean, z_log_var, z = self.encoder(inputs)
        decoded = self.decoder(z)
        
        self.autoencoder = tf.keras.Model(inputs, decoded, name=self.name)
        
        # Create models for individual components
        self.encoder_mean_var = tf.keras.Model(inputs, [z_mean, z_log_var], name=f"{self.name}_encoder_mean_var")
        
        # Initialize optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
    
    def encode(self, x: tf.Tensor, return_distribution: bool = False) -> tf.Tensor:
        """
        Encode input data to latent space.
        
        Args:
            x: Input data tensor
            return_distribution: If True, return (mean, log_var, sample), else just sample
            
        Returns:
            Encoded representation(s)
        """
        if return_distribution:
            return self.encoder(x)
        else:
            z_mean, z_log_var, z = self.encoder(x)
            return z
    
    def encode_mean(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Encode input to mean and log variance (no sampling).
        
        Args:
            x: Input data tensor
            
        Returns:
            Tuple of (mean, log_var)
        """
        return self.encoder_mean_var(x)
    
    @tf.function
    def _train_step(self, x: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        Perform a single training step for VAE.
        
        Args:
            x: Input data batch
            
        Returns:
            Dictionary of loss values
        """
        with tf.GradientTape() as tape:
            # Forward pass
            z_mean, z_log_var, z = self.encoder(x, training=True)
            x_reconstructed = self.decoder(z, training=True)
            
            # Compute VAE loss
            losses = self.vae_loss_fn(x, x_reconstructed, z_mean, z_log_var)
            total_loss = losses['total_loss']
        
        # Compute gradients
        gradients = tape.gradient(total_loss, self.autoencoder.trainable_variables)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.autoencoder.trainable_variables))
        
        return losses
    
    def _compute_loss(self, x: tf.Tensor, x_reconstructed: tf.Tensor, **kwargs) -> Dict[str, tf.Tensor]:
        """
        Compute VAE loss components.
        
        Args:
            x: Original input data
            x_reconstructed: Reconstructed data
            
        Returns:
            Dictionary containing loss components
        """
        # Get latent parameters
        z_mean, z_log_var = self.encoder_mean_var(x)
        
        # Compute VAE loss
        return self.vae_loss_fn(x, x_reconstructed, z_mean, z_log_var)
    
    def generate(self, num_samples: int = 1, latent_samples: tf.Tensor = None) -> tf.Tensor:
        """
        Generate new samples from the learned distribution.
        
        Args:
            num_samples: Number of samples to generate
            latent_samples: Optional latent samples (if None, sample from prior)
            
        Returns:
            Generated samples
        """
        if latent_samples is None:
            # Sample from standard normal prior
            latent_samples = tf.random.normal((num_samples, self.latent_dim))
        
        return self.decoder(latent_samples, training=False)
    
    def interpolate(self, x1: tf.Tensor, x2: tf.Tensor, num_steps: int = 10) -> tf.Tensor:
        """
        Interpolate between two points in latent space.
        
        Args:
            x1: First input point
            x2: Second input point
            num_steps: Number of interpolation steps
            
        Returns:
            Interpolated reconstructions
        """
        # Encode to latent space (use mean, not sample)
        z1_mean, _ = self.encoder_mean_var(x1)
        z2_mean, _ = self.encoder_mean_var(x2)
        
        # Create interpolation ratios
        ratios = tf.linspace(0.0, 1.0, num_steps)
        ratios = tf.reshape(ratios, [-1, 1])
        
        # Interpolate in latent space
        z_interpolated = []
        for ratio in ratios:
            z_interp = (1 - ratio) * z1_mean + ratio * z2_mean
            z_interpolated.append(z_interp)
        
        z_interpolated = tf.concat(z_interpolated, axis=0)
        
        # Decode interpolated latent vectors
        return self.decoder(z_interpolated, training=False)
    
    def get_latent_statistics(self, x: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        Get statistics of the latent distribution.
        
        Args:
            x: Input data
            
        Returns:
            Dictionary with latent statistics
        """
        z_mean, z_log_var = self.encoder_mean_var(x)
        z_std = tf.exp(0.5 * z_log_var)
        
        return {
            'mean': tf.reduce_mean(z_mean, axis=0),
            'std': tf.reduce_mean(z_std, axis=0),
            'mean_of_means': tf.reduce_mean(z_mean),
            'mean_of_stds': tf.reduce_mean(z_std),
            'latent_dim_variance': tf.reduce_var(z_mean, axis=0)
        }
    
    def get_config(self) -> Dict:
        """Get configuration dictionary."""
        config = super().get_config()
        config.update({
            'beta': self.beta,
            'reconstruction_loss': self.reconstruction_loss_type,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm
        })
        return config