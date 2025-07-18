"""
Denoising autoencoder implementation.
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Union
from .vanilla import VanillaAutoencoder


class DenoisingAutoencoder(VanillaAutoencoder):
    """
    Denoising autoencoder that learns to reconstruct clean data from corrupted input.
    
    Corrupts input data with various types of noise and trains the model to
    recover the original clean data, leading to more robust feature learning.
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
        noise_type: str = 'gaussian',
        noise_level: float = 0.1,
        dropout_rate: float = 0.0,
        use_batch_norm: bool = False,
        name: str = "denoising_autoencoder"
    ):
        """
        Initialize denoising autoencoder.
        
        Args:
            input_dim: Dimension of input data
            latent_dim: Dimension of latent space
            encoder_layers: List of hidden layer sizes for encoder
            decoder_layers: List of hidden layer sizes for decoder
            activation: Activation function for hidden layers
            output_activation: Activation function for output layer
            learning_rate: Learning rate for optimizer
            loss_type: Type of reconstruction loss
            noise_type: Type of noise to add ('gaussian', 'masking', 'salt_and_pepper')
            noise_level: Level/intensity of noise to add
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
            name: Name of the model
        """
        self.noise_type = noise_type
        self.noise_level = noise_level
        
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
    
    def add_noise(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        """
        Add noise to input data.
        
        Args:
            x: Clean input data
            training: Whether in training mode (noise only added during training)
            
        Returns:
            Noisy input data
        """
        if not training:
            return x
        
        if self.noise_type == 'gaussian':
            return self._add_gaussian_noise(x)
        elif self.noise_type == 'masking':
            return self._add_masking_noise(x)
        elif self.noise_type == 'salt_and_pepper':
            return self._add_salt_and_pepper_noise(x)
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")
    
    def _add_gaussian_noise(self, x: tf.Tensor) -> tf.Tensor:
        """Add Gaussian noise to input."""
        noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=self.noise_level)
        return x + noise
    
    def _add_masking_noise(self, x: tf.Tensor) -> tf.Tensor:
        """Add masking noise (randomly set values to 0)."""
        mask = tf.random.uniform(tf.shape(x)) > self.noise_level
        return x * tf.cast(mask, x.dtype)
    
    def _add_salt_and_pepper_noise(self, x: tf.Tensor) -> tf.Tensor:
        """Add salt and pepper noise."""
        # Generate random values
        random_vals = tf.random.uniform(tf.shape(x))
        
        # Salt noise (set to 1)
        salt_mask = random_vals < self.noise_level / 2
        
        # Pepper noise (set to 0)
        pepper_mask = random_vals > (1 - self.noise_level / 2)
        
        # Apply noise
        noisy_x = tf.where(salt_mask, tf.ones_like(x), x)
        noisy_x = tf.where(pepper_mask, tf.zeros_like(noisy_x), noisy_x)
        
        return noisy_x
    
    @tf.function
    def _train_step(self, x: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        Perform a single training step with noise corruption.
        
        Args:
            x: Clean input data batch
            
        Returns:
            Dictionary of loss values
        """
        with tf.GradientTape() as tape:
            # Add noise to input
            x_noisy = self.add_noise(x, training=True)
            
            # Forward pass with noisy input
            x_reconstructed = self.autoencoder(x_noisy, training=True)
            
            # Compute loss against clean target
            losses = self._compute_loss(x, x_reconstructed)
            total_loss = losses['total_loss']
        
        # Compute gradients
        gradients = tape.gradient(total_loss, self.autoencoder.trainable_variables)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.autoencoder.trainable_variables))
        
        return losses
    
    def fit(
        self,
        x_train: np.ndarray,
        batch_size: int = 32,
        epochs: int = 100,
        validation_data: Union[np.ndarray, None] = None,
        verbose: int = 1,
        callbacks: List = None
    ) -> Dict[str, List[float]]:
        """
        Train the denoising autoencoder.
        
        Args:
            x_train: Clean training data
            batch_size: Size of training batches
            epochs: Number of training epochs
            validation_data: Optional clean validation data
            verbose: Verbosity level
            callbacks: Optional list of callback functions
            
        Returns:
            Training history dictionary
        """
        # Convert to tensor dataset
        train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
        train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        if validation_data is not None:
            val_dataset = tf.data.Dataset.from_tensor_slices(validation_data)
            val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        # Training loop
        for epoch in range(epochs):
            epoch_losses = []
            
            # Training phase
            for batch in train_dataset:
                losses = self._train_step(batch)
                epoch_losses.append({k: float(v) for k, v in losses.items()})
            
            # Compute average losses for epoch
            avg_losses = {}
            for key in epoch_losses[0].keys():
                avg_losses[key] = np.mean([loss[key] for loss in epoch_losses])
            
            # Store in history
            self.history['loss'].append(avg_losses['total_loss'])
            self.history['reconstruction_loss'].append(avg_losses.get('reconstruction_loss', 0.0))
            self.history['regularization_loss'].append(avg_losses.get('regularization_loss', 0.0))
            
            # Validation phase (with clean data)
            if validation_data is not None:
                val_losses = []
                for val_batch in val_dataset:
                    # Add noise to validation input
                    val_noisy = self.add_noise(val_batch, training=False)  # No noise during validation
                    val_reconstructed = self.autoencoder(val_noisy, training=False)
                    val_loss_dict = self._compute_loss(val_batch, val_reconstructed)
                    val_losses.append({k: float(v) for k, v in val_loss_dict.items()})
                
                avg_val_loss = np.mean([loss['total_loss'] for loss in val_losses])
                
                if 'val_loss' not in self.history:
                    self.history['val_loss'] = []
                self.history['val_loss'].append(avg_val_loss)
            
            # Print progress
            if verbose > 0 and (epoch + 1) % verbose == 0:
                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"  Loss: {avg_losses['total_loss']:.6f}")
                if validation_data is not None:
                    print(f"  Val Loss: {avg_val_loss:.6f}")
                print()
            
            # Execute callbacks
            if callbacks:
                for callback in callbacks:
                    callback(epoch, avg_losses, self)
        
        return self.history
    
    def denoise(self, x_noisy: tf.Tensor) -> tf.Tensor:
        """
        Denoise input data.
        
        Args:
            x_noisy: Noisy input data
            
        Returns:
            Denoised reconstruction
        """
        return self.autoencoder(x_noisy, training=False)
    
    def test_denoising(self, x_clean: tf.Tensor, x_noisy: tf.Tensor = None) -> Dict[str, tf.Tensor]:
        """
        Test denoising performance on clean/noisy data pairs.
        
        Args:
            x_clean: Clean reference data
            x_noisy: Noisy input data (if None, noise will be added to x_clean)
            
        Returns:
            Dictionary with denoising metrics
        """
        if x_noisy is None:
            x_noisy = self.add_noise(x_clean, training=True)
        
        # Get reconstructions
        x_denoised = self.denoise(x_noisy)
        
        # Compute metrics
        mse_noisy = tf.reduce_mean(tf.square(x_clean - x_noisy))
        mse_denoised = tf.reduce_mean(tf.square(x_clean - x_denoised))
        
        # Signal-to-noise ratio improvement
        snr_improvement = 10 * tf.math.log(mse_noisy / mse_denoised) / tf.math.log(10.0)
        
        return {
            'mse_noisy': mse_noisy,
            'mse_denoised': mse_denoised,
            'snr_improvement_db': snr_improvement,
            'denoising_ratio': mse_noisy / mse_denoised
        }
    
    def get_config(self) -> Dict:
        """Get configuration dictionary."""
        config = super().get_config()
        config.update({
            'noise_type': self.noise_type,
            'noise_level': self.noise_level
        })
        return config