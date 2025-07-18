"""
Base autoencoder class providing common functionality for all autoencoder variants.
"""

import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Callable, Any
import warnings


class BaseAutoencoder(ABC):
    """
    Abstract base class for all autoencoder implementations.
    
    This class provides common functionality including:
    - Model building framework
    - Training loop with custom gradient computation
    - Encoding and decoding methods
    - Model saving and loading
    - Visualization hooks
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder_layers: List[int],
        decoder_layers: List[int],
        activation: str = 'relu',
        output_activation: str = 'sigmoid',
        learning_rate: float = 0.001,
        name: str = "autoencoder"
    ):
        """
        Initialize the base autoencoder.
        
        Args:
            input_dim: Dimension of input data
            latent_dim: Dimension of latent space
            encoder_layers: List of hidden layer sizes for encoder
            decoder_layers: List of hidden layer sizes for decoder
            activation: Activation function for hidden layers
            output_activation: Activation function for output layer
            learning_rate: Learning rate for optimizer
            name: Name of the model
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.activation = activation
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        self.name = name
        
        # Initialize components
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.optimizer = None
        self.history = {
            'loss': [],
            'reconstruction_loss': [],
            'regularization_loss': []
        }
        
        # Build the model
        self._build_model()
        
    @abstractmethod
    def _build_encoder(self) -> tf.keras.Model:
        """Build the encoder network."""
        pass
    
    @abstractmethod
    def _build_decoder(self) -> tf.keras.Model:
        """Build the decoder network."""
        pass
    
    @abstractmethod
    def _compute_loss(self, x: tf.Tensor, x_reconstructed: tf.Tensor, **kwargs) -> Dict[str, tf.Tensor]:
        """Compute the total loss and individual loss components."""
        pass
    
    def _build_model(self):
        """Build the complete autoencoder model."""
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        
        # Build the full autoencoder
        inputs = tf.keras.Input(shape=(self.input_dim,), name=f"{self.name}_input")
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        
        self.autoencoder = tf.keras.Model(inputs, decoded, name=self.name)
        
        # Initialize optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
    def encode(self, x: tf.Tensor) -> tf.Tensor:
        """
        Encode input data to latent space.
        
        Args:
            x: Input data tensor
            
        Returns:
            Encoded representation in latent space
        """
        return self.encoder(x)
    
    def decode(self, z: tf.Tensor) -> tf.Tensor:
        """
        Decode latent representation to data space.
        
        Args:
            z: Latent space tensor
            
        Returns:
            Reconstructed data
        """
        return self.decoder(z)
    
    def reconstruct(self, x: tf.Tensor) -> tf.Tensor:
        """
        Reconstruct input data through encoder-decoder pipeline.
        
        Args:
            x: Input data tensor
            
        Returns:
            Reconstructed data
        """
        return self.autoencoder(x)
    
    @tf.function
    def _train_step(self, x: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        Perform a single training step.
        
        Args:
            x: Input data batch
            
        Returns:
            Dictionary of loss values
        """
        with tf.GradientTape() as tape:
            # Forward pass
            x_reconstructed = self.autoencoder(x, training=True)
            
            # Compute losses
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
        validation_data: Optional[np.ndarray] = None,
        verbose: int = 1,
        callbacks: Optional[List[Callable]] = None
    ) -> Dict[str, List[float]]:
        """
        Train the autoencoder.
        
        Args:
            x_train: Training data
            batch_size: Size of training batches
            epochs: Number of training epochs
            validation_data: Optional validation data
            verbose: Verbosity level (0, 1, or 2)
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
            
            # Validation phase
            if validation_data is not None:
                val_losses = []
                for val_batch in val_dataset:
                    val_reconstructed = self.autoencoder(val_batch, training=False)
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
                if 'reconstruction_loss' in avg_losses:
                    print(f"  Reconstruction Loss: {avg_losses['reconstruction_loss']:.6f}")
                if 'regularization_loss' in avg_losses:
                    print(f"  Regularization Loss: {avg_losses['regularization_loss']:.6f}")
                if validation_data is not None:
                    print(f"  Val Loss: {avg_val_loss:.6f}")
                print()
            
            # Execute callbacks
            if callbacks:
                for callback in callbacks:
                    callback(epoch, avg_losses, self)
        
        return self.history
    
    def save(self, filepath: str) -> None:
        """
        Save the autoencoder model.
        
        Args:
            filepath: Path to save the model
        """
        self.autoencoder.save(filepath)
    
    def load(self, filepath: str) -> None:
        """
        Load a saved autoencoder model.
        
        Args:
            filepath: Path to the saved model
        """
        self.autoencoder = tf.keras.models.load_model(filepath)
        # Extract encoder and decoder from loaded model
        # This is a simplified approach - in practice, you might want to save/load
        # encoder and decoder separately for more flexibility
        warnings.warn(
            "Loading complete models may not preserve encoder/decoder separation. "
            "Consider using save_weights/load_weights for better compatibility."
        )
    
    def save_weights(self, filepath: str) -> None:
        """
        Save only the model weights.
        
        Args:
            filepath: Path to save the weights
        """
        self.autoencoder.save_weights(filepath)
    
    def load_weights(self, filepath: str) -> None:
        """
        Load saved model weights.
        
        Args:
            filepath: Path to the saved weights
        """
        self.autoencoder.load_weights(filepath)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration dictionary for this autoencoder.
        
        Returns:
            Configuration dictionary
        """
        return {
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'encoder_layers': self.encoder_layers,
            'decoder_layers': self.decoder_layers,
            'activation': self.activation,
            'output_activation': self.output_activation,
            'learning_rate': self.learning_rate,
            'name': self.name
        }
    
    def summary(self) -> None:
        """Print model summaries."""
        print("=== ENCODER ===")
        self.encoder.summary()
        print("\n=== DECODER ===")
        self.decoder.summary()
        print("\n=== AUTOENCODER ===")
        self.autoencoder.summary()