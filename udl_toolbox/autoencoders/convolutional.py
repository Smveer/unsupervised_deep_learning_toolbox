"""
Convolutional autoencoder implementation for image data.
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Union
from .base import BaseAutoencoder
from ..losses.reconstruction import MeanSquaredError, BinaryCrossentropy


class ConvolutionalAutoencoder(BaseAutoencoder):
    """
    Convolutional autoencoder for image data.
    
    Uses convolutional layers for encoding and transposed convolutions
    (deconvolutions) for decoding, preserving spatial structure.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int],  # (height, width, channels)
        latent_dim: int,
        encoder_filters: List[int] = None,
        decoder_filters: List[int] = None,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        strides: Union[int, Tuple[int, int]] = 2,
        activation: str = 'relu',
        output_activation: str = 'sigmoid',
        learning_rate: float = 0.001,
        loss_type: str = 'mse',
        dropout_rate: float = 0.0,
        use_batch_norm: bool = True,
        name: str = "convolutional_autoencoder"
    ):
        """
        Initialize convolutional autoencoder.
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            latent_dim: Dimension of latent space
            encoder_filters: List of filter counts for encoder conv layers
            decoder_filters: List of filter counts for decoder conv layers
            kernel_size: Size of convolutional kernels
            strides: Stride for convolutions
            activation: Activation function for hidden layers
            output_activation: Activation function for output layer
            learning_rate: Learning rate for optimizer
            loss_type: Type of reconstruction loss
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
            name: Name of the model
        """
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.strides = strides
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.loss_type = loss_type
        
        # Set default filter configurations
        if encoder_filters is None:
            encoder_filters = [32, 64, 128]
        if decoder_filters is None:
            decoder_filters = encoder_filters[::-1]
        
        self.encoder_filters = encoder_filters
        self.decoder_filters = decoder_filters
        
        # Calculate input dimension for base class
        input_dim = np.prod(input_shape)
        
        # Initialize loss function
        if loss_type == 'mse':
            self.loss_fn = MeanSquaredError()
        elif loss_type == 'binary_crossentropy':
            self.loss_fn = BinaryCrossentropy()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Calculate encoded shape after convolutions
        self.encoded_shape = self._calculate_encoded_shape()
        
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            encoder_layers=[],  # Not used for conv layers
            decoder_layers=[],  # Not used for conv layers
            activation=activation,
            output_activation=output_activation,
            learning_rate=learning_rate,
            name=name
        )
    
    def _calculate_encoded_shape(self) -> Tuple[int, int, int]:
        """Calculate the shape after all encoder convolutions."""
        h, w, c = self.input_shape
        
        for _ in self.encoder_filters:
            if isinstance(self.strides, int):
                h = h // self.strides
                w = w // self.strides
            else:
                h = h // self.strides[0]
                w = w // self.strides[1]
        
        # Final number of channels is the last filter count
        final_channels = self.encoder_filters[-1]
        
        return (h, w, final_channels)
    
    def _build_encoder(self) -> tf.keras.Model:
        """Build the convolutional encoder network."""
        inputs = tf.keras.Input(shape=self.input_shape, name=f"{self.name}_encoder_input")
        x = inputs
        
        # Convolutional layers
        for i, filters in enumerate(self.encoder_filters):
            x = tf.keras.layers.Conv2D(
                filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding='same',
                activation=self.activation,
                name=f"{self.name}_encoder_conv_{i}"
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
        
        # Flatten for dense layer
        x_flattened = tf.keras.layers.Flatten(name=f"{self.name}_flatten")(x)
        
        # Dense layer to latent space
        latent = tf.keras.layers.Dense(
            self.latent_dim,
            activation='linear',
            name=f"{self.name}_latent"
        )(x_flattened)
        
        return tf.keras.Model(inputs, latent, name=f"{self.name}_encoder")
    
    def _build_decoder(self) -> tf.keras.Model:
        """Build the convolutional decoder network."""
        inputs = tf.keras.Input(shape=(self.latent_dim,), name=f"{self.name}_decoder_input")
        
        # Dense layer to reshape to encoded shape
        encoded_size = np.prod(self.encoded_shape)
        x = tf.keras.layers.Dense(
            encoded_size,
            activation=self.activation,
            name=f"{self.name}_decoder_dense"
        )(inputs)
        
        # Reshape to encoded shape
        x = tf.keras.layers.Reshape(
            self.encoded_shape,
            name=f"{self.name}_reshape"
        )(x)
        
        # Transposed convolutional layers
        for i, filters in enumerate(self.decoder_filters[:-1]):  # All but last
            x = tf.keras.layers.Conv2DTranspose(
                filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding='same',
                activation=self.activation,
                name=f"{self.name}_decoder_conv_transpose_{i}"
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
        
        # Final output layer
        outputs = tf.keras.layers.Conv2DTranspose(
            self.input_shape[2],  # Output channels = input channels
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='same',
            activation=self.output_activation,
            name=f"{self.name}_output"
        )(x)
        
        return tf.keras.Model(inputs, outputs, name=f"{self.name}_decoder")
    
    def _compute_loss(self, x: tf.Tensor, x_reconstructed: tf.Tensor, **kwargs) -> Dict[str, tf.Tensor]:
        """
        Compute the reconstruction loss for convolutional autoencoder.
        
        Args:
            x: Original input images
            x_reconstructed: Reconstructed images
            
        Returns:
            Dictionary containing loss components
        """
        reconstruction_loss = self.loss_fn(x, x_reconstructed)
        
        return {
            'total_loss': reconstruction_loss,
            'reconstruction_loss': reconstruction_loss,
            'regularization_loss': tf.constant(0.0)
        }
    
    def encode_images(self, images: tf.Tensor) -> tf.Tensor:
        """
        Encode images to latent space.
        
        Args:
            images: Input images tensor
            
        Returns:
            Encoded latent representations
        """
        return self.encoder(images)
    
    def decode_images(self, latent: tf.Tensor) -> tf.Tensor:
        """
        Decode latent representations to images.
        
        Args:
            latent: Latent space tensor
            
        Returns:
            Reconstructed images
        """
        return self.decoder(latent)
    
    def reconstruct_images(self, images: tf.Tensor) -> tf.Tensor:
        """
        Reconstruct images through encoder-decoder pipeline.
        
        Args:
            images: Input images tensor
            
        Returns:
            Reconstructed images
        """
        return self.autoencoder(images)
    
    def get_feature_maps(self, images: tf.Tensor, layer_name: str = None) -> tf.Tensor:
        """
        Get intermediate feature maps from encoder.
        
        Args:
            images: Input images
            layer_name: Name of layer to extract features from (if None, returns all)
            
        Returns:
            Feature maps from specified layer
        """
        if layer_name is None:
            # Return all intermediate outputs
            intermediate_model = tf.keras.Model(
                inputs=self.encoder.input,
                outputs=[layer.output for layer in self.encoder.layers if 'conv' in layer.name]
            )
            return intermediate_model(images)
        else:
            # Return specific layer output
            layer = self.encoder.get_layer(layer_name)
            intermediate_model = tf.keras.Model(
                inputs=self.encoder.input,
                outputs=layer.output
            )
            return intermediate_model(images)
    
    def calculate_receptive_field(self) -> int:
        """
        Calculate the receptive field size of the encoder.
        
        Returns:
            Receptive field size in pixels
        """
        receptive_field = 1
        
        for _ in self.encoder_filters:
            if isinstance(self.kernel_size, int):
                kernel = self.kernel_size
                stride = self.strides if isinstance(self.strides, int) else self.strides[0]
            else:
                kernel = self.kernel_size[0]
                stride = self.strides[0] if isinstance(self.strides, tuple) else self.strides
            
            receptive_field = (receptive_field - 1) * stride + kernel
        
        return receptive_field
    
    def get_config(self) -> Dict:
        """Get configuration dictionary."""
        config = {
            'input_shape': self.input_shape,
            'latent_dim': self.latent_dim,
            'encoder_filters': self.encoder_filters,
            'decoder_filters': self.decoder_filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'activation': self.activation,
            'output_activation': self.output_activation,
            'learning_rate': self.learning_rate,
            'loss_type': self.loss_type,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
            'name': self.name
        }
        return config
    
    def summary(self) -> None:
        """Print model summaries with shape information."""
        print("=== ENCODER ===")
        self.encoder.summary()
        print(f"\nEncoded shape: {self.encoded_shape}")
        print(f"Receptive field: {self.calculate_receptive_field()} pixels")
        print("\n=== DECODER ===")
        self.decoder.summary()
        print("\n=== AUTOENCODER ===")
        self.autoencoder.summary()