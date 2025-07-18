"""
Data preprocessing utilities for autoencoders.
"""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict, Any, Union
import warnings


class DataPreprocessor:
    """
    Comprehensive data preprocessing utilities for autoencoder training.
    
    Provides methods for scaling, normalization, train/validation splitting,
    and data format conversion for various autoencoder types.
    """
    
    def __init__(self, scaling_method: str = 'standard'):
        """
        Initialize data preprocessor.
        
        Args:
            scaling_method: Method for scaling ('standard', 'minmax', 'robust', 'none')
        """
        self.scaling_method = scaling_method
        self.scaler = None
        self.is_fitted = False
        
        # Initialize scaler
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaling_method == 'robust':
            self.scaler = RobustScaler()
        elif scaling_method == 'none':
            self.scaler = None
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}")
    
    def prepare_data(
        self,
        data: np.ndarray,
        validation_split: float = 0.2,
        test_split: Optional[float] = None,
        random_state: int = 42,
        shuffle: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Prepare data for autoencoder training.
        
        Args:
            data: Input data array
            validation_split: Fraction of data to use for validation
            test_split: Optional fraction for test set
            random_state: Random seed for reproducibility
            shuffle: Whether to shuffle data before splitting
            
        Returns:
            Dictionary with train/val/test splits
        """
        data = np.array(data)
        
        # Handle different data shapes
        if len(data.shape) > 2:
            original_shape = data.shape
            data_flat = data.reshape(data.shape[0], -1)
        else:
            original_shape = None
            data_flat = data
        
        # Scale the data
        if self.scaler is not None:
            data_scaled = self.scaler.fit_transform(data_flat)
            self.is_fitted = True
        else:
            data_scaled = data_flat
        
        # Reshape back if needed
        if original_shape is not None:
            data_scaled = data_scaled.reshape(original_shape)
        
        # Split data
        if test_split is not None:
            # Three-way split
            X_temp, X_test = train_test_split(
                data_scaled, test_size=test_split, 
                random_state=random_state, shuffle=shuffle
            )
            X_train, X_val = train_test_split(
                X_temp, test_size=validation_split/(1-test_split),
                random_state=random_state, shuffle=shuffle
            )
            return {
                'train': X_train,
                'validation': X_val,
                'test': X_test
            }
        else:
            # Two-way split
            X_train, X_val = train_test_split(
                data_scaled, test_size=validation_split,
                random_state=random_state, shuffle=shuffle
            )
            return {
                'train': X_train,
                'validation': X_val
            }
    
    def preprocess_images(
        self,
        images: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None,
        normalize: bool = True,
        augment: bool = False
    ) -> np.ndarray:
        """
        Preprocess image data for convolutional autoencoders.
        
        Args:
            images: Image array of shape (n_samples, height, width, channels)
            target_size: Optional target size for resizing
            normalize: Whether to normalize pixel values to [0, 1]
            augment: Whether to apply data augmentation
            
        Returns:
            Preprocessed images
        """
        images = np.array(images)
        
        # Ensure 4D shape
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=-1)
        
        # Resize if requested
        if target_size is not None:
            try:
                from tensorflow.image import resize
                images = resize(images, target_size).numpy()
            except ImportError:
                warnings.warn("TensorFlow image resize not available, skipping resize")
        
        # Normalize pixel values
        if normalize:
            if images.dtype == np.uint8:
                images = images.astype(np.float32) / 255.0
            elif images.max() > 1.0:
                images = images / images.max()
        
        # Data augmentation (simple transformations)
        if augment:
            images = self._apply_image_augmentation(images)
        
        return images
    
    def _apply_image_augmentation(self, images: np.ndarray) -> np.ndarray:
        """Apply simple image augmentation."""
        # This is a placeholder for more sophisticated augmentation
        # In practice, you'd use tf.image or imgaug
        return images
    
    def preprocess_time_series(
        self,
        time_series: np.ndarray,
        window_size: int,
        step_size: int = 1,
        normalize_windows: bool = True
    ) -> np.ndarray:
        """
        Preprocess time series data for sequence autoencoders.
        
        Args:
            time_series: Time series data (n_series, n_timesteps, n_features)
            window_size: Size of sliding windows
            step_size: Step size for sliding windows
            normalize_windows: Whether to normalize each window
            
        Returns:
            Windowed time series data
        """
        if len(time_series.shape) == 1:
            time_series = time_series.reshape(1, -1, 1)
        elif len(time_series.shape) == 2:
            time_series = np.expand_dims(time_series, axis=-1)
        
        windows = []
        
        for series in time_series:
            series_windows = []
            for i in range(0, len(series) - window_size + 1, step_size):
                window = series[i:i + window_size]
                
                if normalize_windows:
                    # Z-score normalization per window
                    window_mean = np.mean(window, axis=0)
                    window_std = np.std(window, axis=0)
                    window_std = np.where(window_std == 0, 1, window_std)  # Avoid division by zero
                    window = (window - window_mean) / window_std
                
                series_windows.append(window)
            
            windows.extend(series_windows)
        
        return np.array(windows)
    
    def add_noise(
        self,
        data: np.ndarray,
        noise_type: str = 'gaussian',
        noise_level: float = 0.1,
        random_state: int = 42
    ) -> np.ndarray:
        """
        Add noise to data (useful for denoising autoencoders).
        
        Args:
            data: Input data
            noise_type: Type of noise ('gaussian', 'uniform', 'salt_pepper')
            noise_level: Level of noise to add
            random_state: Random seed
            
        Returns:
            Noisy data
        """
        np.random.seed(random_state)
        noisy_data = data.copy()
        
        if noise_type == 'gaussian':
            noise = np.random.normal(0, noise_level, data.shape)
            noisy_data = data + noise
        elif noise_type == 'uniform':
            noise = np.random.uniform(-noise_level, noise_level, data.shape)
            noisy_data = data + noise
        elif noise_type == 'salt_pepper':
            # Salt and pepper noise
            prob = noise_level
            random_matrix = np.random.random(data.shape)
            noisy_data[random_matrix < prob/2] = 0  # Pepper noise
            noisy_data[random_matrix > 1 - prob/2] = 1  # Salt noise
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        # Clip values to valid range
        if data.min() >= 0 and data.max() <= 1:
            noisy_data = np.clip(noisy_data, 0, 1)
        
        return noisy_data
    
    def create_corrupted_pairs(
        self,
        data: np.ndarray,
        corruption_ratio: float = 0.3,
        corruption_type: str = 'masking'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create corrupted input-target pairs for denoising autoencoders.
        
        Args:
            data: Clean input data
            corruption_ratio: Fraction of data to corrupt
            corruption_type: Type of corruption ('masking', 'noise', 'dropout')
            
        Returns:
            Tuple of (corrupted_inputs, clean_targets)
        """
        corrupted_data = data.copy()
        
        if corruption_type == 'masking':
            # Random masking
            mask = np.random.random(data.shape) < corruption_ratio
            corrupted_data[mask] = 0
        elif corruption_type == 'noise':
            # Add Gaussian noise
            corrupted_data = self.add_noise(data, 'gaussian', corruption_ratio)
        elif corruption_type == 'dropout':
            # Random dropout
            mask = np.random.random(data.shape) < corruption_ratio
            corrupted_data[mask] = 0
        else:
            raise ValueError(f"Unknown corruption type: {corruption_type}")
        
        return corrupted_data, data
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            data: New data to transform
            
        Returns:
            Transformed data
        """
        if not self.is_fitted and self.scaler is not None:
            raise ValueError("Preprocessor must be fitted before transform")
        
        data = np.array(data)
        
        # Handle different data shapes
        if len(data.shape) > 2:
            original_shape = data.shape
            data_flat = data.reshape(data.shape[0], -1)
        else:
            original_shape = None
            data_flat = data
        
        # Scale the data
        if self.scaler is not None:
            data_scaled = self.scaler.transform(data_flat)
        else:
            data_scaled = data_flat
        
        # Reshape back if needed
        if original_shape is not None:
            data_scaled = data_scaled.reshape(original_shape)
        
        return data_scaled
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform data back to original scale.
        
        Args:
            data: Transformed data
            
        Returns:
            Data in original scale
        """
        if not self.is_fitted and self.scaler is not None:
            raise ValueError("Preprocessor must be fitted before inverse transform")
        
        if self.scaler is None:
            return data
        
        data = np.array(data)
        
        # Handle different data shapes
        if len(data.shape) > 2:
            original_shape = data.shape
            data_flat = data.reshape(data.shape[0], -1)
        else:
            original_shape = None
            data_flat = data
        
        # Inverse transform
        data_original = self.scaler.inverse_transform(data_flat)
        
        # Reshape back if needed
        if original_shape is not None:
            data_original = data_original.reshape(original_shape)
        
        return data_original
    
    def get_preprocessing_stats(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Get statistics about the data for preprocessing analysis.
        
        Args:
            data: Input data
            
        Returns:
            Dictionary of data statistics
        """
        data = np.array(data)
        
        stats = {
            'shape': data.shape,
            'dtype': data.dtype,
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'n_samples': data.shape[0],
            'memory_usage_mb': data.nbytes / (1024 * 1024)
        }
        
        # Check for common data issues
        stats['has_nan'] = bool(np.any(np.isnan(data)))
        stats['has_inf'] = bool(np.any(np.isinf(data)))
        stats['is_normalized'] = bool(stats['min'] >= 0 and stats['max'] <= 1)
        stats['is_standardized'] = bool(abs(stats['mean']) < 0.1 and abs(stats['std'] - 1.0) < 0.1)
        
        return stats
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary."""
        return {
            'scaling_method': self.scaling_method,
            'is_fitted': self.is_fitted
        }