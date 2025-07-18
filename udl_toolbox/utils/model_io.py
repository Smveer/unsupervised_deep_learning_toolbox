"""
Model saving and loading utilities.
"""

import os
import json
import pickle
import tensorflow as tf
import numpy as np
from typing import Dict, Any, Optional
import warnings


class ModelSaver:
    """
    Comprehensive model saving and loading utilities for autoencoders.
    
    Provides methods for saving complete models, weights only, configurations,
    and custom serialization for different autoencoder types.
    """
    
    def __init__(self):
        """Initialize model saver."""
        pass
    
    def save_autoencoder(
        self,
        autoencoder,
        save_path: str,
        save_format: str = 'complete',
        include_optimizer: bool = True,
        save_config: bool = True
    ) -> None:
        """
        Save autoencoder model with various options.
        
        Args:
            autoencoder: Autoencoder instance to save
            save_path: Base path for saving (without extension)
            save_format: Format for saving ('complete', 'weights', 'savedmodel')
            include_optimizer: Whether to save optimizer state
            save_config: Whether to save model configuration
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        if save_format == 'complete':
            # Save the complete Keras model
            autoencoder.autoencoder.save(f"{save_path}_autoencoder.h5")
            autoencoder.encoder.save(f"{save_path}_encoder.h5")
            autoencoder.decoder.save(f"{save_path}_decoder.h5")
            
        elif save_format == 'weights':
            # Save only weights
            autoencoder.autoencoder.save_weights(f"{save_path}_autoencoder_weights.h5")
            autoencoder.encoder.save_weights(f"{save_path}_encoder_weights.h5")
            autoencoder.decoder.save_weights(f"{save_path}_decoder_weights.h5")
            
        elif save_format == 'savedmodel':
            # Save in TensorFlow SavedModel format
            tf.saved_model.save(autoencoder.autoencoder, f"{save_path}_autoencoder")
            tf.saved_model.save(autoencoder.encoder, f"{save_path}_encoder")
            tf.saved_model.save(autoencoder.decoder, f"{save_path}_decoder")
            
        else:
            raise ValueError(f"Unknown save format: {save_format}")
        
        # Save configuration
        if save_config:
            config = autoencoder.get_config()
            config['model_type'] = type(autoencoder).__name__
            
            with open(f"{save_path}_config.json", 'w') as f:
                json.dump(config, f, indent=2)
        
        # Save training history if available
        if hasattr(autoencoder, 'history') and autoencoder.history:
            with open(f"{save_path}_history.json", 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                history_serializable = {}
                for key, value in autoencoder.history.items():
                    if isinstance(value, np.ndarray):
                        history_serializable[key] = value.tolist()
                    elif isinstance(value, list):
                        history_serializable[key] = value
                    else:
                        history_serializable[key] = str(value)
                
                json.dump(history_serializable, f, indent=2)
        
        print(f"Autoencoder saved to {save_path} with format '{save_format}'")
    
    def load_autoencoder(
        self,
        save_path: str,
        autoencoder_class,
        load_format: str = 'complete'
    ):
        """
        Load autoencoder model.
        
        Args:
            save_path: Base path where model was saved
            autoencoder_class: Class of the autoencoder to instantiate
            load_format: Format to load from ('complete', 'weights', 'savedmodel')
            
        Returns:
            Loaded autoencoder instance
        """
        # Load configuration
        config_path = f"{save_path}_config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Remove non-constructor parameters
            model_type = config.pop('model_type', None)
            
            # Create autoencoder instance
            autoencoder = autoencoder_class(**config)
            
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load model weights/architecture
        if load_format == 'complete':
            try:
                autoencoder.autoencoder = tf.keras.models.load_model(f"{save_path}_autoencoder.h5")
                autoencoder.encoder = tf.keras.models.load_model(f"{save_path}_encoder.h5")
                autoencoder.decoder = tf.keras.models.load_model(f"{save_path}_decoder.h5")
            except Exception as e:
                warnings.warn(f"Failed to load complete models: {e}. Trying weights-only loading.")
                load_format = 'weights'
        
        if load_format == 'weights':
            # Load weights only
            autoencoder.autoencoder.load_weights(f"{save_path}_autoencoder_weights.h5")
            autoencoder.encoder.load_weights(f"{save_path}_encoder_weights.h5")
            autoencoder.decoder.load_weights(f"{save_path}_decoder_weights.h5")
            
        elif load_format == 'savedmodel':
            # Load from SavedModel format
            autoencoder.autoencoder = tf.saved_model.load(f"{save_path}_autoencoder")
            autoencoder.encoder = tf.saved_model.load(f"{save_path}_encoder")
            autoencoder.decoder = tf.saved_model.load(f"{save_path}_decoder")
        
        # Load training history if available
        history_path = f"{save_path}_history.json"
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
                autoencoder.history = history
        
        print(f"Autoencoder loaded from {save_path}")
        return autoencoder
    
    def save_model_checkpoint(
        self,
        autoencoder,
        checkpoint_path: str,
        epoch: int,
        loss: float,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save model checkpoint during training.
        
        Args:
            autoencoder: Autoencoder instance
            checkpoint_path: Path for checkpoint
            epoch: Current epoch number
            loss: Current loss value
            additional_info: Additional information to save
        """
        checkpoint_dir = os.path.dirname(checkpoint_path)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save weights
        autoencoder.autoencoder.save_weights(f"{checkpoint_path}_epoch_{epoch}_weights.h5")
        
        # Save checkpoint metadata
        checkpoint_info = {
            'epoch': epoch,
            'loss': float(loss),
            'model_config': autoencoder.get_config(),
            'timestamp': str(tf.timestamp()),
        }
        
        if additional_info:
            checkpoint_info.update(additional_info)
        
        with open(f"{checkpoint_path}_epoch_{epoch}_info.json", 'w') as f:
            json.dump(checkpoint_info, f, indent=2)
        
        print(f"Checkpoint saved at epoch {epoch} with loss {loss:.6f}")
    
    def load_model_checkpoint(
        self,
        checkpoint_path: str,
        epoch: int,
        autoencoder_class,
        autoencoder_instance = None
    ):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Base checkpoint path
            epoch: Epoch number to load
            autoencoder_class: Autoencoder class
            autoencoder_instance: Existing instance to load weights into
            
        Returns:
            Autoencoder instance with loaded weights
        """
        # Load checkpoint info
        info_path = f"{checkpoint_path}_epoch_{epoch}_info.json"
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"Checkpoint info not found: {info_path}")
        
        with open(info_path, 'r') as f:
            checkpoint_info = json.load(f)
        
        # Create or use existing autoencoder instance
        if autoencoder_instance is None:
            config = checkpoint_info['model_config']
            autoencoder = autoencoder_class(**config)
        else:
            autoencoder = autoencoder_instance
        
        # Load weights
        weights_path = f"{checkpoint_path}_epoch_{epoch}_weights.h5"
        autoencoder.autoencoder.load_weights(weights_path)
        
        print(f"Checkpoint loaded from epoch {epoch}")
        return autoencoder, checkpoint_info
    
    def save_preprocessor(
        self,
        preprocessor,
        save_path: str
    ) -> None:
        """
        Save data preprocessor.
        
        Args:
            preprocessor: DataPreprocessor instance
            save_path: Path to save preprocessor
        """
        with open(save_path, 'wb') as f:
            pickle.dump(preprocessor, f)
        print(f"Preprocessor saved to {save_path}")
    
    def load_preprocessor(self, save_path: str):
        """
        Load data preprocessor.
        
        Args:
            save_path: Path to load preprocessor from
            
        Returns:
            Loaded preprocessor instance
        """
        with open(save_path, 'rb') as f:
            preprocessor = pickle.load(f)
        print(f"Preprocessor loaded from {save_path}")
        return preprocessor
    
    def save_training_results(
        self,
        save_path: str,
        autoencoder,
        train_data: np.ndarray,
        validation_data: Optional[np.ndarray] = None,
        additional_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save comprehensive training results.
        
        Args:
            save_path: Base path for saving results
            autoencoder: Trained autoencoder
            train_data: Training data
            validation_data: Optional validation data
            additional_metrics: Additional metrics to save
        """
        # Save the model
        self.save_autoencoder(autoencoder, save_path, save_format='complete')
        
        # Compute and save reconstruction metrics
        train_reconstructions = autoencoder.reconstruct(train_data).numpy()
        train_mse = np.mean((train_data - train_reconstructions) ** 2)
        
        results = {
            'train_mse': float(train_mse),
            'train_samples': len(train_data),
            'model_type': type(autoencoder).__name__,
            'latent_dim': autoencoder.latent_dim,
        }
        
        if validation_data is not None:
            val_reconstructions = autoencoder.reconstruct(validation_data).numpy()
            val_mse = np.mean((validation_data - val_reconstructions) ** 2)
            results['val_mse'] = float(val_mse)
            results['val_samples'] = len(validation_data)
        
        if additional_metrics:
            results.update(additional_metrics)
        
        # Save results
        with open(f"{save_path}_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Training results saved to {save_path}_results.json")
    
    def create_model_archive(
        self,
        autoencoder,
        archive_path: str,
        include_data_samples: bool = False,
        data_samples: Optional[np.ndarray] = None
    ) -> None:
        """
        Create a complete archive of the model and related files.
        
        Args:
            autoencoder: Autoencoder instance
            archive_path: Path for the archive
            include_data_samples: Whether to include sample data
            data_samples: Sample data to include
        """
        import zipfile
        import tempfile
        import shutil
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = os.path.join(temp_dir, 'model')
            
            # Save all model components
            self.save_autoencoder(autoencoder, base_path, save_format='complete')
            
            # Save sample data if requested
            if include_data_samples and data_samples is not None:
                np.save(os.path.join(temp_dir, 'data_samples.npy'), data_samples)
                
                # Generate and save sample reconstructions
                reconstructions = autoencoder.reconstruct(data_samples).numpy()
                np.save(os.path.join(temp_dir, 'sample_reconstructions.npy'), reconstructions)
            
            # Create archive
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, temp_dir)
                        zipf.write(file_path, arcname)
        
        print(f"Model archive created: {archive_path}")
    
    def extract_model_archive(
        self,
        archive_path: str,
        extract_path: str,
        autoencoder_class
    ):
        """
        Extract and load model from archive.
        
        Args:
            archive_path: Path to the archive
            extract_path: Path to extract to
            autoencoder_class: Autoencoder class to instantiate
            
        Returns:
            Loaded autoencoder instance
        """
        import zipfile
        
        # Extract archive
        with zipfile.ZipFile(archive_path, 'r') as zipf:
            zipf.extractall(extract_path)
        
        # Load model
        model_path = os.path.join(extract_path, 'model')
        autoencoder = self.load_autoencoder(model_path, autoencoder_class)
        
        print(f"Model extracted and loaded from {archive_path}")
        return autoencoder
    
    def list_saved_models(self, directory: str) -> list:
        """
        List all saved models in a directory.
        
        Args:
            directory: Directory to search
            
        Returns:
            List of model information dictionaries
        """
        models = []
        
        for file in os.listdir(directory):
            if file.endswith('_config.json'):
                config_path = os.path.join(directory, file)
                base_name = file.replace('_config.json', '')
                
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    model_info = {
                        'name': base_name,
                        'type': config.get('model_type', 'Unknown'),
                        'latent_dim': config.get('latent_dim', 'Unknown'),
                        'config_path': config_path
                    }
                    
                    # Check for history file
                    history_path = os.path.join(directory, f"{base_name}_history.json")
                    if os.path.exists(history_path):
                        model_info['has_history'] = True
                    
                    models.append(model_info)
                    
                except Exception as e:
                    print(f"Error reading config for {base_name}: {e}")
        
        return models