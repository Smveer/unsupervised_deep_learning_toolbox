"""
Comprehensive example demonstrating the UDL Toolbox autoencoder implementations.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, fetch_openml
import tensorflow as tf

# Import our autoencoder implementations
from udl_toolbox.autoencoders import (
    VanillaAutoencoder,
    SparseAutoencoder,
    DenoisingAutoencoder,
    VariationalAutoencoder,
    ConvolutionalAutoencoder
)

# Import utilities
from udl_toolbox.utils import DataPreprocessor, ModelSaver
from udl_toolbox.visualization import (
    LatentSpaceVisualizer,
    ReconstructionVisualizer,
    LossVisualizer
)
from udl_toolbox.projections import PCAProjection, LatentSpaceInterpolation


def create_synthetic_data(n_samples=1000, n_features=50, noise_level=0.1):
    """Create synthetic data for testing."""
    # Create clustered data
    X, y = make_blobs(n_samples=n_samples, centers=3, n_features=n_features, 
                      cluster_std=2.0, random_state=42)
    
    # Add noise
    X += np.random.normal(0, noise_level, X.shape)
    
    return X, y


def example_vanilla_autoencoder():
    """Demonstrate vanilla autoencoder."""
    print("=== Vanilla Autoencoder Example ===")
    
    # Create data
    X, y = create_synthetic_data(n_samples=500, n_features=20)
    
    # Preprocess data
    preprocessor = DataPreprocessor(scaling_method='standard')
    data_splits = preprocessor.prepare_data(X, validation_split=0.2)
    
    # Create autoencoder
    autoencoder = VanillaAutoencoder(
        input_dim=20,
        latent_dim=5,
        encoder_layers=[15, 10],
        decoder_layers=[10, 15],
        learning_rate=0.001
    )
    
    print("Model architecture:")
    autoencoder.summary()
    
    # Train
    history = autoencoder.fit(
        data_splits['train'],
        validation_data=data_splits['validation'],
        epochs=50,
        batch_size=32,
        verbose=10
    )
    
    # Visualize results
    vis = LatentSpaceVisualizer(autoencoder)
    vis.plot_2d_latent_space(data_splits['train'], labels=y[:len(data_splits['train'])], 
                             method='pca', title="Vanilla Autoencoder Latent Space")
    
    # Reconstruction visualization
    recon_vis = ReconstructionVisualizer(autoencoder)
    recon_vis.print_reconstruction_summary(data_splits['validation'])
    
    # Loss visualization
    loss_vis = LossVisualizer()
    loss_vis.plot_training_curves(history, title="Vanilla Autoencoder Training")
    
    return autoencoder, preprocessor


def example_sparse_autoencoder():
    """Demonstrate sparse autoencoder."""
    print("\n=== Sparse Autoencoder Example ===")
    
    # Create data
    X, y = create_synthetic_data(n_samples=500, n_features=20)
    
    # Preprocess data
    preprocessor = DataPreprocessor(scaling_method='minmax')  # Better for sparse AE
    data_splits = preprocessor.prepare_data(X, validation_split=0.2)
    
    # Create sparse autoencoder
    autoencoder = SparseAutoencoder(
        input_dim=20,
        latent_dim=10,
        encoder_layers=[15],
        decoder_layers=[15],
        sparsity_target=0.05,
        sparsity_weight=1.0,
        activation='sigmoid'  # Better for sparsity
    )
    
    # Train
    history = autoencoder.fit(
        data_splits['train'],
        validation_data=data_splits['validation'],
        epochs=50,
        batch_size=32,
        verbose=10
    )
    
    # Analyze sparsity
    autoencoder.visualize_sparsity(data_splits['train'])
    
    # Visualize latent space
    vis = LatentSpaceVisualizer(autoencoder)
    vis.plot_2d_latent_space(data_splits['train'], labels=y[:len(data_splits['train'])], 
                             method='pca', title="Sparse Autoencoder Latent Space")
    
    return autoencoder


def example_denoising_autoencoder():
    """Demonstrate denoising autoencoder."""
    print("\n=== Denoising Autoencoder Example ===")
    
    # Create data
    X, y = create_synthetic_data(n_samples=500, n_features=20)
    
    # Preprocess data
    preprocessor = DataPreprocessor(scaling_method='minmax')
    data_splits = preprocessor.prepare_data(X, validation_split=0.2)
    
    # Create denoising autoencoder
    autoencoder = DenoisingAutoencoder(
        input_dim=20,
        latent_dim=8,
        encoder_layers=[15, 12],
        decoder_layers=[12, 15],
        noise_type='gaussian',
        noise_level=0.2
    )
    
    # Train
    history = autoencoder.fit(
        data_splits['train'],
        validation_data=data_splits['validation'],
        epochs=50,
        batch_size=32,
        verbose=10
    )
    
    # Test denoising capability
    clean_test = data_splits['validation'][:10]
    noisy_test = autoencoder.add_noise(clean_test, training=True)
    denoising_metrics = autoencoder.test_denoising(clean_test, noisy_test)
    
    print(f"Denoising performance:")
    for metric, value in denoising_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return autoencoder


def example_variational_autoencoder():
    """Demonstrate variational autoencoder."""
    print("\n=== Variational Autoencoder Example ===")
    
    # Create data
    X, y = create_synthetic_data(n_samples=500, n_features=20)
    
    # Preprocess data
    preprocessor = DataPreprocessor(scaling_method='standard')
    data_splits = preprocessor.prepare_data(X, validation_split=0.2)
    
    # Create VAE
    autoencoder = VariationalAutoencoder(
        input_dim=20,
        latent_dim=5,
        encoder_layers=[15, 10],
        decoder_layers=[10, 15],
        beta=1.0,
        reconstruction_loss='mse'
    )
    
    # Train
    history = autoencoder.fit(
        data_splits['train'],
        validation_data=data_splits['validation'],
        epochs=50,
        batch_size=32,
        verbose=10
    )
    
    # Generate new samples
    print("Generating new samples...")
    generated_samples = autoencoder.generate(num_samples=5)
    print(f"Generated samples shape: {generated_samples.shape}")
    
    # Analyze latent statistics
    latent_stats = autoencoder.get_latent_statistics(data_splits['train'])
    print("Latent space statistics:")
    for stat, value in latent_stats.items():
        if hasattr(value, 'shape'):
            print(f"  {stat}: shape {value.shape}, mean {np.mean(value):.4f}")
        else:
            print(f"  {stat}: {value:.4f}")
    
    # Interpolation example
    if len(data_splits['train']) >= 2:
        interpolator = LatentSpaceInterpolation(autoencoder)
        z_interp, x_interp = interpolator.interpolate_data_points(
            data_splits['train'][0], data_splits['train'][1], num_steps=5
        )
        print(f"Interpolation result shape: {x_interp.shape}")
    
    return autoencoder


def example_image_autoencoder():
    """Demonstrate convolutional autoencoder with image data."""
    print("\n=== Convolutional Autoencoder Example ===")
    
    # Create simple synthetic image data
    def create_synthetic_images(n_samples=200, img_size=(28, 28)):
        """Create synthetic image data with geometric patterns."""
        images = np.zeros((n_samples, img_size[0], img_size[1], 1))
        
        for i in range(n_samples):
            # Create random geometric patterns
            img = np.zeros(img_size)
            
            # Add random rectangles
            for _ in range(np.random.randint(1, 4)):
                x1, y1 = np.random.randint(0, img_size[1]//2, 2)
                x2, y2 = x1 + np.random.randint(5, img_size[1]//4), y1 + np.random.randint(5, img_size[0]//4)
                x2, y2 = min(x2, img_size[1]), min(y2, img_size[0])
                img[y1:y2, x1:x2] = np.random.random()
            
            images[i, :, :, 0] = img
        
        return images
    
    # Create synthetic image data
    images = create_synthetic_images(n_samples=200)
    
    # Preprocess images
    preprocessor = DataPreprocessor()
    images_processed = preprocessor.preprocess_images(images, normalize=True)
    data_splits = preprocessor.prepare_data(images_processed, validation_split=0.2)
    
    # Create convolutional autoencoder
    autoencoder = ConvolutionalAutoencoder(
        input_shape=(28, 28, 1),
        latent_dim=10,
        encoder_filters=[16, 32],
        decoder_filters=[32, 16],
        kernel_size=3,
        strides=2
    )
    
    print("Convolutional autoencoder architecture:")
    autoencoder.summary()
    
    # Train
    history = autoencoder.fit(
        data_splits['train'],
        validation_data=data_splits['validation'],
        epochs=20,
        batch_size=16,
        verbose=5
    )
    
    # Visualize reconstructions
    recon_vis = ReconstructionVisualizer(autoencoder)
    recon_vis.plot_reconstruction_comparison(
        data_splits['validation'][:5],
        num_samples=5,
        title="Convolutional Autoencoder Reconstructions"
    )
    
    return autoencoder


def example_model_saving_loading():
    """Demonstrate model saving and loading."""
    print("\n=== Model Saving and Loading Example ===")
    
    # Create and train a simple autoencoder
    X, _ = create_synthetic_data(n_samples=200, n_features=10)
    preprocessor = DataPreprocessor()
    data_splits = preprocessor.prepare_data(X, validation_split=0.2)
    
    autoencoder = VanillaAutoencoder(
        input_dim=10,
        latent_dim=3,
        encoder_layers=[8, 5],
        decoder_layers=[5, 8]
    )
    
    # Train briefly
    autoencoder.fit(data_splits['train'], epochs=10, verbose=0)
    
    # Save model
    saver = ModelSaver()
    save_path = "/tmp/test_autoencoder"
    saver.save_autoencoder(autoencoder, save_path, save_format='weights')
    
    # Create new instance and load weights
    new_autoencoder = VanillaAutoencoder(
        input_dim=10,
        latent_dim=3,
        encoder_layers=[8, 5],
        decoder_layers=[5, 8]
    )
    
    # Load weights
    new_autoencoder.load_weights(f"{save_path}_autoencoder_weights.h5")
    
    # Verify they produce same results
    original_output = autoencoder.reconstruct(data_splits['validation'][:5])
    loaded_output = new_autoencoder.reconstruct(data_splits['validation'][:5])
    
    mse = np.mean((original_output - loaded_output) ** 2)
    print(f"MSE between original and loaded model outputs: {mse:.10f}")
    
    if mse < 1e-6:
        print("✓ Model loading successful!")
    else:
        print("✗ Model loading failed!")


def example_projection_analysis():
    """Demonstrate data projection utilities."""
    print("\n=== Data Projection Analysis Example ===")
    
    # Create high-dimensional data
    X, y = create_synthetic_data(n_samples=300, n_features=50)
    
    # PCA projection
    pca = PCAProjection(n_components=2)
    X_pca = pca.fit_transform(X)
    
    print(f"PCA explained variance ratio: {pca.get_explained_variance_ratio()}")
    print(f"Cumulative variance: {pca.get_cumulative_variance_ratio()}")
    
    # Plot PCA results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('PCA Projection')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(pca.get_explained_variance_ratio()) + 1), 
             pca.get_cumulative_variance_ratio(), 'o-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """Run all examples."""
    print("UDL Toolbox - Comprehensive Autoencoder Examples")
    print("=" * 50)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Run examples
    try:
        # Basic autoencoders
        vanilla_ae, preprocessor = example_vanilla_autoencoder()
        sparse_ae = example_sparse_autoencoder()
        denoising_ae = example_denoising_autoencoder()
        vae = example_variational_autoencoder()
        
        # Convolutional autoencoder
        conv_ae = example_image_autoencoder()
        
        # Utility examples
        example_model_saving_loading()
        example_projection_analysis()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully! ✓")
        print("The UDL Toolbox is ready for use.")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()