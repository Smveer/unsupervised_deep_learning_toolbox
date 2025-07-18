"""
Basic tests for the UDL Toolbox autoencoder implementations.
"""

import numpy as np
import tensorflow as tf
import unittest
import tempfile
import os

# Import the modules to test
from udl_toolbox.autoencoders import (
    VanillaAutoencoder,
    SparseAutoencoder,
    DenoisingAutoencoder,
    VariationalAutoencoder,
    ConvolutionalAutoencoder
)
from udl_toolbox.losses import (
    MeanSquaredError,
    BinaryCrossentropy,
    KLDivergence,
    SparsityRegularization,
    VAELoss
)
from udl_toolbox.projections import PCAProjection, LatentSpaceInterpolation
from udl_toolbox.utils import DataPreprocessor, ModelSaver


class TestLossFunctions(unittest.TestCase):
    """Test custom loss functions."""
    
    def setUp(self):
        self.y_true = tf.constant([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        self.y_pred = tf.constant([[0.9, 0.1, 0.8], [0.2, 0.8, 0.1]])
    
    def test_mse_loss(self):
        """Test MSE loss function."""
        mse = MeanSquaredError()
        loss = mse(self.y_true, self.y_pred)
        
        # Manually compute expected MSE
        expected = tf.reduce_mean(tf.square(self.y_true - self.y_pred))
        
        self.assertAlmostEqual(float(loss), float(expected), places=5)
    
    def test_binary_crossentropy_loss(self):
        """Test binary crossentropy loss function."""
        bce = BinaryCrossentropy()
        loss = bce(self.y_true, self.y_pred)
        
        # Should be a positive scalar
        self.assertGreater(float(loss), 0)
        self.assertEqual(loss.shape, ())
    
    def test_kl_divergence(self):
        """Test KL divergence loss."""
        kl = KLDivergence()
        mu = tf.constant([[0.1, -0.2, 0.3], [0.0, 0.1, -0.1]])
        log_var = tf.constant([[-1.0, -0.5, -1.5], [-0.8, -1.2, -0.9]])
        
        loss = kl.standard_normal_kl(mu, log_var)
        
        # KL divergence should be non-negative
        self.assertGreaterEqual(float(loss), 0)
    
    def test_sparsity_regularization(self):
        """Test sparsity regularization."""
        sparsity = SparsityRegularization(sparsity_target=0.05, sparsity_weight=1.0)
        activations = tf.constant([[0.1, 0.0, 0.05, 0.8], [0.0, 0.02, 0.0, 0.1]])
        
        loss = sparsity(activations)
        
        # Should be a positive scalar
        self.assertGreater(float(loss), 0)


class TestAutoencoders(unittest.TestCase):
    """Test autoencoder implementations."""
    
    def setUp(self):
        # Create simple test data
        np.random.seed(42)
        tf.random.set_seed(42)
        self.data = np.random.random((50, 10)).astype(np.float32)
        self.image_data = np.random.random((20, 16, 16, 1)).astype(np.float32)
    
    def test_vanilla_autoencoder(self):
        """Test vanilla autoencoder."""
        ae = VanillaAutoencoder(
            input_dim=10,
            latent_dim=5,
            encoder_layers=[8],
            decoder_layers=[8]
        )
        
        # Test encoding
        encoded = ae.encode(self.data)
        self.assertEqual(encoded.shape, (50, 5))
        
        # Test decoding
        decoded = ae.decode(encoded)
        self.assertEqual(decoded.shape, (50, 10))
        
        # Test reconstruction
        reconstructed = ae.reconstruct(self.data)
        self.assertEqual(reconstructed.shape, self.data.shape)
        
        # Test training (brief)
        history = ae.fit(self.data, epochs=2, verbose=0)
        self.assertIn('loss', history)
        self.assertEqual(len(history['loss']), 2)
    
    def test_sparse_autoencoder(self):
        """Test sparse autoencoder."""
        ae = SparseAutoencoder(
            input_dim=10,
            latent_dim=8,
            encoder_layers=[],
            decoder_layers=[],
            sparsity_target=0.05,
            sparsity_weight=1.0
        )
        
        # Test basic functionality
        encoded = ae.encode(self.data)
        self.assertEqual(encoded.shape, (50, 8))
        
        # Test sparsity statistics
        stats = ae.get_sparsity_statistics(self.data)
        self.assertIn('sparsity_ratio', stats)
        self.assertIn('active_neurons', stats)
    
    def test_denoising_autoencoder(self):
        """Test denoising autoencoder."""
        ae = DenoisingAutoencoder(
            input_dim=10,
            latent_dim=5,
            encoder_layers=[8],
            decoder_layers=[8],
            noise_type='gaussian',
            noise_level=0.1
        )
        
        # Test noise addition
        noisy_data = ae.add_noise(self.data, training=True)
        self.assertEqual(noisy_data.shape, self.data.shape)
        self.assertNotEqual(np.sum(np.abs(noisy_data - self.data)), 0)
        
        # Test denoising metrics
        metrics = ae.test_denoising(self.data[:5])
        self.assertIn('mse_noisy', metrics)
        self.assertIn('mse_denoised', metrics)
    
    def test_variational_autoencoder(self):
        """Test variational autoencoder."""
        vae = VariationalAutoencoder(
            input_dim=10,
            latent_dim=5,
            encoder_layers=[8],
            decoder_layers=[8],
            beta=1.0
        )
        
        # Test encoding with distribution
        z_mean, z_log_var, z = vae.encode(self.data, return_distribution=True)
        self.assertEqual(z_mean.shape, (50, 5))
        self.assertEqual(z_log_var.shape, (50, 5))
        self.assertEqual(z.shape, (50, 5))
        
        # Test generation
        generated = vae.generate(num_samples=5)
        self.assertEqual(generated.shape, (5, 10))
        
        # Test interpolation
        if len(self.data) >= 2:
            interpolated = vae.interpolate(
                self.data[0:1], self.data[1:2], num_steps=3
            )
            self.assertEqual(interpolated.shape, (3, 10))
    
    def test_convolutional_autoencoder(self):
        """Test convolutional autoencoder."""
        cae = ConvolutionalAutoencoder(
            input_shape=(16, 16, 1),
            latent_dim=8,
            encoder_filters=[8, 16],
            decoder_filters=[16, 8],
            kernel_size=3,
            strides=2
        )
        
        # Test with image data
        encoded = cae.encode_images(self.image_data)
        self.assertEqual(encoded.shape, (20, 8))
        
        decoded = cae.decode_images(encoded)
        self.assertEqual(decoded.shape, self.image_data.shape)
        
        # Test receptive field calculation
        rf = cae.calculate_receptive_field()
        self.assertIsInstance(rf, int)
        self.assertGreater(rf, 0)


class TestProjections(unittest.TestCase):
    """Test projection utilities."""
    
    def setUp(self):
        np.random.seed(42)
        self.data = np.random.random((100, 20))
    
    def test_pca_projection(self):
        """Test PCA projection."""
        pca = PCAProjection(n_components=5)
        
        # Test fit and transform
        transformed = pca.fit_transform(self.data)
        self.assertEqual(transformed.shape, (100, 5))
        
        # Test inverse transform
        reconstructed = pca.inverse_transform(transformed)
        self.assertEqual(reconstructed.shape, self.data.shape)
        
        # Test explained variance
        var_ratio = pca.get_explained_variance_ratio()
        self.assertEqual(len(var_ratio), 5)
        self.assertTrue(all(var_ratio >= 0))
        self.assertTrue(all(var_ratio <= 1))
    
    def test_latent_space_interpolation(self):
        """Test latent space interpolation."""
        # Create a simple autoencoder for testing
        ae = VanillaAutoencoder(
            input_dim=20,
            latent_dim=5,
            encoder_layers=[10],
            decoder_layers=[10]
        )
        
        interpolator = LatentSpaceInterpolation(ae)
        
        # Test linear interpolation
        z1 = np.random.random(5)
        z2 = np.random.random(5)
        
        z_interp = interpolator.linear_interpolation(z1, z2, num_steps=5)
        self.assertEqual(z_interp.shape, (5, 5))
        
        # Test data point interpolation
        x1 = self.data[0]
        x2 = self.data[1]
        
        z_interp, x_interp = interpolator.interpolate_data_points(x1, x2, num_steps=3)
        self.assertEqual(z_interp.shape, (3, 5))
        self.assertEqual(x_interp.shape, (3, 20))


class TestUtils(unittest.TestCase):
    """Test utility classes."""
    
    def setUp(self):
        np.random.seed(42)
        self.data = np.random.random((100, 10))
    
    def test_data_preprocessor(self):
        """Test data preprocessor."""
        preprocessor = DataPreprocessor(scaling_method='standard')
        
        # Test data preparation
        splits = preprocessor.prepare_data(self.data, validation_split=0.2)
        
        self.assertIn('train', splits)
        self.assertIn('validation', splits)
        self.assertEqual(splits['train'].shape[0], 80)
        self.assertEqual(splits['validation'].shape[0], 20)
        
        # Test transform
        new_data = np.random.random((10, 10))
        transformed = preprocessor.transform(new_data)
        self.assertEqual(transformed.shape, new_data.shape)
        
        # Test noise addition
        noisy = preprocessor.add_noise(self.data, noise_type='gaussian', noise_level=0.1)
        self.assertEqual(noisy.shape, self.data.shape)
        self.assertGreater(np.mean(np.abs(noisy - self.data)), 0)
    
    def test_model_saver(self):
        """Test model saving and loading."""
        # Create a simple autoencoder
        ae = VanillaAutoencoder(
            input_dim=10,
            latent_dim=5,
            encoder_layers=[8],
            decoder_layers=[8]
        )
        
        # Train briefly
        ae.fit(self.data, epochs=1, verbose=0)
        
        # Test saving
        saver = ModelSaver()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "test_model")
            
            # Save model
            saver.save_autoencoder(ae, save_path, save_format='weights')
            
            # Check files exist
            self.assertTrue(os.path.exists(f"{save_path}_config.json"))
            self.assertTrue(os.path.exists(f"{save_path}_autoencoder_weights.h5"))


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestLossFunctions,
        TestAutoencoders,
        TestProjections,
        TestUtils
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")
    exit(0 if success else 1)