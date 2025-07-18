# UDL Toolbox - Unsupervised Deep Learning Toolbox

🧠 **Comprehensive Autoencoder Implementations with Custom Loss Functions and Visualization Tools**

A complete implementation of autoencoder architectures with custom loss functions, data projection utilities, and visualization tools, built from scratch using TensorFlow/Keras.

## ✨ Features

### 🏗️ Autoencoder Architectures
- **Vanilla Autoencoder**: Basic encoder-decoder architecture
- **Sparse Autoencoder**: Enforces sparsity in hidden representations
- **Denoising Autoencoder**: Learns robust representations from corrupted inputs
- **Variational Autoencoder (VAE)**: Probabilistic latent space modeling
- **Convolutional Autoencoder**: For image data with spatial structure preservation

### 🎯 Custom Loss Functions
- **Reconstruction Losses**: MSE, Binary Crossentropy, Huber Loss
- **Regularization Terms**: KL Divergence, Sparsity Regularization, L1/L2
- **VAE-Specific**: Combined reconstruction + KL loss with β-VAE support

### 📊 Data Projection & Analysis
- **PCA Projection**: Linear dimensionality reduction from scratch
- **t-SNE Projection**: Non-linear embedding for visualization
- **Latent Space Interpolation**: Smooth transitions in learned representations

### 📈 Visualization Tools
- **Latent Space Visualization**: 2D/3D plots, distribution analysis
- **Reconstruction Quality**: Error analysis, before/after comparisons
- **Training Progress**: Loss curves, component analysis, convergence metrics

### 🛠️ Utilities
- **Data Preprocessing**: Scaling, normalization, augmentation
- **Model I/O**: Save/load models, checkpoints, configurations
- **Custom Training Loops**: Gradient computation, progress tracking

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Smveer/unsupervised_deep_learning_toolbox.git
cd unsupervised_deep_learning_toolbox

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```python
import numpy as np
from udl_toolbox.autoencoders import VanillaAutoencoder
from udl_toolbox.utils import DataPreprocessor
from udl_toolbox.visualization import LatentSpaceVisualizer

# Prepare data
data = np.random.random((1000, 50))  # 1000 samples, 50 features
preprocessor = DataPreprocessor(scaling_method='standard')
data_splits = preprocessor.prepare_data(data, validation_split=0.2)

# Create and train autoencoder
autoencoder = VanillaAutoencoder(
    input_dim=50,
    latent_dim=10,
    encoder_layers=[30, 20],
    decoder_layers=[20, 30]
)

history = autoencoder.fit(
    data_splits['train'],
    validation_data=data_splits['validation'],
    epochs=100,
    batch_size=32
)

# Visualize results
visualizer = LatentSpaceVisualizer(autoencoder)
visualizer.plot_2d_latent_space(data_splits['train'], method='pca')
```

## 📚 Examples

### Sparse Autoencoder
```python
from udl_toolbox.autoencoders import SparseAutoencoder

sparse_ae = SparseAutoencoder(
    input_dim=784,  # MNIST-like data
    latent_dim=128,
    encoder_layers=[512, 256],
    decoder_layers=[256, 512],
    sparsity_target=0.05,  # Target 5% activation
    sparsity_weight=1.0,
    activation='sigmoid'
)

# Train and analyze sparsity
history = sparse_ae.fit(train_data, epochs=100)
sparse_ae.visualize_sparsity(train_data)
```

### Variational Autoencoder
```python
from udl_toolbox.autoencoders import VariationalAutoencoder

vae = VariationalAutoencoder(
    input_dim=784,
    latent_dim=20,
    encoder_layers=[400, 200],
    decoder_layers=[200, 400],
    beta=1.0,  # β-VAE parameter
    reconstruction_loss='binary_crossentropy'
)

# Train VAE
history = vae.fit(train_data, epochs=100)

# Generate new samples
generated_samples = vae.generate(num_samples=10)

# Interpolate between points
interpolated = vae.interpolate(sample1, sample2, num_steps=10)
```

### Convolutional Autoencoder
```python
from udl_toolbox.autoencoders import ConvolutionalAutoencoder

conv_ae = ConvolutionalAutoencoder(
    input_shape=(28, 28, 1),  # MNIST images
    latent_dim=64,
    encoder_filters=[32, 64, 128],
    decoder_filters=[128, 64, 32],
    kernel_size=3,
    strides=2
)

# Train on image data
history = conv_ae.fit(image_data, epochs=50)

# Visualize feature maps
feature_maps = conv_ae.get_feature_maps(test_images)
```

### Custom Loss Functions
```python
from udl_toolbox.losses import VAELoss, SparsityRegularization

# Custom VAE loss with β annealing
vae_loss = VAELoss(
    reconstruction_loss='mse',
    beta=0.1,  # Start with low β
    reduction='mean'
)

# Sparsity regularization
sparsity_reg = SparsityRegularization(
    sparsity_target=0.02,
    sparsity_weight=2.0
)
```

## 🔧 Advanced Features

### Data Projection Analysis
```python
from udl_toolbox.projections import PCAProjection, LatentSpaceInterpolation

# PCA analysis
pca = PCAProjection(n_components=10)
pca_data = pca.fit_transform(high_dim_data)
explained_var = pca.get_explained_variance_ratio()

# Latent space interpolation
interpolator = LatentSpaceInterpolation(autoencoder)
path = interpolator.latent_arithmetic(
    base_point=latent_vector,
    direction_vector=semantic_direction,
    scales=[-2, -1, 0, 1, 2]
)
```

### Model Saving & Loading
```python
from udl_toolbox.utils import ModelSaver

saver = ModelSaver()

# Save complete model
saver.save_autoencoder(
    autoencoder,
    save_path="./models/my_autoencoder",
    save_format='complete',
    include_optimizer=True
)

# Load model
loaded_ae = saver.load_autoencoder(
    "./models/my_autoencoder",
    VanillaAutoencoder
)
```

### Comprehensive Visualization
```python
from udl_toolbox.visualization import ReconstructionVisualizer, LossVisualizer

# Reconstruction analysis
recon_vis = ReconstructionVisualizer(autoencoder)
recon_vis.plot_reconstruction_comparison(test_data, num_samples=5)
recon_vis.plot_reconstruction_error_distribution(test_data)
recon_vis.print_reconstruction_summary(test_data)

# Training analysis
loss_vis = LossVisualizer()
loss_vis.plot_training_curves(history)
loss_vis.plot_loss_components(history)
loss_vis.plot_training_summary(history)
```

## 🧪 Running Examples and Tests

### Comprehensive Example
```bash
python examples/comprehensive_example.py
```

This will run demonstrations of all autoencoder types with synthetic data.

### Tests
```bash
python tests/test_basic_functionality.py
```

Run the test suite to verify all components work correctly.

## 📁 Project Structure

```
udl_toolbox/
├── autoencoders/          # Autoencoder implementations
│   ├── base.py           # Base autoencoder class
│   ├── vanilla.py        # Vanilla autoencoder
│   ├── sparse.py         # Sparse autoencoder
│   ├── denoising.py      # Denoising autoencoder
│   ├── variational.py    # Variational autoencoder
│   └── convolutional.py  # Convolutional autoencoder
├── losses/               # Custom loss functions
│   ├── reconstruction.py # Reconstruction losses
│   ├── regularization.py # Regularization terms
│   └── vae_loss.py      # VAE-specific losses
├── projections/          # Data projection utilities
│   ├── pca.py           # PCA implementation
│   ├── tsne.py          # t-SNE utilities
│   └── interpolation.py # Latent space interpolation
├── visualization/        # Visualization tools
│   ├── latent_space.py  # Latent space visualization
│   ├── reconstruction.py # Reconstruction quality
│   └── training.py      # Training progress
└── utils/               # Utility functions
    ├── data_preprocessing.py # Data preprocessing
    └── model_io.py          # Model I/O operations
```

## 🎯 Key Design Principles

1. **Modular Architecture**: Each component is self-contained and reusable
2. **Custom Implementation**: All core algorithms implemented from scratch
3. **No Code Duplication**: Inheritance and composition prevent redundancy
4. **Comprehensive Testing**: Full test coverage for reliability
5. **Educational Focus**: Clear, documented code for learning
6. **Production Ready**: Efficient, scalable implementations

## 🤝 Use Cases

- **Research**: Experiment with different autoencoder architectures
- **Education**: Learn autoencoder concepts with clear implementations
- **Prototyping**: Quickly test ideas with comprehensive tooling
- **Production**: Deploy robust autoencoder solutions
- **Analysis**: Deep dive into model behavior with visualization tools

## 📋 Requirements

- Python >= 3.8
- TensorFlow >= 2.13.0
- NumPy >= 1.21.0
- Matplotlib >= 3.5.0
- Seaborn >= 0.11.0
- Scikit-learn >= 1.0.0
- Pandas >= 1.3.0
- Plotly >= 5.0.0

## 🆘 Troubleshooting

### Common Issues

1. **Memory Issues with Large Models**: Use smaller batch sizes or gradient accumulation
2. **Convergence Problems**: Adjust learning rate, add batch normalization, or try different initializations
3. **Overfitting**: Increase regularization, add dropout, or reduce model complexity
4. **Slow Training**: Enable GPU acceleration, optimize data loading, or reduce model size

### Performance Tips

- Use mixed precision training for faster computation
- Preprocess data once and cache results
- Use TensorFlow data pipelines for efficient loading
- Monitor GPU utilization during training

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📞 Support

If you encounter any issues or have questions, please open an issue on GitHub.

---

**Happy Learning! 🚀**
