"""
VAE-specific loss function combining reconstruction and KL divergence.
"""

import tensorflow as tf
from .reconstruction import MeanSquaredError, BinaryCrossentropy
from .regularization import KLDivergence


class VAELoss:
    """
    Variational Autoencoder loss function.
    
    Combines reconstruction loss with KL divergence regularization.
    Loss = Reconstruction_Loss + β * KL_Divergence
    """
    
    def __init__(
        self,
        reconstruction_loss: str = 'mse',
        beta: float = 1.0,
        reduction: str = 'mean',
        **kwargs
    ):
        """
        Initialize VAE loss.
        
        Args:
            reconstruction_loss: Type of reconstruction loss ('mse', 'binary_crossentropy')
            beta: Weight for KL divergence term (β-VAE parameter)
            reduction: Type of reduction to apply
            **kwargs: Additional arguments for reconstruction loss
        """
        self.beta = beta
        self.reduction = reduction
        
        # Initialize reconstruction loss
        if reconstruction_loss == 'mse':
            self.reconstruction_fn = MeanSquaredError(reduction=reduction)
        elif reconstruction_loss == 'binary_crossentropy':
            self.reconstruction_fn = BinaryCrossentropy(reduction=reduction, **kwargs)
        else:
            raise ValueError(f"Unknown reconstruction loss: {reconstruction_loss}")
        
        # Initialize KL divergence
        self.kl_fn = KLDivergence(reduction=reduction)
        
        self.reconstruction_loss_type = reconstruction_loss
    
    def __call__(
        self,
        x_true: tf.Tensor,
        x_reconstructed: tf.Tensor,
        mu: tf.Tensor,
        log_var: tf.Tensor
    ) -> dict:
        """
        Compute VAE loss.
        
        Args:
            x_true: Original input data
            x_reconstructed: Reconstructed data from decoder
            mu: Mean of latent distribution from encoder
            log_var: Log variance of latent distribution from encoder
            
        Returns:
            Dictionary containing individual loss components and total loss
        """
        # Reconstruction loss
        reconstruction_loss = self.reconstruction_fn(x_true, x_reconstructed)
        
        # KL divergence loss (with standard normal prior)
        kl_loss = self.kl_fn.standard_normal_kl(mu, log_var)
        
        # Total VAE loss
        total_loss = reconstruction_loss + self.beta * kl_loss
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'kl_loss': kl_loss,
            'regularization_loss': self.beta * kl_loss
        }
    
    def get_config(self):
        """Get configuration dictionary."""
        config = {
            'reconstruction_loss': self.reconstruction_loss_type,
            'beta': self.beta,
            'reduction': self.reduction
        }
        return config


class BetaVAELoss(VAELoss):
    """
    β-VAE loss with adjustable β parameter for disentanglement.
    
    Higher β values encourage disentangled representations at the cost
    of reconstruction quality.
    """
    
    def __init__(self, beta: float = 4.0, **kwargs):
        """
        Initialize β-VAE loss.
        
        Args:
            beta: β parameter controlling disentanglement vs reconstruction trade-off
            **kwargs: Other arguments passed to VAELoss
        """
        super().__init__(beta=beta, **kwargs)


class AnnealedVAELoss(VAELoss):
    """
    VAE loss with annealed β parameter.
    
    Gradually increases β during training to balance reconstruction
    and regularization.
    """
    
    def __init__(
        self,
        beta_start: float = 0.0,
        beta_end: float = 1.0,
        anneal_steps: int = 1000,
        **kwargs
    ):
        """
        Initialize annealed VAE loss.
        
        Args:
            beta_start: Initial β value
            beta_end: Final β value
            anneal_steps: Number of steps to anneal β
            **kwargs: Other arguments passed to VAELoss
        """
        super().__init__(beta=beta_start, **kwargs)
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.anneal_steps = anneal_steps
        self.current_step = 0
    
    def update_beta(self, step: int):
        """
        Update β parameter based on current training step.
        
        Args:
            step: Current training step
        """
        self.current_step = step
        if step >= self.anneal_steps:
            self.beta = self.beta_end
        else:
            # Linear annealing
            self.beta = self.beta_start + (self.beta_end - self.beta_start) * (step / self.anneal_steps)
    
    def __call__(self, x_true, x_reconstructed, mu, log_var):
        """Compute annealed VAE loss."""
        # Use current β value
        old_beta = self.beta
        result = super().__call__(x_true, x_reconstructed, mu, log_var)
        return result