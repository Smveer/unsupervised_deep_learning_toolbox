"""
Custom regularization loss functions for autoencoders.
"""

import tensorflow as tf
from typing import Optional


class KLDivergence:
    """
    Kullback-Leibler divergence loss implemented from scratch.
    
    Commonly used in Variational Autoencoders to enforce latent space distribution.
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Initialize KL divergence loss.
        
        Args:
            reduction: Type of reduction to apply ('mean', 'sum', 'none')
        """
        self.reduction = reduction
    
    def __call__(
        self, 
        mu: tf.Tensor, 
        log_var: tf.Tensor, 
        prior_mu: Optional[tf.Tensor] = None,
        prior_log_var: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """
        Compute KL divergence between two multivariate Gaussian distributions.
        
        For VAE, this is typically KL(q(z|x) || p(z)) where:
        - q(z|x) is the encoder distribution with parameters (mu, log_var)
        - p(z) is the prior distribution (usually standard normal)
        
        Args:
            mu: Mean of the approximate posterior
            log_var: Log variance of the approximate posterior
            prior_mu: Mean of the prior (defaults to 0)
            prior_log_var: Log variance of the prior (defaults to 0)
            
        Returns:
            KL divergence tensor
        """
        if prior_mu is None:
            prior_mu = tf.zeros_like(mu)
        if prior_log_var is None:
            prior_log_var = tf.zeros_like(log_var)
        
        # KL divergence formula for multivariate Gaussians:
        # KL = 0.5 * sum(log(var_prior/var_posterior) + (var_posterior + (mu_posterior - mu_prior)^2) / var_prior - 1)
        
        # Convert log variances to variances
        var = tf.exp(log_var)
        prior_var = tf.exp(prior_log_var)
        
        # Compute KL divergence terms
        log_var_ratio = prior_log_var - log_var
        var_ratio = var / prior_var
        mu_diff_squared = tf.square(mu - prior_mu) / prior_var
        
        # KL divergence per dimension
        kl_per_dim = 0.5 * (log_var_ratio + var_ratio + mu_diff_squared - 1.0)
        
        # Sum over latent dimensions
        kl = tf.reduce_sum(kl_per_dim, axis=-1)
        
        # Apply reduction
        if self.reduction == 'mean':
            return tf.reduce_mean(kl)
        elif self.reduction == 'sum':
            return tf.reduce_sum(kl)
        elif self.reduction == 'none':
            return kl
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
    
    def standard_normal_kl(self, mu: tf.Tensor, log_var: tf.Tensor) -> tf.Tensor:
        """
        Compute KL divergence with standard normal prior (most common case for VAE).
        
        Args:
            mu: Mean of the approximate posterior
            log_var: Log variance of the approximate posterior
            
        Returns:
            KL divergence with N(0,I) prior
        """
        # Simplified formula for standard normal prior
        kl_per_dim = 0.5 * (tf.square(mu) + tf.exp(log_var) - log_var - 1.0)
        kl = tf.reduce_sum(kl_per_dim, axis=-1)
        
        # Apply reduction
        if self.reduction == 'mean':
            return tf.reduce_mean(kl)
        elif self.reduction == 'sum':
            return tf.reduce_sum(kl)
        elif self.reduction == 'none':
            return kl
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
    
    def get_config(self):
        """Get configuration dictionary."""
        return {'reduction': self.reduction}


class SparsityRegularization:
    """
    Sparsity regularization loss for sparse autoencoders.
    
    Encourages sparse activations in the hidden layer using KL divergence
    between actual and target activation distributions.
    """
    
    def __init__(self, sparsity_target: float = 0.05, sparsity_weight: float = 1.0, reduction: str = 'mean'):
        """
        Initialize sparsity regularization.
        
        Args:
            sparsity_target: Target average activation (rho)
            sparsity_weight: Weight for sparsity term (beta)
            reduction: Type of reduction to apply ('mean', 'sum', 'none')
        """
        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight
        self.reduction = reduction
        self.epsilon = 1e-8  # Small constant to avoid log(0)
    
    def __call__(self, activations: tf.Tensor) -> tf.Tensor:
        """
        Compute sparsity regularization loss.
        
        Args:
            activations: Hidden layer activations (batch_size, hidden_dim)
            
        Returns:
            Sparsity regularization loss
        """
        # Compute average activation for each neuron across the batch
        rho_hat = tf.reduce_mean(activations, axis=0)
        
        # Clip to avoid numerical issues
        rho_hat = tf.clip_by_value(rho_hat, self.epsilon, 1.0 - self.epsilon)
        rho = tf.clip_by_value(self.sparsity_target, self.epsilon, 1.0 - self.epsilon)
        
        # KL divergence between Bernoulli distributions
        # KL(rho || rho_hat) = rho * log(rho / rho_hat) + (1-rho) * log((1-rho) / (1-rho_hat))
        kl_div = (rho * tf.math.log(rho / rho_hat) + 
                  (1.0 - rho) * tf.math.log((1.0 - rho) / (1.0 - rho_hat)))
        
        # Sum across all neurons
        sparsity_loss = tf.reduce_sum(kl_div)
        
        # Apply sparsity weight
        sparsity_loss = self.sparsity_weight * sparsity_loss
        
        # Note: reduction doesn't apply here as we want one scalar per batch
        return sparsity_loss
    
    def get_config(self):
        """Get configuration dictionary."""
        return {
            'sparsity_target': self.sparsity_target,
            'sparsity_weight': self.sparsity_weight,
            'reduction': self.reduction
        }


class L1Regularization:
    """
    L1 (Lasso) regularization loss.
    
    Encourages sparsity by penalizing the absolute values of parameters.
    """
    
    def __init__(self, l1_weight: float = 0.01, reduction: str = 'mean'):
        """
        Initialize L1 regularization.
        
        Args:
            l1_weight: Weight for L1 penalty
            reduction: Type of reduction to apply ('mean', 'sum', 'none')
        """
        self.l1_weight = l1_weight
        self.reduction = reduction
    
    def __call__(self, weights: tf.Tensor) -> tf.Tensor:
        """
        Compute L1 regularization loss.
        
        Args:
            weights: Model weights tensor
            
        Returns:
            L1 regularization loss
        """
        l1_loss = tf.reduce_sum(tf.abs(weights))
        return self.l1_weight * l1_loss
    
    def get_config(self):
        """Get configuration dictionary."""
        return {
            'l1_weight': self.l1_weight,
            'reduction': self.reduction
        }


class L2Regularization:
    """
    L2 (Ridge) regularization loss.
    
    Prevents overfitting by penalizing large parameter values.
    """
    
    def __init__(self, l2_weight: float = 0.01, reduction: str = 'mean'):
        """
        Initialize L2 regularization.
        
        Args:
            l2_weight: Weight for L2 penalty
            reduction: Type of reduction to apply ('mean', 'sum', 'none')
        """
        self.l2_weight = l2_weight
        self.reduction = reduction
    
    def __call__(self, weights: tf.Tensor) -> tf.Tensor:
        """
        Compute L2 regularization loss.
        
        Args:
            weights: Model weights tensor
            
        Returns:
            L2 regularization loss
        """
        l2_loss = tf.reduce_sum(tf.square(weights))
        return self.l2_weight * l2_loss
    
    def get_config(self):
        """Get configuration dictionary."""
        return {
            'l2_weight': self.l2_weight,
            'reduction': self.reduction
        }