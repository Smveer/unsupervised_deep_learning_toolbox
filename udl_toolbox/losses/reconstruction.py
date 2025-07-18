"""
Custom reconstruction loss functions implemented from scratch.
"""

import tensorflow as tf
from typing import Optional


class MeanSquaredError:
    """
    Mean Squared Error loss function implemented from scratch.
    
    This is equivalent to L2 loss and is commonly used for continuous data.
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Initialize MSE loss.
        
        Args:
            reduction: Type of reduction to apply ('mean', 'sum', 'none')
        """
        self.reduction = reduction
    
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute MSE loss.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            MSE loss tensor
        """
        # Compute squared differences
        squared_diff = tf.square(y_true - y_pred)
        
        # Apply reduction
        if self.reduction == 'mean':
            return tf.reduce_mean(squared_diff)
        elif self.reduction == 'sum':
            return tf.reduce_sum(squared_diff)
        elif self.reduction == 'none':
            return squared_diff
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
    
    def get_config(self):
        """Get configuration dictionary."""
        return {'reduction': self.reduction}


class BinaryCrossentropy:
    """
    Binary crossentropy loss function implemented from scratch.
    
    Commonly used for binary data or when outputs are sigmoid-activated.
    """
    
    def __init__(self, from_logits: bool = False, epsilon: float = 1e-7, reduction: str = 'mean'):
        """
        Initialize binary crossentropy loss.
        
        Args:
            from_logits: Whether predictions are logits or probabilities
            epsilon: Small constant to avoid log(0)
            reduction: Type of reduction to apply ('mean', 'sum', 'none')
        """
        self.from_logits = from_logits
        self.epsilon = epsilon
        self.reduction = reduction
    
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute binary crossentropy loss.
        
        Args:
            y_true: Ground truth values (0 or 1)
            y_pred: Predicted values
            
        Returns:
            Binary crossentropy loss tensor
        """
        if self.from_logits:
            # Apply sigmoid if inputs are logits
            y_pred = tf.nn.sigmoid(y_pred)
        
        # Clip predictions to avoid log(0)
        y_pred = tf.clip_by_value(y_pred, self.epsilon, 1.0 - self.epsilon)
        
        # Compute binary crossentropy: -[y*log(p) + (1-y)*log(1-p)]
        bce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        
        # Apply reduction
        if self.reduction == 'mean':
            return tf.reduce_mean(bce)
        elif self.reduction == 'sum':
            return tf.reduce_sum(bce)
        elif self.reduction == 'none':
            return bce
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
    
    def get_config(self):
        """Get configuration dictionary."""
        return {
            'from_logits': self.from_logits,
            'epsilon': self.epsilon,
            'reduction': self.reduction
        }


class CategoricalCrossentropy:
    """
    Categorical crossentropy loss function implemented from scratch.
    
    Used for multi-class classification problems.
    """
    
    def __init__(self, from_logits: bool = False, epsilon: float = 1e-7, reduction: str = 'mean'):
        """
        Initialize categorical crossentropy loss.
        
        Args:
            from_logits: Whether predictions are logits or probabilities
            epsilon: Small constant to avoid log(0)
            reduction: Type of reduction to apply ('mean', 'sum', 'none')
        """
        self.from_logits = from_logits
        self.epsilon = epsilon
        self.reduction = reduction
    
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute categorical crossentropy loss.
        
        Args:
            y_true: Ground truth one-hot encoded labels
            y_pred: Predicted probabilities or logits
            
        Returns:
            Categorical crossentropy loss tensor
        """
        if self.from_logits:
            # Apply softmax if inputs are logits
            y_pred = tf.nn.softmax(y_pred)
        
        # Clip predictions to avoid log(0)
        y_pred = tf.clip_by_value(y_pred, self.epsilon, 1.0 - self.epsilon)
        
        # Compute categorical crossentropy: -sum(y * log(p))
        cce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        
        # Apply reduction
        if self.reduction == 'mean':
            return tf.reduce_mean(cce)
        elif self.reduction == 'sum':
            return tf.reduce_sum(cce)
        elif self.reduction == 'none':
            return cce
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
    
    def get_config(self):
        """Get configuration dictionary."""
        return {
            'from_logits': self.from_logits,
            'epsilon': self.epsilon,
            'reduction': self.reduction
        }


class Huber:
    """
    Huber loss function implemented from scratch.
    
    Combines MSE and MAE - less sensitive to outliers than MSE.
    """
    
    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        """
        Initialize Huber loss.
        
        Args:
            delta: Threshold for switching between quadratic and linear loss
            reduction: Type of reduction to apply ('mean', 'sum', 'none')
        """
        self.delta = delta
        self.reduction = reduction
    
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute Huber loss.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            Huber loss tensor
        """
        error = y_true - y_pred
        abs_error = tf.abs(error)
        
        # Quadratic loss for small errors, linear for large errors
        quadratic = 0.5 * tf.square(error)
        linear = self.delta * abs_error - 0.5 * tf.square(self.delta)
        
        huber_loss = tf.where(abs_error <= self.delta, quadratic, linear)
        
        # Apply reduction
        if self.reduction == 'mean':
            return tf.reduce_mean(huber_loss)
        elif self.reduction == 'sum':
            return tf.reduce_sum(huber_loss)
        elif self.reduction == 'none':
            return huber_loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
    
    def get_config(self):
        """Get configuration dictionary."""
        return {
            'delta': self.delta,
            'reduction': self.reduction
        }