"""
PCA projection implementation from scratch.
"""

import tensorflow as tf
import numpy as np
from typing import Optional, Tuple
from sklearn.preprocessing import StandardScaler


class PCAProjection:
    """
    Principal Component Analysis (PCA) implementation from scratch.
    
    Reduces dimensionality by projecting data onto principal components
    that capture the most variance in the data.
    """
    
    def __init__(self, n_components: int, standardize: bool = True):
        """
        Initialize PCA projection.
        
        Args:
            n_components: Number of principal components to keep
            standardize: Whether to standardize data before PCA
        """
        self.n_components = n_components
        self.standardize = standardize
        
        # Fitted parameters
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        self.scaler_ = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray) -> 'PCAProjection':
        """
        Fit PCA to the data.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Self for method chaining
        """
        X = np.array(X)
        
        # Standardize if requested
        if self.standardize:
            self.scaler_ = StandardScaler()
            X_processed = self.scaler_.fit_transform(X)
        else:
            X_processed = X.copy()
        
        # Center the data
        self.mean_ = np.mean(X_processed, axis=0)
        X_centered = X_processed - self.mean_
        
        # Compute covariance matrix
        n_samples = X_centered.shape[0]
        cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Keep only the requested number of components
        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / np.sum(eigenvalues)
        
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted PCA.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Transformed data of shape (n_samples, n_components)
        """
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before transform")
        
        X = np.array(X)
        
        # Apply same preprocessing as during fit
        if self.standardize:
            X_processed = self.scaler_.transform(X)
        else:
            X_processed = X.copy()
        
        # Center the data
        X_centered = X_processed - self.mean_
        
        # Project onto principal components
        return np.dot(X_centered, self.components_.T)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit PCA and transform data in one step.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Transformed data of shape (n_samples, n_components)
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Transform data back to original space.
        
        Args:
            X_transformed: Transformed data of shape (n_samples, n_components)
            
        Returns:
            Reconstructed data of shape (n_samples, n_features)
        """
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before inverse transform")
        
        X_transformed = np.array(X_transformed)
        
        # Project back to original space
        X_reconstructed = np.dot(X_transformed, self.components_) + self.mean_
        
        # Reverse standardization if applied
        if self.standardize:
            X_reconstructed = self.scaler_.inverse_transform(X_reconstructed)
        
        return X_reconstructed
    
    def get_explained_variance_ratio(self) -> np.ndarray:
        """
        Get the explained variance ratio for each component.
        
        Returns:
            Array of explained variance ratios
        """
        if not self.is_fitted:
            raise ValueError("PCA must be fitted first")
        return self.explained_variance_ratio_
    
    def get_cumulative_variance_ratio(self) -> np.ndarray:
        """
        Get cumulative explained variance ratio.
        
        Returns:
            Array of cumulative explained variance ratios
        """
        if not self.is_fitted:
            raise ValueError("PCA must be fitted first")
        return np.cumsum(self.explained_variance_ratio_)
    
    def find_n_components_for_variance(self, target_variance: float = 0.95) -> int:
        """
        Find number of components needed to explain target variance.
        
        Args:
            target_variance: Target cumulative variance ratio
            
        Returns:
            Number of components needed
        """
        if not self.is_fitted:
            raise ValueError("PCA must be fitted first")
        
        cumvar = self.get_cumulative_variance_ratio()
        n_components = np.argmax(cumvar >= target_variance) + 1
        return min(n_components, len(cumvar))
    
    def reconstruction_error(self, X: np.ndarray) -> float:
        """
        Compute reconstruction error for given data.
        
        Args:
            X: Input data to compute error for
            
        Returns:
            Mean squared reconstruction error
        """
        X_transformed = self.transform(X)
        X_reconstructed = self.inverse_transform(X_transformed)
        return np.mean((X - X_reconstructed) ** 2)
    
    def get_principal_components(self) -> np.ndarray:
        """
        Get the principal components (eigenvectors).
        
        Returns:
            Principal components matrix of shape (n_components, n_features)
        """
        if not self.is_fitted:
            raise ValueError("PCA must be fitted first")
        return self.components_
    
    def project_new_data(self, X_new: np.ndarray) -> np.ndarray:
        """
        Project new data onto the fitted PCA space.
        
        Args:
            X_new: New data to project
            
        Returns:
            Projected data
        """
        return self.transform(X_new)
    
    def get_config(self) -> dict:
        """Get configuration dictionary."""
        return {
            'n_components': self.n_components,
            'standardize': self.standardize
        }