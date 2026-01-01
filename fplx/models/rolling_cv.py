"""Rolling cross-validation for time-series models."""

import numpy as np
from typing import Generator


class RollingCV:
    """
    Generates indices for rolling cross-validation splits.
    
    This is adapted from the MLSP project for time-series validation.
    
    Parameters
    ----------
    initial_train_size : int
        Size of the initial training set.
    test_size : int
        Size of the test set (forecast horizon).
    step : int
        Step size to move the training window forward.
    """
    
    def __init__(self, initial_train_size: int, test_size: int, step: int = 1):
        if initial_train_size <= 0 or test_size <= 0 or step <= 0:
            raise ValueError("initial_train_size, test_size, and step must be positive integers.")
        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.step = step

    def split(self, X) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices to split data into training and test sets.
        
        Parameters
        ----------
        X : array-like
            Time series data.
            
        Yields
        ------
        train_indices : np.ndarray
            The training set indices for that split.
        test_indices : np.ndarray
            The testing set indices for that split.
        """
        n_samples = len(X)
        if self.initial_train_size + self.test_size > n_samples:
            raise ValueError("initial_train_size + test_size is larger than the number of samples.")

        train_start = 0
        while train_start + self.initial_train_size + self.test_size <= n_samples:
            train_end = train_start + self.initial_train_size
            test_end = train_end + self.test_size
            
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(train_end, test_end)
            
            yield train_indices, test_indices
            
            train_start += self.step


__all__ = ['RollingCV']
