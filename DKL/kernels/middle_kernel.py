import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import rbf_kernel


class MiddleKernel:
    def __init__(self, kernel_type, **kwargs):
        """
        Initialize the EarlyKernel class.

        Args:
            kernel_type (str): The type of kernel to use (default: "rbf").
            **kwargs: Additional parameters for the kernel function.
        """
        self.kernel_type = kernel_type
        self.params = kwargs

    def get_kernel(self, X):
        pass

    def fit(self, X,y):
        pass

    def transform(self, X):
        pass

    def __str__(self):
        return  f"MiddleKernel(kernel_type={self.kernel_type}, params={self.params})"