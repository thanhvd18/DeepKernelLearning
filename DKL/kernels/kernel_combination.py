import os
import numpy as np
from ..model import DeepKernelLearning
class KernelCombination:
    def __init__(self, method="sum"):
        self.method = method
        self.combined_kernel = None
        self.model = None
    def fit(self, KX_train, KY_train, KX_train_test, KX_test_test):
        if self.method == "sum":
            pass
        elif self.method == "DKL":
            self.model = DeepKernelLearning()
            self.model.fit(KX_train, KY_train, KX_train_test, KX_test_test)

        else:
            raise ValueError("Unknown kernel combination method")
        return self

    def transform(self, KX):
        if self.method == "sum":
            return np.sum(KX,axis=0)

        elif self.method == "DKL":
            return self.model.transform(KX)
        else:
            raise ValueError("Unknown kernel combination method")