import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import rbf_kernel


class LateKernel:
    def __init__(self, kernel_type = None, **kwargs):
        """
        Initialize the LateKernel class.

        Args:
            kernel_type (str): The type of kernel to use (default: None).
            **kwargs: Additional parameters for the kernel function.
        """
        self.n_trees = kwargs.get('n_trees', 500)
        self.params = kwargs
        self.rf_classifier = None
        self.X_fit = None
        self.params = kwargs
        self.leaf_indices_ = None

    def get_kernel(self, X):
        if self.rf_classifier is None:
            raise ValueError("Please fit the model first.")
        return self._compute_rf_kernel_same_data(X)

    def fit(self, X,y):
        self.X_fit = np.array(X)

        self.rf_classifier = RandomForestClassifier(n_estimators=self.n_trees)
        self.rf_classifier.fit(X, y)

        self.leaf_indices_ = np.array([
            tree.apply(X)
            for tree in self.rf_classifier.estimators_
        ])

        return self
    
    def _compute_rf_kernel(self, leaf_indices_1, leaf_indices_2):
        """
        Tính kernel giữa hai tập mẫu dựa trên leaf indices của RandomForest.

        - leaf_indices_1: shape (n_estimators, n_samples1)
        - leaf_indices_2: shape (n_estimators, n_samples2)
        """
        n1 = leaf_indices_1.shape[1]
        n2 = leaf_indices_2.shape[1]
        
        # pairwise_sim[i, j] = số lần mẫu i (thuộc tập 1) và mẫu j (thuộc tập 2) cùng leaf
        pairwise_sim = np.zeros((n1, n2), dtype=float)
        
        # Với mỗi cây, xét leaf index của tập 1 và tập 2
        for t in range(self.n_trees):
            # Lấy leaf index của từng mẫu trong 2 tập
            leaves_1 = leaf_indices_1[t]
            leaves_2 = leaf_indices_2[t]
            
            # Tăng counter nếu leaves_1[i] == leaves_2[j]
            for i in range(n1):
                for j in range(n2):
                    if leaves_1[i] == leaves_2[j]:
                        pairwise_sim[i, j] += 1
        
        # Chuẩn hoá bằng số cây
        pairwise_sim /= self.n_trees
        return pairwise_sim
    
    def _compute_rf_kernel_same_data(self, X, y=None):
        """
        Tính kernel trên chính tập dữ liệu X (pairwise giữa X và X).
        """
        # Nếu muốn sắp xếp dữ liệu theo y (tùy nhu cầu), bạn có thể thêm đoạn:
        if y is not None:
            index = np.argsort(y)
            y = np.array(y)[index]
            X = np.array(X)[index, :]

        # Tạo mảng pairwise similarities
        pairwise_similarities = [[0] * len(X) for _ in range(len(X))]
        
        # Lấy từng estimator (cây) ra
        for tree in self.rf_classifier.estimators_:
            leaf_indices = tree.apply(X)
            
            # Cập nhật cặp (i, j) nếu cùng leaf
            for i in range(len(X)):
                for j in range(i + 1, len(X)):
                    if leaf_indices[i] == leaf_indices[j]:
                        pairwise_similarities[i][j] += 1
                        pairwise_similarities[j][i] += 1
        
        # Chuẩn hoá
        normalized_pairwise_similarities = [
            [sim / self.n_trees for sim in row] 
            for row in pairwise_similarities
        ]
        K = np.array(normalized_pairwise_similarities)
        return K

    def transform(self, X):
        """
        Trả về kernel giữa X (mới) và X_fit (đã lưu sau khi fit).
        Shape đầu ra: (n_samples_moi, n_samples_train).
        """
        if self.rf_classifier is None or self.leaf_indices_ is None:
            raise ValueError("Random Forest chưa được fit hoặc leaf_indices_ không tồn tại.")
        
        # leaf_indices của X mới
        new_leaf_indices = np.array([
            tree.apply(X)
            for tree in self.rf_classifier.estimators_
        ])  # shape (n_estimators, n_samples_moi)
        
        # Sử dụng leaf_indices_ của X_fit
        return  self._compute_rf_kernel(self.leaf_indices_ ,new_leaf_indices), self._compute_rf_kernel_same_data(X)



    def fit_transform(self, X, y):
        self.fit(X,y)
        return self.transform(X)

    def __str__(self):
        return  f"LateKernel(kernel_type={self.kernel_type}, params={self.params})"