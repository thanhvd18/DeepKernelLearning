import os
import sys

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
sys.path.append(os.path.join(os.getcwd(), '..', '..', 'DKL'))

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC


import DKL
from DKL import config
from DKL.kernels import KernelConstructor
from DKL.model.model import DeepCNN
from DKL.data import DataLoader, CrossValidator, DataLoaderAdvanced


if __name__ == '__main__':

    data_dir = "../data/AD_CN"
    #desrcibe data structure of data saved in csv files
    data_config = {
        "MRI": os.path.join(data_dir, "MRI.csv"),
        # "PET": os.path.join(data_dir, "PET.csv"),
        # "CSF": os.path.join(data_dir, "CSF.csv"),
        # "SNP": os.path.join(data_dir, "SNP.csv"),
        "label": os.path.join(data_dir, "AD_CN_label.csv")
    }
    data_loader = DataLoader(data_config)

    # representation_type = "kernel" #representation_types = ["feature", "kernel"]
    kernel_level = "early" # kernel_levels = ["early", "middle", "late"]
    kernel_constructor = KernelConstructor(kernel_level, method="linear")
    cv = CrossValidator(n_splits=5,n_repeats=1, stratified=False, random_state=1)

    splits, idxes = DKL.utils.train_test_kernel_cv_split(data_loader,cv,kernel_constructor)
    idx = next(idxes)
    kernel_split, feature_split = splits[0]
    [Xs_kernel_train, Y_K_train, Xs_kernel_train_test,Xs_kernel_test_test, Y_K_test] = kernel_split
    [ Xs_train, Y_train, Xs_test, Y_test] = feature_split

    Xs_kernel_train = DKL.utils.stack_kernel_matrix_from_dict(Xs_kernel_train)
    Xs_kernel_train_test = DKL.utils.stack_kernel_matrix_from_dict(Xs_kernel_train_test)
    Xs_kernel_test_test = DKL.utils.stack_kernel_matrix_from_dict(Xs_kernel_test_test)

    kernel_combiner = DKL.kernels.KernelCombination(method="sum")
    kernel_combiner.fit(Xs_kernel_train, Y_K_train,Xs_kernel_train_test, Xs_kernel_test_test)
    combined_train_kernel = kernel_combiner.transform(Xs_kernel_train)
    combined_test_kernel = kernel_combiner.transform(Xs_kernel_train_test)


    clf = DKL.kernels.KernelClassifier()
    # feature_selector = FeatureSelector(clf, kernel_combiner)
    clf.fit(combined_train_kernel, Y_K_train)
    y_pred = clf.predict(combined_test_kernel)
    cf = confusion_matrix(Y_test, y_pred)
    report = pd.DataFrame(classification_report(Y_test, y_pred, output_dict=True))

    print(cf)
    print(report)
    print("Done!")

    X = data_loader.data['MRI']
    y = data_loader.data['label']
    #

    X_train, X_test, y_train, y_test = X[idx[0]], X[idx[1]], y[idx[0]], y[idx[1]]
    clf = SVC(kernel="linear")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cf = confusion_matrix(y_test, y_pred)
    report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True))
    print(cf)
