import os
import sys

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
sys.path.append(os.path.join(os.getcwd(), '..', '..', 'DKL'))
from DKL.model.model import DeepCNN
from DKL.dataloader import DataLoader, CrossValidator, DataLoaderAdvanced
# from DKL.train import train
from DKL.loss import my_loss
import torch
from DKL import config
from DKL.kernels import KernelConstructor
import DKL
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':

    data_dir = os.path.join(os.getcwd(), "..", "..", "data/AD_CN")
    #desrcibe data structure of data saved in csv files
    data_config = {
        "MRI": os.path.join(data_dir, "MRI.csv"),
        "PET": os.path.join(data_dir, "PET.csv"),
        "CSF": os.path.join(data_dir, "CSF.csv"),
        # "SNP": os.path.join(data_dir, "SNP.csv"),
        "label": os.path.join(data_dir, "AD_CN_label.csv")
    }
    data_loader = DataLoader(data_config)
    # print(data_loader.get_modalities())
    # print(data_loader.get_data("MRI"))

    # representation_type = "kernel" #representation_types = ["feature", "kernel"]
    # kernel_level = "early" # kernel_levels = ["early", "middle", "late"]
    # kernel_constructor = KernelConstructor(kernel_level, method="polynomial")

    # kernel_level = "middle" # kernel_levels = ["early", "middle", "late"]
    # kernel_constructor = KernelConstructor(kernel_level, method="rbf")

    kernel_level = "late" # kernel_levels = ["early", "middle", "late"]
    kernel_constructor = KernelConstructor(kernel_level, method=None)

    # K_X = kernel_constructor.fit_transform(data_loader.get_data("MRI"), data_loader.get_data("label"))
    # print(K_X.shape)

    # #visualize kernel construction for each modality
    # K_X = []
    # modalities = []
    # for key in data_loader.get_modalities():
    #     K_X1, _ = kernel_constructor.fit_transform(data_loader.get_data(key), data_loader.get_data("label"))
    #     K_X.append(K_X1)
    #     modalities.append(key)
    # plt.figure(figsize=(8, 4))
    # plt.subplot(1, 3, 1)
    # plt.title(modalities[0])
    # plt.imshow(K_X[0])
    # plt.subplot(1, 3, 2)
    # plt.title(modalities[1])
    # plt.imshow(K_X[1])
    # plt.subplot(1, 3, 3)
    # plt.title(modalities[2])
    # plt.imshow(K_X[2])
    # plt.show()

    
    cv = CrossValidator(n_splits=5,n_repeats=1, stratified=False, random_state=1)

    splits = DKL.utils.train_test_kernel_cv_split(data_loader,cv,kernel_constructor)
    kernel_split, feature_split = splits[0]
    [Xs_kernel_train, Y_K_train, Xs_kernel_train_test,Xs_kernel_test_test, Y_K_test] = kernel_split
    [ Xs_train, Y_train, Xs_test, Y_test] = feature_split

    Xs_kernel_train = DKL.utils.stack_kernel_matrix_from_dict(Xs_kernel_train)
    Xs_kernel_train_test = DKL.utils.stack_kernel_matrix_from_dict(Xs_kernel_train_test)
    Xs_kernel_test_test = DKL.utils.stack_kernel_matrix_from_dict(Xs_kernel_test_test)

    print("====="*5)
    print("Xs_kernel_train: ", Xs_kernel_train.shape)
    print("Xs_kernel_train_test",Xs_kernel_train_test.shape)
    print("Xs_kernel_test_test",Xs_kernel_test_test.shape)
    print("====="*5)

    kernel_combiner = DKL.kernels.KernelCombination(method="DKL")
    kernel_combiner.fit(Xs_kernel_train, Y_K_train,Xs_kernel_train_test, Xs_kernel_test_test)
    combined_train_kernel = kernel_combiner.transform(Xs_kernel_train)
    combined_test_kernel = kernel_combiner.transform(Xs_kernel_train_test)

    print("====="*5)
    print(combined_train_kernel.shape)
    print(combined_test_kernel.shape)
    print("====="*5)

    plt.imshow(combined_train_kernel)
    plt.show()



    clf = DKL.kernels.KernelClassifier()
    # feature_selector = FeatureSelector(clf, kernel_combiner)
    clf.fit(combined_train_kernel, Y_K_train)
    y_pred = clf.predict(combined_test_kernel)
    cf = confusion_matrix(Y_test, y_pred)
    report = pd.DataFrame(classification_report(Y_test, y_pred, output_dict=True))



    # clf = DKL.kernels.KernelClassifier()
    # # feature_selector = FeatureSelector(clf, kernel_combiner)
    # clf.fit(Xs_kernel_train[1], Y_K_train)
    # y_pred = clf.predict(Xs_kernel_train_test[1])
    # cf = confusion_matrix(Y_test, y_pred)
    # report = pd.DataFrame(classification_report(Y_test, y_pred, output_dict=True))

    print(cf)
    print(report)
    print("Done!")
