import os
import sys
import argparse

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
import numpy as np
from DKL.kernels.middle_kernel import MiddleKernel
from DKL.kernels.enhanced_middle_kernel import EnhancedMiddleKernel

if __name__ == '__main__':
    
    # Khai báo argparse
    parser = argparse.ArgumentParser(description="Deep Kernel Learning Experiment")
    parser.add_argument(
        "--modality", 
        type=str, 
        default="PET", 
        choices=["PET", "GM", "MRI","CSF", "concat"],
        help="Modality to use"
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="Version of the experiment"
    )
    # DEFINE MIDDLE KERNEL PARAMS
    parser.add_argument(
        "--kernel_type",
        type=str,
        default="rbf",
        help="Kernel type to use"
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=30,
        help="Latent dimension"
    )
    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=[1000, 50],
        help="Hidden dimensions"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--lambda_",
        type=float,
        default=0.75,
        help="Lambda"
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.1,
        help="Dropout rate"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=200,
        help="Patience"
    )



    args = parser.parse_args()
    
    chosen_modality = args.modality
    version = args.version


    
    

    data_dir = os.path.join(os.getcwd(), "..", "..", "data/AD_CN")
    # Mặc định vẫn khai báo tất cả file CSV
    data_config = {
        "GM": os.path.join(data_dir, "GM.csv"),
        "PET": os.path.join(data_dir, "PET.csv"),
        "CSF": os.path.join(data_dir, "CSF.csv"),
        "MRI": os.path.join(data_dir, "MRI.csv"),
        "concat": os.path.join(data_dir, "concat.csv"),
        "label": os.path.join(data_dir, "AD_CN_label.csv")
    }
    
    # Để chỉ load đúng modality mà bạn muốn (và label), 
    # filter lại data_config, bỏ những modality không dùng.
    keep_keys = [chosen_modality, "label"]
    data_config = {k: v for k, v in data_config.items() if k in keep_keys}
    
    data_loader = DataLoader(data_config)


    kernel_constructor = KernelConstructor(kernel_level="middle", method="", 
                                           kernel_method=MiddleKernel(kernel_type=args.kernel_type,
                                                                      latent_dim=args.latent_dim,
                                                                      hidden_dims=args.hidden_dims,
                                                                        activation=torch.nn.ReLU(),
                                                                      epochs=args.epochs,
                                                                      batch_size=args.batch_size,
                                                                      lr=args.lr,
                                                                      lambda_=args.lambda_,
                                                                      dropout_rate=args.dropout_rate,
                                                                      patience=args.patience,
                                                                      device=None))
    # kernel_constructor = KernelConstructor(kernel_level="enhanced_middle", method="",
    #                                        kernel_method=EnhancedMiddleKernel(kernel_type=args.kernel_type,
    #                                                                   latent_dim=args.latent_dim,
    #                                                                   hidden_dims=args.hidden_dims,
    #                                                                     activation=torch.nn.ReLU(),
    #                                                                   epochs=args.epochs,
    #                                                                   batch_size=args.batch_size,
    #                                                                   lr=args.lr,
    #                                                                   lambda_=args.lambda_,
    #                                                                   dropout_rate=args.dropout_rate,
    #                                                                   patience=args.patience,
    #                                                                   device=None))

    # Tạo CrossValidator
    cv = CrossValidator(n_splits=5, n_repeats=1, stratified=False, random_state=1)
    # Tạo các splits
    splits, idxes = DKL.utils.train_test_kernel_cv_split(data_loader, cv, kernel_constructor)
    idx = next(idxes)

    confusion_matrices = []
    classification_reports = []
    accuracies = []

    # Split K folds
    for fold_idx, (kernel_split, feature_split) in enumerate(splits):
        print(f"===== FOLD {fold_idx} =====")
        [Xs_kernel_train, Y_K_train, Xs_kernel_train_test, Xs_kernel_test_test, Y_K_test] = kernel_split
        [Xs_train, Y_train, Xs_test, Y_test] = feature_split

        # Vì chỉ load một modality, Xs_kernel_train / Xs_kernel_train_test / Xs_kernel_test_test
        X_train_kernel = Xs_kernel_train[chosen_modality]         # kernel train
        X_train_test_kernel = Xs_kernel_train_test[chosen_modality] # kernel train-test (cross term)
        X_test_test_kernel = Xs_kernel_test_test[chosen_modality]   # kernel test-test

        clf = DKL.kernels.KernelClassifier()
        clf.fit(X_train_kernel, Y_K_train)
        
        # Dự đoán trên phần kernel tương ứng
        y_pred = clf.predict(X_train_test_kernel)

        # Tính confusion matrix, classification report
        cf = confusion_matrix(Y_test, y_pred)
        report = pd.DataFrame(classification_report(Y_test, y_pred, output_dict=True))

        confusion_matrices.append(cf)
        classification_reports.append(report)

        accuracy = accuracy_score(Y_test, y_pred)
        accuracies.append(accuracy)

        print(cf)
        print(report)

    # Gộp kết quả classification reports
    combined_report = pd.concat(
        classification_reports, 
        keys=[f'Fold_{i}' for i in range(len(classification_reports))]
    )
    
    os.makedirs(f'../../results/{version}', exist_ok=True)
    # combined_report.to_csv(f'../../results/{version}/classification_reports_{chosen_modality}.csv')

    print("====="*5)
    average_accuracy = np.mean(accuracies)
    accuracy_variance = np.std(accuracies)
    print(f"Modality: {chosen_modality}")
    print(f"Average Accuracy: {average_accuracy * 100:.2f}%")
    print(f"Accuracy Std: {accuracy_variance * 100:.2f}%")

    # Save average accuracy and accuracy std to a CSV file
    metrics_df = pd.DataFrame({
        'Modality': [chosen_modality],
        'Average Accuracy': [round(average_accuracy * 100, 2)],
        'Accuracy Std': [round(accuracy_variance * 100, 2)]
    })
    metrics_df.to_csv(f'../../results/{version}/accuracy_metrics_{chosen_modality}.csv', index=False)
