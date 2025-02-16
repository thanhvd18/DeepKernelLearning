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

if __name__ == '__main__':
    """
    Ví dụ chạy:
        python main.py --modality PET
        python main.py --modality GM
        python main.py --modality MRI
    """
    
    # Khai báo argparse
    parser = argparse.ArgumentParser(description="Deep Kernel Learning Experiment")
    parser.add_argument(
        "--modality", 
        type=str, 
        default="PET", 
        choices=["PET", "GM", "CSF", "concat"],
        help="Modality bạn muốn chạy thí nghiệm: PET, GM, CSF, hoặc concat."
    )
    args = parser.parse_args()
    
    # Lấy modality được chọn
    chosen_modality = args.modality
    
    data_dir = os.path.join(os.getcwd(), "..", "..", "data/AD_CN")
    
    # Mặc định vẫn khai báo tất cả file CSV
    data_config = {
        "GM": os.path.join(data_dir, "GM.csv"),
        "PET": os.path.join(data_dir, "PET.csv"),
        "CSF": os.path.join(data_dir, "CSF.csv"),
        "concat": os.path.join(data_dir, "concat.csv"),
        "label": os.path.join(data_dir, "AD_CN_label.csv")
    }
    
    # Để chỉ load đúng modality mà bạn muốn (và label), 
    # bạn có thể filter lại data_config, bỏ những modality không dùng.
    # Nếu muốn chạy ghép nhiều modality, bạn có thể tự điều chỉnh logic này.
    keep_keys = [chosen_modality, "label"]
    data_config = {k: v for k, v in data_config.items() if k in keep_keys}
    
    data_loader = DataLoader(data_config)

    # Bạn vẫn giữ nguyên kernel_level, method,... như ban đầu
    kernel_level = "enhancedMiddle" # kernel_levels = ["early", "middle", "late", "enhancedMiddle"]
    kernel_constructor = KernelConstructor(kernel_level, method="linear")

    # Tạo CrossValidator
    cv = CrossValidator(n_splits=5, n_repeats=1, stratified=True, random_state=1)

    # Tạo các splits
    splits = DKL.utils.train_test_kernel_cv_split(data_loader, cv, kernel_constructor)

    confusion_matrices = []
    classification_reports = []
    accuracies = []

    # Chạy vòng lặp huấn luyện và đánh giá
    for fold_idx, (kernel_split, feature_split) in enumerate(splits):
    
        [Xs_kernel_train, Y_K_train, Xs_kernel_train_test, Xs_kernel_test_test, Y_K_test] = kernel_split
        [Xs_train, Y_train, Xs_test, Y_test] = feature_split

        # Vì chỉ load một modality, Xs_kernel_train / Xs_kernel_train_test / Xs_kernel_test_test
        # sẽ là dict có đúng 1 key = chosen_modality. Ta có thể trích xuất ra như sau:
        X_train_kernel = Xs_kernel_train[chosen_modality]         # kernel train
        X_train_test_kernel = Xs_kernel_train_test[chosen_modality] # kernel train-test (cross term)
        X_test_test_kernel = Xs_kernel_test_test[chosen_modality]   # kernel test-test

        print("====="*5)
        print("X_train_kernel shape: ", X_train_kernel.shape)
        print("X_train_test_kernel shape:", X_train_test_kernel.shape)
        print("X_test_test_kernel shape:", X_test_test_kernel.shape)
        print("====="*5)

        # Ví dụ dùng KernelClassifier đơn giản
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

        print(f"===== FOLD {fold_idx} =====")
        print(cf)
        print(report)
        print("Done!\n")

    # Gộp kết quả classification reports
    combined_report = pd.concat(
        classification_reports, 
        keys=[f'Fold_{i}' for i in range(len(classification_reports))]
    )
    
    # Lưu lại vào file CSV nếu muốn
    # (Ở đây thay đường dẫn theo ý bạn)
    combined_report.to_csv(f'/Users/macbook/Documents/WorkSpace/DeepKernelLearning/results/version_9/classification_reports_{chosen_modality}.csv')

    print("====="*5)
    average_accuracy = np.mean(accuracies)
    accuracy_variance = np.std(accuracies)
    print(f"Modality: {chosen_modality}")
    print(f"Average Accuracy: {average_accuracy * 100:.2f}%")
    print(f"Accuracy Std: {accuracy_variance * 100:.2f}%")

    # Save average accuracy and accuracy std to a CSV file
    metrics_df = pd.DataFrame({
        'Modality': [chosen_modality],
        'Average Accuracy': [average_accuracy * 100],
        'Accuracy Std': [accuracy_variance * 100]
    })
    metrics_df.to_csv(f'/Users/macbook/Documents/WorkSpace/DeepKernelLearning/results/version_9/accuracy_metrics_{chosen_modality}.csv', index=False)
