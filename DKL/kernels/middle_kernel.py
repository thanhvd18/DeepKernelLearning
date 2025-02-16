# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.metrics.pairwise import (
#     rbf_kernel,
#     linear_kernel,
#     polynomial_kernel
# )


# def fro_norm(A):
#     if isinstance(A, np.ndarray):
#         A = torch.from_numpy(A)
#     return torch.norm(A, p='fro')

# class Autoencoder(nn.Module):
#     """
#     Mô hình autoencoder cơ bản: 
#       - Encoder: nén dữ liệu đầu vào thành latent vector Z
#       - Decoder: tái tạo lại dữ liệu từ Z
#     """
#     def __init__(self, input_dim, hidden_dims, latent_dim, activation=nn.ReLU()):
#         super(Autoencoder, self).__init__()
#         self.encoder = self._build_encoder(input_dim, hidden_dims, latent_dim, activation)
#         self.decoder = self._build_decoder(input_dim, hidden_dims, latent_dim, activation)

#     def _build_encoder(self, input_dim, hidden_dims, latent_dim, activation):
#         layers = []
#         prev_dim = input_dim
#         for h in hidden_dims:
#             layers.append(nn.Linear(prev_dim, h))
#             layers.append(activation)
#             prev_dim = h
#         layers.append(nn.Linear(prev_dim, latent_dim))
#         return nn.Sequential(*layers)

#     def _build_decoder(self, input_dim, hidden_dims, latent_dim, activation):
#         layers = []
#         prev_dim = latent_dim
#         for h in reversed(hidden_dims):
#             layers.append(nn.Linear(prev_dim, h))
#             layers.append(activation)
#             prev_dim = h
#         layers.append(nn.Linear(prev_dim, input_dim))  # Reconstruction
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         z = self.encode(x)
#         x_hat = self.decode(z)
#         return x_hat

#     def encode(self, x):
#         return self.encoder(x)

#     def decode(self, z):
#         return self.decoder(z)

# def code_loss(Z, P):
#     """
#     Tính code loss theo chuẩn Frobenius (chuẩn hóa hai ma trận).
#     Z: (batch_size, latent_dim) hoặc (n_samples, latent_dim)
#     P: (n_samples, n_samples) - ma trận kernel cho trước
    
#     Return:
#       Một scalar tensor tương ứng với ||C/||C||F - P/||P||F||_F
#       với C = Z * Z^T (kích thước n x n).
#     """
#     # Ma trận C = Z x Z^T
#     C = torch.matmul(Z, Z.t())  # (n x d) * (d x n) = (n x n)


#     C_norm = C / fro_norm(C).clamp_min(1e-12)    # đề phòng chia 0
#     P_norm = P / fro_norm(P).clamp_min(1e-12)   

#     # Code loss = || C_norm - P_norm ||_F
#     return fro_norm(C_norm - P_norm)


# class MiddleKernel:
#     """
#     MiddleKernel sử dụng Autoencoder để trích xuất latent features
#     từ dữ liệu đầu vào X, sau đó tính kernel dựa trên latent features.
#     """

#     def __init__(
#         self,
#         kernel_type="rbf",
#         latent_dim=2000,
#         hidden_dims=[500,500,2000],
#         activation=nn.ReLU(),
#         epochs=50,
#         batch_size=32,
#         lr=1e-3,
#         lambda_=0.75,
#         device=None,
#         **kwargs
#     ):
#         """
#         Args:
#             kernel_type (str): Loại kernel ("rbf", "linear", "poly", ...) 
#                                để tính trên latent features.
#             latent_dim (int): Kích thước của latent vector.
#             hidden_dims (list): Danh sách số neuron trong các hidden layer
#                                 của encoder/decoder.
#             activation (nn.Module): Lớp kích hoạt (mặc định là ReLU).
#             epochs (int): Số epochs để huấn luyện Autoencoder.
#             batch_size (int): Kích thước batch để huấn luyện.
#             lr (float): Tốc độ học (learning rate) cho optimizer.
#             device (torch.device): Thiết bị (cpu hoặc cuda) để train.
#             **kwargs: Tham số bổ sung cho hàm kernel (ví dụ gamma cho RBF, 
#                       degree cho polynomial,...).
#         """
#         self.lambda_ = lambda_
#         self.kernel_type = kernel_type
#         self.latent_dim = latent_dim
#         self.hidden_dims = hidden_dims
#         self.activation = activation
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.lr = lr
#         self.params = kwargs  # Tham số cho hàm kernel (vd: gamma, degree,...)

#         # Nếu device=None thì mặc định tự chọn
#         self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         self.model = None
#         self.input_dim = None

#         # Khởi tạo Autoencoder
#         # self.model = Autoencoder(input_dim, hidden_dims, latent_dim, activation).to(self.device)
        
#         # # Loss MSE cho reconstruction
#         # self.recon_loss_fn = nn.MSELoss()
        
#         # # Optimizer
#         # self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

#     def fit(self, X, y=None):
#         """
#         Huấn luyện Autoencoder trên dữ liệu X.

#         Args:
#             X (np.ndarray or torch.Tensor): Dữ liệu đầu vào, shape (num_samples, num_features).
#             y (np.ndarray): Nhãn (nếu cần cho bài toán supervised). 
#                             Với autoencoder ta thường không dùng y.
#             flag_train (bool): True if training, False if testing
#         """
#         if isinstance(X, np.ndarray):
#             X = torch.from_numpy(X).float()
        
#         # Prior knowledge: kernel matrix P
#         P = rbf_kernel(X, X, gamma=self.params.get("gamma", 1.0))

#         self.input_dim = X.shape[1]

#         # Tạo dataset và dataloader cho quá trình train
#         dataset = TensorDataset(X, X)  # Tự giám sát, input = target
#         dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

#         # Khởi tạo Autoencoder
#         self.model = Autoencoder(
#             input_dim=self.input_dim,
#             hidden_dims=self.hidden_dims,
#             latent_dim=self.latent_dim,
#             activation=self.activation
#         ).to(self.device)

#         # Định nghĩa loss và optimizer
#         criterion = nn.MSELoss()
#         optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

#         print("Start training... for creating latent features")
#         # Vòng lặp huấn luyện
#         self.model.train()
#         for epoch in range(self.epochs):
#             epoch_loss = 0.0
#             for batch_X, _ in dataloader:
#                 batch_X = batch_X.to(self.device)
#                 # Forward
#                 X_recon = self.model(batch_X)
#                 loss = criterion(X_recon, batch_X)

#                 # Để tính code_loss, ta cần encode hết X (full dataset) => tốn compute.
#                 # Thực tế có thể tối ưu nếu dataset lớn.
#                 Z = self.model.encode(X)  # encode toàn bộ X
#                 c_loss = code_loss(Z, P)

#                 # Tổng loss = MSE loss + code loss
#                 total_loss = (1 - self.lambda_) * loss + self.lambda_ * c_loss

#                 # Backward
#                 optimizer.zero_grad()
#                 total_loss.backward()
#                 optimizer.step()

#                 epoch_loss += total_loss.item() * batch_X.size(0)

#             epoch_loss /= len(dataloader.dataset)
#             # tiến trình training
#             print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}")
               
#         """
#         Lấy latent features từ encoder.

#         Args:
#             X (np.ndarray or torch.Tensor): Dữ liệu đầu vào, shape (num_samples, num_features).

#         Returns:
#             np.ndarray: Latent features, shape (num_samples, latent_dim).
#         """
#         print("Finish training... for creating latent features")
        
#         if isinstance(X, np.ndarray):
#             X = torch.from_numpy(X).float()
#         self.model.eval()
#         with torch.no_grad():
#             X = X.to(self.device)
#             Z = self.model.encode(X)
        
#         self.X_fit = Z.cpu().numpy()
#         return self

#     def transform(self, X):
        
#         """
#         Transform the input data to kernel space using the fitted data.

#         Args:
#             X (np.ndarray): Feature matrix to transform.

#         Returns:
#             np.ndarray: Kernel matrix between the fitted data and the input data.

#         Raises:
#             ValueError: If the model has not been fitted.
#         """
#         if isinstance(X, np.ndarray):
#             X = torch.from_numpy(X).float()
#         self.model.eval()
#         with torch.no_grad():
#             X = X.to(self.device)
#             X_latent = self.model.encode(X)


#         if self.X_fit is None:
#             raise ValueError("The kernel model must be fitted before calling transform.")
#         return self.get_kernel(self.X_fit, X_latent), self.get_kernel(X_latent, X_latent)
        
    
#     def fit_transform(self, X, y=None):
#         """
#         Huấn luyện Autoencoder và trả về latent features.

#         Args:
#             X (np.ndarray or torch.Tensor): Dữ liệu đầu vào, shape (num_samples, num_features).

#         Returns:
#             np.ndarray: Latent features, shape (num_samples, latent_dim).
#         """
#         self.fit(X, y)
#         return self.transform(X)

#     def get_kernel(self, X1, X2):
#         """
#         Tính kernel trên latent features thay vì raw features.

#         Args:
#             X (np.ndarray or torch.Tensor): Dữ liệu đầu vào, shape (num_samples, num_features).

#         Returns:
#             np.ndarray: Ma trận kernel (num_samples, num_samples).
#         """
#         # Bước 1: Biến đổi X thành latent features
#         # X_latent = self.transform(X)
#         # Bước 2: Tính kernel dựa trên X_latent
#         if self.kernel_type == "rbf":
#             # rbf_kernel có thêm tham số gamma
#             gamma = self.params.get("gamma", None)
#             return rbf_kernel(X1, X2, gamma=gamma)
#         elif self.kernel_type == "linear":
#             return linear_kernel(X1, X2.T)
#         elif self.kernel_type == "poly":
#             degree = self.params.get("degree", 3)
#             coef0 = self.params.get("coef0", 1)
#             gamma = self.params.get("gamma", None)
#             return polynomial_kernel(
#                 X1, X2.T, 
#                 degree=degree, 
#                 gamma=gamma, 
#                 coef0=coef0
#             )
#         else:
#             raise ValueError(f"Unsupported kernel_type: {self.kernel_type}")

#     def __str__(self):
#         return (
#             f"MiddleKernel(kernel_type={self.kernel_type}, "
#             f"latent_dim={self.latent_dim}, hidden_dims={self.hidden_dims}, "
#             f"params={self.params})"
#         )


# if __name__ == "__main__":
#     # Ví dụ sử dụng MiddleKernel

#     # Giả sử có dữ liệu X ngẫu nhiên
#     X = np.random.rand(100, 20).astype(np.float32)  # 100 samples, 20 features

#     # Tạo kernel object
#     mk = MiddleKernel(
#         kernel_type="rbf",
#         latent_dim=8,
#         hidden_dims=[32, 16],
#         activation=nn.ReLU(),
#         epochs=5,
#         batch_size=16,
#         lr=1e-3,
#         gamma=0.5  # tham số cho rbf_kernel
#     )

#     # Huấn luyện autoencoder
#     mk.fit(X)

#     # Trích xuất latent features
#     Z = mk.transform(X)
#     print("Latent shape:", Z.shape)  # (100, 8)

#     # Tính kernel trên latent features
#     K = mk.get_kernel(X)
#     print("Kernel shape:", K.shape)  # (100, 100)
#     print("Kernel sample:\n", K[:5, :5])

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, polynomial_kernel

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def fro_norm(A):
    if isinstance(A, np.ndarray):
        A = torch.from_numpy(A)
    return torch.norm(A, p='fro')

class Autoencoder(nn.Module):
    """
    Mô hình autoencoder cơ bản:
      - Encoder: nén dữ liệu đầu vào thành latent vector Z
      - Decoder: tái tạo lại dữ liệu từ Z
    """
    def __init__(self, input_dim, hidden_dims, latent_dim, activation=nn.ReLU(), dropout_rate=0.2):
        super(Autoencoder, self).__init__()
        self.encoder = self._build_encoder(input_dim, hidden_dims, latent_dim, activation, dropout_rate)
        self.decoder = self._build_decoder(input_dim, hidden_dims, latent_dim, activation, dropout_rate)

    def _build_encoder(self, input_dim, hidden_dims, latent_dim, activation, dropout_rate):
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation)
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, latent_dim))
        return nn.Sequential(*layers)

    def _build_decoder(self, input_dim, hidden_dims, latent_dim, activation, dropout_rate):
        layers = []
        prev_dim = latent_dim
        for h in reversed(hidden_dims):
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation)
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, input_dim))
        return nn.Sequential(*layers)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

def code_loss(Z, P):
    """
    Tính code loss theo chuẩn Frobenius.
    Z: (batch_size, latent_dim) hoặc (n_samples, latent_dim)
    P: (n_samples, n_samples) - ma trận kernel cho trước
    Return:
      Một scalar tensor tương ứng với ||C/||C||F - P/||P||F||_F
      với C = Z * Z^T.
    """
    C = torch.matmul(Z, Z.t())
    C_norm = C / fro_norm(C).clamp_min(1e-12)
    P_norm = P / fro_norm(P).clamp_min(1e-12)
    return fro_norm(C_norm - P_norm)

class MiddleKernel:
    """
    Sử dụng Autoencoder để trích xuất latent features và tính kernel dựa trên latent features.
    """
    def __init__(
        self,
        kernel_type="rbf",
        latent_dim=2000,
        hidden_dims=[1024,1024,2000],
        activation=nn.ReLU(),
        epochs=100,
        batch_size=32,
        lr=1e-3,
        lambda_=0.75,
        dropout_rate=0.2,
        patience=10,  # Số epoch không cải thiện trước khi dừng sớm
        device=None,
        **kwargs
    ):
        self.lambda_ = lambda_
        self.kernel_type = kernel_type
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dropout_rate = dropout_rate
        self.params = kwargs
        self.patience = patience

        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.input_dim = None
        self.X_fit = None

    def fit(self, X, y=None):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        
        # Tính prior kernel matrix P sử dụng rbf
        P_np = rbf_kernel(X, X, gamma=self.params.get("gamma", 1.0))
        P = torch.from_numpy(P_np).float().to(self.device)
        self.input_dim = X.shape[1]

        dataset = TensorDataset(X, X)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = Autoencoder(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim,
            activation=self.activation,
            dropout_rate=self.dropout_rate
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

        best_loss = float('inf')
        trigger = 0

        print("Start training... for creating latent features")
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, _ in dataloader:
                batch_X = batch_X.to(self.device)
                optimizer.zero_grad()
                X_recon = self.model(batch_X)
                recon_loss = criterion(X_recon, batch_X)
                
                # Tính code loss trên toàn bộ dữ liệu (có thể tính theo batch nếu dữ liệu lớn)
                Z = self.model.encode(X.to(self.device))
                c_loss = code_loss(Z, P)

                total_loss = (1 - self.lambda_) * recon_loss + self.lambda_ * c_loss
                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item() * batch_X.size(0)
            
            epoch_loss /= len(dataloader.dataset)
            scheduler.step(epoch_loss)
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}")

            # Early Stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_state = self.model.state_dict()
                trigger = 0
            else:
                trigger += 1
                if trigger >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model weights
        self.model.load_state_dict(best_model_state)
        print("Finish training... for creating latent features")
        
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            Z = self.model.encode(X)
        self.X_fit = Z.cpu().numpy()
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            X_latent = self.model.encode(X)
        if self.X_fit is None:
            raise ValueError("The kernel model must be fitted before calling transform.")
        return self.get_kernel(self.X_fit, X_latent.cpu().numpy()), self.get_kernel(X_latent.cpu().numpy(), X_latent.cpu().numpy())
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def get_kernel(self, X1, X2):
        if self.kernel_type == "rbf":
            gamma = self.params.get("gamma", None)
            return rbf_kernel(X1, X2, gamma=gamma)
        elif self.kernel_type == "linear":
            return linear_kernel(X1, X2)
        elif self.kernel_type == "poly":
            degree = self.params.get("degree", 3)
            coef0 = self.params.get("coef0", 1)
            gamma = self.params.get("gamma", None)
            return polynomial_kernel(X1, X2, degree=degree, gamma=gamma, coef0=coef0)
        else:
            raise ValueError(f"Unsupported kernel_type: {self.kernel_type}")

    def __str__(self):
        return (f"MiddleKernel(kernel_type={self.kernel_type}, latent_dim={self.latent_dim}, "
                f"hidden_dims={self.hidden_dims}, params={self.params})")

if __name__ == "__main__":
    # Ví dụ sử dụng MiddleKernel với dữ liệu ngẫu nhiên
    X = np.random.rand(100, 20).astype(np.float32)

    mk = MiddleKernel(
        kernel_type="rbf",
        latent_dim=32,
        hidden_dims=[256,128,64],
        activation=nn.ReLU(),
        epochs=50,
        batch_size=16,
        lr=1e-3,
        lambda_=0.75,
        dropout_rate=0.2,
        gamma=0.5,
        patience=10
    )

    mk.fit(X)
    latent_kernel, self_kernel = mk.transform(X)
    print("Latent kernel shape:", latent_kernel.shape)
    print("Self kernel shape:", self_kernel.shape)
