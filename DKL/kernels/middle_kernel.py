import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics.pairwise import (
    rbf_kernel,
    linear_kernel,
    polynomial_kernel
)


class Autoencoder(nn.Module):
    """
    Mô hình Autoencoder cơ bản gồm 2 phần:
    - Encoder: nén dữ liệu đầu vào thành latent vector.
    - Decoder: tái tạo lại dữ liệu từ latent vector.
    """

    def __init__(self, input_dim, hidden_dims, latent_dim, activation=nn.ReLU()):
        """
        Args:
            input_dim (int): Số chiều của dữ liệu đầu vào.
            hidden_dims (list): Danh sách kích thước các hidden layer.
            latent_dim (int): Số chiều của latent vector.
            activation (nn.Module): Lớp kích hoạt (ReLU, LeakyReLU, ...).
        """
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.X_fit = None

        # ---------------------
        # Xây dựng encoder
        # ---------------------
        encoder_layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            encoder_layers.append(activation)
            prev_dim = dim
        # Latent layer
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        encoder_layers.append(activation)

        self.encoder = nn.Sequential(*encoder_layers)

        # ---------------------
        # Xây dựng decoder
        # ---------------------
        decoder_layers = []
        prev_dim = latent_dim
        # Duyệt ngược hidden_dims
        for dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, dim))
            decoder_layers.append(activation)
            prev_dim = dim
        # Reconstruction layer
        decoder_layers.append(nn.Linear(prev_dim, input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """
        Truyền dữ liệu qua Encoder để được latent vector,
        sau đó Decoder để tái tạo lại dữ liệu.
        """
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

    def encode(self, x):
        """
        Chỉ lấy phần Encoder: nén dữ liệu đầu vào thành latent vector.
        """
        return self.encoder(x)

    def decode(self, z):
        """
        Chỉ lấy phần Decoder: khôi phục dữ liệu từ latent vector.
        """
        return self.decoder(z)


class MiddleKernel:
    """
    MiddleKernel sử dụng Autoencoder để trích xuất latent features
    từ dữ liệu đầu vào X, sau đó tính kernel dựa trên latent features.
    """

    def __init__(
        self,
        kernel_type="rbf",
        latent_dim=32,
        hidden_dims=[256,128,64],
        activation=nn.ReLU(),
        epochs=50,
        batch_size=32,
        lr=1e-3,
        device=None,
        **kwargs
    ):
        """
        Args:
            kernel_type (str): Loại kernel ("rbf", "linear", "poly", ...) 
                               để tính trên latent features.
            latent_dim (int): Kích thước của latent vector.
            hidden_dims (list): Danh sách số neuron trong các hidden layer
                                của encoder/decoder.
            activation (nn.Module): Lớp kích hoạt (mặc định là ReLU).
            epochs (int): Số epochs để huấn luyện Autoencoder.
            batch_size (int): Kích thước batch để huấn luyện.
            lr (float): Tốc độ học (learning rate) cho optimizer.
            device (torch.device): Thiết bị (cpu hoặc cuda) để train.
            **kwargs: Tham số bổ sung cho hàm kernel (ví dụ gamma cho RBF, 
                      degree cho polynomial,...).
        """
        self.kernel_type = kernel_type
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.params = kwargs  # Tham số cho hàm kernel (vd: gamma, degree,...)

        # Nếu device=None thì mặc định tự chọn
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.input_dim = None

    def fit(self, X, y=None):
        """
        Huấn luyện Autoencoder trên dữ liệu X.

        Args:
            X (np.ndarray or torch.Tensor): Dữ liệu đầu vào, shape (num_samples, num_features).
            y (np.ndarray): Nhãn (nếu cần cho bài toán supervised). 
                            Với autoencoder ta thường không dùng y.
            flag_train (bool): True if training, False if testing
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()

        self.input_dim = X.shape[1]

        # Tạo dataset và dataloader cho quá trình train
        dataset = TensorDataset(X, X)  # Tự giám sát, input = target
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Khởi tạo Autoencoder
        self.model = Autoencoder(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim,
            activation=self.activation
        ).to(self.device)

        # Định nghĩa loss và optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        print("Start training... for creating latent features")
        # Vòng lặp huấn luyện
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, _ in dataloader:
                batch_X = batch_X.to(self.device)
                # Forward
                X_recon = self.model(batch_X)
                loss = criterion(X_recon, batch_X)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_X.size(0)

            epoch_loss /= len(dataloader.dataset)
            # tiến trình training
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}")
               
        """
        Lấy latent features từ encoder.

        Args:
            X (np.ndarray or torch.Tensor): Dữ liệu đầu vào, shape (num_samples, num_features).

        Returns:
            np.ndarray: Latent features, shape (num_samples, latent_dim).
        """
        print("Finish training... for creating latent features")
        
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            Z = self.model.encode(X)
        
        self.X_fit = Z.cpu().numpy()
        return self

    def transform(self, X):
        
        """
        Transform the input data to kernel space using the fitted data.

        Args:
            X (np.ndarray): Feature matrix to transform.

        Returns:
            np.ndarray: Kernel matrix between the fitted data and the input data.

        Raises:
            ValueError: If the model has not been fitted.
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            X_latent = self.model.encode(X)


        if self.X_fit is None:
            raise ValueError("The kernel model must be fitted before calling transform.")
        return self.get_kernel(self.X_fit, X_latent), self.get_kernel(X_latent, X_latent)
        
    
    def fit_transform(self, X, y=None):
        """
        Huấn luyện Autoencoder và trả về latent features.

        Args:
            X (np.ndarray or torch.Tensor): Dữ liệu đầu vào, shape (num_samples, num_features).

        Returns:
            np.ndarray: Latent features, shape (num_samples, latent_dim).
        """
        self.fit(X, y)
        return self.transform(X)

    def get_kernel(self, X1, X2):
        """
        Tính kernel trên latent features thay vì raw features.

        Args:
            X (np.ndarray or torch.Tensor): Dữ liệu đầu vào, shape (num_samples, num_features).

        Returns:
            np.ndarray: Ma trận kernel (num_samples, num_samples).
        """
        # Bước 1: Biến đổi X thành latent features
        # X_latent = self.transform(X)
        # Bước 2: Tính kernel dựa trên X_latent
        if self.kernel_type == "rbf":
            # rbf_kernel có thêm tham số gamma
            gamma = self.params.get("gamma", None)
            return rbf_kernel(X1, X2, gamma=gamma)
        elif self.kernel_type == "linear":
            return linear_kernel(X1, X2.T)
        elif self.kernel_type == "poly":
            degree = self.params.get("degree", 3)
            coef0 = self.params.get("coef0", 1)
            gamma = self.params.get("gamma", None)
            return polynomial_kernel(
                X1, X2.T, 
                degree=degree, 
                gamma=gamma, 
                coef0=coef0
            )
        else:
            raise ValueError(f"Unsupported kernel_type: {self.kernel_type}")

    def __str__(self):
        return (
            f"MiddleKernel(kernel_type={self.kernel_type}, "
            f"latent_dim={self.latent_dim}, hidden_dims={self.hidden_dims}, "
            f"params={self.params})"
        )


if __name__ == "__main__":
    # Ví dụ sử dụng MiddleKernel

    # Giả sử có dữ liệu X ngẫu nhiên
    X = np.random.rand(100, 20).astype(np.float32)  # 100 samples, 20 features

    # Tạo kernel object
    mk = MiddleKernel(
        kernel_type="rbf",
        latent_dim=8,
        hidden_dims=[32, 16],
        activation=nn.ReLU(),
        epochs=5,
        batch_size=16,
        lr=1e-3,
        gamma=0.5  # tham số cho rbf_kernel
    )

    # Huấn luyện autoencoder
    mk.fit(X)

    # Trích xuất latent features
    Z = mk.transform(X)
    print("Latent shape:", Z.shape)  # (100, 8)

    # Tính kernel trên latent features
    K = mk.get_kernel(X)
    print("Kernel shape:", K.shape)  # (100, 100)
    print("Kernel sample:\n", K[:5, :5])
