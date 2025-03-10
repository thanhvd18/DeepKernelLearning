import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, polynomial_kernel
from cvxopt import matrix, solvers, mul

# --------------------- Helper Functions --------------------- #
def set_seed(seed=42):
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

# --------------------- EasyMKL Class --------------------- #
class EasyMKL():
    '''
    EasyMKL is a scalable multiple kernel learning algorithm.
    The parameter lam (lambda) must be in the range [0,1].
    '''
    def __init__(self, lam=0.1, tracenorm=True):
        self.lam = lam
        self.tracenorm = tracenorm
        self.list_Ktr = None
        self.labels = None
        self.gamma = None
        self.weights = None
        self.traces = []

    def sum_kernels(self, list_K, weights=None):
        ''' Returns the kernel created by averaging all the kernels '''
        k = matrix(0.0, (list_K[0].size[0], list_K[0].size[1]))
        if weights is None:
            for ker in list_K:
                k += ker
        else:
            for w, ker in zip(weights, list_K):
                k += w * ker            
        return k
    
    def traceN(self, k):
        return sum([k[i, i] for i in range(k.size[0])]) / k.size[0]
    
    def train(self, list_Ktr, labels):
        ''' 
        list_Ktr : list of kernel matrices (training examples)
        labels   : training labels (must be binary: -1 and +1)
        '''
        self.list_Ktr = list_Ktr  
        for k in self.list_Ktr:
            self.traces.append(self.traceN(k))
        if self.tracenorm:
            self.list_Ktr = [k / self.traceN(k) for k in list_Ktr]

        set_labels = set(labels)
        if len(set_labels) != 2:
            raise ValueError('Number of distinct labels must be 2.')
        elif (-1 in set_labels and 1 in set_labels):
            self.labels = matrix(np.array(labels, dtype=float))
        else:
            poslab = max(set_labels)
            self.labels = matrix(np.array([1.0 if i==poslab else -1.0 for i in labels]))
        
        # Sum of kernels
        ker_matrix = matrix(self.sum_kernels(self.list_Ktr))

        YY = matrix(np.diag(list(matrix(self.labels))))
        KLL = (1.0 - self.lam) * YY * ker_matrix * YY
        LID = matrix(np.diag([self.lam] * len(self.labels)))
        Q = 2 * (KLL + LID)
        p = matrix([0.0] * len(self.labels))
        G = -matrix(np.diag([1.0] * len(self.labels)))
        h = matrix([0.0] * len(self.labels), (len(self.labels), 1))
        A = matrix([[1.0 if lab == +1 else 0 for lab in self.labels],
                    [1.0 if lab == -1 else 0 for lab in self.labels]]).T
        b = matrix([[1.0], [1.0]], (2, 1))
        
        solvers.options['show_progress'] = False
        sol = solvers.qp(Q, p, G, h, A, b)
        self.gamma = sol['x']     
        
        # Evaluate weights:
        yg = mul(self.gamma.T, self.labels.T)
        self.weights = []
        for kermat in self.list_Ktr:
            b_val = yg * kermat * yg.T
            self.weights.append(b_val[0])
            
        norm2 = sum(self.weights)
        self.weights = [w / norm2 for w in self.weights]

        if self.tracenorm: 
            for idx, val in enumerate(self.traces):
                self.weights[idx] = self.weights[idx] / val        
        
        # Refine gamma using the combined kernel
        ker_matrix = matrix(self.sum_kernels(self.list_Ktr, self.weights))
        YY = matrix(np.diag(list(matrix(self.labels))))
        KLL = (1.0 - self.lam) * YY * ker_matrix * YY
        LID = matrix(np.diag([self.lam] * len(self.labels)))
        Q = 2 * (KLL + LID)
        p = matrix([0.0] * len(self.labels))
        G = -matrix(np.diag([1.0] * len(self.labels)))
        h = matrix([0.0] * len(self.labels), (len(self.labels), 1))
        A = matrix([[1.0 if lab == +1 else 0 for lab in self.labels],
                    [1.0 if lab == -1 else 0 for lab in self.labels]]).T
        b = matrix([[1.0], [1.0]], (2, 1))
        
        solvers.options['show_progress'] = False
        sol = solvers.qp(Q, p, G, h, A, b)
        self.gamma = sol['x']
        
        return self
    
    def rank(self, list_Ktest):
        '''
        list_Ktest : list of kernel matrices for the test examples.
        Returns a ranking vector computed on the test kernel.
        '''
        if self.weights is None:
            raise ValueError('EasyMKL has to be trained first!')
        YY = matrix(np.diag(list(matrix(self.labels))))
        ker_matrix = matrix(self.sum_kernels(list_Ktest, self.weights))
        z = ker_matrix * YY * self.gamma
        return z

# --------------------- Autoencoder Class --------------------- #
class Autoencoder(nn.Module):
    """
    A basic autoencoder:
      - Encoder: compresses input data into a latent vector.
      - Decoder: reconstructs the data from the latent vector.
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
    Computes the code loss based on the Frobenius norm of the difference between the normalized
    matrices C = Z * Z^T and the given prior kernel matrix P.
    """
    C = torch.matmul(Z, Z.t())
    C_norm = C / fro_norm(C).clamp_min(1e-12)
    P_norm = P / fro_norm(P).clamp_min(1e-12)
    return fro_norm(C_norm - P_norm)

# --------------------- EnhancedMiddleKernel with Prior Code Loss --------------------- #
class EnhancedMiddleKernel:
    """
    EnhancedMiddleKernel trains an autoencoder to extract latent features from the input data.
    It then computes three prior kernel matrices (RBF, linear, polynomial) on the raw data to serve
    as prior knowledge for the code loss. The individual code losses are combined using weights learned
    via EasyMKL. The overall loss is a combination of the reconstruction loss and the weighted code loss.
    """
    def __init__(self,
                 latent_dim=2000,
                 hidden_dims=[500, 500, 2000],
                 activation=nn.ReLU(),
                 epochs=100,
                 batch_size=32,
                 lr=1e-3,
                 lambda_=0.75,
                 dropout_rate=0.2,
                 patience=10,
                 device=None,
                 kernel_params=None,
                 **kwargs):
        """
        Args:
            latent_dim (int): Size of the latent vector.
            hidden_dims (list): Hidden layer sizes for the autoencoder.
            activation (nn.Module): Activation function.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            lr (float): Learning rate.
            lambda_ (float): Weighting factor between reconstruction loss and code loss.
            dropout_rate (float): Dropout rate in the autoencoder.
            patience (int): Patience for early stopping.
            device (torch.device): Device for training.
            kernel_params (dict): Parameters for kernel functions (e.g., gamma for RBF).
        """
        self.kernel_type = "rbf"
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lambda_ = lambda_
        self.dropout_rate = dropout_rate
        self.patience = patience
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.kernel_params = kernel_params if kernel_params is not None else {}
        self.params = kwargs
        self.model = None
        self.input_dim = None
        self.X_fit = None
        self.easy_mkl = None  # For learning the weights for combining prior code losses

    def fit(self, X, y):
        """
        Train the autoencoder and learn the kernel combination weights for the prior code loss.
        
        Args:
            X (np.ndarray or torch.Tensor): Input data.
            y (array-like): Binary labels (-1 and +1) for the training data.
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        
        self.input_dim = X.shape[1]
        
        # ---------------------- Compute Prior Kernel Matrices on Raw Data ---------------------- #
        gamma = self.kernel_params.get("gamma", 1.0)
        degree = self.kernel_params.get("degree", 3)
        coef0 = self.kernel_params.get("coef0", 1)
        
        # RBF prior kernel
        P_rbf_np = rbf_kernel(X, X, gamma=gamma)
        # Linear prior kernel
        P_linear_np = linear_kernel(X, X)
        # Polynomial prior kernel
        P_poly_np = polynomial_kernel(X, X, degree=degree, gamma=gamma, coef0=coef0)
        
        # Convert to torch tensors (for code loss computation)
        P_rbf = torch.from_numpy(P_rbf_np).float().to(self.device)
        P_linear = torch.from_numpy(P_linear_np).float().to(self.device)
        P_poly = torch.from_numpy(P_poly_np).float().to(self.device)
        
        # ---------------------- Learn Weights via EasyMKL on the Prior Kernels ---------------------- #
        # prior_kernels = [matrix(P_rbf_np), matrix(P_linear_np), matrix(P_poly_np)]
        prior_kernels = [
            matrix(P_rbf_np.astype(np.float64)),
            matrix(P_linear_np.astype(np.float64)),
            matrix(P_poly_np.astype(np.float64))
        ]
        self.easy_mkl = EasyMKL(lam=0.1, tracenorm=True)
        self.easy_mkl.train(prior_kernels, y)
        # The learned weights for the three prior kernels
        weights = self.easy_mkl.weights
        
        # ---------------------- Prepare DataLoader for Autoencoder Training ---------------------- #
        dataset = TensorDataset(X, X)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize the autoencoder
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

        print("Start training autoencoder for latent feature extraction...")
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, _ in dataloader:
                batch_X = batch_X.to(self.device)
                optimizer.zero_grad()
                X_recon = self.model(batch_X)
                recon_loss = criterion(X_recon, batch_X)
                
                # Compute code loss on the entire training set (using the current autoencoder)
                self.model.eval()
                with torch.no_grad():
                    X_full = X.to(self.device)
                Z_full = self.model.encode(X_full)
                self.model.train()
                
                # Compute code losses for each prior kernel
                c_loss_rbf = code_loss(Z_full, P_rbf)
                c_loss_linear = code_loss(Z_full, P_linear)
                c_loss_poly = code_loss(Z_full, P_poly)
                # Combine the code losses using EasyMKL weights
                combined_code_loss = weights[0] * c_loss_rbf + weights[1] * c_loss_linear + weights[2] * c_loss_poly
                
                total_loss = (1 - self.lambda_) * recon_loss + self.lambda_ * combined_code_loss
                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item() * batch_X.size(0)
            
            epoch_loss /= len(dataloader.dataset)
            scheduler.step(epoch_loss)
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}")
            
            # Early stopping
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
        print("Finished training autoencoder.")
        
        # Extract latent features on the full training set
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
    
    def fit_transform(self, X, y):
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
        return (f"EnhancedMiddleKernel(latent_dim={self.latent_dim}, hidden_dims={self.hidden_dims}, "
                f"epochs={self.epochs}, lr={self.lr}, lambda_={self.lambda_}, dropout_rate={self.dropout_rate})")

# --------------------- Main Example --------------------- #
if __name__ == "__main__":
    # Example usage with random data.
    X = np.random.rand(100, 20).astype(np.float32)  # 100 samples, 20 features
    # Generate random binary labels (-1 and +1)
    y = [1 if i < 50 else -1 for i in range(100)]
    
    emk = EnhancedMiddleKernel(
        latent_dim=2000,
        hidden_dims=[500, 500, 2000],
        activation=nn.ReLU(),
        epochs=50,
        batch_size=16,
        lr=1e-3,
        lambda_=0.75,
        dropout_rate=0.2,
        patience=10,
        kernel_params={"gamma": 0.5, "degree": 3, "coef0": 1}
    )
    
    emk.fit(X, y)
    latent_features = emk.transform(X)
    print("Latent features shape:", latent_features.shape)
