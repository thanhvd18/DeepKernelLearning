import numpy as np
from .model import ConvolutionalGatingMLP, DeepCNN, DeepCNN1
from ..loss import my_loss
import torch
import torch.nn.functional as F

class DeepKernelLearning(DeepCNN1):
    def __init__(self,n_kernel=5,kernel_size = 1, n_layer=8):
        super().__init__(n_kernel, kernel_size, n_layer)
        self.model = None
        self.kernel = None


    def fit(self, KX_train, KY_train, KX_train_test, KX_test_test):
        # Combine the kernel matrices
        assert KX_train.shape[0] == KX_train_test.shape[0] == KX_test_test.shape[0], "Number of modalities must match."

        num_modalities = KX_train.shape[0]
        train_size = KX_train.shape[1]
        self.train_size = train_size
        test_size = KX_test_test.shape[1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        combine_K = np.zeros((num_modalities, train_size + test_size, train_size + test_size))


        for i in range(num_modalities):
            combine_K[i, :train_size, :train_size] = KX_train[i]  # Train x Train
            combine_K[i, :train_size, train_size:] = KX_train_test[i]  # Train x Test
            combine_K[i, train_size:, :train_size] = KX_train_test[i].T  # Test x Train
            combine_K[i, train_size:, train_size:] = KX_test_test[i]  # Test x Test

        # convert numpy to torch
        self.combine_K = torch.tensor(combine_K, dtype=torch.float32).to(device)

        # # normalize the kernel matrix 
       

        KY_train = torch.tensor(KY_train, dtype=torch.float32).to(device)
    # Handle label mask matrix
        combine_y_mask = np.full((train_size + test_size, train_size + test_size), None, dtype=object)
        combine_y_mask[:train_size, :train_size] = KY_train

        self.model = DeepCNN1(n_kernel=3,kernel_size = 1, n_layer=3).to(device)
        # self.model = ConvolutionalGatingMLP(num_kernels=num_modalities, kernel_size=train_size+test_size).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)
        epochs = 50
        for epoch in range(epochs):
            print("Epoch: ", epoch)
            self.model.train()
            outputs = self.model(self.combine_K)
            # print("before",outputs.shape)
            outputs = outputs[:,:train_size, :train_size]
            # print("after",outputs.shape)
            loss = my_loss(KY_train,outputs,"centeral_alignment_cris")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"epoch = {epoch}, loss: {loss.item()}")
        return self

    def transform(self, Xs=None):

        outputs = self.model(self.combine_K)
        outputs = torch.squeeze(outputs)
        if Xs.shape[2] != self.train_size: # training
            return  outputs[self.train_size:, :self.train_size].detach().numpy()
        else: #test
            return outputs[:self.train_size, :self.train_size].detach().numpy()