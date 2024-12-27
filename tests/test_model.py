import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..', 'DKL'))
from DKL.model.model import DeepCNN
import torch


def test_output():
    nModality = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ = torch.randn(nModality, 64, 64).to(device)
    model = DeepCNN(n_kernel=nModality, kernel_size=1, n_layer=8).to(device)
    output = model(input_)
    print(output.shape)
    assert output.shape == torch.Size([1, 64, 64])
    print("DeepCNN test passed")

if __name__ == '__main__':
    test_output()
