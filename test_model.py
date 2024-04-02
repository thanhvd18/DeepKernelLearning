import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import *


def test_DeepCNN():
    model = DeepCNN(n_kernel=5,kernel_size = 1, n_layer=8)
    x = torch.randn(1,5,100,100)
    y = model(x)
    assert y.shape == torch.Size([1,1,100,100])
    print("DeepCNN test passed")