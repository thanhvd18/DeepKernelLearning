import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DeepCNN(nn.Module):
    def __init__(self,n_kernel=5,kernel_size = 1, n_layer=8):
        super(DeepCNN, self).__init__()
        super(DeepCNN, self).__init__()
        if kernel_size == 3:
          self.conv1 = nn.Conv2d(n_kernel, 8, kernel_size=3,padding=1)
          self.conv2 = nn.Conv2d(8, 16, kernel_size=3,padding=1)
          self.conv3 = nn.Conv2d(16, 32, kernel_size=3,padding=1)
          self.conv4 = nn.Conv2d(32, 64, kernel_size=3,padding=1)
          self.conv5 = nn.Conv2d(64, 128, kernel_size=3,padding=1)
          self.conv6 = nn.Conv2d(128, 256, kernel_size=3,padding=1)
          self.conv7 = nn.Conv2d(256, 512, kernel_size=3,padding=1)
          self.conv8 = nn.Conv2d(512, 1024, kernel_size=3,padding=1)
          self.conv9 = nn.Conv2d(1024, 2048, kernel_size=3,padding=1)
          self.conv10 = nn.Conv2d(2048,4096, kernel_size=3,padding=1)


        elif kernel_size == 1:
          self.conv1 = nn.Conv2d(n_kernel, 8, kernel_size=1)
          self.conv2 = nn.Conv2d(8, 16, kernel_size=1)
          self.conv3 = nn.Conv2d(16, 32, kernel_size=1)
          self.conv4 = nn.Conv2d(32, 64, kernel_size=1)
          self.conv5 = nn.Conv2d(64, 128, kernel_size=1)
          self.conv6 = nn.Conv2d(128, 256, kernel_size=1)
          self.conv7 = nn.Conv2d(256, 512, kernel_size=1)
          self.conv8 = nn.Conv2d(512, 1024, kernel_size=1)
          self.conv9 = nn.Conv2d(1024, 2048, kernel_size=1)
          self.conv10 = nn.Conv2d(2048,4096, kernel_size=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.n_layer = n_layer

    def forward(self, x):
        n_kernel_list = [8,16,32,64,128,256,512,1024,2048,4096]
        conv_list = [self.conv1, self.conv2,self.conv3,self.conv4,self.conv5,
                      self.conv6,self.conv7,self.conv8,self.conv9,self.conv10]

        out_list = []
        if self.n_layer == 1:
          x = conv_list[0](x)
        else:
          for i in range(self.n_layer):
              # print(f"===== {i}", x.shape)
              x = conv_list[i](x)
              if i<self.n_layer-1:
                x = self.relu(x)


        lastConv =  nn.Conv2d(n_kernel_list[self.n_layer-1],1, kernel_size=1)
        x = lastConv(x)
        x = self.sigmoid(x)
        return x