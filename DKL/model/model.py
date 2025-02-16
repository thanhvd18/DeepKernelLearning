import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F


class DeepCNN(nn.Module):
    def __init__(self,n_kernel=5,kernel_size = 1, n_layer=8):
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
              x = conv_list[i](x)
              if i<self.n_layer-1:
                x = self.relu(x)


        lastConv =  nn.Conv2d(n_kernel_list[self.n_layer-1],1, kernel_size=1)
        x = lastConv(x)
        x = self.sigmoid(x)
        return x
    

# Convolutional Gating MLP Model
class ConvolutionalGatingMLP(nn.Module):
    def __init__(self, num_kernels=3, kernel_size=231):
        super(ConvolutionalGatingMLP, self).__init__()
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        
        # Gating Network (Convolutional)
        self.conv1 = nn.Conv2d(in_channels=num_kernels, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        
        # MLP for combining kernels
        self.fc1 = nn.Linear(num_kernels * kernel_size * kernel_size, 512)
        self.fc2 = nn.Linear(512, kernel_size * kernel_size)
    
    def forward(self, kernels):
        # Gating Mechanism
        gating_weights = self.sigmoid(self.conv2(self.conv1(kernels)))
        gated_kernels = kernels * gating_weights  # Element-wise multiplication
        
        # Flatten and MLP
        combined_input = gated_kernels.view(1, -1)
        mlp_output = F.relu(self.fc1(combined_input))
        combined_kernel = self.fc2(mlp_output).view(1, self.kernel_size, self.kernel_size)
        return combined_kernel
    


class DeepCNN1(nn.Module):
    def __init__(self, n_kernel=5, kernel_size=1, n_layer=8):
        super(DeepCNN1, self).__init__()
        self.n_layer = n_layer
        
        # Define convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = n_kernel
        out_channels_list = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        
        for i in range(n_layer):
            out_channels = out_channels_list[i]
            padding = (kernel_size // 2) if kernel_size > 1 else 0
            self.conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
            )
            in_channels = out_channels
        
        # Define additional layers for global feature extraction
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.global_fc = nn.Linear(out_channels_list[n_layer-1] * 2, out_channels_list[n_layer-1])
        
        # Final convolution to reduce channels to 1
        self.final_conv = nn.Conv2d(out_channels_list[n_layer-1] * 2, 1, kernel_size=1)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Pass through convolutional layers with ReLU activation
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            if i < self.n_layer - 1:
                x = self.relu(x)
        
        # Global feature extraction (before final_conv)
        global_avg = self.global_avg_pool(x)  # Shape: [batch, channels, 1, 1]
        global_max = self.global_max_pool(x)  # Shape: [batch, channels, 1, 1]
        
        # Concatenate global features
        global_features = torch.cat([global_avg, global_max], dim=1)  # Shape: [batch, 2 * channels, 1, 1]
        print("before",global_features.shape)
        global_features = global_features.view(global_features.size(0), -1)  # Shape: [batch, 2 * channels]
        print("aft",global_features.shape)
        # Pass through fully connected layer
        global_features = self.global_fc(global_features)  # Shape: [batch, channels]
        global_features = global_features.unsqueeze(-1).unsqueeze(-1)  # Shape: [batch, channels, 1, 1]
        
        # Expand global features to match spatial dimensions
        global_features = global_features.expand(-1, -1, x.size(2), x.size(3))  # Shape: [batch, channels, H, W]
        
        # Concatenate local and global features
        combined_features = torch.cat([x, global_features], dim=1)  # Shape: [batch, 2 * channels, H, W]
        
        # Final convolution and activation
        out = self.final_conv(combined_features)  # Shape: [batch, 1, H, W]
        out = self.sigmoid(out)
        
        return out



