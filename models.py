import torch
import torch.nn as nn
import torch.nn.functional as F

class VariableInput3DCNN(nn.Module):
    def __init__(self, num_classes, N=2):
        super(VariableInput3DCNN, self).__init__()
        self.name="VariableInput3DCNN"
        
        # 3D Convolutional Layers
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=N*2, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(N*2)
        
        self.conv2 = nn.Conv3d(in_channels=N*2, out_channels=N*4, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(N*4)
        
        self.conv3 = nn.Conv3d(in_channels=N*4, out_channels=N*8, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(N*8)
        
        self.conv4 = nn.Conv3d(in_channels=N*8, out_channels=N*16, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm3d(N*16)
        
        # Adaptive Pooling Layer
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Fully Connected Layer
        self.fc = nn.Linear(N*16, num_classes)
    
    def forward(self, x):
        # Input x shape: [batch_size, channels, depth, height, width]
        
        # Pass through the 3D Conv layers with ReLU and BatchNorm
        x = F.relu(self.bn1(self.conv1(x)))  # -> [batch_size, 32, depth, height, width]
        x = F.max_pool3d(x, 2)  # -> [batch_size, 32, depth/2, height/2, width/2]
        
        x = F.relu(self.bn2(self.conv2(x)))  # -> [batch_size, 64, depth/2, height/2, width/2]
        x = F.max_pool3d(x, 2)  # -> [batch_size, 64, depth/4, height/4, width/4]
        
        x = F.relu(self.bn3(self.conv3(x)))  # -> [batch_size, 128, depth/4, height/4, width/4]
        x = F.max_pool3d(x, 2)  # -> [batch_size, 128, depth/8, height/8, width/8]
        
        x = F.relu(self.bn4(self.conv4(x)))  # -> [batch_size, 256, depth/8, height/8, width/8]
        x = F.max_pool3d(x, 2)  # -> [batch_size, 256, depth/16, height/16, width/16]
        
        # Apply Adaptive Pooling to get a fixed-size output
        x = self.adaptive_pool(x)  # -> [batch_size, 256, 1, 1, 1]
        
        # Flatten the output
        x = x.view(x.size(0), -1)  # -> [batch_size, 256]
        
        # Final Fully Connected Layer for classification
        x = self.fc(x)  # -> [batch_size, num_classes]
        
        return x