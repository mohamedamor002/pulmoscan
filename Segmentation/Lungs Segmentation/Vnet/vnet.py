import torch
import torch.nn as nn
import torch.nn.functional as F

class InputBlock(nn.Module):
    """Initial block that maps input to first feature maps"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm3d(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class DownBlock(nn.Module):
    """Downsampling block with residual connection"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Left pathway
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2, stride=2)
        self.bn1 = nn.BatchNorm3d(out_channels)
        
        # Right pathway (residual)
        self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=2)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
    def forward(self, x):
        # Main pathway
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        # Residual pathway
        residual = self.conv2(x)
        residual = self.bn2(residual)
        
        return F.relu(out + residual, inplace=True)

class UpBlock(nn.Module):
    """Upsampling block with residual connection"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Transposed convolution for upsampling
        self.upconv = nn.ConvTranspose3d(
            in_channels, out_channels, 
            kernel_size=2, stride=2
        )
        
        # Convolutional layers
        self.conv1 = nn.Conv3d(2*out_channels, out_channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm3d(out_channels)
        
    def forward(self, x, skip):
        # Upsample
        x = self.upconv(x)
        
        # Crop skip connection if needed and concatenate
        diffZ = skip.size()[2] - x.size()[2]
        diffY = skip.size()[3] - x.size()[3]
        diffX = skip.size()[4] - x.size()[4]
        
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2,
                      diffZ // 2, diffZ - diffZ // 2])
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        return F.relu(x, inplace=True)

class OutputBlock(nn.Module):
    """Final output block"""
    def __init__(self, in_channels, out_channels):
        super(OutputBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        return x  # No sigmoid here!

class VNet(nn.Module):
    """Complete V-Net architecture"""
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        
        # Encoder path
        self.in_block = InputBlock(in_channels, 16)
        self.down1 = DownBlock(16, 32)
        self.down2 = DownBlock(32, 64)
        self.down3 = DownBlock(64, 128)
        self.down4 = DownBlock(128, 256)
        
        # Decoder path
        self.up1 = UpBlock(256, 128)
        self.up2 = UpBlock(128, 64)
        self.up3 = UpBlock(64, 32)
        self.up4 = UpBlock(32, 16)
        
        # Output
        self.out_block = OutputBlock(16, out_channels)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Encoder
        x1 = self.in_block(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        return self.out_block(x)