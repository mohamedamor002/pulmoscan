import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Residual block for 3D UNet"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels)
        )
        
        # Skip connection with 1x1 conv if channel dimensions don't match
        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.skip(x)
        out = self.double_conv(x)
        out += residual
        return self.relu(out)

class Down(nn.Module):
    """Downscaling with maxpool then residual block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            ResidualBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then residual block"""
    def __init__(self, in_channels, out_channels, attention=True):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ResidualBlock(in_channels, out_channels)
        self.attention = attention
        
        if attention:
            self.attention_gate = AttentionGate(in_channels // 2, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Apply attention if enabled
        if self.attention:
            x2 = self.attention_gate(x2, x1)
        
        # Handle size differences
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class AttentionGate(nn.Module):
    """Attention Gate for focusing on relevant features"""
    def __init__(self, in_channels, gate_channels):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(gate_channels, gate_channels, kernel_size=1, bias=True),
            nn.BatchNorm3d(gate_channels)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(in_channels, gate_channels, kernel_size=1, bias=True),
            nn.BatchNorm3d(gate_channels)
        )
        
        self.psi = nn.Sequential(
            nn.Conv3d(gate_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, use_attention=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.use_attention = use_attention

        # Initial layer with more filters
        self.inc = ResidualBlock(n_channels, 32)
        
        # Encoder path
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        
        # Additional deeper level
        self.down5 = Down(512, 1024)
        
        # Bridge / bottleneck
        self.bridge = nn.Sequential(
            nn.Dropout3d(0.2),  # Add dropout to prevent overfitting
            ResidualBlock(1024, 1024)
        )
        
        # Decoder path
        self.up1 = Up(1024, 512, attention=use_attention)
        self.up2 = Up(512, 256, attention=use_attention)
        self.up3 = Up(256, 128, attention=use_attention)
        self.up4 = Up(128, 64, attention=use_attention)
        self.up5 = Up(64, 32, attention=use_attention)
        
        # Output layer
        self.outc = OutConv(32, n_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        
        # Bridge
        x6 = self.bridge(x6)
        
        # Decoder with skip connections
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        
        # Output
        logits = self.outc(x)
        return logits  # No sigmoid here, we'll apply it in the loss function 