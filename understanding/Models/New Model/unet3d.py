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
    def __init__(self, in_channels=1, out_channels=1, use_attention=True, dropout_rates=(0.0, 0.1, 0.2, 0.3, 0.4)):
        super().__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.use_attention = use_attention
        self.dropout_rates = dropout_rates

        # Initial layer with more filters
        self.inc = ResidualBlock(in_channels, 32)
        
        # Encoder path
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        
        # Additional deeper level
        self.down5 = Down(512, 1024)
        
        # Bridge / bottleneck
        self.bridge = nn.Sequential(
            nn.Dropout3d(0.4),  # Increased dropout in bottleneck
            ResidualBlock(1024, 1024)
        )
        
        # Decoder path
        self.up1 = Up(1024, 512, attention=use_attention)
        self.up2 = Up(512, 256, attention=use_attention)
        self.up3 = Up(256, 128, attention=use_attention)
        self.up4 = Up(128, 64, attention=use_attention)
        self.up5 = Up(64, 32, attention=use_attention)
        
        # Output layer
        self.outc = OutConv(32, out_channels)
        
        # Updated detection head in UNet3D class __init__ method
        self.det_head = nn.Sequential(
            # Global average pooling to collapse spatial dimensions
            nn.AdaptiveAvgPool3d(1),
            # Flatten to vector
            nn.Flatten(),
            # MLP for coordinate prediction
            nn.Linear(32, 64),  # from the final feature map (32 channels)
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Increased dropout in detection head
            nn.Linear(64, 4),  # [z,y,x,r] coordinates
            nn.Sigmoid()  # Add sigmoid to ensure outputs are in [0,1] range
        )
        
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
        # Encoder with spatial dropout after each block
        x1 = self.inc(x)
        x1 = F.dropout3d(x1, p=self.dropout_rates[0], training=self.training)
        
        x2 = self.down1(x1)
        x2 = F.dropout3d(x2, p=self.dropout_rates[1], training=self.training)
        
        x3 = self.down2(x2)
        x3 = F.dropout3d(x3, p=self.dropout_rates[2], training=self.training)
        
        x4 = self.down3(x3)
        x4 = F.dropout3d(x4, p=self.dropout_rates[3], training=self.training)
        
        x5 = self.down4(x4)
        x5 = F.dropout3d(x5, p=self.dropout_rates[4], training=self.training)
        
        x6 = self.down5(x5)
        
        # Bridge
        x6 = self.bridge(x6)
        
        # Decoder with spatial dropout after each block
        x = self.up1(x6, x5)
        x = F.dropout3d(x, p=self.dropout_rates[3], training=self.training)
        
        x = self.up2(x, x4)
        x = F.dropout3d(x, p=self.dropout_rates[2], training=self.training)
        
        x = self.up3(x, x3)
        x = F.dropout3d(x, p=self.dropout_rates[1], training=self.training)
        
        x = self.up4(x, x2)
        x = F.dropout3d(x, p=self.dropout_rates[0], training=self.training)
        
        feature_maps = self.up5(x, x1)
        
        # Main segmentation output
        logits = self.outc(feature_maps)
        
        # Detection output - completely separate path from the same features
        det_coords = self.det_head(feature_maps)
        
        return logits, det_coords 