import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet3D(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super().__init__()
        
        # Encoder
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(32, 64)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(64, 128)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(128, 256)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(256, 512)
        )
        
        # Decoder
        self.up1 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(128, 64)
        self.up4 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.up_conv4 = DoubleConv(64, 32)
        
        # Output
        self.outc = nn.Conv3d(32, n_classes, kernel_size=1)
        
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
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder
        x = self.up1(x5)
        x = self.up_conv1(torch.cat([x4, x], dim=1))
        x = self.up2(x)
        x = self.up_conv2(torch.cat([x3, x], dim=1))
        x = self.up3(x)
        x = self.up_conv3(torch.cat([x2, x], dim=1))
        x = self.up4(x)
        x = self.up_conv4(torch.cat([x1, x], dim=1))
        
        # Output
        logits = self.outc(x)
        return torch.sigmoid(logits) 