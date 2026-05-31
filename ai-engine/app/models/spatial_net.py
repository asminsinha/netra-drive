import torch
import torch.nn as nn

class SeparableConv2d(nn.Module):
    """Depthwise separable convolution for fast, low-latency edge-frame extraction."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class NetraDriveSpatialNet(nn.Module):
    """
    Custom Multi-Task Vision Network built from scratch.
    Accepts raw gray/RGB crops and estimates:
    1. 68-point Facial Landmarks Matrix
    2. Absolute Head Pose Orientation Angles
    3. Photometric Illumination Quality Estimation
    """
    def __init__(self):
        super().__init__()
        
        # Shared Feature Extracting Backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), # 112x112
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            SeparableConv2d(16, 32, kernel_size=3, stride=2, padding=1), # 56x56
            SeparableConv2d(32, 64, kernel_size=3, stride=2, padding=1), # 28x28
            SeparableConv2d(64, 128, kernel_size=3, stride=2, padding=1), # 14x14
            SeparableConv2d(128, 256, kernel_size=3, stride=2, padding=1), # 7x7
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Task Heads
        # 1. Regression for 68 points (x, y coords) -> 136 output units
        self.landmark_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 136)
        )
        
        # 2. Regression for Head Pose Angles (Pitch, Yaw, Roll) -> 3 units
        self.pose_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3)
        )
        
        # 3. Classification for Cabin Lighting Quality -> 1 unit scalar
        self.illumination_head = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        
        landmarks = self.landmark_head(features)
        pose = self.pose_head(features)
        illumination = self.illumination_head(features)
        
        return {
            "landmarks": landmarks,
            "pose": pose,
            "illumination_quality": illumination
        }