from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ResLight10Detector", "reslight10_detector"]


# --- Convolution helpers ---------------------------------------------------
def _conv3x3x3(in_channels: int, out_channels: int, stride: int | tuple[int, int, int] = 1) -> nn.Conv3d:
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


# --- Residual building block ----------------------------------------------
class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int | tuple[int, int, int] = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = _conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


# --- ResLight detector backbone -------------------------------------------
class ResLight10Detector(nn.Module):
    """3D ResNet-Lite-10 backbone for gesture detection."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        sample_size: int = 112,
        sample_duration: int = 8,
        use_adaptive_pool: bool = False,
    ) -> None:
        super().__init__()
        self.inplanes = 16
        self._use_adaptive_pool = use_adaptive_pool
        self._pool_kernel: tuple[int, int, int] | None = None
        if not self._use_adaptive_pool:
            last_duration = max(1, math.ceil(sample_duration / 16))
            last_size = max(1, math.ceil(sample_size / 32))
            self._pool_kernel = (last_duration, last_size, last_size)

        self.conv1 = nn.Conv3d(
            in_channels,
            16,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.layer1 = self._make_layer(16, blocks=1)
        self.layer2 = self._make_layer(32, blocks=1, stride=2)
        self.layer3 = self._make_layer(64, blocks=1, stride=2)
        self.layer4 = self._make_layer(128, blocks=1, stride=2)

        if self._use_adaptive_pool or self._pool_kernel is None:
            self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avgpool = nn.AvgPool3d(self._pool_kernel, stride=1)
        
        self.dropout = nn.Dropout(p=0.5)
        
        self.fc = nn.Linear(128 * _BasicBlock.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, planes: int, blocks: int, stride: int | tuple[int, int, int] = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * _BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    planes * _BasicBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(planes * _BasicBlock.expansion),
            )

        layers = [_BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * _BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(_BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self._use_adaptive_pool or self._pool_kernel is None:
            x = self.avgpool(x)
        else:
            kernel_t = min(x.size(2), self._pool_kernel[0])
            kernel_h = min(x.size(3), self._pool_kernel[1])
            kernel_w = min(x.size(4), self._pool_kernel[2])
            if (kernel_t, kernel_h, kernel_w) == self._pool_kernel:
                x = self.avgpool(x)
            else:
                x = F.avg_pool3d(x, kernel_size=(kernel_t, kernel_h, kernel_w), stride=1)
        
        x = torch.flatten(x, 1)
        x = self.dropout(x)  # Apply dropout before final FC layer
        x = self.fc(x)
        return x


# --- Convenience factory ---------------------------------------------------
def reslight10_detector(in_channels: int = 3, num_classes: int = 2) -> ResLight10Detector:
    return ResLight10Detector(in_channels=in_channels, num_classes=num_classes)
