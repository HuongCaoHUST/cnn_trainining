import torch
import torch.nn as nn

class Block(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(Block, self).__init__()
        # Depthwise convolution
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, 
                               padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # Pointwise convolution
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, 
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class MobileNet(nn.Module):
    # (output_channels, stride)
    cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2), 
           512, 512, 512, 512, 512, (1024, 2), 1024]

    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        self.layers = self._make_layers(in_channels=32)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def _make_layers(self, in_channels):
        layers = []
        for x in self.cfg:
            out_channels = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_channels, out_channels, stride))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

