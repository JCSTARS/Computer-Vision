import math
import torch
import torch.nn as nn
from torchinfo import summary

# Cifar 数据集是32*32的图片
img_height = 32
img_width = 32


# 3x3 卷积定义
def conv3x3(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                     stride=stride, padding=padding, bias=False)


# Resnet_50  中的残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.mid_channels = out_channels//4
        self.conv1 = conv3x3(in_channels, self.mid_channels, kernel_size=1, padding=0)
        # Resnet50 中，从第二个残差块开始每个layer的第一个残差块都需要一次downsample
        self.bn1 = nn.BatchNorm2d(self.mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(self.mid_channels, self.mid_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(self.mid_channels)
        self.conv3 = conv3x3(self.mid_channels, out_channels, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.downsample_0 = nn.Sequential()
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample:
            residual = self.downsample(x)
        else:
            residual = self.downsample_0(x)

        out += residual
        out = self.relu(out)
        return out


# ResNet定义
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.conv = conv3x3(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(64)
        # self.max_pool = nn.MaxPool2d(3,2,padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 64, 256, layers[0], 1)
        self.layer2 = self.make_layer(block, 256, 512, layers[1], 2)
        self.layer3 = self.make_layer(block, 512, 1024, layers[2], 2)
        self.layer4 = self.make_layer(block, 1024, 2048, layers[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(math.ceil(img_height/32)*math.ceil(img_width/32)*2048, num_classes)

    def make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels))
        layers = [block(in_channels, out_channels, stride, downsample)]
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        # out = self.max_pool(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out



if __name__ == '__main__':
    model = ResNet(ResidualBlock, [3, 4, 6, 3])
    print(model)
    summary(model, input_size=(1, 3, 32, 32))
