import torch.nn as nn
import torch.nn.functional as F

class Conv1d(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=1, stride=1, padding=0, bn=True, activation=F.leaky_relu):
        super().__init__()
        self.conv = nn.Conv1d(in_size, out_size, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_size) if bn else None
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class Conv2d(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bn=True, activation=F.leaky_relu):
        super().__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_size) if bn else None
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
