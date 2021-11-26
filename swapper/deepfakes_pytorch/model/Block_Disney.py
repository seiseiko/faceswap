import torch
import torch.nn as nn
import torch.nn.functional as F

class FromRGB(nn.Module):
    def __init__(self, level):
        super(FromRGB, self).__init__()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        if level < 3:
            self.Conv = nn.Conv2d(in_channels=3,
                                out_channels=512,
                                kernel_size=1,
                                stride=1)
            # self.normal = nn.BatchNorm2d(512)
        else:
            self.Conv = nn.Conv2d(in_channels=3,
                                out_channels=2**(12-level),
                                kernel_size=1,
                                stride=1)
            # self.normal = nn.BatchNorm2d(2**(12-level))

    def forward(self, x):
        out = self.Conv(x)
        out = self.LeakyReLU(out)
        # out = self.normal(out)
        return out

class ToRGB(nn.Module):
    def __init__(self, level):
        super(ToRGB, self).__init__()
        self.Sigmoid = torch.sigmoid
        if level < 3:
            self.Conv = nn.Conv2d(in_channels=512,
                                out_channels=3,
                                kernel_size=1,
                                stride=1)
        else:
            self.Conv = nn.Conv2d(in_channels=2**(12-level),
                                out_channels=3,
                                kernel_size=1,
                                stride=1)

    def forward(self, x):
        out = self.Conv(x)
        out = self.Sigmoid(out)
        return out

class EncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1, padding=1):
        super(EncoderLayer, self).__init__()
        self.Conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=strides,
                              padding=padding)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        # self.normal = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.Conv(x)
        out = self.LeakyReLU(out)
        # out = self.normal(out)
        return out

class DecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1, padding=1):
        super(DecoderLayer, self).__init__()
        self.Conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=strides,
                              padding=padding)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        # self.normal = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.Conv(x)
        out = self.LeakyReLU(out)
        # out = self.normal(out)
        return out

class DecoderZeroLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, strides=1, padding=0):
        super(DecoderZeroLayer, self).__init__()
        self.Conv = nn.ConvTranspose2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=strides,
                              padding=padding)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        # self.normal = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.Conv(x)
        out = self.LeakyReLU(out)
        # out = self.normal(out)
        return out

class EncoderLevel(nn.Module):
    def __init__(self, level):
        super(EncoderLevel, self).__init__()
        self.level = level
        if level == 0:
            self.layer1 = EncoderLayer(512, 512)
            self.layer2 = EncoderLayer(512, 512, 4, 1, 0)
        elif level < 4:
            self.layer1 = EncoderLayer(512, 512)
            self.layer2 = EncoderLayer(512, 512)
        else:
            self.layer1 = EncoderLayer(2**(12-level), 2**(12-level))
            self.layer2 = EncoderLayer(2**(12-level), 2**(12-level+1))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        if self.level != 0:
            out = F.interpolate(out, [2**(self.level+1), 2**(self.level+1)])
        return out

class DecoderLevel(nn.Module):
    def __init__(self, level):
        super(DecoderLevel, self).__init__()
        self.level = level
        if level == 0:
            self.layer1 = DecoderZeroLayer(512, 512)
            self.layer2 = DecoderLayer(512, 512)
        elif level < 4:
            self.layer1 = DecoderLayer(512, 512)
            self.layer2 = DecoderLayer(512, 512)
        else:
            self.layer1 = DecoderLayer(2**(12-level+1), 2**(12-level))
            self.layer2 = DecoderLayer(2**(12-level), 2**(12-level))

    def forward(self, x):
        if self.level != 0:
            out = F.interpolate(x, [2**(self.level+2), 2**(self.level+2)])
            out = self.layer1(out)
        else:
            out = self.layer1(x)
        out = self.layer2(out)
        
        return out