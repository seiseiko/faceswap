import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .Block_Disney import FromRGB, ToRGB, EncoderLevel, DecoderLevel


class Encoder(nn.Module):
    def __init__(self, res=128, common=0, device="cuda"):
        super(Encoder, self).__init__()
        #3*256*256 //  3*128*128
        self.res = res
        self.level = int(np.log2(self.res))-2
        self.common = common
        self.from_RGB = FromRGB(self.level)
        self.layers = nn.ModuleList()
        self.device = device
        for i in range(self.level, -1, -1):
            self.layers.append(EncoderLevel(i).to(self.device))

        if self.common != -1:
            self.common_layers = nn.ModuleList()
            for i in range(0, self.common+1):
                self.common_layers.append(DecoderLevel(i).to(self.device))

    def next_level(self):
        """
        """
        self.res *= 2
        self.level += 1
        self.old_from_RGB = self.from_RGB
        self.from_RGB = FromRGB(self.level).to(self.device)
        self.layers.insert(0, EncoderLevel(self.level).to(self.device))

    def update_RGB(self):
        del self.old_from_RGB

    def set_RGB(self):
        self.old_from_RGB = FromRGB(self.level-1)

    def forward(self, x, alpha=1):
        out = self.from_RGB(x)
        if self.level != self.common+1:
            if alpha != 1:
                out = self.layers[0](out)
                old_x = F.interpolate(x, [int(self.res/2), int(self.res/2)])
                old_out = self.old_from_RGB(old_x)
                out = alpha*out+(1-alpha)*old_out
                for i in range(1, len(self.layers)):
                    out = self.layers[i](out)
            else:
                for i in range(len(self.layers)):
                    out = self.layers[i](out)
        else:
            out = self.layers[0](out)

        # out += torch.from_numpy(np.random.normal(0, 1,
        #                                          (out.shape[0], out.shape[1], out.shape[2], out.shape[3]))).to("cuda")

        if self.common != -1:
            for i in range(len(self.common_layers)):
                out = self.common_layers[i](out)

        return out


class Decoder(nn.Module):
    def __init__(self, res=128, common=0, device="cuda"):
        super(Decoder, self).__init__()
        self.res = res
        self.level = int(np.log2(self.res))-2
        self.common = common
        self.device = device
        self.layers = nn.ModuleList()
        self.to_RGB = ToRGB(self.level)
        for i in range(self.common+1, self.level+1):
            self.layers.append(DecoderLevel(i).to(self.device))

    def next_level(self):
        self.res *= 2
        self.level += 1
        self.old_to_RGB = self.to_RGB
        self.to_RGB = ToRGB(self.level).to(self.device)
        self.layers.append(DecoderLevel(self.level).to(self.device))

    def update_RGB(self):
        del self.old_to_RGB

    def set_RGB(self):
        self.old_to_RGB = ToRGB(self.level-1)

    def forward(self, x, alpha=1):
        out = self.layers[0](x)
        if self.level != self.common+1:
            if alpha != 1:
                for i in range(1, len(self.layers)-1):
                    out = self.layers[i](out)
                old_out = self.old_to_RGB(out)
                old_out = F.interpolate(old_out, [self.res, self.res])
                out = self.layers[-1](out)
                out = self.to_RGB(out)
                out = alpha*out+(1-alpha)*old_out
            else:
                for i in range(1, len(self.layers)):
                    out = self.layers[i](out)
                out = self.to_RGB(out)
        else:
            out = self.to_RGB(out)

        return out
