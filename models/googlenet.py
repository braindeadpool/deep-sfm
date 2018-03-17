# Implements the GoogLeNet architecture from the paper
# "Going deeper with convolutions" https://arxiv.org/pdf/1409.4842.pdf .

import torch
from torch import nn
from torch import functional


class GoogLeNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *input):
        pass

    # Implement Local Response Norm or wait for next release of pytorch to use the new implementation