from model import common

import torch
import torch.nn as nn


class ChannelGate(nn.Module):
    def __init__(self, g):
        pass

class GSCNN(nn.Module):

    def __init__(self, args):
        super(GSCNN, self).__init__()
