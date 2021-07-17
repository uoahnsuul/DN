import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def make_model(args, parent=False):
    return GRDN(args)

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

class RDB_Conv(nn.Module):
    def __init__(self, in_channels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        inC = in_channels
        G = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(inC, G, kSize, padding=1, stride=1),
            nn.ReLU(inplace=True)
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        nC = nConvLayers

        layers = []

        for c in range(nC) :
            layers.append(RDB_Conv(G0 + c*G, G))
        self.layers = nn.Sequential(*layers)

        self.LF = nn.Conv2d(G0 + nC*G, G0, 1, padding=0,stride=1)

    def forward(self, x):
        out = self.LF(self.layers(x))
        return out + x

class GRDB(nn.Module):
    def __init__(self, growRate0):
        super(GRDB, self).__init__()

        G0 = growRate0

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = 16, 8, 64

        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0=G0, growRate=G, nConvLayers=C)
            )

        self.GF = nn.Conv2d(self.D * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        out = x

        RDBs_out = []
        for i in range(self.D):
            out = self.RDBs[i](out)
            RDBs_out.append(x)

        out = self.GF(torch.cat(RDBs_out, 1))

        return out + x


class GRDN(nn.Module):
    def __init__(self, args):
        super(GRDN, self).__init__()
        r = args.scale[0]
        G0 = args.G0
        kSize = args.RDNkSize

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = 16, 8, 64

        self.SFENet = nn.Conv2d(3, G0, kSize, padding=(kSize-1)//2, stride=1)

        self.convDown = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=2)

        self.GRDBs = nn.ModuleList()
        for i in range(self.D):
            self.GRDBs.append(
                GRDB(growRate0=G0)
            )

        self.convUp = nn.Sequential(*[
                nn.ConvTranspose2d(G0, G0, kSize, padding=(kSize-1)//2, stride=2, output_padding=1)
            ])

        self.cbam = CBAM(gate_channels=G0)
        self.final_conv = nn.Conv2d(G0, args.n_colors, kernel_size=3, padding=1)

    def forward(self, x):
        y = self.SFENet(x)
        y = self.convDown(y)
        for i in range(self.D):
            y = self.GRDBs[i](y)

        y = self.convUp(y)
        y = self.cbam(y)
        y = self.final_conv(y)
        return y + x



if __name__ == '__main__':
    from help_func.my_torchsummary import summary
    from option import args


    model = GRDN(args)
    summary(model, (3, 96, 96), device='cpu')