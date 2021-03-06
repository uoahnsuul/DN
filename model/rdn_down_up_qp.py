# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797

#from model import common

import torch
import torch.nn as nn


def make_model(args, parent=False):
    return RDN(args)


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class RDN(nn.Module):
    def __init__(self, args):
        super(RDN, self).__init__()
        r = 2
        G0 = args.G0*2
        kSize = args.RDNkSize

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 128),
        }[args.RDNconfig]

        # Shallow feature extraction net
        self.SFENet1_input5 = nn.Conv2d(5, G0//2, kSize, padding=(kSize - 1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0//2, G0, kSize, padding=(kSize - 1) // 2, stride=2)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0=G0, growRate=G, nConvLayers=C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

        if r == 1:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, args.n_colors, kSize, padding=(kSize - 1) // 2, stride=1)
            ])

        # Up-sampling net
        elif r == 2 or r == 3:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0//2, G * r * r, kSize, padding=(kSize - 1) // 2, stride=1),
                nn.PixelShuffle(r),
                nn.Conv2d(G, 1, kSize, padding=(kSize - 1) // 2, stride=1)
            ])
        elif r == 4:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G * 4, kSize, padding=(kSize - 1) // 2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, G * 4, kSize, padding=(kSize - 1) // 2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, 3, kSize, padding=(kSize - 1) // 2, stride=1)
            ])
        else:
            raise ValueError("scale must be 2 or 3 or 4.")

        self.chroma = nn.Conv2d(64, 2, kSize, padding=(kSize - 1) // 2)
        self.down = nn.Conv2d(5, 5, kSize, padding=(kSize - 1) // 2, stride=2)

    def forward(self, x):

        # Y = x[:, 0, :, :]
        # UV = x[:, 1:3, :, :]
        # QP = x[:, 3, :, :]
        # CU = x[:, 4, :, :]

        _x = self.SFENet1_input5(x)
        _x = self.SFENet2(_x)

        RDBs_out = []
        for i in range(self.D):
            _x = self.RDBs[i](_x)
            RDBs_out.append(_x)

        _x = self.GFF(torch.cat(RDBs_out, 1))
        # x += f__1

        outY = _x[:, 0:64, :, :]
        outUV = _x[:, 64:, :, :]

        outY = self.UPNet(outY)[:, 0, :, :] + x[:, 0, ...]
        outUV = self.chroma(outUV)
        chroma_org = self.down(x)[:, 1:3, ...]
        outUV = outUV + chroma_org

        #out = self.UPNet(_x) + x[:,:3,...]

        out = [outY, outUV]

        return out


if __name__ == '__main__':
    from help_func.my_torchsummary import summary
    from option import args

    model = RDN(args)
    summary(model, (5, 96, 96), device='cpu')
