import torch.nn as nn
import torch

def make_model(args, parent=False):
    return IFnD(args)


class IFnD(nn.Module):
    def __init__(self, args):
        super(IFnD, self).__init__()

        d = 14
        conv = []
        conv.append(nn.Conv2d(8, 96, 3, padding=1, stride=1))
        for i in range(d):
            conv.append(nn.Conv2d(96, 96, 3, padding=1, stride=1))
            conv.append(nn.LeakyReLU())
        conv.append(nn.Conv2d(96, 6, 3, padding=1, stride=1))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):

        y = torch.cat((torch.unsqueeze(x[:, 0, :72, :72], 1), torch.unsqueeze(x[:, 0, 72:, :72], 1),
                       torch.unsqueeze(x[:, 0, :72, 72:], 1), torch.unsqueeze(x[:, 0, 72:, 72:], 1),
                       torch.unsqueeze(x[:, 1, ::2, ::2], 1), torch.unsqueeze(x[:, 2, ::2, ::2], 1),
                       torch.unsqueeze(x[:, 3, :72, :72], 1), torch.unsqueeze(x[:, 4, ::2, ::2], 1)), 1)
        # Yx4, U, V, QP, CU

        out = self.conv(y)
        out += y[:, :6, :, :]
        output = torch.cat((torch.cat((torch.unsqueeze(out[:, 0, :, :], 1), torch.unsqueeze(out[:, 1, :, :], 1)), 2),
                            torch.cat((torch.unsqueeze(out[:, 2, :, :], 1), torch.unsqueeze(out[:, 3, :, :], 1)), 2)), 3)

        output = torch.cat((output, out[:, 4, :, :].repeat(2, axis=2).repeat(2, axis=3)))
        return output


if __name__ == '__main__':
    from help_func.my_torchsummary import summary
    from option import args

    model = IFnD(args)
    summary(model, (5, 144, 144), device='cpu')
