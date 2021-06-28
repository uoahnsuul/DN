from help_func.typedef import *
import os
import numpy as np

from help_func.Unit import CU
from help_func.Unit import PU
from help_func.Unit import TU
from help_func.Unit import CompArea



uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])


def setClassInstance(path):
    clases = {'CU':CU, 'PU':PU, 'TU':TU}
    for key, value in clases.items():
        if os.path.exists(os.path.join(path, key + '_INFO.npz')):
            unitinfoes = list(np.load(os.path.join(path, key + '_INFO.npz'))['NAME'])
            for i, unitinfo in enumerate(unitinfoes):
                setattr(value, unitinfo, i)
                value.INDEX_DICT[unitinfo] = i


class Units:
    def __init__(self, channeltype, cu, pu, tu):
        self.channeltype = channeltype
        self.culist = np.array([CU(c) for c in cu.T])
        self.pulist = np.array([PU(p, self.culist) for p in pu.T])
        self.tulist = np.array([TU(t, self.culist) for t in tu.T])

class Picture:
    def __init__(self, datapath, loadpic, loadunit):
        self.filename = os.path.basename(datapath)
        self.dirname = os.path.dirname(os.path.dirname(datapath))
        self.pelbuf = {}
        self.units = {}
        dirsplit = os.path.splitext(self.filename)[0].split('_')
        self.poc = int(dirsplit[-1].split('POC')[1])
        self.qp = int(dirsplit[-2])
        self.chromaformat = ChromaFormat.YCbCr4_2_0 if dirsplit[-3]=='P420' else ChromaFormat.YCbCr4_4_4 if dirsplit[-3]=='P444' else ChromaFormat.YCbCr4_0_0


        for idx ,peltype in PictureFormat.INDEX_DIC.items():
            if idx in loadpic:
                self.pelbuf[peltype] = np.load(os.path.join(self.dirname, '{}/{}'.format(peltype, self.filename)))
            else:
                self.pelbuf[peltype] = None

        setClassInstance(self.dirname)
        for idx, unittype in UnitFormat.INDEX_DIC.items():
            if idx in loadunit:
                self.units[unittype] = np.load(os.path.join(self.dirname, '{}/{}'.format(unittype, self.filename)))
            else:
                self.units[unittype] = None

        self.lumaunits = Units(ChannelType.CHANNEL_TYPE_LUMA, self.units['CU'][ChannelType.STR_LUMA],
                               self.units['PU'][ChannelType.STR_LUMA], self.units['TU'][ChannelType.STR_LUMA])
        self.chromaunits = Units(ChannelType.CHANNEL_TYPE_CHROMA, self.units['CU'][ChannelType.STR_CHROMA],
                               self.units['PU'][ChannelType.STR_CHROMA], self.units['TU'][ChannelType.STR_CHROMA])




    def UpSamplingChroma(self, UVPic):
        if self.chromaformat == ChromaFormat.YCbCr4_2_0:
            return UVPic.repeat(2, axis=0).repeat(2, axis=1)
        elif self.chromaformat == ChromaFormat.YCbCr4_4_4:
            return UVPic
        else:
            return None


if __name__ == '__main__':
    u = Picture('/Users/YangwooKim/PycharmProjects/CNN_Intra_Pred_Block/Dataset/BY_PIC/Training/TU/AI_PNG_0002_2040x1848_8bit_P420_37_POC0.npz',
                [PictureFormat.ORIGINAL, PictureFormat.PREDICTION, PictureFormat.UNFILTEREDRECON, PictureFormat.RECONSTRUCTION],
                [UnitFormat.CU, UnitFormat.PU, UnitFormat.TU])

