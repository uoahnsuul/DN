CU_UNIT = ['QP', 'DEPTH', 'QT_DEPTH', 'BT_DEPTH', 'MT_DEPTH', 'ispMode']
PU_UNIT = ['CU_IDX', 'MODE', 'IS_MIP', 'MULTI_REF_IDX']
TU_UNIT = ['CU_IDX', 'DEPTH', 'HOR_TR', 'VER_TR']

import os
import numpy as np

from help_func.CompArea import Area
from help_func.CompArea import Position
from help_func.CompArea import Size


from help_func.typedef import *

_width = 0
_height = 1
_x_pos = 2
_y_pos = 3

def reverseDict(dics):
    assert isinstance(dics, dict)
    new_dic = {}
    for key, value in dics.items():
        new_dic[value] = key
    return new_dic

def recalcPosition(_cf, srcCld, dstCld, pos):
    if (1 if srcCld == Component.COMPONENT_Y else 0) == (1 if dstCld == Component.COMPONENT_Y else 0):
        return pos
    if srcCld == Component.COMPONENT_Y and dstCld != Component.COMPONENT_Y:
        return Position(pos.x >> ChromaScale, pos.y >> ChromaScale)
    return Position(pos.x << ChromaScale, pos.y << ChromaScale)


class CompArea(Area):
    def __init__(self, _compID, _cf, _area):
        super(CompArea, self).__init__(_area.w, _area.h, _area.x, _area.y)
        self.chromaFormat = _cf
        self.compID = _compID

    def chromaPos(self):
        if self.compID == Component.COMPONENT_Y:
            return Position(self.x >> ChromaScale, self.y >> ChromaScale)
        else:
            return Position(self.x, self.y)

    def lumaSize(self):
        if self.compID != Component.COMPONENT_Y:
            return Position(self.x << ChromaScale, self.y << ChromaScale)
        return Position(self.x, self.y)

    def chromaSize(self):
        if self.compID == Component.COMPONENT_Y:
            return Position(self.x >> ChromaScale, self.y >> ChromaScale)
        return Position(self.x, self.y)

    def lumaPos(self):
        if self.compID != Component.COMPONENT_Y:
            return Position(self.x << ChromaScale, self.y << ChromaScale)
        return Position(self.x, self.y)

    def compPos(self):
        return self.lumaPos() if self.compID == Component.COMPONENT_Y else self.chromaPos()

    def topLeftComp(self, _compID):
        return recalcPosition(self.chromaFormat, self.compID, _compID, self)

    def topRightComp(self, _compID):
        return recalcPosition(self.chromaFormat, self.compID, _compID,
                              Position(self.x + self.width - 1, self.y))

    def bottomLeftComp(self, _compID):
        return recalcPosition(self.chromaFormat, self.compID, _compID,
                              Position(self.x, self.y + self.height - 1))

    def bottomRightComp(self, _compID):
        return recalcPosition(self.chromaFormat, self.compID, _compID,
                              Position(self.x + self.width - 1, self.y + self.height - 1))

    def __eq__(self, other):
        if self.chromaFormat != other.chromaFormat:
            return False
        if self.compID != other.compID:
            return False
        return Position.__eq__(self, other) and Size.__eq__(self, other)

class UnitArea:
    def __init__(self, chromaFormat, blkY, blkCb):
        self.chromaFormat = chromaFormat
        self.block = [blkY, blkCb]

    def Y(self):
        return self.block[Component.COMPONENT_Y]

    def Cb(self):
        return self.block[Component.COMPONENT_Cb]

    def Cr(self):
        return self.block[Component.COMPONENT_Cb]

    def block(self, comp):
        return self.block[comp]


class _UnitArea:
    def __init__(self, dataPath):
        np_load = np.load(dataPath)
        self.luma = np_load['LUMA'] if 'LUMA' in np_load else np.array([])
        self.chroma = np_load['CHROMA'] if 'CHROMA' in np_load else np.array([])

    def makeBufbyArea(self, area, isLuma=True, sampling_ratio=1):
        l = self.targetTree(isLuma)
        filtered = l[:,~np.any([(l[_height]+l[_y_pos]) < area.y,
                               (l[_width]+l[_x_pos] < area.x),
                               l[_y_pos] > (area.y + area.height),
                               l[_x_pos] > (area.x + area.width)
                               ], axis=0)]
        filtered[_x_pos] -= area.x
        filtered[_y_pos] -= area.y
        tmp = filtered[_x_pos] < 0
        filtered[_width, tmp] += filtered[_x_pos, tmp]
        filtered[_x_pos, tmp] = 0
        tmp = filtered[_y_pos] < 0
        filtered[_height, tmp] += filtered[_y_pos, tmp]
        filtered[_y_pos, tmp] = 0
        tmp = filtered[_height] + filtered[_y_pos] - area.height
        filtered[_height, tmp>0] = area.height - filtered[_y_pos, tmp>0]
        tmp = filtered[_width] + filtered[_x_pos] - area.width
        filtered[_width, tmp>0] = area.width - filtered[_x_pos, tmp>0]
        return filtered

    def targetTree(self, isLuma):
        if isLuma:
            return self.luma
        else:
            return self.chroma

class CU(Area):
    QP = None
    DEPTH = None
    QT_DEPTH = None
    BT_DEPTH = None
    MT_DEPTH = None
    ispMode = None
    UNIT_INFO = None
    INDEX_DICT = {'UNIT_INFO':UNIT_INFO, 'QP':QP, 'DEPTH':DEPTH, 'QT_DEPTH':QT_DEPTH,
                  'BT_DEPTH':BT_DEPTH, 'MT_DEPTH':MT_DEPTH,
                  'ispMode':ispMode}

    def __init__(self, info):
        super(CU, self).__init__(*info[:4])
        self.pulist = []
        self.tulist = []
        self.info = info[4:]

class PU(Area):
    CU_IDX = None
    MODE = None
    IS_MIP = None
    MULTI_REF_IDX = None
    INDEX_DICT = {'CU_IDX':CU_IDX, 'MODE':MODE,
                  'IS_MIP':IS_MIP, 'MULTI_REF_IDX':MULTI_REF_IDX}
    UNIT_INFO = None

    def __init__(self, info, culist):
        super(PU, self).__init__(*info[:4])
        self.cu = culist[info[5]]
        self.cu.pulist.append(self)
        self.info = info[5:]

class TU(Area):
    CU_IDX = None
    DEPTH = None
    HOR_TR = None
    VER_TR = None
    INDEX_DICT = {'CU_IDX':CU_IDX, 'DEPTH':DEPTH,
                  'HOR_TR':HOR_TR, 'VER_TR':VER_TR}

    def __init__(self, info, culist):
        super(TU, self).__init__(*info[:4])
        self.cu = culist[info[5]]
        self.cu.tulist.append(self)
        self.info = info[5:]








if __name__=='__main__':
    # setClassInstance('\\\\223.195.38.111/Sequences/By_PIC_ALL/Training')
    u = _UnitArea('/Users/YangwooKim/PycharmProjects/CNN_Intra_Pred_Block/Dataset/BY_PIC/Training/TU/AI_PNG_0002_2040x1848_8bit_P420_37_POC0.npz')
    u.makeBufbyArea(area=Area(100,100, 100, 100))
    print(' ')