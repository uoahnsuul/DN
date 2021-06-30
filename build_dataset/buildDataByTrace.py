
import os
import time
import subprocess
import struct
import numpy as np
from PIL import Image
import math
from threading import Thread
import threading
from multiprocessing import Pool, Lock
from help_func.logging import LoggingHelper
from help_func.help_python import myUtil
import shutil
import tqdm

import copy
import random

from help_func.CompArea import TuList
from help_func.CompArea import UnitBuf
from help_func.CompArea import Component
from help_func.CompArea import ChromaFormat
from help_func.CompArea import Area



from collections import namedtuple
import csv

from help_func.CompArea import PictureFormat
from help_func.CompArea import LearningIndex

import re


import os
import ruamel.yaml

datatype = namedtuple('datatype', ['type', 'binlist', 'opt_num'])
if __name__ == '__main__':
    os.chdir("../")


logger = LoggingHelper.get_instance(always=True).logger

class ConfigMember(dict):
    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = ConfigMember(value)
        return value

class Config(dict):
    yaml = ruamel.yaml.YAML()
    yaml.allow_duplicate_keys = True

    def __init__(self, file_path, log):
        # print(os.getcwd())
        assert os.path.exists(file_path), "ERROR: Config File doesn't exist."
        with open(file_path, 'r') as f:
            self.member = self.yaml.load(f)
            f.close()
        self.logger = log

    def __getattr__(self, name):
        if name not in self.member:
            if self.logger is None:
                print("Miss no name '%s' in config ", name)
            else:
                self.logger.error("Miss no name '%s' in config ", name)
            return False
        value = self.member[name]
        if isinstance(value, dict):
            value = ConfigMember(value)
        return value

    def isExist(self, name):
        if name in self.member:
            return True
        return False


    def write_yml(self):
        path = self.member['NET_INFO_PATH']
        with open(path, 'w+') as fp:
            self.yaml.dump(self.member, fp)

reblockstat = r'^BlockStat: POC @\('


class investText:
    infotext = 'initinfo.txt'
    blockdir = 'block'
    def __init__(self, rootdir):
        self.rootdir = rootdir
        self.infopath = os.path.join(rootdir, investText.infotext)
        self.blockdir = os.path.join(rootdir, investText.blockdir)

        self.info = open(self.infopath, mode='r').readlines()
        self.width, self.height = self.getWidthHeight()
        self.rules = self.getinfoRules()


    def getWidthHeight(self):
        rex = re.compile("# Sequence size: \[(?P<width>\d+)x (?P<height>\d+)\]")
        for line in self.info:
            if rex.match(line):
                r = rex.match(line).groupdict()
                return int(r['width']), int(r['height'])
        else:
            assert 0

    def getinfoRules(self):
        rex = re.compile("# Block Statistic Type: ")
        scale = re.compile('Scale: (?P<parm>\d+)')
        ranges = re.compile('\[(?P<start>-*\d+), (?P<end>-*\d+)\]')
        rules = []
        for line in self.info:
            if rex.match(line):
                line = rex.sub('', line)
                tmp = line.split('; ')
                if scale.match(tmp[-1]):
                    tmp[-1] = ('scale', int(scale.match(tmp[-1]).group('parm')))
                elif ranges.match(tmp[-1]):
                    d = ranges.match(tmp[-1]).groupdict()
                    tmp[-1] = [int(d['start']), int(d['end'])]
                else:
                    tmp[-1] = ''
                rules.append(tmp)
        return rules

class investPOCBlocks:
    types = ['BlockStat: POC ',
             'PPSStat: POC '
             ]
    numpy_types = ['int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']

    def __init__(self, initinfo, poc):
        self.globalinfo = initinfo
        self.poc = poc
        self.ppsdic = {}
        self.blockstats = open(os.path.join(self.globalinfo.blockdir, (str(poc) + '.txt')), 'r').readlines()
        self.block_named_tuple, self.block_type,self.blockdic = self.initBlockDic()
        self.normalBlockRex = re.compile('(BlockStat: POC )(?P<poc>\d+) @\(\s*(?P<xpos>\d+),\s*(?P<ypos>\d+)\) \[\s*(?P<w>\d+)x\s*(?P<h>\d+)\] (?P<name>[^=]*)=\{*(?P<value>[-\d\s,]*)\}*')
        self.geoMVRex = re.compile('(BlockStat: POC )(?P<poc>\d+) @\[(?P<vectors>.*)--] (?P<name>.*)=\{*(?P<value>[-\d\s,]*)\}*')
        self.ppsRex = re.compile('(PPSStat: POC )(?P<poc>\d+) @ (?P<name>.*)=(?P<value>[-\d,]*)')
        self.matching = [(self.normalBlockRex, self.setNormalBlock), (self.geoMVRex, self.setGeoMV), (self.ppsRex, self.setPPSparm)]
        self.getinfo()

    def initBlockDic(self):
        blockType = namedtuple('blockType', ['name', 'type', 'scale', 'range'])
        tmp = {}
        tmp2 = {}
        tmp3  = {}
        for info in self.globalinfo.rules:
            tmp[info[0]] = blockType(info[0], info[1], info[2][1] if isinstance(info[2], tuple) else 0, info[2] if isinstance(info[2], list) else [])
            tmp3[info[0]] = info[1] if info[1] in investPOCBlocks.numpy_types else 'int16'
            tmp2[info[0]] = list()
        return tmp, tmp3, tmp2



    @staticmethod
    def decompGeoVect(value):
        arr = [int(y) for x in value.split('--') for y in x[1:-1].split(',')]
        arr = arr + [-1] * (10 - len(arr)) + [len(arr)]
        return arr

    def setGeoMV(self, dic):
        self.blockdic[dic['name']].append(self.decompGeoVect(dic['vectors']) + list(map(int, dic['value'].split(','))))
        return

    def setNormalBlock(self, dic):
        assert len(dic['value'])
        self.blockdic[dic['name']].append([dic['xpos'], dic['ypos'], dic['w'], dic['h'], *list(map(int, dic['value'].split(',')))])
        return

    def setPPSparm(self, dic):
        self.ppsdic[dic['name']] = list(map(int, dic['value'].split(',')))
        return

    def getinfo(self):
        for line in self.blockstats:
            for m, func in self.matching:
                tmp = m. match(line)
                if tmp:
                    func(tmp.groupdict())
                    break
            else:
                assert 0



#
#
# it = investText('C:\\Users\\YangwooKim\\Desktop\\Codec\\VVC-8.0\\VVCSoftware_VTM-VTM-8.0-getdata\\VVCSoftware_VTM-VTM-8.0-getdata\\build\\source\\App\\DecoderApp\\RA_BQTerrace_1920x1080_60_8bit_32_RS0')
# ipb = investPOCBlocks(it, 16)


class BuildData(object):
    COLOR_BLACK = [0, 512, 512]
    COLOR_RED = [304, 336, 1020]
    COLOR_GREEN = np.array([149, 43, 21], dtype='int16') << 2
    COLOR_BLUE = np.array([29, 255, 107], dtype='int16') << 2
    configpath = './build_dataset/data_config.yml'


    #get data set
    ORIGINAL = True
    PREDICTION = False
    RECON = True
    UNFILTERED = True
    BLOCK = True

    #Debug
    PRINT_PSNR = True

    #autoset
    def __init__(self, Datasetpath):
        # Datasetpath = './Dataset/' + Datasetpath
        self.Datasetpath = Datasetpath
        os.makedirs(Datasetpath, exist_ok=True)
        os.makedirs(os.path.join(Datasetpath, PictureFormat.INDEX_DIC[PictureFormat.ORIGINAL]), exist_ok=True)
        os.makedirs(os.path.join(Datasetpath, PictureFormat.INDEX_DIC[PictureFormat.PREDICTION]), exist_ok=True)
        os.makedirs(os.path.join(Datasetpath, PictureFormat.INDEX_DIC[PictureFormat.RECONSTRUCTION]), exist_ok=True)
        os.makedirs(os.path.join(Datasetpath, PictureFormat.INDEX_DIC[PictureFormat.UNFILTEREDRECON]), exist_ok=True)
        os.makedirs(os.path.join(Datasetpath, 'BLOCK'), exist_ok=True)


imginfo = namedtuple('imginfo', ['ORIGINAL', 'PREDICTION', 'RECON', 'UNFILTERED', 'BLOCK'])

class imgInfo(BuildData):
    thlock = threading.Lock()

    # os.makedirs(os.path.join(TrainingSetPath, '32x32'), exist_ok=True)
    # os.makedirs(os.path.join(TestSetPath, '32x32'), exist_ok=True)
    def __init__(self, dir, savepath, condition_dic=None):
        super().__init__(savepath)
        self.example_image_get = False
        self.root = dir
        self.basename = os.path.splitext(os.path.basename(self.root))[0]
        self.org_path = os.path.join(dir, PictureFormat.INDEX_DIC[PictureFormat.ORIGINAL])
        self.pred_path = os.path.join(dir, PictureFormat.INDEX_DIC[PictureFormat.PREDICTION])
        self.recon_path = os.path.join(dir, PictureFormat.INDEX_DIC[PictureFormat.RECONSTRUCTION])
        self.unfiltered_path = os.path.join(dir, PictureFormat.INDEX_DIC[PictureFormat.UNFILTEREDRECON])
        self.block_path = os.path.join(dir, 'BLOCK')
        self.path_list = [self.org_path, self.pred_path, self.recon_path, self.unfiltered_path, self.block_path]

        self.investText = investText(dir)
        self.width, self.height = self.investText.width, self.investText.height
        self.cwidth, self.cheight = self.width//2, self.height//2
        self.area, self.carea = self.width*self.height, (self.cwidth)*(self.cheight)
        self.pels = [self.area, self.carea, self.carea]
        self.totalpels = np.sum(self.pels)
        self.pelCumsum = np.cumsum(self.pels)
        self.poc_dic = self.makePOC_dic()
        self.depth = 10
        if condition_dic is None:
            self.condition_dic = {PictureFormat.ORIGINAL:self.ORIGINAL,
                                  PictureFormat.PREDICTION:self.PREDICTION,
                                  PictureFormat.RECONSTRUCTION:self.RECON,
                                  PictureFormat.UNFILTEREDRECON:self.UNFILTERED
                                  }
        else:
            self.condition_dic = condition_dic
        self.logger = LoggingHelper.get_instance(always=True).logger


    def makePOC_dic(self):
        filelist = os.listdir(self.org_path)
        pocdic = {}
        for f in filelist:
            intf = (int)(f.split('.bin')[0])
            pocdic[intf] = imginfo(*[os.path.join(x, f) for x in self.path_list])
        return pocdic

    def PrintPSNR(self, name, pic_buffer):
        try:
            for i in [PictureFormat.PREDICTION, PictureFormat.RECONSTRUCTION, PictureFormat.UNFILTEREDRECON]:
                if i in pic_buffer:
                    self.logger.debug("  %s %s PSNR : [Y : %s], [U : %s], [V : %s]"
                                      % (
                                          name, PictureFormat.INDEX_DIC[i],
                                          self.getPSNR(pic_buffer[PictureFormat.ORIGINAL]['Y'], pic_buffer[i]['Y']),
                                          self.getPSNR(pic_buffer[PictureFormat.ORIGINAL]['Cb'], pic_buffer[i]['Cb']),
                                          self.getPSNR(pic_buffer[PictureFormat.ORIGINAL]['Cr'], pic_buffer[i]['Cr'])))

        except Exception as e:
            self.logger.error(e)
            self.logger.error('   %s cannot calc PSNR' % (name))
            raise

    def getDataByTrace(self):
        self.logger.info("%s binfile get training set.." % self.root)
        for poc, img in self.poc_dic.items():
            basename = '{}_POC{}'.format(self.basename, poc)
            try:
                tmp_pic_buff = {}
                for key, value in self.condition_dic.items():
                    if value:
                        y, cb, cr = self.imgUnpack(os.path.join(self.root, PictureFormat.INDEX_DIC[key], str(poc)+'.bin'))
                        tmp_pic_buff[key] = {'Y':y, 'Cb':cb, 'Cr':cr}
                        np.savez(os.path.join(self.Datasetpath, PictureFormat.INDEX_DIC[key], basename),
                                            Y=y, Cb=cb, Cr=cr)
                if self.PRINT_PSNR:
                    self.PrintPSNR(basename, tmp_pic_buff)
                del tmp_pic_buff

                blockInPOC = investPOCBlocks(self.investText, poc)
                os.makedirs(os.path.join(self.Datasetpath, 'BLOCK', basename), exist_ok=True)
                blockpath = os.path.join(self.Datasetpath, 'BLOCK', basename)
                for key, value in blockInPOC.blockdic.items():
                    if value:
                        np.save(os.path.join(blockpath, key), np.array(value, dtype=blockInPOC.block_type[key]))
                # ppsparam = [*blockInPOC.ppsdic.items()]
                np.savez(os.path.join(blockpath, 'PPSParam.npz'), **blockInPOC.ppsdic)
            except Exception as e:
                self.logger.error("NAME : <{}>, {}".format(basename, e))
                raise

    def getPSNR(self, org, control):
        if self.depth == 10:
            MAX_VALUE = 1023.0
        else:
            MAX_VALUE = 255.0
        try:
            mse = np.square(np.subtract(org, control).astype(np.int32)).mean()
            if mse == 0:
                return 100
            return 10 * math.log10((MAX_VALUE * MAX_VALUE) / mse)
        except:
            msum = np.array([])
            for i in range(len(org)):
                msum = np.concatenate((msum, np.square(np.subtract(org[i], control[i])).astype(np.int32).flatten()), axis=0)
            return 10 * math.log10((MAX_VALUE * MAX_VALUE) / msum.mean())


    def imgUnpack(self, imgpath):
        with open(imgpath, 'rb') as img:
            if self.depth == 10:
                endian = '<'
                fmt = 'h'
                perbyte = 2
            else:
                assert 0, 'Not support 8bit'

            image = np.array(struct.unpack(endian + str(self.totalpels) + fmt,
                                           img.read((perbyte) * self.totalpels)),
                             dtype='int16')
            image = np.split(image, self.pelCumsum)
            return image[0].reshape(self.height, self.width), image[1].reshape(self.cheight, self.cwidth), image[2].reshape(self.cheight, self.cwidth)

    def DownSamplingLuma(self, lumaPic):
        pos00 = lumaPic[::2, ::2, :]
        pos10 = lumaPic[1::2, ::2, :]
        return (pos00 + pos10 + 1) >> 1

    def UpSamplingChroma(self, UVPic):
        return UVPic.repeat(2, axis=0).repeat(2, axis=1)

    def saveImage(self, YUV, real_tulist, candi_list, name, select_mark=True, poc=''):
        if poc:
            poc = '_' + poc
        uv = self.UpSamplingChroma(np.concatenate((YUV[1].reshape(1, YUV[1].shape[0], YUV[1].shape[1]),
                                                   YUV[2].reshape(1, YUV[2].shape[0], YUV[2].shape[1])),
                                                  axis=0).transpose((1, 2, 0)))
        y = YUV[0].reshape((1, YUV[0].shape[0], YUV[0].shape[1])).transpose((1, 2, 0))
        YUV = np.concatenate((y, uv), axis=2)
        if self.depth == 10:
            YUV = np.uint8(YUV >> 2)
        else:
            YUV = np.uint8(YUV)

        # 0: width, 1: height, 2: x_pos 3: y_pos, 4 : qp, 5 : mode
        for tu in real_tulist.T:
            YUV[tu[3]:tu[3] + tu[1], tu[2], 0] = 0
            YUV[tu[3], tu[2]:tu[2] + tu[0], 0] = 0
            if tu[2] + tu[0] < YUV.shape[1] and tu[3] + tu[1] < YUV.shape[0]:
                YUV[tu[3] + tu[1], tu[2]:tu[2] + tu[0], 0] = 0
                YUV[tu[3]:tu[3] + tu[1], tu[2] + tu[0], 0] = 0
        if select_mark:
            for tu in candi_list.T:
                YUV[tu[3]:tu[3] + tu[1], tu[2]:tu[2] + tu[0], 1] = self.COLOR_BLUE[1]
                YUV[tu[3]:tu[3] + tu[1], tu[2]:tu[2] + tu[0], 2] = self.COLOR_BLUE[2]

        rgbimg = Image.fromarray(YUV, 'YCbCr')
        rgbimg.convert('RGBA').save(os.path.join(self.sample_path, name + self.name + poc + ".png"))
        # self.logger.info("%s image file save." % self.name)
        # plt.imshow(rgbimg)
        # plt.show()
        # rgbimg.show()

# r = imgInfo('C:\\Users\\YangwooKim\\Desktop\\Codec\\VVC-8.0\\VVCSoftware_VTM-VTM-8.0-getdata\\VVCSoftware_VTM-VTM-8.0-getdata\\build\\source\\App\\DecoderApp\\RA_BQTerrace_1920x1080_60_8bit_32_RS0')
# r.getDataByTrace()

class SplitManager:
    cfg = Config('./build_dataset/data_config.yml', logger)
    rasreg = re.compile('.*_RS(?P<ras>\d+).')
    def __init__(self):
        self.logger = LoggingHelper.get_instance(always=True).logger

        self.corenum = self.cfg.PARALLEL_DECODE
        self.traininglist = self.getFileListsFromList(self.cfg.TRAINING_BIN_PATH, pattern='.bin', isfiltering=True)
        self.testlist = self.getFileListsFromList(self.cfg.TEST_BIN_PATH, pattern='.bin', isfiltering=True)
        self.validationlist = self.getFileListsFromList(self.cfg.VALIDATION_BIN_PATH, pattern='.bin', isfiltering=True)

        self.datatype = {
            'Training': datatype('Training', self.traininglist, 1),
            'Validation': datatype('Validation', self.validationlist, 2),
            'Test': datatype('Test', self.testlist, 0)}
        self.video_orglist = self.getFileListsFromList(self.cfg.VIDEO_ORG_PATH, pattern='.yuv')
        self.png_orglist = self.getFileListsFromList(self.cfg.PNG_ORG_PATH, pattern='.yuv')
        self.logpath = self.cfg.DECODE_LOG_PATH
        os.makedirs(self.logpath, exist_ok=True)
        # self.myftp = self.tryReconnectFTP()
        # self.seq = self.initSeqeuences()

    def getFileListsFromList(self, list, pattern='.bin', isfiltering=False):
        filelist = []
        for files in list:
            filelist += myUtil.getFileList(files, pattern)

        return filelist

    def initCommand(self, filelist, isTraining):
        seqs = []
        bdDic = {}
        # frameDic = {}
        # framenum = 0
        with open("./build_dataset/Sequences.csv", 'r') as reader:
            # with open("./SequenceSetting/CTCSequences.csv", 'r') as reader:
            data = reader.read()
            lines = data.strip().split('\n')
            for line in lines:
                seqs.append(line.split(','))
        seqs = seqs[1:]
        for seq in seqs:
            # if seq[0].split("_")[0].lower() != "netflix":
            bdDic[seq[0].lower()] = seq[2]
            # else:
            #
            #     bdDic[''.join(seq[0].split('_')[1:]).lower()] = seq[2]
            # frameDic[seq[0].lower()] = seq[8]
        commands = []
        png_org_list = []
        video_org_list = []
        for org in self.png_orglist:
            png_org_list.append('_'.join(str(os.path.basename(org).split('.yuv')[0]).split('_')[:3]).lower())
        for org in self.video_orglist:
            tmp = str(os.path.basename(org).split(".yuv")[0])
            if tmp.split("_")[0].lower() != "netflix":
                tmp = '_'.join(tmp.split("_")[:3]).lower()
            else:
                tmp = '_'.join(tmp.split("_")[:4]).lower()
            video_org_list.append(tmp)

        for binfile in filelist:
            if os.path.basename(binfile).split('_')[1] == 'PNG':
                for tmp, org in zip(png_org_list, self.png_orglist):
                    if tmp == '_'.join(os.path.basename(binfile).split('_')[2:5]):
                        command = self.cfg.DECODER_PATH + ' -b ' + binfile + ' -i ' + org + ' -bd 8'
                        commands.append((os.path.basename(binfile), command))
                        break
            else:
                for tmp, org in zip(video_org_list, self.video_orglist):
                    if os.path.basename(binfile).lower().split('_')[1] != 'netflix':
                        if '_'.join(os.path.basename(binfile).lower().split("_")[1:4]).lower() == tmp:
                            # command = Decoderpath + ' -b ' + binfile + ' -i ' + org + ' -bd ' + bdDic[tmp]
                            command = self.cfg.DECODER_PATH + ' -b ' + binfile + ' -i ' + org + ' -bd ' + bdDic[tmp]
                            commands.append((os.path.basename(binfile), command))
                            break
                    else:
                        if '_'.join(os.path.basename(binfile).lower().split("_")[1:5]).lower() == tmp:
                            # command = Decoderpath + ' -b ' + binfile + ' -i ' + org + ' -bd ' + bdDic[tmp]
                            command = self.cfg.DECODER_PATH + ' -b ' + binfile + ' -i ' + org + ' -bd ' + bdDic[tmp]
                            commands.append((os.path.basename(binfile), command))
                            break
        if isTraining == LearningIndex.TRAINING:
            tmp_com = []
            for c in commands:
                ras = self.getRAS(c[0])
                if ras is None:
                    tmp_com.append(c)
                else:
                    if ras%2 == 0:
                        tmp_com.append(c)
            commands = tmp_com
        print("command len : {}".format(len(commands)))


        for i in range(len(commands)):
            commands[i] = (commands[i][0], commands[i][1] + " --TraceFile=DecTrace.txt --TraceRule=\"D_BLOCK_STATISTICS_ALL:POC>=0\"")
        return commands

    def runDecoder(self, command):
        (name, command) = command
        logpath = name.replace(".bin", ".log")
        logpath = os.path.join(self.logpath, logpath)
        self.logger.info(command)
        self.logger.info("%s start" % name)
        with open(logpath, 'w') as fp:
            sub_proc = subprocess.Popen(command, stdout=fp)
            sub_proc.wait()
        self.logger.info("%s done" % name)
        return name.replace(".bin", "")

    def runThreading(self, command, isTraining):
        name = self.runDecoder(command)
        splitimg = imgInfo(name, os.path.join(self.cfg.DATASET_PATH, LearningIndex.INDEX_DIC[isTraining]), self.getconditionduc(isTraining))
        splitimg.getDataByTrace()
        try:
            shutil.rmtree(name)
        except Exception as e:
            self.logger.error(e)


    def getconditionduc(self, isTraining):
        if(isTraining==LearningIndex.TRAINING):
            return {PictureFormat.ORIGINAL:1, PictureFormat.PREDICTION:0, PictureFormat.RECONSTRUCTION:0, PictureFormat.UNFILTEREDRECON:1}
        return {PictureFormat.ORIGINAL:1, PictureFormat.PREDICTION:1, PictureFormat.RECONSTRUCTION:1, PictureFormat.UNFILTEREDRECON:1}

    # def CalculateTargetNum(self, cur_num, target_num, remain):
    #     return (target_num - cur_num) //  remain

    @staticmethod
    def extend_list(_list):
        temp = []
        for i in _list:
            if i is None:
                continue
            assert isinstance(i, list)
            temp.extend(i)
        return temp

    def getDataset(self, kind_of_data, obj_POC=100):
        commands = self.initCommand(self.datatype[kind_of_data].binlist, self.datatype[kind_of_data].opt_num)
        with Pool(self.corenum) as pool:
            for i, _ in enumerate(pool.starmap(self.runThreading,
                                             zip(commands, [self.datatype[kind_of_data].opt_num] * len(commands)))):
                print("Progress {} / {}".format(i+1, len(commands)))
            # pool.starmap(self.runThreading,
            #          zip(commands, [self.datatype[kind_of_data].opt_num] * len(commands)))
        return

    def getRAS(self, name):
        if self.rasreg.match(name) is None:
            return None
        return int(self.rasreg.match(name).group('ras'))

    @staticmethod
    def initHeaderAndWriteCSV(dir, header, csv_list):
        if not os.path.exists(dir):
            with open(dir, 'w', newline='') as f:
                headerlinewriter = csv.writer(f)
                headerlinewriter.writerow(header)
                for row in csv_list:
                    headerlinewriter.writerow(row)
        else:
            with open(dir, 'a', newline='') as f:
                headerlinewriter = csv.writer(f)
                for row in csv_list:
                    headerlinewriter.writerow(row)






if __name__ == '__main__':
    # os.chdir("../")
    print(os.getcwd())
    sp = SplitManager()
    sp.getDataset('Training')
    sp.getDataset('Validation')
    sp.getDataset('Test')

