import bm3d
from help_func.CompArea import PredMode
import numpy as np
from PIL import Image

class BM3D:

    def __init__(self):
        self.psd_dic = {PredMode.MODE_INTRA:{}, PredMode.MODE_INTER:{}}

    def setPsd(self, isIntra, qp, org, noisy):
        if isIntra:
            mode = PredMode.MODE_INTRA
        else:
            mode = PredMode.MODE_INTER
        noise_standard_variance = self.getStd(org, noisy)
        if qp in self.psd_dic[mode]:
            self.psd_dic[mode][qp].append(noise_standard_variance)
        else:
            self.psd_dic[mode][qp] = [noise_standard_variance]

    def getPsd(self, mode, qp):
        if qp not in self.psd_dic[mode]:
            assert 0, '{} not in psd dic'.format(qp)
        return np.mean(self.psd_dic[mode][qp])

    @staticmethod
    def getStd(org, noisy):
        return np.std((org - noisy).reshape(-1))

    def operate_Bm3d(self, image_noisy, mode, qp):
        bm3d.bm3d(image_noisy, sigma_psd=self.getPsd(mode,qp))

def DownSamplingLuma(lumaPic):
    pos00 = lumaPic[::2, ::2, :]
    pos10 = lumaPic[1::2, ::2, :]
    return (pos00 + pos10 + 1) >> 1
im = Image.open('C:\\Users\\YangwooKim\\Desktop\\B201누수_2.jpg')
image_noisy = np.asarray(im).astype(dtype=np.uint16)
image_noisy = DownSamplingLuma(image_noisy)
image_noisy = DownSamplingLuma(image_noisy)
image_noisy = DownSamplingLuma(image_noisy).astype(dtype=np.uint8)
image_noisy = image_noisy.astype('float16')*4
# image_noisy /= 255
# image = Image.fromarray(image_noisy, 'RGB')
# image.show()
# Image.fromarray(image_noisy).show()
# image_noisy = np.random.randn(256,256,3)
denoised_image = bm3d.bm3d(image_noisy, sigma_psd=30/255)/4
denoised_image = Image.fromarray(denoised_image.astype('uint8'), 'RGB')
denoised_image.show()
