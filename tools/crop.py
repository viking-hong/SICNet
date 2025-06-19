
import os
import numpy as np
import cv2
import copy
from PIL import Image
from multiprocessing import Pool
from functools import partial

def custombasename(fullname):
    return os.path.basename(os.path.splitext(fullname)[0])

def GetFileFromThisRootDir(dir,ext = None):
  allfiles = []
  needExtFilter = (ext != None)
  for root,dirs,files in os.walk(dir):
    for filespath in files:
      filepath = os.path.join(root, filespath)
      extension = os.path.splitext(filepath)[1][1:]
      if needExtFilter and extension in ext:
        allfiles.append(filepath)
      elif not needExtFilter:
        allfiles.append(filepath)
  return allfiles
def split_single_warp(name, split_base, rate, extent):
    split_base.SplitSingle(name, rate, extent)
class splitbase():
    def __init__(self,
                 srcpath,
                 dstpath,
                 gap=200,
                 subsize=1024,
                 ext='.jpg',
                 padding=False,
                 num_process=32,
                 mask = False):
        self.srcpath = srcpath
        self.outpath = dstpath
        self.gap = gap
        self.subsize = subsize
        self.slide = self.subsize - self.gap
        self.srcpath = srcpath
        self.dstpath = dstpath
        self.ext = ext
        self.padding = padding
        self.pool = Pool(num_process)
        self.mask = mask
        if not os.path.isdir(self.outpath):
            os.mkdir(self.outpath)

    def saveimagepatches(self, img, subimgname, left, up, ext='.jpg'):
        subimg = copy.deepcopy(img[up: (up + self.subsize), left: (left + self.subsize)])
        outdir = os.path.join(self.dstpath, subimgname + ext)
        h, w, c = np.shape(subimg)
        if (self.padding):
            outimg = np.zeros((self.subsize, self.subsize, 3))
            outimg[0:h, 0:w, :] = subimg
            cv2.imwrite(outdir, outimg)
        else:
            cv2.imwrite(outdir, subimg)

    def savemaskpatches(self, img, subimgname, left, up, ext='.jpg'):
        subimg = copy.deepcopy(img[up: (up + self.subsize), left: (left + self.subsize)])
        outdir = os.path.join(self.dstpath, subimgname + ext)
        h, w, c = np.shape(subimg)
        if (self.padding):
            outimg = np.zeros((self.subsize, self.subsize, 1))
            outimg[0:h, 0:w, :] = subimg
            outputImg = Image.fromarray(np.uint8(outimg)).convert('L')
            outputImg.save(outdir)
            #cv2.imwrite(outdir, outimg)
        else:
            #cv2.imwrite(outdir, subimg)
            outputImg = Image.fromarray(np.uint8(subimg)).convert('L')
            outputImg.save(outdir)

    def SplitSingle(self, name, rate, extent):
        img = cv2.imread(os.path.join(self.srcpath, name + extent))
        #assert np.shape(img) != ()

        if (rate != 1):
            resizeimg = cv2.resize(img, None, fx=rate, fy=rate, interpolation=cv2.INTER_CUBIC)
        else:
            resizeimg = img
        outbasename = name + '__' + str(int(rate*100)) + '__'

        weight = np.shape(resizeimg)[1]
        height = np.shape(resizeimg)[0]

        # if (max(weight, height) < self.subsize/2):
        #     return

        left, up = 0, 0
        while (left < weight):
            if (left + self.subsize >= weight):
                left = max(weight - self.subsize, 0)
            up = 0
            while (up < height):
                if (up + self.subsize >= height):
                    up = max(height - self.subsize, 0)
                subimgname = outbasename + str(left) + '___' + str(up)
                if self.mask == False:
                    self.saveimagepatches(resizeimg, subimgname, left, up)
                else:
                    self.savemaskpatches(resizeimg, subimgname, left, up)
                if (up + self.subsize >= height):
                    break
                else:
                    up = up + self.slide
            if (left + self.subsize >= weight):
                break
            else:
                left = left + self.slide

    def splitdata(self, rate):

        imagelist = GetFileFromThisRootDir(self.srcpath)
        imagenames = [custombasename(x) for x in imagelist if (custombasename(x) != 'Thumbs')]

        # worker = partial(self.SplitSingle, rate=rate, extent=self.ext)
        worker = partial(split_single_warp, split_base=self, rate=rate, extent=self.ext)
        self.pool.map(worker, imagenames)
        #
        for name in imagenames:
            self.SplitSingle(name, rate, self.ext)
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)


if __name__ == '__main__':
    #scales = [1.0, 0.5]
    split = splitbase(r'H:\Datasets\Raft1/image/',
                      r'H:\Datasets\Raft1/image1/',
                      gap=200,
                      subsize=512,
                      num_process=8,
                      mask=False
                      )
    split.splitdata(2.00)
    split.splitdata(1.75)
    split.splitdata(1.50)
    split.splitdata(1.25)
    split.splitdata(1.00)
    #split.splitdata(0.50)
    #split.splitdata(0.25)

    split_mask = splitbase(r'H:\Datasets\Raft1/mask/',
                      r'H:\Datasets\Raft1/mask1/',
                      gap=200,
                      subsize=512,
                      num_process=8,
                      mask=True
                      )
    split_mask.splitdata(2.00)
    split_mask.splitdata(1.75)
    split_mask.splitdata(1.50)
    split_mask.splitdata(1.25)
    split_mask.splitdata(1.00)
    #split_mask.splitdata(0.50)
    #split_mask.splitdata(0.25)

