from torch.utils.data.dataset import Dataset
import torch
from sklearn.feature_extraction.image import extract_patches_2d
from os import listdir,mkdir,system
from os.path import join,isdir
import sys
from PIL.Image import open as imopen
import numpy as np
from torch.nn.functional import pad,unfold
from scipy.io import loadmat
class BlurDataset(Dataset):
    def __init__(self, gdt, y=None):
        super(BlurDataset, self).__init__()

        self.gdt = gdt
        self.y = y

    def __len__(self):
        return self.gdt.shape[0]

    def __getitem__(self, item):
        if self.y!=None:
            return self.gdt[item], self.y[item]
        return self.gdt[item]


def patchImg(imPath, patchSize, device, patchPerImg):
    img = torch.from_numpy(np.asarray(
        imopen(imPath)).transpose((2, 0, 1))).float().to(device)/255.

    _, H, W = img.shape
    vpad = patchSize - \
        (H-patchSize) % patchSize if (H -
                                      patchSize) % patchSize != 0 else 0
    hpad = patchSize - \
        (W-patchSize) % patchSize if (W -
                                      patchSize) % patchSize != 0 else 0
    img = pad(img, (0, vpad, 0, hpad), 'reflect').unsqueeze(0)
    patchedImg = unfold(img, kernel_size=patchSize,
                        stride=patchSize).permute((0, 2, 1))
    patchedImg = patchedImg.reshape(
        patchedImg.shape[1], 3, patchSize, patchSize).detach().cpu()
    patchedImg = patchedImg[np.random.choice(
        np.arange(patchedImg.shape[0]), min(patchedImg.shape[0], patchPerImg),replace=False), ...]
    return patchedImg

def dataPrepare(trainPath,valPath,patchSize,patchPerImg,device):
    #train
    patchedImgList = []
    if isinstance(trainPath,list):
        for p in trainPath:
            imList=listdir(p)
            for imName in imList:
                patchedImg = patchImg(
                    join(p, imName), patchSize=patchSize, device=device, patchPerImg=patchPerImg)
                patchedImgList.append(patchedImg)
        trainImgs=torch.cat(patchedImgList,dim=0)
    else:
        imList = listdir(trainPath)
        for imName in imList:
            patchedImg = patchImg(
                join(trainPath, imName), patchSize=patchSize, device=device, patchPerImg=patchPerImg)
            patchedImgList.append(patchedImg)
        trainImgs = torch.cat(patchedImgList, dim=0)
    patchedImgList = []
    if isinstance(valPath, list):
        for p in valPath:
            imList = listdir(p)
            for imName in imList:
                patchedImg = patchImg(
                    join(p, imName), patchSize=patchSize, device=device, patchPerImg=patchPerImg)
                patchedImgList.append(patchedImg)
        valImgs = torch.cat(patchedImgList, dim=0)
    else:
        imList = listdir(valPath)
        for imName in imList:
            patchedImg = patchImg(
                join(valPath, imName), patchSize=patchSize, device=device, patchPerImg=patchPerImg)
            patchedImgList.append(patchedImg)
        valImgs = torch.cat(patchedImgList, dim=0)
    
    return trainImgs,valImgs


def get_matrix(path,tp):

    # Load kernels
    # kernels = sio.loadmat(config['kernal_datapath'])['kernels'][0]
    # blur_kernel = kernels[kernel_tp].astype(np.float64)
    kernels = loadmat(path,
                      squeeze_me=True)[tp]
    blur_kernel = kernels.astype(np.float64)
    blur_kernel_trans = blur_kernel[::-1, ::-1]

    # Convert from numpy to torch:
    bk = torch.from_numpy(blur_kernel.copy()).type(
        torch.FloatTensor).unsqueeze(0).unsqueeze(0)
    bkt = torch.from_numpy(blur_kernel_trans.copy()).type(
        torch.FloatTensor).unsqueeze(0).unsqueeze(0)
    bk = torch.stack([bk, bk, bk], dim=1).squeeze(2).permute(1, 0, 2, 3)
    bkt = torch.stack([bkt, bkt, bkt], dim=1).squeeze(2).permute(1, 0, 2, 3)
    return bk, bkt
