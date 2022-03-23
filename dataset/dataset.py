from genericpath import isfile
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
from omegaconf import ListConfig
from model.blur import BlurClass
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
    if isinstance(trainPath,ListConfig):
        for p in trainPath:
            imList = listdir(p)
            for imName in imList:
                if isfile(join(p, imName)):
                    patchedImg = patchImg(
                        join(p, imName), patchSize=patchSize, device=device, patchPerImg=patchPerImg)
                    patchedImgList.append(patchedImg)
        trainImgs = torch.cat(patchedImgList,dim=0)
    else:
        imList = listdir(trainPath)
        for imName in imList:
            if isfile(join(trainPath, imName)):
                patchedImg = patchImg(
                    join(trainPath, imName), patchSize=patchSize, device=device, patchPerImg=patchPerImg)
                patchedImgList.append(patchedImg)
        trainImgs = torch.cat(patchedImgList, dim=0)
    patchedImgList = []
    if isinstance(valPath, ListConfig):
        for p in valPath:
            imList = listdir(p)
            for imName in imList:
                if isfile(join(p, imName)):
                    patchedImg = patchImg(
                        join(p, imName), patchSize=patchSize, device=device, patchPerImg=patchPerImg)
                    patchedImgList.append(patchedImg)
        valImgs = torch.cat(patchedImgList, dim=0)
    else:
        imList = listdir(valPath)
        for imName in imList:
            if isfile(join(valPath, imName)):
                patchedImg = patchImg(
                    join(valPath, imName), patchSize=patchSize, device=device, patchPerImg=patchPerImg)
                patchedImgList.append(patchedImg)
        valImgs = torch.cat(patchedImgList, dim=0)
    
    return trainImgs.detach(),valImgs.detach()


class BlurDatasetHD(Dataset):
    def __init__(self, path):
        super(BlurDatasetHD, self).__init__()
        self.lst=listdir(path)
        self.path=path

    def __len__(self):
        return len(self.lst)

    def __getitem__(self, item):
        return torch.load(join(self.path,self.lst[item]),map_location='cpu')

def patchNSave(imPath, patchSize, device, patchPerImg,savePath,fixedNoise,*args):
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
    saveIdx=np.random.choice(
        np.arange(patchedImg.shape[0]), min(patchedImg.shape[0], patchPerImg), replace=False)
    
    for ii,i in enumerate(saveIdx):
        im = patchedImg[i, ...]
        imName = imPath.split('/')[-1].split('.')[0]
        if fixedNoise:
            y=BlurClass.imfilter(
                im.unsqueeze(0), args[0])+torch.FloatTensor(im.size()).normal_(0, std=args[1]/255.)
            y=y.squeeze()
            torch.save((im, y), join(savePath, f'{imName}_{ii}.pt'))
        else:
            torch.save(im, join(savePath, f'{imName}_{ii}.pt'))
    

def dataPrepareHD(trainPath, valPath, patchSize, patchPerImg, fixedNoise, savePath, device, *args):
    mkdir(join(savePath,'train'))
    mkdir(join(savePath,'val'))

    if isinstance(trainPath, ListConfig):
        for p in trainPath:
            imList = listdir(p)
            for imName in imList:
                if isfile(join(p, imName)):
                    patchNSave(
                        join(p, imName), patchSize, device, patchPerImg, join(savePath, 'train'), fixedNoise, *args)
    else:
        imList = listdir(trainPath)
        for imName in imList:
            if isfile(join(trainPath, imName)):
                patchNSave(
                    join(trainPath, imName), patchSize, device, patchPerImg, join(savePath, 'train'), fixedNoise, *args)
    
    if isinstance(valPath, ListConfig):
        for p in valPath:
            imList = listdir(p)
            for imName in imList:
                if isfile(join(p, imName)):
                    patchNSave(
                        join(p, imName), patchSize, device, patchPerImg, join(savePath,'val'), fixedNoise, *args)
    else:
        imList = listdir(valPath)
        for imName in imList:
            if isfile(join(valPath, imName)):
                patchNSave(
                    join(valPath, imName), patchSize, device, patchPerImg, join(savePath,'val'),fixedNoise,*args)

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
