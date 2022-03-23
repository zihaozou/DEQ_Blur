import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from os import mkdir
from os.path import join
import numpy as np
import torch
import hydra
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scher
import model.deq as deq
from model.blur import BlurClass
from model.dnn import DnCNN
from model.jacob import jacobinNet
from dataset.dataset import BlurDataset, BlurDatasetHD,dataPrepare,get_matrix,dataPrepareHD
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch.nn import DataParallel
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch.nn.functional import mse_loss
@hydra.main(config_path="conf", config_name="config")
def main(config):
    ## dataset
    if config.data.hard_disk!=None:
        bk, bkt = get_matrix(config.blur.data.kernel_path,
                             config.blur.data.kernel_type)
        dataPrepareHD(config.blur.data.train_path,
                      config.blur.data.val_path, 
                      config.blur.data.patch_size,
                      config.blur.data.patch_per_img,
                      config.blur.data.fixed_noise,
                      config.data.hard_disk,
                      config.blur.train.devices[0],
                      bk, config.blur.data.sigma)
        trainset = BlurDatasetHD(join(config.data.hard_disk, 'train'))
        valset = BlurDatasetHD(join(config.data.hard_disk, 'val'))
    else:
        trainset,valset=dataPrepare(config.blur.data.train_path,config.blur.data.val_path,config.blur.data.patch_size,config.blur.data.patch_per_img,config.blur.train.devices[0])
        torch.cuda.empty_cache()
        bk,bkt=get_matrix(config.blur.data.kernel_path,config.blur.data.kernel_type)
        if config.blur.data.fixed_noise:
            trainset = (trainset, BlurClass.imfilter(
                trainset, bk)+torch.FloatTensor(trainset.size()).normal_(0,std=config.blur.data.sigma/255.))
            valset = (valset, BlurClass.imfilter(
                valset, bk)+torch.FloatTensor(valset.size()).normal_(0, std=config.blur.data.sigma/255.))
        else:
            trainset=[trainset]
            valset=[valset]
        trainset=BlurDataset(*trainset)
        valset = BlurDataset(*valset)
    trainLoader=DataLoader(trainset,batch_size=config.blur.data.batch_size,shuffle=True,num_workers=config.blur.data.num_workers,pin_memory=True,drop_last=True)
    valLoader = DataLoader(valset, batch_size=config.blur.data.batch_size,
                           shuffle=False, num_workers=config.blur.data.num_workers, pin_memory=True)


    ## model
    dObj=BlurClass(bk,bkt)
    rObj = jacobinNet(DnCNN(depth=config.blur.model.cnn.depth, 
                            in_channels=config.blur.model.cnn.in_chans, 
                            out_channels=config.blur.model.cnn.out_chans,
                            init_features=config.blur.model.cnn.init_feats,
                            kernel_size=config.blur.model.cnn.kernel_size))
    ured = deq.URED(dObj=dObj, dnn=rObj,
                    gamma=config.blur.model.red.gamma, tau=config.blur.model.red.tau)
    model=deq.DEQFixedPoint(ured,solver_img=deq.nesterov,solver_grad=deq.anderson,**config.blur.model.deq.kwargs)
    if config.blur.model.cnn.pretrained != None:
        model.module.load_state_dict(torch.load(
            config.blur.model.cnn.pretrained, map_location='cpu'))
    model=model.to(f'cuda:{config.blur.train.devices[0]}')
    model = DataParallel(model, device_ids=config.blur.train.devices)
    

    ## optimizer and scheduler
    optimizer = getattr(optim, config.blur.train.optimizer.type)(model.parameters(),**config.blur.train.optimizer.kwargs)
    scheduler = getattr(lr_scher, config.blur.train.scheduler.type)(
        optimizer, **config.blur.train.scheduler.kwargs)
    
    ## logger
    if config.blur.model.warmup != None:
        warmUpDict = torch.load(
            config.model.warmup, map_location='cpu')
        model.module.load_state_dict(warmUpDict['model'])
        optimizer.load_state_dict(warmUpDict['optimizer'])
        scheduler.load_state_dict(warmUpDict['scheduler'])
    logger=SummaryWriter(log_dir='logs')
    mkdir('ckpts')



    ## train
    for e in (bar:=tqdm(iterable=range(config.blur.train.num_epoches))):
        model.train()
        epochPSNR=0
        for b,batch in enumerate(trainLoader):
            optimizer.zero_grad()
            if config.blur.data.fixed_noise:
                gt,y=batch
            else:
                gt=batch
                y = BlurClass.imfilter(gt, bk)+torch.FloatTensor(gt.size()).normal_(mean=0, std=config.blur.data.sigma/255.)
            gt = gt.to(f'cuda:{config.blur.train.devices[0]}')
            y = y.to(f'cuda:{config.blur.train.devices[0]}')
            predX=model(y)
            loss = mse_loss(predX,gt)
            loss.backward()
            optimizer.step()
            batchPSNR=psnr(gt.detach().cpu().numpy(),predX.detach().cpu().numpy(),data_range=1)
            logger.add_scalar('train/batchPSNR',batchPSNR,e*len(trainLoader)+b)
            logger.add_scalar('train/batchLoss', loss.item(),
                                e*len(trainLoader)+b)

            logger.add_scalar('train/tau',model.module.f.tau.data.item(), e*len(trainLoader)+b)
            epochPSNR+=loss.item()
            if b%15==0:
                bar.set_description(f'PSNR:{batchPSNR:.2f}')
        epochPSNR/=len(trainLoader)
        logger.add_scalar('train/epochPSNR', epochPSNR, e)

        model.eval()
        epochPSNR = 0
        for b,batch in enumerate(valLoader):
            if config.blur.data.fixed_noise:
                gt, y = batch
            else:
                gt = batch
                y = BlurClass.imfilter(
                    gt, bk)+torch.FloatTensor(gt.size()).normal_(mean=0, std=config.blur.data.sigma/255.)
            gt = gt.to(f'cuda:{config.blur.train.devices[0]}')
            y = y.to(f'cuda:{config.blur.train.devices[0]}')
            predX = model(y)
            loss = mse_loss(predX, gt)
            batchPSNR = psnr(gt.detach().cpu().numpy(),
                             predX.detach().cpu().numpy(), data_range=1)
            logger.add_scalar('val/batchPSNR', batchPSNR,
                              e*len(trainLoader)+b)
            logger.add_scalar('val/batchLoss', loss.item(),
                              e*len(trainLoader)+b)
        epochPSNR /= len(trainLoader)
        bestModel = {'model': model.module.state_dict(), 'optimizer': optimizer.state_dict(
        ), 'scheduler': scheduler.state_dict(), 'psnr': epochPSNR}
        torch.save(bestModel, 'ckpts', _use_new_zipfile_serialization=False)

if __name__=='__main__':
    main()
