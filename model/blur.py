import torch.nn as nn
import torch
from torch.nn.functional import pad,conv2d
class BlurClass(nn.Module):
    def __init__(self, kernel,kernelTran):
        super(BlurClass, self).__init__()
        self.bk=kernel
        self.bkt=kernelTran
    def grad(self, x,y):
        with torch.no_grad():
            delta_g = self.imfilter(self.imfilter(x, self.bk) - y, self.bkt)
        return delta_g
    def fwd_bwd(self,x):
        with torch.no_grad():
            g = self.imfilter(self.imfilter(x, self.bk), self.bkt)
        return g
    @staticmethod
    def imfilter(x, k):
        '''
        x: image, NxcxHxW
        k: kernel, cx1xhxw
        '''
        k=k.to(x.device)
        x = pad(x, pad=((k.shape[-2] - 1) // 2,(k.shape[-2] - 1) // 2, (k.shape[-1] - 1) // 2,(k.shape[-1] - 1) // 2), mode='circular')
        x = conv2d(x, k, groups=x.shape[1])
        k=k.cpu()
        #torch.cuda.empty_cache()
        return x.detach()
