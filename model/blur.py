import torch.nn as nn
import torch
from torch.nn.functional import pad,conv2d
class BlurClass(nn.Module):
    def __init__(self, kernel,kernelTran):
        super(BlurClass, self).__init__()
        self.register_buffer('bk',kernel)
        self.register_buffer('bkt',kernelTran)
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
        with torch.no_grad():
            k = torch.stack([k, k, k], dim=1).squeeze(2).permute(1, 0, 2, 3)
            x = pad(x, pad=((k.shape[-2] - 1) // 2,(k.shape[-2] - 1) // 2, (k.shape[-1] - 1) // 2,(k.shape[-1] - 1) // 2), mode='circular')
            x = conv2d(x, k, groups=x.shape[1])
        return x