import torch
from torch.nn.functional import pad,unfold,fold
def patchify(img,kSize,stride):
    _, _, H, W = img.shape
    vpad = stride-(H-kSize) % stride if (H -kSize) % stride != 0 else 0
    hpad = stride-(W-kSize) % stride if (W -kSize) % stride != 0 else 0
    padded = pad(img, (0, vpad, 0, hpad), 'reflect')
    patched = unfold(padded, kernel_size=kSize,
                    stride=stride).permute((0, 2, 1))
    mask = fold(unfold(torch.ones_like(padded), kernel_size=kSize,
                    stride=stride), (padded.shape[-2:]), kSize, stride=stride)
    patched = patched.reshape(
        patched.shape[1], padded.shape[1], kSize, kSize).detach()
    return patched,mask,H,W

def unify(img,mask,imH,imW,kSize,stride):
    unified = fold(img.reshape(1, img.shape[0], -1).permute(
        (0, 2, 1)), (mask.shape[-2:]), kSize, stride=stride)
    unified=unified/mask
    unified = unified[:, :, 0:imH, 0:imW]
    return unified
    
