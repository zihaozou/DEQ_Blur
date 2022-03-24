import torch
import torch.nn as nn
import torch.autograd as autograd
def nesterov(f, x0, max_iter=150):
    """ nesterov acceleration for fixed point iteration. """
    res = []
    imgs = []

    x = x0
    s = x.clone()
    t = torch.tensor(1., dtype=torch.float32)
    for k in range(max_iter):

        xnext = f(s)
        
        # acceleration

        tnext = 0.5*(1+torch.sqrt(1+4*t*t))

        s = xnext + ((t-1)/tnext)*(xnext-x)
        
        # update
        t = tnext
        x = xnext
        
        # res.append((x - s).norm().item()/(1e-5 + x.norm().item()))
        # if (res[-1] < tol):
        #     break

    return x

def anderson(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-5, beta = 1.0):
    """ Anderson acceleration for fixed point iteration. """
    bsz, d, H, W = x0.shape
    X = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
    X[:,0], F[:,0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
    X[:,1], F[:,1] = F[:,0], f(F[:,0].view_as(x0)).view(bsz, -1)
    
    H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
    H[:,0,1:] = H[:,1:,0] = 1
    y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
    y[:,0] = 1
    
    res = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:,:n]-X[:,:n]
        H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
        # (bsz x n)
        alpha = torch.linalg.solve(
            H[:, :n+1, :n+1], y[:, :n+1])[:, 1:n+1, 0]
        
        X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
        F[:,k%m] = f(X[:,k%m].view_as(x0)).view(bsz, -1)
        res.append((F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm().item()))
        if (res[-1] < tol):
            break
    return X[:,k%m].view_as(x0), res, None

class URED(nn.Module):

    def __init__(self, dObj, dnn: nn.Module, gamma,tau):
        #gamma_inti=3e-3, tau_inti=1e-1, batch_size=60, device='cuda:0'):

        super(URED, self).__init__()
        self.dnn = dnn
        self.dObj = dObj
        self.gamma = torch.tensor(gamma, dtype=torch.float32)
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32))

    def denoise(self, n_ipt, create_graph=True, strict=True):
        denoiser = self.dnn(n_ipt, create_graph, strict)
        return denoiser

    def forward(self, n_ipt, n_y, create_graph=False, strict=False):
        #tau = self.tau if self.tau - 0.28 <= 0 else 0.28
        delta_g = self.dObj.grad(n_ipt, n_y)
        xSubD    = torch.abs(self.tau) * (self.dnn(n_ipt, create_graph, strict))
        xnext  =  n_ipt - self.gamma * (delta_g.detach() + xSubD) # torch.Size([1, 1, H, W, 2])
        xnext[xnext<=0] = 0
        # snr_ = compare_snr(xnext, gt).item()
        # print(snr_)
        return xnext

class DEQFixedPoint(nn.Module):
    def __init__(self, f, solver_img, solver_grad, **kwargs):
        super(DEQFixedPoint, self).__init__()
        self.f = f
        self.solver_img = solver_img
        self.solver_grad = solver_grad
        self.kwargs = kwargs
        
    def forward(self, n_y):
        #tauList, gammaList,n_ipt_periter = None, None, None
        #bk,bkt = self.f.dObj.init(gt)
        # compute forward pass and re-engage autograd tape
        z = self.solver_img(lambda z : self.f(z, n_y,  create_graph=False, strict=False), n_y, max_iter=100).detach()
        #torch.cuda.empty_cache()
        z =  self.f(z, n_y,  create_graph=self.training, strict=self.training)
        # set up Jacobian vector product (without additional forward calls)
        if self.training:
            z0 = z.clone().detach().requires_grad_()
            # f0 = self.f(z0, yStoc, emStoc, meas_list, create_graph=create_graph, strict=strict)
            r0 = self.f.gamma * self.f.tau * self.f.denoise(z0)
            def backward_hook(grad):
                if self.hook is not None:
                    self.hook.remove()
                # print(grad.shape)
                fTg = lambda y : y - self.f.gamma*self.f.dObj.fwd_bwd(y)-autograd.grad(r0, z0, y, retain_graph=True)[0] + grad #
                g, self.backward_res, _ = self.solver_grad(fTg, grad, max_iter=50, **self.kwargs)
                return g
            self.hook=z.register_hook(backward_hook)
        return z
