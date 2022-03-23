import torch
from torch import nn
import torch.autograd as autograd


class jacobinNet(nn.Module):
    """Unfold network models, i.e. (online) PnP/RED"""

    def __init__(self, dnn: nn.Module):
        super(jacobinNet, self).__init__()
        self.dnn = dnn

    def forward(self, x, create_graph, strict):
        J = autograd.functional.jacobian(
            self.dnn, x, create_graph=create_graph, strict=strict)

        # N = self.dnn(x)
        # JN = autograd.grad(N, x, grad_outputs=x - N, create_graph=create_graph, retain_graph=retain_graph, only_inputs=only_inputs)[0]
        # Dg = N + JN

        # with torch.no_grad():
        #     self.dnn.eval()
        #     loss_h = self.dnn(x)
        #     # print(loss_h)
        return J  # , loss_h.item()
