import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

def sg(x):
    """
    Stop gradient propagation.
    """
    return x.detach()

def st(x):
    """
    Pass gradient through by subtracting detached version.
    """
    return x - sg(x)

def maybe_remat(module, enabled):
    """
    Optionally wrap a module with gradient checkpointing.

    Args:
        module (nn.Module): The module to be wrapped.
        enabled (bool): Whether to enable gradient checkpointing.

    Returns:
        nn.Module: The wrapped module if enabled, otherwise the original module.
    """
    if enabled:
        return RematModule(module)
    else:
        return module


class RematModule(nn.Module):
    def __init__(self, module):
        super(RematModule, self).__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return checkpoint(self.module.forward, *args, **kwargs)