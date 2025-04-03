import torch
import torch.nn as nn
from transformer_vq.nn.types import Dtype


class LayerNorm(nn.Module):
    def __init__(self,
                 input_dim: int,
                 param_dtype: Dtype,
                 center: bool = False,  # RMS norm默认不进行中心化
                 norm: bool = True,
                 gain: bool = True,
                 bias: bool =True):
        
        super(LayerNorm, self).__init__()
        self.input_dim = input_dim
        self.param_dtype = param_dtype
        self.center = center
        self.norm = norm
        self.gain = gain
        self.bias = bias

        if self.gain:
            self.g = nn.Parameter(torch.ones(self.input_dim, dtype=self.param_dtype))
        if self.bias:
            self.b = nn.Parameter(torch.zeros(self.input_dim, dtype=self.param_dtype))

    def forward(self, x, eps=1e-6):
        dtype = x.dtype
        x = x.to(torch.float32)  # 保证计算精度

        if self.center:
            x = x - x.mean(dim=-1, keepdim=True)
        if self.norm:
            x = x * torch.rsqrt(eps + x.pow(2).mean(dim=-1, keepdim=True))

        if self.gain:
            x = x * self.g
        if self.bias:
            x = x + self.b

        return x.to(dtype)