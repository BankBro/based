import dataclasses
import torch
import torch.nn as nn

from transformer_vq.nn.types import TransformerConfig


def get_sinusoid_embs(length, width, lam, flip, start=0):
    pos_seq = start + torch.arange(length)
    assert pos_seq.shape == (length,)

    inv_lams = 1 / (lam ** (torch.arange(0, width, 2) / width))
    pre = pos_seq[:, None] * inv_lams[None, :]  # (length, width/2)

    sin = torch.sin(pre)
    cos = torch.cos(pre)

    cat = torch.cat([sin, cos], dim=-1)  # TODO: 正弦余弦交错
    assert cat.shape == (length, width)

    if not flip:
        return cat
    return torch.flip(cat, dims=[0])  # (length, width)  UD


class ScaledSin(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(ScaledSin, self).__init__()
        self.config = config
        self.apply_config()
        self.scale = nn.Parameter(torch.empty(1, dtype=torch.float32))
        nn.init.normal_(self.scale)

    def apply_config(self):
        for k, v in dataclasses.asdict(self.config).items():
            setattr(self, k, v)

    def forward(self, length, offset):
        embs = get_sinusoid_embs(
            length=length, start=offset, width=self.d_model, lam=self.pe_lam, flip=False
        )  # UD
        return (self.scale * embs).to(self.dtype)