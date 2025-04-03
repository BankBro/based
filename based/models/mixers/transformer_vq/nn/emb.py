import dataclasses
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_vq.nn.types import TransformerConfig


class Embeddings(nn.Module):
    def __init__(self, config: TransformerConfig):
        # TODO: 简单点可以直接调用 nn.Embedding
        super(Embeddings, self).__init__()
        self.config = config
        self.apply_config()
        self.embs = nn.Parameter(torch.empty(self.n_vocab, self.d_model, dtype=self.param_dtype))
        self.bias_out = nn.Parameter(torch.empty(self.n_vocab, dtype=self.param_dtype))
        self.reset_parameters()

    def reset_parameters(self):
        self.e_init(self.embs)
        self.b_init(self.bias_out)
        # nn.init.normal_(self.embs, mean=0.0, std=self.e_init.stddev)
        # nn.init.constant_(self.bias_out, 0.0)

    def apply_config(self):
        for k, v in self.config.__dict__.items():
            setattr(self, k, v)

    def forward(self, x):
        x = torch.nn.functional.embedding(x, self.embs)  # B
        return x.type(self.dtype)

    def logits(self, x):  # BUD
        x = x.type(torch.float32)
        x = torch.matmul(x, self.embs.t().type(torch.float32))  # BUC
        x += self.bias_out.type(torch.float32).unsqueeze(0).unsqueeze(0)
        return x  # BUC