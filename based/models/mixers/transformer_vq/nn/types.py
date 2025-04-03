from dataclasses import fields, dataclass
import torch
import torch.nn as nn
from typing import Any
from typing import Callable
from typing import List

PRNGKey = Any
Shape = List[int]
Dtype = torch.dtype
Initializer = Callable[..., None]  # TODO

@dataclass
class TransformerConfig:
    param_dtype: Dtype      # 模型参数的数据类型，用于初始化和训练过程。
    dtype: Dtype            # 模型在运行时使用的数据类型，用于前向传播等计算过程。
    global_batch_size: int  # B (全局批量大小)
    sequence_len: int       # T (序列总长度)
    update_len: int         # U (更新步长)
    block_len: int          # L (块长度)
    mem_len: int            # M (记忆长度, W = L + M)
    d_model: int            # D (模型隐藏维度)
    d_k: int                # K (码本向量维度)
    d_v: int                # V (值向量维度)
    n_head: int             # H (注意力头数)
    n_code: int             # S (码本大小)
    n_layer: int            # N (层数)
    n_vocab: int            # C (词表大小)
    d_ff: int
    grad_thru_cache: bool
    agg_cache: bool
    pe_abs: bool
    pe_lam: float
    p_dropemb: float
    p_dropsin: float
    p_dropres: float
    p_droplyr: float
    p_nucleus: float
    c_beta: float
    c_gamma: float
    e_tie: bool  # 控制输入嵌入矩阵和输出投影层的权重共享, True: 输入token
                 # 嵌入矩阵(Embeddings)和输出logits投影层(out_proj)共享同一组权重
    e_preln: bool  # True: 在投影到词表空间前进行层归一化
    e_scale: str
    is_train: bool
    e_init: Initializer
    w_init: Initializer
    r_init: Initializer
    b_init: Initializer
    no_emb: bool = False  # 控制模型是否跳过嵌入层, False表示输入没有嵌入, 会使用嵌入层

    @classmethod
    def create(cls, **kwargs):
        signature = {field.name: field.type for field in fields(TransformerConfig)}
        filtered = {k: v for k, v in kwargs.items() if k in signature}

        if isinstance(filtered["param_dtype"], str):
            filtered["param_dtype"] = torch.dtype(filtered["param_dtype"])

        if isinstance(filtered["dtype"], str):
            filtered["dtype"] = torch.dtype(filtered["dtype"])

        for k, v in filtered.items():
            if signature[k] is bool and v in {0, 1}:
                filtered[k] = bool(v)

        filtered["e_init"] = lambda tensor: nn.init.normal_(tensor, mean=0.0, std=1.0)
        filtered["w_init"] = lambda tensor: nn.init.xavier_normal_(tensor, gain=1.0)
        filtered["r_init"] = lambda tensor: nn.init.xavier_normal_(tensor, gain=1.0)
        filtered["b_init"] = lambda tensor: nn.init.zeros_(tensor)

        return cls(**filtered)
