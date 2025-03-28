"""
Linear attention in Based. 
"""
import math

import torch
import torch.nn as nn
from einops import rearrange

from based.generation import InferenceParams

########### VARIOUS KERNEL OPTIONS ###########

try:
    from train.csrc.causal_dot_prod import causal_dot_product  # linear attention cuda kernel
    print(f"Successfully imported the causal dot product kernel! ")
except:
    print(f"Could not import the causal dot product kernel... ")
    causal_dot_product = None

try:
    from fla.ops.based import fused_chunk_based, parallel_based
    from fla.ops.based.naive import naive_parallel_based
    print(f"Successfully imported the FLA triton kernels! ")
except:
    print(f"Could not import the FLA triton kernels... ")

try:
    import thunderkittens as tk
    from fla.modules import RMSNorm
    print(f"Successfully imported tk")
except:
    print(f"Please install the based kernel within ThunderKittens")

########### VARIOUS KERNEL OPTIONS ###########

        
class FeatureMap(nn.Module):
    """
    Parent feature map; default is identity function
    """
    def __init__(self, input_dim: int, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        
    def forward(self, x: torch.Tensor):
        """
        Assume x.shape is (batch_size, n_heads, seq_len, head_dim)
        """
        return x

class TaylorExp(FeatureMap):
    """
    Feature map to compute 2nd-order Taylor approx. of exp(q^T k / sqrt(d))
    """
    def __init__(
            self, 
            input_dim: int, 
            **kwargs: any
        ):
        super().__init__(input_dim, **kwargs)
        self.r2  = math.sqrt(2)
        self.rd  = math.sqrt(input_dim)
        self.rrd = math.sqrt(self.rd)
        self.tril_indices = torch.tril_indices(self.input_dim, self.input_dim, -1)
        
    def forward(self, x: torch.Tensor):  # x: BHLF
        # Get 2nd-order terms (rearrange(x * x), '... m n -> ... (m n)')
        x2 = (x.unsqueeze(-1) * x.unsqueeze(-2)).flatten(start_dim=-2) / self.r2  # BHLFF -> BHL(F*F)
        # SE: raising to power 0 is a hacky way to get ones without calling torch.ones
        # which is incompatible with cuda graph caching 
        return torch.cat(
            [x[..., :1] ** 0, x / self.rrd, x2 / self.rd], 
            dim=-1
        )  # [BHL1, BHLF, BHL(F*F)] -> BHL(1+F+F*F)

class LinearAttention(nn.Module):
    def __init__(
        self,
        d_model: int,           # D: 隐藏层的维度
        feature_map: FeatureMap = TaylorExp, 
        l_max: int = 2048,
        feature_dim: int = 16,  # F: TaylorExp后特征向量的维度
        head_dim: int = None,   # D': 多头注意力的头的维度  D' = D // H
        num_heads: int = 16,    # H: 多头注意力的头的数量
        eps: float = 1e-12,
        layer_idx: int = None,
        parallel_implementation: str="quadratic",  # "linear", "quadratic"
        **kwargs
    ):
        super().__init__()

        self.layer_idx = layer_idx
        self.d_model = d_model
        self.l_max = l_max
        self.eps = eps
        self.parallel_implementation = parallel_implementation

        # set dimension 
        self.num_heads = num_heads
        self.head_dim = self.d_model // self.num_heads if head_dim is None else head_dim      
        self.feature_dim = feature_dim

        # initialize projections and feature map
        self.feature_map = feature_map
        self.proj_q = nn.Linear(self.d_model, self.feature_dim * self.num_heads, bias=False)
        self.proj_k = nn.Linear(self.d_model, self.feature_dim * self.num_heads, bias=False)
        self.proj_v = nn.Linear(self.d_model, self.num_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.d_model, bias=False)

        if self.parallel_implementation == "tk":
            self.g_norm = RMSNorm(self.head_dim, eps=1e-5)

        
    def forward(self, 
        hidden_states: torch.Tensor,  # BLD
        inference_params: InferenceParams = None,
        *args: any, 
        **kwargs: any
    ):
        """
        x (torch.Tensor): tensor of shape (b, d, l)
        y (torch.Tensor): tensor of shape (b, d, l)
        """
        b, l, _ = hidden_states.size()  # BLD

        q = self.proj_q(hidden_states)  # BL(F*H)
        k = self.proj_k(hidden_states)  # BL(F*H)
        v = self.proj_v(hidden_states)  # BL(H*D')
        q = q.view(b, l, self.num_heads, self.feature_dim).transpose(1, 2)  # BHLF
        k = k.view(b, l, self.num_heads, self.feature_dim).transpose(1, 2)  # BHLF
        v = v.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)     # BHLD'

        self.is_inference = inference_params is not None
        if inference_params is None:
            # train
            return self.parallel_forward(hidden_states, q, k, v)  # BLD
        else:
            # inference
            # check if we are doing prefill or generation
            if inference_params.seqlen_offset > 0: 
                # recurrent
                kv_state, k_state = self._get_inference_cache(inference_params)
                q, k = self.feature_map(q), self.feature_map(k)  # BHL(1+F+F*F), BHL(1+F+F*F)
                return self.recurrent_forward(hidden_states, kv_state, k_state, q, k, v)
            else:  
                # prefill
                y, kv_state, k_state = self.parallel_forward(hidden_states, q, k, v)  # BLD, BH1D'(1+F+F*F), BH11(1+F+F*F)
                print("kv_state: ", kv_state.shape)
                print("k_state: ", k_state.shape)
                print("qkv:", q.shape, k.shape, v.shape)
                print("y: ", y.shape)
                if self.layer_idx in inference_params.key_value_memory_dict:
                    # # update the state in-place when graph caching is enabled
                    inference_params.key_value_memory_dict[self.layer_idx][0].copy_(kv_state)
                    inference_params.key_value_memory_dict[self.layer_idx][1].copy_(k_state)
                else: 
                    inference_params.key_value_memory_dict[self.layer_idx] = (kv_state, k_state)
                return y

    def parallel_forward(self, x: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        x: BLD
        q: BHLF
        k: BHLF
        v: BHLD'
        """
        
        if self.parallel_implementation == "tk":

            # WARNING: this mode is currently only for BENCHMARKING and seqlen needs to be a multiple of 64
            # pad to use other seqlens, as in https://github.com/HazyResearch/ThunderKittens/tree/main/demos/based_demo
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            y, kv_state = tk.based( q, k, v )  
            if self.is_inference:
                kv_state = kv_state[:, :, None].transpose(3, 4)
                k_state = None
            # output norm and gating 
            y = self.g_norm(y)
            y = rearrange(y, 'b h l d -> b l (h d)')
        
        elif self.parallel_implementation == "quadratic":  # 默认是这个

            q, k = self.feature_map(q), self.feature_map(k)  # BHL(1+F+F*F), BHL(1+F+F*F)
            A_qk = torch.einsum("bhnd,bhmd->bhnm", q, k)  # BHLL

            try:
                A_qk = torch.tril(A_qk)       
            except:
                # tril is incompatible with certain data types
                b, h, l, l = A_qk.shape
                cumsum_matrix = torch.tril(torch.ones((l, l))).to(q.device, q.dtype)
                A_qk = A_qk * cumsum_matrix

            y = torch.einsum("bhnm,bhme->bhne", A_qk.to(x.dtype), v.to(x.dtype))  # BHLD'
            z = 1 / (torch.einsum("bhld,bhld->bhl", q, k.cumsum(2)) + self.eps)  # BHL
            y = y * z[..., None]  # BHLD'
            y = rearrange(y, 'b h l d -> b l (h d)')  # BL(H*D')

        elif self.parallel_implementation == "linear": 

            q, k = self.feature_map(q), self.feature_map(k)
            v = causal_dot_product(q.contiguous().to(dtype=torch.float32), k.contiguous().to(dtype=torch.float32),v.contiguous().to(dtype=torch.float32),)
            z = 1 / (
                torch.einsum(
                    "bhld,bhld->bhl", 
                    q.to(dtype=torch.float32), 
                    k.to(dtype=torch.float32).cumsum(2)
                ) + self.eps
            )
            y = v * z[..., None]
            y = rearrange(y, 'b h l d -> b l (h d)')

        elif self.parallel_implementation == "fla_parallel":

            """ 
            Computes both the feature map and causal dot products.
            Booleans are for the denominator and the normalization 
            """
            y = parallel_based(q, k, v, True, True)
            y = rearrange(y, 'b h l d -> b l (h d)')

        elif self.parallel_implementation == "fla_chunk":

            """ 
            Computes both the feature map and causal dot products.
            Booleans are for the denominator and the normalization 
            """
            y = fused_chunk_based(q, k, v, True, True)
            y = rearrange(y, 'b h l d -> b l (h d)')

        else: 
            raise ValueError(f"Parallel implementation {self.parallel_implementation} not supported")

        if self.is_inference and self.parallel_implementation != "tk":
            kv_state = torch.einsum("bhnd,bhnf->bhfd", k, v)[:, :, None]  # BH1D'(1+F+F*F)
            k_state = k.sum(dim=2)[:, :, None, None]  # BH11(1+F+F*F)

        if self.is_inference:
            # inference
            return self.out_proj(y.to(x.dtype)), kv_state, k_state  # BLD, BH1D'(1+F+F*F), BH(1+F+F*F)11
        else:
            # train
            return self.out_proj(y.to(x.dtype))  # BLD

    
    def recurrent_forward(self, hidden_states: torch.Tensor, kv_state: torch.Tensor, k_state: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, decay: torch.Tensor=None):
        """
        Compute linear attention with recurrent view
        -> Assume q.shape is (b, h, 1, d); k and v.shape are (b, h, l, d)

        kv_state: BH1D'(1+F+F*F)
        k_state:  BH11(1+F+F*F)
        q:        BHL(1+F+F*F)
        k:        BHL(1+F+F*F)
        v:        BHLD'
        """
        b, h, l, d = q.shape
        assert l == 1, f'q.shape is {q.shape} but should be ({b}, {h}, 1, {d})'
        # Expand dims for broadcasting to compute linear attention
        q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1)  # BHL1(1+F+F*F), BHL1(1+F+F*F), BHLD'1

        kv_state += k[:, :, -1:] * v[:, :, -1:]  # BH1D'(1+F+F*F)
        k_state  += k[:, :, -1:]  # BH11(1+F+F*F)

        # Compute linear attention
        num = (q * kv_state).sum(dim=-1)
        if 'fla' in self.parallel_implementation: 
            eps = 1e-6 # this code uses an alternate eps
        else: 
            eps = 1e-12
        y = num / ((q * k_state).sum(dim=-1) + eps)

        y = rearrange(y, 'b h l d -> b l (h d)').to(q.dtype)
        return self.out_proj(y)
 
    
    def expanded_size(self):
        return self.feature_dim ** 2 + self.feature_dim + 1
    
    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype=None, **kwargs):
        """Creates a state tensor of shape ..."""

        kv_shape = (
            batch_size, self.num_heads, 1, self.head_dim, self.expanded_size()
        )
        k_shape = (
            batch_size, self.num_heads, 1, 1, self.expanded_size()
        )
        kv_state = torch.zeros(*kv_shape, dtype=dtype, device=self.out_proj.weight.device)
        k_state = torch.zeros(*k_shape, dtype=dtype, device=self.out_proj.weight.device)
        return (kv_state, k_state)
     
    def _get_inference_cache(self, inference_params: InferenceParams):
        return inference_params.key_value_memory_dict[self.layer_idx]

