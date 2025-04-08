import dataclasses

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from based.models.mixers.transformer_vq.nn.grad import sg
from based.models.mixers.transformer_vq.nn.norm import LayerNorm
from based.models.mixers.transformer_vq.nn.pe import get_sinusoid_embs
from based.models.mixers.transformer_vq.nn.types import TransformerConfig
from based.models.mixers.transformer_vq.nn.vq import LearnableVQ

from based.models.mixers.transformer_vq.utils.dict import recursive_apply_dict
from based.models.mixers.transformer_vq.utils.tools import one_hot_encode  # , check_dtypes_equal

MASK_INFTY_APPROX = 1e30  # mask value approximating infinity


class VQAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.apply_config()

        self.tau = self.d_k ** 0.5
        self.input_ln = LayerNorm(self.d_model)

        q_ch = self.n_head * self.d_k
        k_ch = self.n_head * self.d_k
        v_ch = self.n_head * self.d_v

        self.q_ln = LayerNorm(self.d_k, gain=False, bias=False)
        self.k_ln = LayerNorm(self.d_k, gain=False, bias=False)

        self.q_proj = nn.Linear(self.d_model, q_ch, bias=False)
        self.kvg_proj = nn.Linear(self.d_model, k_ch + v_ch + v_ch, bias=False)
        self.r_proj = nn.Linear(self.d_model, k_ch, bias=False)
        self.res_proj = nn.Linear(v_ch, self.d_model, bias=False)

        self.xl_u = nn.Parameter(torch.zeros(q_ch))
        self.xl_v = nn.Parameter(torch.zeros(q_ch))
        
        self.quantizer = LearnableVQ(self.config)
        
        self.dropsin = nn.Dropout(self.p_dropsin)
        self.dropres = nn.Dropout(self.p_dropres)

    def apply_config(self):
        for k, v in dataclasses.asdict(self.config).items():
            setattr(self, k, v)
    
    @staticmethod
    def initial_state(config, batch_size, device):
        prefix = (batch_size, config.n_head)
        s = config.n_code
        m = config.mem_len
        d_k = config.d_k
        d_v = config.d_v

        return dict(
            pos_offset = torch.tensor(0, dtype=torch.int32, device=device),
            xlcache = dict(
                z = torch.full((*prefix, m), fill_value=s, dtype=torch.int32, device=device),  # invalid z count 0
                k_hat = torch.zeros((*prefix, m, d_k), device=device),
                v = torch.zeros((*prefix, m, d_v), device=device),
                doc_ids = torch.zeros((batch_size, m), dtype=torch.int32, device=device),
            ),
            aggcache = dict(
                upper_div_lower = torch.zeros((*prefix, s, d_v), device=device),
                lower = torch.zeros((*prefix, s), device=device),
                latest_doc_id = torch.zeros((batch_size,), dtype=torch.int32, device=device),
            ),
        )
    
    @staticmethod
    def rel_shift(x):
        *leading_shape, present_len, past_len = x.shape  # BHLW
        pad_spec = [1, 0]
        x = torch.nn.functional.pad(x, pad_spec)  # BHL(W+1)
        x = x.view(*leading_shape, past_len + 1, present_len)  # BH(W+1)L
        x = x[..., 1:, :]  # BHWL
        x = x.view(*leading_shape, present_len, past_len)  # BHLW
        return x
    
    @staticmethod
    def get_causal_mask(block_len, mem_len, invalid_len, with_locality):
        assert block_len > 0 and mem_len >= 0
        assert invalid_len.ndim == 0
        device=invalid_len.device

        i = torch.arange(block_len, device=device).unsqueeze(-1)  # L1
        j = torch.arange(mem_len + block_len, device=device).unsqueeze(0)  # 1W

        alloc_mask = j >= invalid_len  # 1W, 排除无效的历史段（如文档分片边界前的无效内容）
        causal_mask = j - mem_len <= i  # LW, 因果掩码
        keep_mask = torch.logical_and(alloc_mask, causal_mask)  # LW

        if with_locality:
            """
            局部窗口掩码, 强制只关注当前块内的上下文, 防止相对位置分数的累积。
            支持长上下文窗口的能力依然存在, 因为记忆机制仍然有效。
            但不利用相对位置编码进行修正, 可能导致历史信息的表征不够精细, 可考虑改进。
            """
            window_mask = j >= i  # LW
            keep_mask = torch.logical_and(keep_mask, window_mask)

        return keep_mask  # LW

    @staticmethod
    def get_agg_biases(lower):
        # lower: BHS
        result = torch.where(
            torch.eq(lower, torch.zeros_like(lower)),
            -MASK_INFTY_APPROX,
            torch.log(torch.max(lower, torch.ones_like(lower))),  # this is never nan
        )
        return result  # BHS
    
    def get_q(self, x_tilde):
        bsz, present_len, _ = x_tilde.shape
        q = self.q_proj(x_tilde)  # (B,L,H*K)
        q = q.view(bsz, present_len, self.n_head, self.d_k)  # BLHK
        q = self.q_ln(q) * (self.tau**-0.5)
        q = q.permute(0, 2, 1, 3)  # BHLK
        return q

    def get_kvg(self, x_tilde):
        bsz, present_len, _ = x_tilde.shape

        kvg = self.kvg_proj(x_tilde)  # BL(H*(K+V+V))
        hk, hv = self.n_head * self.d_k, self.n_head * self.d_v
        k, v, g = torch.split(kvg, [hk, hv, hv], dim=-1)  # BL(H*K), BL(H*V), BL(H*V)
        
        assert k.shape == (bsz, present_len, self.n_head * self.d_k)
        assert v.shape == (bsz, present_len, self.n_head * self.d_v)
        assert g.shape == (bsz, present_len, self.n_head * self.d_v)

        k = k.view(bsz, present_len, self.n_head, self.d_k)  # BLHK
        v = v.view(bsz, present_len, self.n_head, self.d_v)  # BLHV
        k = self.k_ln(k) * (self.tau**-0.5)
        v = F.silu(v)
        g = F.silu(g)  # BL(H*V)
        k = k.permute(0, 2, 1, 3)  # BHLK
        v = v.permute(0, 2, 1, 3)  # BHLV

        return k, v, g
    
    def get_xl_helpers(self, device):
        # compute helpers for xl biases (z dai et al., 2019)
        xl_r = get_sinusoid_embs(
            length=self.mem_len + self.block_len,
            width=self.d_model,
            lam=self.pe_lam,
            flip=True,
        ).to(device=device)  # (M+L)D=WD

        xl_r = self.dropsin(xl_r)  # WD
        xl_r = self.r_proj(xl_r)  # (M+L)(H*K)=W(H*K)

        xl_r = xl_r.view(self.mem_len + self.block_len, self.n_head, self.d_k)  # WHK
        xl_r = xl_r.transpose(0, 1)  # HWK
        xl_r = xl_r * (self.tau**-0.5)

        xl_u = self.xl_u.view(1, self.n_head, 1, self.d_k) * (self.tau**-0.5)  # 1H1K
        xl_v = self.xl_v.view(1, self.n_head, 1, self.d_k) * (self.tau**-0.5)  # 1H1K
        return xl_r, xl_u, xl_v  # HWK, 1H1K, 1H1K
    
    def attn(self,
             present_q,  # BHLK
             present_k,  # BHLK
             present_v,  # BHLV
             present_doc_ids,  # BL
             state,
             vq_spec):
        bsz = present_q.shape[0]
        
        # check_dtypes_equal(
        #     present_v,
        #     state["xlcache"]["v"],
        #     state["aggcache"]["upper_div_lower"],
        #     state["aggcache"]["lower"],
        # )
        assert present_q.shape == (bsz, self.n_head, self.block_len, self.d_k)
        assert present_k.shape == (bsz, self.n_head, self.block_len, self.d_k)
        assert present_v.shape == (bsz, self.n_head, self.block_len, self.d_v)

        # quantize keys, compute metrics, commit loss, and codebook surrogate loss
        vq_output_dict = self.quantizer(present_k, vq_spec=vq_spec)

        present_z = vq_output_dict["shortcodes"]  # BHL
        present_k_hat = vq_output_dict["quantized"]  # BHLK
        l_commit = vq_output_dict["l_commit"]
        l_codebook = vq_output_dict["l_codebook"]
        metrics = vq_output_dict["metrics"]

        assert present_z.shape == (bsz, self.n_head, self.block_len)
        assert present_k_hat.shape == (bsz, self.n_head, self.block_len, self.d_k)
        # check_dtypes_equal(present_k_hat, present_k)

        # concatenate sliding window cache k/v onto current block
        xlcache = state["xlcache"]
        aggcache = state["aggcache"]
        assert xlcache["z"].shape == (bsz, self.n_head, self.mem_len)
        assert xlcache["k_hat"].shape == (bsz, self.n_head, self.mem_len, self.d_k)
        assert xlcache["v"].shape == (bsz, self.n_head, self.mem_len, self.d_v)


        recent_z = torch.cat([xlcache["z"], present_z], dim=-1)  # BH(M+L)=BHW
        recent_k_hat = torch.cat([xlcache["k_hat"], present_k_hat], dim=-2)  # BH(M+L)K=BHWK
        recent_v = torch.cat([xlcache["v"], present_v], dim=-2)  # BH(M+L)V=BHWV
        recent_doc_ids = torch.cat([xlcache["doc_ids"], present_doc_ids], dim=-1)  # B(M+L)=BW
        W = self.mem_len + self.block_len
        assert recent_z.shape ==  (bsz, self.n_head, W)
        assert recent_k_hat.shape == (bsz, self.n_head, W, self.d_k)
        assert recent_v.shape == (bsz, self.n_head, W, self.d_v)

        # compute xl bias helpers
        xl_r, xl_u, xl_v = self.get_xl_helpers(device=present_q.device)  # HWK, 1H1K, 1H1K

        # compute aggcache scores
        c = self.quantizer.get_codebook()  # HSK
        cache_scores = torch.einsum("bhlk,hsk->bhls", present_q + xl_u, c)  # BHLS
        cache_biases = VQAttention.get_agg_biases(aggcache["lower"]).unsqueeze(-2)  # BH1S
        cache_scores += cache_biases  # BHLS

        # compute recent scores (present and xlcache)
        # https://zhuanlan.zhihu.com/p/271984518
        recent_scores_ac = torch.einsum("bhlk,bhwk->bhlw", present_q + xl_u, recent_k_hat)  # BHLW

        recent_scores_bd = torch.einsum("bhlk,hwk->bhlw", present_q + xl_v, xl_r)  # BHLW
        recent_scores_bd = self.rel_shift(recent_scores_bd)  # BHLW
        recent_scores_bd = recent_scores_bd * self.get_causal_mask(
            block_len=self.block_len,
            mem_len=self.mem_len,
            invalid_len=torch.relu(self.mem_len - state["pos_offset"]),
            with_locality=True,  # 强制注意力仅限于当前block内的上下文
        ).unsqueeze(0).unsqueeze(0).type(torch.int32)  # 11LW -> BHLW

        recent_scores = recent_scores_ac + recent_scores_bd  # BHLW
        keep_mask = self.get_causal_mask(
            block_len=self.block_len,
            mem_len=self.mem_len,
            invalid_len=torch.relu(self.mem_len - state["pos_offset"]),
            with_locality=not self.agg_cache,
        ).unsqueeze(0).unsqueeze(0).type(torch.int32)  # 11LW
        recent_scores = recent_scores * keep_mask - MASK_INFTY_APPROX * (1 - keep_mask)  # BHLW

        # subtract max score for stability
        cache_max_scores = torch.max(cache_scores, dim=-1).values  # BHL
        recent_max_scores = torch.max(recent_scores, dim=-1).values  # BHL
        max_scores = sg(torch.max(cache_max_scores, recent_max_scores))  # BHL
        assert max_scores.shape == (bsz, self.n_head, self.block_len)
        cache_scores -= max_scores.unsqueeze(-1)  # BHLS
        recent_scores -= max_scores.unsqueeze(-1)  # BHLW
        cache_a = torch.exp(cache_scores)  # BHLS
        recent_a = torch.exp(recent_scores)  # BHLW
        assert cache_a.shape == (bsz, self.n_head, self.block_len, self.n_code)
        assert recent_a.shape == (bsz, self.n_head, self.block_len, W)

        # compute per-query normalizer d and divide unnormalized weights a by it first
        # 注意力机制的核心是通过权重分配, 将q与k的相似度转化为概率分布。
        # 归一化后的权重a满足: sum(a, dim=-1) = 1, 从而保证后续加权求和的数值稳定性。
        d = torch.sum(recent_a, dim=-1)  # BHL
        if self.agg_cache:
            d += torch.sum(cache_a, dim=-1)  # BHL

        wv = torch.einsum("bhlw,bhwv->bhlv", recent_a / d.unsqueeze(-1), recent_v)  # BHLV
        if self.agg_cache:
            wv += torch.einsum(
                "bhls,bhsv->bhlv", cache_a / d.unsqueeze(-1), aggcache["upper_div_lower"]  # ???
            )  # BHLV

        wv = wv.transpose(1, 2)  # BLHV
        wv = wv.reshape(bsz, self.block_len, self.n_head * self.d_v)  # BL(H*V)
        return {
            'attn_out': wv,  # BL(H*V)
            'recent_z': recent_z,  # BHW
            'recent_k_hat': recent_k_hat,  # BHWK
            'recent_v': recent_v,  # BHWV
            'recent_doc_ids': recent_doc_ids,  # BW
            'l_commit': l_commit,
            'l_codebook': l_codebook,
            'metrics': metrics,
        }
    
    def update_state(self,
                     recent_z,  # BHW
                     recent_k_hat,  # BHWK
                     recent_v,  # BHWV
                     recent_doc_ids,  # BW
                     state):
        B = recent_z.shape[0]
        H = self.n_head
        L = self.block_len
        M = self.mem_len
        S = self.n_code
        K = self.d_k
        V = self.d_v

        aggcache = state["aggcache"]
        assert aggcache['upper_div_lower'].shape == (B, H, S, V), f"want {(B, H, S, V)}, got {aggcache['upper_div_lower'].shape}"
        assert aggcache['lower'].shape == (B, H, S)
        assert recent_z[..., : -M].shape == (B, H, L)
        assert recent_v[..., : -M, :].shape == (B, H, L, V)
        assert recent_k_hat[..., : -M, :].shape == (B, H, L, K)

        new_pos_offset = state["pos_offset"] + self.block_len  # ???

		# compute kronecker deltas; invalid z's from xlcache init encode to zero vecs
        delta = one_hot_encode(
            recent_z[..., : -self.mem_len],  # BHL
            num_classes=self.n_code,
            dtype=recent_z.dtype,
            device=recent_z.device,
        )  # BHLS
        new_lower = aggcache["lower"] + torch.sum(delta, dim=-2)  # BHS + BHS, 码本命中计数

		# compute updated upper cache variable (stored in relative format for stability)
        # i.e., we compute new_upper_div_lower by dividing axis S by counts in new_lower
        f1 = aggcache["lower"] / torch.clamp(new_lower, min=1.0)  # BHS
        f2 = delta / torch.clamp(new_lower, min=1.0).unsqueeze(-2)  # BHLS

        new_upper_div_lower = (
            f1.unsqueeze(-1) * aggcache["upper_div_lower"]
            + torch.einsum("bhls,bhlv->bhsv", f2, recent_v[..., : -self.mem_len, :])
        )  # BHSV

        new_state = {
            'pos_offset': new_pos_offset,
            'xlcache': {
                'z': recent_z[..., -self.mem_len :],  # BHM
                'k_hat': recent_k_hat[..., -self.mem_len :, :],  # BHMK
                'v': recent_v[..., -self.mem_len :, :],  # BHMV
                'doc_ids': recent_doc_ids[..., -self.mem_len :],  # BM
            },
            'aggcache': {
                'lower': new_lower,  # BHS
                'upper_div_lower': new_upper_div_lower,  # BHSV
                'latest_doc_id': recent_doc_ids[..., -self.mem_len - 1],  # 当前缓存窗口前一个时间步的文档ID
                # TODO: latest_doc_id 没有被用到
            },
        }

        if not self.grad_thru_cache:
            new_state = recursive_apply_dict(new_state, sg)

        return new_state

    def forward(self, state, input_dict):
        # TODO: 检查漏掉的assert
        doc_ids = input_dict.pop("doc_ids")  # BL
        vq_spec = input_dict.pop("vq_spec")
        x = input_dict.pop("input_features")  # BLD, 是否已经有位置编码了???TODO

        x_tilde = self.input_ln(x)  # BLD
        q = self.get_q(x_tilde=x_tilde)  # BHLK
        k, v, g = self.get_kvg(x_tilde=x_tilde)  # BHLK, BHLV, BL(H*V)

        attn_output_dict = self.attn(q, k, v, doc_ids, state, vq_spec)
        wv = attn_output_dict.get("attn_out")  # BL(H*V)
        o = wv * g  # BL(H*V)
        res = self.res_proj(o)  # BLD
        res = self.dropres(res)  # BLD

        assert res.shape == x.shape, f"Expected shape {x.shape}, got {res.shape}"
        # check_dtypes_equal(res, x)

        new_state = self.update_state(
            recent_z=attn_output_dict.get("recent_z"),  # BHW
            recent_k_hat=attn_output_dict.get("recent_k_hat"),  # BHWK
            recent_v=attn_output_dict.get("recent_v"),  # BHWV
            recent_doc_ids=attn_output_dict.get("recent_doc_ids"),  # BW
            state=state,
        )
        
        output_dict = {
            'res': res,  # BLD
            'metrics': attn_output_dict.get("metrics"),
            'l_commit': attn_output_dict.get("l_commit"),
            'l_codebook': attn_output_dict.get("l_codebook"),
        }
        return new_state, output_dict
