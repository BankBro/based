import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import dataclasses
from dataclasses import dataclass, fields

from based.models.mixers.transformer_vq.nn.grad import sg
from based.models.mixers.transformer_vq.nn.grad import st
from based.models.mixers.transformer_vq.nn.types import TransformerConfig


@dataclass
class VQSpec:
    n_device: torch.Tensor
    n_block_per_update: torch.Tensor
    loss_mask: torch.Tensor

    @classmethod
    def create(cls, **kwargs):
        signature = {field.name: field.type for field in fields(cls)}
        filtered = {k: v for k, v in kwargs.items() if k in signature}
        return cls(**filtered)


def get_shortcodes(vecs, codebook):
    B = vecs.shape[0]
    H = vecs.shape[1]
    L = vecs.shape[2]
    S = codebook.shape[1]
    K = codebook.shape[2]
    
    assert vecs.shape == (B, H, L, K), f"Expected vecs shape {(B, H, L, K)}, got {vecs.shape}"
    assert codebook.shape == (H, S, K), f"Expected codebook shape {(H, S, K)}, got {codebook.shape}"
    
    diffs2 = (
        torch.sum(vecs ** 2, dim=-1, keepdim=True)  # BHL1
        - 2.0 * torch.einsum("bhlk,hsk->bhls", vecs, codebook)  # BHLS
        + torch.sum(codebook ** 2, dim=-1).unsqueeze(0).unsqueeze(2)  # 1H1S
    )  # BHLS
    
    z = torch.argmin(diffs2, dim=-1)  # BHL
    assert z.shape == (B, H, L), f"Expected z shape {(B, H, L)}, got {z.shape}"
    
    errs2 = torch.min(diffs2, dim=-1).values  # BHL
    errs2 = torch.nn.functional.relu(errs2)  # this is a no-op if using infinite precision
    assert errs2.shape == (B, H, L), f"Expected errs2 shape {(B, H, L)}, got {errs2.shape}"
    
    return z, errs2

def get_codewords(shortcodes, codebook):
    B = shortcodes.shape[0]
    H = shortcodes.shape[1]
    L = shortcodes.shape[2]
    S = codebook.shape[1]
    d = codebook.shape[2]
    
    shortcodes = shortcodes.unsqueeze(-1)  # BHL1
    codebook = codebook.unsqueeze(0)  # 1HSd
    assert shortcodes.shape == (B, H, L, 1), f"Expected shape {(B, H, L, 1)}, got {shortcodes.shape}"
    assert codebook.shape == (1, H, S, d), f"Expected shape {(1, H, S, d)}, got {codebook.shape}"
    
    cz = torch.gather(codebook.expand(B, H, S, d), 2, shortcodes.expand(-1, -1, -1, d))  # BHLd
    assert cz.shape == (B, H, L, d), f"Expected shape {(B, H, L, d)}, got {cz.shape}"
    return cz


class LearnableVQ(nn.Module):
    def __init__(self, config:TransformerConfig):
        super(LearnableVQ, self).__init__()
        self.config = config
        self.apply_config()

        self.c_sum = nn.Parameter(torch.empty(self.n_head, self.n_code, self.d_k))
        init.xavier_normal_(self.c_sum)
        self.c_count = nn.Parameter(torch.ones(self.n_head, self.n_code))

    def apply_config(self):
        for k, v in dataclasses.asdict(self.config).items():
            setattr(self, k, v)
    
    @staticmethod
    def _get_codebook(c_sum, c_count):
        c = c_sum / torch.clamp(c_count[..., None], min=0.01)
        return sg(c)  # HSK

    def get_codebook(self):
        return LearnableVQ._get_codebook(self.c_sum, self.c_count)

    @staticmethod
    def get_codebook_ema_targets(vecs, shortcodes, c_sum, c_count, c_gamma, vq_spec:VQSpec):
        B, H, L, D = vecs.shape
        S = c_sum.shape[1]
        
        assert vecs.shape == (B, H, L, D), f"Expected vecs shape {(B, H, L, D)}, got {vecs.shape}"
        assert shortcodes.shape == (B, H, L), f"Expected shortcodes shape {(B, H, L)}, got {shortcodes.shape}"
        assert c_sum.shape == (H, S, D), f"Expected c_sum shape {(H, S, D)}, got {c_sum.shape}"
        assert c_count.shape == (H, S), f"Expected c_count shape {(H, S)}, got {c_count.shape}"
        assert vq_spec.loss_mask.shape == (B, L), f"Expected loss_mask shape {(B, L)}, got {vq_spec.loss_mask.shape}"
        
        g = c_gamma
        d = vq_spec.n_device
        p = vq_spec.n_block_per_update
        
        r = F.one_hot(shortcodes, num_classes=S).type(vecs.dtype)  # BHLS
        r = r * vq_spec.loss_mask.unsqueeze(1).unsqueeze(-1)  # BHLS
        
        # d * p 的作用是将结果放大为整个batch, 弥补了多设备的分片
        c_sum_hat = d * p * torch.einsum("bhls,bhld->hsd", r, vecs)  # HSD
        c_count_hat = d * p * r.sum(dim=(0, 2))  # HS
        
        c_sum_tgt = (1 - g) * c_sum_hat + g * c_sum
        c_count_tgt = (1 - g) * c_count_hat + g * c_count
        
        assert c_sum_tgt.shape == (H, S, D), f"Expected c_sum_tgt shape {(H, S, D)}, got {c_sum_tgt.shape}"
        assert c_count_tgt.shape == (H, S), f"Expected c_count_tgt shape {(H, S)}, got {c_count_tgt.shape}"
        
        return c_sum_tgt, c_count_tgt

    @staticmethod
    def get_codebook_loss(
        vecs,
        shortcodes,
        c_sum,
        c_count,
        c_gamma,
        vq_spec,
    ):
        B = vecs.shape[0]
        H = vecs.shape[1]
        L = vecs.shape[2]
        d = vecs.shape[3]
        S = c_count.shape[1]

        c_sum_tgt, c_count_tgt = LearnableVQ.get_codebook_ema_targets(
            vecs=vecs,
            shortcodes=shortcodes,
            c_sum=c_sum,
            c_count=c_count,
            c_gamma=c_gamma,
            vq_spec=vq_spec,
        )
        assert c_sum_tgt.shape == (H, S, d), f"Expected c_sum_tgt shape {(H, S, d)}, got {c_sum_tgt.shape}"
        assert c_count_tgt.shape == (H, S), f"Expected c_count_tgt shape {(H, S)}, got {c_count_tgt.shape}"
        
        l_codebook_sum = torch.sum(sg(c_sum - c_sum_tgt) * st(c_sum))
        l_codebook_count = torch.sum(sg(c_count - c_count_tgt) * st(c_count))
        l_codebook = l_codebook_count + l_codebook_sum
        return l_codebook
    
    @staticmethod
    def get_quantization_metrics(vecs, vecs_hat, errs2, c_sum, c_count):
        # we'll call stop gradients in the return statement, so no need to call it now
        n_head, n_code = c_count.shape[0], c_count.shape[1]
        eps, errmin, errmax, maskval = 1e-2, 0e1, 1e1, 1e30

        c_count = torch.clamp(c_count, min=eps)  # HS
        c = c_sum / c_count.unsqueeze(-1)  # HSd
        c_norms = torch.clamp(torch.norm(c, dim=-1), min=eps)  # HS
        c_normed = c / c_norms.unsqueeze(-1)  # HSd
        c_sims = torch.einsum("hsd,hzd->hsz", c_normed, c_normed)  # HSS
        c_dists = torch.norm(c.unsqueeze(2) - c.unsqueeze(1), dim=-1)  # HSS

        vec_norms = torch.clamp(torch.norm(vecs, dim=-1), min=eps)  # BHL
        vec_hat_norms = torch.clamp(torch.norm(vecs_hat, dim=-1), min=eps)  # BHL

        errs = torch.sqrt(errs2)  # BHL
        relative_errs = torch.clamp(errs / vec_norms, min=errmin, max=errmax)  # BHL

        probs = c_count / torch.sum(c_count, dim=-1, keepdim=True)  # HS

        c_thresh_oob = torch.logical_or(c_count < 1.0, c_count > 1_000_000)  # HS
        c_thresh_oob = c_thresh_oob.to(dtype=torch.float32)

        # elements will have shape [], [H] or [B, H], then we will tree map
        # to avg over heads/device batch items
        ones = torch.ones([1, n_code, n_code], dtype=torch.float32, device=vecs.device)
        up = torch.triu(ones).to(device=vecs.device)  # upper triangular ones mask
        low = torch.tril(ones).to(device=vecs.device)  # strict lower triangular ones mask
        
        metrics = dict(
            c_sim_min=torch.amin(low * c_sims + maskval * up, dim=(1, 2)),  # [H]
            c_sim_mean=torch.sum(low * c_sims, dim=(1, 2)) / torch.sum(low, dim=(1, 2)),  # [H]
            c_sim_max=torch.amax(low * c_sims - maskval * up, dim=(1, 2)),  # [H]
            c_dist_min=torch.amin(low * c_dists + maskval * up, dim=(1, 2)),  # [H]
            c_dist_mean=torch.sum(low * c_dists, dim=(1, 2)) / torch.sum(low, dim=(1, 2)),  # [H]
            c_dist_max=torch.amax(low * c_dists - maskval * up, dim=(1, 2)),  # [H]
            c_norm_min=torch.min(c_norms, dim=1).values,  # [H]
            c_norm_mean=torch.mean(c_norms, dim=1),  # [H]
            c_norm_max=torch.max(c_norms, dim=1).values,  # [H]
            c_usage_min=torch.min(c_count, dim=1).values,  # [H]
            c_usage_mean=torch.mean(c_count, dim=1),  # [H]
            c_usage_max=torch.max(c_count, dim=1).values,  # [H]
            c_thresh_oob=torch.sum(c_thresh_oob, dim=1),  # [H]
            c_entropy=torch.sum(-probs * torch.log(probs + 1e-20), dim=-1),  # [H]
            vec_norm_mean=torch.mean(vec_norms, dim=2),  # [B, H]
            vec_hat_norm_mean=torch.mean(vec_hat_norms, dim=2),  # [B, H]
            relative_err_min=torch.min(relative_errs, dim=2).values,  # [B, H]
            relative_err_mean=torch.mean(relative_errs, dim=2),  # [B, H]
            relative_err_max=torch.max(relative_errs, dim=2).values,  # [B, H]
        )

        return {k: torch.mean(sg(v)) for k, v in metrics.items()}
    
    def forward(self, vecs, vq_spec):
        self.block_len = self.config.block_len if self.training else vecs.shape[2]

        orig_dtype = vecs.dtype
        vecs_hp = vecs
        c = self.get_codebook()
        z, errs2 = get_shortcodes(vecs=vecs_hp, codebook=c)
        
        cz = get_codewords(shortcodes=z, codebook=c)
        cz = cz.to(orig_dtype)
        vecs_hat = sg(cz) + st(vecs)

        if self.is_train:
            loss_mask = vq_spec.loss_mask  # BL
            # l_commit: BL -> B1L -> BHL -> BL -> [1]
            # 鼓励输入向量接近码本, l_commit只对vecs有梯度
            l_commit = torch.mean(torch.sum(torch.unsqueeze(loss_mask, 1) * errs2, dim=1))

            # 通过EMA更新码本参数, 确保码本随输入分布变化, l_codebook只对c_sum和c_count有梯度
            l_codebook = self.get_codebook_loss(
                vecs=vecs_hp,
                shortcodes=z,
                c_sum=self.c_sum,
                c_count=self.c_count,
                c_gamma=self.c_gamma,
                vq_spec=vq_spec,
            )

            metrics = self.get_quantization_metrics(
                vecs=sg(vecs),
                vecs_hat=sg(vecs_hat),
                errs2=sg(errs2),
                c_sum=sg(self.c_sum),
                c_count=sg(self.c_count)
            )

        else:
            l_commit = torch.zeros()
            l_codebook = torch.zeros()
            metrics = dict()

        return dict(
            quantized=vecs_hat,
            shortcodes=z,
            l_commit=l_commit,
            l_codebook=l_codebook,
            metrics=metrics,
            errs2=errs2,
        )