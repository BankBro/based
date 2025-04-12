import dataclasses
import torch
import torch.nn as nn
from based.models.mixers.transformer_vq.nn.attn import VQAttention
from based.models.mixers.transformer_vq.nn.types import TransformerConfig
from based.models.mixers.transformer_vq.nn.vq import VQSpec

from based.models.mixers.transformer_vq.utils.tools import check_tensor_shape
from based.models.mixers.transformer_vq.utils.dict import average_nested_dicts

class TransformerLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.apply_config()
        
        self.attn1 = VQAttention(self.config)
        self.attn2 = VQAttention(self.config)

        self.droplyr1 = nn.Dropout(self.p_droplyr)
        self.droplyr2 = nn.Dropout(self.p_droplyr)

    def apply_config(self):
        for k, v in dataclasses.asdict(self.config).items():
            setattr(self, k, v)

    @staticmethod
    def initial_state(config, batch_size, device):
        return [
            VQAttention.initial_state(config, batch_size, device),
            VQAttention.initial_state(config, batch_size, device)
        ]
    
    def _adapt_vq_spec(self, vq_spec, n_block):
        """vq_spec: n_device(F1), n_block_per_update(F1), loss_mask(FBL)"""

        if vq_spec is None:
            return [None] * n_block
        
        assert vq_spec.n_device.shape[0] == n_block
        assert vq_spec.n_block_per_update.shape[0] == n_block
        assert vq_spec.loss_mask.shape[0] == n_block
        
        return [
            VQSpec.create(
                n_device=vq_spec.n_device[f],
                n_block_per_update=vq_spec.n_block_per_update[f],
                loss_mask=vq_spec.loss_mask[f]
            )
            for f in range(n_block)
        ]
    
    def process_blocks(self, attn, state, x, doc_ids, vq_spec_list):
        res_outputs, dic_outputs = [], []

        for x_block, doc_ids_block, vq_spec_block in zip(x, doc_ids, vq_spec_list):
            input_dict = dict(
                input_features=x_block,  # BLD
                doc_ids=doc_ids_block,  # BL
                vq_spec=vq_spec_block
            )
            state, output_dict = attn(state, input_dict)
            
            res_outputs.append(output_dict.pop("res"))  # BLD
            dic_outputs.append(output_dict)
        
        res = torch.stack(res_outputs, dim=0)
        dic_outputs = average_nested_dicts(dic_outputs)
        return res, dic_outputs, state

    def forward(self, x, doc_ids, state, vq_spec):
        """
        处理块维度(F)的扫描逻辑:
            x: FBLD
            doc_ids: FBL
            state: [attn1_states, attn2_states]
            vq_spec: n_device(F1), n_block_per_update(F1), loss_mask(FBL)
        """
        self.block_len = self.config.block_len if self.training else x.shape[2]
        F, B, L, D = x.shape[0], x.shape[1], self.block_len, self.d_model

        check_tensor_shape(x, (F, B, L, D))
        check_tensor_shape(doc_ids, (F, B, L))

        attn1_state, attn2_state = state
        vq_spec_list = self._adapt_vq_spec(vq_spec, F)

        r1, attn1_output_dict, new_state1 = self.process_blocks(
            self.attn1, attn1_state, x, doc_ids, vq_spec_list)
        check_tensor_shape(r1, (F, B, L, D))
        x = x + self.droplyr1(r1)

        r2, attn2_output_dict, new_state2 = self.process_blocks(
            self.attn2, attn2_state, x, doc_ids, vq_spec_list)
        check_tensor_shape(r2, (F, B, L, D))
        x = x + self.droplyr2(r2)

        l_commit = attn1_output_dict.pop("l_commit") + attn2_output_dict.pop("l_commit")
        l_codebook = attn1_output_dict.pop("l_codebook") + attn2_output_dict.pop("l_codebook")

        metric_dict = average_nested_dicts([
            attn1_output_dict.pop("metrics"),
            attn2_output_dict.pop("metrics")
        ])

        return dict(
            output_features=x,  # FBLD
            attn_state=[new_state1, new_state2],
            l_commit=l_commit,
            l_codebook=l_codebook,
            metrics=metric_dict,
        )


class TransVQAttention(nn.Module):
    def __init__(self, d_model: int, **kwargs):
        super().__init__()
        # print(f"TransVQAttention init, d_model={d_model}")
        self.config = TransformerConfig.create(d_model=d_model, **kwargs)
        self.apply_all_params(**{'d_model': d_model, **kwargs})

        self.vq_layer = TransformerLayer(self.config)
        self.state = None
        self.loss_metrics = None

    def apply_all_params(self, **kwargs):
        config_field_names = {
            field.name for field in dataclasses.fields(TransformerConfig)
        }

        for k, v in kwargs.items():
            if k in config_field_names:
                setattr(self, k, getattr(self.config, k))
            else:
                setattr(self, k, v)

    def initial_state(self, batch_size, device):
        self.state = TransformerLayer.initial_state(self.config, batch_size, device)

    def get_blocks_from_sequence(self, x):
        """将序列分割为块"""
        batch_size, seq_len, *suffix = x.shape  # BU*
        x = x.view(batch_size, -1, self.block_len, *suffix)  # BFL*
        suffix_axes = list(range(3, x.ndim))
        return x.permute(1, 0, 2, *suffix_axes)  # FBL*

    def get_sequence_from_blocks(self, x):
        """合并块为完整序列"""
        n_block, batch_size, block_len, d_model = x.shape  # FBDL
        x = x.permute(1, 0, 2, 3)  # BFLD
        return x.reshape(batch_size, -1, d_model)  # BUD
    
    def _adapt_vq_spec(self, vq_spec, n_block):
        """
        input:
            vq_spec
                n_device: 1
                n_block_per_update: 1
                loss_mask: BU

        output:
            vq_spec
                n_device: F1
                n_block_per_update: F1
                loss_mask: FBL
        """
        if vq_spec is None:
            return None
        
        n_device = vq_spec.n_device.expand(n_block, -1)  # F1
        n_block_per_update = vq_spec.n_block_per_update.expand(n_block, -1)  # F1
        
        # BU -> BFL -> FBL
        loss_mask = vq_spec.loss_mask.view(-1, n_block, self.block_len).permute(1,0,2)
            
        return VQSpec.create(
            n_device=n_device,  # F1
            n_block_per_update=n_block_per_update,  # F1
            loss_mask=loss_mask  # FBL
        )

    def forward(self, inputs, inference_params, *args, **kwargs):
        """inputs: BUD, 输入长度T默认等于U, 下面用U代替T"""
        self.block_len = self.config.block_len if self.training else inputs.shape[1]

        assert inputs.shape[1] % self.block_len == 0

        B, U = inputs.shape[0], inputs.shape[1]
        L, D = self.block_len, self.d_model
        F = U // L  # n_block_per_update

        x_blocks = self.get_blocks_from_sequence(inputs)  # FBLD
        check_tensor_shape(x_blocks, (F, B, L, D))

        doc_ids = torch.ones([B, U], dtype=torch.int32, device=inputs.device)  # BU
        doc_ids_blocks = self.get_blocks_from_sequence(doc_ids)  # FBL

        device = inputs.device
        vq_spec = VQSpec.create(
            n_device=torch.tensor([self.n_device], device=device),  # TODO: n_device
            n_block_per_update=torch.tensor([F], device=device),
            loss_mask=torch.ones([B, U], dtype=torch.int32, device=device),
        )
        vq_spec = self._adapt_vq_spec(vq_spec, F)

        layer_output_dict = self.vq_layer(
            x_blocks, # FBLD
            doc_ids_blocks,  # FBL
            self.state, 
            vq_spec
        )
        self.state = layer_output_dict.pop('attn_state')

        output_blocks = layer_output_dict.pop('output_features')
        check_tensor_shape(output_blocks, (F, B, L, D))
        output = self.get_sequence_from_blocks(output_blocks)  # BUD
        check_tensor_shape(output, (B, U, D))

        self.vq_loss_metrics = layer_output_dict

        return output
    
__all__ = ["TransVQAttention"]