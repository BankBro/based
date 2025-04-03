import dataclasses
import torch
import torch.nn as nn
from transformer_vq.nn.attn import VQAttention
from transformer_vq.nn.emb import Embeddings
from transformer_vq.nn.norm import LayerNorm
from transformer_vq.nn.pe import ScaledSin
from transformer_vq.nn.types import TransformerConfig
from transformer_vq.nn.vq import VQSpec

from transformer_vq.utils.tools import check_tensor_shape
from transformer_vq.utils.dict import average_nested_dicts

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
    def initial_state(config, batch_size):
        return [
            VQAttention.initial_state(config, batch_size),
            VQAttention.initial_state(config, batch_size)
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
            output_features=x,
            attn_state=[new_state1, new_state2],
            l_commit=l_commit,
            l_codebook=l_codebook,
            metrics=metric_dict,
        )


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.apply_config()
        
        # 初始化嵌入层
        if not self.no_emb or self.e_tie:
            self.token_embedder = Embeddings(self.config)
        
        # 位置编码
        if self.pe_abs:
            self.position_embedder = ScaledSin(self.config)
        
        # 初始化Transformer层
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(self.config) for _ in range(self.n_layer)
        ])
        
        # 输出层
        if self.e_preln:
            self.out_ln = LayerNorm(self.d_model, self.param_dtype)
        
        # 不共享嵌入层
        if not self.e_tie:
            self.out_proj = nn.Linear(self.d_model, self.n_vocab)
        
        # Dropout
        self.dropemb = nn.Dropout(self.p_dropemb)

    def apply_config(self):
        for k, v in dataclasses.asdict(self.config).items():
            setattr(self, k, v)
    
    @staticmethod
    def initial_state(config, batch_size):
        return [
            TransformerLayer.initial_state(config, batch_size)
            for _ in range(config.n_layer)
        ]

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

    def forward(self, inputs, doc_ids, state, vq_spec):
        B, U = inputs.shape[0], inputs.shape[1]
        L, D = self.block_len, self.d_model
        F = U // L
        C = self.n_vocab

        x = inputs  # BU*
        
        if not self.no_emb:
            x = self.token_embedder(x)  # BUD

        if self.pe_abs:
            offset = state[0][0]["pos_offset"]
            emb = self.position_embedder(x.size(1), offset)  # UD
            x = x + emb  # BUD
        
        x = self.dropemb(x)  # BUD
        x_blocks = self.get_blocks_from_sequence(x)  # FBLD
        doc_ids_blocks = self.get_blocks_from_sequence(doc_ids)  # FBL

        new_states = []
        aux = []
        for i, layer in enumerate(self.transformer_layers):
            layer_output_dict = layer(
                x_blocks, # FBLD
                doc_ids_blocks,  # FBL
                state[i], 
                self._adapt_vq_spec(vq_spec, F)
            )
            x_blocks = layer_output_dict['output_features']
            check_tensor_shape(x_blocks, (F, B, L, D))

            new_states.append(layer_output_dict.pop('attn_state'))
            aux.append(layer_output_dict)

        aux = average_nested_dicts(aux)  # dic(l_commit:xxx, l_codebook:xxx, metrics:dic)
        
        x = self.get_sequence_from_blocks(x_blocks)  # BUD
        check_tensor_shape(x, (B, U, D))

        if self.e_preln:
            x = self.out_ln(x)  # BUD
        
        # 生成logits
        if self.e_tie:
            logits = self.token_embedder.logits(x)  # BUC
        else:
            self.out_proj(x)  # BUC
        logits = logits * self.e_scale

        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)  # BUC
        check_tensor_shape(logprobs, (B, U, C))
        
        return {
            'logprobs': logprobs,  # BUC
            'attn_state': new_states,
            **aux
        }