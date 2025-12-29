# coding=utf-8
# Copyright 2025 OpenSeek and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation


class OpenseekConfig(PretrainedConfig):

    model_type = "openseek"
    keys_to_ignore_at_inference = ["past_key_values"]
    # Default tensor parallel plan for base model `OpenseekModel`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.dt_proj": "rowwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.feed_forward.gate_proj": "colwise",
        "layers.*.feed_forward.up_proj": "colwise",
        "layers.*.feed_forward.down_proj": "rowwise",
        "layers.*.feed_forward.queries_proj": "colwise",
        "layers.*.feed_forward.down_embed": "rowwise",
        "layers.*.feed_forward.up_embed": "rowwise",
    }

    def __init__(
        self,
        vocab_size=32768,
        hidden_size=1024,
        intermediate_size=2048,
        num_hidden_layers=32,
        hidden_bias=False,
        hidden_dropout=0.0,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-06,
        use_cache=True,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        tie_word_embeddings=False,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        rope_scaling=None,

        # === for attention ===
        num_attention_heads=8,
        num_key_value_heads=None,
        attention_dropout=0.0,
        # =====================

        # === for MLA ===
        q_lora_rank=256,
        kv_lora_rank=128,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        # ===============

        # === for DMA ===
        keep_window_size=2048,
        dynamic_mask_ratio=0.0,
        # ===============

        # === for MoE ===
        is_moe=False,
        num_experts=2048,
        num_experts_per_tok=8,

        # === for Deepseek MoE ===
        moe_intermediate_size=256,
        num_routed_experts=8,
        routed_scaling_factor=2.5,
        ep_size=1,
        topk_method='noaux_tc',
        n_group=8,
        topk_group=4,
        moe_layer_freq=1,
        first_k_dense_replace=3,
        norm_topk_prob=True,
        scoring_func='sigmoid',
        aux_loss_alpha=0.001,
        seq_aux=True,
        # ======================

        # === for CDMoE ===
        expert_retrieval_size=64,
        # =================
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers

        self.hidden_bias = hidden_bias
        self.hidden_dropout = hidden_dropout
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache

        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        # attention
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.attention_dropout = attention_dropout

        # MLA
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim

        # DMA
        self.keep_window_size = keep_window_size
        self.dynamic_mask_ratio = dynamic_mask_ratio

        # MoE
        self.is_moe = is_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        # Deepseek MoE
        self.moe_intermediate_size = moe_intermediate_size
        self.num_routed_experts = num_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.ep_size = ep_size
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux

        # CDMoE
        self.expert_retrieval_size = expert_retrieval_size

        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, copy it it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        # for backward compatibility
        if num_key_value_heads is None:
            self.num_key_value_heads = num_attention_heads

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["OpenseekConfig"]
