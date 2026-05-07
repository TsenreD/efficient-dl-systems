"""
Baseline Calculator for DDP implementation with baseline model.
"""

from calculators.base import BaseCalculator


class BaselineCalculator(BaseCalculator):
    """
    Calculator for baseline implementation with DDP.
    """
    
    def _calculate_block_params(self) -> int:
        config = self.model
        
        mha_params = 4 * config.hidden_dim * config.hidden_dim
        rmsnorm_params = config.hidden_dim
        swiglu_params = 3 * config.hidden_dim * config.intermediate_dim
        
        return mha_params + swiglu_params + 2 * rmsnorm_params
    
    def calculate_total_params(self) -> int:
        """
        Calculate total model parameters.
        """
        config = self.model
        
        embedding_params = config.vocab_size * config.hidden_dim
        layer_params = config.num_layers * self._calculate_block_params()
        rmsnorm_params = config.hidden_dim
        head_params = config.hidden_dim * config.vocab_size
        
        return embedding_params + layer_params + rmsnorm_params + head_params
    
    def calculate_param_memory(self) -> int:
        """
        DDP: full params on each GPU.

        With AMP: params in bf16 + master params in fp32
        """
        return 6 * self.calculate_total_params()

    def calculate_gradient_memory(self) -> int:
        """
        DDP: full gradients on each GPU (fp32).
        """
        return 4 * self.calculate_total_params()

    def calculate_optimizer_memory(self) -> int:
        """
        DDP: full optimizer states on each GPU (fp32).

        AdEMAMix has 3 states: m (momentum), v (variance), nu (third moment)
        """
        return 12 * self.calculate_total_params()

    def calculate_activation_memory(self) -> int:
        """
        Baseline activation memory (all intermediates saved).
        """
        rms = self._calculate_rmsnorm_act()
        attn = self._calculate_attn_act()
        mlp  = self._calculate_swiglu_act()
        loss_act = self._calculate_ce_act()
        
        layer_act = rms + attn + rms + mlp
        
        return self.model.num_layers * layer_act + (rms * 6 // 5) + loss_act
    
    def _calculate_rmsnorm_act(self) -> int:
        train_conf = self.training
        model_conf = self.model
        
        B = train_conf.batch_size
        S = train_conf.seq_len
        H = model_conf.hidden_dim
        
        return 4 * (6 * B * S * H) # all fp32 in both modes
    
    def _calculate_ce_act(self) -> int:
        train_conf = self.training
        model_conf = self.model
        
        B = train_conf.batch_size # 2
        S = train_conf.seq_len # 8
        V = model_conf.vocab_size # 64
        
        return 2 * B * (S - 1) * V  # all fp32 in both modes
    
    def _calculate_swiglu_act(self) -> int:
        train_conf = self.training
        model_conf = self.model
        
        B = train_conf.batch_size # 3
        S = train_conf.seq_len #  12
        H = model_conf.hidden_dim # 16
        I = model_conf.intermediate_dim # 64
        
        part_1 = 2 * H * I # gate + up proj matrices, do not count
        part_2 = 2 * B * S * H # inputs for projections
        part_3 = 7 * B * S * I # intermediate ops
        part_4 = H * I # down proj, do not count
        part_5 = B * S * I # down proj input
        
        return train_conf.dtype_bytes * (
            2 * B * S * H + 8 * B * S * I
        )
    
    
    def _calculate_attn_act(self) -> int:
        train_conf = self.training
        model_conf = self.model
        
        B = train_conf.batch_size # 3
        S = train_conf.seq_len #  8
        H = model_conf.hidden_dim # 64
        N = model_conf.num_heads # 4
        
        if train_conf.dtype_bytes == 4:
            part_2 = 4 * (3 * B * S * H) # save q, k, v inputs
            part_3 = 4 * (4 * S * (H//N)) # rope q_cos, q_sin, k_cos, k_sin
            part_4 = 4 * (2 * B * N * S * (H//N)) # q rot, k rot
            part_5 = 4 * (B * N * S * S) # q*kT result
            part_6 = 4 * (B * N * S * (H//N)) # v
            part_7 = 4 * (B * N * S * S) # softmax + scale
            
            part_x = 4 * (H * H + B * S * H) # out proj
            
            #return part_2 + part_3 + part_4 + part_5 + part_6 + part_7 + part_x
        
        else:
            part_2 = 2 * (3 * B * S * H) # save q, k, v inputs
            part_3 = 4 * (4 * S * (H//N)) # rope q_cos, q_sin, k_cos, k_sin
            part_4 = 2 * (2 * B * N * S * (H//N)) # q rot, k rot
            part_5 = 2 * (B * N * S * S) # q*kT result
            part_6 = 2 * (B * N * S * (H//N)) # v
            part_7 = 2 * (B * N * S * S) # softmax + scale
            
            part_x = 2 * (H * H + B * S * H) # out proj
            
            #return part_2 + part_3 + part_4 + part_5 + part_6 + part_7 + part_x
        
        return train_conf.dtype_bytes * (
            7 * B * S * H + 2 * B * N * S * S
        ) + 4 * (4 * S * (H // N))
        

        
    
    def calculate_peak_memory(self) -> int:
        """Total peak memory = params + grads + optimizer + activations."""
        return (
            self.calculate_param_memory()
            + self.calculate_gradient_memory()
            + self.calculate_optimizer_memory()
            + self.calculate_activation_memory()
        )

    def time_embedding_ms(self) -> float:
        """
        Embedding lookup time (ms).
        """
        train_conf = self.training
        model_conf = self.model
        
        B = train_conf.batch_size # 2
        S = train_conf.seq_len # 8
        H = model_conf.hidden_dim # 64
        
        flops = 0
        token_id_bytes = 8 * B * S
        selected_row_bytes = self.training.dtype_bytes * 2 * B * S * H
        gather_bandwidth_utilization = 0.3
        memory_bytes = int((token_id_bytes + selected_row_bytes) / gather_bandwidth_utilization)
        
        return self.roofline_time_ms(flops, memory_bytes)
    
    def time_rms_norm_ms(self) -> float:
        """
        RMSNorm time - baseline, not fused (ms).
        """
        train_conf = self.training
        model_conf = self.model
        dtype = self.training.dtype_bytes
        
        B = train_conf.batch_size # 2
        S = train_conf.seq_len # 8
        H = model_conf.hidden_dim # 64
        
        memory_bytes = 0
        flops = 0
        
        memory_bytes += B * S * dtype * H # read input
        memory_bytes += B * S * 4 * H # write input fp32
        
        memory_bytes += 2 * B * S * 4 * H # read, square, write
        flops += 2 * B * S * H # x2 because bf16 flops downstream, here is fp32
        
        memory_bytes += B * S * 4 * H # read, mean, write B * S
        flops += B * S * H
        
        memory_bytes += 2 * B * S * 4 * H # read , mul, write
        flops += 2 * B * S * H # x2 because bf16 flops downstream, here is fp32
        
        memory_bytes += 2 * B * S * 4 * H # read weights, add 1, write
        flops += 2 * B * S * H # x2 because bf16 flops downstream, here is fp32
        
        memory_bytes += 2 * B * S * 4 * H # read sum, mul, write
        flops += 2 * B * S * H # x2 because bf16 flops downstream, here is fp32
        
        return self.roofline_time_ms(flops, memory_bytes)
    
    def time_attention_ms(self) -> float:
        """
        Standard attention time (ms).
        """
        train_conf = self.training
        model_conf = self.model
        dtype = self.training.dtype_bytes
        
        B = train_conf.batch_size # 2
        S = train_conf.seq_len # 8
        H = model_conf.hidden_dim # 64
        n_heads = model_conf.num_heads

        head_dim = H // n_heads
        hidden_elems = B * S * H
        score_elems = B * n_heads * S * S
        hidden_bytes = dtype * hidden_elems
        hidden_fp32_bytes = 4 * hidden_elems
        score_bytes = dtype * score_elems

        total_ms = 0.0

        def op_time_ms(
            flops: int,
            memory_bytes: int,
            *,
            compute_utilization: float = 1.0,
            memory_utilization: float = 1.0,
        ) -> float:
            effective_flops = int(flops / compute_utilization)
            effective_memory_bytes = int(memory_bytes / memory_utilization)
            return self.roofline_time_ms(effective_flops, effective_memory_bytes)

        # q_proj, k_proj, v_proj are three separate eager linear calls.
        qkv_linear_flops = 2 * B * S * H * H
        qkv_linear_bytes = hidden_bytes + dtype * H * H + hidden_bytes
        total_ms += 3 * op_time_ms(
            qkv_linear_flops,
            qkv_linear_bytes,
            compute_utilization=0.9,
        )

        # Baseline RoPE is unfused eager PyTorch. It materializes x.float(),
        # rotated cat, two multiply outputs, an add output, and a cast back.
        rope_cos_sin_bytes = 2 * B * n_heads * S * head_dim * 4
        rope_one_tensor_bytes = (
            hidden_bytes + hidden_fp32_bytes      # x.float()
            + 2 * hidden_fp32_bytes              # torch.cat([-x2, x1])
            + hidden_fp32_bytes + rope_cos_sin_bytes // 2 + hidden_fp32_bytes
            + hidden_fp32_bytes + rope_cos_sin_bytes // 2 + hidden_fp32_bytes
            + 3 * hidden_fp32_bytes              # add the two multiply outputs
            + hidden_fp32_bytes + hidden_bytes   # cast back to original dtype
        )
        rope_flops = 2 * 6 * hidden_elems
        total_ms += op_time_ms(
            rope_flops,
            2 * rope_one_tensor_bytes,
            memory_utilization=0.9,
        )

        # attn_weights = q @ k.T
        total_ms += op_time_ms(
            2 * B * S * S * H,
            2 * hidden_bytes + score_bytes,
            compute_utilization=0.9,
        )

        # The "* scale" after matmul is a separate full score-tensor pass.
        total_ms += op_time_ms(
            score_elems,
            2 * score_bytes,
            memory_utilization=0.9,
        )

        # torch.ones + torch.triu + masked_fill over the full score tensor.
        causal_mask_bytes = S * S
        total_ms += op_time_ms(
            score_elems,
            2 * score_bytes + B * n_heads * causal_mask_bytes + 3 * causal_mask_bytes,
            memory_utilization=0.9,
        )

        # Eager softmax is not just one read and one write. It performs row max,
        # exp/sum, and normalization traffic over the score tensor.
        total_ms += op_time_ms(
            5 * score_elems,
            4 * score_bytes,
            memory_utilization=0.9,
        )

        # Dropout with p=0 is effectively a no-op in these tests.

        # out = attn_weights @ v
        total_ms += op_time_ms(
            2 * B * S * S * H,
            score_bytes + hidden_bytes + hidden_bytes,
            compute_utilization=0.9,
        )

        # out.transpose(1, 2).contiguous()
        total_ms += op_time_ms(
            0,
            2 * hidden_bytes,
            memory_utilization=0.9,
        )

        # out_proj
        total_ms += op_time_ms(
            2 * B * S * H * H,
            hidden_bytes + dtype * H * H + hidden_bytes,
            compute_utilization=0.9,
        )

        return total_ms
    
    def time_mlp_ms(self) -> float:
        """
        MLP time - baseline (ms).
        """
        train_conf = self.training
        model_conf = self.model
        dtype = self.training.dtype_bytes

        B = train_conf.batch_size
        S = train_conf.seq_len
        H = model_conf.hidden_dim
        I = model_conf.intermediate_dim

        tokens = B * S
        hidden_elems = tokens * H
        intermediate_elems = tokens * I
        hidden_bytes = dtype * hidden_elems
        intermediate_bytes = dtype * intermediate_elems

        total_ms = 0.0

        def op_time_ms(
            flops: int,
            memory_bytes: int,
            *,
            compute_utilization: float = 1.0,
            memory_utilization: float = 1.0,
        ) -> float:
            effective_flops = int(flops / compute_utilization)
            effective_memory_bytes = int(memory_bytes / memory_utilization)
            return self.roofline_time_ms(effective_flops, effective_memory_bytes)

        # gate_proj and up_proj are separate eager linear calls.
        up_or_gate_flops = 2 * tokens * H * I
        up_or_gate_bytes = hidden_bytes + dtype * H * I + intermediate_bytes
        total_ms += 2 * op_time_ms(
            up_or_gate_flops,
            up_or_gate_bytes,
            compute_utilization=0.9,
        )

        # gate.clamp(max=limit)
        total_ms += op_time_ms(
            intermediate_elems,
            2 * intermediate_bytes,
            memory_utilization=0.9,
        )

        # up.clamp(min=-limit, max=limit)
        total_ms += op_time_ms(
            intermediate_elems,
            2 * intermediate_bytes,
            memory_utilization=0.9,
        )

        # gate * alpha
        total_ms += op_time_ms(
            intermediate_elems,
            2 * intermediate_bytes,
            memory_utilization=0.9,
        )

        # sigmoid(gate * alpha)
        total_ms += op_time_ms(
            4 * intermediate_elems,
            2 * intermediate_bytes,
            memory_utilization=0.9,
        )

        # gate * sigmoid(...)
        total_ms += op_time_ms(
            intermediate_elems,
            3 * intermediate_bytes,
            memory_utilization=0.9,
        )

        # up + 1
        total_ms += op_time_ms(
            intermediate_elems,
            2 * intermediate_bytes,
            memory_utilization=0.9,
        )

        # (up + 1) * glu
        total_ms += op_time_ms(
            intermediate_elems,
            3 * intermediate_bytes,
            memory_utilization=0.9,
        )

        # down_proj
        total_ms += op_time_ms(
            2 * tokens * I * H,
            intermediate_bytes + dtype * I * H + hidden_bytes,
            compute_utilization=0.9,
        )

        return total_ms
    
    def time_lm_head_ms(self) -> float:
        """
        LM head projection time (ms).
        """
        train_conf = self.training
        model_conf = self.model
        dtype = self.training.dtype_bytes

        B = train_conf.batch_size
        S = train_conf.seq_len
        H = model_conf.hidden_dim
        V = model_conf.vocab_size

        tokens = B * S
        flops = 2 * tokens * H * V
        memory_bytes = dtype * tokens * H + dtype * H * V + dtype * tokens * V

        return self.roofline_time_ms(
            int(flops / 0.7),
            memory_bytes,
        )
    
    def time_loss_ms(self) -> float:
        """
        Cross-entropy loss time - baseline (ms).
        """
        train_conf = self.training
        model_conf = self.model
        dtype = self.training.dtype_bytes

        B = train_conf.batch_size
        S = train_conf.seq_len
        V = model_conf.vocab_size

        tokens = B * (S - 1)
        logits_bytes = dtype * tokens * V
        logits_fp32_bytes = 4 * tokens * V
        label_bytes = 8 * tokens

        total_ms = 0.0

        def op_time_ms(
            flops: int,
            memory_bytes: int,
            *,
            compute_utilization: float = 1.0,
            memory_utilization: float = 1.0,
        ) -> float:
            effective_flops = int(flops / compute_utilization)
            effective_memory_bytes = int(memory_bytes / memory_utilization)
            return self.roofline_time_ms(effective_flops, effective_memory_bytes)

        # shift_logits.contiguous().float() materializes the shifted logits in fp32.
        total_ms += op_time_ms(
            0,
            logits_bytes + logits_fp32_bytes,
            memory_utilization=0.9,
        )

        # shift_labels.contiguous()
        total_ms += op_time_ms(
            0,
            2 * label_bytes,
            memory_utilization=0.9,
        )

        # F.cross_entropy = log_softmax + nll_loss. In eager PyTorch this is
        # several passes over the vocabulary dimension plus label gathers.
        total_ms += op_time_ms(
            5 * tokens * V,
            4 * logits_fp32_bytes + label_bytes,
            memory_utilization=0.9,
        )

        return total_ms

    def calculate_communication_volume(self) -> int:
        """
        DDP all-reduce volume (bytes).
        
        all-reduce: 2 * (n-1)/n * gradient_size
        ≈ 2 * gradient_size for large n
        """
        train_conf = self.training
        
        n = train_conf.num_gpus
        gradient_size = self.calculate_gradient_memory()
        return int(2 * (n - 1) / n * gradient_size)
    
    def time_communication_ms(self) -> float:
        """
        DDP communication time (ms).
        """
        volume = self.calculate_communication_volume()
        interconnect = self.gpu.interconnect_bandwidth_gbps
        
        volume_gb = volume / 1e9
        time_s = volume_gb / interconnect
        return time_s * 1000
        
    
    def overlap_efficiency(self) -> float:
        """
        DDP overlap efficiency (0.0 to 1.0).
        
        DDP overlaps gradient all-reduce with backward computation.
        Estimate based on your analysis.
        """
        # DDP launches gradient all-reduces as parameter buckets become ready
        # during backward. The last bucket and small/bubbled collectives cannot
        # be hidden, so treat most, but not all, communication as overlapped.
        backward_ms = self.time_backward_pass_ms()
        communication_ms = self.time_communication_ms()
        if communication_ms == 0:
            return 1.0

        overlap_window_ms = 0.8 * backward_ms
        overlapped_ms = min(communication_ms, overlap_window_ms)
        return max(0.0, min(1.0, overlapped_ms / communication_ms))
    
    def time_total_step_ms(self) -> float:
        """
        Total step time accounting for compute/comm overlap (ms).
        
        Consider how to combine compute time and communication time
        with overlap efficiency.
        """
        compute_ms = self.time_forward_backward_ms()
        communication_ms = self.time_communication_ms()
        non_overlapped_communication_ms = communication_ms * (1.0 - self.overlap_efficiency())
        return compute_ms + non_overlapped_communication_ms
