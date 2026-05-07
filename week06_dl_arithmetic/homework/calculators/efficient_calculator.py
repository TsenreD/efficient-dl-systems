"""
Efficient Calculator for FSDP implementation with efficient_model.
"""

from calculators.base import BaseCalculator


class EfficientCalculator(BaseCalculator):
    """
    Calculator for efficient implementation with FSDP.
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
        FSDP: sharded params (fp32).
        """
        return 4 * self.calculate_total_params() // self.training.num_gpus

    def calculate_gradient_memory(self) -> int:
        """
        FSDP: sharded gradients after reduce-scatter (fp32).
        """
        return 4 * self.calculate_total_params() // self.training.num_gpus

    def calculate_optimizer_memory(self) -> int:
        """
        FSDP: sharded optimizer states (fp32).
        
        AdEMAMix has 3 states: m (momentum), v (variance), nu (third moment)
        """
        return 12 * self.calculate_total_params() // self.training.num_gpus

    def calculate_fsdp_buffer_memory(self) -> int:
        """
        FSDP communication buffers (bf16).
        
        - 2 All-gather buffers: unsharded params for current unit
        - 2 Reduce-scatter buffers: gradients before sharding
        """
        
        config = self.model
        max_block_size = max(
            self._calculate_block_params(),
            config.vocab_size * config.hidden_dim
        )
        
        return 2 * 4 * max_block_size

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
    
    def _calculate_ce_act(self) -> int:
        train_conf = self.training
        model_conf = self.model
        
        B = train_conf.batch_size
        S = train_conf.seq_len
        H = model_conf.hidden_dim # 16
        V = model_conf.vocab_size # 64
        dtype = train_conf.dtype_bytes
        
        return 4 * (B * S * H)
    
    def _calculate_rmsnorm_act(self) -> int:
        train_conf = self.training
        model_conf = self.model
        
        B = train_conf.batch_size
        S = train_conf.seq_len
        H = model_conf.hidden_dim
        dtype = train_conf.dtype_bytes
        
        return dtype * (B * S * H + H) + 4 * B * S
        
    def _calculate_attn_act(self) -> int:
        train_conf = self.training
        model_conf = self.model
        
        B = train_conf.batch_size 
        S = train_conf.seq_len 
        H = model_conf.hidden_dim
        dtype = train_conf.dtype_bytes
        N = model_conf.num_heads
        
        mem = 0
        mem += dtype * B * S * H # x 
        mem += 4* dtype * B * S * H # q, k, v, out
        mem += 4 * B * N * S
        mem += dtype * B * S * H # out projected
        
        return 6 * dtype* B * S * H + 4 * B * N * S
    
    def _calculate_swiglu_act(self) -> int:
        train_conf = self.training
        model_conf = self.model
        
        B = train_conf.batch_size # 3
        S = train_conf.seq_len #  8
        H = model_conf.hidden_dim # 64
        I = model_conf.intermediate_dim
        dtype = train_conf.dtype_bytes
        
        mem = 0
        mem += dtype * B * S * H # x
        mem += 2 * dtype * B * S * I # gate + up proj
        
        return dtype * (B * S * H + 2 * B * S * I)

    def calculate_peak_memory(self) -> int:
        """Total peak memory including FSDP buffers."""
        return (
            self.calculate_param_memory()
            + self.calculate_gradient_memory()
            + self.calculate_optimizer_memory()
            + self.calculate_fsdp_buffer_memory()
            + self.calculate_activation_memory()
        )

    def time_embedding_ms(self) -> float:
        """Embedding lookup time - same as baseline (ms)."""
        train_conf = self.training
        model_conf = self.model
        
        B = train_conf.batch_size # 2
        S = train_conf.seq_len # 8
        H = model_conf.hidden_dim # 64
        
        flops = 0
        token_id_bytes = 8 * B * S
        selected_row_bytes = self.training.dtype_bytes * 2 * B * S * H
        gather_bandwidth_utilization = 0.9
        memory_bytes = int((token_id_bytes + selected_row_bytes) / gather_bandwidth_utilization)
        
        return self.roofline_time_ms(flops, memory_bytes)

    def time_rms_norm_ms(self) -> float:
        """
        Fused RMSNorm time (ms).
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
        
        flops += 2 * B * S * H # x2 because bf16 flops downstream, here is fp32
        
        flops += B * S * H
    
        flops += 2 * B * S * H # x2 because bf16 flops downstream, here is fp32
        
        flops += 2 * B * S * H # x2 because bf16 flops downstream, here is fp32
        
        memory_bytes += dtype * B * S  * H # read sum, mul, write to dtype
        flops += 2 * B * S * H # x2 because bf16 flops downstream, here is fp32
        
        return 6 * self.roofline_time_ms(flops, memory_bytes)

    def time_attention_ms(self) -> float:
        """
        Efficient attention time (ms).

        The efficient path uses one fused QKV projection, RoPE, FlashAttention,
        and one output projection. It does not materialize the S x S attention
        matrix, mask, or softmax tensors like the baseline path.
        """
        train_conf = self.training
        model_conf = self.model
        dtype = self.training.dtype_bytes
        
        B = train_conf.batch_size # 2
        S = train_conf.seq_len # 8
        H = model_conf.hidden_dim # 64
        n_heads = model_conf.num_heads

        tokens = B * S
        hidden_elems = B * S * H
        hidden_bytes = dtype * hidden_elems
        hidden_fp32_bytes = 4 * hidden_elems

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

        # qkv_proj: one larger GEMM, (B*S, H) @ (H, 3H).
        total_ms += op_time_ms(
            2 * tokens * H * (3 * H),
            hidden_bytes + dtype * H * (3 * H) + 3 * hidden_bytes,
            compute_utilization=0.75,
        )

        # RoPE on q and k. This is still expressed through PyTorch ops in the
        # model, so use a conservative memory utilization for the rowwise casts,
        # trig-table reads, multiplies, adds, and final bf16 outputs.
        rope_bytes = (
            2 * hidden_bytes
            + 4 * hidden_fp32_bytes
            + 2 * B * n_heads * S * (H // n_heads) * 4
            + 2 * hidden_bytes
        )
        total_ms += op_time_ms(
            12 * hidden_elems,
            rope_bytes,
            memory_utilization=0.5,
        )

        # FlashAttention performs QK^T, causal masking, online softmax, and PV
        # without writing the full attention matrix. Its effective utilization is
        # lower than a large square GEMM for this benchmark shape.
        flash_flops = 4 * B * S * S * H
        flash_bytes = 4 * hidden_bytes + 4 * B * n_heads * S
        total_ms += op_time_ms(
            flash_flops,
            flash_bytes,
            compute_utilization=0.12,
        )

        # out_proj.
        total_ms += op_time_ms(
            2 * tokens * H * H,
            hidden_bytes + dtype * H * H + hidden_bytes,
            compute_utilization=0.75,
        )

        return total_ms

    def time_mlp_ms(self) -> float:
        """
        Fused SwiGLU time (ms).
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

        # gate_proj, up_proj, and down_proj dominate the forward. The Triton
        # activation kernel avoids the many eager elementwise passes from the
        # baseline, so aggregate the full MLP as a compute-heavy fused block.
        flops = 6 * tokens * H * I + 8 * intermediate_elems
        memory_bytes = (
            2 * hidden_bytes
            + 2 * dtype * H * I
            + 3 * intermediate_bytes
            + dtype * I * H
            + hidden_bytes
        )

        return self.roofline_time_ms(
            int(flops / 0.7),
            memory_bytes,
        )

    def time_lm_head_ms(self) -> float:
        """
        LM head with fused loss (ms).
        """
        train_conf = self.training
        model_conf = self.model
        dtype = self.training.dtype_bytes

        B = train_conf.batch_size
        S = train_conf.seq_len
        H = model_conf.hidden_dim
        V = model_conf.vocab_size

        # The efficient loss shifts tokens, casts hidden states and lm_head
        # weights to fp32, then calls Liger fused linear cross entropy. It avoids
        # materializing full logits, but the actual work is fp32-ish and much
        # slower than the H100 bf16 tensor-core peak used by the roofline helper.
        tokens = B * (S - 1)
        flops = 2 * tokens * H * V + 4 * tokens * V
        memory_bytes = (
            dtype * tokens * H
            + 4 * tokens * H
            + dtype * H * V
            + 4 * H * V
            + 8 * tokens
        )

        return self.roofline_time_ms(
            int(flops / 0.055),
            memory_bytes,
        )

    def time_loss_ms(self) -> float:
        """
        Fused linear cross-entropy time (ms).
        """
        return 0.0

    def calculate_allgather_volume(self) -> int:
        """
        FSDP all-gather volume - forward pass (bytes).
        """
        n = self.training.num_gpus
        return int(2 * (n-1) * self.calculate_total_params() / n)

    def calculate_reducescatter_volume(self) -> int:
        """
        FSDP reduce-scatter volume - backward pass (bytes).
        """
        n = self.training.num_gpus
        return int(2 * (n-1) * self.calculate_total_params() / n)

    def calculate_communication_volume(self) -> int:
        """
        Total FSDP communication volume.
        
        = 2 * all-gather (forward + backward) + reduce-scatter (backward)
        """
        return 2 * self.calculate_allgather_volume() + self.calculate_reducescatter_volume()

    def time_communication_ms(self) -> float:
        """
        FSDP communication time (ms).
        
        time = total_volume / interconnect_bandwidth
        """
        volume = self.calculate_communication_volume()
        interconnect = self.gpu.interconnect_bandwidth_gbps
        
        volume_gb = volume / 1e9
        time_s = volume_gb / interconnect
        return time_s * 1000

    def overlap_efficiency(self) -> float:
        """
        FSDP overlap efficiency (0.0 to 1.0).
        
        FSDP can overlap:
        - All-gather of next layer with compute of current layer
        - Reduce-scatter of current layer with backward of next layer
        
        Estimate based on your analysis.
        """
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
