"""
Runtime-backed tests for calculators/efficient_calculator.py.

These tests derive expectations from real efficient_model tensors and then
apply the FSDP sharding/collective rules the calculator is supposed to model.
"""

import pytest
import torch
from time import perf_counter
from torch.autograd.graph import saved_tensors_hooks

from calculators.base import GPUSpec, ModelConfig, TrainingConfig
from calculators.efficient_calculator import EfficientCalculator
from config import TransformerConfig
from efficient_model.attention import MultiHeadAttention
from efficient_model.loss import CrossEntropyLoss
from efficient_model.norm import RMSNorm
from efficient_model.swiglu import SwiGLUFeedForward
from efficient_model.transformer import EfficientTransformer


@pytest.fixture
def gpu():
    return GPUSpec(
        name="H100 SXM NVLink",
        memory_bandwidth_gbps=2400,
        flops_bf16=800,
        interconnect_bandwidth_gbps=400,
    )


@pytest.fixture
def model_config():
    return ModelConfig(
        vocab_size=64,
        hidden_dim=16,
        num_heads=4,
        num_layers=2,
        intermediate_dim=32,
        max_seq_len=32,
    )


@pytest.fixture
def training_config():
    return TrainingConfig(
        batch_size=2,
        seq_len=16,
        num_gpus=4,
        dtype_bytes=torch.tensor([], dtype=torch.bfloat16).element_size(),
    )


@pytest.fixture
def transformer_config(model_config):
    return TransformerConfig(
        vocab_size=model_config.vocab_size,
        hidden_dim=model_config.hidden_dim,
        num_heads=model_config.num_heads,
        num_layers=model_config.num_layers,
        intermediate_dim=model_config.intermediate_dim,
        max_seq_len=model_config.max_seq_len,
        dropout=0.0,
    )


@pytest.fixture
def model(transformer_config, device):
    return EfficientTransformer(transformer_config).to(device=device)


@pytest.fixture
def calculator(model_config, training_config, gpu):
    return EfficientCalculator(model_config, training_config, gpu)


def model_param_bytes(model, dtype):
    return sum(torch.empty_like(param, dtype=dtype).nbytes for param in model.parameters())


def largest_block_param_bytes(model, dtype):
    blocks = list(model.layers) + [model.embedding, model.ln_f, model.lm_head]
    return max(
        sum(torch.empty_like(param, dtype=dtype).nbytes for param in block.parameters())
        for block in blocks
    )


def saved_tensor_bytes_for_forward_backward(model, training_config):
    device = next(model.parameters()).device
    input_ids = torch.randint(
        0,
        model.config.vocab_size,
        (training_config.batch_size, training_config.seq_len),
        device=device,
    )
    labels = input_ids.clone()
    excluded_ptrs = {param.data_ptr() for param in model.parameters()}
    total = 0

    def pack(tensor):
        nonlocal total
        if tensor.data_ptr() not in excluded_ptrs:
            total += tensor.nbytes
        return tensor

    def unpack(tensor):
        return tensor

    model.zero_grad(set_to_none=True)
    with saved_tensors_hooks(pack, unpack):
        loss = model(input_ids, labels=labels)
        loss.backward()

    return total


def synchronize_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure_ms(fn, device, warmup, repeats):
    synchronize_if_needed(device)

    for _ in range(warmup):
        fn()
        synchronize_if_needed(device)

    if device.type == "cuda":
        with torch.cuda.device(device):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            for _ in range(repeats):
                fn()
            end.record()
            torch.cuda.synchronize(device)

            return start.elapsed_time(end) / repeats

    start = perf_counter()
    for _ in range(repeats):
        fn()
    end = perf_counter()

    return (end - start) * 1000 / repeats


def measure_forward_ms(fn, device, warmup=5, repeats=10):
    with torch.no_grad():
        return _measure_ms(fn, device, warmup, repeats)


def measure_forward_backward_ms(fn, device, warmup=3, repeats=5):
    return _measure_ms(fn, device, warmup, repeats)


def measure_forward_ms_wall_clock(fn, device, warmup=5, repeats=10):
    """Wall-clock fallback for debugging CPU launch overhead."""
    with torch.no_grad():
        synchronize_if_needed(device)
        for _ in range(warmup):
            fn()
            synchronize_if_needed(device)

        start = perf_counter()
        for _ in range(repeats):
            fn()
        synchronize_if_needed(device)
        end = perf_counter()

    return (end - start) * 1000 / repeats


# class TestEfficientMemory:
#     def test_calculate_total_params_matches_real_model(self, calculator, model):
#         real_model_params = sum(param.numel() for param in model.parameters())

#         assert calculator.calculate_total_params() == real_model_params

#     def test_calculate_param_memory_matches_fp32_sharded_params(self, calculator, model):
#         expected = model_param_bytes(model, torch.float32) // calculator.training.num_gpus

#         assert calculator.calculate_param_memory() == expected

#     def test_calculate_gradient_memory_matches_fp32_sharded_gradients(self, calculator, model):
#         expected = model_param_bytes(model, torch.float32) // calculator.training.num_gpus

#         assert calculator.calculate_gradient_memory() == expected

#     def test_calculate_optimizer_memory_matches_three_sharded_fp32_states(self, calculator, model):
#         expected = 3 * model_param_bytes(model, torch.float32) // calculator.training.num_gpus

#         assert calculator.calculate_optimizer_memory() == expected

#     def test_calculate_fsdp_buffer_memory_matches_runtime_largest_unit_buffers(self, calculator, model):
#         largest_unit_bytes = largest_block_param_bytes(model, torch.bfloat16)
#         expected = 4 * largest_unit_bytes

#         assert calculator.calculate_fsdp_buffer_memory() == expected

#     @pytest.mark.parametrize(
#         "model_dtype,dtype_bytes",
#         [
#             (torch.float32, 4),
#             (torch.bfloat16, 2),
#         ],
#     )
#     def test_calculate_activation_memory_matches_saved_tensors(
#         self,
#         calculator,
#         model,
#         training_config,
#         gpu,
#         model_dtype,
#         dtype_bytes,
#     ):
#         model = model.to(dtype=model_dtype)
#         dtype_training_config = TrainingConfig(
#             batch_size=training_config.batch_size,
#             seq_len=training_config.seq_len,
#             num_gpus=training_config.num_gpus,
#             dtype_bytes=dtype_bytes,
#         )
#         dtype_calculator = EfficientCalculator(calculator.model, dtype_training_config, gpu)
#         saved_bytes = saved_tensor_bytes_for_forward_backward(model, training_config)

#         assert dtype_calculator.calculate_activation_memory() == pytest.approx(saved_bytes, rel=0.35)

#     def test_calculate_peak_memory_sums_all_reported_components(self, calculator):
#         expected = (
#             calculator.calculate_param_memory()
#             + calculator.calculate_gradient_memory()
#             + calculator.calculate_optimizer_memory()
#             + calculator.calculate_fsdp_buffer_memory()
#             + calculator.calculate_activation_memory()
#         )

#         assert calculator.calculate_peak_memory() == expected


class TestEfficientTime:
    @pytest.fixture
    def benchmark_model_config(self):
        return ModelConfig(
            vocab_size=16384,
            hidden_dim=1024,
            num_heads=16,
            num_layers=2,
            intermediate_dim=4096,
            max_seq_len=1024,
        )

    @pytest.fixture
    def benchmark_training_config(self):
        return TrainingConfig(
            batch_size=4,
            seq_len=1024,
            num_gpus=4,
            dtype_bytes=torch.tensor([], dtype=torch.bfloat16).element_size(),
        )

    @pytest.fixture
    def benchmark_transformer_config(self, benchmark_model_config):
        return TransformerConfig(
            vocab_size=benchmark_model_config.vocab_size,
            hidden_dim=benchmark_model_config.hidden_dim,
            num_heads=benchmark_model_config.num_heads,
            num_layers=benchmark_model_config.num_layers,
            intermediate_dim=benchmark_model_config.intermediate_dim,
            max_seq_len=benchmark_model_config.max_seq_len,
            dropout=0.0,
        )

    @pytest.fixture
    def benchmark_calculator(self, benchmark_model_config, benchmark_training_config, gpu):
        return EfficientCalculator(benchmark_model_config, benchmark_training_config, gpu)

    @pytest.fixture
    def runtime_device(self):
        if not torch.cuda.is_available():
            pytest.skip("real time vs roofline tests require CUDA")
        return torch.device("cuda")

    @pytest.fixture
    def runtime_dtype(self):
        return torch.bfloat16

    def assert_measured_time_matches_prediction(
        self,
        fn,
        predicted_ms,
        runtime_device,
        rel=0.3,
    ):
        measured_ms = measure_forward_ms(fn, runtime_device)

        assert measured_ms > 0
        assert predicted_ms > 0
        assert torch.isfinite(torch.tensor(measured_ms))
        assert measured_ms == pytest.approx(predicted_ms, rel=rel)

    def test_time_loss_ms_is_fused_into_lm_head(self, calculator):
        assert calculator.time_loss_ms() == pytest.approx(0.0)

    def test_embedding_real_time_tracks_calculator_prediction(
        self,
        benchmark_transformer_config,
        benchmark_training_config,
        benchmark_calculator,
        runtime_device,
        runtime_dtype,
    ):
        embedding = torch.nn.Embedding(
            benchmark_transformer_config.vocab_size,
            benchmark_transformer_config.hidden_dim,
        ).to(device=runtime_device, dtype=runtime_dtype)
        input_ids = torch.randint(
            0,
            benchmark_transformer_config.vocab_size,
            (benchmark_training_config.batch_size, benchmark_training_config.seq_len),
            device=runtime_device,
        )

        self.assert_measured_time_matches_prediction(
            lambda: embedding(input_ids),
            benchmark_calculator.time_embedding_ms(),
            runtime_device,
        )

    def test_rms_norm_real_time_tracks_calculator_prediction(
        self,
        benchmark_transformer_config,
        benchmark_training_config,
        benchmark_calculator,
        runtime_device,
        runtime_dtype,
    ):
        rms_norm = RMSNorm(benchmark_transformer_config.hidden_dim).to(
            device=runtime_device,
            dtype=runtime_dtype,
        )
        x = torch.randn(
            benchmark_training_config.batch_size,
            benchmark_training_config.seq_len,
            benchmark_transformer_config.hidden_dim,
            device=runtime_device,
            dtype=runtime_dtype,
        )

        self.assert_measured_time_matches_prediction(
            lambda: rms_norm(x),
            benchmark_calculator.time_rms_norm_ms(),
            runtime_device,
        )

    def test_attention_real_time_tracks_calculator_prediction(
        self,
        benchmark_transformer_config,
        benchmark_training_config,
        benchmark_calculator,
        runtime_device,
        runtime_dtype,
    ):
        attention = MultiHeadAttention(benchmark_transformer_config).to(
            device=runtime_device,
            dtype=runtime_dtype,
        )
        x = torch.randn(
            benchmark_training_config.batch_size,
            benchmark_training_config.seq_len,
            benchmark_transformer_config.hidden_dim,
            device=runtime_device,
            dtype=runtime_dtype,
        )

        self.assert_measured_time_matches_prediction(
            lambda: attention(x),
            benchmark_calculator.time_attention_ms(),
            runtime_device,
        )

    def test_mlp_real_time_tracks_calculator_prediction(
        self,
        benchmark_transformer_config,
        benchmark_training_config,
        benchmark_calculator,
        runtime_device,
        runtime_dtype,
    ):
        mlp = SwiGLUFeedForward(
            benchmark_transformer_config.hidden_dim,
            benchmark_transformer_config.intermediate_dim,
        ).to(device=runtime_device, dtype=runtime_dtype)
        x = torch.randn(
            benchmark_training_config.batch_size,
            benchmark_training_config.seq_len,
            benchmark_transformer_config.hidden_dim,
            device=runtime_device,
            dtype=runtime_dtype,
        )

        self.assert_measured_time_matches_prediction(
            lambda: mlp(x),
            benchmark_calculator.time_mlp_ms(),
            runtime_device,
        )

    def test_fused_lm_head_loss_real_time_tracks_calculator_prediction(
        self,
        benchmark_transformer_config,
        benchmark_training_config,
        benchmark_calculator,
        runtime_device,
        runtime_dtype,
    ):
        loss_fn = CrossEntropyLoss()
        hidden_states = torch.randn(
            benchmark_training_config.batch_size,
            benchmark_training_config.seq_len,
            benchmark_transformer_config.hidden_dim,
            device=runtime_device,
            dtype=runtime_dtype,
        )
        lm_head = torch.nn.Linear(
            benchmark_transformer_config.hidden_dim,
            benchmark_transformer_config.vocab_size,
            bias=False,
        ).to(device=runtime_device, dtype=runtime_dtype)
        labels = torch.randint(
            0,
            benchmark_transformer_config.vocab_size,
            (benchmark_training_config.batch_size, benchmark_training_config.seq_len),
            device=runtime_device,
        )

        self.assert_measured_time_matches_prediction(
            lambda: loss_fn(hidden_states, lm_head.weight, labels),
            benchmark_calculator.time_lm_head_ms(),
            runtime_device,
        )

    def test_time_forward_pass_ms_sums_model_operations(self, calculator):
        expected = (
            calculator.time_embedding_ms()
            + calculator.model.num_layers
            * (
                2 * calculator.time_rms_norm_ms()
                + calculator.time_attention_ms()
                + calculator.time_mlp_ms()
            )
            + calculator.time_rms_norm_ms()
            + calculator.time_lm_head_ms()
            + calculator.time_loss_ms()
        )

        assert calculator.time_forward_pass_ms() == pytest.approx(expected)

    def test_time_backward_pass_ms_defaults_to_twice_forward(self, calculator):
        assert calculator.time_backward_pass_ms() == pytest.approx(2 * calculator.time_forward_pass_ms())

    def test_time_forward_backward_ms_sums_forward_and_backward(self, calculator):
        expected = calculator.time_forward_pass_ms() + calculator.time_backward_pass_ms()

        assert calculator.time_forward_backward_ms() == pytest.approx(expected)

    @pytest.mark.parametrize("num_gpus", [1, 2])
    def test_total_step_time_tracks_real_model_forward_backward(
        self,
        num_gpus,
        benchmark_model_config,
        benchmark_transformer_config,
        benchmark_training_config,
        gpu,
        runtime_device,
        runtime_dtype,
    ):
        training_config = TrainingConfig(
            batch_size=benchmark_training_config.batch_size,
            seq_len=benchmark_training_config.seq_len,
            num_gpus=num_gpus,
            dtype_bytes=benchmark_training_config.dtype_bytes,
        )
        calculator = EfficientCalculator(benchmark_model_config, training_config, gpu)
        model = EfficientTransformer(benchmark_transformer_config).to(
            device=runtime_device,
            dtype=runtime_dtype,
        )
        input_ids = torch.randint(
            0,
            benchmark_transformer_config.vocab_size,
            (benchmark_training_config.batch_size, benchmark_training_config.seq_len),
            device=runtime_device,
        )
        labels = input_ids.clone()

        def forward_backward():
            model.zero_grad(set_to_none=True)
            loss = model(input_ids, labels=labels)
            loss.backward()

        measured_ms = measure_forward_backward_ms(forward_backward, runtime_device)
        modeled_non_overlapped_comm_ms = (
            calculator.time_communication_ms()
            * (1.0 - calculator.overlap_efficiency())
        )
        measured_plus_modeled_comm_ms = measured_ms + modeled_non_overlapped_comm_ms
        predicted_ms = calculator.time_total_step_ms()

        assert measured_ms > 0
        assert predicted_ms > 0
        assert torch.isfinite(torch.tensor(measured_ms))
        assert measured_ms == pytest.approx(calculator.time_forward_backward_ms(), rel=0.3)
        assert measured_plus_modeled_comm_ms == pytest.approx(predicted_ms, rel=0.3)


class TestEfficientCommunication:
    def test_calculate_allgather_volume_matches_fsdp_collective_on_full_param_bytes(
        self,
        calculator,
        model,
    ):
        total_bytes = model_param_bytes(model, torch.bfloat16)
        expected = int(
            (calculator.training.num_gpus - 1)
            / calculator.training.num_gpus
            * total_bytes
        )

        assert calculator.calculate_allgather_volume() == expected

    def test_calculate_reducescatter_volume_matches_fsdp_collective_on_full_grad_bytes(
        self,
        calculator,
        model,
    ):
        total_bytes = model_param_bytes(model, torch.bfloat16)
        expected = int(
            (calculator.training.num_gpus - 1)
            / calculator.training.num_gpus
            * total_bytes
        )

        assert calculator.calculate_reducescatter_volume() == expected

    def test_calculate_communication_volume_sums_fsdp_collectives(self, calculator):
        expected = 2 * calculator.calculate_allgather_volume() + calculator.calculate_reducescatter_volume()

        assert calculator.calculate_communication_volume() == expected

    def test_time_communication_ms_uses_interconnect_bandwidth(self, calculator):
        expected = (
            calculator.calculate_communication_volume()
            / (calculator.gpu.interconnect_bandwidth_gbps * 1e9)
            * 1000
        )

        assert calculator.time_communication_ms() == pytest.approx(expected)

    def test_overlap_efficiency_is_a_fraction(self, calculator):
        assert 0.0 <= calculator.overlap_efficiency() <= 1.0

    def test_time_total_step_ms_adds_non_overlapped_communication(self, calculator):
        expected = (
            calculator.time_forward_backward_ms()
            + calculator.time_communication_ms() * (1.0 - calculator.overlap_efficiency())
        )

        assert calculator.time_total_step_ms() == pytest.approx(expected)

    def test_time_total_step_ms_equals_forward_backward_for_single_gpu(
        self,
        model_config,
        training_config,
        gpu,
    ):
        single_gpu_training = TrainingConfig(
            batch_size=training_config.batch_size,
            seq_len=training_config.seq_len,
            num_gpus=1,
            dtype_bytes=training_config.dtype_bytes,
        )
        calculator = EfficientCalculator(model_config, single_gpu_training, gpu)

        assert calculator.calculate_communication_volume() == 0
        assert calculator.time_communication_ms() == pytest.approx(0.0)
        assert calculator.time_total_step_ms() == pytest.approx(calculator.time_forward_backward_ms())

    def test_time_total_step_ms_adds_only_non_overlapped_fsdp_tail_for_two_gpus(
        self,
        model_config,
        training_config,
        gpu,
    ):
        two_gpu_training = TrainingConfig(
            batch_size=training_config.batch_size,
            seq_len=training_config.seq_len,
            num_gpus=2,
            dtype_bytes=training_config.dtype_bytes,
        )
        calculator = EfficientCalculator(model_config, two_gpu_training, gpu)

        compute_ms = calculator.time_forward_backward_ms()
        communication_ms = calculator.time_communication_ms()
        overlap = calculator.overlap_efficiency()
        expected = compute_ms + communication_ms * (1.0 - overlap)

        assert communication_ms > 0
        assert 0.0 <= overlap <= 1.0
        assert calculator.time_total_step_ms() >= compute_ms
        assert calculator.time_total_step_ms() <= compute_ms + communication_ms
        assert calculator.time_total_step_ms() == pytest.approx(expected)
