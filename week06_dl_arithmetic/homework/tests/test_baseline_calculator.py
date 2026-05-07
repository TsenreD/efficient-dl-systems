"""
Runtime-backed tests for calculators/baseline_calculator.py.

The expected values are derived from real PyTorch model tensors where possible,
so these tests can be used as a guide while implementing each calculator method.
"""

import pytest
import torch
from time import perf_counter
from torch.autograd.graph import saved_tensors_hooks

from calculators.base import GPUSpec, ModelConfig, TrainingConfig
from calculators.baseline_calculator import BaselineCalculator
from config import TransformerConfig
from model.loss import cross_entropy_loss
from model.norm import RMSNorm
from model.swiglu import SwiGLUFeedForward
from model.attention import MultiHeadAttention
from model.transformer import BaselineTransformer
from optimizer.ademamix import AdEMAMix


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
def model(transformer_config):
    return BaselineTransformer(transformer_config)


@pytest.fixture
def calculator(model_config, training_config, gpu):
    return BaselineCalculator(model_config, training_config, gpu)


def model_param_bytes(model, dtype):
    return sum(torch.empty_like(param, dtype=dtype).nbytes for param in model.parameters())


def optimizer_state_bytes_after_one_step(model):
    optimizer = AdEMAMix(model.parameters(), lr=1e-3)
    for param in model.parameters():
        param.grad = torch.ones_like(param)
    optimizer.step()

    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                total += value.nbytes
    return total


def saved_tensor_bytes_for_forward_backward(model, training_config):
    input_ids = torch.randint(
        0,
        model.config.vocab_size,
        (training_config.batch_size, training_config.seq_len),
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
        logits = model(input_ids)
        loss = cross_entropy_loss(logits, labels)
        loss.backward()

    return total


def synchronize_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_forward_ms(fn, device, warmup=2, repeats=5):
    with torch.no_grad():
        for _ in range(warmup):
            fn()

        synchronize_if_needed(device)
        start = perf_counter()
        for _ in range(repeats):
            fn()
        synchronize_if_needed(device)
        end = perf_counter()

    return (end - start) * 1000 / repeats


def measure_forward_backward_ms(fn, device, warmup=1, repeats=3):
    for _ in range(warmup):
        fn()

    synchronize_if_needed(device)
    start = perf_counter()
    for _ in range(repeats):
        fn()
    synchronize_if_needed(device)
    end = perf_counter()

    return (end - start) * 1000 / repeats


class TestBaselineMemory:
    def test_calculate_block_params_matches_one_real_transformer_block(self, calculator, model):
        real_block_params = sum(param.numel() for param in model.layers[0].parameters())

        assert calculator._calculate_block_params() == real_block_params

    def test_calculate_total_params_matches_real_model(self, calculator, model):
        real_model_params = sum(param.numel() for param in model.parameters())

        assert calculator.calculate_total_params() == real_model_params

    def test_calculate_param_memory_matches_bf16_params_plus_fp32_master(self, calculator, model):
        expected = model_param_bytes(model, torch.bfloat16) + model_param_bytes(model, torch.float32)

        assert calculator.calculate_param_memory() == expected

    def test_calculate_gradient_memory_matches_fp32_gradient_buffers(self, calculator, model):
        expected = model_param_bytes(model, torch.float32)

        assert calculator.calculate_gradient_memory() == expected

    def test_calculate_optimizer_memory_matches_real_ademamix_state(self, calculator, model):
        expected = optimizer_state_bytes_after_one_step(model)

        assert calculator.calculate_optimizer_memory() == expected

    @pytest.mark.parametrize(
        "model_dtype,dtype_bytes",
        [
            (torch.float32, 4),
            (torch.bfloat16, 2),
        ],
    )
    def test_calculate_activation_memory_matches_saved_tensors(
        self,
        calculator,
        model,
        training_config,
        gpu,
        model_dtype,
        dtype_bytes,
    ):
        model = model.to(dtype=model_dtype)
        dtype_training_config = TrainingConfig(
            batch_size=training_config.batch_size,
            seq_len=training_config.seq_len,
            num_gpus=training_config.num_gpus,
            dtype_bytes=dtype_bytes,
        )
        dtype_calculator = BaselineCalculator(calculator.model, dtype_training_config, gpu)
        saved_bytes = saved_tensor_bytes_for_forward_backward(model, training_config)

        assert dtype_calculator.calculate_activation_memory() == pytest.approx(saved_bytes, rel=0.35)

    def test_calculate_peak_memory_sums_all_reported_components(self, calculator):
        expected = (
            calculator.calculate_param_memory()
            + calculator.calculate_gradient_memory()
            + calculator.calculate_optimizer_memory()
            + calculator.calculate_activation_memory()
        )

        assert calculator.calculate_peak_memory() == expected



class TestBaselineRealTime:
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
        return BaselineCalculator(benchmark_model_config, benchmark_training_config, gpu)

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
            rel=0.3
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

    def test_lm_head_real_time_tracks_calculator_prediction(
        self,
        benchmark_transformer_config,
        benchmark_training_config,
        benchmark_calculator,
        runtime_device,
        runtime_dtype,
    ):
        lm_head = torch.nn.Linear(
            benchmark_transformer_config.hidden_dim,
            benchmark_transformer_config.vocab_size,
            bias=False,
        ).to(device=runtime_device, dtype=runtime_dtype)
        x = torch.randn(
            benchmark_training_config.batch_size,
            benchmark_training_config.seq_len,
            benchmark_transformer_config.hidden_dim,
            device=runtime_device,
            dtype=runtime_dtype,
        )

        self.assert_measured_time_matches_prediction(
            lambda: lm_head(x),
            benchmark_calculator.time_lm_head_ms(),
            runtime_device,
        )

    def test_loss_real_time_tracks_calculator_prediction(
        self,
        benchmark_transformer_config,
        benchmark_training_config,
        benchmark_calculator,
        runtime_device,
        runtime_dtype,
    ):
        logits = torch.randn(
            benchmark_training_config.batch_size,
            benchmark_training_config.seq_len,
            benchmark_transformer_config.vocab_size,
            device=runtime_device,
            dtype=runtime_dtype,
        )
        labels = torch.randint(
            0,
            benchmark_transformer_config.vocab_size,
            (benchmark_training_config.batch_size, benchmark_training_config.seq_len),
            device=runtime_device,
        )

        self.assert_measured_time_matches_prediction(
            lambda: cross_entropy_loss(logits, labels),
            benchmark_calculator.time_loss_ms(),
            runtime_device,
        )

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
        single_gpu_training_config = TrainingConfig(
            batch_size=benchmark_training_config.batch_size,
            seq_len=benchmark_training_config.seq_len,
            num_gpus=num_gpus,
            dtype_bytes=benchmark_training_config.dtype_bytes,
        )
        calculator = BaselineCalculator(
            benchmark_model_config,
            single_gpu_training_config,
            gpu,
        )
        model = BaselineTransformer(benchmark_transformer_config).to(
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
            logits = model(input_ids)
            loss = cross_entropy_loss(logits, labels)
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


# class TestBaselineCommunication:
#     def test_calculate_communication_volume_matches_ddp_all_reduce(self, calculator, model):
#         gradient_bytes = model_param_bytes(model, torch.float32)
#         expected = int(
#             2
#             * (calculator.training.num_gpus - 1)
#             / calculator.training.num_gpus
#             * gradient_bytes
#         )

#         assert calculator.calculate_communication_volume() == expected

#     def test_time_communication_ms_uses_interconnect_bandwidth(self, calculator):
#         expected = (
#             calculator.calculate_communication_volume()
#             / (calculator.gpu.interconnect_bandwidth_gbps * 1e9)
#             * 1000
#         )

#         assert calculator.time_communication_ms() == pytest.approx(expected)

#     def test_overlap_efficiency_is_a_fraction(self, calculator):
#         assert 0.0 <= calculator.overlap_efficiency() <= 1.0

#     def test_time_total_step_ms_adds_non_overlapped_communication(self, calculator):
#         expected = (
#             calculator.time_forward_backward_ms()
#             + calculator.time_communication_ms() * (1.0 - calculator.overlap_efficiency())
#         )

#         assert calculator.time_total_step_ms() == pytest.approx(expected)
