import pytest
import torch

import flashinfer
from flashinfer.utils import get_compute_capability


@pytest.mark.cuda
def test_trtllm_ragged_kv_large_stride_overflow():
    """
    Test that ragged KV with large numel (>2^31) doesn't cause TMA descriptor error.

    Constructs a scenario where key.numel() = 131072 * 128 * 192 > 2^31, which
    triggers int32 overflow in kStrideBatch. Before the fix, this caused negative
    stride and TMA descriptor error. After the fix, negative strideBatch is clamped
    to 0 for ragged layouts.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    if not hasattr(flashinfer.prefill, "trtllm_ragged_attention_deepseek"):
        pytest.skip("trtllm_ragged_attention_deepseek is not available in this build")

    device = torch.device("cuda")
    compute_capability = get_compute_capability(device)
    if compute_capability[0] != 10:
        pytest.skip(
            "TRTLLM-gen ragged attention requires SM100 and SM103 GPUs, "
            f"got sm{compute_capability[0]}{compute_capability[1]}"
        )

    torch.manual_seed(42)

    # Configuration that triggers numel > 2^31
    batch_size = 16
    max_kv_len = 8192
    num_kv_heads = 128
    head_dim_qk = 192
    head_dim_vo = 128

    # Construct ragged Q
    seq_lens_q = torch.randint(
        low=50, high=150, size=(batch_size,), device=device, dtype=torch.int32
    )
    cum_seq_lens_q = torch.cat(
        [
            torch.zeros(1, device=device, dtype=torch.int32),
            torch.cumsum(seq_lens_q, dim=0, dtype=torch.int32),
        ],
        dim=0,
    )
    total_q = int(cum_seq_lens_q[-1].item())
    max_q_len = int(seq_lens_q.max().item())

    q = torch.randn(
        total_q,
        num_kv_heads,
        head_dim_qk,
        device=device,
        dtype=torch.bfloat16,
    )

    # Construct ragged KV: total_kv = 16 * 8192 = 131072
    # key.numel() = 131072 * 128 * 192 = 3,221,225,472 (0xC0000000) > 2^31
    seq_lens_kv = torch.full(
        (batch_size,), max_kv_len, device=device, dtype=torch.int32
    )
    cum_seq_lens_kv = torch.arange(
        0,
        (batch_size + 1) * max_kv_len,
        max_kv_len,
        device=device,
        dtype=torch.int32,
    )
    total_kv = int(cum_seq_lens_kv[-1].item())

    k = torch.randn(
        total_kv,
        num_kv_heads,
        head_dim_qk,
        device=device,
        dtype=torch.bfloat16,
    )
    v = torch.randn(
        total_kv,
        num_kv_heads,
        head_dim_vo,
        device=device,
        dtype=torch.bfloat16,
    )

    workspace_buffer = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    scale = float(1.0 / (head_dim_qk**0.5))

    # Should not raise "buildNdTmaDescriptor: invalid argument" error
    output = flashinfer.prefill.trtllm_ragged_attention_deepseek(
        query=q,
        key=k,
        value=v,
        workspace_buffer=workspace_buffer,
        seq_lens=seq_lens_kv,
        max_q_len=max_q_len,
        max_kv_len=max_kv_len,
        bmm1_scale=scale,
        bmm2_scale=1.0,
        o_sf_scale=1.0,
        batch_size=batch_size,
        window_left=-1,
        cum_seq_lens_q=cum_seq_lens_q,
        cum_seq_lens_kv=cum_seq_lens_kv,
        enable_pdl=False,
        is_causal=True,
        return_lse=False,
    )

    # Basic shape check
    assert output.shape[0] == total_q
    assert output.shape[1] == num_kv_heads
    assert output.shape[2] == head_dim_vo


@pytest.mark.cuda
def test_trtllm_ragged_kv_positive_stride_wrap_noncausal_correctness():
    """
    Non-causal ragged DeepSeek attention should not inherit a fake batch stride.

    This uses a total KV token count whose key.numel() exceeds 2^32 and would
    wrap to a positive int32 value if used as kStrideBatch. Ragged KV has a
    singleton batch dimension in the TMA descriptor, so the only correct batch
    stride is zero.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    if not hasattr(flashinfer.prefill, "trtllm_ragged_attention_deepseek"):
        pytest.skip("trtllm_ragged_attention_deepseek is not available in this build")

    device = torch.device("cuda")
    compute_capability = get_compute_capability(device)
    if compute_capability[0] != 10:
        pytest.skip(
            "TRTLLM-gen ragged attention requires SM100 and SM103 GPUs, "
            f"got sm{compute_capability[0]}{compute_capability[1]}"
        )

    free_mem, _ = torch.cuda.mem_get_info(device)
    required_mem = 24 * 1024**3
    if free_mem < required_mem:
        pytest.skip(
            f"requires at least {required_mem / 1024**3:.0f} GiB free GPU memory"
        )

    torch.manual_seed(42)

    batch_size = 8
    max_q_len = 1
    max_kv_len = 24576
    num_kv_heads = 128
    head_dim_qk = 192
    head_dim_vo = 128

    total_kv = batch_size * max_kv_len
    key_numel = total_kv * num_kv_heads * head_dim_qk
    assert key_numel > 2**32
    assert key_numel % 2**32 < 2**31

    cum_seq_lens_q = torch.arange(
        0, batch_size + 1, device=device, dtype=torch.int32
    )
    seq_lens_kv = torch.full(
        (batch_size,), max_kv_len, device=device, dtype=torch.int32
    )
    cum_seq_lens_kv = torch.arange(
        0,
        (batch_size + 1) * max_kv_len,
        max_kv_len,
        device=device,
        dtype=torch.int32,
    )

    q = (
        torch.randn(
            int(cum_seq_lens_q[-1].item()),
            num_kv_heads,
            head_dim_qk,
            device=device,
            dtype=torch.float32,
        )
        * 0.1
    ).to(torch.bfloat16)
    k = (
        torch.randn(
            total_kv,
            num_kv_heads,
            head_dim_qk,
            device=device,
            dtype=torch.float32,
        )
        * 0.1
    ).to(torch.bfloat16)
    v = (
        torch.randn(
            total_kv,
            num_kv_heads,
            head_dim_vo,
            device=device,
            dtype=torch.float32,
        )
        * 0.1
    ).to(torch.bfloat16)

    workspace_buffer = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    workspace_buffer_ref = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device=device
    )
    scale = float(1.0 / (head_dim_qk**0.5))

    wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer_ref,
        kv_layout="NHD",
        backend="cutlass",
    )
    wrapper.plan(
        cum_seq_lens_q,
        cum_seq_lens_kv,
        num_kv_heads,
        num_kv_heads,
        head_dim_qk,
        head_dim_vo=head_dim_vo,
        causal=False,
        sm_scale=scale,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    output_ref, _ = wrapper.run(q, k, v, return_lse=True)

    output = torch.empty_like(output_ref)
    output_trtllm, lse_trtllm = flashinfer.prefill.trtllm_ragged_attention_deepseek(
        query=q,
        key=k,
        value=v,
        workspace_buffer=workspace_buffer,
        seq_lens=seq_lens_kv,
        max_q_len=max_q_len,
        max_kv_len=max_kv_len,
        bmm1_scale=scale,
        bmm2_scale=1.0,
        o_sf_scale=1.0,
        batch_size=batch_size,
        window_left=-1,
        cum_seq_lens_q=cum_seq_lens_q,
        cum_seq_lens_kv=cum_seq_lens_kv,
        enable_pdl=False,
        is_causal=False,
        return_lse=True,
        out=output,
    )

    assert torch.isfinite(output_trtllm).all()
    assert torch.isfinite(lse_trtllm).all()
    torch.testing.assert_close(output_trtllm, output_ref, atol=2e-2, rtol=2e-2)
