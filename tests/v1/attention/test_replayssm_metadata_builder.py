# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ReplaySSM metadata from BaseMambaAttentionMetadataBuilder.

Triton ReplaySSM plans write_pos/is_flush on the GPU (see
tests/kernels/mamba/test_replayssm_plan.py); the builder only supplies the
persistent buffers that planning fills, plus align-mode sequence lengths.
"""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    MockMambaBuilder,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.config.mamba import MambaBackendEnum
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.short_conv_attn import (
    PleShortConvAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec


def test_gdn_replay_admission_reset_includes_decode_promoted_prompt_tail():
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
    from vllm.v1.worker.utils import AttentionGroup

    config = create_vllm_config(max_num_seqs=4, max_num_batched_tokens=32)
    config.cache_config.use_replayssm_spec = True
    config.cache_config.mamba_cache_mode = "none"
    config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=3
    )
    device = torch.device("cuda")
    # Only request/attention metadata is exercised; no model forward is needed.
    state = MambaHybridModelState(config, torch.nn.Identity(), None, device)
    spec = MambaSpec(
        block_size=16,
        shapes=((7, 8), (1, 8, 8)),
        dtypes=(torch.float32, torch.float32),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )
    group = AttentionGroup(
        MambaAttentionBackendEnum.GDN_ATTN.get_class(), ["gdn"], spec, 0
    )
    group.create_metadata_builders(config, device)
    kv = KVCacheConfig(
        num_blocks=1000,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["gdn"], spec)],
    )
    batch = InputBatch.make_dummy(1, 4, InputBuffers(4, 32, device), is_padding=False)
    batch = replace(
        batch,
        seq_lens=torch.tensor([20], dtype=torch.int32, device=device),
        seq_lens_cpu_upper_bound=torch.tensor([20], dtype=torch.int32),
        is_prefilling_np=np.array([True]),
        has_prefill=True,
        prefill_runs_as_decode_np=np.array([True]),
        num_draft_tokens=3,
        num_draft_tokens_per_req=np.array([3], dtype=np.int32),
    )
    state.add_request(
        0,
        NewRequestData(
            req_id=batch.req_ids[0],
            prompt_token_ids=[1] * 17,
            mm_features=[],
            sampling_params=None,
            pooling_params=None,
            block_ids=([1, 2],),
            num_computed_tokens=16,
            lora_request=None,
        ),
    )
    block_table = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32, device=device)

    def build(current):
        return state.prepare_attn(
            current,
            CUDAGraphMode.NONE,
            (block_table,),
            torch.zeros((1, 4), dtype=torch.int64, device=device),
            [[group]],
            kv,
        )["gdn"]

    tail = build(batch)
    assert tail.num_spec_decodes == 1 and tail.num_prefills == 0
    block_id = tail.spec_state_indices_tensor[0, 0].item()
    assert tail.spec_write_pos_d[block_id].item() == 0
    # The admission reset must be consumed by the promoted prompt tail, not
    # delayed until the next step, which would discard accepted ring entries.
    decode = replace(
        batch,
        is_prefilling_np=np.array([False]),
        has_prefill=False,
        prefill_runs_as_decode_np=None,
    )
    assert build(decode).spec_write_pos_d[block_id].item() == 1
    assert build(decode).spec_write_pos_d[block_id].item() == 2


def test_gdn_replay_metadata_keeps_group_local_ring_cursors():
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridAttnMetadata
    from vllm.v1.worker.utils import AttentionGroup

    config = create_vllm_config(max_num_seqs=4, max_num_batched_tokens=32)
    config.cache_config.use_replayssm_spec = True
    config.cache_config.mamba_cache_mode = "none"
    config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=3
    )
    device = torch.device("cuda")
    spec = MambaSpec(
        block_size=16,
        shapes=((7, 8), (1, 8, 8)),
        dtypes=(torch.float32, torch.float32),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )
    groups = [
        AttentionGroup(
            MambaAttentionBackendEnum.GDN_ATTN.get_class(), [f"gdn{i}"], spec, i
        )
        for i in range(2)
    ]
    for group in groups:
        group.create_metadata_builders(config, device)
    kv = KVCacheConfig(
        num_blocks=1000,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(g.layer_names, spec) for g in groups],
    )
    metadata = build_attn_metadata(
        [[g] for g in groups],
        num_reqs=1,
        num_tokens=4,
        query_start_loc_gpu=torch.tensor([0, 4], device=device, dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 4], dtype=torch.int32),
        max_query_len=4,
        seq_lens=torch.tensor([20], device=device, dtype=torch.int32),
        max_seq_len=20,
        block_tables=[
            torch.tensor([[b]], device=device, dtype=torch.int32) for b in (2, 8)
        ],
        slot_mappings=torch.zeros((2, 4), device=device, dtype=torch.int64),
        kv_cache_config=kv,
        seq_lens_cpu_upper_bound=torch.tensor([20], dtype=torch.int32),
        model_specific_attn_metadata=MambaHybridAttnMetadata(
            is_prefilling=torch.tensor([False]),
            num_accepted_tokens=torch.tensor([1], device=device, dtype=torch.int32),
            num_decode_draft_tokens_cpu=torch.tensor([3], dtype=torch.int32),
            replayssm_reset_ring_cpu=torch.tensor([0], dtype=torch.int8),
        ),
    )
    for i in range(2):
        meta = metadata[f"gdn{i}"]
        block_id = meta.spec_state_indices_tensor[0, 0].item()
        assert meta.spec_write_pos_d[block_id].item() == 1
    assert (
        metadata["gdn0"].spec_write_pos_d.data_ptr()
        != metadata["gdn1"].spec_write_pos_d.data_ptr()
    )


BLOCK_SIZE = 16
DEVICE = torch.device("cpu")


def _make_spec_replay_config(num_spec: int, mode: str):
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            use_replayssm=mode == "decode",
            use_replayssm_spec=mode == "spec",
            replayssm_buffer_len=16,
            mamba_cache_mode="align",
        ),
        mamba_config=SimpleNamespace(backend=MambaBackendEnum.TRITON),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.NONE,
            max_cudagraph_capture_size=0,
        ),
        speculative_config=SimpleNamespace(
            num_speculative_tokens=num_spec, parallel_drafting=False
        ),
        num_speculative_tokens=num_spec,
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )


@pytest.mark.parametrize("num_spec", [2, 3])
# "decode" (global --use-replayssm) relied on Mamba2-only scoping of the shared builder, dropped with the
# Mamba2 speculative path; this build runs ReplaySSM for GDN layers only.
@pytest.mark.parametrize("mode", ["none", "spec"])
def test_ple_spec_metadata_with_global_replayssm(
    num_spec: int, mode: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("vllm.utils.torch_utils.PIN_MEMORY", False)
    config = _make_spec_replay_config(num_spec, mode)
    spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((9 + num_spec, 8),),
        dtypes=(torch.float32,),
        mamba_type=MambaAttentionBackendEnum.SHORT_CONV,
    )
    builder = PleShortConvAttentionMetadataBuilder(spec, ["ple"], config, DEVICE)
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[32, 48], query_lens=[num_spec + 1] * 2),
        BLOCK_SIZE,
        DEVICE,
    )
    accepted = torch.tensor([1, num_spec], dtype=torch.int32)
    metadata = builder.build(
        0,
        common,
        num_accepted_tokens=accepted,
        num_decode_draft_tokens_cpu=torch.tensor([num_spec] * 2),
    )
    assert metadata.num_spec_decodes == 2
    assert metadata.spec_query_start_loc is not None
    assert metadata.spec_query_start_loc.tolist() == [
        0,
        num_spec + 1,
        2 * (num_spec + 1),
    ]
    assert torch.equal(metadata.num_accepted_tokens, accepted)
    assert torch.equal(
        metadata.spec_state_indices_tensor, common.block_table_tensor[[0, 1], [1, 2]]
    )
    assert not builder.use_replayssm
    assert config.cache_config.use_replayssm_spec == (mode == "spec")


@pytest.mark.parametrize("mamba_cache_mode", ["none", "align"])
def test_triton_replayssm_planning_buffers(mamba_cache_mode):
    builder = _create_replayssm_builder(16, mamba_cache_mode)
    case = ReplaySSMBuildCase(
        seq_lens=[106, 113, 140],
        query_lens=[1, 1, 34],
        is_prefilling=[False, False, True],
    )
    first = _build(builder, case)
    second = _build(builder, case)

    assert first.num_decodes == 2
    for meta in (first, second):
        assert meta.write_pos_d is not None and meta.is_flush_d is not None
        assert meta.write_pos_d.shape == (2,) and meta.is_flush_d.shape == (2,)
        assert meta.bc_pre_scratch is not None
    # Persistent buffers: the planning kernel writes them in place.
    assert second.write_pos_d.data_ptr() == first.write_pos_d.data_ptr()
    assert second.is_flush_d.data_ptr() == first.is_flush_d.data_ptr()

    if mamba_cache_mode == "align":
        assert first.replayssm_align_block_size == BLOCK_SIZE
        assert first.replayssm_seq_lens_d is not None
        assert first.replayssm_seq_lens_d.tolist() == case.seq_lens
    else:
        assert first.replayssm_align_block_size == 0
        assert first.replayssm_seq_lens_d is None


def test_spec_decode_single_token_chunk_synthesizes_acceptance_metadata():
    builder = _create_replayssm_builder(16, num_speculative_tokens=3)
    # Single-token prefill chunk with prior state, run as a decode row.
    case = ReplaySSMBuildCase(seq_lens=[100], query_lens=[1], is_prefilling=[True])

    meta = _build(builder, case)

    assert meta.query_start_loc_d is not None
    assert meta.query_start_loc_d.tolist() == [0, 1]
    assert meta.num_accepted_tokens is not None
    assert meta.num_accepted_tokens.tolist() == [1]


def test_flashinfer_replayssm_state_indices_are_stable_for_full_cudagraph():
    checkpointing_ssu = pytest.importorskip("flashinfer.mamba.checkpointing_ssu")
    if not hasattr(checkpointing_ssu, "allocate_checkpointing_ssu_scratch"):
        pytest.skip("requires FlashInfer ReplaySSM autotuning support")

    builder = _create_replayssm_builder(
        16,
        mamba_backend=MambaBackendEnum.FLASHINFER,
        num_speculative_tokens=3,
    )
    builder.compilation_config.cudagraph_mode = CUDAGraphMode.FULL

    first = _build(
        builder,
        ReplaySSMBuildCase(
            seq_lens=[106, 106],
            query_lens=[1, 1],
            is_prefilling=[False, False],
        ),
    )
    first_indices = first.replayssm_state_indices_d
    assert first_indices is not None
    assert first_indices.is_contiguous()
    first_ptr = first_indices.data_ptr()

    second = _build(
        builder,
        ReplaySSMBuildCase(
            seq_lens=[122, 122],
            query_lens=[1, 1],
            is_prefilling=[False, False],
        ),
    )
    second_indices = second.replayssm_state_indices_d
    assert second_indices is not None
    assert second_indices.data_ptr() == first_ptr
    assert torch.equal(second_indices, second.state_indices_tensor_d[:, 0])
