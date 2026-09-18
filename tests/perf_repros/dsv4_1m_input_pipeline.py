#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Reproduce the DeepSeek-V4 1M input-pipeline and CP-metadata latency.

This is intentionally a performance repro, not a correctness ST.  It exercises
production TorchTitan v0.3.0 data loading and DSV4 metadata code while avoiding
full-model memory/compute requirements.

The default run collects three signals:

1. 4K synchronous DataLoader next() latency (reference).
2. 1M synchronous DataLoader next() latency plus 1-worker/prefetch=8 overlap.
   The prefetch case mirrors Trainer.train_step(): fetch 8 accumulation batches
   in a burst, then keep those 8 CPU batches alive while releasing one every
   15 seconds to model the observed ~15 s/microbatch compute window.  Step 2's
   fetch burst therefore measures steady-state overlap after the cold step.
3. DSV4 metadata cost for 4K/logical-CP1 and 1M/logical-CP128 using the actual
   create_varlen_metadata_for_document(), build_compressed_varlen_metadata(),
   and build_cp_plan() implementations.  Logical CP128 needs only one device:
   build_cp_plan() already derives every rank's plan locally.

The script writes one JSON file suitable for before/after comparison across the
baseline and the optimization branch.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import shutil
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.distributed.tensor.experimental._context_parallel._load_balancer import (
    _HeadTailLoadBalancer,
)
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.common.attention import create_varlen_metadata_for_document

from torchtitan_npu.models.deepseek_v4.metadata import (
    build_compressed_varlen_metadata,
)
from torchtitan_npu.models.deepseek_v4.token_dispatcher import build_cp_plan


RATIOS = [1, 4, 128]
WINDOW_SIZE = 128


def _git(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _device_sync(device: torch.device) -> None:
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _resolve_device(spec: str) -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _device_name(device: torch.device) -> str:
    try:
        if device.type == "npu" and hasattr(torch, "npu"):
            return str(torch.npu.get_device_name(device))
        if device.type == "cuda":
            return str(torch.cuda.get_device_name(device))
    except Exception:
        pass
    return platform.processor() or "cpu"


def _shm_snapshot() -> dict[str, int] | None:
    path = Path("/dev/shm")
    if not path.exists():
        return None
    usage = shutil.disk_usage(path)
    return {
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }


def _summary_ms(values: list[float]) -> dict[str, Any]:
    return {
        "values_ms": [round(v, 3) for v in values],
        "mean_ms": round(statistics.fmean(values), 3),
        "median_ms": round(statistics.median(values), 3),
        "max_ms": round(max(values), 3),
    }


def _make_loader(
    *,
    tokenizer: HuggingFaceTokenizer,
    dataset: str,
    dataset_path: str,
    seq_len: int,
    num_workers: int,
    prefetch_factor: int | None,
) -> HuggingFaceTextDataLoader:
    config = HuggingFaceTextDataLoader.Config(
        dataset=dataset,
        dataset_path=dataset_path,
        infinite=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=False,
    )
    return HuggingFaceTextDataLoader(
        config,
        dp_world_size=1,
        dp_rank=0,
        tokenizer=tokenizer,
        seq_len=seq_len,
        local_batch_size=1,
        snapshot_every_n_steps=None,
    )


def _measure_fetch_bursts(
    *,
    name: str,
    tokenizer: HuggingFaceTokenizer,
    dataset: str,
    dataset_path: str,
    seq_len: int,
    grad_accum: int,
    num_workers: int,
    prefetch_factor: int | None,
    steps: int,
    microbatch_compute_seconds: float,
) -> dict[str, Any]:
    loader = _make_loader(
        tokenizer=tokenizer,
        dataset=dataset,
        dataset_path=dataset_path,
        seq_len=seq_len,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    t0 = time.perf_counter()
    iterator = iter(loader)
    iterator_init_ms = (time.perf_counter() - t0) * 1e3
    step_results: list[dict[str, Any]] = []

    for step in range(steps):
        held_batches: list[Any] = []
        batch_latencies: list[float] = []
        shm_before = _shm_snapshot()
        burst_start = time.perf_counter()
        for _ in range(grad_accum):
            start = time.perf_counter()
            batch = next(iterator)
            batch_latencies.append((time.perf_counter() - start) * 1e3)
            held_batches.append(batch)
        burst_ms = (time.perf_counter() - burst_start) * 1e3
        shm_after_fetch = _shm_snapshot()

        step_results.append(
            {
                "step": step + 1,
                "burst_ms": round(burst_ms, 3),
                "batch_next": _summary_ms(batch_latencies),
                "shm_before": shm_before,
                "shm_after_fetch": shm_after_fetch,
            }
        )

        # Only the inter-step gap matters for overlap.  Keep the fetched CPU
        # batches alive and release one after each simulated microbatch compute
        # window so worker shared-memory pressure resembles Trainer.train_step().
        if step + 1 < steps and microbatch_compute_seconds > 0:
            for idx in range(len(held_batches)):
                time.sleep(microbatch_compute_seconds)
                held_batches[idx] = None
        held_batches.clear()
        gc.collect()

    del iterator, loader
    gc.collect()
    return {
        "name": name,
        "seq_len": seq_len,
        "grad_accum": grad_accum,
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
        "prefetch_factor": prefetch_factor if num_workers > 0 else None,
        "steps": steps,
        "microbatch_compute_seconds_between_steps": microbatch_compute_seconds,
        "iterator_init_ms": round(iterator_init_ms, 3),
        "step_results": step_results,
    }


def _fetch_one_positions(
    *,
    tokenizer: HuggingFaceTokenizer,
    dataset: str,
    dataset_path: str,
    seq_len: int,
) -> tuple[torch.Tensor, float]:
    loader = _make_loader(
        tokenizer=tokenizer,
        dataset=dataset,
        dataset_path=dataset_path,
        seq_len=seq_len,
        num_workers=0,
        prefetch_factor=None,
    )
    iterator = iter(loader)
    start = time.perf_counter()
    input_dict, _labels = next(iterator)
    next_ms = (time.perf_counter() - start) * 1e3
    positions = input_dict["positions"]
    del iterator, loader
    gc.collect()
    return positions, next_ms


def _measure_metadata(
    *,
    tokenizer: HuggingFaceTokenizer,
    dataset: str,
    dataset_path: str,
    seq_len: int,
    cp_size: int,
    repeats: int,
    device: torch.device,
) -> dict[str, Any]:
    if seq_len % cp_size != 0:
        raise ValueError(f"seq_len={seq_len} must be divisible by cp_size={cp_size}")

    positions_cpu, dataset_next_ms = _fetch_one_positions(
        tokenizer=tokenizer,
        dataset=dataset,
        dataset_path=dataset_path,
        seq_len=seq_len,
    )

    start = time.perf_counter()
    positions = positions_cpu.to(device)
    _device_sync(device)
    h2d_ms = (time.perf_counter() - start) * 1e3

    create_varlen_ms: list[float] = []
    compressed_ms: list[float] = []
    cp_plan_ms: list[float] = []
    n_documents = None

    for _ in range(repeats):
        _device_sync(device)
        start = time.perf_counter()
        global_varlen = create_varlen_metadata_for_document(positions)
        _device_sync(device)
        create_varlen_ms.append((time.perf_counter() - start) * 1e3)
        if n_documents is None:
            n_documents = int(global_varlen.cu_seq_q.numel() - 1)

        _device_sync(device)
        start = time.perf_counter()
        compressed = build_compressed_varlen_metadata(global_varlen, RATIOS)
        _device_sync(device)
        compressed_ms.append((time.perf_counter() - start) * 1e3)
        del compressed

        load_balancer = (
            _HeadTailLoadBalancer(seq_len, cp_size, device.type)
            if cp_size > 1
            else None
        )
        _device_sync(device)
        start = time.perf_counter()
        cp_plan = build_cp_plan(
            global_varlen,
            load_balancer,
            rank=0,
            cp_size=cp_size,
            shard_len=seq_len // cp_size,
            window_size=WINDOW_SIZE,
            ratios=RATIOS,
        )
        _device_sync(device)
        cp_plan_ms.append((time.perf_counter() - start) * 1e3)
        del cp_plan, global_varlen, load_balancer
        gc.collect()

    return {
        "seq_len": seq_len,
        "logical_cp_size": cp_size,
        "shard_len": seq_len // cp_size,
        "ratios": RATIOS,
        "window_size": WINDOW_SIZE,
        "repeats": repeats,
        "dataset_next_ms": round(dataset_next_ms, 3),
        "positions_h2d_ms": round(h2d_ms, 3),
        "n_documents": n_documents,
        "create_varlen": _summary_ms(create_varlen_ms),
        "global_compressed_metadata": _summary_ms(compressed_ms),
        "build_cp_plan": _summary_ms(cp_plan_ms),
    }


def _environment(device: torch.device) -> dict[str, Any]:
    torch_npu_version = None
    try:
        import torch_npu

        torch_npu_version = getattr(torch_npu, "__version__", None)
    except ImportError:
        pass

    return {
        "timestamp_epoch_s": time.time(),
        "hostname": socket.gethostname(),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "torch_npu_version": torch_npu_version,
        "device": str(device),
        "device_name": _device_name(device),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "shm": _shm_snapshot(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-assets-path", required=True)
    parser.add_argument("--dataset", default="c4_test")
    parser.add_argument("--dataset-path", default="tests/assets/c4_test")
    parser.add_argument("--output", default="dsv4_1m_input_pipeline_repro.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=8)
    parser.add_argument(
        "--microbatch-compute-seconds",
        type=float,
        default=15.0,
        help="Inter-step compute window used only by the prefetched 1M case.",
    )
    parser.add_argument("--metadata-repeats", type=int, default=3)
    parser.add_argument(
        "--skip-dataloader",
        action="store_true",
        help="Skip DataLoader burst/overlap measurements.",
    )
    parser.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Skip DSV4 metadata measurements.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.grad_accum <= 0:
        raise ValueError("--grad-accum must be > 0")
    if args.metadata_repeats <= 0:
        raise ValueError("--metadata-repeats must be > 0")

    device = _resolve_device(args.device)
    tokenizer = HuggingFaceTokenizer(tokenizer_path=args.hf_assets_path)
    result: dict[str, Any] = {
        "environment": _environment(device),
        "config": vars(args),
    }

    if not args.skip_dataloader:
        result["dataloader"] = [
            _measure_fetch_bursts(
                name="4k_sync_reference",
                tokenizer=tokenizer,
                dataset=args.dataset,
                dataset_path=args.dataset_path,
                seq_len=4096,
                grad_accum=args.grad_accum,
                num_workers=0,
                prefetch_factor=None,
                steps=1,
                microbatch_compute_seconds=0.0,
            ),
            _measure_fetch_bursts(
                name="1m_sync_baseline",
                tokenizer=tokenizer,
                dataset=args.dataset,
                dataset_path=args.dataset_path,
                seq_len=1048576,
                grad_accum=args.grad_accum,
                num_workers=0,
                prefetch_factor=None,
                steps=1,
                microbatch_compute_seconds=0.0,
            ),
            _measure_fetch_bursts(
                name="1m_worker1_prefetch8",
                tokenizer=tokenizer,
                dataset=args.dataset,
                dataset_path=args.dataset_path,
                seq_len=1048576,
                grad_accum=args.grad_accum,
                num_workers=1,
                prefetch_factor=args.prefetch_factor,
                steps=2,
                microbatch_compute_seconds=args.microbatch_compute_seconds,
            ),
        ]

    if not args.skip_metadata:
        result["metadata"] = [
            _measure_metadata(
                tokenizer=tokenizer,
                dataset=args.dataset,
                dataset_path=args.dataset_path,
                seq_len=4096,
                cp_size=1,
                repeats=args.metadata_repeats,
                device=device,
            ),
            _measure_metadata(
                tokenizer=tokenizer,
                dataset=args.dataset,
                dataset_path=args.dataset_path,
                seq_len=1048576,
                cp_size=128,
                repeats=args.metadata_repeats,
                device=device,
            ),
        ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"\nWrote repro results to {output}")


if __name__ == "__main__":
    main()
