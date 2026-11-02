#!/usr/bin/env python3
"""Trusted HCCL rank supervisor; the source payload is loaded only on an NPU."""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    args = parser.parse_args(argv)
    source = Path(os.environ["LITE_GITCODE_SOURCE_ROOT"]).resolve()
    payload = args.payload.resolve()
    if (
        args.payload.is_symlink()
        or not payload.is_file()
        or not payload.is_relative_to(source)
    ):
        raise ValueError("payload must come from the staged merge source")
    identity = json.loads(os.environ["LITE_GITCODE_IDENTITY"])
    output = Path(os.environ["LITE_GITCODE_OUTPUT"])
    world = int(os.environ["WORLD_SIZE"])
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if (
        world != identity["world_size"]
        or not 0 <= rank < world
        or not 0 <= local_rank < 8
    ):
        raise ValueError("actual rank topology differs from the trusted case")
    import torch
    import torch.distributed as dist
    import torch_npu

    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl", timeout=datetime.timedelta(seconds=55))
    try:
        spec = importlib.util.spec_from_file_location(
            "_lite_gitcode_source_payload", payload
        )
        if spec is None or spec.loader is None:
            raise ValueError("payload loader unavailable")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if (
            type(getattr(module, "PAYLOAD_API_VERSION", None)) is not int
            or module.PAYLOAD_API_VERSION != 1
        ):
            raise ValueError("source payload interface version mismatch")
        tensor = torch.tensor(
            [float(rank)], device=f"npu:{local_rank}", dtype=torch.float32
        )
        module.all_reduce(tensor)
        actual = float(tensor.item())
        expected = world * (world - 1) // 2
        if actual != expected:
            raise AssertionError(f"all_reduce returned {actual}, expected {expected}")
        dist.barrier()
        if rank == 0:
            evidence = {
                **identity,
                "status": "PASS",
                "all_reduce_sum": actual,
                "payload_path": str(payload),
                "torch_version": str(torch.__version__),
                "torch_npu_version": str(torch_npu.__version__),
            }
            (output / "smoke-rank0.json").write_text(
                json.dumps(evidence, sort_keys=True) + "\n"
            )
            print(
                "GITCODE_HCCL_SUM "
                + json.dumps({"world_size": world, "sum": actual}, sort_keys=True),
                flush=True,
            )
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
