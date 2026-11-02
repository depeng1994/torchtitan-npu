"""Proposed-source payload for GitCode merge-result HCCL testing.

Only the leased execution-domain runner imports this module. Topology, timeout,
expected SUM, result publication and cleanup are owned by trusted control code.
"""

PAYLOAD_API_VERSION = 1


def all_reduce(tensor):
    import torch.distributed as dist

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
