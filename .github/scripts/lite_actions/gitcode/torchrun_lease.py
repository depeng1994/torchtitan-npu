#!/usr/bin/env python3
"""Trusted torchrun shim for approved source recipes in a leased NPU container.

The source shell still supplies its model/optimizer arguments. Resource topology,
rendezvous and the instrumentation module are owned by the frozen control runner.
This also prevents the historical single-node localhost:0 rendezvous from escaping
formal CI's allocated master port. Explicit manual standalone is a separate path.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import sys

OPTION_NAMES = (
    "nnodes",
    "node_rank",
    "nproc_per_node",
    "master_addr",
    "master_port",
    "rdzv_backend",
    "rdzv_endpoint",
    "rdzv_id",
    "rdzv_conf",
    "local_ranks_filter",
    "role",
    "tee",
    "max_restarts",
    "log_dir",
)


def launch_args(argv, env):
    try:
        split = argv.index("-m")
    except ValueError:
        raise ValueError("trusted training entrypoint requires module launch") from None
    if split + 1 >= len(argv) or argv[split + 1] != "_lite_gitcode_train":
        raise ValueError("source recipe bypassed the trusted training wrapper")
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for name in OPTION_NAMES:
        names = ["--" + name]
        if "_" in name:
            names.append("--" + name.replace("_", "-"))
        parser.add_argument(*names)
    parser.parse_args(argv[:split])
    identity = json.loads(env["LITE_GITCODE_IDENTITY"])
    nodes, devices, rank = (int(env[key]) for key in ("NNODES", "NGPU", "NODE_RANK"))
    if nodes not in (1, 2) or devices != 8 or not 0 <= rank < nodes:
        raise ValueError("unapproved leased training topology")
    if (identity["world_size"], identity["node_rank"]) != (nodes * devices, rank):
        raise ValueError("training lease identity mismatch")
    address = env["MASTER_ADDR"]
    ipaddress.IPv4Address(address)
    port = int(env["LITE_CI_MASTER_PORT"])
    if not 1024 <= port <= 65535 or int(env["MASTER_PORT"]) != port:
        raise ValueError("training rendezvous port lease mismatch")
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes",
        str(nodes),
        "--nproc-per-node",
        str(devices),
        "--node-rank",
        str(rank),
        "--master-addr",
        address,
        "--master-port",
        str(port),
        "--max-restarts",
        "0",
        "--local-ranks-filter",
        "0",
        "--role",
        "rank",
        "--tee",
        "3",
        "-m",
        "_lite_gitcode_train",
        *argv[split + 2 :],
    ]


def main(argv=None):
    command = launch_args(sys.argv[1:] if argv is None else argv, os.environ)
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
