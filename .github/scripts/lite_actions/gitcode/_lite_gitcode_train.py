"""Trusted torchrun entry wrapper, packaged separately from proposed source.

The approved model scripts select this module via TRAIN_FILE. A script that
bypasses the wrapper will lack per-rank module evidence and cannot pass trusted
verification. This is source provenance, not a sandbox against hostile insiders.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import runpy
import sys
from pathlib import Path


def main():
    source = Path(os.environ["LITE_GITCODE_SOURCE_ROOT"]).resolve()
    output = Path(os.environ["LITE_GITCODE_OUTPUT"])
    identity = json.loads(os.environ["LITE_GITCODE_IDENTITY"])
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != identity["world_size"] or not 0 <= rank < world:
        raise ValueError("training rank identity mismatch")
    model = os.environ["LITE_GITCODE_EXPECTED_MODEL"]
    if model != "torchtitan_npu.models.deepseek_v4":
        raise ValueError("unapproved training model")
    arguments = sys.argv[1:]
    values = []
    for index, argument in enumerate(arguments):
        if argument == "--module" and index + 1 < len(arguments):
            values.append(arguments[index + 1])
        elif argument.startswith("--module="):
            values.append(argument.partition("=")[2])
    if values != [model]:
        raise ValueError("actual training module differs from the frozen recipe")
    sys.path.insert(0, str(source))
    package = importlib.import_module("torchtitan_npu")
    module = importlib.import_module(model)
    spec = importlib.util.find_spec("torchtitan_npu.train")
    if spec is None or spec.origin is None:
        raise ValueError("merged-source training entrypoint missing")
    paths = {
        "package_path": str(Path(package.__file__).resolve()),
        "model_path": str(Path(module.__file__).resolve()),
        "training_path": str(Path(spec.origin).resolve()),
    }
    if any(not Path(value).is_relative_to(source) for value in paths.values()):
        raise ValueError("training package resolved outside the tested merge tree")
    evidence = {**identity, "rank": rank, **paths}
    destination = output / f"module-path-rank-{rank}.json"
    with destination.open("x") as stream:
        stream.write(json.dumps(evidence, sort_keys=True) + "\n")
    print("GITCODE_TRAIN_SOURCE " + json.dumps(evidence, sort_keys=True), flush=True)
    runpy.run_module("torchtitan_npu.train", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
