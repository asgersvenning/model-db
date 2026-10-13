"""Restricted PyTorch artifact inspection; never construct or execute a model."""
from collections.abc import Mapping
import json
from pathlib import Path
import re
import sys
import zipfile

VERSION = "1"
CONTAINERS = ("state_dict", "model_state_dict", "model", "ema")


def inspect(path):
    import torch

    result = {"inspection_version": VERSION, "torch_version": torch.__version__,
              "runtime_parameter_count": None, "candidates": []}
    # torch.load may dispatch TorchScript archives to torch.jit.load even with
    # weights_only requested. Reject these before invoking the loader.
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            if any(n.endswith("/constants.pkl") or "/code/" in n
                   for n in archive.namelist()):
                return dict(result, status="unsupported", reason="TorchScript archive")
    try:
        from torch._subclasses.fake_tensor import FakeTensorMode
        # Read tensor metadata without allocating checkpoint-sized CPU storage.
        with FakeTensorMode():
            checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    except Exception as error:
        unsupported = re.search(r"Unsupported global: GLOBAL ([\w.]+)", str(error))
        if unsupported:
            return dict(result, status="unsupported",
                        reason=f"Custom serialized object is not supported: {unsupported.group(1)}")
        return dict(result, status="unresolved",
                    reason=f"Restricted loader rejected checkpoint: {type(error).__name__}: {str(error).splitlines()[0][:300]}")

    def is_state(value):
        return (isinstance(value, Mapping) and bool(value)
                and all(isinstance(k, str) and isinstance(v, torch.Tensor)
                        for k, v in value.items()))

    if is_state(checkpoint):
        candidates = [("root", checkpoint)]
    elif isinstance(checkpoint, Mapping):
        candidates = [(key, checkpoint[key]) for key in CONTAINERS
                      if key in checkpoint and is_state(checkpoint[key])]
        rejected = [key for key in CONTAINERS if key in checkpoint
                    and not is_state(checkpoint[key])]
        if rejected:
            result["unresolved_containers"] = rejected
    else:
        candidates = []
    for name, state in candidates:
        by_dtype = {}
        for tensor in state.values():
            dtype = str(tensor.dtype)
            entry = by_dtype.setdefault(dtype, {"tensors": 0, "elements": 0})
            entry["tensors"] += 1
            entry["elements"] += tensor.numel()
        result["candidates"].append({
            "container": name, "tensor_count": len(state),
            "tensor_elements": sum(v["elements"] for v in by_dtype.values()),
            "by_dtype": by_dtype,
        })
    if not candidates or result.get("unresolved_containers"):
        return dict(result, status="unresolved", reason="Unsupported or incomplete state-dictionary layout")
    return dict(result, status="ok")


if __name__ == "__main__":
    try:
        result = inspect(Path(sys.argv[1]))
    except Exception as error:
        result = {"status": "failed", "reason": f"{type(error).__name__}: {error}",
                  "runtime_parameter_count": None, "candidates": []}
    Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + "\n")
