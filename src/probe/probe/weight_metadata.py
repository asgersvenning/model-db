"""Read PyTorch pickle metadata using inert stand-ins, never imported classes.

Only data.pkl is read: tensor storage payloads are not allocated or decoded.
No global referenced by the pickle is imported or executed.
"""
from dataclasses import dataclass
import io
import math
import pickle
import pickletools
import zipfile


class Unsupported(ValueError):
    pass


class Object:
    """An inert record for any serialized class or reducer we do not interpret."""
    global_name = ""

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj.args, obj.state, obj.items, obj.sequence = args, None, {}, []
        return obj

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.state = state

    def __setitem__(self, key, value):
        self.items[key] = value

    def append(self, value):
        self.sequence.append(value)

    def extend(self, values):
        self.sequence.extend(values)


@dataclass(eq=False)
class Storage:
    dtype: str
    size: int


@dataclass(eq=False)
class Tensor:
    shape: tuple
    dtype: str


DTYPES = {"Float": "float32", "Double": "float64", "Half": "float16",
          "BFloat16": "bfloat16", "Long": "int64", "Int": "int32",
          "Short": "int16", "Char": "int8", "Byte": "uint8", "Bool": "bool",
          "ComplexFloat": "complex64", "ComplexDouble": "complex128",
          "QInt8": "qint8", "QUInt8": "quint8", "QInt32": "qint32"}


class MetadataReader(pickle.Unpickler):
    def __init__(self, stream):
        super().__init__(stream)
        self.classes = {}

    def persistent_load(self, pid):
        if not isinstance(pid, tuple) or len(pid) < 5 or pid[0] != "storage":
            raise Unsupported("Unknown persistent storage reference")
        name = getattr(pid[1], "global_name", "")
        kind = name.removeprefix("torch.").removesuffix("Storage")
        dtype = DTYPES.get(kind, "unknown")
        if not isinstance(pid[4], int) or pid[4] < 0:
            raise Unsupported("Invalid storage size")
        return Storage(dtype, pid[4])

    def tensor(self, storage, offset, size, stride, *rest):
        if not isinstance(storage, Storage) or not isinstance(size, (tuple, list)):
            raise Unsupported("Unknown tensor reconstruction layout")
        if not all(type(n) is int and n >= 0 for n in size):
            raise Unsupported("Invalid tensor dimensions")
        return Tensor(tuple(size), "torch." + storage.dtype)

    def parameter(self, tensor, *rest):
        if not isinstance(tensor, Tensor):
            raise Unsupported("Unknown parameter reconstruction layout")
        return tensor

    def find_class(self, module, name):
        full = f"{module}.{name}"
        if module == "torch._utils":
            if name in ("_rebuild_tensor", "_rebuild_tensor_v2", "_rebuild_qtensor"):
                return self.tensor
            if name in ("_rebuild_parameter", "_rebuild_parameter_with_state"):
                return self.parameter
        if full not in self.classes:
            # NEWOBJ needs a type; REDUCE may call it. Both create only this
            # inert local Object, with no methods from the checkpoint's class.
            self.classes[full] = type("SerializedObject", (Object,), {"global_name": full})
        return self.classes[full]


EXCLUDE = {"ema", "model_ema", "ema_state_dict", "optimizer", "optimizers",
           "optimizer_state_dict", "optimizer_states", "scheduler", "lr_scheduler",
           "lr_schedulers", "scaler", "grad_scaler", "_buffers"}


def mapping(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, Object) and value.global_name == "collections.OrderedDict":
        return value.items
    return None


def summarize(value):
    seen, tensors, excluded = set(), [], set()
    stack = [("", value)]
    while stack:
        path, obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, Tensor):
            tensors.append(obj)
            continue
        entries = mapping(obj)
        if entries is not None:
            for key, child in entries.items():
                label = str(key)
                child_path = f"{path}.{label}" if path else label
                if label.lower() in EXCLUDE:
                    excluded.add(child_path)
                else:
                    stack.append((child_path, child))
        elif isinstance(obj, Object):
            if obj.global_name.startswith("torch.") and (
                    "rebuild" in obj.global_name or obj.global_name.endswith("._load_from_bytes")):
                raise Unsupported("Unrecognized tensor reducer: " + obj.global_name)
            state = mapping(obj.state)
            if state is not None and "_parameters" in state and "_modules" in state:
                # nn.Module records explicitly separate registered parameters
                # and child modules from buffers, caches and other attributes.
                stack.extend([(path + "._parameters", state["_parameters"]),
                              (path + "._modules", state["_modules"])])
                if "_buffers" in state:
                    excluded.add(path + "._buffers")
            else:
                stack.extend([(path, obj.state), (path, obj.items),
                              (path, obj.sequence), (path, obj.args)])
        elif isinstance(obj, (tuple, list, set, frozenset)):
            stack.extend((f"{path}[{i}]", child) for i, child in enumerate(obj))
    dtypes = {}
    for tensor in tensors:
        count = dtypes.setdefault(tensor.dtype, {"tensors": 0, "elements": 0})
        count["tensors"] += 1
        count["elements"] += math.prod(tensor.shape)
    return {"tensor_count": len(tensors),
            "tensor_elements": sum(d["elements"] for d in dtypes.values()),
            "by_dtype": dtypes, "excluded_paths": sorted(excluded)}


def inspect_archive(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if any(n.endswith("/constants.pkl") or "/code/" in n for n in names):
            raise Unsupported("TorchScript archive")
        entries = [n for n in names if n == "data.pkl" or n.endswith("/data.pkl")]
        if len(entries) != 1:
            raise Unsupported("Expected one PyTorch data.pkl")
        if archive.getinfo(entries[0]).file_size > 64 * 1024**2:
            raise Unsupported("Pickle metadata exceeds 64 MiB")
        data = archive.read(entries[0])
        # Extension opcodes can use pickle's process-global extension cache,
        # bypassing find_class. Do not resolve them or external buffers.
        if any(op.name in ("EXT1", "EXT2", "EXT4", "NEXT_BUFFER")
               for op, _, _ in pickletools.genops(data)):
            raise Unsupported("Pickle extension or external-buffer opcode")
        reader = MetadataReader(io.BytesIO(data))
        checkpoint = reader.load()
    root = mapping(checkpoint)
    candidates = [(key, root[key]) for key in ("state_dict", "model_state_dict", "model")
                  if root is not None and key in root and root[key] is not None]
    if not candidates:
        candidates = [("root", checkpoint)]
    results = []
    for name, obj in candidates:
        summary = summarize(obj)
        if summary["tensor_count"]:
            results.append(dict(summary, container=name))
    return {"status": "ok" if results else "unresolved",
            "method": "pickle_tensor_metadata", "candidates": results,
            "excluded_branches": sorted(k for k in (root or {}) if str(k).lower() in EXCLUDE),
            **({} if results else {"reason": "No model tensors identified"})}
