import importlib
import sys
import types
from collections import OrderedDict

import torch
from torch.serialization import (
    add_safe_globals,
    clear_safe_globals,
    get_unsafe_globals_in_checkpoint,
    safe_globals,
)


ckpt_path = "outputs/pusht/checkpoints/model_latest.pth"


def ensure_module(path: str):
    parts = path.split(".")
    for i in range(1, len(parts) + 1):
        name = ".".join(parts[:i])
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    return sys.modules[path]


def resolve_or_stub(fqname: str):
    """
    Return the Python object for a fully-qualified name.
    If it can't be imported, create a stub class/function on the fly.
    """
    # Handle some common builtins/numpy/codecs names that show up frequently
    if fqname.startswith("builtins."):
        import builtins

        obj = getattr(builtins, fqname.split(".", 1)[1], None)
        if obj is not None:
            return obj
    if fqname.startswith("numpy."):
        # e.g. numpy.core.multiarray.scalar
        try:
            modname, attr = fqname.rsplit(".", 1)
            mod = importlib.import_module(modname)
            return getattr(mod, attr)
        except Exception:
            pass
    if fqname.startswith("_codecs."):
        import _codecs as _c

        return getattr(_c, fqname.split(".", 1)[1])

    # Try normal import
    try:
        modname, attr = fqname.rsplit(".", 1)
        mod = importlib.import_module(modname)
        return getattr(mod, attr)
    except Exception:
        # Stub it
        modname, attr = fqname.rsplit(".", 1)
        mod = ensure_module(modname)
        if not hasattr(mod, attr):
            # Heuristic: create a class stub by default
            Stub = type(attr, (), {})
            setattr(mod, attr, Stub)
        return getattr(mod, attr)


# 1) Inspect the checkpoint to get all blocked globals
blocked = get_unsafe_globals_in_checkpoint(ckpt_path)
print("Blocked globals:", blocked)

# 2) Resolve or stub each and add to the allowlist
objs = [resolve_or_stub(name) for name in blocked]
clear_safe_globals()  # start clean
add_safe_globals(objs)  # persistent allowlist
with safe_globals(objs):  # also add for this context
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

# 3) Extract state_dict
state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

# 4) (Optional) strip common prefixes
state_dict = OrderedDict((k.replace("model.", "").replace("module.", ""), v) for k, v in state_dict.items())

print(f"Loaded {len(state_dict)} tensors. Example keys:", list(state_dict.keys())[:10])
