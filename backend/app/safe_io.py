"""Safe(ish) loading of third-party model files.

A .joblib file is a pickle: loading an untrusted one can run arbitrary code. These loaders only allow the
scikit-learn / numpy / scipy classes a normal model needs and refuse everything else (os, subprocess, eval...).
"""
import pickle
from pathlib import Path

_NUMPY_MODULES = {"numpy", "numpy._core.multiarray", "numpy.core.multiarray", "numpy._core.numeric", "numpy.core.numeric"}
_NUMPY_NAMES = {"dtype", "ndarray", "_reconstruct", "scalar", "_frombuffer", "float64", "float32", "float16",
                "int64", "int32", "int16", "int8", "uint8", "uint16", "uint32", "uint64", "bool_", "str_", "object_"}
_BUILTINS = {("builtins", n) for n in ("object", "set", "frozenset", "slice", "complex", "tuple", "list", "dict")}
_COLLECTIONS = {("collections", "OrderedDict"), ("collections", "defaultdict")}


def _allowed(module: str, name: str) -> bool:
    if module in _NUMPY_MODULES:
        return name in _NUMPY_NAMES
    if module.startswith(("sklearn.", "scipy.sparse")):
        return True
    if module == "joblib.numpy_pickle":
        return name == "NumpyArrayWrapper"
    return (module, name) in _BUILTINS or (module, name) in _COLLECTIONS


def load_joblib_safe(path):
    from joblib.numpy_pickle import NumpyUnpickler

    path = Path(path)

    class Safe(NumpyUnpickler):
        def find_class(self, module, name):
            if not _allowed(module, name):
                raise pickle.UnpicklingError(f"Blocked unsafe class {module}.{name} in {path.name}")
            return super().find_class(module, name)

    with open(path, "rb") as fh:
        try:
            u = Safe(str(path), fh, ensure_native_byte_order=True, mmap_mode=None)
        except TypeError:  # older joblib
            u = Safe(str(path), fh, mmap_mode=None)
        return u.load()


def load_reference_npz(path):
    """embeddings (n,d) float32 + labels (n,) + texts (n,) as plain (non-pickled) arrays."""
    import numpy as np

    try:
        z = np.load(path, allow_pickle=False)
        return z["embeddings"], z["labels"], z["texts"]
    except ValueError as exc:
        raise ValueError(
            f"{Path(path).name} contains pickled object arrays. Convert it once with "
            f"`python tools/convert_reference.py {path}` (it verifies the file first)."
        ) from exc
