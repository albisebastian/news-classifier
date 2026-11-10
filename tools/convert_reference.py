"""Convert a semantic_reference.npz that stores texts as a pickled object array into a pickle-free npz.
The object array is read with a restricted unpickler that only permits numpy array reconstruction
(anything else, e.g. os.system, raises), so the file cannot execute code.

    python tools/convert_reference.py backend/models/semantic_reference.npz
"""
import pickle, sys, zipfile
import numpy as np
import numpy.lib.format as npf

SAFE = {("numpy", "ndarray"), ("numpy", "dtype"), ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "_reconstruct")}

class Restricted(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in SAFE:
            raise pickle.UnpicklingError(f"blocked {module}.{name}")
        return super().find_class(module, name)

def read_member(z, info):
    with z.open(info) as f:
        ver = npf.read_magic(f)
        shape, fortran, dtype = (npf.read_array_header_1_0 if ver == (1, 0) else npf.read_array_header_2_0)(f)
        if dtype.hasobject:
            return Restricted(f).load()
    with z.open(info) as f:
        return npf.read_array(f, allow_pickle=False)

if __name__ == "__main__":
    src = sys.argv[1]
    with zipfile.ZipFile(src) as z:
        d = {i.filename[:-4]: read_member(z, i) for i in z.infolist()}
    out = src.replace(".npz", ".converted.npz")
    np.savez_compressed(out, embeddings=np.asarray(d["embeddings"], dtype="float32"),
                        labels=np.asarray(d["labels"]).astype(str), texts=np.array([str(t) for t in d["texts"]]))
    print("written", out, "- rename it to semantic_reference.npz")
