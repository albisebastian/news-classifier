"""Loads the trained Keras model and turns text into ranked category probabilities.

If the model file is missing, a small keyword-based DEMO classifier is used so the
UI can be tried end-to-end. /api/health reports which mode is active.
"""
from __future__ import annotations

import json
import math
import pickle
import re
from typing import List

import numpy as np

from . import config as C

NOTEBOOK_LABELS = ["Technology", "Business", "Sports", "Entertainment"]
DEMO_LABELS = ["Politics", "Business", "Sports", "Technology", "Entertainment", "Health", "Science", "World"]
DEMO_KEYWORDS = {
    "Politics": "election minister president senate parliament vote government party policy law bill campaign",
    "Business": "market stocks economy company profit shares bank trade inflation revenue ceo investors earnings",
    "Sports": "match league team cup coach goal player season championship tournament win score final",
    "Technology": "ai software app startup chip cyber data cloud device smartphone robot internet tech",
    "Entertainment": "film movie actor singer album festival award star series music box office celebrity",
    "Health": "hospital vaccine disease doctors patients virus health medical cancer drug treatment",
    "Science": "scientists research study space nasa climate species planet discovery physics lab",
    "World": "war border treaty united nations crisis refugees summit conflict diplomat embassy foreign",
}


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower() if C.LOWERCASE else text


class Classifier:
    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.labels: List[str] = DEMO_LABELS
        self.mode = "demo"
        self.error: str | None = None
        self.max_len = C.MAX_LEN
        self.vocab = None
        self._load()

    # ------------------------------------------------------------------ load
    def _load(self) -> None:
        if not C.MODEL_PATH.exists():
            self.error = f"Model not found at {C.MODEL_PATH} - running in demo mode."
            return
        try:
            import tensorflow as tf  # imported lazily: heavy

            self.model = tf.keras.models.load_model(C.MODEL_PATH)
            self.labels = self._load_labels()
            self.tokenizer = self._load_tokenizer()
            if not self.max_len:
                shape = getattr(self.model, "input_shape", None)
                if isinstance(shape, list):
                    shape = shape[0]
                self.max_len = int(shape[1]) if shape and len(shape) > 1 and shape[1] else 50
            self.mode = "model"
            try:
                vec = next(l for l in self.model.layers if l.__class__.__name__ == "TextVectorization")
                self.vocab = set(vec.get_vocabulary())
            except Exception:
                self.vocab = None
        except Exception as exc:  # keep the app alive, surface the reason
            self.model, self.mode = None, "demo"
            self.error = f"Could not load model ({exc}) - running in demo mode."

    def _load_labels(self) -> List[str]:
        if C.LABELS_PATH.exists():
            data = json.loads(C.LABELS_PATH.read_text())
            if isinstance(data, dict):  # {"0": "Sports", ...}
                data = [data[k] for k in sorted(data, key=lambda k: int(k))]
            return [str(x).title() for x in data]
        enc = C.MODELS_DIR / "label_encoder.pkl"
        if enc.exists():
            with open(enc, "rb") as f:
                return [str(c) for c in pickle.load(f).classes_]
        n = self.model.output_shape[-1]
        if n == len(NOTEBOOK_LABELS):  # the original Colab notebook's order
            return NOTEBOOK_LABELS
        return [f"Class {i}" for i in range(n)]

    def _load_tokenizer(self):
        if C.TOKENIZER_PATH.exists():
            if C.TOKENIZER_PATH.suffix == ".json":
                from tensorflow.keras.preprocessing.text import tokenizer_from_json

                return tokenizer_from_json(C.TOKENIZER_PATH.read_text())
            with open(C.TOKENIZER_PATH, "rb") as f:
                return pickle.load(f)
        alt = C.TOKENIZER_PATH.with_suffix(".json")
        if alt.exists():
            from tensorflow.keras.preprocessing.text import tokenizer_from_json

            return tokenizer_from_json(alt.read_text())
        return None  # model is assumed to contain its own TextVectorization layer

    # --------------------------------------------------------------- predict
    def predict_batch(self, texts: List[str]) -> List[List[dict]]:
        texts = [_clean(t) for t in texts]
        probs = self._model_probs(texts) if self.model is not None else self._demo_probs(texts)
        out = []
        for row in probs:
            order = np.argsort(row)[::-1]
            out.append([{"label": self.labels[i], "score": round(float(row[i]), 4)} for i in order])
        return out

    def _model_probs(self, texts: List[str]) -> np.ndarray:
        import tensorflow as tf

        if self.tokenizer is not None:
            from tensorflow.keras.preprocessing.sequence import pad_sequences

            seq = self.tokenizer.texts_to_sequences(texts)
            x = pad_sequences(seq, maxlen=self.max_len, padding=C.PADDING, truncating=C.TRUNCATING)
        else:
            x = tf.constant(texts)
        p = np.asarray(self.model.predict(x, verbose=0))
        if p.ndim == 1:
            p = p[None, :]
        # Normalise in case the final layer emitted logits
        if not np.allclose(p.sum(axis=1), 1.0, atol=1e-2) or (p < 0).any():
            e = np.exp(p - p.max(axis=1, keepdims=True))
            p = e / e.sum(axis=1, keepdims=True)
        return p

    def _demo_probs(self, texts: List[str]) -> np.ndarray:
        rows = []
        for t in texts:
            words = set(re.findall(r"[a-z]+", t.lower()))
            logits = np.array([len(words & set(DEMO_KEYWORDS[l].split())) * 2.2 for l in self.labels], dtype=float)
            logits += np.array([(hash((t, l)) % 100) / 400 for l in self.labels])  # tiny deterministic jitter
            e = np.exp(logits - logits.max())
            rows.append(e / e.sum())
        return np.array(rows)


classifier = Classifier()


# ---- unified engine interface (so main.py can treat Keras and the intelligent engine alike) -------------------
class StandardEngine:
    """Wraps the trained Keras model (standard version)."""
    id = "standard"
    label = "Standard"

    def __init__(self, clf: Classifier) -> None:
        self.clf = clf

    @property
    def available(self) -> bool:
        return self.clf.mode == "model"

    @property
    def labels(self):
        return self.clf.labels

    def _meta(self, base: dict, text: str) -> dict:
        v = self.clf.vocab
        if v:
            words = [w for w in re.findall(r"[a-z0-9']+", text.lower()) if len(w) >= 3]
            if words and sum(w in v for w in words) / len(words) < C.MIN_KNOWN_RATIO:
                base.update(unclear=True, reason="most words are not recognised")
        return base

    def predict_batch(self, texts):
        return [{"predictions": p[:8], "meta": self._meta({"engine": "standard", "mode": "headline"}, t)}
                for p, t in zip(self.clf.predict_batch(texts), texts)]

    def predict_stories(self, stories):
        # the Keras model was trained on "headline + short description": give it the story opening only
        texts = [f'{s["text"]}. {" ".join(s.get("context", "").split()[:C.CONTEXT_WORDS])}'.strip() for s in stories]
        return [{"predictions": p[:8], "meta": self._meta({"engine": "standard", "mode": "story"}, t)}
                for p, t in zip(self.clf.predict_batch(texts), texts)]

    def info(self):
        return {"id": self.id, "label": self.label, "available": self.available, "labels": self.labels,
                "semantic": None, "embedder": None, "notes": [self.clf.error] if self.clf.error else []}
