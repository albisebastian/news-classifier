"""The "intelligent" engine: SVM (word+char TF-IDF) fused with a semantic nearest-neighbour layer.

* SVM        - LinearSVC margins -> softmax probabilities.
* Semantic   - sentence embedding of the text vs 930 labelled reference articles; similarity-weighted k-NN vote.
               Long stories are split into windows and averaged. Loaded in the background; if the embedding model
               cannot be found/verified the engine keeps working on the SVM alone and says so.
* Stories    - for an uploaded newspaper, the whole story (headline + body) is analysed and blended with the
               headline-only view; short/garbled bodies fall back to the headline.
"""
from __future__ import annotations

import re
import threading
from typing import List, Optional

import numpy as np

from . import config as C
from .safe_io import load_joblib_safe, load_reference_npz


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _clean(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


def _make_embedder(name: str):
    try:
        from sentence_transformers import SentenceTransformer

        m = SentenceTransformer(name)
        return lambda texts: np.asarray(m.encode(list(texts), normalize_embeddings=True, batch_size=16,
                                                 show_progress_bar=False), dtype=np.float32)
    except ImportError:
        from fastembed import TextEmbedding

        m = TextEmbedding(model_name=name)

        def enc(texts):
            v = np.array(list(m.embed(list(texts))), dtype=np.float32)
            return v / np.linalg.norm(v, axis=1, keepdims=True)

        return enc


class IntelligentEngine:
    id = "intelligent"
    label = "Intelligent"

    def __init__(self) -> None:
        self.pipe = None
        self.classes: List[str] = []
        self.labels: List[str] = []
        self.available = False
        self.notes: List[str] = []
        self.semantic = "off"            # off | loading | ready
        self._embed = None
        self._ref_emb: Optional[np.ndarray] = None
        self._ref_y: Optional[np.ndarray] = None
        self.embedder_name: Optional[str] = None
        self._vocab = None
        self._load_svm()
        if self.available:
            self._start_semantic()

    # ------------------------------------------------------------------ loading
    def _load_svm(self) -> None:
        if not C.SVM_PATH.exists():
            self.notes.append(f"{C.SVM_PATH.name} not found")
            return
        try:
            self.pipe = load_joblib_safe(C.SVM_PATH)
            self.classes = [str(c) for c in self.pipe.classes_]
            self.labels = [c.title() for c in self.classes]
            self.available = True
        except Exception as exc:
            self.notes.append(f"could not load SVM model: {exc}")

    def _start_semantic(self) -> None:
        if not C.SEMANTIC_PATH.exists():
            self.notes.append("semantic layer off: semantic_reference.npz not found")
            return
        try:
            emb, labels, texts = load_reference_npz(C.SEMANTIC_PATH)
        except Exception as exc:
            self.notes.append(f"semantic layer off: {exc}")
            return
        idx = {c: i for i, c in enumerate(self.classes)}
        keep = np.array([str(l) in idx for l in labels])
        self._ref_emb = emb[keep].astype(np.float32)
        self._ref_y = np.array([idx[str(l)] for l in labels[keep]])
        self._ref_texts = [str(t) for t in texts[keep]]
        self.semantic = "loading"
        threading.Thread(target=self._find_embedder, daemon=True).start()

    def _find_embedder(self) -> None:
        pinned = C.MODELS_DIR / "embedder.txt"
        cands = [C.EMBEDDER_MODEL] if C.EMBEDDER_MODEL else (
            ([pinned.read_text().strip()] if pinned.exists() else []) + C.EMBEDDER_CANDIDATES)
        probe = self._ref_texts[:6]
        for name in dict.fromkeys(c for c in cands if c):
            try:
                enc = _make_embedder(name)
                sim = float(np.mean(np.sum(enc(probe) * self._ref_emb[:6], axis=1)))
                if sim >= C.EMBED_MATCH:
                    self._embed, self.embedder_name, self.semantic = enc, name, "ready"
                    try:
                        pinned.write_text(name)
                    except OSError:
                        pass
                    return
                self.notes.append(f"embedder {name} did not match stored embeddings (cos={sim:.2f})")
            except Exception as exc:
                self.notes.append(f"embedder {name} unavailable: {type(exc).__name__}")
                if isinstance(exc, ImportError):   # no embedding library at all: stop trying
                    break
        self.semantic = "off"
        self.notes.append("semantic layer off: no matching embedding model - running on the SVM alone. "
                          "Install `sentence-transformers` and allow internet access once, or set EMBEDDER_MODEL.")

    # ------------------------------------------------------------------ scoring
    def _svm_probs(self, texts: List[str]) -> np.ndarray:
        return _softmax(self.pipe.decision_function(texts) / C.SVM_TEMP)

    def _sem_probs(self, texts: List[str]) -> np.ndarray:
        # split long texts into windows, embed all windows in one batch, average each text's window votes
        chunks, owner = [], []
        for i, t in enumerate(texts):
            w = t.split()
            parts = [" ".join(w[j:j + C.CHUNK_WORDS]) for j in range(0, max(len(w), 1), C.CHUNK_WORDS)][: C.MAX_CHUNKS]
            chunks += parts
            owner += [i] * len(parts)
        E = self._embed(chunks)
        sims = E @ self._ref_emb.T
        k = min(C.KNN_K, sims.shape[1])
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        votes = np.zeros((len(chunks), len(self.classes)))
        for r in range(len(chunks)):
            s = sims[r, top[r]]
            w = np.exp((s - s.max()) / C.KNN_TAU)
            np.add.at(votes[r], self._ref_y[top[r]], w)
        votes /= votes.sum(axis=1, keepdims=True)
        out = np.zeros((len(texts), len(self.classes)))
        cnt = np.zeros(len(texts))
        for r, i in enumerate(owner):
            out[i] += votes[r]; cnt[i] += 1
        return out / cnt[:, None]

    def _fuse(self, texts: List[str]):
        ps = self._svm_probs(texts)
        if self.semantic == "ready":
            try:
                pm = self._sem_probs(texts)
                # The semantic layer compares against long reference articles and is the weaker signal (~61 % leave-one-out
                # vs ~93 % for the SVM), so it only gets a real say on long texts; for a short headline it barely counts.
                n = np.array([len(t.split()) for t in texts], dtype=float)
                w = (C.W_SEM * np.minimum(1.0, n / C.SEM_FULL_WORDS))[:, None]
                p = (C.W_SVM * ps + w * pm) / (C.W_SVM + w)
                return p, ps, pm
            except Exception as exc:       # never let the optional layer break a request
                self.notes.append(f"semantic scoring failed: {exc}")
        return ps, ps, None

    def known_ratio(self, text: str) -> float:
        """Share of the text's words (3+ letters) that exist in the SVM's word vocabulary."""
        if self._vocab is None:
            wv = self.pipe.named_steps["tfidf"].transformer_list[0][1]
            self._vocab, self._tok, self._pre = wv.vocabulary_, wv.build_tokenizer(), wv.build_preprocessor()
        words = [w for w in self._tok(self._pre(text)) if len(w) >= 3]
        return sum(w in self._vocab for w in words) / len(words) if words else 0.0

    # ------------------------------------------------------------------ public API
    def _rank(self, p: np.ndarray, limit: int = 8) -> List[dict]:
        order = np.argsort(p)[::-1][:limit]
        return [{"label": self.labels[i], "score": round(float(p[i]), 4)} for i in order]

    def predict_batch(self, texts: List[str]) -> List[dict]:
        """Short texts / typed headlines."""
        texts = [_clean(t) for t in texts]
        p, ps, pm = self._fuse(texts)
        out = [self._result(p[i], ps[i], None if pm is None else pm[i], "headline", 0) for i in range(len(texts))]
        for r, txt in zip(out, texts):
            self._guard(r, txt)
        return out

    def predict_stories(self, stories: List[dict]) -> List[dict]:
        """stories: [{'text': headline, 'context': body}] -> whole-story analysis."""
        heads = [_clean(s["text"]) for s in stories]
        bodies = [_clean(s.get("context", "")) for s in stories]
        full_idx = [i for i, b in enumerate(bodies) if len(b.split()) >= C.MIN_BODY_WORDS]
        ph, psh, pmh = self._fuse(heads)
        P, PS, PM = ph.copy(), psh.copy(), None if pmh is None else pmh.copy()
        if full_idx:
            pf, psf, pmf = self._fuse([f"{heads[i]}. {bodies[i]}" for i in full_idx])
            a = C.HEADLINE_BLEND
            for r, i in enumerate(full_idx):
                P[i] = (1 - a) * pf[r] + a * ph[i]
                PS[i] = (1 - a) * psf[r] + a * psh[i]
                if PM is not None and pmf is not None:
                    PM[i] = (1 - a) * pmf[r] + a * pmh[i]
        out = [self._result(P[i], PS[i], None if PM is None else PM[i],
                            "story" if i in full_idx else "headline", len(bodies[i].split()))
               for i in range(len(stories))]
        for r, h, b in zip(out, heads, bodies):
            self._guard(r, f"{h} {b}")
        return out

    def _guard(self, result: dict, text: str) -> None:
        k = self.known_ratio(text)
        if k < C.MIN_KNOWN_RATIO:
            result["meta"]["unclear"] = True
            result["meta"]["reason"] = "most words are not recognised"

    def _result(self, p, ps, pm, mode, words) -> dict:
        meta = {"engine": "intelligent", "mode": mode, "body_words": words,
                "svm": self.labels[int(np.argmax(ps))], "semantic": None, "agree": None}
        if pm is not None:
            meta["semantic"] = self.labels[int(np.argmax(pm))]
            meta["agree"] = bool(np.argmax(ps) == np.argmax(pm))
        return {"predictions": self._rank(p), "meta": meta}

    def info(self) -> dict:
        return {"id": self.id, "label": self.label, "available": self.available, "labels": self.labels,
                "semantic": self.semantic, "embedder": self.embedder_name, "notes": self.notes[-3:]}
