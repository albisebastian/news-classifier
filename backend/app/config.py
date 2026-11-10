"""All tunables in one place. Override any value with an environment variable."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models"))

# ---- Files you upload (see README) -----------------------------------------
def _first_existing(*names):
    for n in names:
        if (MODELS_DIR / n).exists():
            return MODELS_DIR / n
    return MODELS_DIR / names[0]


MODEL_PATH = Path(os.getenv("MODEL_PATH") or _first_existing("news_category_classifier.keras", "news_classifier.keras"))
TOKENIZER_PATH = Path(os.getenv("TOKENIZER_PATH", MODELS_DIR / "tokenizer.pkl"))  # or tokenizer.json
LABELS_PATH = Path(os.getenv("LABELS_PATH", MODELS_DIR / "labels.json"))  # or label_encoder.pkl

# ---- Preprocessing (must match training) -----------------------------------
MAX_LEN = int(os.getenv("MAX_LEN", "0"))  # 0 = infer from the model's input shape
PADDING = os.getenv("PADDING", "post")     # 'pre' | 'post'
TRUNCATING = os.getenv("TRUNCATING", "post")
LOWERCASE = os.getenv("LOWERCASE", "1") == "1"  # set 0 if you did not lowercase in training

# ---- Headline extraction ---------------------------------------------------
HEADLINE_RATIO = float(os.getenv("HEADLINE_RATIO", "1.35"))  # headline font size / body size
MIN_WORDS = int(os.getenv("MIN_WORDS", "2"))
MAX_WORDS = int(os.getenv("MAX_WORDS", "28"))
PDF_DPI = int(os.getenv("PDF_DPI", "200"))
MAX_PAGES = int(os.getenv("MAX_PAGES", "40"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "4" if os.getenv("VERCEL") else "40"))  # Vercel request bodies are capped at ~4.5 MB
CONTEXT_WORDS = int(os.getenv("CONTEXT_WORDS", "40"))  # body words appended to a headline when classifying (0 = headline only)

# ---- "Intelligent" engine (svm_model.joblib + semantic_reference.npz) ----------------------------
DEFAULT_ENGINE = os.getenv("ENGINE", "intelligent")           # intelligent | standard
SVM_PATH = Path(os.getenv("SVM_PATH", MODELS_DIR / "svm_model.joblib"))
SEMANTIC_PATH = Path(os.getenv("SEMANTIC_PATH", MODELS_DIR / "semantic_reference.npz"))
# Sentence-embedding model that produced semantic_reference.npz. Leave empty to auto-detect: each candidate is
# downloaded, and the first one that reproduces the stored embeddings (cosine >= EMBED_MATCH) is used + remembered.
EMBEDDER_MODEL = os.getenv("EMBEDDER_MODEL", "")
EMBEDDER_CANDIDATES = [
    "sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/paraphrase-MiniLM-L6-v2",
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1", "sentence-transformers/all-MiniLM-L12-v2",
    "BAAI/bge-small-en-v1.5", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]
EMBED_MATCH = float(os.getenv("EMBED_MATCH", "0.97"))
SVM_TEMP = float(os.getenv("SVM_TEMP", "0.20"))      # softmax temperature turning SVM margins into probabilities
W_SVM = float(os.getenv("W_SVM", "0.6"))             # fusion weights (semantic layer optional)
W_SEM = float(os.getenv("W_SEM", "0.4"))
KNN_K = int(os.getenv("KNN_K", "12"))                # nearest reference articles that vote
KNN_TAU = float(os.getenv("KNN_TAU", "0.05"))        # vote sharpness (lower = nearest neighbours dominate)
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "150"))   # long stories are embedded in windows of this many words
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "5"))
HEADLINE_BLEND = float(os.getenv("HEADLINE_BLEND", "0.3"))  # weight of the headline-only view in a story's score
MIN_BODY_WORDS = int(os.getenv("MIN_BODY_WORDS", "25"))     # shorter bodies: classify the headline alone
STORY_WORDS = int(os.getenv("STORY_WORDS", "600"))          # max body words collected under each detected headline

# ---- input guard: refuse to name a category for text the model has mostly never seen (gibberish, OCR noise) ----
MIN_KNOWN_RATIO = float(os.getenv("MIN_KNOWN_RATIO", "0.34"))   # share of words that must be in the model's vocabulary
