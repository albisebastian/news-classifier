# %% [markdown]
# # News Category Classifier — 20-class taxonomy (real HuffPost + synthetic taxonomy data)
# Works as a plain script (`python train/train_model.py --json ... --csv ...`) AND as the Colab notebook
# (the notebook is generated from this file; cells are separated by `# %%`).
#
# Data handling (important):
# * **Real data** (HuffPost News Category v3) is mapped onto the 20-class taxonomy of the synthetic file.
# * **Synthetic CSV** is templated and its descriptions literally contain the label text
#   ("This example concerns <category>") -> using them would be label leakage. We train on its
#   **headlines only**, capped and down-weighted.
# * Scores are reported on **held-out real headlines** (synthetic test scores are inflated by templates).

# %%
import argparse, csv, io, json, random, sys, zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models

ap = argparse.ArgumentParser()
ap.add_argument("--json", default="News_Category_Dataset_v3.json", help="HuffPost JSON-lines file (.json or .zip)")
ap.add_argument("--csv", default="world_news_taxonomy_50000_synthetic.csv", help="synthetic taxonomy CSV")
ap.add_argument("--out", default=None, help="output folder (default backend/models if present, else model_out)")
args, _ = ap.parse_known_args()   # parse_known_args so this also runs inside Colab/Jupyter

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
OUT_DIR = Path(args.out) if args.out else (ROOT / "backend" / "models" if (ROOT / "backend").exists() else Path("model_out"))

# ---- the 20 classes the app will predict (names come from the synthetic taxonomy file) ----
TAXONOMY = ["POLITICS & GOVERNMENT", "WORLD & INTERNATIONAL", "BUSINESS & ECONOMY", "FINANCE", "TECHNOLOGY",
            "SCIENCE", "HEALTH & MEDICINE", "ENVIRONMENT", "ENERGY", "SPORTS", "ENTERTAINMENT",
            "CULTURE & SOCIETY", "LIFESTYLE", "FOOD & AGRICULTURE", "TRAVEL & TOURISM", "CRIME & JUSTICE",
            "EDUCATION", "AUTOMOTIVE", "REAL ESTATE", "OPINION & ANALYSIS"]

# HuffPost category -> taxonomy class. ENERGY / AUTOMOTIVE / REAL ESTATE / OPINION have no real
# counterpart in HuffPost, so they are learnt from the synthetic file only (weaker on real text).
REAL_MAP = {
    "POLITICS": "POLITICS & GOVERNMENT",
    "WORLD NEWS": "WORLD & INTERNATIONAL", "THE WORLDPOST": "WORLD & INTERNATIONAL", "WORLDPOST": "WORLD & INTERNATIONAL",
    "BUSINESS": "BUSINESS & ECONOMY", "MONEY": "FINANCE", "TECH": "TECHNOLOGY", "SCIENCE": "SCIENCE",
    "WELLNESS": "HEALTH & MEDICINE", "HEALTHY LIVING": "HEALTH & MEDICINE",
    "ENVIRONMENT": "ENVIRONMENT", "GREEN": "ENVIRONMENT", "SPORTS": "SPORTS",
    "ENTERTAINMENT": "ENTERTAINMENT", "COMEDY": "ENTERTAINMENT", "ARTS": "ENTERTAINMENT",
    "ARTS & CULTURE": "ENTERTAINMENT", "CULTURE & ARTS": "ENTERTAINMENT",
    "RELIGION": "CULTURE & SOCIETY", "QUEER VOICES": "CULTURE & SOCIETY", "BLACK VOICES": "CULTURE & SOCIETY",
    "LATINO VOICES": "CULTURE & SOCIETY", "WOMEN": "CULTURE & SOCIETY", "MEDIA": "CULTURE & SOCIETY",
    "STYLE & BEAUTY": "LIFESTYLE", "STYLE": "LIFESTYLE", "HOME & LIVING": "LIFESTYLE", "PARENTING": "LIFESTYLE",
    "PARENTS": "LIFESTYLE", "WEDDINGS": "LIFESTYLE", "DIVORCE": "LIFESTYLE",
    "FOOD & DRINK": "FOOD & AGRICULTURE", "TASTE": "FOOD & AGRICULTURE",
    "TRAVEL": "TRAVEL & TOURISM", "CRIME": "CRIME & JUSTICE", "EDUCATION": "EDUCATION", "COLLEGE": "EDUCATION",
}
REAL_CAP = 4500      # max real rows per class (stops Politics swamping the rest)
SYN_CAP = 1500       # max synthetic rows per class
SYN_WEIGHT = 0.5     # synthetic rows count half as much as real ones
MAX_TOKENS, BATCH, EPOCHS, SEED = 60000, 128, 12, 42
random.seed(SEED); tf.keras.utils.set_random_seed(SEED)

# %%
def open_text(path):
    path = Path(path)
    if path.suffix == ".zip":
        z = zipfile.ZipFile(path)
        name = next(n for n in z.namelist() if n.endswith((".json", ".csv")) and not n.startswith("__MACOSX"))
        return io.TextIOWrapper(z.open(name), encoding="utf-8")
    return open(path, encoding="utf-8")

def load_real(path):
    rows = []
    with open_text(path) as f:
        for line in f:
            r = json.loads(line)
            lab = REAL_MAP.get(str(r.get("category", "")).upper().strip())
            head = (r.get("headline") or "").strip()
            if lab and len(head.split()) >= 3:
                rows.append((head, (r.get("short_description") or "").strip(), TAXONOMY.index(lab)))
    return cap(rows, REAL_CAP)

def load_syn(path):
    rows = []
    with open_text(path) as f:
        for r in csv.DictReader(f):
            lab = r["category"].upper().strip()
            if lab in TAXONOMY:
                rows.append((r["headline"].strip(), "", TAXONOMY.index(lab)))   # NO description: label leakage
    return cap(rows, SYN_CAP)

def cap(rows, limit):
    random.shuffle(rows)
    seen, out, cnt = set(), [], Counter()
    for h, d, y in rows:
        if h.lower() in seen or cnt[y] >= limit:
            continue
        seen.add(h.lower()); cnt[y] += 1; out.append((h, d, y))
    return out

real, syn = load_real(args.json), load_syn(args.csv)
print("real rows:", len(real), " synthetic rows:", len(syn))
print("real per class:", {TAXONOMY[k]: v for k, v in sorted(Counter(r[2] for r in real).items())})

# %%
# Split: real 85/15 (test = honest benchmark), synthetic 90/10. Val comes out of train.
r_tr, r_te = train_test_split(real, test_size=0.15, random_state=SEED, stratify=[r[2] for r in real])
s_tr, s_te = train_test_split(syn, test_size=0.10, random_state=SEED, stratify=[r[2] for r in syn])
r_tr, r_va = train_test_split(r_tr, test_size=0.10, random_state=SEED, stratify=[r[2] for r in r_tr])

def expand(rs, synthetic):
    X, y, w = [], [], []
    for h, d, lab in rs:
        X.append(h); y.append(lab); w.append(SYN_WEIGHT if synthetic else 1.0)
        if d:                                   # real rows also get a "headline. description" variant
            X.append(f"{h}. {d}"); y.append(lab); w.append(1.0)
    return X, y, w

a = expand(r_tr, False); b = expand(s_tr, True)
Xtr, ytr, wtr = a[0] + b[0], np.array(a[1] + b[1]), np.array(a[2] + b[2])
Xva, yva, _ = expand(r_va, False)
yva = np.array(yva)
n = len(TAXONOMY)
cw = compute_class_weight("balanced", classes=np.arange(n), y=ytr)
wtr = wtr * cw[ytr]                      # class balance x real/synthetic weight
wtr = wtr / wtr.mean()
print(f"train {len(Xtr)}  val {len(Xva)}  real-test {len(r_te)}  synthetic-test {len(s_te)}")

# %%
# TF-IDF over unigrams + bigrams -> small MLP. Raw text goes in, probabilities come out (vectorizer is inside).
vec = layers.TextVectorization(max_tokens=MAX_TOKENS, output_mode="tf_idf", ngrams=2,
                               standardize="lower_and_strip_punctuation")
vec.adapt(tf.data.Dataset.from_tensor_slices(np.array(Xtr)).batch(512))

model = models.Sequential([
    layers.Input(shape=(), dtype=tf.string),
    vec,
    layers.Dropout(0.2),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(n, activation="softmax"),
])
model.compile(optimizer=tf.keras.optimizers.AdamW(1e-3, weight_decay=1e-4),
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05), metrics=["accuracy"])

oh = lambda y: tf.one_hot(y, n)
train_ds = (tf.data.Dataset.from_tensor_slices((np.array(Xtr), oh(ytr), wtr.astype("float32")))
            .shuffle(30000, seed=SEED).batch(BATCH).prefetch(tf.data.AUTOTUNE))
val_ds = tf.data.Dataset.from_tensor_slices((np.array(Xva), oh(yva))).batch(BATCH)

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True, monitor="val_loss"),
                     tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=1, min_lr=1e-5)])

# %%
def evaluate(name, X, y, only_present=True):
    p = model.predict(tf.constant(list(X)), batch_size=256, verbose=0).argmax(1)
    y = np.array(y); acc = float((p == y).mean())
    labs = sorted(set(y)) if only_present else list(range(n))
    print(f"\n=== {name}: accuracy {acc:.3f} ===")
    print(classification_report(y, p, labels=labs, target_names=[TAXONOMY[i] for i in labs], digits=3, zero_division=0))
    return acc

acc = evaluate("REAL headlines only (what users type)", [r[0] for r in r_te], [r[2] for r in r_te])
evaluate("REAL headline + description", [f"{r[0]}. {r[1]}" if r[1] else r[0] for r in r_te], [r[2] for r in r_te])
evaluate("SYNTHETIC headlines (templated - optimistic)", [r[0] for r in s_te], [r[2] for r in s_te], False)
if acc < 0.5:
    raise SystemExit(f"Training FAILED to learn (real accuracy {acc:.2f}). Model NOT saved.")

# %%
def classify(text, k=3):
    p = model.predict(tf.constant([text]), verbose=0)[0]
    top = p.argsort()[::-1][:k]
    print(f"{text}\n   " + "  |  ".join(f"{TAXONOMY[i].title()} {p[i]*100:.1f}%" for i in top))

for t in ["Mohammed Shami announces retirement from Test cricket after 229 wickets",
          "Central bank holds interest rates as inflation cools",
          "New AI chip promises faster on-device processing",
          "Senate passes sweeping climate and energy bill",
          "Electric vehicle maker recalls thousands of cars over battery fault",
          "Housing prices climb as mortgage rates ease",
          "Hospital warns of rising flu cases this winter"]:
    classify(t)

# %%
OUT_DIR.mkdir(parents=True, exist_ok=True)
model.save(OUT_DIR / "news_category_classifier.keras")
(OUT_DIR / "labels.json").write_text(json.dumps([t.title() for t in TAXONOMY]))
print("Saved to", OUT_DIR.resolve())
try:   # Colab: bundle + download
    from google.colab import files
    import shutil
    shutil.make_archive("model_files", "zip", OUT_DIR)
    files.download("model_files.zip")
except ImportError:
    pass
