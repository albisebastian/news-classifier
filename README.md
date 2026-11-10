# Headline — News Category Classifier

Upload a newspaper page (image / PDF) → headlines are detected → each is classified with probabilities.
Or type a headline and classify it directly.

## 1. Run it in VS Code (3 steps)

**Prerequisites:** Python 3.10–3.12 (3.13 also works). Images / scanned PDFs are read with Tesseract if installed, otherwise
automatically with the pip-installed `rapidocr_onnxruntime` (already in requirements.txt, no extra install). Text-based PDFs need neither.
- Windows: install from https://github.com/UB-Mannheim/tesseract/wiki and add it to PATH
- macOS: `brew install tesseract`   ·   Linux: `sudo apt install tesseract-ocr`

Open the `news-classifier` folder in VS Code (**File → Open Folder**), open the terminal (**Ctrl+`**) and run:

**macOS / Linux**
```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
**Windows (PowerShell)**
```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
Open http://localhost:8000. For the optional **Standard** (Keras) engine or retraining, install `pip install -r requirements-standard.txt` instead. (Shortcut: `./run.sh` or `run.bat` does the same; or press **F5** — a launch config is included.)

## Engines: Intelligent (default) and Standard

| | Intelligent | Standard |
|---|---|---|
| Files in `backend/models/` | `svm_model.joblib`, `semantic_reference.npz` | `news_category_classifier.keras`, `labels.json` |
| How it works | word+char TF-IDF **SVM** fused with a **semantic k-NN** over 930 labelled reference articles (sentence embeddings) | TF-IDF neural net (20-class taxonomy) |
| Classes | 17: Arts & Culture, Business, Comedy, Crime, Education, Entertainment, Environment, Health, Lifestyle, Media, Politics, Religion, Science, Sports, Technology, War, Women | 20 |
| Best at | whole stories (headline + full article body) | short headlines |

Switch engines with the toggle in the header, or set `ENGINE=standard` before starting the server.

**What the backend does with a newspaper upload** (`/api/analyze`):
1. Finds each headline (PDF text layer or Tesseract OCR) and collects the *whole story body* under it.
2. Intelligent engine reads the complete story: the SVM scores the full text; the semantic layer embeds it in windows
   (long stories are chunked and averaged) and votes against the reference articles; both are fused, then blended with the
   headline-only view (`HEADLINE_BLEND`). Bodies under 25 words fall back to the headline alone.
3. Every story gets its own category, probabilities, and an *engines agree / mixed signal* flag.
4. Results are returned grouped by category (and by page); the UI can filter by category.
A single typed headline is simply classified, with no story analysis.

**Semantic layer.** `semantic_reference.npz` holds 384-dim embeddings; the matching embedding model is found automatically on
first start (candidates are tried and each is *verified* against the stored vectors, the match is remembered in
`models/embedder.txt`). This needs `sentence-transformers` and internet access **once** (~90 MB download). If it cannot be
verified the engine keeps running on the SVM alone and the status pill says "SVM only". Pin a model with `EMBEDDER_MODEL=...`.

**Measured notes (be aware):**
- The SVM improves with text length on the reference articles: 78 % (first 10 words) -> 85 % (20) -> 90 % (40) -> ~93 % (150+).
  These come from the reference set, which may overlap its training data, so treat them as optimistic.
- On bare HuffPost headlines (different label definitions) it scored only ~31 %, far below the Standard engine (~63 % on its own
  taxonomy). So: Intelligent for uploaded newspapers/full stories; for terse single headlines Standard may be better if you have it.
- The semantic layer could not be tested end-to-end in the build sandbox (no access to the embedding model); its fusion logic
  was tested with a stand-in embedder only.
- Security: `.joblib` files are pickles. The app loads them through `app/safe_io.py`, which only allows scikit-learn/numpy/scipy
  classes and blocks everything else. Only load model files you trust. `tools/convert_reference.py` converts an npz that stores
  texts as pickled objects into a pickle-free one (already done for the supplied file).
- `scikit-learn` is pinned to 1.6.1 because that is the version the SVM was saved with.

Tuning (env vars, see `config.py`): `SVM_TEMP`, `W_SVM`/`W_SEM`, `KNN_K`, `HEADLINE_BLEND`, `CHUNK_WORDS`, `STORY_WORDS`.

## Where the model files go

```
backend/models/
├── svm_model.joblib                 <- Intelligent engine (included)
├── semantic_reference.npz           <- Intelligent engine, semantic layer (included)
├── news_category_classifier.keras   <- Standard engine (optional, from train/)
└── labels.json                      <- Standard engine class names
```
The status pill (top right) shows **model online · N classes** when the model loads, or **demo mode** when it does not.
Your Colab model works as-is: if `labels.json` is missing and the model has 4 outputs, the notebook's order
(Technology, Business, Sports, Entertainment) is used.

## 3. Training / retraining (20 categories)

`train/train_model.py` (and its Colab twin `train/News_Category_Classifier_Colab.ipynb`) trains on **both** datasets:
- **HuffPost News Category v3** (real headlines) mapped onto the 20-class taxonomy of the synthetic CSV.
- **world_news_taxonomy_50000_synthetic.csv** - headlines only, capped and down-weighted. Its descriptions are **not**
  used because every one literally says "This example concerns <category>" (label leakage).

Measured on held-out *real* headlines: **62.8 % headline-only, 65.5 % headline+description** across 16 real classes
(20 classes in total). Strong: Sports 79 %, Food 76 %, Crime 71 %. Weak: Culture & Society 47 %, Business 51 %
(these categories overlap heavily in HuffPost). Energy, Automotive, Real Estate and Opinion exist only in the synthetic
file, so they are weaker on real-world text. Better real data for those classes is the best way to improve them.

Local retrain:
```bash
cd backend && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
python ../train/train_model.py --json data/News_Category_Dataset_v3.json.zip --csv data/world_news_taxonomy_50000_synthetic.csv
```
It writes `news_category_classifier.keras` + `labels.json` into `backend/models/` (about 10 min on CPU).
Or open the notebook in Google Colab, run all cells, upload both files, then unzip the downloaded `model_files.zip` into `backend/models/`.

## 4. Project layout
```
backend/app/   main.py (API) · classifier.py (model) · extractor.py (OCR/PDF headlines) · config.py (settings)
backend/models/ your model files        train/train_model.py  retraining script
frontend/      index.html · style.css · app.js   (no build step)
```

## Tuning headline detection
Lines larger than `HEADLINE_RATIO` (default 1.35) × the body text size are headlines. Lower it if headlines are missed,
raise it if body text is picked up. Set env vars, e.g. `HEADLINE_RATIO=1.2 uvicorn app.main:app`.

## Colour semantics
High ≥ 60 % (green) · Neutral 30–60 % (amber) · Low < 30 % (coral).

## Deploy (Hugging Face Spaces, free, Docker)

The repo ships a `Dockerfile` (CPU torch + OCR, Intelligent engine only, no TensorFlow). Port 7860.
1. Create a Space at https://huggingface.co/new-space -> SDK **Docker** (blank template).
2. Clone the Space, copy this project into it, and replace its `README.md` with `deploy/hf-space-README.md`
   (the YAML header tells Spaces to use Docker on port 7860). Then `git push` to the Space.
3. First boot downloads the embedding model (cached afterwards); the status pill shows "SVM + semantic" when ready.

Any container host works the same way: `docker build -t headline-ai . && docker run -p 7860:7860 headline-ai`.
The 166 MB Keras model is not in the repo; add `news_category_classifier.keras` to `backend/models/` (and use
`requirements-standard.txt`) if you want the Standard engine.

## Deploy on Vercel (free tier, slim version)

`vercel.json`, `api/index.py` and the root `requirements.txt` configure a slim deployment. What runs there:

| Works | Not available (too large for a serverless function) |
|---|---|
| Typed headlines, Intelligent engine (SVM) | Semantic layer (needs PyTorch) |
| PDFs with a selectable text layer | OCR for images / scanned PDFs |
| Per-story analysis, grouping by category | Keras "Standard" engine (TensorFlow) |

Limits: request bodies are capped at about 4.5 MB on Vercel, so uploads are limited to 4 MB (set automatically when `VERCEL` is present).
Bundle is roughly 345 MB of Python packages; Vercel's Python limit is 500 MB. If a build ever exceeds it, set the
environment variable `VERCEL_ANALYZE_BUILD_OUTPUT=1` and redeploy to see which packages are largest.

Steps: vercel.com -> **Add New... -> Project** -> import `albisebastian/news-classifier` -> leave the defaults -> **Deploy**.
Every push to `main` redeploys. For the full app (OCR for photos/scans + semantic layer) use the Docker route above.
