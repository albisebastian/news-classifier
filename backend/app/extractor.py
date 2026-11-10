"""Pull headlines out of images and PDFs.

Strategy: collect text lines with a size (font size for digital PDFs, pixel height
from OCR for scans), estimate the body-text size, and treat clearly larger, short
lines as headlines. Adjacent large lines in the same block are merged.
"""
from __future__ import annotations

import io
import statistics
from dataclasses import dataclass
from typing import List

from . import config as C


@dataclass
class Line:
    text: str
    size: float
    top: float
    block: int


def _pick_headlines(lines: List[Line]) -> List[dict]:
    """Returns [{'text': headline, 'context': body text that follows it}]"""
    lines = [l for l in lines if l.text.strip()]
    if len(lines) < 2:
        return []
    # weight by characters so a few huge titles don't skew the "body" size
    sizes = [l.size for l in lines for _ in range(max(1, len(l.text) // 8))]
    body = statistics.median(sizes)
    threshold = body * C.HEADLINE_RATIO

    heads: List[List[Line]] = []
    bodies: List[List[str]] = []
    for l in sorted(lines, key=lambda x: (x.block, x.top)):
        if l.size < threshold:
            if bodies and len(" ".join(bodies[-1]).split()) < C.STORY_WORDS:
                bodies[-1].append(l.text.strip())
            continue
        prev = heads[-1][-1] if heads else None
        if prev and prev.block == l.block and abs(prev.size - l.size) / l.size < 0.32 and 0 < l.top - prev.top < l.size * 2.6:
            heads[-1].append(l)
        else:
            heads.append([l])
            bodies.append([])

    out = []
    for group, body in zip(heads, bodies):
        text = " ".join(l.text.strip() for l in group)
        text = " ".join(text.split()).replace("- ", "") if text.endswith("-") else " ".join(text.split())
        if C.MIN_WORDS <= len(text.split()) <= C.MAX_WORDS and sum(c.isalpha() for c in text) > len(text) * 0.6:
            ctx = " ".join(" ".join(body).split()[: C.STORY_WORDS])
            out.append({"text": text, "context": ctx})
    return out


# ------------------------------------------------------------------- images
def _tesseract_lines(image) -> List[Line]:
    import os
    import shutil

    import pytesseract
    from pytesseract import Output

    # Windows installs are often not on PATH: honour TESSERACT_CMD, then the default install folders
    if not shutil.which("tesseract"):
        for cand in (os.getenv("TESSERACT_CMD"),
                     r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                     r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
            if cand and os.path.exists(cand):
                pytesseract.pytesseract.tesseract_cmd = cand
                break

    d =pytesseract.image_to_data(image, output_type=Output.DICT, config="--psm 3")
    groups: dict = {}
    for i, word in enumerate(d["text"]):
        if not word.strip() or float(d["conf"][i]) < 40:
            continue
        key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
        groups.setdefault(key, []).append((d["left"][i], word, d["height"][i], d["top"][i]))
    lines = []
    for (block, _, _), words in groups.items():
        words.sort()
        heights = [w[2] for w in words]
        lines.append(Line(" ".join(w[1] for w in words), statistics.median(heights), min(w[3] for w in words), block))
    return lines


def _rapid_lines(image) -> List[Line]:
    """Pure-pip OCR fallback (no system install). Lines are assigned to newspaper columns by their left edge."""
    import numpy as np
    from rapidocr_onnxruntime import RapidOCR

    global _RAPID
    if _RAPID is None:
        _RAPID = RapidOCR()
    result, _ = _RAPID(np.array(image.convert("RGB")))
    if not result:
        return []
    items = [(min(p[0] for p in box), min(p[1] for p in box), max(p[1] for p in box) - min(p[1] for p in box), txt)
             for box, txt, score in result if float(score) > 0.4 and txt.strip()]
    gap = image.width * 0.06
    col_starts: List[float] = []
    for x, *_ in sorted(items):
        if not col_starts or x - col_starts[-1] > gap:
            col_starts.append(x)
    return [Line(txt, float(h), float(y), max(i for i, c in enumerate(col_starts) if c <= x + gap))
            for x, y, h, txt in items]


_RAPID = None


def _ocr_lines(image) -> List[Line]:
    try:
        return _tesseract_lines(image)
    except Exception as exc:
        if "tesseract" not in str(exc).lower() and type(exc).__name__ != "TesseractNotFoundError":
            raise
        try:
            return _rapid_lines(image)
        except ImportError:
            raise RuntimeError("tesseract not installed and no fallback OCR: run  pip install rapidocr_onnxruntime  and restart")


def headlines_from_image(image) -> List[dict]:
    from PIL import ImageOps

    img = ImageOps.exif_transpose(image).convert("L")
    if max(img.size) < 1600:  # small scans OCR poorly
        s = 1600 / max(img.size)
        img = img.resize((int(img.width * s), int(img.height * s)))
    return _pick_headlines(_ocr_lines(img))


# --------------------------------------------------------------------- PDFs
def _pdf_text_lines(page) -> List[Line]:
    lines = []
    for b_idx, block in enumerate(page.get_text("dict")["blocks"]):
        for line in block.get("lines", []):
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = "".join(s["text"] for s in line["spans"])
            size = max(s["size"] for s in spans)
            lines.append(Line(text, size, line["bbox"][1], b_idx))
    return lines


def analyze_file(data: bytes, filename: str, content_type: str | None) -> List[dict]:
    """Returns [{'page': 1, 'headlines': ['...']}, ...]"""
    from PIL import Image

    name = (filename or "").lower()
    if name.endswith(".pdf") or content_type == "application/pdf":
        import pymupdf as fitz

        doc = fitz.open(stream=data, filetype="pdf")
        pages = []
        for i, page in enumerate(doc):
            if i >= C.MAX_PAGES:
                break
            lines = _pdf_text_lines(page)
            if sum(len(l.text) for l in lines) < 80:  # scanned page -> OCR
                pix = page.get_pixmap(dpi=C.PDF_DPI)
                heads = headlines_from_image(Image.open(io.BytesIO(pix.tobytes("png"))))
            else:
                heads = _pick_headlines(lines)
            pages.append({"page": i + 1, "headlines": heads})
        return pages

    img = Image.open(io.BytesIO(data))
    return [{"page": 1, "headlines": headlines_from_image(img)}]
