"""Technical checks — no API required.

Each check returns a dict:
  { id, name, status, value, threshold, message }
where status is "pass" | "warn" | "fail".
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageCms

# English dictionary for garbled-text detection (loaded once)
_DICT: set[str] = set()
for _dpath in ("/usr/share/dict/words", "/usr/share/dict/web2"):
    if os.path.exists(_dpath):
        try:
            with open(_dpath, encoding="latin-1") as _f:
                _DICT = {w.strip().lower() for w in _f if len(w.strip()) >= 2}
            break
        except Exception:  # noqa: BLE001
            pass
_HAS_TESSERACT = shutil.which("tesseract") is not None


# ── helpers ──────────────────────────────────────────────────────────────────

def _result(id: str, name: str, status: str, value: Any, threshold: str, message: str) -> dict:
    return dict(id=id, name=name, status=status, value=value, threshold=threshold, message=message)


def _to_cv2(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


# ── individual checks ─────────────────────────────────────────────────────────

def check_format(path: Path, img: Image.Image) -> dict:
    fmt = img.format or ""
    if fmt.upper() == "JPEG":
        return _result("format", "File Format", "pass", fmt, "JPEG", "JPEG format ✓")
    return _result("format", "File Format", "fail", fmt, "JPEG",
                   f"Adobe Stock requires JPEG. This is {fmt or 'unknown'}.")


def check_file_size(path: Path) -> dict:
    size_mb = os.path.getsize(path) / (1024 * 1024)
    val = round(size_mb, 2)
    if size_mb > 45:
        return _result("file_size", "File Size", "fail", f"{val} MB", "≤ 45 MB",
                       f"File is {val} MB — exceeds 45 MB limit.")
    if size_mb > 40:
        return _result("file_size", "File Size", "warn", f"{val} MB", "≤ 45 MB",
                       f"File is {val} MB — close to 45 MB limit.")
    return _result("file_size", "File Size", "pass", f"{val} MB", "≤ 45 MB", f"{val} MB ✓")


def check_resolution(img: Image.Image) -> dict:
    w, h = img.size
    mp = round(w * h / 1_000_000, 1)
    val = f"{w}×{h} ({mp} MP)"
    if mp < 4:
        return _result("resolution", "Resolution", "fail", val, "4–100 MP",
                       f"Only {mp} MP — Adobe Stock requires at least 4 MP.")
    if mp > 100:
        return _result("resolution", "Resolution", "fail", val, "4–100 MP",
                       f"{mp} MP exceeds 100 MP limit.")
    return _result("resolution", "Resolution", "pass", val, "4–100 MP", f"{mp} MP ✓")


def check_color_profile(img: Image.Image) -> dict:
    """Read ICC profile description properly. Only fail on clearly wide-gamut profiles."""
    icc = img.info.get("icc_profile") or b""
    if not icc:
        # No ICC profile — JPEG without one is treated as sRGB by most software. Warn only.
        return _result("color_profile", "Color Profile", "warn", "No ICC profile", "sRGB",
                       "No embedded profile. Embed sRGB to be safe (usually still accepted).")
    # Parse the real profile description
    desc = ""
    try:
        prof = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        desc = (ImageCms.getProfileName(prof) or "").strip()
    except Exception:  # noqa: BLE001
        desc = ""
    d = desc.lower()
    if "srgb" in d or not desc:
        return _result("color_profile", "Color Profile", "pass", desc or "sRGB (assumed)", "sRGB", "sRGB ✓")
    # Clearly wide-gamut profiles that Adobe Stock rejects
    WIDE = ("adobe rgb", "prophoto", "display p3", "p3", "ecirgb", "wide gamut", "rec2020", "rec. 2020")
    if any(w in d for w in WIDE):
        return _result("color_profile", "Color Profile", "fail", desc, "sRGB",
                       f"Profile is '{desc}', not sRGB. Convert to sRGB before submitting.")
    # Unknown/other profile present — warn, don't hard-fail
    return _result("color_profile", "Color Profile", "warn", desc, "sRGB",
                   f"Profile '{desc}' may not be sRGB. Convert to sRGB to be safe.")


def check_exposure(img: Image.Image) -> dict:
    gray = np.array(img.convert("L"))
    mean = float(gray.mean())
    # Compute % of pixels that are blown (>250) or crushed (<5)
    blown = float((gray > 250).mean() * 100)
    crushed = float((gray < 5).mean() * 100)
    val = f"mean={mean:.0f}, blown={blown:.1f}%, crushed={crushed:.1f}%"

    # Dark/high-contrast textures are valid, so only fail on EXTREME clipping.
    if blown > 15:
        return _result("exposure", "Exposure", "fail", val, "Not blown",
                       f"{blown:.1f}% of pixels are blown out (lost highlight detail).")
    if crushed > 50:
        return _result("exposure", "Exposure", "fail", val, "Not crushed",
                       f"{crushed:.1f}% of pixels are pure black (lost shadow detail).")
    if blown > 6:
        return _result("exposure", "Exposure", "warn", val, "Not blown",
                       f"{blown:.1f}% blown highlights — check brightest areas.")
    if crushed > 35:
        return _result("exposure", "Exposure", "warn", val, "Not crushed",
                       f"{crushed:.1f}% pure-black pixels — intentional for dark textures, otherwise check.")
    return _result("exposure", "Exposure", "pass", val, "OK", "Exposure OK ✓")


def check_noise(img: Image.Image) -> dict:
    """Estimate noise via high-frequency residual on a flat region."""
    bgr = _to_cv2(img)
    gray_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Laplacian noise estimate — blur on uint8 (some OpenCV builds reject float32 GaussianBlur)
    blur = cv2.GaussianBlur(gray_u8, (3, 3), 0).astype(np.float32)
    residual = gray_u8.astype(np.float32) - blur
    noise = float(np.std(residual))
    val = f"{noise:.2f}"

    if noise > 22:
        return _result("noise", "Noise Level", "fail", val, "< 22",
                       f"High noise/grain detected (score {noise:.1f}).")
    if noise > 15:
        return _result("noise", "Noise Level", "warn", val, "< 22",
                       f"Moderate noise (score {noise:.1f}). Check at 100%.")
    return _result("noise", "Noise Level", "pass", val, "< 22", f"Low noise ✓ (score {noise:.1f})")


def check_sharpness(img: Image.Image) -> dict:
    """Laplacian variance — measures focus/sharpness."""
    bgr = _to_cv2(img)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    val = f"{lap_var:.1f}"

    # Whole-image Laplacian is unreliable for textures/gradients (smooth ≠ blurry),
    # so this is WARN-only — never a hard fail. Only flag near-flat / heavily soft images.
    if lap_var < 6:
        return _result("sharpness", "Sharpness", "warn", val, "detail present",
                       f"Very low detail (score {lap_var:.0f}). If it should be sharp, check focus.")
    if lap_var < 25:
        return _result("sharpness", "Sharpness", "warn", val, "detail present",
                       f"Soft/low detail (score {lap_var:.0f}). Fine for smooth textures, else check 100%.")
    return _result("sharpness", "Sharpness", "pass", val, "detail present", f"Detail OK ✓ (score {lap_var:.0f})")


def check_jpeg_artifacts(path: Path, img: Image.Image) -> dict:
    """Detect JPEG block artifacts by checking DCT block boundaries."""
    bgr = _to_cv2(img)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape

    # Compute mean absolute difference at 8-pixel block boundaries vs interior
    def boundary_energy(axis: int) -> float:
        if axis == 0:  # horizontal boundaries
            rows = np.arange(8, h, 8)
            if len(rows) == 0:
                return 0.0
            bd = np.abs(gray[rows, :] - gray[rows - 1, :]).mean()
            interior = np.abs(np.diff(gray, axis=0)).mean()
        else:  # vertical boundaries
            cols = np.arange(8, w, 8)
            if len(cols) == 0:
                return 0.0
            bd = np.abs(gray[:, cols] - gray[:, cols - 1]).mean()
            interior = np.abs(np.diff(gray, axis=1)).mean()
        return float(bd / (interior + 1e-6))

    ratio = (boundary_energy(0) + boundary_energy(1)) / 2
    val = f"{ratio:.2f}"

    if ratio > 2.2:
        return _result("jpeg_artifacts", "JPEG Artifacts", "fail", val, "< 2.2",
                       f"Strong JPEG block artifacts detected (score {ratio:.2f}). Use higher quality export.")
    if ratio > 1.6:
        return _result("jpeg_artifacts", "JPEG Artifacts", "warn", val, "< 2.2",
                       f"Moderate compression artifacts (score {ratio:.2f}).")
    return _result("jpeg_artifacts", "JPEG Artifacts", "pass", val, "< 2.2",
                   f"No significant JPEG artifacts ✓ (score {ratio:.2f})")


def check_chromatic_aberration(img: Image.Image) -> dict:
    """Detect color fringing at high-contrast edges."""
    bgr = _to_cv2(img)
    # Detect edges on luminance
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 200)
    # Dilate edges slightly to sample neighborhood
    kernel = np.ones((3, 3), np.uint8)
    edge_zone = cv2.dilate(edges, kernel, iterations=2)

    # At edge zones, measure std dev of each channel separately
    b, g, r = cv2.split(bgr)
    mask = edge_zone > 0
    if mask.sum() < 100:
        return _result("chroma_aberration", "Chromatic Aberration", "pass", "N/A",
                       "Low", "Insufficient edges to measure.")

    # Channel misalignment proxy: std of (R-G) and (B-G) at edges
    rg = (r.astype(np.float32) - g.astype(np.float32))[mask]
    bg_ch = (b.astype(np.float32) - g.astype(np.float32))[mask]
    ca_score = float((np.std(rg) + np.std(bg_ch)) / 2)
    val = f"{ca_score:.2f}"
    # Colorful textures naturally have high channel variance at edges, so only flag extremes.
    if ca_score > 60:
        return _result("chroma_aberration", "Chromatic Aberration", "fail", val, "< 60",
                       f"Strong color fringing at edges (score {ca_score:.1f}).")
    if ca_score > 42:
        return _result("chroma_aberration", "Chromatic Aberration", "warn", val, "< 60",
                       f"Possible color fringing (score {ca_score:.1f}). Check edges at 200%.")
    return _result("chroma_aberration", "Chromatic Aberration", "pass", val, "< 60",
                   f"No significant CA ✓ (score {ca_score:.1f})")


def check_saturation(img: Image.Image) -> dict:
    """Saturation is a stylistic choice — WARN only, never a hard fail."""
    hsv = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = hsv[:, :, 1]  # 0-255
    mean_sat = float(sat.mean())
    max_sat = float(np.percentile(sat, 99))
    val = f"mean={mean_sat:.0f}/255, p99={max_sat:.0f}/255"

    if mean_sat > 235:
        return _result("saturation", "Saturation", "warn", val, "stylistic",
                       f"Very high average saturation ({mean_sat:.0f}/255). Confirm it's intentional.")
    if mean_sat < 8:
        return _result("saturation", "Saturation", "warn", val, "stylistic",
                       f"Almost no color ({mean_sat:.0f}/255). Fine if intentionally monochrome.")
    return _result("saturation", "Saturation", "pass", val, "stylistic",
                   f"Saturation OK ✓ (mean {mean_sat:.0f}/255)")


def check_watermark(img: Image.Image) -> dict:
    """Heuristic: look for semi-transparent uniform-color overlays in corners."""
    arr = np.array(img.convert("RGBA"))
    h, w = arr.shape[:2]
    region_size = min(h, w) // 6

    # Check 4 corners for unusually low-variance (flat color) regions
    corners = [
        arr[:region_size, :region_size],
        arr[:region_size, -region_size:],
        arr[-region_size:, :region_size],
        arr[-region_size:, -region_size:],
    ]
    suspicious = 0
    for c in corners:
        std = float(c[:, :, :3].std())
        if std < 8:  # very flat region
            suspicious += 1

    if suspicious >= 2:
        return _result("watermark", "Watermark/Overlay", "warn", f"{suspicious}/4 corners flat",
                       "No watermarks", "Possible watermark or overlay detected in corners. Adobe Stock rejects watermarked images.")
    return _result("watermark", "Watermark/Overlay", "pass", "None detected",
                   "No watermarks", "No obvious watermark detected ✓")


def check_borders(img: Image.Image) -> dict:
    """Detect solid white/black border bands at the edges (Adobe rejects frames/matting)."""
    gray = np.array(img.convert("L")).astype(np.float32)
    h, w = gray.shape

    def band_thickness(lines) -> int:
        """Count consecutive edge lines that form a flat WHITE matte/letterbox band.
        (Dark textures legitimately have near-black edges, so only white bands count.)"""
        t = 0
        for ln in lines:
            m, s = float(ln.mean()), float(ln.std())
            if s < 4 and m > 248:
                t += 1
            else:
                break
        return t

    top    = band_thickness([gray[i, :] for i in range(min(h // 3, 400))])
    bottom = band_thickness([gray[h - 1 - i, :] for i in range(min(h // 3, 400))])
    left   = band_thickness([gray[:, i] for i in range(min(w // 3, 400))])
    right  = band_thickness([gray[:, w - 1 - i] for i in range(min(w // 3, 400))])

    worst = max(top / h, bottom / h, left / w, right / w)  # fraction of dimension
    px = max(top, bottom, left, right)
    val = f"{px}px ({worst*100:.1f}%)"
    if worst > 0.015:
        return _result("borders", "Borders / Frame", "fail", val, "edge-to-edge",
                       f"Solid border/letterbox band ({px}px) at an edge. Adobe rejects borders/frames.")
    return _result("borders", "Borders / Frame", "pass", val, "edge-to-edge", "Fills frame ✓")


def check_text_legibility(img: Image.Image) -> dict:
    """OCR for garbled AI text (newspapers, signs). Local, free. Skips if tesseract missing.

    AI-rendered text is a top Adobe rejection. We CLAHE-boost contrast then OCR (psm 11,
    sparse text). Lots of text-like regions with low confidence = garbled lettering.
    Calibrated on real accepted textures (max 13 regions) → fail above 15.
    """
    if not _HAS_TESSERACT:
        return _result("text_legibility", "Text Legibility", "warn", "OCR unavailable",
                       "real words",
                       "Tesseract not installed — can't check for garbled text. `brew install tesseract`")
    work = img.convert("L")
    if max(work.size) > 2200:
        work.thumbnail((2200, 2200), Image.LANCZOS)
    arr = np.array(work)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(arr)
    buf = io.BytesIO(); Image.fromarray(clahe).save(buf, "PNG")
    try:
        proc = subprocess.run(["tesseract", "stdin", "stdout", "--psm", "11", "tsv"],
                              input=buf.getvalue(), capture_output=True, timeout=120)
        out = proc.stdout.decode("utf-8", "ignore")
    except Exception as e:  # noqa: BLE001
        return _result("text_legibility", "Text Legibility", "warn", "OCR error", "real words", str(e))

    words = []
    for line in out.splitlines()[1:]:
        cols = line.split("\t")
        if len(cols) != 12:
            continue
        try:
            conf = float(cols[10])
        except ValueError:
            continue
        t = "".join(ch for ch in cols[11] if ch.isalpha()).lower()
        if conf >= 40 and len(t) >= 3:
            words.append(t)

    n = len(words)
    real = sum(1 for w in words if w in _DICT)
    ratio = (real / n) if n else 0.0
    val = f"{n} text regions, {ratio*100:.0f}% real words"

    # Many text-like regions that are mostly NOT real words = garbled AI lettering.
    # Calibrated on real Adobe samples: clean textures peak at 15 regions, garbled
    # text starts at 17 → fail at 16.
    if n >= 16 and ratio < 0.80:
        return _result("text_legibility", "Text Legibility", "fail", val, "no garbled text",
                       f"Garbled/AI text detected ({n} regions, only {ratio*100:.0f}% real words). "
                       "Unreadable AI text is a top Adobe rejection — remove or fix it.")
    if n >= 10 and ratio < 0.70:
        return _result("text_legibility", "Text Legibility", "warn", val, "no garbled text",
                       f"Possible garbled text ({n} regions) — inspect the lettering at 100%.")
    return _result("text_legibility", "Text Legibility", "pass", val, "no garbled text",
                   "No significant garbled text ✓")


# ── run all ───────────────────────────────────────────────────────────────────

def run_all(path: Path) -> list[dict]:
    img = Image.open(path)
    img.load()  # force decode

    results = [
        check_format(path, img),
        check_file_size(path),
        check_resolution(img),
        check_color_profile(img),
        check_exposure(img),
        check_noise(img),
        check_sharpness(img),
        check_jpeg_artifacts(path, img),
        check_chromatic_aberration(img),
        check_saturation(img),
        check_borders(img),
        check_text_legibility(img),
        check_watermark(img),
    ]
    return results
