"""Vision checks — Cloud vision-language model (NVIDIA API).

Uses the OpenAI Python client to talk to NVIDIA NIM APIs.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI
from PIL import Image, ImageOps

try:
    import cv2
except Exception:  # pragma: no cover - cv2 ships with the project
    cv2 = None

# ── config (override via env) ────────────────────────────────────────────────
MODEL_ID   = os.getenv("STOCKLINT_VLM_MODEL", "nvidia/nemotron-nano-12b-v2-vl")
MAX_DIM    = int(os.getenv("STOCKLINT_VLM_MAX_DIM", "2048"))
MAX_TOKENS = int(os.getenv("STOCKLINT_VLM_MAX_TOKENS", "700"))

# Fallback models queried per image.
# We maintain a pool of fallback API clients across providers.

# Deterministic small-face gate. The VLM cannot reliably eyeball whether a face is
# 0.2% or 5% of the frame, so a measured detector handles "humans too small/distant
# for features to be trustworthy". If a face is detected but occupies less than this
# fraction of the frame → force anatomy=fail. Calibrated: bad samples ≤0.20%, good
# samples ≥0.62%, so 0.5% cleanly separates them. Set to 0 to disable.
FACE_MIN_FRAC = float(os.getenv("STOCKLINT_FACE_MIN_FRAC", "0.005"))
_YUNET_PATH   = Path(__file__).resolve().parent.parent / "assets" / "face_detection_yunet_2023mar.onnx"
_FACE_DET_W   = 1024  # long side the detector runs at

_CHECK_META: dict[str, str] = {
    "text_in_image": "Text / Lettering",
    "anatomy": "Anatomy",
    "ai_artifacts": "AI Artifacts",
    "overall_commercial_quality": "Commercial Quality",
}

_SYSTEM = """You are a professional photography and illustration quality reviewer. \
Please evaluate this image for commercial stock viability. Inspect the visual details carefully \
and report your findings objectively.

Return ONLY valid JSON (no markdown, no commentary) in EXACTLY this shape:
{
  "text_in_image": {"status": "pass|fail", "value": "<short>", "message": "<why>"},
  "anatomy": {"status": "pass|fail", "value": "<short>", "message": "<why>"},
  "ai_artifacts": {"status": "pass|fail", "value": "<short>", "message": "<why>"},
  "overall_commercial_quality": {"status": "pass|fail", "value": "<short>", "message": "<why>"},
  "summary": "<one short sentence>"
}

LENGTH LIMIT (critical): every "value" ≤ 8 words, every "message" ≤ 15 words.
NEVER transcribe whole sentences or paragraphs — only a few words. Be terse.

How to judge each check:

1. text_in_image
   STEP A: Into "value", transcribe ONLY the 3-5 most prominent words exactly as the
           letters spell them (even if nonsense). Max ~8 words. No text → value="no text", status="pass".
   STEP B: Judge ONLY what the letters actually spell:
     - REAL, correctly-spelled words in a real language → status="pass".
     - Scrambled, misspelled, or illegible letters that are not real language → status="fail".
   Read what is really there: clean text passes, genuinely garbled text fails.

2. anatomy
   Examine every person and animal. Check for structural correctness in hands, fingers,
   faces, and limbs. If there are structural inconsistencies (e.g., incorrect number of
   fingers, unaligned facial features, structural irregularities) → status="fail".
   No people/animals OR anatomy is clearly correct → status="pass".
   When a face is clearly visible and close, scrutinize the eyes and teeth for melted,
   fused, or asymmetric features. Do NOT fail merely because a subject is small or distant.

3. ai_artifacts
   Check for unnatural visual blending, impossible geometry, irregular repeating
   patterns, or inconsistent structural details → status="fail" if obvious,
   otherwise "pass".

4. overall_commercial_quality
   A believable, professional image a buyer would trust → "pass".
   Lacking professional polish or containing obvious generation errors → "fail".

Evaluate honestly and specifically — flag an issue only when visually apparent."""

_USER = ("Review this image for quality requirements. Read any text present, and evaluate "
         "the structural correctness of all subjects. Return the JSON only.")


def _prep_image_b64(path: Path) -> str:
    """Downscales image to MAX_DIM and encodes it as base64 for API submission."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max(img.size) > MAX_DIM:
        img.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# ── deterministic small-face gate ──────────────────────────────────────────────
_face_det = None

def _largest_face_fraction(path: Path) -> float | None:
    """Largest detected face's bbox area as a fraction of the frame.

    0.0 = no face detected; None = detection unavailable (cv2/model missing or
    read error). YuNet DNN — accurate on small faces, few false positives.
    """
    global _face_det
    if cv2 is None or not _YUNET_PATH.exists():
        return None
    try:
        img = cv2.imread(str(path))
        if img is None:
            return None
        h, w = img.shape[:2]
        scale = _FACE_DET_W / max(h, w)
        rw, rh = max(1, int(w * scale)), max(1, int(h * scale))
        rimg = cv2.resize(img, (rw, rh))
        if _face_det is None:
            _face_det = cv2.FaceDetectorYN.create(str(_YUNET_PATH), "", (rw, rh), 0.7, 0.3, 5000)
        _face_det.setInputSize((rw, rh))
        _, faces = _face_det.detect(rimg)
        if faces is None or len(faces) == 0:
            return 0.0
        frame = rw * rh
        return max((f[2] * f[3]) / frame for f in faces)
    except Exception:
        return None


def _parse_json(text: str) -> dict | None:
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t).rstrip("`").strip()
    start = t.find("{")
    if start == -1:
        return None
    try:
        data = json.loads(t[start:])
        if isinstance(data, dict):
            checks = []
            for k, v in data.items():
                if k != "summary" and isinstance(v, dict) and "status" in v:
                    v["id"] = k
                    checks.append(v)
            return {"checks": checks, "summary": data.get("summary", "")}
    except json.JSONDecodeError:
        pass
        
    checks = []
    for m in re.finditer(r'"([a-z_]+)"\s*:\s*(\{[^{}]+\})', t):
        try:
            cid = m.group(1)
            obj = json.loads(m.group(2))
            if isinstance(obj, dict) and "status" in obj:
                obj["id"] = cid
                checks.append(obj)
        except json.JSONDecodeError:
            continue
            
    summary = ""
    sm = re.search(r'"summary"\s*:\s*"([^"]*)"', t)
    if sm:
        summary = sm.group(1)
        
    if checks or summary:
        return {"checks": checks, "summary": summary}
        
    for suffix in ("}", '"}', '", "summary": ""} }'):
        try:
            obj = json.loads(t + suffix)
            if isinstance(obj, dict):
                checks = []
                for k, v in obj.items():
                    if k != "summary" and isinstance(v, dict) and "status" in v:
                        v["id"] = k
                        checks.append(v)
                if checks:
                    return {"checks": checks, "summary": obj.get("summary", "")}
        except json.JSONDecodeError:
            pass

    return None

def _norm_status(s: str) -> str:
    s = (s or "").strip().lower()
    PASS = ("no fail", "not fail", "no issue", "no problem", "none", "pass", "ok",
            "good", "acceptable", "clean", "fine", "yes", "correct", "no defect")
    if any(p in s for p in PASS):
        return "pass"
    if "fail" in s or s in ("reject", "rejected", "bad", "no"):
        return "fail"
    return "warn"

try:
    from .technical import _DICT as _DICT  # noqa: F401
except Exception:
    _DICT = set()

def _judge_text(vlm_status: str, value: str) -> tuple[str, str]:
    words = re.findall(r"[A-Za-z]{3,}", value or "")
    if _DICT and len(words) >= 3:
        real = sum(1 for w in words if w.lower() in _DICT)
        frac = real / len(words)
        if frac < 0.5:
            return "fail", f"Transcribed text is mostly not real words ({real}/{len(words)} legible) — looks like garbled AI lettering."
        if frac >= 0.8:
            return "pass", f"Text reads as real words ({real}/{len(words)}) ✓"
    return (vlm_status or "pass"), ""

def _result(id: str, name: str, status: str, value, message: str) -> dict:
    return dict(id=id, name=name, status=status, value=value,
                threshold="Adobe Stock", message=message)

def _unavailable(msg: str) -> list[dict]:
    return [_result("vision_unavailable", "AI Vision", "warn", "unavailable", msg)]

# ── Fallback Clients ──────────────────────────────────────────────────────────
_fallback_clients = None

def _get_fallback_clients() -> list[tuple[OpenAI, str]]:
    global _fallback_clients
    if _fallback_clients is not None:
        return _fallback_clients
        
    clients = []
    
    # 1. NVIDIA NIM (Free Tier)
    # Check NVIDIA_API_KEYS (comma separated) or fallback to NVIDIA_API_KEY
    nvidia_keys_env = os.getenv("NVIDIA_API_KEYS") or os.getenv("NVIDIA_API_KEY", "")
    nvidia_keys = [k.strip() for k in nvidia_keys_env.split(",") if k.strip()]
    for key in nvidia_keys:
        client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=key)
        clients.append((client, MODEL_ID))
        
    # 2. OpenRouter
    or_keys_env = os.getenv("OPENROUTER_API_KEYS", "")
    or_keys = [k.strip() for k in or_keys_env.split(",") if k.strip()]
    or_model = os.getenv("OPENROUTER_MODEL", "google/gemini-1.5-flash")
    for key in or_keys:
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
        clients.append((client, or_model))
        
    # 3. Gemini
    gemini_keys_env = os.getenv("GEMINI_API_KEYS", "")
    gemini_keys = [k.strip() for k in gemini_keys_env.split(",") if k.strip()]
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    for key in gemini_keys:
        client = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=key)
        clients.append((client, gemini_model))
        
    _fallback_clients = clients
    return _fallback_clients

# ── run all ───────────────────────────────────────────────────────────────────

def _query_model(client: OpenAI, model: str, b64_img: str) -> dict | None:
    """Sends the image to one model; returns parsed {checks, summary} or None."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _SYSTEM + "\n\n" + _USER},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
                        }
                    ]
                }
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.0
        )
        text = response.choices[0].message.content
    except Exception as e:
        print(f"Vision model {model} on {client.base_url} failed: {type(e).__name__}: {e}")
        return None

    print(f"RAW VLM OUTPUT [{model}]:\n{text}\n{'='*40}")
    data = _parse_json(text)
    if not data or "checks" not in data:
        return None
    return data


def run_all(path: Path) -> list[dict]:
    clients = _get_fallback_clients()
    if not clients:
        return _unavailable("No API keys found for NVIDIA, OpenRouter, or Gemini. Add them to .env.")

    try:
        b64_img = _prep_image_b64(path)
    except Exception as e:
        return _unavailable(f"Could not prepare image: {type(e).__name__}: {e}")

    # Query using fallback chain. First successful response wins.
    datas: list[dict] = []
    for client, model in clients:
        time.sleep(1.5)  # stay under rate limits when falling back
        d = _query_model(client, model, b64_img)
        if d:
            datas.append(d)
            break

    if not datas:
        return _unavailable("All fallback vision providers failed or returned unparseable responses.")

    # Deterministic small-face gate (overrides the anatomy verdict).
    face_frac = _largest_face_fraction(path) if FACE_MIN_FRAC > 0 else None
    gate_fail = face_frac is not None and 0.0 < face_frac < FACE_MIN_FRAC

    summary = next((d.get("summary", "") for d in datas if d.get("summary")), "")
    results: list[dict] = []
    for cid, name in _CHECK_META.items():
        # collect this check from every model that reported it
        votes: list[tuple[str, str, str]] = []  # (status, value, message)
        for d in datas:
            c = next((x for x in d.get("checks", [])
                      if isinstance(x, dict) and x.get("id") == cid), None)
            if not c:
                continue
            status = _norm_status(c.get("status", ""))
            value = c.get("value", "—")
            message = c.get("message", "") or summary
            if cid == "text_in_image":
                status, note = _judge_text(status, str(value))
                if note:
                    message = note
            votes.append((status, value, message))

        # anatomy: deterministic gate overrides the VLM when faces are too small
        if cid == "anatomy" and gate_fail:
            results.append(_result(cid, name, "fail",
                f"faces too small ({face_frac*100:.2f}% of frame)",
                f"Human face(s) occupy only {face_frac*100:.2f}% of the image "
                f"(< {FACE_MIN_FRAC*100:.1f}% threshold) — subjects too small/distant, "
                f"facial features cannot be verified and are likely distorted."))
            continue

        if not votes:
            results.append(_result(cid, name, "warn", "—", "No model reported this check."))
            continue

        # union rule: any fail -> fail; report the failing model's value/message
        fail = next((v for v in votes if v[0] == "fail"), None)
        if fail:
            results.append(_result(cid, name, "fail", fail[1], fail[2]))
        else:
            status, value, message = votes[0]
            results.append(_result(cid, name, status, value, message))
    return results
