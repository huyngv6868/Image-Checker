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

# ── config (override via env) ────────────────────────────────────────────────
MODEL_ID   = os.getenv("STOCKLINT_VLM_MODEL", "meta/llama-3.2-11b-vision-instruct")
MAX_DIM    = int(os.getenv("STOCKLINT_VLM_MAX_DIM", "2048"))
MAX_TOKENS = int(os.getenv("STOCKLINT_VLM_MAX_TOKENS", "700"))

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

# ── API Client Singleton ──────────────────────────────────────────────────────
_client = None

def _get_client() -> OpenAI | None:
    global _client
    if _client is not None:
        return _client
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        return None
    _client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )
    return _client

# ── run all ───────────────────────────────────────────────────────────────────

def run_all(path: Path) -> list[dict]:
    client = _get_client()
    if not client:
        return _unavailable("NVIDIA_API_KEY is not set in environment variables. Add it to .env.")

    # Prevent hitting 40 RPM rate limit (1.5s delay guarantees <40 RPM)
    time.sleep(1.5)

    try:
        b64_img = _prep_image_b64(path)
        
        response = client.chat.completions.create(
            model=MODEL_ID,
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
        return _unavailable(f"Vision inference failed via NVIDIA API: {type(e).__name__}: {e}")

    print(f"RAW VLM OUTPUT:\n{text}\n{'='*40}")
    data = _parse_json(text)
    if not data or "checks" not in data:
        return _unavailable("Vision model returned an unparseable response.")

    by_id = {c.get("id"): c for c in data.get("checks", []) if isinstance(c, dict)}
    results: list[dict] = []
    for cid, name in _CHECK_META.items():
        c = by_id.get(cid)
        if not c:
            results.append(_result(cid, name, "warn", "—", "Model did not report this check."))
            continue
        status = _norm_status(c.get("status", ""))
        value = c.get("value", "—")
        message = c.get("message", "") or data.get("summary", "")
        if cid == "text_in_image":
            status, note = _judge_text(status, str(value))
            if note:
                message = note
        results.append(_result(cid, name, status, value, message))
    return results
