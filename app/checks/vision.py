"""Vision checks — Gemini 2.5 Flash reads the image for AI quality issues.

The big one for AI stock art is GARBLED TEXT (newspapers, signs, labels) — pixel
metrics can't read text, only a vision model can. Returns the same dict format as
technical.py: {id, name, status, value, threshold, message}.
"""
from __future__ import annotations

import io
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MAX_DIM = 2200  # downscale longest side before sending (keep text legible enough to read)

_SYSTEM = """You are a strict Adobe Stock quality reviewer for AI-generated images.
Adobe REJECTS images for these quality issues. Inspect carefully and report honestly.

Return ONLY valid JSON (no markdown):
{
  "overall": "pass" | "fail",
  "checks": [{"id": string, "name": string, "status": "pass"|"fail", "value": string, "message": string}],
  "summary": string
}

Check ALL of these (include every one in the checks array):

1. id="text_in_image" name="Text / Lettering"  ← THE MOST IMPORTANT CHECK
   STEP 1: In the "value" field, TRANSCRIBE the 3-5 most prominent words/headlines you can see
           (write exactly what the letters spell, even if nonsense). If no text at all, value="no text".
   STEP 2: Judge. If the image contains text (newspaper, headlines, signs, labels, documents, print):
     - AI-generated text is almost ALWAYS gibberish: fake words, scrambled/melted letters,
       wrong spelling, words that aren't real language (e.g. "Hautfis", "vonistawal", "Petsrl fhes",
       "Ramarels", "ntnager"). If the prominent text is NOT clean, real, correctly-spelled words
       → status="fail".
     - Only pass if the visible text reads as genuine, correctly-spelled real words.
     - A newspaper/document whose body text is an unreadable smear when it should be legible → fail.
   - If there is genuinely NO text anywhere → status="pass".
   Be ruthless here: garbled text is the #1 Adobe rejection for AI images. When in doubt → fail.

2. id="borders_frames" name="Borders / Frames"
   - Solid white or black bands/margins/letterboxing along any edge → FAIL (Adobe rejects borders, frames, matting).
   - Image must fill the frame edge-to-edge. pass if no border.

3. id="anatomy" name="Anatomy"
   - Malformed hands/fingers, distorted faces/eyes, extra limbs, wrong proportions → FAIL.
   - No people/animals OR anatomy correct → pass.

4. id="ai_artifacts" name="AI Artifacts"
   - Unnatural repeating/tiling patterns, halos around objects, melted/merged shapes,
     impossible structures, smeared details → FAIL if obvious. Otherwise pass.

5. id="overall_commercial_quality" name="Commercial Quality"
   - Looks like a believable, professional photo/texture a buyer would trust → pass.
   - Looks obviously fake, cheap, or broken → FAIL.

"overall": "fail" if ANY check is fail; otherwise "pass".
"summary": one sentence naming the main problem, or confirming it looks clean.
Be strict on text — garbled text is the #1 reason these get rejected.
"""


def _downscale_bytes(path: Path) -> tuple[bytes, str]:
    img = Image.open(path)
    img.load()
    if max(img.size) > MAX_DIM:
        img.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=88)
    return buf.getvalue(), "image/jpeg"


def run_all(path: Path) -> list[dict]:
    api_keys = [k.strip() for k in os.getenv("GEMINI_API_KEYS", "").split(",") if k.strip()]
    if not api_keys:
        return [_unavailable("No GEMINI_API_KEYS configured in .env")]

    try:
        img_bytes, mime = _downscale_bytes(path)
    except Exception as e:  # noqa: BLE001
        return [_unavailable(f"Could not read image: {e}")]

    from google import genai
    from google.genai import types

    last_err = None
    for key in api_keys:
        try:
            client = genai.Client(api_key=key)
            resp = client.models.generate_content(
                model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type=mime),
                    "Review this image for Adobe Stock quality rejection issues.",
                ],
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM,
                    response_mime_type="application/json",
                    max_output_tokens=2048,
                ),
            )
            data = _parse((resp.text or "").strip())
            if data:
                checks = data.get("checks", [])
                for c in checks:
                    c.setdefault("threshold", "Adobe Stock")
                checks.append({
                    "id": "vision_summary",
                    "name": "AI Vision Summary",
                    "status": data.get("overall", "warn"),
                    "value": str(data.get("overall", "—")).upper(),
                    "threshold": "pass",
                    "message": data.get("summary", ""),
                })
                return checks
            last_err = "Could not parse Gemini response"
        except Exception as e:  # noqa: BLE001
            last_err = str(e)

    return [_unavailable(f"Vision check failed: {last_err}")]


def _parse(text: str) -> dict | None:
    text = re.sub(r"```json\s*|\s*```", "", text, flags=re.IGNORECASE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    return None


def _unavailable(msg: str) -> dict:
    return {
        "id": "vision_unavailable", "name": "AI Vision Check", "status": "warn",
        "value": "Unavailable", "threshold": "—", "message": msg,
    }
