# StockLint

Local app that checks images against Adobe Stock quality & technical requirements before upload.

## Project layout

```
app/
  main.py          FastAPI server (port 8000)
  checker.py       orchestrator — runs all checks on one image
  checks/
    technical.py   PIL + OpenCV checks (no API): resolution, exposure, noise, sharpness, artifacts
    vision.py      Gemini vision checks (API): AI artifacts, anatomy, text, patterns
  static/
    index.html     single-file web UI

plugins/           Claude Code plugins (caveman, ruflo, understand-anything) — dev tools only
runtime/           cloakbrowser, gh-cli — not part of the app
```

## Running

```bash
source .venv/bin/activate
cp .env.example .env   # add GEMINI_API_KEYS
uvicorn app.main:app --reload
# open http://127.0.0.1:8000
```

## Key facts

- Technical checks: PIL/numpy/OpenCV — no API, instant. Includes:
  resolution, file size, format, sRGB profile (ICC-parsed), exposure, noise,
  sharpness (warn-only), JPEG artifacts, chromatic aberration, saturation (warn-only),
  white-border/matte detection, and **garbled-text OCR** (tesseract + dict check).
- Garbled-text check needs Tesseract: `brew install tesseract` (called via stdin, no temp files).
  Calibrated on real Adobe accept/reject samples: 20/20 accepted pass (no false positives).
- Vision checks: Gemini vision — reads garbled text/anatomy/borders. BUT free tier is
  ~20 req/day per model → impractical for bulk; needs paid tier or a higher-limit model.
- Thresholds were calibrated against /Users/huy/Downloads/Accepted vs Not Accepted.
  Most "quality issue" rejections = garbled AI text; pixel metrics can't read text.
- Adobe Stock min resolution: 4 MP | max: 100 MP | max file size: 45 MB | format: JPEG | profile: sRGB
