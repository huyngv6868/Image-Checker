# StockLint

Local app that checks images against Adobe Stock quality & technical requirements before upload.

## Project layout

```
app/
  main.py          FastAPI server (port 8000)
  checker.py       orchestrator — runs all checks on one image
  checks/
    technical.py   PIL + OpenCV checks (no API): resolution, exposure, noise, sharpness, artifacts
    vision.py      LOCAL VLM checks (Qwen2.5-VL via MLX, on-device): text, anatomy, AI artifacts, quality
  static/
    index.html     single-file web UI

plugins/           Claude Code plugins (caveman, ruflo, understand-anything) — dev tools only
runtime/           cloakbrowser, gh-cli — not part of the app
```

## Running

```bash
source .venv/bin/activate
# No API key needed — AI vision runs locally. First run downloads the model (~8.5GB).
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
- Vision checks: **LOCAL** vision-language model run via `mlx-vlm` on Apple Silicon.
  No API, no Ollama, fully offline & free. Reads garbled text / anatomy (hands, faces,
  eyes, teeth) / AI artifacts / commercial quality.
  - **Default model = `Qwen2.5-VL-3B-Instruct-4bit` (~2GB), MAX_DIM=1024** — sized to be
    SAFE on a 24GB Mac that's also running an IDE/browser. ⚠️ The 8-bit 7B model (~9GB
    resident) at 1960px exhausted 24GB RAM → full swap → machine froze. Don't raise the
    default unless there's spare RAM. MLX memory is hard-capped (STOCKLINT_VLM_MEM_LIMIT_GB).
  - Model loads once (lazy singleton); GPU inference is serialized (one Metal device);
    `mx.clear_cache()` runs after each image so memory doesn't grow image-to-image.
  - For bulk, run technical-only first (skip_vision), then vision on flagged images.
    Override via STOCKLINT_VLM_MODEL / _MAX_DIM / _MEM_LIMIT_GB.
  - **transformers is pinned to 4.49.0** — ≥4.52 forces a torch-only "fast" image
    processor that breaks mlx-vlm 0.1.15's numpy preprocessing.
  - The VLM READS text reliably but JUDGES pass/fail inconsistently, so vision.py verifies
    its transcription against the English dictionary (hybrid in `_judge_text`); inference
    uses `repetition_penalty` to avoid greedy decoding loops on OCR.
- Thresholds were calibrated against /Users/huy/Downloads/Accepted vs Not Accepted.
  Most "quality issue" rejections = garbled AI text; pixel metrics can't read text.
- Adobe Stock min resolution: 4 MP | max: 100 MP | max file size: 45 MB | format: JPEG | profile: sRGB
