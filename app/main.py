"""StockLint — FastAPI server."""
from __future__ import annotations

import asyncio
import csv
import subprocess
import io
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from .checker import run  # noqa: E402

APP_DIR    = Path(__file__).resolve().parent
UPLOAD_DIR = Path(tempfile.gettempdir()) / "stocklint_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
JPEG_EXTS  = {".jpg", ".jpeg", ".JPG", ".JPEG"}

app = FastAPI(title="StockLint")


@app.get("/", response_class=HTMLResponse)
def index():
    return (APP_DIR / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/status")
def status():
    keys = [k.strip() for k in os.getenv("GEMINI_API_KEYS", "").split(",") if k.strip()]
    return {"keys": len(keys), "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash")}


# ── Native folder picker (macOS AppleScript) ─────────────────────────────────
@app.get("/api/pick-folder")
def pick_folder():
    """Open native macOS folder picker dialog, return selected path."""
    try:
        result = subprocess.run(
            ["osascript", "-e",
             'POSIX path of (choose folder with prompt "Select your image folder")'],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            # User cancelled
            return {"cancelled": True, "path": ""}
        path = result.stdout.strip().rstrip("/")
        return {"cancelled": False, "path": path}
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Scan folder ───────────────────────────────────────────────────────────────
@app.post("/api/scan-folder")
def scan_folder(body: dict):
    folder = Path(body.get("path", "")).expanduser()
    if not folder.is_dir():
        return JSONResponse({"error": f"Folder not found: {folder}"}, status_code=400)
    files = sorted(p for p in folder.rglob("*") if p.suffix in JPEG_EXTS and p.is_file())
    return {"folder": str(folder), "count": len(files)}


# ── Check folder — SSE stream ─────────────────────────────────────────────────
@app.get("/api/check-folder")
async def check_folder(path: str, skip_vision: str = "false"):
    folder = Path(path).expanduser()
    if not folder.is_dir():
        async def err():
            yield f'data: {json.dumps({"type":"error","message":"Folder not found"})}\n\n'
        return StreamingResponse(err(), media_type="text/event-stream")

    files = sorted(p for p in folder.rglob("*") if p.suffix in JPEG_EXTS and p.is_file())
    skip  = skip_vision.lower() in ("true", "1", "yes")

    async def generate():
        yield f'data: {json.dumps({"type":"start","total":len(files)})}\n\n'
        for i, img_path in enumerate(files):
            try:
                report = await asyncio.to_thread(run, img_path, skip_vision=skip)
                report["original_path"] = str(img_path)
                report["file"]  = img_path.name
                report["index"] = i
                report["type"]  = "result"
            except Exception as e:  # noqa: BLE001
                report = {
                    "type": "result", "original_path": str(img_path),
                    "file": img_path.name, "index": i,
                    "overall": "fail", "fail_count": 1, "warn_count": 0,
                    "checks": [], "error": str(e),
                }
            yield f"data: {json.dumps(report)}\n\n"
            await asyncio.sleep(0)
        yield f'data: {json.dumps({"type":"done","total":len(files)})}\n\n'

    return StreamingResponse(
        generate(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Check single uploaded file ────────────────────────────────────────────────
@app.post("/api/check")
async def check_image(file: UploadFile = File(...), skip_vision: str = Form("false")):
    suffix = Path(file.filename or "img.jpg").suffix.lower() or ".jpg"
    tmp = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    try:
        with open(tmp, "wb") as f:
            shutil.copyfileobj(file.file, f)
        skip   = skip_vision.lower() in ("true", "1", "yes")
        report = await asyncio.to_thread(run, tmp, skip_vision=skip)
        report["original_name"] = file.filename
        report["original_path"] = file.filename
        return JSONResponse(report)
    finally:
        tmp.unlink(missing_ok=True)


# ── Delete failed images ──────────────────────────────────────────────────────
@app.post("/api/delete-failed")
def delete_failed(body: dict):
    paths = body.get("paths", [])
    deleted, errors = [], []
    for p in paths:
        path = Path(p)
        try:
            if path.is_file():
                path.unlink()
                deleted.append(str(path))
            else:
                errors.append(f"Not found: {p}")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{p}: {e}")
    return {"deleted": len(deleted), "errors": errors}


# ── Export CSV ────────────────────────────────────────────────────────────────
@app.post("/api/export-csv")
async def export_csv(reports: list[dict]):
    buf = io.StringIO()
    w   = csv.writer(buf)
    w.writerow(["File", "Path", "Overall", "Fails", "Warns",
                "Format", "File Size", "Resolution", "Color Profile",
                "Exposure", "Noise", "Sharpness", "JPEG Artifacts",
                "Chroma Aberration", "Saturation", "Watermark",
                "Anatomy", "Text in Image", "AI Artifacts",
                "Skin Texture", "Background", "Commercial Quality"])
    check_ids = [
        "format", "file_size", "resolution", "color_profile", "exposure",
        "noise", "sharpness", "jpeg_artifacts", "chroma_aberration", "saturation",
        "watermark", "anatomy", "text_in_image", "ai_artifacts",
        "skin_texture", "background_quality", "overall_commercial_quality",
    ]
    for r in reports:
        by_id = {c["id"]: c for c in r.get("checks", [])}
        row   = [r.get("file"), r.get("original_path", ""),
                 r.get("overall", ""), r.get("fail_count", ""), r.get("warn_count", "")]
        for cid in check_ids:
            c = by_id.get(cid)
            row.append(c["status"].upper() if c else "—")
        w.writerow(row)
    return Response(content=buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=stocklint_results.csv"})
