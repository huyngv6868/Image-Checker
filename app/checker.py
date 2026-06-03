"""Orchestrator — runs technical + vision checks on one image, returns full report."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

from .checks import technical, vision


def _overall(checks: list[dict]) -> str:
    statuses = {c["status"] for c in checks}
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    return "pass"


def run(path: Path, *, skip_vision: bool = False) -> dict:
    """Run all checks. Returns full report dict."""
    t0 = time.time()

    tech_checks = technical.run_all(path)
    vis_checks = vision.run_all(path) if not skip_vision else [{
        "id": "vision_skipped",
        "name": "AI Vision Check",
        "status": "warn",
        "value": "Skipped",
        "threshold": "—",
        "message": "Vision check skipped (no API key or disabled).",
    }]

    all_checks = tech_checks + vis_checks
    overall = _overall(all_checks)

    fails = [c for c in all_checks if c["status"] == "fail"]
    warns = [c for c in all_checks if c["status"] == "warn"]

    return {
        "file": path.name,
        "path": str(path),
        "overall": overall,
        "fail_count": len(fails),
        "warn_count": len(warns),
        "checks": all_checks,
        "duration_s": round(time.time() - t0, 2),
    }
