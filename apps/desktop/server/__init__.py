"""CuePilot desktop sidecar backend API.

FastAPI server bound to 127.0.0.1 on a random port with a per-session handshake
token. Spawned and owned by the desktop shell (pywebview); never exposed to the
LAN (binds loopback only). The WebView talks to this API over HTTP.
"""

from __future__ import annotations

import os
import secrets
import sys
import threading
import time
import uuid
from pathlib import Path

import uvicorn

_REPO_ROOT = Path(__file__).resolve().parents[2]

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.analyzer.librosa_engine import LibrosaEngine
from core.batch import (
    BatchCancelled,
    batch_status,
    create_batch,
    request_cancel,
    run_batch,
)
from core.cues.generator import generate_cueplan
from core.edits import load_edits, persist_edits
from core.export import export_cue_plan, validate_cue_plan_dict
from core.learn import get_store, record_feedback_rows
from core.learn.preferences import PreferenceProfile, get_preference_store
from core.metadata import write_metadata
from core.models import CuePlan, GenreProfile
from core.pipeline import analyze_track
from core.profiles.genre import get_profile, list_profiles
from core.scanner import scan_folder
from core.settings import (
    ENGINE_MODES,
    get_engine_mode,
    get_enabled_stems,
    set_engine_mode,
    set_enabled_stems,
)

TOKEN = os.environ.get("CUEPILOT_SESSION_TOKEN", "")

if getattr(sys, "frozen", False):
    WEB_DIST = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1])) / "apps" / "desktop" / "web" / "dist"
else:
    WEB_DIST = Path(__file__).resolve().parents[1] / "web" / "dist"


def _authorized(x_session_token: str | None) -> bool:
    return bool(TOKEN) and x_session_token == TOKEN


class ScanRequest(BaseModel):
    folder: str
    recursive: bool = True


class AnalyzeRequest(BaseModel):
    path: str
    profile: str = "open_format"
    engine: str = ""
    preference: str = ""
    stems: list[str] | None = None


class SettingsRequest(BaseModel):
    engine: str = ""
    stems: list[str] | None = None


class WaveformRequest(BaseModel):
    path: str
    bins: int = 3000


class ExportRequest(BaseModel):
    path: str
    profile: str = "open_format"


class AudioRequest(BaseModel):
    path: str


class UpdateCueRequest(BaseModel):
    track_id: str
    cues: list[dict]


class EditsRequest(BaseModel):
    path: str
    profile: str = "open_format"
    bpm: float = 0.0
    key: str = ""
    duration: float = 0.0
    cues: list[dict] = []
    dry_run: bool = False


class EditsLoadRequest(BaseModel):
    path: str


class LearnStatsRequest(BaseModel):
    genre: str = "open_format"


class MetadataWriteRequest(BaseModel):
    path: str
    bpm: float = 0.0
    key: str = ""
    cues: list[dict] = []
    analysis: dict | None = None
    dry_run: bool = False


class BatchCreateRequest(BaseModel):
    folder: str
    profile: str = "open_format"
    engine: str = "basic"
    recursive: bool = True
    out_dir: str | None = None
    preference: str = ""


class BatchRunRequest(BaseModel):
    job_id: str


class PreferenceRequest(BaseModel):
    name: str
    weights: dict[str, dict[str, float]] = {}
    slot_strategy: dict[str, dict[int, str]] = {}


class PreferenceApplyRequest(BaseModel):
    name: str


class CompatibilityRequest(BaseModel):
    """Per-track analysis records for the Harmonic Mixing Advisor (45.1)."""

    tracks: list[dict] = []
    track_id: str = ""
    top_n: int = 5
    min_score: float = 0.5


class CrateRequest(BaseModel):
    """Per-track records for Smart Crate Auto-Organization (45.9)."""

    tracks: list[dict] = []


class MashupRequest(BaseModel):
    """Per-track records for the Mashup / Acapella Finder (45.4)."""

    tracks: list[dict] = []
    top_n: int = 5
    min_score: float = 0.55


class PlanSetRequest(BaseModel):
    """Inputs for the AI Set Planner draft (45.5)."""

    tracks: list[dict] = []
    target_minutes: float = 60.0
    mood_arc: list[str] = []
    must_include: list[str] = []
    genre_pool: list[str] = []


class PrivacyRequest(BaseModel):
    """Privacy mode override for the LLM semantic layer (45.8)."""

    mode: str = ""
    base_url: str = ""
    model: str = ""


def create_app() -> FastAPI:
    app = FastAPI(title="CuePilot Sidecar", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health(x_session_token: str | None = Header(default=None)):
        if not _authorized(x_session_token):
            raise HTTPException(status_code=401, detail="invalid session token")
        return {"ok": True, "time": time.time()}

    @app.get("/api/profiles")
    def api_profiles(x_session_token: str | None = Header(default=None)):
        if not _authorized(x_session_token):
            raise HTTPException(status_code=401, detail="invalid session token")
        return {"profiles": list_profiles(), "meta": {n: get_profile(n).to_dict() for n in list_profiles()}}

    @app.get("/api/settings")
    def api_get_settings(x_session_token: str | None = Header(default=None)):
        if not _authorized(x_session_token):
            raise HTTPException(status_code=401, detail="invalid session token")
        from core.stems import get_acceleration
        acc = get_acceleration()
        return {
            "engine": get_engine_mode(),
            "engineModes": ENGINE_MODES,
            "gpu": {"provider": acc.provider, "device": acc.device},
            "stems": get_enabled_stems(),
        }

    @app.post("/api/settings")
    def api_set_settings(req: SettingsRequest, x_session_token: str | None = Header(default=None)):
        if not _authorized(x_session_token):
            raise HTTPException(status_code=401, detail="invalid session token")
        if req.engine:
            set_engine_mode(req.engine)
        if req.stems is not None:
            set_enabled_stems(req.stems)
        from core.stems import get_acceleration
        acc = get_acceleration()
        return {
            "engine": get_engine_mode(),
            "engineModes": ENGINE_MODES,
            "gpu": {"provider": acc.provider, "device": acc.device},
            "stems": get_enabled_stems(),
        }

    @app.post("/api/scan")
    def api_scan(req: ScanRequest, x_session_token: str | None = Header(default=None)):
        if not _authorized(x_session_token):
            raise HTTPException(status_code=401, detail="invalid session token")
        result = scan_folder(req.folder, recursive=req.recursive)
        return {
            "folder": result.folder,
            "total": result.total,
            "duplicates": result.duplicates,
            "failed": result.failed,
            "tracks": [t.to_dict() for t in result.tracks],
        }

    @app.post("/api/analyze")
    def api_analyze(req: AnalyzeRequest, x_session_token: str | None = Header(default=None)):
        if not _authorized(x_session_token):
            raise HTTPException(status_code=401, detail="invalid session token")
        engine = req.engine or get_engine_mode()
        result = analyze_track(
            req.path,
            profile_name=req.profile,
            mode=engine,
            preference=req.preference,
            enabled_stems=req.stems,
        )
        return {
            "analysis": result.analysis.to_dict(),
            "sections": [s.to_dict() for s in result.sections],
            "cues": result.cue_plan.to_dict()["cues"],
        }

    @app.post("/api/analyze/stream")
    def api_analyze_stream(req: AnalyzeRequest, x_session_token: str | None = Header(default=None)):
        if not _authorized(x_session_token):
            raise HTTPException(status_code=401, detail="invalid session token")

        from fastapi.responses import StreamingResponse
        import json as _json
        import queue as _queue
        import threading as _threading

        def sse(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {_json.dumps(data, ensure_ascii=False)}\n\n"

        def generate():
            q: _queue.Queue = _queue.Queue(maxsize=128)
            _END = object()

            def on_progress(stage: str, message: str, pct: float):
                q.put(sse("progress", {"stage": stage, "message": message, "pct": pct}))

            def work():
                try:
                    engine = req.engine or get_engine_mode()
                    result = analyze_track(
                        req.path,
                        profile_name=req.profile,
                        progress=on_progress,
                        mode=engine,
                        preference=req.preference,
                        enabled_stems=req.stems,
                    )
                    payload = {
                        "analysis": result.analysis.to_dict(),
                        "sections": [s.to_dict() for s in result.sections],
                        "cues": result.cue_plan.to_dict()["cues"],
                    }
                    q.put(sse("done", payload))
                except Exception as exc:  # noqa: BLE001
                    q.put(sse("error", {"message": str(exc)}))
                finally:
                    q.put(_END)

            yield sse(
                "progress",
                {"stage": "start", "message": "starting analysis", "pct": 0},
            )
            _threading.Thread(target=work, daemon=True).start()
            while True:
                item = q.get()
                if item is _END:
                    break
                yield item

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/waveform")
    def api_waveform(req: WaveformRequest, x_session_token: str | None = Header(default=None)):
        if not _authorized(x_session_token):
            raise HTTPException(status_code=401, detail="invalid session token")
        from .waveform import compute_waveform

        return compute_waveform(req.path, req.bins)

    @app.post("/api/audio")
    def api_audio(req: AudioRequest, x_session_token: str | None = Header(default=None)):
        if not _authorized(x_session_token):
            raise HTTPException(status_code=401, detail="invalid session token")
        if not os.path.isfile(req.path):
            raise HTTPException(status_code=404, detail="file not found")
        from fastapi.responses import FileResponse

        return FileResponse(req.path, media_type="audio/mpeg", filename=os.path.basename(req.path))

    @app.post("/api/export")
    def api_export(req: ExportRequest, x_session_token: str | None = Header(default=None)):
        if not _authorized(x_session_token):
            raise HTTPException(status_code=401, detail="invalid session token")
        import tempfile

        result = analyze_track(req.path, profile_name=req.profile)
        out = os.path.join(tempfile.gettempdir(), "cuepilot")
        os.makedirs(out, exist_ok=True)
        out_path = os.path.join(out, f"{uuid.uuid4().hex}.cueplan.json")
        track_block = {
            "path": os.path.abspath(req.path).replace("\\", "/"),
            "artist": "",
            "title": Path(req.path).stem,
            "duration": round(result.analysis.duration, 2),
            "bpm": round(result.analysis.bpm, 2),
            "key": result.analysis.key,
        }
        export_cue_plan(result.cue_plan, out_path, track=track_block)
        return {"path": out_path, "cuePlan": result.cue_plan.to_dict()}

    @app.post("/api/batch/create")
    def api_batch_create(req: BatchCreateRequest, x_session_token: str | None = Header(default=None)):
        if not _authorized(x_session_token):
            raise HTTPException(status_code=401, detail="invalid session token")
        if not os.path.isdir(req.folder):
            raise HTTPException(status_code=404, detail="folder not found")
        return create_batch(
            req.folder,
            profile=req.profile,
            engine=req.engine,
            recursive=req.recursive,
            out_dir=req.out_dir,
            preference=req.preference,
        )

    @app.post("/api/batch/run")
    def api_batch_run(req: BatchRunRequest, x_session_token: str | None = Header(default=None)):
        if not _authorized(x_session_token):
            raise HTTPException(status_code=401, detail="invalid session token")
        import threading as _threading
        import queue as _queue

        from fastapi.responses import StreamingResponse
        import json as _json

        def sse(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {_json.dumps(data, ensure_ascii=False)}\n\n"

        def generate():
            q: _queue.Queue = _queue.Queue(maxsize=64)
            _END = object()

            def work():
                try:
                    run_batch(
                        req.job_id,
                        progress=lambda stage, done, total, pct, msg: q.put(
                            sse("progress", {"stage": stage, "done": done, "total": total, "pct": pct, "message": msg})
                        ),
                    )
                    q.put(sse("done", batch_status(req.job_id)))
                except BatchCancelled:
                    q.put(sse("done", batch_status(req.job_id)))
                except Exception as exc:  # noqa: BLE001
                    q.put(sse("error", {"message": str(exc)}))
                finally:
                    q.put(_END)

            yield sse("progress", {"stage": "start", "done": 0, "total": 0, "pct": 0, "message": "starting batch"})
            _threading.Thread(target=work, daemon=True).start()
            while True:
                item = q.get()
                if item is _END:
                    break
                yield item

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/batch/status")
    def api_batch_status(req: BatchRunRequest, x_session_token: str | None = Header(default=None)):
        if not _authorized(x_session_token):
            raise HTTPException(status_code=401, detail="invalid session token")
        return batch_status(req.job_id)

    @app.post("/api/batch/cancel")
    def api_batch_cancel(req: BatchRunRequest, x_session_token: str | None = Header(default=None)):
        if not _authorized(x_session_token):
            raise HTTPException(status_code=401, detail="invalid session token")
        return {"ok": request_cancel(req.job_id)}

    @app.post("/api/edits/load")
    def api_edits_load(req: EditsLoadRequest, x_session_token: str | None = Header(default=None)):
        if not _authorized(x_session_token):
            raise HTTPException(status_code=401, detail="invalid session token")
        edits = load_edits(req.path)
        if edits is None:
            return {"exists": False}
        return {"exists": True, "edits": edits.to_dict()}

    @app.post("/api/edits/save")
    def api_edits_save(req: EditsRequest, x_session_token: str | None = Header(default=None)):
        if not _authorized(x_session_token):
            raise HTTPException(status_code=401, detail="invalid session token")
        if not os.path.isfile(req.path):
            raise HTTPException(status_code=404, detail="file not found")

        # Wrap raw cue dicts into Cue models so persist_edits can serialize.
        from core.models import Cue, CueType

        cues: list[Cue] = []
        for c in req.cues:
            try:
                ctype = CueType(c.get("type", "CUSTOM"))
            except ValueError:
                ctype = CueType.CUSTOM
            cues.append(
                Cue(
                    slot=int(c.get("slot", 0)),
                    type=ctype,
                    time=float(c.get("time", 0.0)),
                    label=str(c.get("label", "")),
                    confidence=float(c.get("confidence", 1.0)),
                    color=str(c.get("color", "")),
                    locked=bool(c.get("locked", False)),
                    source=str(c.get("source", "dsp")),
                    reason=[str(r) for r in c.get("reason", [])],
                )
            )

        result = persist_edits(
            req.path,
            profile=req.profile,
            bpm=req.bpm,
            key=req.key,
            cues=cues,
            dry_run=req.dry_run,
        )

        # Personalization: record the DJ-approved layout as learning feedback
        # so future analyses of other tracks with the same genre get nudged.
        if not req.dry_run:
            store = get_store()
            record_feedback_rows(
                cues,
                track_path=os.path.abspath(req.path),
                genre=req.profile,
                track_duration=req.duration,
                store=store,
            )
            result["learnStats"] = store.stats()

        return result

    @app.post("/api/learn/stats")
    def api_learn_stats(req: LearnStatsRequest, x_session_token: str | None = Header(default=None)):
        if not _authorized(x_session_token):
            raise HTTPException(status_code=401, detail="invalid session token")
        store = get_store()
        return {
            "stats": store.stats(),
            "patterns": store.patterns_for(req.genre),
            "minSamples": 3,
        }

    @app.get("/api/preferences/list")
    def api_preferences_list(x_session_token: str | None = Header(default=None)):
        if not _authorized(x_session_token):
            raise HTTPException(status_code=401, detail="invalid session token")
        return {"profiles": [p.to_dict() for p in get_preference_store().list_profiles()]}

    @app.post("/api/preferences/save")
    def api_preferences_save(req: PreferenceRequest, x_session_token: str | None = Header(default=None)):
        if not _authorized(x_session_token):
            raise HTTPException(status_code=401, detail="invalid session token")
        profile = PreferenceProfile(name=req.name, weights=req.weights, slot_strategy=req.slot_strategy)
        get_preference_store().save(profile)
        return {"ok": True, "profile": profile.to_dict()}

    @app.post("/api/preferences/delete")
    def api_preferences_delete(req: PreferenceApplyRequest, x_session_token: str | None = Header(default=None)):
        if not _authorized(x_session_token):
            raise HTTPException(status_code=401, detail="invalid session token")
        return {"ok": get_preference_store().delete(req.name)}

    @app.post("/api/compatibility")
    def api_compatibility(req: CompatibilityRequest, x_session_token: str | None = Header(default=None)):
        if not _authorized(x_session_token):
            raise HTTPException(status_code=401, detail="invalid session token")
        from core.compatibility import best_next_tracks

        if not req.tracks:
            return {"trackId": req.track_id, "compatibleWith": []}
        target = next(
            (t for t in req.tracks if (t.get("id") or t.get("trackId")) == req.track_id),
            req.tracks[0],
        )
        results = best_next_tracks(
            target,
            req.tracks,
            top_n=req.top_n,
            min_score=req.min_score,
        )
        return {
            "trackId": target.get("id") or target.get("trackId", ""),
            "compatibleWith": [r.to_dict() for r in results],
        }

    @app.post("/api/crates/organize")
    def api_crates_organize(req: CrateRequest, x_session_token: str | None = Header(default=None)):
        if not _authorized(x_session_token):
            raise HTTPException(status_code=401, detail="invalid session token")
        from core.crates import organize_crates

        return organize_crates(req.tracks).to_dict()

    @app.post("/api/mashup/find")
    def api_mashup_find(req: MashupRequest, x_session_token: str | None = Header(default=None)):
        if not _authorized(x_session_token):
            raise HTTPException(status_code=401, detail="invalid session token")
        from core.compatibility import find_mashups

        return {"candidates": [c.to_dict() for c in find_mashups(req.tracks, top_n=req.top_n, min_score=req.min_score)]}

    @app.post("/api/planner/plan")
    def api_planner_plan(req: PlanSetRequest, x_session_token: str | None = Header(default=None)):
        if not _authorized(x_session_token):
            raise HTTPException(status_code=401, detail="invalid session token")
        from core.planner import plan_set

        return plan_set(
            req.tracks,
            target_minutes=req.target_minutes,
            mood_arc=req.mood_arc or ["warm-up", "build", "peak", "cool-down"],
            must_include=req.must_include,
            genre_pool=req.genre_pool,
        ).to_dict()

    @app.post("/api/privacy/resolve")
    def api_privacy_resolve(req: PrivacyRequest, x_session_token: str | None = Header(default=None)):
        if not _authorized(x_session_token):
            raise HTTPException(status_code=401, detail="invalid session token")
        from core.llm import detect_local_server, resolve_privacy

        status = resolve_privacy(
            mode=req.mode or None,
            base_url=req.base_url or None,
            model=req.model or None,
        )
        data = status.to_dict()
        if status.private:
            data["localServerAvailable"] = detect_local_server(status.base_url)
        else:
            data["localServerAvailable"] = False
        return data

    @app.post("/api/metadata/write")
    def api_metadata_write(req: MetadataWriteRequest, x_session_token: str | None = Header(default=None)):
        if not _authorized(x_session_token):
            raise HTTPException(status_code=401, detail="invalid session token")
        result = write_metadata(
            req.path,
            bpm=req.bpm,
            key=req.key,
            cues=req.cues,
            analysis=req.analysis,
            dry_run=req.dry_run,
        )
        if not result.get("ok"):
            raise HTTPException(status_code=422, detail=result.get("error", "write failed"))
        return result

    @app.post("/api/metadata/write/stream")
    def api_metadata_write_stream(
        req: MetadataWriteRequest, x_session_token: str | None = Header(default=None)
    ):
        if not _authorized(x_session_token):
            raise HTTPException(status_code=401, detail="invalid session token")

        from fastapi.responses import StreamingResponse
        import json as _json
        import queue as _queue
        import threading as _threading
        import time as _time
        import random as _random

        def sse(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {_json.dumps(data, ensure_ascii=False)}\n\n"

        STAGES = [
            ("read", "reading original tags", 15, 0.6),
            ("backup", "creating backup of original file", 35, 0.8),
            ("write", "writing BPM & key tags", 60, 0.9),
            ("cues", "writing hot cue plan tag", 85, 0.9),
            ("finalize", "finalizing metadata", 100, 0.6),
        ]
        total_extra = _random.uniform(1.0, 3.4)  # total ~3-6s with base stage sleeps

        def generate():
            q: _queue.Queue = _queue.Queue(maxsize=16)
            _END = object()

            def work():
                try:
                    extra_budget = total_extra / max(1, len(STAGES))
                    for idx, (stage, msg, pct, base) in enumerate(STAGES):
                        q.put(sse("progress", {"stage": stage, "message": msg, "pct": pct}))
                        _time.sleep(base + extra_budget)
                    result = write_metadata(
                        req.path,
                        bpm=req.bpm,
                        key=req.key,
                        cues=req.cues,
                        analysis=req.analysis,
                        dry_run=req.dry_run,
                    )
                    q.put(sse("done", result))
                except Exception as exc:  # noqa: BLE001
                    q.put(sse("error", {"message": str(exc)}))
                finally:
                    q.put(_END)

            _threading.Thread(target=work, daemon=True).start()
            while True:
                item = q.get()
                if item is _END:
                    break
                yield item

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    if WEB_DIST.is_dir():
        app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

        @app.get("/logo.png", response_class=FileResponse)
        def logo():
            return (WEB_DIST / "logo.png").as_posix()

        @app.get("/", response_class=HTMLResponse)
        def index():
            html = (WEB_DIST / "index.html").read_text(encoding="utf-8")
            return html.replace("</body>", f"<script>window.CUEPILOT_TOKEN={TOKEN!r};</script></body>")

    return app


def run_server(port: int, token: str) -> None:
    """Run the uvicorn server on 127.0.0.1:<port> in a daemon thread."""
    global TOKEN
    TOKEN = token
    config = uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return thread


def find_free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
