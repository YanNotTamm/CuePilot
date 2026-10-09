"""CuePilot CLI.

Commands: scan, analyze, cues, export.
GUI (Tauri) and Serato adapter come in later phases; the CLI must remain
available regardless (Section 21).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from core.analyzer import get_engine
from core.batch import batch_status, create_batch, run_batch
from core.export import export_cue_plan, validate_cue_plan_dict
from core.learn import PreferenceProfile, get_preference_store
from core.models import format_timestamp
from core.pipeline import analyze_track
from core.profiles.genre import list_profiles
from core.scanner import scan_folder

from .integrate import add_integration_parser


def _cmd_scan(args) -> int:
    if not os.path.isdir(args.folder):
        print(f"error: not a directory: {args.folder}", file=sys.stderr)
        return 1

    result = scan_folder(args.folder, recursive=not args.no_recursive)

    print(f"folder: {result.folder}")
    print(f"tracks: {result.total}")
    print(f"duplicates skipped: {result.duplicates}")
    print(f"failed: {result.failed}")
    print(f"elapsed: {result.elapsed_seconds:.2f}s")
    for track in result.tracks:
        status = track.status.value
        extra = track.error if track.error else track.display_name
        print(f"  [{status}] {track.path}  {extra}")

    if args.json:
        out = {
            "folder": result.folder,
            "total": result.total,
            "duplicates": result.duplicates,
            "failed": result.failed,
            "tracks": [t.to_dict() for t in result.tracks],
        }
        print(json.dumps(out, indent=2, ensure_ascii=False))

    return 0


def _cmd_analyze(args) -> int:
    if not os.path.isfile(args.file):
        print(f"error: not a file: {args.file}", file=sys.stderr)
        return 1
    try:
        result = analyze_track(
            args.file,
            profile_name=args.profile,
            engine=get_engine(args.engine),
            preference=args.preference,
        )
    except Exception as exc:
        print(f"error analyzing {args.file}: {exc}", file=sys.stderr)
        return 1

    a = result.analysis
    print(f"TRACK: {result.track_id}")
    print(f"BPM     {a.bpm:.0f}")
    print(f"KEY     {a.key}")
    print(f"ENGINE  {a.engine} {a.engine_version}")
    print(f"DURATION {format_timestamp(a.duration)}")
    print()
    print("BEATS:", len(a.beats), "DOWNBEATS:", len(a.downbeats), "PHRASES:", len(a.phrases))
    print()
    print("SECTIONS")
    for sec in result.sections:
        print(
            f"  {sec.type.value:<12} {format_timestamp(sec.start)} -> "
            f"{format_timestamp(sec.end)}  conf={sec.confidence:.0%}"
        )
    print()
    print("CUES")
    for cue in result.cue_plan.cues:
        print(
            f"  {cue.slot} {cue.type.value:<12} {format_timestamp(cue.time)} "
            f"{cue.confidence:.0%} {cue.bucket.value}"
        )
    return 0


def _cmd_cues(args) -> int:
    """Generate cues for a folder of tracks (batch) and write JSON per track."""
    if not os.path.isdir(args.folder):
        print(f"error: not a directory: {args.folder}", file=sys.stderr)
        return 1

    out_dir = args.out or tempfile.mkdtemp(prefix="cuepilot_cues_")
    os.makedirs(out_dir, exist_ok=True)

    scan = scan_folder(args.folder)
    tracks = [t for t in scan.tracks if t.status.value in ("QUEUED",)]
    print(f"analyzing {len(tracks)} of {scan.total} tracks -> {out_dir}")

    ok = 0
    failed = 0
    for i, track in enumerate(tracks, 1):
        try:
            result = analyze_track(track.path, track_id=track.id, profile_name=args.profile)
            track_block = {
                "path": track.path.replace("\\", "/"),
                "artist": track.artist,
                "title": track.title,
                "duration": round(track.duration, 2),
                "bpm": round(result.analysis.bpm, 2),
                "key": result.analysis.key,
            }
            base = os.path.splitext(os.path.basename(track.path))[0]
            out_path = os.path.join(out_dir, f"{base}.cueplan.json")
            export_cue_plan(result.cue_plan, out_path, track=track_block)
            ok += 1
            print(f"[{i}/{len(tracks)}] OK {track.path}")
        except Exception as exc:
            failed += 1
            print(f"[{i}/{len(tracks)}] FAIL {track.path}: {exc}", file=sys.stderr)

    print(f"done: {ok} exported, {failed} failed")
    return 0 if failed == 0 else 1


def _cmd_export(args) -> int:
    with open(args.input, "r", encoding="utf-8") as f:
        payload = json.load(f)
    errors = validate_cue_plan_dict(payload)
    if errors:
        print("invalid cueplan:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _cmd_batch(args) -> int:
    """Create and run a resumable batch job."""
    if not os.path.isdir(args.folder):
        print(f"error: not a directory: {args.folder}", file=sys.stderr)
        return 1

    if args.resume:
        job_id = args.resume
    else:
        job = create_batch(
            args.folder,
            profile=args.profile,
            engine=args.engine,
            out_dir=args.out,
            max_retries=args.retries,
        )
        job_id = job["jobId"]
        print(f"batch: {job_id} ({job['total']} tracks)")

    if args.create_only:
        print(json.dumps(batch_status(job_id), indent=2, ensure_ascii=False))
        return 0

    def progress(stage: str, done: int, total: int, pct: float, message: str) -> None:
        if stage == "analyzing":
            print(f"[{done}/{total}] analyzing: {message}")

    result = run_batch(job_id, progress=progress)
    print(f"done: ok={result.ok} failed={result.failed} skipped={result.skipped} in {result.elapsed_seconds}s")
    for f in result.failures:
        print(f"  FAIL {f['path']}: {f['error']}", file=sys.stderr)
    return 0 if result.failed == 0 else 1


def _cmd_preferences_list(args) -> int:
    """List saved DJ preference profiles."""
    store = get_preference_store()
    profiles = store.list_profiles()
    if not profiles:
        print("no preference profiles saved")
        return 0
    for p in profiles:
        genres = ", ".join(sorted(p.weights)) or "(no overrides)"
        print(f"{p.name:<24} genres: {genres}")
    return 0


def _cmd_preferences_show(args) -> int:
    store = get_preference_store()
    profile = store.get(args.name)
    if profile is None:
        print(f"error: no profile named {args.name}", file=sys.stderr)
        return 1
    print(json.dumps(profile.to_dict(), indent=2, ensure_ascii=False))
    return 0


def _cmd_preferences_save(args) -> int:
    store = get_preference_store()
    try:
        weights = json.loads(args.weights)
    except json.JSONDecodeError as exc:
        print(f"error: invalid JSON: {exc}", file=sys.stderr)
        return 1
    if not isinstance(weights, dict):
        print("error: weights must be a JSON object", file=sys.stderr)
        return 1
    profile = PreferenceProfile(name=args.name, weights=weights)
    store.save(profile)
    print(f"saved profile: {args.name}")
    return 0


def _cmd_preferences_delete(args) -> int:
    store = get_preference_store()
    if store.delete(args.name):
        print(f"deleted profile: {args.name}")
        return 0
    print(f"error: no profile named {args.name}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cuepilot", description="AI Hotcue Generator for Serato")
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="scan a music folder")
    p_scan.add_argument("folder", help="music folder path")
    p_scan.add_argument("--no-recursive", action="store_true")
    p_scan.add_argument("--json", action="store_true")
    p_scan.set_defaults(func=_cmd_scan)

    p_analyze = sub.add_parser("analyze", help="analyze a single track")
    p_analyze.add_argument("file", help="audio file path")
    p_analyze.add_argument("--profile", default="open_format", help="genre profile")
    p_analyze.add_argument("--engine", default="librosa", help="analyzer engine")
    p_analyze.add_argument("--preference", default="", help="DJ preference profile name")
    p_analyze.set_defaults(func=_cmd_analyze)

    p_cues = sub.add_parser("cues", help="batch-generate cues for a folder")
    p_cues.add_argument("folder", help="music folder path")
    p_cues.add_argument("--profile", default="open_format", help="genre profile")
    p_cues.add_argument("--out", default=None, help="output directory for cueplan files")
    p_cues.set_defaults(func=_cmd_cues)

    p_export = sub.add_parser("export", help="validate and print a cueplan JSON")
    p_export.add_argument("input", help="cueplan JSON file")
    p_export.set_defaults(func=_cmd_export)

    p_batch = sub.add_parser("batch", help="create/run a resumable batch analysis job")
    p_batch.add_argument("folder", help="music folder path")
    p_batch.add_argument("--profile", default="open_format", help="genre profile")
    p_batch.add_argument("--engine", default="basic", help="analyzer engine mode (basic/intellistem/onnx)")
    p_batch.add_argument("--out", default=None, help="output directory for cueplan files")
    p_batch.add_argument("--retries", type=int, default=2, help="max retries per failed track")
    p_batch.add_argument("--resume", default=None, metavar="JOB_ID", help="resume an existing batch job")
    p_batch.add_argument("--create-only", action="store_true", help="only create the job, do not run")
    p_batch.set_defaults(func=_cmd_batch)

    p_pref = sub.add_parser("preferences", help="manage DJ preference profiles")
    p_pref_sub = p_pref.add_subparsers(dest="pref_command", required=True)
    p_pref_list = p_pref_sub.add_parser("list", help="list saved profiles")
    p_pref_list.set_defaults(func=_cmd_preferences_list)
    p_pref_show = p_pref_sub.add_parser("show", help="show a saved profile")
    p_pref_show.add_argument("name", help="profile name")
    p_pref_show.set_defaults(func=_cmd_preferences_show)
    p_pref_save = p_pref_sub.add_parser("save", help="save a profile from per-genre weight overrides")
    p_pref_save.add_argument("name", help="profile name")
    p_pref_save.add_argument("weights", help="JSON: {genre: {subscore: weight}}")
    p_pref_save.set_defaults(func=_cmd_preferences_save)
    p_pref_del = p_pref_sub.add_parser("delete", help="delete a profile")
    p_pref_del.add_argument("name", help="profile name")
    p_pref_del.set_defaults(func=_cmd_preferences_delete)

    p_profiles = sub.add_parser("profiles", help="list available genre profiles")
    p_profiles.set_defaults(func=lambda _: (print("\n".join(list_profiles())), 0)[1])

    add_integration_parser(sub)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
