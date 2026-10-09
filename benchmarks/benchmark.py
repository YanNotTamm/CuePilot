"""DSP benchmark harness.

Runs the full pipeline over synthetic tracks with known ground truth and
reports:

- BPM accuracy (half/double-time tolerant)
- beat accuracy (F1 within tolerance)
- downbeat accuracy (F1 within tolerance)
- section boundary F1
- cue timing error vs the section it labels

Usage:
    python -m benchmarks.benchmark [--out benchmarks/results.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.analyzer import LibrosaEngine
from core.analyzer.base import AudioData
from core.beatgrid import detect_beatgrid
from core.cues import generate_cueplan
from core.profiles.genre import get_profile
from core.structure import detect_structure
from tests.fixtures.synth import structure_boundaries, synth_track

BEAT_TOL = 0.05  # seconds, half a kick at 120 BPM is 0.25s; 50ms is strict
DOWNBEAT_TOL = 0.15
BOUNDARY_TOL = 2.5  # seconds, allows phrase-alignment drift

STRUCTURES: dict[str, list[tuple[str, float]]] = {
    "edm": [
        ("INTRO", 0.25),
        ("BUILD", 0.55),
        ("DROP", 1.00),
        ("BREAKDOWN", 0.15),
        ("BUILD", 0.60),
        ("SECOND_DROP", 1.00),
        ("OUTRO", 0.20),
    ],
    "techno": [
        ("INTRO", 0.30),
        ("DROP", 1.00),
        ("BUILD", 0.65),
        ("DROP", 1.00),
        ("OUTRO", 0.25),
    ],
    "hip_hop": [
        ("INTRO", 0.35),
        ("VOCAL", 0.80),
        ("DROP", 0.95),
        ("BREAK", 0.30),
        ("DROP", 1.00),
        ("OUTRO", 0.40),
    ],
}

CASES = [
    dict(name="edm_128", bpm=128.0, seconds=96.0, structure=STRUCTURES["edm"], profile="edm"),
    dict(name="edm_140", bpm=140.0, seconds=96.0, structure=STRUCTURES["edm"], profile="edm"),
    dict(name="techno_132", bpm=132.0, seconds=64.0, structure=STRUCTURES["techno"], profile="techno"),
    dict(name="hiphop_95", bpm=95.0, seconds=64.0, structure=STRUCTURES["hip_hop"], profile="hip_hop"),
]


def _ground_beats(bpm: float, seconds: float) -> list[float]:
    period = 60.0 / bpm
    n = int(seconds / period) + 1
    return [i * period for i in range(n)]


def _ground_downbeats(bpm: float, seconds: float) -> list[float]:
    period = 60.0 / bpm
    n = int(seconds / period) + 1
    return [i * period for i in range(0, n, 4)]


def _match_pairs(pred: list[float], truth: list[float], tol: float) -> tuple[int, int, int]:
    """Return (tp, fp, fn) after greedy nearest-neighbour matching."""
    tp = 0
    used = set()
    for p in pred:
        best = min(range(len(truth)), key=lambda j: abs(truth[j] - p), default=None)
        if best is not None and best not in used and abs(truth[best] - p) <= tol:
            used.add(best)
            tp += 1
    fp = len(pred) - tp
    fn = len(truth) - tp
    return tp, fp, fn


def _f1(tp: int, fp: int, fn: int) -> float:
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def run_case(case: dict) -> dict:
    t0 = time.perf_counter()
    samples, sr = synth_track(bpm=case["bpm"], seconds=case["seconds"], structure=case["structure"])
    audio = AudioData(samples=samples[np.newaxis, :], sample_rate=int(sr), channels=1, duration=case["seconds"])

    feats = LibrosaEngine().extract(audio)
    grid = detect_beatgrid(feats)
    profile = get_profile(case["profile"])
    sections = detect_structure(feats, grid, profile)
    plan = generate_cueplan(None, feats, grid, sections, profile)

    # --- BPM accuracy (half/double-time tolerant) ---
    gt_bpm = case["bpm"]
    det_bpm = grid.bpm
    bpm_err = abs(det_bpm - gt_bpm) / gt_bpm
    bpm_ok = bpm_err <= 0.05 or (bpm_err >= 0.95 and bpm_err <= 1.05) or (bpm_err >= 1.95 and bpm_err <= 2.05)

    # --- beat / downbeat accuracy ---
    gt_beats = _ground_beats(case["bpm"], case["seconds"])
    gt_downbeats = _ground_downbeats(case["bpm"], case["seconds"])
    tp_b, fp_b, fn_b = _match_pairs([b.time for b in grid.beats], gt_beats, BEAT_TOL)
    beat_f1 = _f1(tp_b, fp_b, fn_b)
    tp_d, fp_d, fn_d = _match_pairs(grid.downbeat_times, gt_downbeats, DOWNBEAT_TOL)
    downbeat_f1 = _f1(tp_d, fp_d, fn_d)

    # --- section boundary F1 ---
    gt_bounds = structure_boundaries(case["bpm"], case["seconds"], case["structure"])
    det_bounds = [s.end for s in sections[:-1]] if sections else []
    tp_s, fp_s, fn_s = _match_pairs(det_bounds, gt_bounds, BOUNDARY_TOL)
    boundary_f1 = _f1(tp_s, fp_s, fn_s)

    # --- cue timing error (label-aware: each cue vs its ground segment start) ---
    gt_starts = structure_boundaries(case["bpm"], case["seconds"], case["structure"])
    cue_errs = []
    used = set()
    for c in plan.cues:
        lab = c.type.value
        if lab in ("INTRO", "OUTRO", "CUSTOM"):
            continue
        # Ground segment start for the first unused segment whose label matches.
        best = None
        best_err = None
        for j, (glab, gamp) in enumerate(case["structure"]):
            if glab == lab and j not in used:
                gt_time = gt_starts[j - 1] if j >= 1 else 0.0
                err = abs(gt_time - c.time)
                if best_err is None or err < best_err:
                    best_err = err
                    best = j
        if best is not None:
            used.add(best)
            cue_errs.append(best_err)
    cue_timing_err = float(np.mean(cue_errs)) if cue_errs else None

    elapsed = time.perf_counter() - t0
    return {
        "case": case["name"],
        "bpm": {"ground": gt_bpm, "detected": round(det_bpm, 3), "err": round(bpm_err, 4), "ok": bpm_ok},
        "beats": {"count_gt": len(gt_beats), "count_det": len(grid.beats), "f1": round(beat_f1, 3)},
        "downbeats": {"count_gt": len(gt_downbeats), "count_det": len(grid.downbeat_times), "f1": round(downbeat_f1, 3)},
        "sections": {
            "count_gt": len(case["structure"]),
            "count_det": len(sections),
            "boundary_f1": round(boundary_f1, 3),
            "detected": [s.type.value for s in sections],
        },
        "cues": {"count": len(plan.cues), "mean_timing_err": round(cue_timing_err, 3) if cue_timing_err is not None else None},
        "elapsed_s": round(elapsed, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmarks/results.json")
    args = ap.parse_args()

    results = [run_case(c) for c in CASES]

    print(f"{'case':<14}{'bpm':>12}{'beatF1':>8}{'downF1':>8}{'boundF1':>9}{'cueErr':>8}{'sections':>22}  time")
    for r in results:
        b = r["bpm"]
        bpm_s = f"{b['detected']:.1f}" + ("*" if not b["ok"] else "")
        print(
            f"{r['case']:<14}{bpm_s:>12}{r['beats']['f1']:>8.2f}{r['downbeats']['f1']:>8.2f}"
            f"{r['sections']['boundary_f1']:>9.2f}{str(r['cues']['mean_timing_err']):>8}"
            f"{'/'.join(r['sections']['detected']):>22}  {r['elapsed_s']}s"
        )

    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
