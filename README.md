# CuePilot

<p align="center">
  <img src="logo.png" alt="CuePilot Logo" width="220" />
</p>

<p align="center">
  <strong>Intelligent Hot Cue & Beatgrid Generator for Serato and Rekordbox</strong><br>
  <em>Local-first, genre-aware audio intelligence engine for modern DJs.</em>
</p>

---

## Overview

**CuePilot** automates music preparation for DJs. It analyzes audio files locally, detects tempo and beatgrids, segments musical structures (intro, verse, build, drop, breakdown, chorus, outro), and generates downbeat-aligned hot cues tailored to the track's musical genre.

Designed to be **100% local-first**, CuePilot processes your audio entirely on your machine with zero cloud dependencies and no audio data egress.

### Key Capabilities

- **Genre-Routed Cue Placement**: Instead of a generic one-size-fits-all approach, CuePilot routes tracks through genre-aware detection strategies covering 30+ genres (House, Tech House, Techno, Hip-Hop, Drum & Bass, Baile Funk, Indobounce, Afrobeats, and more).
- **Phrase & Downbeat Precision**: High-precision beatgrid detection, bar calculations, and musical phrase alignment (8/16/32-bar phrasing).
- **Stem Intelligence Engine (SIE)**: Leverages 4-channel source separation (vocal, bass, drums, other) to detect vocal entries, drop impacts, and energy shifts with pinpoint accuracy.
- **DJ Library Integration**:
  - **Serato DJ Pro**: Non-destructive ID3 / Vorbis metadata tagging (BPM, Camelot Key, cue tags) and canonical `.cueplan.json` exports.
  - **Pioneer Rekordbox**: Direct SQLite sync into Rekordbox's `master.db` with an automated backup, verify, and rollback safety lifecycle.
- **Desktop Application & CLI**: Includes both an interactive desktop application with interactive waveform scrubbing and a high-performance CLI for batch-processing music libraries.
- **Gig Readiness & Audio QC**: Detects audio clipping, dynamic range anomalies, corrupt headers, and provides a gig readiness score for every track.

---

## Pipeline

```text
Audio File (MP3, WAV, AIFF, FLAC, OGG)
                │
                ▼
      ┌───────────────────┐
      │  Audio Analyzer   │  (DSP Feature Extraction & Key Detection)
      └─────────┬─────────┘
                │
                ▼
      ┌───────────────────┐
      │ Beatgrid Engine   │  (BPM, Downbeats, Bars, Phrasing)
      └─────────┬─────────┘
                │
                ▼
      ┌───────────────────┐
      │ Structure Router  │  (Intro, Verse, Build, Drop, Breakdown, Outro)
      └─────────┬─────────┘
                │
                ▼
      ┌───────────────────┐
      │   Cue Generator   │  (DJ Slot Mapping & Downbeat Scoring)
      └─────────┬─────────┘
                │
        ┌───────┴───────┐
        ▼               ▼
 ┌─────────────┐ ┌─────────────┐
 │ Serato Tags │ │  Rekordbox  │
 │ & CuePlan   │ │  SQLite DB  │
 └─────────────┘ └─────────────┘
```

---

## Installation

### Prerequisites

- Python 3.11 or higher
- FFmpeg (recommended for decoding a wide range of audio formats)

### Setup

```bash
# Clone the repository
git clone https://github.com/YanNotTamm/CuePilot.git
cd CuePilot

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS / Linux:
source .venv/bin/activate

# Install dependencies and editable package
pip install -e .
```

---

## Usage

### Command Line Interface (CLI)

```bash
# Scan a directory recursively and verify file integrity
cuepilot scan "D:\Music\DJ Library"

# Analyze a single track with a specific genre profile
cuepilot analyze "D:\Music\track.mp3" --profile tech-house

# Batch-generate cue plans for an entire folder
cuepilot cues "D:\Music\EDM" --profile edm --out ./cues

# Create and run a resumable background batch job
cuepilot batch create "D:\Music\Setlist" --profile house
cuepilot batch run <job-id>

# Export and validate a CuePlan against the canonical schema
cuepilot export ./cues/track.cueplan.json

# Discover local Serato or Rekordbox library installations
cuepilot serato discover
cuepilot rekordbox discover
```

### Desktop UI

CuePilot provides a desktop interface featuring real-time waveform visualization, cue point editing, and library management:

```bash
# Launch the desktop app
python -m apps.desktop.entry
```

---

## Repository Layout

```text
CuePilot/
├── core/
│   ├── analyzer/        # Modular DSP analysis engine, Librosa & key detection
│   ├── beatgrid/        # BPM, beat, downbeat, bar, and phrase detection
│   ├── structure/       # Section segmentation and classification
│   ├── scoring/         # CueScore formula and downbeat alignment weighting
│   ├── profiles/        # Genre-specific cueing and phrasing profiles
│   ├── cues/            # DJ slot mapping and cue generator
│   ├── stems/           # Stem Intelligence Engine (vocal/bass/drums energy)
│   ├── integrations/    # Serato and Rekordbox library sync adapters
│   ├── quality/         # Audio QC and Gig Readiness scoring
│   ├── compatibility/   # Harmonic mixing & transition advisor
│   ├── crates/          # Smart crate auto-organization
│   ├── learn/           # Personalization and correction learning store
│   ├── scanner.py       # Audio folder scanner and deduplication
│   ├── pipeline.py      # End-to-end analysis orchestration
│   └── export.py        # Canonical JSON export and schema validation
├── apps/
│   └── desktop/         # Desktop application (FastAPI sidecar + Web UI)
├── schemas/             # JSON schemas for canonical CuePlan data
├── cli/                 # Command-line interface
├── tests/               # Unit, integration, and synthetic audio test suite
├── benchmarks/          # DSP benchmark harness and accuracy metrics
└── models/              # Pretrained models directory for stem separation
```

---

## Benchmarks

CuePilot includes an offline benchmark suite (`benchmarks.benchmark`) to evaluate DSP accuracy against synthetic ground truth tracks:

```bash
python -m benchmarks.benchmark
```

| Benchmark Case | Detected BPM | Beat F1 | Downbeat F1 | Boundary F1 | Cue Error |
|---|---|---|---|---|---|
| `edm_128` | 128.2 | 1.00 | 1.00 | 1.00 | < 0.05s |
| `edm_140` | 140.1 | 1.00 | 1.00 | 0.95 | < 0.10s |
| `techno_132` | 132.2 | 1.00 | 1.00 | 1.00 | < 0.02s |
| `hiphop_95` | 95.0 | 1.00 | 1.00 | 0.92 | < 0.02s |

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
