# CHANGELOG — Audit Remediation (Bilingual / Bilingual)

> Status: perubahan ini sudah di-push ke branch `audit-fix-jojo-the-cat` di repo utama YanNotTamm/CuePilot dan **belum di-merge ke main**.
> This changelog covers work pushed to branch `audit-fix-jojo-the-cat`
> in the main repository YanNotTamm/CuePilot. **Nothing has been merged
> to main yet.**
>
> Prinsip: tidak ada klaim fitur yang belum terverifikasi. Angka benchmark
> hanya yang terukur di `benchmarks/results.json`; klaim lama yang
> bertentangan dengannya dicabut, bukan diganti dengan angka baru.
> Principle: no unverified feature is claimed. Benchmark figures are only
> those measured in `benchmarks/results.json`; old claims contradicting
> them are withdrawn, not replaced with new unmeasured figures.

## Unreleased — Perbaikan hasil audit / Audit remediation

### Kritis / Critical

#### K-1 — Serato: batasan jujur / Honest Serato scope
- **ID:** Hot cue Serato **BELUM didukung**. CuePilot untuk Serato hanya
  menulis BPM/key + tag `CUEPILOT_CUES` pada file audio dan mengekspor
  `.cueplan.json`. Format Serato Markers / DatabaseV2 **tidak** ditulis
  dan **tidak** ditebak, karena format tersebut terenkripsi dan belum
  terverifikasi. Pesan hasil apply, dry-run, dan docstring kini
  menyatakan batasan ini secara eksplisit.
- **EN:** Serato hot cues are **NOT yet supported**. For Serato, CuePilot
  writes only BPM/key + the `CUEPILOT_CUES` tag on the audio file and
  exports `.cueplan.json`. The Serato Markers / DatabaseV2 format is
  **not** written and **not** guessed, as it is encrypted and unverified.
  Apply results, dry-runs, and docstrings now state this explicitly.

#### K-2 — Rekordbox: apply idempotent, verifikasi berbasis set / Idempotent Rekordbox apply, set-based verification
- **ID:** Apply hot cue Rekordbox kini idempotent per slot: hapus + sisip
  per `ContentID` + `CueNumber` dalam satu transaksi, UUID stabil per
  lagu + slot (`uuid5`). Verifikasi membandingkan pasangan
  `(CueNumber, CueTiming)` sebagai set, bukan sekadar jumlah baris, dan
  melaporkan cue yang hilang, duplikat, atau ekstra. Apply dua kali
  atau menimpa cue manual pada slot yang sama tidak lagi menimbulkan
  duplikat. Cue manual pada slot di luar rencana dipertahankan dan
  dilaporkan sebagai ekstra, bukan dihapus diam-diam.
- **EN:** Rekordbox hot-cue apply is now idempotent per slot: delete +
  insert per `ContentID` + `CueNumber` in a single transaction, with a
  stable UUID per track + slot (`uuid5`). Verification compares
  `(CueNumber, CueTiming)` pairs as a set, not just a row count, and
  reports missing, duplicated, or extra cues. Re-applying, or
  overwriting a manual cue on the same slot, no longer duplicates.
  Manual cues on slots outside the plan are preserved and reported as
  extra, not silently deleted.

#### K-3 — Backup dan rollback benar-benar memulihkan / Backup and rollback that actually restore
- **ID:** Backup library **tidak lagi dihapus otomatis** setelah apply
  berhasil (`_commit` mempertahankannya dan mengembalikan lokasi
  backup, sehingga rollback tetap bisa dipakai setelah commit).
  Backup direktori (mis. `Subcrates`) kini disalin rekursif dengan hash
  direktori, bukan dilewati diam-diam. File audio yang tag-nya ditulis
  kini masuk target backup Serato, dan rollback memulihkan tag audio,
  dengan fallback ke salinan terbaru di `.cuepilot-backup`.
- **EN:** Library backups are **no longer auto-deleted** after a
  successful apply (`_commit` retains them and returns the backup
  location, so rollback remains possible after commit). Directory
  backups (e.g. `Subcrates`) are now copied recursively with a
  directory hash, not silently skipped. Audio files whose tags were
  written are now backup targets for Serato, and rollback restores
  audio tags, falling back to the newest copy in `.cuepilot-backup`.

#### K-4 — Rekordbox: pencocokan path dan kunci database / Rekordbox path matching and database locking
- **ID:** Pencocokan lagu dinormalisasi (casefold, penyatuan separator
  `\`/`/`, pencocokan nama file + akhir path), bukan lagi gabungan
  string path yang rapuh. Tulis ditolak bila Rekordbox tampak masih
  terbuka: file `-wal`/`-shm` ada atau probe lock SQLite gagal.
  **Tutup Rekordbox sebelum apply.**
- **EN:** Track matching is normalised (casefold, separator unification,
  filename + path-suffix matching), not fragile string concatenation.
  Writes are refused while Rekordbox still looks open: `-wal`/`-shm`
  files present or a SQLite lock probe failing. **Close Rekordbox
  before applying.**

### Tinggi / High

#### T-1 — Benchmark: klaim lama dicabut, bug metrik diperbaiki / Benchmark: old claims withdrawn, metric bug fixed
- **ID:** Klaim akurasi cue lama (`edm_128` <0,05 dtk, `edm_140`
  <0,10 dtk) **dicabut**. Data terukur terakhir di
  `benchmarks/results.json`: `edm_128` 0,647 dtk, `edm_140` 7,434 dtk
  (boundary F1 0,667), `techno_132` 0,019 dtk, `hiphop_95` 0,019 dtk.
  Penyebab utama `edm_140` adalah klasifikasi section dan pergeseran
  boundary pada 140 BPM, bukan beatgrid (beat/downbeat F1 1,00).
  **Klasifikasi/boundary DSP `edm_140` belum diperbaiki** — mengubah
  ambang tanpa bisa menjalankan benchmark ber-librosa akan menjadi
  perbaikan tak terverifikasi. Satu bug metrik yang terbukti
  diperbaiki: pencocokan label benchmark kini memetakan BREAK →
  BREAKDOWN dan VOCAL → VOCAL/CHORUS, sehingga cue BREAK tidak lagi
  hilang diam-diam dari `mean_timing_err`. `results.json` tidak diubah
  tanpa data run baru.
- **EN:** Previous cue-accuracy claims (`edm_128` <0.05 s, `edm_140`
  <0.10 s) are **withdrawn**. Last measured data in
  `benchmarks/results.json`: `edm_128` 0.647 s, `edm_140` 7.434 s
  (boundary F1 0.667), `techno_132` 0.019 s, `hiphop_95` 0.019 s.
  The main cause for `edm_140` is section classification and boundary
  drift at 140 BPM, not the beatgrid (beat/downbeat F1 1.00).
  **The `edm_140` DSP classification/boundary failure is not yet
  fixed** — tuning thresholds without a librosa benchmark run would be
  an unverified fix. One proven metric bug is fixed: benchmark label
  matching now maps BREAK → BREAKDOWN and VOCAL → VOCAL/CHORUS, so
  BREAK cues no longer silently vanish from `mean_timing_err`.
  `results.json` was not changed without new run data.

#### T-2 — Slot CHORUS tidak lagi menabrak OUTRO / CHORUS slot no longer collides with OUTRO
- **ID:** CHORUS kini terpetakan eksplisit. Karena skema 8 slot tidak
  muat untuk 8 tipe section + START, CHORUS berbagi slot 3 dengan
  VOCAL **by design** (keduanya `CueType.VOCAL`, first-wins), dan
  tabrakan slot tidak lagi dibuang diam-diam melainkan dicatat di
  reason cue yang bertahan. Tipe yang tidak terpetakan tidak lagi
  jatuh diam-diam ke slot 8.
- **EN:** CHORUS is now mapped explicitly. Because the 8-slot schema
  cannot fit 8 section types plus START, CHORUS shares slot 3 with
  VOCAL **by design** (both `CueType.VOCAL`, first-wins), and slot
  collisions are no longer dropped silently but recorded in the
  surviving cue's reasons. Unmapped types no longer fall back silently
  to slot 8.

#### T-3 — Genre: label jujur, bukan deteksi palsu / Genre: honest labels, not fake detection
- **ID:** CuePilot **tidak memiliki detektor genre**. `genre_detected`
  kini sengaja kosong dan `genre_confidence` 0,0. Rekaman jujur adalah
  field kanonik baru `genre_profile_used` (diekspor sebagai
  `genreProfileUsed`) plus blok `extras["genre"]`. Profil yang tidak
  dikenal dan jatuh ke `open_format` kini memberi peringatan keras,
  bukan diam-diam.
- **EN:** CuePilot **has no genre detector**. `genre_detected` is now
  intentionally empty and `genre_confidence` 0.0. The honest record is
  the new canonical field `genre_profile_used` (exported as
  `genreProfileUsed`) plus the `extras["genre"]` block. An unknown
  profile falling back to `open_format` now emits a hard warning,
  not silence.

#### T-4 — Beatgrid: batasan 4/4 dan tempo global dinyatakan / Beatgrid: 4/4 and global-tempo limits stated
- **ID:** Grid hanya mendukung meter 4/4 yang diasumsikan dan tempo
  global — belum ada deteksi meter maupun pelacakan tempo variabel,
  dan keduanya **tidak diklaim**. Field `BeatGrid` baru
  (`meter`, `meter_detected`, `tempo_model`) menyatakan batasan ini.
  Klasifikasi subdivisi kini informasional saja dan tidak mengubah
  snapping; klaim adaptif dihapus. Peta hint subdivisi diperbaiki
  (termasuk `indobounce`/`bassline_bounce`), floor confidence 0,35
  dihapus, dan dua bug bukti-nol diperbaiki.
- **EN:** The grid only supports an assumed 4/4 meter and a global
  tempo — no meter detection or variable-tempo tracking yet, and
  neither is claimed. New `BeatGrid` fields (`meter`,
  `meter_detected`, `tempo_model`) state these limits. Subdivision
  classification is now informational only and does not change
  snapping; the adaptive claim was removed. The subdivision hint map
  was fixed (including `indobounce`/`bassline_bounce`), the 0.35
  confidence floor was removed, and two zero-evidence bugs were fixed.

#### T-5 — Confidence dan readiness yang tidak menipu / Non-deceptive confidence and readiness
- **ID:** Confidence beatgrid kini dari konsistensi interval antar-beat
  dan kekuatan onset (`beatgrid_confidence`), bukan dari jumlah beat.
  Fallback klasifikasi buta DROP diganti `SectionType.UNKNOWN` yang
  eksplisit (confidence dicap 0,4, dan generator tidak membuat cue
  untuk section UNKNOWN). SECOND_DROP hanya bila ada section kontras
  di antaranya. Koreksi manual tidak lagi menurunkan skor Gig
  Readiness: tanpa koreksi komponen verifikasi 0,8 (hanya DSP, belum
  diverifikasi manusia), tiap koreksi +0,05 hingga 1,0. Konsekuensinya,
  rencana borderline yang belum diverifikasi kini benar masuk
  `needs_review`, bukan `ready`.
- **EN:** Beatgrid confidence now comes from inter-beat interval
  consistency and onset strength (`beatgrid_confidence`), not beat
  count. The blind DROP classification fallback is replaced by an
  explicit `SectionType.UNKNOWN` (confidence capped at 0.4, and the
  generator creates no cue for UNKNOWN sections). SECOND_DROP now
  requires a contrasting section in between. Manual corrections no
  longer lower the Gig Readiness score: no corrections = 0.8
  verification component (DSP only, unverified), each correction
  +0.05 up to 1.0. Consequently, an unverified borderline plan now
  correctly lands in `needs_review`, not `ready`.

#### T-6 — Metadata: jangan timpa tag bagus dengan nilai kosong / Metadata: never overwrite good tags with empty values
- **ID:** BPM ≤ 0 (atau tak ter-parse) dan key kosong/blank **tidak
  pernah ditulis**. Field yang dilewati dicatat di `skipped_fields`
  pada hasil apply, sehingga tag BPM/key yang sudah benar tidak
  rusak diam-diam.
- **EN:** BPM ≤ 0 (or unparseable) and blank keys are **never
  written**. Skipped fields are recorded in `skipped_fields` in the
  apply result, so good existing BPM/key tags are not silently
  destroyed.

#### T-7 — Stem: label backend yang benar-benar jalan / Stems: label the backend that actually ran
- **ID:** Semua stem di-clip ke [-1, 1]. Separasi DSP dinyatakan
  sebagai aproksimasi filter, bukan separasi ML, dan confidence
  section dari jalur DSP dikurangi 0,15; pipeline menerapkan penalti
  yang sama ke cue dari section fallback. Laporan tidak pernah lagi
  menulis `engine=onnx` saat yang jalan adalah DSP — labelnya
  `dsp-fallback` dengan alasan fallback yang eksplisit.
- **EN:** All stems are clipped to [-1, 1]. DSP separation is stated
  as a filter approximation, not ML separation, and section
  confidence from the DSP path is reduced by 0.15; the pipeline
  applies the same penalty to cues from fallback sections. Reports
  never again write `engine=onnx` when DSP actually ran — the label
  is `dsp-fallback` with an explicit fallback reason.

#### T-8 — Batch/CLI: perintah yang dijanjikan kini jalan / Batch/CLI: promised commands now work
- **ID:** Subcommand `batch create`, `batch run <job-id>`, dan
  `batch status <job-id>` kini benar-benar jalan; bentuk legacy
  `batch <folder>` tetap didukung. Tanpa `--out`, cueplan diekspor
  ke folder `cueplans/` di direktori job. Track DONE dilewati,
  track gagal dicoba ulang sampai `--retries`, dan track yang
  dilewati dilaporkan beserta alasannya. `worker_limit` divalidasi
  dan ditegakkan (`None`/1 = sekuensial, ≥2 = ThreadPoolExecutor
  terbatas dengan penulisan state tetap berurutan), dan `_JobLock`
  kini benar-benar dipakai per path job.
- **EN:** Subcommands `batch create`, `batch run <job-id>`, and
  `batch status <job-id>` now actually work; the legacy
  `batch <folder>` form still works. Without `--out`, cueplans are
  exported to the job directory's `cueplans/` folder. DONE tracks are
  skipped, failed tracks are retried up to `--retries`, and skipped
  tracks are reported with their reason. `worker_limit` is validated
  and enforced (`None`/1 = sequential, ≥2 = bounded ThreadPoolExecutor
  with state writes kept in original order), and `_JobLock` is now
  actually used per job path.

#### T-9 — Packaging dan desktop: dependensi dan build dinyatakan / Packaging and desktop: dependencies and build stated
- **ID:** `pyproject.toml` kini memakai `find_packages` (mencakup
  `core.stems` dan `core.learn` yang sebelumnya hilang), dengan extra
  `desktop` (FastAPI, uvicorn, pywebview) dan `onnx` (onnxruntime),
  bukan dependensi wajib. `scripts/build_web.sh` (baru) membangun
  `apps/desktop/web/dist/` (dist tidak ikut repo dan wajib dibangun
  di checkout tujuan). `shell.py` kini mendelegasikan ke `entry.py`
  yang kanonik, bukan duplikat implementasi.
- **EN:** `pyproject.toml` now uses `find_packages` (covering
  `core.stems` and `core.learn`, previously missing), with `desktop`
  (FastAPI, uvicorn, pywebview) and `onnx` (onnxruntime) extras, not
  mandatory dependencies. New `scripts/build_web.sh` builds
  `apps/desktop/web/dist/` (dist is not committed and must be built
  in the target checkout). `shell.py` now delegates to canonical
  `entry.py` instead of duplicating the implementation.

### Sedang / Medium

#### S-1 — Edit dan belajar: tidak hilang diam-diam, tervalidasi, di-snap / Edits and learning: not silently lost, validated, re-snapped
- **ID:** Sidecar edit yang rusak kini dikembalikan sebagai objek
  `corrupt=True` dengan warning, bukan `None` yang ditelan; pipeline
  tidak menerapkannya dan memberi peringatan. `apply_edits_to_cues`
  memvalidasi slot 1–8 (ditolak) dan waktu 0 ≤ time ≤ duration
  (di-clamp dengan notifikasi); pipeline kini meneruskan durasi dan
  notifikasi tersebut. Geseran pola belajar dibatasi dari 15% durasi
  menjadi **maksimal 4 detik** dan di-snap ulang ke beatgrid
  terdekat; pipeline kini meneruskan `beat_times`/`downbeat_times`
  sehingga snap tersebut aktif di produksi (sebelumnya tidak
  ter-wire). File preferensi yang rusak di-backup sebelum ditimpa.
- **EN:** Corrupt edit sidecars are now returned as `corrupt=True`
  objects with a warning, not swallowed into `None`; the pipeline
  does not apply them and warns. `apply_edits_to_cues` validates
  slots 1–8 (rejected) and time 0 ≤ time ≤ duration (clamped with
  notification); the pipeline now passes duration and notifications.
  Learned-pattern shifts are capped from 15% of duration to **at most
  4 seconds** and re-snapped to the nearest beatgrid; the pipeline
  now passes `beat_times`/`downbeat_times` so that snap is active in
  production (previously unwired). Corrupt preference files are
  backed up before being overwritten.

#### S-2 — Loop: ujung dari downbeat grid / Loops: ends from the downbeat grid
- **ID:** Durasi loop kini `bar × ketuk-per-bar × durasi ketuk`
  (bukan rumus `BARS_PER_PHRASE` yang menyesatkan), dan ujung loop
  diambil dari downbeat grid bila tersedia; bila tidak, fallback
  ditandai `endFromGrid=False`.
- **EN:** Loop duration is now `bars × beats-per-bar × beat
  duration` (not the misleading `BARS_PER_PHRASE` formula), and loop
  ends come from grid downbeats when available; otherwise the
  fallback is flagged `endFromGrid=False`.

#### S-3 — QC: jendela penuh, label dBFS RMS, silence simetris / QC: full-track windows, dBFS RMS label, symmetric silence
- **ID:** Analisis rolloff frekuensi tinggi kini memakai hingga 32
  jendela yang tersebar di seluruh lagu, bukan hanya ~12 detik
  pertama. Metrik loudness kini **dBFS RMS, bukan LUFS** (tanpa
  K-weighting/gating); kunci `lufs` lama dihapus, alias parameter
  lama dipertahankan sementara dengan `DeprecationWarning`. Edge
  silence kini simetris: tidak ada frame audible → `(durasi, durasi)`.
- **EN:** High-frequency rolloff analysis now uses up to 32 windows
  spread across the whole track, not just the first ~12 seconds. The
  loudness metric is now **dBFS RMS, not LUFS** (no K-weighting/
  gating); the old `lufs` key was removed, with the old parameter
  alias kept temporarily with a `DeprecationWarning`. Edge silence
  is now symmetric: no audible frame → `(duration, duration)`.

#### S-4 — Privasi: local-first, cloud hanya opt-in eksplisit / Privacy: local-first, cloud only explicit opt-in
- **ID:** Default LLM berubah dari cloud menjadi `local-light`.
  Mode cloud hanya aktif dengan persetujuan eksplisit
  (`explicit_consent=True` atau env `CUEPILOT_LLM_CLOUD_CONSENT`);
  tanpa itu, permintaan cloud jatuh ke local-light dengan
  `requires_explicit_consent=True` dan peringatan. Endpoint desktop
  `/api/privacy/resolve` kini menerima field `explicit_consent`
  (UI wajib memakai checkbox konfirmasi untuk mengisinya).
- **EN:** The LLM default changed from cloud to `local-light`.
  Cloud mode is only active with explicit consent
  (`explicit_consent=True` or env `CUEPILOT_LLM_CLOUD_CONSENT`);
  without it, cloud requests fall back to local-light with
  `requires_explicit_consent=True` and a warning. The desktop
  `/api/privacy/resolve` endpoint now accepts an `explicit_consent`
  field (the UI must use a confirmation checkbox to set it).

#### S-5 — Scanner, settings, planner, kompatibilitas / Scanner, settings, planner, compatibility
- **ID:** ID track scanner kini case-sensitive (path apa adanya +
  normalisasi separator saja, tanpa `lower()`); file yang hanya
  beda kapitalisasi tidak lagi berbagi ID (perubahan perilaku yang
  disengaja: ID lama berbasis lower-case akan berubah). File
  `settings.json` yang rusak dipindah ke
  `settings.json.corrupt-<waktu>` dengan warning, bukan ditimpa
  diam-diam. Planner tidak lagi memutasi dict input caller, dan
  flag `_must_include` hanya untuk track yang benar-benar
  must-include. `key_to_camelot` menerima nama nada **dan** kode
  Camelot (mis. `8A`), jarak Camelot memakai modulo 24
  (12B→1A = 1), definisi energy boost diperbaiki menjadi satu nomor
  Camelot lebih tinggi dengan huruf sama (8A→9A), dan rekomendasi
  transisi OUTRO→INTRO hanya ditampilkan bila kedua cue benar-benar
  ada (bila tidak, `suggested_transition=None`).
- **EN:** Scanner track IDs are now case-sensitive (path as-is plus
  separator normalisation only, no `lower()`); files differing only
  by capitalisation no longer share an ID (intentional behaviour
  change: old lower-case-based IDs will change). A corrupt
  `settings.json` is moved aside to
  `settings.json.corrupt-<timestamp>` with a warning, not silently
  overwritten. The planner no longer mutates the caller's input
  dicts, and the `_must_include` flag is only set for tracks that
  truly are must-include. `key_to_camelot` accepts note names **and**
  Camelot codes (e.g. `8A`), Camelot distance uses modulo 24
  (12B→1A = 1), energy boost is corrected to one Camelot number
  higher, same letter (8A→9A), and an OUTRO→INTRO transition is only
  recommended when both cues actually exist (otherwise
  `suggested_transition=None`).

#### S-6 — File rusak/pendek/sample rate: error jelas, test ditambah / Corrupt/short/sample-rate files: clear errors, tests added
- **ID:** `librosa_engine.load()` kini memvalidasi sebelum decode
  (path kosong, file hilang, 0 byte, ekstensi tak didukung dengan
  daftar yang didukung, exception decoder dibungkus, sampel kosong /
  sample rate invalid / durasi 0 ditolak). Pipeline membungkus
  exception decode menjadi `AudioDecodeError` berisi nama file +
  alasan + saran, dan menolak hasil decode kosong/durasi 0. Test
  baru mencakup file corrupt, kosong, lagu sangat pendek (5–10
  detik, tanpa cue di luar durasi), mono vs stereo, dan sample rate
  44100/48000 (bookkeeping durasi dari sample rate asli).
- **EN:** `librosa_engine.load()` now validates before decoding
  (empty path, missing file, 0 bytes, unsupported extension with the
  supported list named, decoder exceptions wrapped, empty samples /
  invalid sample rate / zero duration rejected). The pipeline wraps
  decode exceptions into `AudioDecodeError` with file name + reason
  + suggestion, and rejects empty/zero-duration decodes. New tests
  cover corrupt, empty, very short (5–10 s, no cue outside duration),
  mono vs stereo, and 44100/48000 sample-rate files (duration
  bookkeeping from the original sample rate).

### Rendah / Low

#### R-1 s.d. R-6 — Subdivisi, cache, desktop, cap, duplikasi entry, retensi backup / Subdivision, cache, desktop, caps, duplicate entry, backup retention
- **ID:** Peta hint subdivisi diperbaiki dan floor confidence dihapus
  (bukti nol → `unknown`, 0,0). Kunci cache kini mencakup parameter
  analisis (sample rate/hop); pipeline kini meneruskan parameter
  tersebut saat load/store, sehingga entri lama tanpa parameter atau
  dengan parameter berbeda menjadi miss, bukan dipakai diam-diam.
  CORS desktop dibatasi ke origin loopback (regex
  localhost/127.0.0.1, server tetap bind 127.0.0.1), dan sleep acak
  di stream penulisan metadata dihapus (stage kini diemisikan
  langsung; waktu stream mencerminkan kerja nyata). Cap confidence
  **tidak** disatukan antara jalur klasik (0,7 untuk lokal-custom)
  dan jalur stem (hingga 0,98): jalur stem tidak menerima input
  strategi, sehingga membatasi semua section stem ke 0,7 akan
  menghukum genre non-lokal secara keliru — kedua jalur
  didokumentasikan sebagai tidak langsung sebanding, dan penalti
  fallback DSP 0,15 tetap berlaku terpisah. Entry desktop
  digabungkan (`shell.py` mendelegasikan ke `entry.py` yang
  kanonik). Retensi backup metadata: fungsi baru `prune_backups()`
  di `core/metadata.py` menyimpan backup terbaru per track
  (default 3 terbaru, tidak pernah menghapus backup terbaru walau
  sudah tua) dan hanya menghapus file di dalam folder
  `.cuepilot-backup` milik track tersebut; fungsi ini **tidak**
  dijalankan otomatis pada setiap penulisan, agar rollback terbaru
  tidak pernah terhapus tanpa sengaja — jalankan secara eksplisit
  saat perawatan library.
- **EN:** The subdivision hint map was fixed and the confidence floor
  removed (zero evidence → `unknown`, 0.0). Cache keys now include
  analysis parameters (sample rate/hop); the pipeline now passes
  those parameters on load/store, so old entries without parameters
  or with different parameters miss instead of being silently
  reused. Desktop CORS is restricted to loopback origins (regex
  localhost/127.0.0.1, server still binds 127.0.0.1), and the random
  sleep in the metadata-write stream was removed (stages are now
  emitted directly; stream time reflects real work). Confidence caps
  were **not** unified between the classic path (0.7 for lokal-custom)
  and the stem path (up to 0.98): the stem path receives no strategy
  input, so capping all stem sections at 0.7 would wrongly penalise
  non-lokal genres — both paths are documented as not directly
  comparable, and the separate DSP-fallback penalty of 0.15 still
  applies. Desktop entries were merged (`shell.py` delegates to
  canonical `entry.py`). Metadata backup retention: new
  `prune_backups()` in `core/metadata.py` keeps the newest backups
  per track (default newest 3, never deleting the newest backup even
  if old) and only deletes files inside that track's own
  `.cuepilot-backup` folder; it is **not** run automatically on
  every write, so the newest rollback is never deleted by accident —
  run it explicitly during library maintenance.

### Perubahan perilaku yang disengaja (breaking changes) / Intentional behaviour changes (breaking)
- **ID:** `genre_detected` kosong / `genre_confidence` 0,0 (bukan lagi
  nama profil + confidence beatgrid); field baru `genre_profile_used`.
  Default LLM cloud → local-light, cloud butuh consent eksplisit.
  Metrik `lufs` → `dbfsRms` (alias lama sementara dengan warning).
  Cap geseran belajar 15% durasi → maksimal 4 detik + snap beatgrid.
  `load_edits` corrupt kini mengembalikan objek, bukan `None`.
  ID scanner case-sensitive (ID lama berubah). Backup library tidak
  lagi dihapus otomatis setelah apply berhasil. Rencana borderline
  yang belum diverifikasi kini `needs_review`, bukan `ready`.
  `MAX_SHIFT_RATIO` dihapus, diganti `MAX_LEARNED_SHIFT_SECONDS`.
- **EN:** `genre_detected` empty / `genre_confidence` 0.0 (no longer
  profile name + beatgrid confidence); new `genre_profile_used`
  field. LLM default cloud → local-light, cloud needs explicit
  consent. Metric `lufs` → `dbfsRms` (old alias temporarily with
  warning). Learned shift cap 15% of duration → at most 4 seconds +
  beatgrid snap. Corrupt `load_edits` now returns an object, not
  `None`. Scanner IDs case-sensitive (old IDs change). Library
  backups are no longer auto-deleted after successful apply.
  Unverified borderline plans are now `needs_review`, not `ready`.
  `MAX_SHIFT_RATIO` removed, replaced by
  `MAX_LEARNED_SHIFT_SECONDS`.

## Yang TIDAK diperbaiki (jujur) / What was NOT fixed (honest)
- **ID:** (1) Hot cue Serato DatabaseV2/Markers — format terenkripsi
  dan tidak terverifikasi; mengarangnya berisiko merusak library
  pengguna. (2) Klasifikasi section dan boundary `edm_140` pada
  140 BPM — butuh run benchmark ber-librosa; ambang tidak diubah
  tanpa bukti run. (3) Deteksi meter dan pelacakan tempo variabel —
  tidak diimplementasikan, hanya dinyatakan batasannya. (4) Snapping
  adaptif subdivisi — tidak diimplementasikan, klaimnya dihapus.
  (5) Inferensi ONNX stem end-to-end — tidak ada model di
  `models/stems/` dan tidak ada onnxruntime/librosa di lingkungan
  verifikasi, sehingga jalur ONNX nyata belum dijalankan.
  (6) Deteksi "Rekordbox tertutup" 100% — tidak ada API resmi;
  yang diberikan adalah gabungan terbaik yang dapat diuji (sidecar
  + probe lock + proses). (7) Penghapusan total cue manual Rekordbox
  di luar rencana — tidak diasumsikan; upsert per slot
  dipertahankan, penghapusan total butuh persetujuan eksplisit.
  (8) File-lock antar-proses untuk batch — `_JobLock` kini nyata
  dalam satu proses, bukan file-lock antar-proses (butuh dependensi
  baru yang tidak ditambahkan diam-diam). (9) Suite test penuh
  ber-librosa dan benchmark penuh — tidak dapat dijalankan di
  lingkungan verifikasi ini (lihat Verifikasi di bawah); angka
  benchmark baru tidak diklaim.
- **EN:** (1) Serato DatabaseV2/Markers hot cues — format encrypted
  and unverified; inventing it risks corrupting user libraries.
  (2) `edm_140` section classification and boundary at 140 BPM —
  needs a librosa benchmark run; thresholds were not tuned without
  run evidence. (3) Meter detection and variable-tempo tracking —
  not implemented, only their limits stated. (4) Adaptive
  subdivision snapping — not implemented, its claim removed.
  (5) ONNX stem inference end-to-end — no model in `models/stems/`
  and no onnxruntime/librosa in the verification environment, so the
  real ONNX path was not run. (6) 100% "Rekordbox closed" detection —
  no official API; the best testable combination (sidecar + lock
  probe + process) is provided. (7) Total deletion of manual
  Rekordbox cues outside the plan — not assumed; per-slot upsert is
  kept, total deletion needs explicit approval. (8) Cross-process
  file lock for batch — `_JobLock` is now real within one process,
  not a cross-process file lock (would need a new dependency, not
  added silently). (9) Full librosa test suite and full benchmark —
  could not be run in this verification environment (see Verification
  below); new benchmark figures are not claimed.

## Verifikasi / Verification
- **ID:** `py_compile` untuk semua file Python yang diubah: lolos.
  Suite yang bisa jalan tanpa librosa (venv sekali pakai di `/tmp`
  dengan pytest, mutagen, soundfile, numpy, scipy): **189 passed**
  pada set test yang dipilih (termasuk semua file `test_*_fix.py`
  baru, test konsolidasi, dan test lama yang tidak butuh librosa:
  integrations, metadata, compatibility, planner, settings, scanner,
  quality, learn, privacy, loops, batch-fix, slot-collision,
  beatgrid-fix, pipeline-fix, decode-fix, edits-fix, stems-fix).
  **Tidak dijalankan:** suite penuh yang butuh librosa
  (`test_beatgrid.py`, `test_pipeline.py`, test bands/pipeline
  ber-audio, dan sejenisnya) serta `python -m benchmarks.benchmark`
  penuh, karena librosa tidak tersedia
  di lingkungan verifikasi ini. Dua kegagalan awal pada run
  konsolidasi (fake `apply_learned_patterns` di test pipeline yang
  tidak menerima kwarg baru, dan test readiness bucket lama yang
  menegaskan kebijakan lama) sudah diperbaiki dengan memperbarui
  fake test dan test tersebut agar sesuai kebijakan baru yang
  terdokumentasi, bukan dengan melemahkan ambang. Build web
  desktop dicoba oleh fixer batch dan berhasil di lingkungannya
  (node v24, npm 10; `dist/` tetap gitignored dan wajib dibangun
  ulang di checkout tujuan via `scripts/build_web.sh`).
- **EN:** `py_compile` for all changed Python files: passed.
  The librosa-free runnable set (throwaway venv in `/tmp` with
  pytest, mutagen, soundfile, numpy, scipy): **189 passed** on the
  selected test set (including all new `test_*_fix.py` files, the
  consolidation test, and legacy tests not needing librosa:
  integrations, metadata, compatibility, planner, settings, scanner,
  quality, learn, privacy, loops, batch-fix, slot-collision,
  beatgrid-fix, pipeline-fix, decode-fix, edits-fix, stems-fix).
  **Not run:** the full librosa-dependent suite (`test_beatgrid.py`,
  `test_pipeline.py`, audio-bearing bands/pipeline tests, and
  similar) and the full `python -m benchmarks.benchmark`, because
  librosa was unavailable in this verification environment. Two initial consolidation-run failures
  (pipeline-test fakes for `apply_learned_patterns` not accepting
  the new kwarg, and the old readiness-bucket test asserting the old
  policy) were fixed by updating those test fakes/tests to the new
  documented policy, not by weakening thresholds. The desktop web
  build was tried by the batch fixer and succeeded in its environment
  (node v24, npm 10; `dist/` remains gitignored and must be rebuilt
  in the target checkout via `scripts/build_web.sh`).

## Langkah berikut sebelum fork/git / Next steps before fork/git
- **ID:** Jalankan di lingkungan ber-librosa: suite pytest penuh,
  `python -m benchmarks.benchmark` (perbarui `results.json` dan
  README hanya dari hasil run itu), dan uji manual sekali jalan:
  apply Rekordbox pada DB tiruan lalu buka di Rekordbox, serta
  apply Serato lalu periksa BPM/key/tag di Serato (hot cue Serato
  tetap tidak diharapkan). Setelah itu baru fork/commit dengan
  changelog ini.
- **EN:** Run in a librosa environment: the full pytest suite,
  `python -m benchmarks.benchmark` (update `results.json` and README
  only from that run's results), and one manual end-to-end check:
  Rekordbox apply on a test DB then open it in Rekordbox, and Serato
  apply then check BPM/key/tags in Serato (Serato hot cues are still
  not expected). Only then fork/commit with this changelog.
