import { createContext, useContext, useState, ReactNode } from "react";

export type Lang = "id" | "en";

const DICT: Record<string, { id: string; en: string }> = {
  subtitle: { id: "Asisten hot cue untuk DJ", en: "Hot cue assistant for DJs" },

  welcomeTitle: {
    id: "Siapkan musikmu, tinggal pilih folder",
    en: "Ready to prep your music, just pick a folder",
  },
  welcomeSub: {
    id: "CuePilot memindai lagu-lagumu, menemukan bagian penting (intro, drop, breakdown, vokal), lalu membuat hot cue otomatis. Kamu tinggal dengar dan simpan.",
    en: "CuePilot scans your tracks, finds the important parts (intro, drop, breakdown, vocals), and creates hot cues automatically. Just listen and save.",
  },
  pickFolderButton: { id: "Pilih Folder Musik", en: "Pick Music Folder" },
  pickFolder: { id: "Pilih folder", en: "Pick a folder" },
  orTypePath: { id: "atau ketik alamat folder di bawah", en: "or type the folder path below" },
  folderPlaceholder: {
    id: "Contoh: D:\\Music\\EDM",
    en: "Example: D:\\Music\\EDM",
  },
  scanButton: { id: "Pindai Folder", en: "Scan Folder" },
  scanning: { id: "Memindai…", en: "Scanning…" },
  scanningTitle: { id: "Memindai folder…", en: "Scanning folder…" },
  scanningSub: {
    id: "Mencari lagu & membaca info lagu",
    en: "Finding tracks & reading track info",
  },
  whatIsHotcueTitle: { id: "Apa itu hot cue?", en: "What is a hot cue?" },
  whatIsHotcueDesc: {
    id: "Hot cue adalah penanda titik penting di dalam lagu (contoh: saat drop dimulai, bagian vokal, atau breakdown). Dengan hot cue, kamu bisa langsung lompat ke bagian itu saat DJ-ing — tanpa perlu mencari-cari lagi.",
    en: "A hot cue is a marker for an important point in a track (like when the drop starts, a vocal part, or the breakdown). With hot cues you can jump straight to that part while DJ-ing — no more hunting.",
  },
  howItWorksTitle: { id: "Cara kerjanya", en: "How it works" },
  how1Title: { id: "Pilih folder", en: "Pick a folder" },
  how1Desc: { id: "Folder tempat musikmu berada", en: "The folder where your music lives" },
  how2Title: { id: "Pilih lagu", en: "Pick a track" },
  how2Desc: { id: "CuePilot menganalisis otomatis", en: "CuePilot analyzes automatically" },
  how3Title: { id: "Dengar & simpan", en: "Listen & save" },
  how3Desc: { id: "Setujui hot cue, lalu simpan ke file", en: "Approve the cues, then save to file" },

  tracksTitle: { id: "Lagu di folder ini", en: "Tracks in this folder" },
  trackCount: { id: "lagu", en: "tracks" },
  noTracks: {
    id: "Tidak ada lagu yang ditemukan di folder ini.",
    en: "No tracks found in this folder.",
  },
  noTracksHint: {
    id: "Pastikan folder berisi file MP3, WAV, FLAC, AIFF, atau OGG.",
    en: "Make sure the folder contains MP3, WAV, FLAC, AIFF, or OGG files.",
  },
  analyzeAllButton: {
    id: "Buat Hot Cue untuk Semua Lagu",
    en: "Create Hot Cues for All Tracks",
  },
  analyzeAllHint: {
    id: "CuePilot memproses semua lagu secara berurutan. Kamu bisa membatalkannya kapan saja.",
    en: "CuePilot processes all tracks one by one. You can cancel anytime.",
  },
  pickATrack: { id: "Pilih satu lagu untuk mulai", en: "Pick a track to start" },
  pickATrackHint: {
    id: "Klik lagu apa pun untuk melihat hot cue-nya.",
    en: "Click any track to see its hot cues.",
  },
  backToFolder: { id: "Ganti Folder", en: "Change Folder" },

  analyzingTitle: { id: "Menganalisis lagu…", en: "Analyzing track…" },
  analyzingSub: {
    id: "Menemukan bagian penting & membuat hot cue",
    en: "Finding the important parts & creating hot cues",
  },
  readyTitle: { id: "Siap memulai", en: "Ready when you are" },
  readySub: {
    id: "Pilih folder musik dulu, lalu klik lagu untuk melihat hot cue-nya.",
    en: "Pick a music folder first, then click a track to see its hot cues.",
  },

  tempo: { id: "Tempo", en: "Tempo" },
  bpm: { id: "Tempo", en: "Tempo" },
  key: { id: "Kunci", en: "Key" },
  duration: { id: "Durasi", en: "Duration" },
  changeGenre: { id: "Ubah Jenis Musik", en: "Change Genre" },
  changeGenreHint: {
    id: "Jenis musik menentukan cara CuePilot menyusun hot cue. Pilih yang paling cocok.",
    en: "The genre decides how CuePilot builds hot cues. Pick the closest match.",
  },

  play: { id: "Putar", en: "Play" },
  pause: { id: "Jeda", en: "Pause" },
  stop: { id: "Berhenti", en: "Stop" },
  listenHere: { id: "Dengar bagian ini", en: "Listen to this part" },

  hotCuesTitle: { id: "Hot Cue", en: "Hot Cues" },
  hotCuesHint: {
    id: "Klik penanda untuk langsung mendengar bagian itu. Hot cue otomatis diberi warna berbeda.",
    en: "Click a marker to jump straight to that part. Auto cues use different colors.",
  },
  cueTypeStart: { id: "Awal lagu", en: "Track start" },
  cueTypeIntro: { id: "Intro", en: "Intro" },
  cueTypeMixIn: { id: "Mix in", en: "Mix in" },
  cueTypeVocal: { id: "Vokal", en: "Vocal" },
  cueTypeHook: { id: "Hook", en: "Hook" },
  cueTypeMelody: { id: "Melodi", en: "Melody" },
  cueTypeGroove: { id: "Groove", en: "Groove" },
  cueTypePreGroove: { id: "Pra-groove", en: "Pre-groove" },
  cueTypeBuild: { id: "Build", en: "Build" },
  cueTypePreDrop: { id: "Pra-drop", en: "Pre-drop" },
  cueTypeDrop: { id: "Drop", en: "Drop" },
  cueTypeClimax: { id: "Puncak", en: "Climax" },
  cueTypeTransition: { id: "Transisi", en: "Transition" },
  cueTypeChorus: { id: "Reff", en: "Chorus" },
  cueTypeBreakdown: { id: "Breakdown", en: "Breakdown" },
  cueTypeEnergyPeak: { id: "Puncak energi", en: "Energy peak" },
  cueTypeSecondDrop: { id: "Drop kedua", en: "Second drop" },
  cueTypeOutro: { id: "Outro", en: "Outro" },
  cueTypeMixOut: { id: "Mix out", en: "Mix out" },
  cueTypeAlternative: { id: "Alternatif", en: "Alternative" },
  cueTypeCustom: { id: "Manual", en: "Custom" },

  editCues: { id: "Edit Manual", en: "Manual Edit" },
  editCuesHint: {
    id: "Pindahkan hot cue dengan mengklik gelombang suara, atau hapus penanda yang tidak kamu suka.",
    en: "Move hot cues by clicking the waveform, or delete markers you don't like.",
  },
  doneEditing: { id: "Selesai Edit", en: "Done Editing" },
  save: { id: "Simpan", en: "Save" },
  saveCues: { id: "Simpan perubahan", en: "Save changes" },
  categories: { id: "Slot", en: "Slots" },
  clickToPlace: {
    id: "Klik gelombang suara untuk menempatkan penanda",
    en: "Click the waveform to place a marker",
  },
  saved: { id: "Tersimpan", en: "Saved" },
  lock: { id: "Kunci penanda ini (tetap saat analisis ulang)", en: "Lock this marker (keep on re-analysis)" },
  unlock: { id: "Buka kunci", en: "Unlock" },
  deleteCue: { id: "Hapus penanda", en: "Delete marker" },
  editHint: { id: "Klik gelombang suara untuk menempatkan penanda", en: "Click the waveform to place a marker" },

  saveToFileButton: { id: "Simpan ke File Musik", en: "Save to Music File" },
  saveToFileHint: {
    id: "Menulis tempo, kunci & hot cue ke file lagu (MP3/WAV/FLAC). File asli otomatis di-backup dulu — aman.",
    en: "Writes tempo, key & hot cues into the track file (MP3/WAV/FLAC). The original file is backed up first — safe.",
  },
  writtenOk: {
    id: "Hot cue berhasil disimpan ke file",
    en: "Hot cues saved to file",
  },
  writtenOkDetail: {
    id: "Tempo, kunci & {count} hot cue tersimpan. File asli sudah di-backup.",
    en: "Tempo, key & {count} hot cues saved. Original file was backed up.",
  },
  writtenFail: { id: "Gagal menyimpan", en: "Could not save" },
  close: { id: "Tutup", en: "Close" },
  writeMetadataHint: {
    id: "Simpan tempo, kunci & hot cue ke dalam file lagu",
    en: "Save tempo, key & hot cues into the track file",
  },
  writingTitle: { id: "Menyimpan…", en: "Saving…" },
  writingSub: {
    id: "Menulis tempo, kunci & hot cue ke file lagu",
    en: "Writing tempo, key & hot cues into the track file",
  },
  mst_read: { id: "Membaca file asli", en: "Reading original file" },
  mst_backup: { id: "Membuat cadangan file asli", en: "Backing up original file" },
  mst_write: { id: "Menulis tempo & kunci", en: "Writing tempo & key" },
  mst_cues: { id: "Menulis paket hot cue", en: "Writing hot cue plan" },
  mst_finalize: { id: "Menyelesaikan…", en: "Finalizing…" },
  mst_writeCuePlan: { id: "Membuat paket hot cue", en: "Building hot cue plan" },

  zoom: { id: "Zoom", en: "Zoom" },
  zoomIn: { id: "Perbesar", en: "Zoom in" },
  zoomOut: { id: "Perkecil", en: "Zoom out" },
  zoomReset: { id: "Reset zoom", en: "Reset zoom" },
  legend: { id: "Legenda", en: "Legend" },
  played: { id: "Sudah diputar", en: "Played" },
  unplayed: { id: "Belum diputar", en: "Unplayed" },
  playhead: { id: "Posisi putar", en: "Playhead" },
  hotcueLegend: { id: "Hot cue", en: "Hot cue" },

  sectionIntro: { id: "Intro", en: "Intro" },
  sectionVocal: { id: "Vokal", en: "Vocal" },
  sectionBuild: { id: "Build", en: "Build" },
  sectionDrop: { id: "Drop", en: "Drop" },
  sectionBreakdown: { id: "Breakdown", en: "Breakdown" },
  sectionChorus: { id: "Reff", en: "Chorus" },
  sectionSecondDrop: { id: "Drop kedua", en: "Second drop" },
  sectionOutro: { id: "Outro", en: "Outro" },

  chooseGenre: { id: "Pilih jenis musik", en: "Choose the genre" },
  genreHint: {
    id: "Jenis musik menentukan cara CuePilot menyusun hot cue. Pilih yang paling cocok agar hasilnya akurat.",
    en: "The genre decides how CuePilot builds hot cues. Pick the closest match for accurate results.",
  },

  learnedCount: { id: "Pola dipelajari", en: "Patterns learned" },
  learnedTracks: { id: "dari lagu", en: "from tracks" },
  learnHint: {
    id: "CuePilot belajar dari edit kamu dan menerapkannya otomatis di lagu lain dengan jenis musik yang sama.",
    en: "CuePilot learns from your edits and applies them automatically to other tracks of the same genre.",
  },
  preferenceProfile: { id: "Profil DJ", en: "DJ Profile" },
  preferenceHint: {
    id: "Profil menyimpan preferensi hot cue kamu per jenis musik, jadi hasilnya makin pas dengan gayamu.",
    en: "A profile saves your hot cue preferences per genre, so results match your style better.",
  },
  noPreference: { id: "Tanpa profil (otomatis)", en: "No profile (automatic)" },

  settingsTitle: { id: "Pengaturan", en: "Settings" },
  engineLabel: { id: "Mesin Analisis", en: "Analysis Engine" },
  engineBasic: { id: "Cepat & ringan", en: "Fast & light" },
  engineIntelli: { id: "Cerdas (stem)", en: "Smart (stems)" },
  engineOnnx: { id: "Presisi (ML)", en: "Precise (ML)" },
  engineBasicHint: {
    id: "Cepat, cocok untuk laptop biasa. Cukup untuk kebanyakan lagu.",
    en: "Fast, great for most laptops. Enough for most tracks.",
  },
  engineIntelliHint: {
    id: "Lebih teliti dengan memisahkan vokal/drum/bass, tapi sedikit lebih lambat.",
    en: "More thorough by separating vocals/drums/bass, but a bit slower.",
  },
  engineOnnxHint: {
    id: "Paling presisi memakai model Machine Learning lokal. Butuh GPU yang mumpuni.",
    en: "Most precise using local Machine Learning models. Needs a capable GPU.",
  },
  engineFallbackNote: {
    id: "Model ML belum terpasang — memakai pemisah stem DSP",
    en: "No ML model installed — using DSP stem separator",
  },
  languageLabel: { id: "Bahasa", en: "Language" },
  indonesian: { id: "Indonesia", en: "Indonesian" },
  english: { id: "English", en: "English" },

  stemLabel: { id: "Stem", en: "Stems" },
  stemVocal: { id: "Vokal", en: "Vocals" },
  stemDrums: { id: "Drum", en: "Drums" },
  stemBass: { id: "Bass", en: "Bass" },
  stemOther: { id: "Melodi", en: "Melody" },
  stemsAnalysisLabel: { id: "Stem yang Dipakai Analisis", en: "Stems Used in Analysis" },
  stemsAnalysisHint: {
    id: "Pilih bagian lagu (vokal, drum, bass, melodi) yang diikutkan saat menganalisis struktur lagu. Matikan yang mengganggu.",
    en: "Choose which parts (vocals, drums, bass, melody) are included when analyzing the structure. Turn off the ones that get in the way.",
  },
  stemsOverlayLabel: { id: "Tampilkan di Gelombang", en: "Show on Waveform" },
  stemsOverlayHint: {
    id: "Klik untuk menampilkan atau menyembunyikan tiap bagian di gelombang suara.",
    en: "Click to show or hide each part on the waveform.",
  },

  onnxConfirmTitle: { id: "Aktifkan mode presisi?", en: "Enable precise mode?" },
  onnxConfirmDesc: {
    id: "Mode ini lebih lambat dan butuh GPU yang cepat. Untuk laptop biasa, kami sarankan mode 'Cerdas' atau 'Cepat'.",
    en: "This mode is slower and needs a fast GPU. For regular laptops, we suggest 'Smart' or 'Fast' mode.",
  },
  onnxConfirmNoGpu: {
    id: "GPU yang kompatibel tidak terdeteksi. Proses akan berjalan sangat lambat di CPU.",
    en: "No compatible GPU detected. The process will run very slowly on CPU.",
  },
  onnxConfirmGpuOk: {
    id: "GPU terdeteksi ({device}). Proses akan lebih cepat.",
    en: "GPU detected ({device}). The process will be faster.",
  },
  onnxConfirmYes: { id: "Ya, Aktifkan", en: "Yes, Activate" },
  onnxConfirmNo: { id: "Batal", en: "Cancel" },

  stage_stems: { id: "Memisahkan stem (vokal / drum / bass)", en: "Separating stems (vocals / drums / bass)" },
  stage_stem_features: { id: "Mengekstrak fitur stem", en: "Extracting stem features" },
  stage_stem_energy: { id: "Menghitung kurva energi", en: "Computing energy curve" },
  stage_stem_sections: { id: "Mendeteksi seksi dari stem", en: "Detecting sections from stems" },
  stage_start: { id: "Memulai analisis…", en: "Starting analysis…" },
  stage_load: { id: "Membaca file audio", en: "Reading audio file" },
  stage_extract: { id: "Mengekstrak fitur suara", en: "Extracting audio features" },
  stage_profile: { id: "Memuat jenis musik", en: "Loading genre" },
  stage_key: { id: "Mendeteksi kunci musik", en: "Detecting musical key" },
  stage_beatgrid: { id: "Menganalisis ketukan", en: "Analyzing beatgrid" },
  stage_structure: { id: "Mendeteksi struktur lagu", en: "Detecting song structure" },
  stage_cues: { id: "Menyusun hot cue", en: "Creating hot cues" },
  stage_learn: { id: "Menerapkan pola belajar kamu", en: "Applying your learned patterns" },
  stage_done: { id: "Hot cue siap", en: "Hot cues ready" },

  batchTitle: { id: "Membuat hot cue untuk semua lagu", en: "Creating hot cues for all tracks" },
  batchProgress: { id: "Proses", en: "Progress" },
  batchDone: { id: "Selesai", en: "Done" },
  batchFailed: { id: "Gagal", en: "Failed" },
  batchQueued: { id: "Menunggu", en: "Queued" },
  batchCancel: { id: "Hentikan", en: "Stop" },
  batchResume: { id: "Lanjutkan", en: "Resume" },
  batchFinished: { id: "Selesai semua!", en: "All done!" },
  batchFinishedFail: {
    id: "Selesai. Beberapa lagu gagal diproses.",
    en: "Finished. Some tracks could not be processed.",
  },
  batchEmpty: {
    id: "Tidak ada lagu yang bisa diproses.",
    en: "No tracks to process.",
  },

  errorTitle: { id: "Terjadi kendala", en: "Something went wrong" },
  errorHint: { id: "Coba lagi, atau pilih folder yang lain.", en: "Try again, or pick another folder." },
  loadingError: { id: "Gagal memuat", en: "Failed to load" },

  readinessReady: { id: "Siap Gig", en: "Gig Ready" },
  readinessNeedsReview: { id: "Perlu Dicek", en: "Needs Review" },
  readinessNotAnalyzed: { id: "Belum Dianalisis", en: "Not Analyzed" },
  readinessTitle: { id: "Kesiapan Gig", en: "Gig Readiness" },
  readinessHint: {
    id: "Skor 0-100 seberapa siap lagu ini dipakai di gig, dari akurasi hot cue, kualitas audio, kelengkapan struktur, dan stabilitas edit.",
    en: "A 0-100 score of how gig-ready this track is, from cue accuracy, audio quality, structure completeness, and edit stability.",
  },
  readinessCueConfidence: { id: "Akurasi cue", en: "Cue accuracy" },
  readinessAudioQuality: { id: "Kualitas audio", en: "Audio quality" },
  readinessStructure: { id: "Kelengkapan struktur", en: "Structure" },
  readinessStability: { id: "Stabilitas edit", en: "Edit stability" },
  qcTitle: { id: "Pemeriksaan Audio", en: "Audio QC" },
  qcScore: { id: "Skor kualitas", en: "Quality score" },
  qcOk: { id: "Kualitas audio bagus", en: "Audio quality is good" },
  qcIssues: { id: "Catatan kualitas", en: "Quality notes" },
  qcClipping: { id: "Kliping", en: "Clipping" },
  qcLoudness: { id: "Volume tidak konsisten", en: "Loudness inconsistency" },
  qcMonoCompat: { id: "Masalah mono-compat", en: "Mono-compatibility" },
  qcTranscode: { id: "Bitrate rendah / transcode", en: "Low bitrate / transcode" },
  qcEdgeSilence: { id: "Diam di awal/akhir", en: "Edge silence" },
  qcPeak: { id: "Puncak", en: "Peak" },
  qcLufs: { id: "Kekerasan (LUFS)", en: "Loudness (LUFS)" },
  qcHighFreq: { id: "Frekuensi tinggi", en: "High-freq" },
  qcStereoCorr: { id: "Korelasi stereo", en: "Stereo correlation" },

  loopsTitle: { id: "Saved Loop Terdeteksi", en: "Detected Saved Loops" },
  loopsHint: {
    id: "Region 4/8/16 bar yang mulus diulang — klik untuk mendengar.",
    en: "Seamless 4/8/16-bar regions — click to preview.",
  },
  loopsSeamless: { id: "Kehalusan", en: "Seamless" },
};

export const tr = (lang: Lang, key: string, fallback?: string): string => {
  const entry = DICT[key];
  if (!entry) return fallback ?? key;
  return lang === "id" ? entry.id : entry.en;
};

interface Ctx {
  lang: Lang;
  setLang: (l: Lang) => void;
  t: (key: string, fallback?: string) => string;
}

const LangContext = createContext<Ctx>({ lang: "id", setLang: () => {}, t: (k) => k });

export function LangProvider({ children }: { children: ReactNode }) {
  const [lang, setLang] = useState<Lang>("id");
  const t = (key: string, fallback?: string) => tr(lang, key, fallback);
  return (
    <LangContext.Provider value={{ lang, setLang, t }}>{children}</LangContext.Provider>
  );
}

export function useLang(): Ctx {
  return useContext(LangContext);
}