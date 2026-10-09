import { useCallback, useEffect, useRef, useState } from "react";
import {
  api,
  AnalyzeProgress,
  AnalyzeResult,
  BatchJobStatus,
  Cue,
  GenreProfileMeta,
  LearnStatsResult,
  PreferenceProfileItem,
  ScanResult,
  TrackItem,
  StemName,
  WaveformData,
} from "./api";
import { Waveform } from "./Waveform";
import { CueGrid } from "./CueGrid";
import { TrackList } from "./TrackList";
import { LoadingDisc } from "./LoadingDisc";
import { EmptyState } from "./EmptyState";
import { GenreModal } from "./GenreModal";
import { FolderStep } from "./FolderStep";
import { SettingsPanel } from "./SettingsPanel";
import { ReadinessBadge, ReadinessPanel } from "./Readiness";
import { LoopsPanel } from "./Loops";
import { useLang } from "./i18n";

const DEFAULT_PROFILE = "open_format";
const NUM_SLOTS = 8;

const ROLE_COLORS: Record<string, string> = {
  START: "#2ED98B",
  INTRO: "#2ECC71",
  MIX_IN: "#7FB069",
  VOCAL: "#FFC24B",
  HOOK: "#FFB300",
  MELODY: "#B06BFF",
  GROOVE: "#FF8A3D",
  PRE_GROOVE: "#FFA066",
  BUILD: "#FF8A3D",
  PRE_DROP: "#FF6D4D",
  DROP: "#FF4D6D",
  CLIMAX: "#FF2E55",
  TRANSITION: "#46C8FF",
  CHORUS: "#B06BFF",
  BREAKDOWN: "#46C8FF",
  ENERGY_PEAK: "#FF9E2E",
  SECOND_DROP: "#FF2E55",
  OUTRO: "#2AE0C8",
  MIX_OUT: "#4FB8A0",
  ALTERNATIVE: "#9AA1AB",
};

export default function App() {
  const { t, lang, setLang } = useLang();
  const [step, setStep] = useState<"folder" | "library" | "result">("folder");
  const [folder, setFolder] = useState("");
  const [scan, setScan] = useState<ScanResult | null>(null);
  const [scanning, setScanning] = useState(false);
  const [selected, setSelected] = useState<TrackItem | null>(null);
  const [result, setResult] = useState<AnalyzeResult | null>(null);
  const [waveform, setWaveform] = useState<WaveformData | null>(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState<AnalyzeProgress | null>(null);
  const [error, setError] = useState("");
  const [profiles, setProfiles] = useState<string[]>([DEFAULT_PROFILE]);
  const [profileMeta, setProfileMeta] = useState<Record<string, GenreProfileMeta>>({});
  const [pendingTrack, setPendingTrack] = useState<TrackItem | null>(null);
  const [activeProfile, setActiveProfile] = useState(DEFAULT_PROFILE);
  const [activePreference, setActivePreference] = useState("");
  const [preferences, setPreferences] = useState<PreferenceProfileItem[]>([]);
  const [engine, setEngine] = useState<"basic" | "intellistem" | "onnx">("basic");
  const [activeStems, setActiveStems] = useState<StemName[]>(["vocal", "drums", "bass", "other"]);
  const [gpuInfo, setGpuInfo] = useState<{ provider: string; device: string } | null>(null);
  const [showOnnxConfirm, setShowOnnxConfirm] = useState(false);
  const [showSettings, setShowSettings] = useState(false);

  const [audioUrl, setAudioUrl] = useState("");
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const rafTimeRef = useRef<number>(0);

  const [editing, setEditing] = useState(false);
  const [selectedSlot, setSelectedSlot] = useState(1);
  const [savedMsg, setSavedMsg] = useState("");
  const [learnStats, setLearnStats] = useState<LearnStatsResult | null>(null);
  const [writingMeta, setWritingMeta] = useState(false);
  const [metaResult, setMetaResult] = useState<{ ok: boolean; count?: number; error?: string } | null>(
    null
  );
  const [metaProgress, setMetaProgress] = useState<AnalyzeProgress | null>(null);

  const [batchStatus, setBatchStatus] = useState<BatchJobStatus | null>(null);
  const [analyzingAll, setAnalyzingAll] = useState(false);
  const cancelBatchRef = useRef<() => boolean>(() => false);

  useEffect(() => {
    if (!playing) return;
    const tick = () => {
      const a = audioRef.current;
      if (a) setCurrentTime(a.currentTime);
      rafTimeRef.current = requestAnimationFrame(tick);
    };
    rafTimeRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafTimeRef.current);
  }, [playing]);

  useEffect(() => {
    api.profiles().then((r) => {
      setProfiles(r.profiles);
      setProfileMeta(r.meta || {});
    }).catch(() => {});
  }, []);

  useEffect(() => {
    api.preferencesList().then((r) => setPreferences(r.profiles || [])).catch(() => {});
  }, []);

  useEffect(() => {
    api.settings().then((r) => {
      setEngine(r.engine);
      if (r.gpu) setGpuInfo(r.gpu);
      if (r.stems && r.stems.length > 0) setActiveStems(r.stems);
    }).catch(() => {});
  }, []);

  const switchEngine = useCallback(async (next: "basic" | "intellistem" | "onnx", force = false) => {
    if (next === "onnx" && !force) {
      setShowOnnxConfirm(true);
      return;
    }
    setEngine(next);
    try {
      const r = await api.setSettings(next, activeStems);
      if (r.gpu) setGpuInfo(r.gpu);
      if (r.stems && r.stems.length > 0) setActiveStems(r.stems);
    } catch {
      /* keep local state */
    }
  }, [activeStems]);

  const stopAudio = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
    setPlaying(false);
    setCurrentTime(0);
    setAudioUrl("");
  }, []);

  const analyzeTrack = useCallback(
    async (t: TrackItem, profile: string) => {
      setLoading(true);
      setError("");
      setResult(null);
      setWaveform(null);
      setActiveProfile(profile);
      setEditing(false);
      setSavedMsg("");
      setProgress(null);
      stopAudio();
      try {
        const [res, wf, url] = await Promise.all([
          api.analyzeStream(t.path, profile, setProgress, engine, activeStems),
          api.waveform(t.path),
          api.audioUrl(t.path),
        ]);
        setWaveform(wf);
        setResult(res);
        setAudioUrl(url);
        setStep("result");
      } catch (e) {
        setError(String(e));
      } finally {
        setLoading(false);
      }
    },
    [stopAudio, engine, activeStems]
  );

  const selectTrack = useCallback((t: TrackItem) => {
    setSelected(t);
    setError("");
    setPendingTrack(t);
  }, []);

  const onPickGenre = useCallback(
    (profile: string) => {
      const t = pendingTrack;
      setPendingTrack(null);
      if (t) {
        analyzeTrack(t, profile);
      }
    },
    [pendingTrack, analyzeTrack]
  );

  const doScan = useCallback(async () => {
    const dir = folder.trim();
    if (!dir || scanning) return;
    setScanning(true);
    setError("");
    setScan(null);
    setSelected(null);
    setPendingTrack(null);
    setResult(null);
    setWaveform(null);
    setEditing(false);
    setSavedMsg("");
    stopAudio();
    try {
      const s = await api.scan(dir);
      setScan(s);
      setStep("library");
    } catch (e) {
      setError(String(e));
    } finally {
      setScanning(false);
    }
  }, [folder, scanning, stopAudio]);

  const openFolderPicker = useCallback(async () => {
    interface PyWebviewApi {
      pick_folder?: () => Promise<string | null>;
    }
    const w = window as unknown as {
      pywebview?: { api?: PyWebviewApi };
    };
    try {
      const dir = await w.pywebview?.api?.pick_folder?.();
      if (dir) {
        setFolder(dir);
        await doScan();
      }
    } catch {
      // ignore; user can type the path manually
    }
  }, [doScan]);

  const goToFolder = useCallback(() => {
    setStep("folder");
    setSelected(null);
    setPendingTrack(null);
    setResult(null);
    setWaveform(null);
    setEditing(false);
    stopAudio();
  }, [stopAudio]);

  const togglePlay = useCallback(() => {
    const a = audioRef.current;
    if (!a) return;
    if (playing) {
      a.pause();
    } else {
      a.play().catch(() => {});
    }
  }, [playing]);

  const seekTo = useCallback((time: number) => {
    const a = audioRef.current;
    if (!a) return;
    a.currentTime = time;
    setCurrentTime(time);
  }, []);

  const jumpToCue = useCallback(
    (cue: Cue) => {
      seekTo(cue.time);
      if (audioRef.current && audioRef.current.paused) {
        audioRef.current.play().catch(() => {});
      }
    },
    [seekTo]
  );

  const reset = useCallback(() => {
    const a = audioRef.current;
    if (a) {
      a.pause();
      a.currentTime = 0;
    }
    setPlaying(false);
    setCurrentTime(0);
  }, []);

  const reanalyze = useCallback(() => {
    if (selected) setPendingTrack(selected);
  }, [selected]);

  const refreshLearnStats = useCallback(() => {
    api.learnStats(activeProfile).then(setLearnStats).catch(() => {});
  }, [activeProfile]);

  useEffect(() => {
    if (result) refreshLearnStats();
  }, [result, refreshLearnStats]);

  const roleForSlot = useCallback(
    (slot: number): string => {
      return (
        profileMeta[activeProfile]?.cueStrategy?.[String(slot)] ||
        profileMeta[activeProfile]?.cueStrategy?.[slot] ||
        ""
      );
    },
    [activeProfile, profileMeta]
  );

  const cueColorsForSlots = useCallback((): Record<number, string> => {
    const map: Record<number, string> = {};
    for (let s = 1; s <= NUM_SLOTS; s++) {
      const role = roleForSlot(s);
      map[s] = ROLE_COLORS[role] || "#22c55e";
    }
    return map;
  }, [roleForSlot]);

  const patchCues = useCallback((fn: (cues: Cue[]) => Cue[]) => {
    setResult((r) => (r ? { ...r, cues: fn(r.cues) } : r));
  }, []);

  const placeCue = useCallback(
    (time: number) => {
      if (!result || !selectedSlot) return;
      const role = roleForSlot(selectedSlot);
      const color = ROLE_COLORS[role] || "#22c55e";
      patchCues((cues) => {
        const idx = cues.findIndex((c) => c.slot === selectedSlot);
        const newCue: Cue = {
          slot: selectedSlot,
          type: role || "CUSTOM",
          time: Math.max(0, Math.min(result.analysis.duration, time)),
          label: role || `Cue ${selectedSlot}`,
          confidence: 1,
          color,
          locked: false,
          source: "user",
          reason: ["user"],
        };
        if (idx >= 0) {
          const next = [...cues];
          next[idx] = { ...newCue, ...(cues[idx].locked ? { locked: true } : {}) };
          return next;
        }
        return [...cues, newCue].sort((a, b) => a.slot - b.slot);
      });
      seekTo(time);
    },
    [result, selectedSlot, roleForSlot, patchCues, seekTo]
  );

  const deleteCue = useCallback((slot: number) => {
    patchCues((cues) => cues.filter((c) => c.slot !== slot));
  }, [patchCues]);

  const toggleLock = useCallback(
    (slot: number) => {
      patchCues((cues) =>
        cues.map((c) => (c.slot === slot ? { ...c, locked: !c.locked } : c))
      );
    },
    [patchCues]
  );

  const saveCues = useCallback(async () => {
    if (!selected || !result) return;
    setSavedMsg("");
    try {
      const payload = {
        path: selected.path,
        profile: activeProfile,
        bpm: result.analysis.bpm,
        key: result.analysis.key,
        duration: result.analysis.duration,
        cues: result.cues.map((c) => ({
          slot: c.slot,
          type: c.type,
          time: c.time,
          label: c.label,
          confidence: c.confidence,
          color: c.color,
          locked: c.locked,
          source: c.source,
          reason: c.reason,
        })),
      };
      const res = await api.editsSave(payload);
      setSavedMsg(
        `✓ ${t("saved")}${res.learnStats ? ` · ${t("learnedCount")}: ${res.learnStats.total}` : ""}`
      );
      refreshLearnStats();
    } catch (e) {
      setError(String(e));
    }
  }, [selected, result, activeProfile, refreshLearnStats, t]);

  const writeToMetadata = useCallback(async () => {
    if (!selected || !result) return;
    setWritingMeta(true);
    setMetaResult(null);
    setMetaProgress({ stage: "read", message: "reading original tags", pct: 0 });
    try {
      const res = await api.metadataWriteStream(
        {
          path: selected.path,
          bpm: result.analysis.bpm,
          key: result.analysis.key,
          cues: result.cues.map((c) => ({
            slot: c.slot,
            type: c.type,
            time: c.time,
            label: c.label,
            confidence: c.confidence,
            color: c.color,
            locked: c.locked,
            source: c.source,
            reason: c.reason,
          })),
          analysis: { bpm: result.analysis.bpm, key: result.analysis.key },
        },
        setMetaProgress
      );
      if (res.ok) {
        setMetaResult({ ok: true, count: res.cueCount ?? result.cues.length });
      } else {
        setMetaResult({ ok: false, error: res.error || "write failed" });
      }
    } catch (e) {
      setMetaResult({ ok: false, error: String(e) });
    } finally {
      setWritingMeta(false);
      setMetaProgress(null);
    }
  }, [selected, result]);

  const startBatch = useCallback(async () => {
    if (!scan || analyzingAll) return;
    setError("");
    setBatchStatus(null);
    setAnalyzingAll(true);
    cancelBatchRef.current = () => false;
    try {
      const created = await api.batchCreate({
        folder: scan.folder,
        profile: activeProfile,
        engine,
        preference: activePreference,
      });
      setBatchStatus(created);
      const onProgress = () => {};
      const status = await api.batchRunStream(created.jobId!, onProgress, () => cancelBatchRef.current());
      setBatchStatus(status);
    } catch (e) {
      const msg = String(e);
      if (msg.includes("cancelled") || msg.includes("cancel")) {
        // user cancelled — not an error
        if (batchStatus?.jobId) {
          try {
            setBatchStatus(await api.batchStatus(batchStatus.jobId));
          } catch {
            /* ignore */
          }
        }
      } else {
        setError(msg);
      }
    } finally {
      setAnalyzingAll(false);
    }
  }, [scan, analyzingAll, activeProfile, engine, activePreference, batchStatus?.jobId]);

  const stopBatch = useCallback(() => {
    cancelBatchRef.current = () => true;
    if (batchStatus?.jobId) {
      api.batchCancel(batchStatus.jobId).catch(() => {});
    }
  }, [batchStatus?.jobId]);

  const refreshBatch = useCallback(async () => {
    if (batchStatus?.jobId) {
      try {
        setBatchStatus(await api.batchStatus(batchStatus.jobId));
      } catch {
        /* ignore */
      }
    }
  }, [batchStatus?.jobId]);

  useEffect(() => {
    if (!analyzingAll) return;
    const id = window.setInterval(refreshBatch, 1200);
    return () => window.clearInterval(id);
  }, [analyzingAll, refreshBatch]);

  const onEngineChange = useCallback(
    (e: "basic" | "intellistem" | "onnx") => {
      setShowSettings(false);
      switchEngine(e);
    },
    [switchEngine]
  );

  const onProfileChange = useCallback(
    (p: string) => {
      setActiveProfile(p);
      setShowSettings(false);
      if (selected) setPendingTrack(selected);
    },
    [selected]
  );

  const onPreferenceChange = useCallback(
    (p: string) => {
      setActivePreference(p);
      setShowSettings(false);
      if (selected) setPendingTrack(selected);
    },
    [selected]
  );

  const onStemsChange = useCallback(
    (stems: StemName[]) => {
      setActiveStems(stems);
      setShowSettings(false);
      api.setSettings(undefined, stems).catch(() => {});
      if (selected) setPendingTrack(selected);
    },
    [selected]
  );

  const stepIndex = step === "folder" ? 0 : step === "library" ? 1 : 2;

  return (
    <div className="app">
      <audio
        ref={audioRef}
        src={audioUrl}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
        onTimeUpdate={(e) => setCurrentTime((e.target as HTMLAudioElement).currentTime)}
      />

      <header className="topbar">
        <div className="brand">
          <img src="/logo.png" alt="CuePilot" className="brand-logo" />
          <h1>CuePilot</h1>
        </div>
        <span className="subtitle">{t("subtitle")}</span>
        <div className="topbar-right">
          <button className="settings-btn" onClick={() => setShowSettings(true)}>
            ⚙ {t("settingsTitle")}
          </button>
          <button
            className="lang-toggle"
            onClick={() => setLang(lang === "id" ? "en" : "id")}
            title={lang === "id" ? "Switch to English" : "Ganti ke Indonesia"}
          >
            {lang === "id" ? "EN" : "ID"}
          </button>
        </div>
      </header>

      <div className="steps">
        <div className={`step ${stepIndex === 0 ? "is-active" : ""} ${stepIndex > 0 ? "is-done" : ""}`}>
          <span className="step-num">{stepIndex > 0 ? "✓" : "1"}</span>
          {t("pickFolder")}
        </div>
        <span className="step-arrow">→</span>
        <div className={`step ${stepIndex === 1 ? "is-active" : ""} ${stepIndex > 1 ? "is-done" : ""}`}>
          <span className="step-num">{stepIndex > 1 ? "✓" : "2"}</span>
          {t("pickATrack")}
        </div>
        <span className="step-arrow">→</span>
        <div className={`step ${stepIndex === 2 ? "is-active" : ""}`}>
          <span className="step-num">3</span>
          {t("saveToFileButton")}
        </div>
      </div>

      {error && <div className="error">{error}</div>}

      {step === "folder" && (
        <main className="content">
          {(loading || scanning) && (
            <LoadingDisc variant={scanning ? "scanning" : "analyzing"} progress={progress ?? undefined} />
          )}
          {!loading && !scanning && (
            <FolderStep
              folder={folder}
              setFolder={setFolder}
              scanning={scanning}
              onPickFolder={openFolderPicker}
              onScan={doScan}
            />
          )}
        </main>
      )}

      {step === "library" && (
        <div className="layout">
          <TrackList
            scan={scan}
            selectedId={selected?.id}
            onSelect={selectTrack}
            analyzingAll={analyzingAll}
            canBatch={!!scan && scan.total > 0}
            onBatch={startBatch}
            batchStatus={batchStatus}
            onStopBatch={stopBatch}
            onPickFolder={goToFolder}
          />
          <main className="content">
            {loading && <LoadingDisc variant="analyzing" progress={progress ?? undefined} />}
            {!loading && !result && <EmptyState scanning={scanning} />}
            {!loading && result && waveform && (
              <ResultView
                selected={selected}
                result={result}
                waveform={waveform}
                profileMeta={profileMeta}
                activeProfile={activeProfile}
                currentTime={currentTime}
                playing={playing}
                editing={editing}
                selectedSlot={selectedSlot}
                savedMsg={savedMsg}
                learnStats={learnStats}
                cueColors={cueColorsForSlots()}
                activeStems={activeStems}
                onTogglePlay={togglePlay}
                onReset={reset}
                onJumpToCue={jumpToCue}
                onSeek={seekTo}
                onReanalyze={reanalyze}
                onWriteMetadata={writeToMetadata}
                onToggleEditing={() => setEditing((v) => !v)}
                onSelectSlot={setSelectedSlot}
                onSaveCues={saveCues}
                onDeleteCue={deleteCue}
                onToggleLock={toggleLock}
                onPlaceCue={placeCue}
                onBack={goToFolder}
              />
            )}
          </main>
        </div>
      )}

      {step === "result" && (
        <div className="layout">
          <TrackList
            scan={scan}
            selectedId={selected?.id}
            onSelect={selectTrack}
            analyzingAll={analyzingAll}
            canBatch={!!scan && scan.total > 0}
            onBatch={startBatch}
            batchStatus={batchStatus}
            onStopBatch={stopBatch}
            onPickFolder={goToFolder}
          />
          <main className="content">
            {loading && <LoadingDisc variant="analyzing" progress={progress ?? undefined} />}
            {!loading && result && waveform && (
              <ResultView
                selected={selected}
                result={result}
                waveform={waveform}
                profileMeta={profileMeta}
                activeProfile={activeProfile}
                currentTime={currentTime}
                playing={playing}
                editing={editing}
                selectedSlot={selectedSlot}
                savedMsg={savedMsg}
                learnStats={learnStats}
                cueColors={cueColorsForSlots()}
                activeStems={activeStems}
                onTogglePlay={togglePlay}
                onReset={reset}
                onJumpToCue={jumpToCue}
                onSeek={seekTo}
                onReanalyze={reanalyze}
                onWriteMetadata={writeToMetadata}
                onToggleEditing={() => setEditing((v) => !v)}
                onSelectSlot={setSelectedSlot}
                onSaveCues={saveCues}
                onDeleteCue={deleteCue}
                onToggleLock={toggleLock}
                onPlaceCue={placeCue}
                onBack={goToFolder}
              />
            )}
          </main>
        </div>
      )}

      <footer className="footer">
        <span className="footer-text">
          ♥{" "}
          <b>YanNotTamm</b>
        </span>
        <span className="footer-hint">
          {lang === "id"
            ? "CuePilot — Kenali Drop, Kuasai Momen"
            : "CuePilot — Know the Drop, Control the Moment"}
        </span>
      </footer>

      {pendingTrack && (
        <GenreModal
          trackName={pendingTrack.title || pendingTrack.filename}
          profiles={profiles}
          meta={profileMeta}
          onPick={onPickGenre}
          onCancel={() => setPendingTrack(null)}
        />
      )}

      {showSettings && (
        <SettingsPanel
          engine={engine}
          gpuInfo={gpuInfo}
          profiles={profiles}
          profileMeta={profileMeta}
          activeProfile={activeProfile}
          preferences={preferences}
          activePreference={activePreference}
          activeStems={activeStems}
          onEngine={onEngineChange}
          onStems={onStemsChange}
          onLang={setLang}
          onProfile={onProfileChange}
          onPreference={onPreferenceChange}
          onClose={() => setShowSettings(false)}
        />
      )}

      {writingMeta && (
        <div className="modal-backdrop">
          <div className="modal modal-writing">
            <LoadingDisc variant="analyzing" progress={metaProgress ?? undefined} />
          </div>
        </div>
      )}

      {metaResult && (
        <div className="modal-backdrop" onClick={() => setMetaResult(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            {metaResult.ok ? (
              <>
                <div className="meta-result-icon">✓</div>
                <h2>{t("writtenOk")}</h2>
                <p className="modal-hint">
                  {t("writtenOkDetail").replace("{count}", String(metaResult.count ?? 0))}
                </p>
              </>
            ) : (
              <>
                <div className="meta-result-icon err">✕</div>
                <h2>{t("writtenFail")}</h2>
                <p className="modal-hint">{metaResult.error}</p>
              </>
            )}
            <button className="modal-close" onClick={() => setMetaResult(null)}>
              {t("close")}
            </button>
          </div>
        </div>
      )}

      {showOnnxConfirm && (
        <div className="modal-backdrop" onClick={() => setShowOnnxConfirm(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()} style={{ maxWidth: "450px" }}>
            <h2>{t("onnxConfirmTitle")}</h2>
            <p className="modal-hint" style={{ margin: "12px 0", lineHeight: "1.5" }}>
              {t("onnxConfirmDesc")}
            </p>
            <div
              className={`gpu-status-alert ${gpuInfo && gpuInfo.provider !== "cpu" ? "success" : "warning"}`}
              style={{
                padding: "10px 14px",
                borderRadius: "6px",
                fontSize: "13px",
                margin: "16px 0",
                background: gpuInfo && gpuInfo.provider !== "cpu" ? "#ecfdf5" : "#fffbeb",
                color: gpuInfo && gpuInfo.provider !== "cpu" ? "#065f46" : "#92400e",
                border: `1px solid ${gpuInfo && gpuInfo.provider !== "cpu" ? "#a7f3d0" : "#fef3c7"}`,
                textAlign: "left",
              }}
            >
              {gpuInfo && gpuInfo.provider !== "cpu"
                ? t("onnxConfirmGpuOk").replace("{device}", `${gpuInfo.device} [${gpuInfo.provider.toUpperCase()}]`)
                : t("onnxConfirmNoGpu")}
            </div>
            <div style={{ display: "flex", gap: "10px", marginTop: "20px" }}>
              <button
                className="scan-btn"
                onClick={() => {
                  setShowOnnxConfirm(false);
                  switchEngine("onnx", true);
                }}
                style={{ flex: 1 }}
              >
                {t("onnxConfirmYes")}
              </button>
              <button
                className="browse-btn"
                onClick={() => setShowOnnxConfirm(false)}
                style={{ flex: 1, margin: 0 }}
              >
                {t("onnxConfirmNo")}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

interface ResultViewProps {
  selected: TrackItem | null;
  result: AnalyzeResult;
  waveform: WaveformData;
  profileMeta: Record<string, GenreProfileMeta>;
  activeProfile: string;
  currentTime: number;
  playing: boolean;
  editing: boolean;
  selectedSlot: number;
  savedMsg: string;
  learnStats: LearnStatsResult | null;
  cueColors: Record<number, string>;
  activeStems: StemName[];
  onTogglePlay: () => void;
  onReset: () => void;
  onJumpToCue: (c: Cue) => void;
  onSeek: (t: number) => void;
  onReanalyze: () => void;
  onWriteMetadata: () => void;
  onToggleEditing: () => void;
  onSelectSlot: (s: number) => void;
  onSaveCues: () => void;
  onDeleteCue: (slot: number) => void;
  onToggleLock: (slot: number) => void;
  onPlaceCue: (t: number) => void;
  onBack: () => void;
}

function ResultView(props: ResultViewProps) {
  const { t } = useLang();
  const {
    selected,
    result,
    waveform,
    profileMeta,
    activeProfile,
    currentTime,
    playing,
    editing,
    selectedSlot,
    savedMsg,
    learnStats,
    cueColors,
    activeStems,
    onTogglePlay,
    onReset,
    onJumpToCue,
    onSeek,
    onReanalyze,
    onWriteMetadata,
    onToggleEditing,
    onSelectSlot,
    onSaveCues,
    onDeleteCue,
    onToggleLock,
    onPlaceCue,
    onBack,
  } = props;

  return (
    <>
      <div className="info">
        <button className="back-btn" onClick={onBack}>
          ← {t("backToFolder")}
        </button>
        <span className="info-cell">
          <label>{t("bpm")}</label>
          <b>{result.analysis.bpm.toFixed(1)}</b>
        </span>
        <span className="info-cell">
          <label>{t("key")}</label>
          <b>{result.analysis.key}</b>
        </span>
        <span className="info-cell">
          <label>{t("duration")}</label>
          <b>{fmt(result.analysis.duration)}</b>
        </span>
        <span className="spacer" />
        <ReadinessBadge readiness={result.analysis.extras?.readiness} qc={result.analysis.extras?.quality} />
        <button className="genre-badge" onClick={onReanalyze} title={t("changeGenre")}>
          {profileMeta[activeProfile]?.displayName || activeProfile.replace("_", " ")}
        </button>
        <button
          className="meta-write-btn"
          onClick={onWriteMetadata}
          title={t("writeMetadataHint")}
        >
          💾 {t("saveToFileButton")}
        </button>
        <span className="now-playing">
          {selected?.artist ? `${selected.artist} - ${selected.title}` : selected?.title}
        </span>
      </div>

      <div className="transport">
        <button className="trans-btn play" onClick={onTogglePlay}>
          {playing ? "❚❚" : "▶"}
        </button>
        <button className="trans-btn" onClick={onReset} title={t("stop")}>
          ■
        </button>
        <div className="time">
          {fmt(currentTime)} <span className="time-sep">/</span> {fmt(result.analysis.duration)}
        </div>
        <div className="cue-jump">
          {result.cues.map((c) => (
            <button
              key={c.slot}
              className="jump-chip"
              style={{ "--chip": c.color } as React.CSSProperties}
              onClick={() => onJumpToCue(c)}
              title={`${t("listenHere")} · ${fmt(c.time)}`}
            >
              {c.slot}
            </button>
          ))}
        </div>
        <div className="edit-controls">
          <button
            className={`edit-toggle ${editing ? "is-on" : ""}`}
            onClick={onToggleEditing}
            title={editing ? t("doneEditing") : t("editCues")}
          >
            {editing ? "✓ " + t("doneEditing") : "✎ " + t("editCues")}
          </button>
          {editing && (
            <button className="save-btn" onClick={onSaveCues} title={t("saveCues")}>
              {t("save")}
            </button>
          )}
        </div>
      </div>

      <ReadinessPanel
        readiness={result.analysis.extras?.readiness}
        qc={result.analysis.extras?.quality}
      />

      <LoopsPanel loops={result.analysis.extras?.loops ?? []} onJump={onSeek} />

      {editing && (
        <div className="category-bar">
          <span className="cat-label">{t("categories")}</span>
          {Array.from({ length: NUM_SLOTS }, (_, i) => i + 1).map((s) => {
            const role = cueRole(activeProfile, profileMeta, s);
            const color = ROLE_COLORS[role] || "#22c55e";
            const used = result.cues.some((c) => c.slot === s);
            return (
              <button
                key={s}
                className={`cat-chip ${selectedSlot === s ? "is-selected" : ""}`}
                style={{ "--cat": color } as React.CSSProperties}
                onClick={() => onSelectSlot(s)}
                title={`${s} ${role}`}
              >
                <b>{s}</b>
                <span>{role || `slot ${s}`}</span>
                {used && <i className="cat-dot" />}
              </button>
            );
          })}
          <span className="cat-hint">{t("clickToPlace")}</span>
        </div>
      )}

      {savedMsg && <div className="saved-banner">{savedMsg}</div>}

      {learnStats && learnStats.patterns.length > 0 && (
        <div className="learn-banner">
          <span className="learn-dot" />
          {t("learnedCount")}: {learnStats.stats.total} · {t("learnedTracks")}: {learnStats.stats.tracks}
        </div>
      )}

      <Waveform
        waveform={waveform}
        sections={result.sections}
        cues={result.cues}
        currentTime={currentTime}
        playing={playing}
        onSeek={onSeek}
        editing={editing}
        selectedSlot={selectedSlot}
        onPlaceCue={onPlaceCue}
        cueColors={cueColors}
        stems={result.analysis.extras?.stems ?? null}
        activeStems={activeStems}
      />

      <CueGrid
        cues={result.cues}
        onJump={onJumpToCue}
        editing={editing}
        onDelete={onDeleteCue}
        onToggleLock={onToggleLock}
      />
    </>
  );
}

function cueRole(
  profile: string,
  meta: Record<string, GenreProfileMeta>,
  slot: number
): string {
  return (
    meta[profile]?.cueStrategy?.[String(slot)] ||
    meta[profile]?.cueStrategy?.[slot] ||
    ""
  );
}

export function fmt(t: number): string {
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  const ms = Math.floor((t % 1) * 1000);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${String(ms).padStart(3, "0")}`;
}

export function sectionColor(type: string): string {
  const map: Record<string, string> = {
    INTRO: "#2ED98B",
    VOCAL: "#FFC24B",
    BUILD: "#FF8A3D",
    DROP: "#FF4D6D",
    BREAKDOWN: "#46C8FF",
    CHORUS: "#B06BFF",
    SECOND_DROP: "#FF2E55",
    OUTRO: "#2AE0C8",
  };
  return map[type] || "#9AA1AB";
}