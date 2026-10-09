declare global {
  interface Window {
    CUEPILOT_TOKEN?: string;
  }
}

export interface TrackItem {
  id: string;
  path: string;
  artist: string;
  title: string;
  filename: string;
  duration: number;
  status: string;
  error?: string;
}

export interface ScanResult {
  folder: string;
  total: number;
  duplicates: number;
  failed: number;
  tracks: TrackItem[];
}

export interface Cue {
  slot: number;
  type: string;
  time: number;
  label: string;
  confidence: number;
  color: string;
  locked: boolean;
  source: string;
  reason: string[];
}

export interface Section {
  type: string;
  start: number;
  end: number;
  confidence: number;
  source: string;
}

export interface QcIssue {
  code: string;
  severity: "error" | "warning" | "info";
  label: string;
  detail: string;
}

export interface QcReport {
  score: number;
  ok: boolean;
  issues: QcIssue[];
  metrics: Record<string, number>;
}

export interface GigReadiness {
  score: number;
  bucket: "ready" | "needs_review" | "not_analyzed";
  displayBucket: string;
  components: Record<string, number>;
  reasons: string[];
}

export interface SavedLoop {
  type: string;
  bars: number;
  start: number;
  end: number;
  seamlessScore: number;
  useCase: string;
  phraseIndex: number;
  reasons: string[];
}

export interface Analysis {
  trackId: string;
  duration: number;
  sampleRate: number;
  bpm: number;
  key: string;
  beats: number[];
  downbeats: number[];
  energyCurve: { time: number; energy: number }[];
  confidence: number;
  engine?: string;
  engineVersion?: string;
  extras?: {
    engine?: string;
    backend?: string;
    status?: string;
    enabledStems?: string[];
    stems?: StemCurves;
    quality?: QcReport;
    readiness?: GigReadiness;
    loops?: SavedLoop[];
  };
}

export interface StemPoint {
  t: number;
  v: number;
}

export interface StemCurves {
  times: number[];
  vocal: StemPoint[];
  drums: StemPoint[];
  bass: StemPoint[];
  other: StemPoint[];
}

export type StemName = "vocal" | "drums" | "bass" | "other";

export interface AnalyzeResult {
  analysis: Analysis;
  sections: Section[];
  cues: Cue[];
}

export interface WaveformPeak {
  t: number;
  min: number;
  max: number;
}

export interface WaveformData {
  duration: number;
  sampleRate: number;
  peaks: WaveformPeak[];
}

export interface GenreProfileMeta {
  name: string;
  displayName: string;
  description: string;
  descriptionId: string;
  bpmRange: [number, number];
  sectionOrder: string[];
  slotMap: Record<string, number>;
  cueStrategy: Record<string, string>;
  detectionWeights: Record<string, number>;
  preferredPhraseBars: number[];
  energyProfile: string;
  transitionStyle: string;
}

export interface ProfilesResult {
  profiles: string[];
  meta: Record<string, GenreProfileMeta>;
}

export interface SettingsResult {
  engine: "basic" | "intellistem" | "onnx";
  engineModes: string[];
  gpu?: { provider: string; device: string };
  stems?: StemName[];
}

const BASE = "";

function headers(): Record<string, string> {
  const h: Record<string, string> = { "Content-Type": "application/json" };
  if (window.CUEPILOT_TOKEN) {
    h["X-Session-Token"] = window.CUEPILOT_TOKEN;
  }
  return h;
}

async function request<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: body === undefined ? "GET" : "POST",
    headers: headers(),
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status}: ${text}`);
  }
  return (await res.json()) as T;
}

export interface CueEditPayload {
  slot: number;
  type: string;
  time: number;
  label: string;
  confidence: number;
  color: string;
  locked: boolean;
  source: string;
  reason: string[];
}

export interface EditsResult {
  path: string;
  backup: string | null;
  dryRun: boolean;
  written: boolean;
  cueCount: number;
  updatedAt?: string;
  preview?: unknown;
}

export interface EditsLoadResult {
  exists: boolean;
  edits?: {
    schemaVersion: string;
    trackPath: string;
    profile: string;
    bpm: number;
    key: string;
    cues: CueEditPayload[];
    updatedAt: string;
    appVersion: string;
  };
}

export interface LearnPattern {
  slot: number;
  avgPosition: number;
  samples: number;
  roles: string[];
  trusted: boolean;
}

export interface LearnStatsResult {
  stats: { total: number; tracks: number; genres: number };
  patterns: LearnPattern[];
  minSamples: number;
}

export interface MetadataWriteResult {
  ok: boolean;
  written: boolean;
  dryRun?: boolean;
  bpm?: number;
  key?: string;
  cueCount?: number;
  backup?: string;
  error?: string;
}

export interface AnalyzeProgress {
  stage: string;
  message: string;
  pct: number;
}

export interface BatchTrackState {
  path: string;
  status: string;
  retries: number;
  error: string;
  cueplan: string;
}

export interface BatchJobStatus {
  found: boolean;
  jobId?: string;
  folder?: string;
  profile?: string;
  engine?: string;
  total?: number;
  done?: number;
  failed?: number;
  queued?: number;
  pct?: number;
  tracks?: BatchTrackState[];
  message?: string;
}

export interface PreferenceProfileItem {
  name: string;
  weights: Record<string, Record<string, number>>;
  slotStrategy: Record<string, Record<number, string>>;
}

export interface CompatibleTrack {
  trackId: string;
  keyCompatibility: string;
  bpmCompatibility: string;
  bpmDelta: number;
  score: number;
  suggestedTransition: { fromCue: string; toCue: string } | null;
  reasons: string[];
}

export interface CompatibilityResult {
  trackId: string;
  compatibleWith: CompatibleTrack[];
}

export const api = {
  health: () => request<{ ok: boolean }>("/api/health"),
  profiles: () => request<ProfilesResult>("/api/profiles"),
  settings: () => request<SettingsResult>("/api/settings"),
  setSettings: (engine?: string, stems?: StemName[]) =>
    request<SettingsResult>("/api/settings", { engine: engine || "", stems: stems ?? null }),
  scan: (folder: string, recursive = true) =>
    request<ScanResult>("/api/scan", { folder, recursive }),
  analyze: (path: string, profile = "open_format", engine?: string) =>
    request<AnalyzeResult>("/api/analyze", { path, profile, engine: engine || "" }),
  analyzeStream: async (
    path: string,
    profile: string,
    onProgress: (p: AnalyzeProgress) => void,
    engine?: string,
    stems?: StemName[]
  ): Promise<AnalyzeResult> => {
    const res = await fetch(`${BASE}/api/analyze/stream`, {
      method: "POST",
      headers: headers(),
      body: JSON.stringify({ path, profile, engine: engine || "", stems: stems ?? null }),
    });
    if (!res.ok || !res.body) {
      const text = await res.text();
      throw new Error(`HTTP ${res.status}: ${text}`);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop() || "";
      for (const block of events) {
        const lines = block.split("\n");
        const dataLine = lines.find((l) => l.startsWith("data:"));
        const evtLine = lines.find((l) => l.startsWith("event:"));
        if (!dataLine) continue;
        const evt = (evtLine?.slice(6) || "message").trim();
        const data = JSON.parse(dataLine.slice(5).trim()) as Record<string, unknown>;
        if (evt === "progress") {
          onProgress(data as unknown as AnalyzeProgress);
        } else if (evt === "done") {
          return data as unknown as AnalyzeResult;
        } else if (evt === "error") {
          throw new Error(String(data.message || "analysis failed"));
        }
      }
    }
    throw new Error("analysis stream ended without a result");
  },
  waveform: (path: string, bins = 3000) =>
    request<WaveformData>("/api/waveform", { path, bins }),
  export: (path: string, profile = "open_format") =>
    request<{ path: string; cuePlan: unknown }>("/api/export", { path, profile }),
  metadataWrite: (payload: {
    path: string;
    bpm: number;
    key: string;
    cues: CueEditPayload[];
    analysis?: { bpm: number; key: string };
    dryRun?: boolean;
  }) => request<MetadataWriteResult>("/api/metadata/write", payload),
  metadataWriteStream: async (
    payload: {
      path: string;
      bpm: number;
      key: string;
      cues: CueEditPayload[];
      analysis?: { bpm: number; key: string };
      dryRun?: boolean;
    },
    onProgress: (p: AnalyzeProgress) => void
  ): Promise<MetadataWriteResult> => {
    const res = await fetch(`${BASE}/api/metadata/write/stream`, {
      method: "POST",
      headers: headers(),
      body: JSON.stringify(payload),
    });
    if (!res.ok || !res.body) {
      const text = await res.text();
      throw new Error(`HTTP ${res.status}: ${text}`);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop() || "";
      for (const block of events) {
        const lines = block.split("\n");
        const dataLine = lines.find((l) => l.startsWith("data:"));
        const evtLine = lines.find((l) => l.startsWith("event:"));
        if (!dataLine) continue;
        const evt = (evtLine?.slice(6) || "message").trim();
        const data = JSON.parse(dataLine.slice(5).trim()) as Record<string, unknown>;
        if (evt === "progress") {
          onProgress(data as unknown as AnalyzeProgress);
        } else if (evt === "done") {
          return data as unknown as MetadataWriteResult;
        } else if (evt === "error") {
          throw new Error(String(data.message || "write failed"));
        }
      }
    }
    throw new Error("metadata stream ended without a result");
  },
  editsLoad: (path: string) => request<EditsLoadResult>("/api/edits/load", { path }),
  editsSave: (payload: {
    path: string;
    profile: string;
    bpm: number;
    key: string;
    duration: number;
    cues: CueEditPayload[];
    dryRun?: boolean;
  }) => request<EditsResult & { learnStats?: { total: number; tracks: number; genres: number } }>("/api/edits/save", payload),
  learnStats: (genre: string = "open_format") =>
    request<LearnStatsResult>("/api/learn/stats", { genre }),
  audioUrl: async (path: string): Promise<string> => {
    const res = await fetch(`${BASE}/api/audio`, {
      method: "POST",
      headers: headers(),
      body: JSON.stringify({ path }),
    });
    if (!res.ok) throw new Error(`audio HTTP ${res.status}`);
    const blob = await res.blob();
    return URL.createObjectURL(blob);
  },
  batchCreate: (payload: { folder: string; profile?: string; engine?: string; out_dir?: string; preference?: string }) =>
    request<BatchJobStatus>("/api/batch/create", payload),
  batchRunStream: async (
    jobId: string,
    onProgress: (p: AnalyzeProgress) => void,
    shouldCancel?: () => boolean
  ): Promise<BatchJobStatus> => {
    const res = await fetch(`${BASE}/api/batch/run`, {
      method: "POST",
      headers: headers(),
      body: JSON.stringify({ job_id: jobId }),
    });
    if (!res.ok || !res.body) {
      const text = await res.text();
      throw new Error(`HTTP ${res.status}: ${text}`);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (shouldCancel?.()) throw new Error("batch cancelled by user");
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop() || "";
      for (const block of events) {
        const lines = block.split("\n");
        const dataLine = lines.find((l) => l.startsWith("data:"));
        const evtLine = lines.find((l) => l.startsWith("event:"));
        if (!dataLine) continue;
        const evt = (evtLine?.slice(6) || "message").trim();
        const data = JSON.parse(dataLine.slice(5).trim()) as Record<string, unknown>;
        if (evt === "progress") {
          onProgress(data as unknown as AnalyzeProgress);
        } else if (evt === "done") {
          return data as unknown as BatchJobStatus;
        } else if (evt === "error") {
          throw new Error(String(data.message || "batch failed"));
        }
      }
    }
    throw new Error("batch stream ended without a result");
  },
  batchStatus: (jobId: string) =>
    request<BatchJobStatus>("/api/batch/status", { job_id: jobId }),
  batchCancel: (jobId: string) =>
    request<{ ok: boolean }>("/api/batch/cancel", { job_id: jobId }),
  compatibility: (tracks: unknown[], trackId: string, topN = 5, minScore = 0.5) =>
    request<CompatibilityResult>("/api/compatibility", { tracks, track_id: trackId, top_n: topN, min_score: minScore }),
  preferencesList: () => request<{ profiles: PreferenceProfileItem[] }>("/api/preferences/list"),
};
