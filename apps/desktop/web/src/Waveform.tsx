import { useCallback, useEffect, useRef, useState } from "react";
import { Cue, Section, StemCurves, StemName, WaveformData } from "./api";
import { fmt, sectionColor } from "./App";
import { useLang } from "./i18n";

interface Props {
  waveform: WaveformData;
  sections: Section[];
  cues: Cue[];
  currentTime: number;
  playing: boolean;
  onSeek: (t: number) => void;
  editing?: boolean;
  selectedSlot?: number;
  onPlaceCue?: (time: number) => void;
  cueColors?: Record<number, string>;
  stems?: StemCurves | null;
  activeStems?: StemName[];
}

const MIN_ZOOM = 1;
const MAX_ZOOM = 128;

function hexA(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

const LEGEND_SECTIONS = ["INTRO", "VOCAL", "BUILD", "DROP", "BREAKDOWN", "CHORUS", "SECOND_DROP", "OUTRO"];

export const STEM_COLORS: Record<StemName, string> = {
  vocal: "#ffd166",
  drums: "#ff6b6b",
  bass: "#6bc5ff",
  other: "#c99bff",
};

export const STEM_ORDER: StemName[] = ["vocal", "drums", "bass", "other"];

export function Waveform({ waveform, sections, cues, currentTime, playing, onSeek, editing, selectedSlot, onPlaceCue, cueColors, stems, activeStems }: Props) {
  const { t } = useLang();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const [zoom, setZoom] = useState(MIN_ZOOM);
  const [showStems, setShowStems] = useState<Record<StemName, boolean>>({
    vocal: true,
    drums: true,
    bass: true,
    other: true,
  });

  const dur = waveform.duration || 1;
  const peaks = waveform.peaks;

  // Visible window: when zoomed the view follows the playhead so it stays in
  // view while scrubbing/playing. Zoom=1 shows the whole track.
  const viewDur = Math.max(0.25, dur / zoom);
  let viewStart = currentTime - viewDur / 2;
  if (viewStart < 0) viewStart = 0;
  if (viewStart + viewDur > dur) viewStart = Math.max(0, dur - viewDur);
  const viewEnd = viewStart + viewDur;

  const timeToX = useCallback(
    (t: number, w: number) => ((t - viewStart) / viewDur) * w,
    [viewStart, viewDur]
  );
  const xToTime = useCallback(
    (x: number, w: number) => viewStart + (x / w) * viewDur,
    [viewStart, viewDur]
  );

  const zoomIn = useCallback(() => setZoom((z) => Math.min(z * 2, MAX_ZOOM)), []);
  const zoomOut = useCallback(() => setZoom((z) => Math.max(z / 2, MIN_ZOOM)), []);
  const zoomReset = useCallback(() => setZoom(MIN_ZOOM), []);

  const draw = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      canvas.width = w * dpr;
      canvas.height = h * dpr;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    // background
    ctx.fillStyle = "#f3f5f9";
    ctx.fillRect(0, 0, w, h);

    // section bands (colored by type, labeled when wide enough)
    for (const s of sections) {
      const x0 = timeToX(s.start, w);
      const x1 = timeToX(s.end, w);
      const bw = x1 - x0;
      if (bw <= 0) continue;
      const col = sectionColor(s.type);
      ctx.fillStyle = hexA(col, 0.14);
      ctx.fillRect(x0, 0, Math.max(1, bw), h);
      ctx.fillStyle = hexA(col, 0.9);
      ctx.beginPath();
      ctx.moveTo(x0, 0);
      ctx.lineTo(x0, h);
      ctx.lineWidth = 1;
      ctx.strokeStyle = hexA(col, 0.25);
      ctx.stroke();
      if (bw > 46) {
        ctx.fillStyle = hexA(col, 0.95);
        ctx.font = "bold 10px 'Space Grotesk', sans-serif";
        ctx.textAlign = "left";
        ctx.fillText(s.type, x0 + 6, 14);
      }
    }

    // waveform bars
    const mid = h / 2;
    const maxAmp = Math.max(...peaks.map((p) => Math.max(Math.abs(p.min), Math.abs(p.max))), 0.0001);
    const scale = h / 2 / maxAmp;
    const playedX = timeToX(currentTime, w);

    // visible peaks only (perf + clarity when zoomed)
    const lo = Math.max(0, Math.floor(((viewStart - 0.001) / dur) * peaks.length));
    const hi = Math.min(peaks.length, Math.ceil(((viewEnd + 0.001) / dur) * peaks.length));

    // stem overlay bands (colored, toggleable)
    if (stems && activeStems && activeStems.length > 1) {
      const bandH = h * 0.34;
      for (const name of STEM_ORDER) {
        if (!showStems[name]) continue;
        if (!activeStems.includes(name)) continue;
        const pts = stems[name];
        if (!pts || pts.length < 2) continue;
        const col = STEM_COLORS[name];
        ctx.beginPath();
        let started = false;
        for (const p of pts) {
          if (p.t < viewStart - 0.01 || p.t > viewEnd + 0.01) continue;
          const x = timeToX(p.t, w);
          const y = h - 4 - (Math.min(1, Math.max(0, p.v)) * bandH * 0.9);
          if (!started) {
            ctx.moveTo(x, y);
            started = true;
          } else {
            ctx.lineTo(x, y);
          }
        }
        ctx.strokeStyle = hexA(col, 0.55);
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.fillStyle = hexA(col, 0.10);
        ctx.lineTo(timeToX(Math.min(viewEnd, dur), w), h - 4);
        ctx.lineTo(timeToX(Math.max(viewStart, 0), w), h - 4);
        ctx.closePath();
        ctx.fill();
      }
    }

    for (let i = lo; i < hi; i++) {
      const x = timeToX(peaks[i].t, w);
      const yTop = mid - Math.max(0, peaks[i].max) * scale;
      const yBot = mid - Math.min(0, peaks[i].min) * scale;
      const hgt = Math.max(1, yBot - yTop);
      const isPlayed = x <= playedX;
      ctx.fillStyle = isPlayed ? "#0ea5c5" : "#c4cad6";
      ctx.fillRect(x, yTop, Math.max(1, barWidth(w, peaks.length)) + 0.5, hgt);
    }

    // playhead
    if (currentTime >= viewStart && currentTime <= viewEnd) {
      const x = timeToX(currentTime, w);
      ctx.fillStyle = "#e11d48";
      ctx.fillRect(x - 1, 0, 2, h);
      ctx.beginPath();
      ctx.arc(x, 0, 5, 0, Math.PI * 2);
      ctx.fill();
    }

    // cue markers
    for (const c of cues) {
      if (c.time < viewStart || c.time > viewEnd) continue;
      const x = timeToX(c.time, w);
      const col = c.color || (cueColors && cueColors[c.slot]) || sectionColor(c.type);
      const isSelected = editing && selectedSlot === c.slot;
      ctx.strokeStyle = col;
      ctx.lineWidth = isSelected ? 3 : 2;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
      ctx.setLineDash([]);
      // badge
      ctx.fillStyle = isSelected ? "#ffffff" : col;
      ctx.strokeStyle = isSelected ? col : "rgba(0,0,0,0.15)";
      ctx.lineWidth = isSelected ? 2 : 1;
      ctx.beginPath();
      ctx.roundRect(x + 3, 6, 18, 18, 4);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = isSelected ? col : "#ffffff";
      ctx.font = "bold 11px 'JetBrains Mono', monospace";
      ctx.textAlign = "center";
      ctx.fillText(String(c.slot), x + 12, 19);
    }

    // editing: ghost guide line at the selected slot color (no cue placed)
    if (editing && selectedSlot && !cues.some((c) => c.slot === selectedSlot)) {
      const col = (cueColors && cueColors[selectedSlot]) || "#22c55e";
      ctx.fillStyle = hexA(col, 0.9);
      ctx.font = "bold 10px 'JetBrains Mono', monospace";
      ctx.textAlign = "left";
      ctx.fillText(`+ slot ${selectedSlot}`, 8, h - 8);
    }
  };

  const barWidth = (w: number, n: number) => Math.max(1, w / Math.max(1, n));

  useEffect(() => {
    draw();
  }, [waveform, sections, cues, currentTime, zoom, viewStart, viewEnd, stems, showStems, activeStems]);

  useEffect(() => {
    const tick = () => {
      draw();
      rafRef.current = requestAnimationFrame(tick);
    };
    if (playing) {
      rafRef.current = requestAnimationFrame(tick);
    }
    return () => cancelAnimationFrame(rafRef.current);
  }, [playing, currentTime, zoom, viewStart, viewEnd, stems, showStems, activeStems]);

  const onPointer = (e: React.PointerEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    const time = xToTime(ratio * rect.width, rect.width);
    if (editing && onPlaceCue) {
      onPlaceCue(time);
    } else {
      onSeek(time);
    }
  };

  return (
    <div className="wave-block">
      <div className="wave-toolbar">
        <span className="wave-zoom-label">{t("zoom")}</span>
        <button className="wave-btn" onClick={zoomOut} title={t("zoomOut")} disabled={zoom <= MIN_ZOOM}>
          −
        </button>
        <button className="wave-btn" onClick={zoomReset} title={t("zoomReset")}>
          {zoom === MIN_ZOOM ? "100%" : `${Math.round(zoom * 100)}%`}
        </button>
        <button className="wave-btn" onClick={zoomIn} title={t("zoomIn")} disabled={zoom >= MAX_ZOOM}>
          +
        </button>
        <span className="wave-zoom-range">
          {fmt(viewStart)} – {fmt(viewEnd)}
        </span>
      </div>

      {stems && activeStems && activeStems.length > 1 && (
        <div className="wave-stem-bar" title={t("stemsOverlayHint")}>
          <span className="wave-stem-label">{t("stemLabel")}</span>
          {STEM_ORDER.filter((s) => activeStems.includes(s)).map((name) => {
            const on = showStems[name];
            return (
              <button
                key={name}
                className={`stem-chip ${on ? "is-on" : ""}`}
                style={{ "--stem": STEM_COLORS[name] } as React.CSSProperties}
                onClick={() =>
                  setShowStems((prev) => ({ ...prev, [name]: !prev[name] }))
                }
              >
                <i className="stem-swatch" />
                {t(`stem${name === "other" ? "Other" : name.charAt(0).toUpperCase() + name.slice(1)}`)}
              </button>
            );
          })}
        </div>
      )}

      <div className="wave-wrap" style={{ cursor: editing ? "pointer" : "crosshair" }}>
        <canvas
          ref={canvasRef}
          className="waveform"
          onPointerDown={onPointer}
          onPointerMove={(e) => e.buttons === 1 && onPointer(e)}
        />
        <div className="wave-time">
          <span>{fmt(viewStart)}</span>
          <span>{fmt(viewEnd)}</span>
        </div>
      </div>

      <div className="wave-legend">
        <span className="wave-legend-title">{t("legend")}</span>
        {LEGEND_SECTIONS.map((s) => (
          <span className="legend-item" key={s}>
            <i className="legend-swatch" style={{ background: sectionColor(s) }} />
            {s}
          </span>
        ))}
        <span className="legend-item">
          <i className="legend-swatch" style={{ background: "#0ea5c5" }} />
          {t("played")}
        </span>
        <span className="legend-item">
          <i className="legend-swatch" style={{ background: "#c4cad6" }} />
          {t("unplayed")}
        </span>
        <span className="legend-item">
          <i className="legend-swatch" style={{ background: "#e11d48" }} />
          {t("playhead")}
        </span>
        <span className="legend-item">
          <i className="legend-swatch" style={{ background: "transparent", border: "2px dashed #9aa1ab" }} />
          {t("hotcueLegend")}
        </span>
      </div>
    </div>
  );
}
