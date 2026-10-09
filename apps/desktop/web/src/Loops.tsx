import { SavedLoop } from "./api";
import { useLang } from "./i18n";

function fmt(t: number): string {
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  const ms = Math.floor((t % 1) * 1000);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${String(ms).padStart(3, "0")}`;
}

interface Props {
  loops: SavedLoop[];
  onJump: (time: number) => void;
}

export function LoopsPanel({ loops, onJump }: Props) {
  const { t } = useLang();
  if (!loops || loops.length === 0) return null;

  return (
    <div className="loops-panel">
      <div className="loops-head">
        <span>{t("loopsTitle")}</span>
        <span className="loops-hint">{t("loopsHint")}</span>
      </div>
      <div className="loops-grid">
        {loops.map((loop, i) => (
          <button
            key={i}
            className="loop-chip"
            onClick={() => onJump(loop.start)}
            title={`${loop.bars} bars · ${fmt(loop.start)} → ${fmt(loop.end)} · ${t("loopsSeamless")}: ${Math.round(loop.seamlessScore * 100)}%`}
          >
            <span className={`loop-use ${loop.useCase}`}>{loop.bars} BAR</span>
            <span className="loop-range">
              {fmt(loop.start)} – {fmt(loop.end)}
            </span>
            <span className="loop-score">{Math.round(loop.seamlessScore * 100)}%</span>
          </button>
        ))}
      </div>
    </div>
  );
}