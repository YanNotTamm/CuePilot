import { Cue } from "./api";
import { fmt, sectionColor } from "./App";
import { useLang } from "./i18n";

interface Props {
  cues: Cue[];
  onJump: (c: Cue) => void;
  editing?: boolean;
  onDelete?: (slot: number) => void;
  onToggleLock?: (slot: number) => void;
}

const FRIENDLY_TYPE: Record<string, string> = {
  START: "cueTypeStart",
  INTRO: "cueTypeIntro",
  MIX_IN: "cueTypeMixIn",
  VOCAL: "cueTypeVocal",
  HOOK: "cueTypeHook",
  MELODY: "cueTypeMelody",
  GROOVE: "cueTypeGroove",
  PRE_GROOVE: "cueTypePreGroove",
  BUILD: "cueTypeBuild",
  PRE_DROP: "cueTypePreDrop",
  DROP: "cueTypeDrop",
  CLIMAX: "cueTypeClimax",
  TRANSITION: "cueTypeTransition",
  CHORUS: "cueTypeChorus",
  BREAKDOWN: "cueTypeBreakdown",
  ENERGY_PEAK: "cueTypeEnergyPeak",
  SECOND_DROP: "cueTypeSecondDrop",
  OUTRO: "cueTypeOutro",
  MIX_OUT: "cueTypeMixOut",
  ALTERNATIVE: "cueTypeAlternative",
  CUSTOM: "cueTypeCustom",
};

export function CueGrid({ cues, onJump, editing, onDelete, onToggleLock }: Props) {
  const { t } = useLang();
  if (cues.length === 0) return null;
  return (
    <div className="cuegrid">
      {cues.map((c) => {
        const col = c.color || sectionColor(c.type);
        const typeKey = FRIENDLY_TYPE[c.type] || "cueTypeCustom";
        const typeLabel = t(typeKey);
        return (
          <div
            key={c.slot}
            className={`cue-card ${c.locked ? "is-locked" : ""}`}
            style={{ "--cue": col } as React.CSSProperties}
          >
            <button className="cue-card-main" onClick={() => onJump(c)} title={t("listenHere")}>
              <div className="cue-slot" style={{ background: col }}>
                {c.slot}
              </div>
              <div className="cue-body">
                <div className="cue-title">
                  {c.label && c.label !== c.type ? c.label : typeLabel}
                </div>
                <div className="cue-time">{fmt(c.time)}</div>
                <div className="cue-meter">
                  <span className="meter-track">
                    <span className="meter-fill" style={{ width: "100%", background: col }} />
                  </span>
                  <span className="cue-conf">{c.source === "user" ? t("cueTypeCustom") : typeLabel}</span>
                </div>
              </div>
            </button>
            {editing && (
              <div className="cue-actions">
                <button
                  className="cue-act"
                  onClick={() => onToggleLock?.(c.slot)}
                  title={c.locked ? t("unlock") : t("lock")}
                >
                  {c.locked ? "🔒" : "🔓"}
                </button>
                <button className="cue-act" onClick={() => onDelete?.(c.slot)} title={t("deleteCue")}>
                  🗑
                </button>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}