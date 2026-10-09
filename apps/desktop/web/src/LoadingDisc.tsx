import { AnalyzeProgress } from "./api";
import { useLang } from "./i18n";

interface Props {
  variant?: "analyzing" | "scanning";
  progress?: AnalyzeProgress;
}

export function LoadingDisc({ variant = "analyzing", progress }: Props) {
  const { t } = useLang();
  const isScan = variant === "scanning";

  const title = progress
    ? t(`stage_${progress.stage}`, t(`mst_${progress.stage}`, progress.message))
    : isScan
      ? t("scanningTitle")
      : t("analyzingTitle");
  const sub = isScan
    ? t("scanningSub")
    : progress
      ? progress.message
      : t("analyzingSub");

  return (
    <div className={`dj-loading ${isScan ? "scanning" : ""}`}>
      <div className="dj-decks">
        <div className="deck">
          <div className="vinyl vinyl-a">
            <div className="vinyl-hole" />
            <div className="vinyl-label">CP</div>
          </div>
          <div className="tonearm tonearm-a" />
          <div className="deck-spin-slow">A</div>
        </div>

        <div className="mixer">
          <div className="xfader" />
          <div className="mixer-knobs">
            <span />
            <span />
            <span />
          </div>
        </div>

        <div className="deck">
          <div className="vinyl vinyl-b">
            <div className="vinyl-hole" />
            <div className="vinyl-label">CP</div>
          </div>
          <div className="tonearm tonearm-b" />
          <div className="deck-spin-fast">B</div>
        </div>
      </div>

      <div className="dj-wave">
        {Array.from({ length: 32 }).map((_, i) => (
          <span
            key={i}
            className="dj-wave-bar"
            style={{
              animationDelay: `${(i % 8) * 110}ms`,
              height: `${24 + ((i * 37) % 76)}%`,
            }}
          />
        ))}
      </div>

      <p className="loading-text">{title}</p>
      <p className="loading-sub">{sub}</p>
      {progress && progress.pct > 0 && (
        <div className="loading-meter">
          <span className="loading-meter-fill" style={{ width: `${progress.pct}%` }} />
        </div>
      )}
    </div>
  );
}
