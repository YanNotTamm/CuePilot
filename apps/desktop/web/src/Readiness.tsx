import { GigReadiness, QcReport } from "./api";
import { useLang } from "./i18n";

interface Props {
  readiness?: GigReadiness;
  qc?: QcReport;
}

export function ReadinessBadge({ readiness, qc }: Props) {
  const { t } = useLang();
  if (!readiness) return null;

  const bucketLabel =
    readiness.bucket === "ready"
      ? t("readinessReady")
      : readiness.bucket === "needs_review"
        ? t("readinessNeedsReview")
        : t("readinessNotAnalyzed");

  return (
    <span className={`readiness-badge ${readiness.bucket}`} title={t("readinessHint")}>
      <span className="readiness-dot" />
      {bucketLabel}
      {readiness.bucket !== "not_analyzed" && (
        <b className="readiness-score">{Math.round(readiness.score)}</b>
      )}
      {qc && !qc.ok && <span className="readiness-qc-warn">!</span>}
    </span>
  );
}

export function ReadinessPanel({ readiness, qc }: Props) {
  const { t } = useLang();
  if (!readiness) return null;
  const comp = readiness.components || {};
  const rows: Array<[string, string, number]> = [
    [t("readinessCueConfidence"), "cue_confidence", comp.cue_confidence ?? 0],
    [t("readinessAudioQuality"), "audio_quality", comp.audio_quality ?? 0],
    [t("readinessStructure"), "structure", comp.structure ?? 0],
    [t("readinessStability"), "correction_stability", comp.correction_stability ?? 0],
  ];

  return (
    <div className="readiness-panel">
      <div className="readiness-head">
        <div>
          <span className="readiness-title">{t("readinessTitle")}</span>
          <p className="readiness-hint">{t("readinessHint")}</p>
        </div>
        <div className={`readiness-score-big ${readiness.bucket}`}>
          {Math.round(readiness.score)}
        </div>
      </div>

      <div className="readiness-bars">
        {rows.map(([label, key, val]) => (
          <div className="readiness-row" key={key}>
            <span className="readiness-row-label">{label}</span>
            <div className="readiness-bar">
              <i className={`fill ${key}`} style={{ width: `${Math.round(val * 100)}%` }} />
            </div>
            <span className="readiness-row-val">{Math.round(val * 100)}</span>
          </div>
        ))}
      </div>

      {readiness.reasons && readiness.reasons.length > 0 && (
        <div className="readiness-reasons">
          {readiness.reasons.map((r, i) => (
            <span className="reason-chip" key={i}>
              {r}
            </span>
          ))}
        </div>
      )}

      {qc && (
        <div className="qc-block">
          <div className="qc-head">
            <span>{t("qcTitle")}</span>
            <b className={`qc-score ${qc.ok ? "ok" : "warn"}`}>
              {t("qcScore")}: {Math.round(qc.score * 100)}
            </b>
          </div>
          {qc.ok && qc.issues.length === 0 && <p className="qc-ok">{t("qcOk")}</p>}
          {qc.issues.length > 0 && (
            <ul className="qc-issues">
              {qc.issues.map((issue, i) => (
                <li key={i} className={`severity-${issue.severity}`}>
                  <b>{issue.label}</b>
                  <span>{issue.detail}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}