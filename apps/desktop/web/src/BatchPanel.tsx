import { BatchJobStatus } from "./api";
import { useLang } from "./i18n";

interface Props {
  status: BatchJobStatus;
  onStop: () => void;
}

export function BatchPanel({ status, onStop }: Props) {
  const { t } = useLang();
  const total = status.total ?? 0;
  const done = status.done ?? 0;
  const failed = status.failed ?? 0;
  const pct = status.pct ?? 0;
  const remaining = total - done - failed;

  const title =
    remaining > 0
      ? t("batchTitle")
      : failed > 0
        ? t("batchFinishedFail")
        : t("batchFinished");

  return (
    <div className="batch-panel">
      <div className="batch-head">
        <strong>{title}</strong>
        <span className="pct">{Math.round(pct)}%</span>
      </div>
      <div className="batch-bar">
        <span className="batch-bar-fill" style={{ width: `${Math.min(100, pct)}%` }} />
      </div>
      <div className="batch-status">
        {done} {t("batchDone").toLowerCase()} · {failed} {t("batchFailed").toLowerCase()} ·{" "}
        {remaining} {t("batchQueued").toLowerCase()}
      </div>
      {remaining > 0 && (
        <button className="batch-stop" onClick={onStop} style={{ marginTop: 10 }}>
          {t("batchCancel")}
        </button>
      )}
    </div>
  );
}