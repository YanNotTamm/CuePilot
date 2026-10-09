import { ScanResult, TrackItem, BatchJobStatus } from "./api";
import { useLang } from "./i18n";
import { BatchPanel } from "./BatchPanel";

interface Props {
  scan: ScanResult | null;
  selectedId?: string;
  onSelect: (t: TrackItem) => void;
  analyzingAll: boolean;
  canBatch: boolean;
  onBatch: () => void;
  batchStatus: BatchJobStatus | null;
  onStopBatch: () => void;
  onPickFolder: () => void;
}

export function TrackList({
  scan,
  selectedId,
  onSelect,
  analyzingAll,
  canBatch,
  onBatch,
  batchStatus,
  onStopBatch,
  onPickFolder,
}: Props) {
  const { t } = useLang();
  return (
    <aside className="tracklist">
      {scan && (
        <>
          <div className="tracklist-head">
            <h2>{t("tracksTitle")}</h2>
            <span className="tracklist-count">{scan.total}</span>
          </div>
          <div className="tracklist-scroll">
            {scan.tracks.map((tr, i) => (
              <button
                key={tr.id}
                className={`track ${selectedId === tr.id ? "active" : ""}`}
                onClick={() => onSelect(tr)}
                title={tr.path}
              >
                <span className="track-idx">{String(i + 1).padStart(2, "0")}</span>
                <span className="track-body">
                  <span className="track-name">
                    {tr.artist ? `${tr.artist} - ${tr.title}` : tr.title || tr.filename}
                  </span>
                  <span className="track-meta">
                    {tr.duration.toFixed(1)}s · {tr.status.toLowerCase()}
                  </span>
                </span>
              </button>
            ))}
          </div>
          <div className="batch-actions">
            <button className="primary-btn" onClick={onBatch} disabled={!canBatch || analyzingAll}>
              {analyzingAll ? t("analyzingTitle") : t("analyzeAllButton")}
            </button>
            <p className="batch-hint">{t("analyzeAllHint")}</p>
            {batchStatus && batchStatus.found && (
              <BatchPanel
                status={batchStatus}
                onStop={onStopBatch}
              />
            )}
          </div>
        </>
      )}
      {!scan && (
        <div className="tracklist-empty">
          <p>{t("pickATrack")}</p>
          <p className="hint">
            {t("pickATrackHint")}
            {" "}
            <button className="back-btn" onClick={onPickFolder}>
              {t("pickFolder")}
            </button>
          </p>
        </div>
      )}
    </aside>
  );
}