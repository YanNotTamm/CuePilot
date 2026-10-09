import { useLang } from "./i18n";

interface Props {
  folder: string;
  setFolder: (v: string) => void;
  scanning: boolean;
  onPickFolder: () => void;
  onScan: () => void;
}

export function FolderStep({ folder, setFolder, scanning, onPickFolder, onScan }: Props) {
  const { t } = useLang();
  return (
    <div className="welcome">
      <div className="welcome-hero">
        <h2>{t("welcomeTitle")}</h2>
        <p>{t("welcomeSub")}</p>

        <button className="big-pick-btn" onClick={onPickFolder} disabled={scanning}>
          <span>📁</span>
          {t("pickFolderButton")}
        </button>

        <p className="or-type">{t("orTypePath")}</p>
        <div className="path-row">
          <input
            value={folder}
            onChange={(e) => setFolder(e.target.value)}
            placeholder={t("folderPlaceholder")}
            onKeyDown={(e) => e.key === "Enter" && onScan()}
          />
          <button className="browse-btn" onClick={onScan} disabled={scanning || !folder.trim()}>
            {scanning ? t("scanning") : t("scanButton")}
          </button>
        </div>
      </div>

      <div className="how-grid">
        <div className="how-card">
          <div className="how-icon">📁</div>
          <h3>{t("how1Title")}</h3>
          <p>{t("how1Desc")}</p>
        </div>
        <div className="how-card">
          <div className="how-icon">🎧</div>
          <h3>{t("how2Title")}</h3>
          <p>{t("how2Desc")}</p>
        </div>
        <div className="how-card">
          <div className="how-icon">💾</div>
          <h3>{t("how3Title")}</h3>
          <p>{t("how3Desc")}</p>
        </div>
      </div>

      <div className="explainer">
        <h3>{t("whatIsHotcueTitle")}</h3>
        <p>{t("whatIsHotcueDesc")}</p>
      </div>
    </div>
  );
}