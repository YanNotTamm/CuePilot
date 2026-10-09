import { GenreProfileMeta } from "./api";
import { useLang } from "./i18n";

interface Props {
  trackName: string;
  profiles: string[];
  meta: Record<string, GenreProfileMeta>;
  onPick: (profile: string) => void;
  onCancel: () => void;
}

export function GenreModal({ trackName, profiles, meta, onPick, onCancel }: Props) {
  const { t, lang } = useLang();
  return (
    <div className="modal-backdrop" onClick={onCancel}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h2>{t("chooseGenre")}</h2>
        <p className="modal-sub">
          <b>{trackName}</b>
        </p>
        <p className="modal-hint">{t("genreHint")}</p>
        <div className="genre-grid">
          {profiles.map((p) => {
            const m = meta[p];
            return (
              <button key={p} className="genre-chip" onClick={() => onPick(p)}>
                <span className="genre-name">{m?.displayName || p}</span>
                <span className="genre-desc">
                  {(lang === "id" ? m?.descriptionId : m?.description) || ""}
                </span>
              </button>
            );
          })}
        </div>
        <button className="modal-close" onClick={onCancel} style={{ marginTop: 18 }}>
          {t("close")}
        </button>
      </div>
    </div>
  );
}