import { GenreProfileMeta, PreferenceProfileItem, StemName } from "./api";
import { Lang, useLang } from "./i18n";

interface Props {
  engine: "basic" | "intellistem" | "onnx";
  gpuInfo: { provider: string; device: string } | null;
  profiles: string[];
  profileMeta: Record<string, GenreProfileMeta>;
  activeProfile: string;
  preferences: PreferenceProfileItem[];
  activePreference: string;
  activeStems: StemName[];
  onEngine: (e: "basic" | "intellistem" | "onnx") => void;
  onStems: (s: StemName[]) => void;
  onLang: (l: Lang) => void;
  onProfile: (p: string) => void;
  onPreference: (p: string) => void;
  onClose: () => void;
}

const STEM_OPTIONS: { name: StemName; color: string }[] = [
  { name: "vocal", color: "#ffd166" },
  { name: "drums", color: "#ff6b6b" },
  { name: "bass", color: "#6bc5ff" },
  { name: "other", color: "#c99bff" },
];

export function SettingsPanel({
  engine,
  gpuInfo,
  profiles,
  profileMeta,
  activeProfile,
  preferences,
  activePreference,
  activeStems,
  onEngine,
  onStems,
  onLang,
  onProfile,
  onPreference,
  onClose,
}: Props) {
  const { t, lang } = useLang();
  const toggleStem = (name: StemName) => {
    const next = activeStems.includes(name)
      ? activeStems.filter((s) => s !== name)
      : [...activeStems, name];
    if (next.length === 0) return; // keep at least one stem
    onStems(next);
  };
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h2>{t("settingsTitle")}</h2>
        <div className="settings-list">
          <div className="settings-row">
            <div className="settings-label">{t("engineLabel")}</div>
            <p className="settings-hint">{t("engineBasicHint")}</p>
            <div className="seg">
              <button
                className={`seg-opt ${engine === "basic" ? "is-on" : ""}`}
                onClick={() => onEngine("basic")}
              >
                <b>{t("engineBasic")}</b>
                <span>{t("engineBasicHint")}</span>
              </button>
              <button
                className={`seg-opt ${engine === "intellistem" ? "is-on" : ""}`}
                onClick={() => onEngine("intellistem")}
              >
                <b>{t("engineIntelli")}</b>
                <span>{t("engineIntelliHint")}</span>
              </button>
              <button
                className={`seg-opt ${engine === "onnx" ? "is-on" : ""}`}
                onClick={() => onEngine("onnx")}
                title={gpuInfo && gpuInfo.provider !== "cpu" ? undefined : t("onnxConfirmNoGpu")}
              >
                <b>{t("engineOnnx")}</b>
                <span>{t("engineOnnxHint")}</span>
              </button>
            </div>
          </div>

          <div className="settings-row">
            <div className="settings-label">{t("stemsAnalysisLabel")}</div>
            <p className="settings-hint">{t("stemsAnalysisHint")}</p>
            <div className="stem-settings">
              {STEM_OPTIONS.map(({ name, color }) => {
                const on = activeStems.includes(name);
                return (
                  <button
                    key={name}
                    className={`stem-chip ${on ? "is-on" : ""}`}
                    style={{ "--stem": color } as React.CSSProperties}
                    onClick={() => toggleStem(name)}
                  >
                    <i className="stem-swatch" />
                    {t(`stem${name === "other" ? "Other" : name.charAt(0).toUpperCase() + name.slice(1)}`)}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="settings-row">
            <div className="settings-label">{t("preferenceProfile")}</div>
            <p className="settings-hint">{t("preferenceHint")}</p>
            <select
              className="select-field"
              value={activePreference}
              onChange={(e) => onPreference(e.target.value)}
            >
              <option value="">{t("noPreference")}</option>
              {preferences.map((p) => (
                <option key={p.name} value={p.name}>
                  {p.name}
                </option>
              ))}
            </select>
          </div>

          <div className="settings-row">
            <div className="settings-label">{t("changeGenre")}</div>
            <p className="settings-hint">{t("changeGenreHint")}</p>
            <select
              className="select-field"
              value={activeProfile}
              onChange={(e) => onProfile(e.target.value)}
            >
              {profiles.map((p) => (
                <option key={p} value={p}>
                  {profileMeta[p]?.displayName || p}
                </option>
              ))}
            </select>
          </div>

          <div className="settings-row">
            <div className="settings-label">{t("languageLabel")}</div>
            <div className="seg">
              <button
                className={`seg-opt ${lang === "id" ? "is-on" : ""}`}
                onClick={() => onLang("id")}
              >
                <b>{t("indonesian")}</b>
              </button>
              <button
                className={`seg-opt ${lang === "en" ? "is-on" : ""}`}
                onClick={() => onLang("en")}
              >
                <b>{t("english")}</b>
              </button>
            </div>
          </div>
        </div>

        <button className="modal-close" onClick={onClose} style={{ marginTop: 22 }}>
          {t("close")}
        </button>
      </div>
    </div>
  );
}