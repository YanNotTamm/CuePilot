import { useLang } from "./i18n";

export function EmptyState({ scanning }: { scanning: boolean }) {
  const { t } = useLang();
  return (
    <div className="empty-state">
      <div className="empty-glyph">♪</div>
      <h2>{scanning ? t("scanningTitle") : t("readyTitle")}</h2>
      <p>{scanning ? t("scanningSub") : t("readySub")}</p>
    </div>
  );
}
