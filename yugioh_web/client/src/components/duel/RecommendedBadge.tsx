/**
 * Shared "recommended by AI Assist" visual tokens.
 *
 * The amber ring (border + glow) marks a recommended action's clickable element;
 * the star badge (below) is pinned to its corner. Both live here so the whole
 * "recommended" look has one source of truth.
 */
export const RECOMMENDED_COLOR = "#ffb020";
export const RECOMMENDED_BORDER = `1px solid ${RECOMMENDED_COLOR}`;
export const RECOMMENDED_SHADOW = "0 0 8px rgba(255,176,32,0.6)";

/**
 * Fill and glow for AI-assist *controls* — the DeckSelector toggle and the
 * board autoplay pill. Softer than RECOMMENDED_SHADOW above, which rings an
 * individual recommended action rather than a control.
 */
export const RECOMMENDED_BACKGROUND = "rgba(255,176,32,0.1)";
export const RECOMMENDED_GLOW = "0 0 12px rgba(255,176,32,0.25)";

/**
 * Amber star badge marking the action recommended by AI Assist.
 *
 * Renders unconditionally; callers gate on their own `isRecommended` flag and
 * supply the positioning context (a `position: relative` wrapper).
 */
export function RecommendedBadge() {
  return (
    <span
      title="Recommended by AI Assist"
      style={{
        position: "absolute",
        top: "-5px",
        left: "-5px",
        width: "18px",
        height: "18px",
        borderRadius: "50%",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontSize: "0.6rem",
        lineHeight: 1,
        background: RECOMMENDED_COLOR,
        color: "#1a1200",
        border: "1px solid #ffd67a",
        boxShadow: "0 0 8px rgba(255,176,32,0.9)",
      }}
    >
      ★
    </span>
  );
}
