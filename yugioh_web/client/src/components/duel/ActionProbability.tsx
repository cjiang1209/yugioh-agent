/**
 * The policy's probability for one action, pinned to the bottom-left corner
 * of its container.
 *
 * The same corner on every panel, so a reader finds it in one place whatever
 * the prompt looks like. Bottom-left is the corner nothing else claims: the
 * recommended star sits top-left, the location badge bottom-right, and a
 * card's attribute icon is printed in the top-right of its art. Renders
 * nothing without a value, so callers can hand it a lookup that may miss;
 * they supply the positioning context (a `position: relative` container), as
 * they already do for the star.
 *
 * Out of flow, so showing it or hiding it never moves anything around it.
 */
export function ActionProbability({ value }: { value?: number | null }) {
  if (value == null) return null;

  return (
    <span
      data-testid="action-probability"
      title={`Policy probability ${(value * 100).toFixed(1)}%`}
      style={{
        position: "absolute",
        bottom: "2px",
        left: "2px",
        padding: "0 3px",
        borderRadius: "2px",
        background: "rgba(6,10,20,0.75)",
        fontFamily: "'Share Tech Mono', monospace",
        fontSize: "0.55rem",
        lineHeight: 1.3,
        color: "#c8d8e8",
      }}
    >
      {Math.round(value * 100)}%
    </span>
  );
}
