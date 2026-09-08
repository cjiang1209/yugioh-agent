/**
 * A dependency-free SVG trace of a value series, symmetric about zero.
 *
 * The y-domain autoscales to the largest magnitude in the series, which makes
 * small movements visible; the printed bound is what stops a +/-0.05 wobble
 * from reading as a swing.
 */

/** Smallest bound to scale against, so a flat-zero series still projects. */
const MIN_BOUND = 0.01;

/**
 * The drawing's own coordinate space, not its rendered size. The element
 * stretches to whatever width the panel gives it and the viewBox maps these
 * x-coordinates onto that, so the trace always spans the full panel. Height is
 * rendered one-to-one, so only the x-axis stretches.
 */
const WIDTH = 200;
const HEIGHT = 88;

function sparklineBound(values: number[]): number {
  return Math.max(MIN_BOUND, ...values.map(Math.abs));
}

interface SparklineProps {
  /** Oldest first. */
  values: number[];
}

export function Sparkline({ values }: SparklineProps) {
  const bound = sparklineBound(values);
  const midY = HEIGHT / 2;

  // A single point has no span to divide across, so pin it mid-width.
  const xAt = (i: number) =>
    values.length > 1 ? (i / (values.length - 1)) * WIDTH : WIDTH / 2;
  const yAt = (v: number) => midY - (v / bound) * midY;

  const points = values.map((v, i) => `${xAt(i)},${yAt(v)}`).join(" ");

  return (
    <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        preserveAspectRatio="none"
        height={HEIGHT}
        role="img"
        aria-label={`Value trace, ${values.length} points, scale plus or minus ${bound.toFixed(2)}`}
        style={{ flex: "1 1 0", minWidth: 0 }}
      >
        {/* The x-axis stretch would thin these out with it, so the strokes opt
            out of the viewBox scaling and stay the width they ask for. */}
        <line
          x1={0}
          y1={midY}
          x2={WIDTH}
          y2={midY}
          stroke="rgba(255,255,255,0.15)"
          strokeWidth={1}
          vectorEffect="non-scaling-stroke"
        />
        {values.length > 0 && (
          <polyline
            points={points}
            fill="none"
            stroke="var(--neon-cyan)"
            strokeWidth={1.5}
            strokeLinejoin="round"
            vectorEffect="non-scaling-stroke"
          />
        )}
      </svg>
      <span
        style={{
          fontFamily: "var(--ui-font-mono)",
          fontSize: "var(--ui-text-md)",
          color: "var(--text-muted)",
          whiteSpace: "nowrap",
        }}
      >
        ±{bound.toFixed(2)}
      </span>
    </div>
  );
}
