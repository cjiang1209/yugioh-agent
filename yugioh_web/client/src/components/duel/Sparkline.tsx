/**
 * A dependency-free SVG trace of a value series, symmetric about zero.
 *
 * The y-domain autoscales to the largest magnitude in the series, which makes
 * small movements visible; the printed bound is what stops a +/-0.05 wobble
 * from reading as a swing.
 */

/** Smallest bound to scale against, so a flat-zero series still projects. */
const MIN_BOUND = 0.01;

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
        width={WIDTH}
        height={HEIGHT}
        role="img"
        aria-label={`Value trace, ${values.length} points, scale plus or minus ${bound.toFixed(2)}`}
        style={{ flexShrink: 0 }}
      >
        <line
          x1={0}
          y1={midY}
          x2={WIDTH}
          y2={midY}
          stroke="rgba(255,255,255,0.15)"
          strokeWidth={1}
        />
        {values.length > 0 && (
          <polyline
            points={points}
            fill="none"
            stroke="var(--neon-cyan)"
            strokeWidth={1.5}
            strokeLinejoin="round"
          />
        )}
      </svg>
      <span
        style={{
          fontFamily: "'Share Tech Mono', monospace",
          fontSize: "0.75rem",
          color: "var(--text-muted)",
          whiteSpace: "nowrap",
        }}
      >
        ±{bound.toFixed(2)}
      </span>
    </div>
  );
}
