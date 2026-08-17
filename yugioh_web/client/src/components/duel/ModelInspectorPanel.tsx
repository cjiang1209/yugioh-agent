import { Sparkline } from "./Sparkline";

/**
 * Developer read-out of what the recommender model thinks of the position.
 *
 * Shows the value head's raw output and its history across the duel's prompts.
 * Deliberately uncalibrated: this is for inspecting a checkpoint, so the
 * number shown is the number the network produced.
 */

/**
 * The one size for this panel's own text -- label, value, prompt count -- set
 * on the container and inherited. Matches the sibling panels in the same
 * column rather than the denser action list, so the left column reads as one
 * scale. Colour, not size, is what marks the value out from the text around it.
 */
const TEXT_SIZE = "0.75rem";

const VALUE_CAVEAT =
  "Value head output for your side: the model's estimate of the discounted " +
  "return — the terminal win or loss plus whatever reward shaping it trained " +
  "with. NOT a win probability, and not comparable across checkpoints " +
  "trained with different discounting or shaping.";

interface ModelInspectorPanelProps {
  /** One sample per prompt so far this duel, oldest first. */
  trace: number[];
}

export function ModelInspectorPanel({ trace }: ModelInspectorPanelProps) {
  // The newest sample is the current evaluation, so an empty trace is the only
  // way to have none.
  const value = trace.length > 0 ? trace[trace.length - 1] : null;

  return (
    <div
      className="h-full overflow-y-auto p-2 flex flex-col gap-1"
      style={{
        fontFamily: "'Share Tech Mono', monospace",
        fontSize: TEXT_SIZE,
      }}
    >
      <div
        title={VALUE_CAVEAT}
        style={{ display: "flex", alignItems: "baseline", gap: "8px" }}
      >
        <span style={{ color: "var(--text-muted)" }}>V(s)</span>
        {value === null ? (
          <span style={{ color: "var(--text-muted)" }}>
            No current evaluation.
          </span>
        ) : (
          <span
            style={{
              color: value >= 0 ? "var(--neon-cyan)" : "var(--neon-pink)",
            }}
          >
            {value >= 0 ? "+" : ""}
            {value.toFixed(3)}
          </span>
        )}
      </div>

      <Sparkline values={trace} />

      <div style={{ color: "var(--text-muted)" }}>
        {trace.length} prompt{trace.length === 1 ? "" : "s"}
      </div>
    </div>
  );
}
