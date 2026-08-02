import type { DuelOutcome } from "../../../../shared/gameTypes";

/** Each action renders only when its handler is provided. */
interface DuelResultOverlayProps {
  outcome: DuelOutcome;
  /** Closing line of the duel log, shown under the title. */
  lastLogLine?: string;
  /** Start another duel with the same decks. */
  onRestart?: () => void;
  /** Return to deck selection. */
  onChangeDecks?: () => void;
  /** Hide the overlay to inspect the final board. */
  onDismiss?: () => void;
}

const glow = (color: string) => `0 0 20px ${color}, 0 0 60px ${color}`;

const RESULTS: Record<DuelOutcome, { title: string; color: string }> = {
  win: { title: "VICTORY", color: "var(--neon-cyan)" },
  loss: { title: "DEFEAT", color: "var(--neon-pink)" },
  draw: { title: "DRAW", color: "var(--text-secondary)" },
};

const BUTTON_BASE = {
  fontFamily: "'Orbitron', sans-serif",
  fontSize: "0.6rem",
  letterSpacing: "0.1em",
} as const;

/** Full-screen end-of-duel result with the follow-up actions. */
export function DuelResultOverlay({
  outcome,
  lastLogLine,
  onRestart,
  onChangeDecks,
  onDismiss,
}: DuelResultOverlayProps) {
  const { title, color } = RESULTS[outcome];
  const actions = [
    {
      label: "DUEL AGAIN",
      onClick: onRestart,
      background: "rgba(0,245,255,0.15)",
      border: "var(--neon-cyan)",
    },
    {
      label: "CHANGE DECKS",
      onClick: onChangeDecks,
      background: "rgba(0,245,255,0.08)",
      border: "rgba(0,245,255,0.4)",
    },
  ].filter(a => a.onClick);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ background: "rgba(0,0,0,0.9)" }}
    >
      <div className="text-center animate-slide-up">
        <div
          className="text-6xl font-black mb-4"
          style={{
            fontFamily: "'Orbitron', sans-serif",
            color,
            textShadow: glow(color),
          }}
        >
          {title}
        </div>
        <div
          className="text-lg opacity-60"
          style={{ color: "var(--text-secondary)" }}
        >
          {lastLogLine}
        </div>

        <div className="flex items-center justify-center gap-3 mt-8">
          {actions.map(a => (
            <button
              key={a.label}
              onClick={a.onClick}
              className="px-5 py-2 rounded transition-all hover:opacity-90"
              style={{
                ...BUTTON_BASE,
                background: a.background,
                border: `1px solid ${a.border}`,
                color: "var(--neon-cyan)",
              }}
            >
              {a.label}
            </button>
          ))}
        </div>

        {onDismiss && (
          <button
            onClick={onDismiss}
            className="mt-4 px-2 py-1 opacity-50 hover:opacity-90 transition-opacity"
            style={{
              ...BUTTON_BASE,
              fontSize: "0.5rem",
              color: "var(--text-secondary)",
            }}
          >
            VIEW BOARD
          </button>
        )}
      </div>
    </div>
  );
}
