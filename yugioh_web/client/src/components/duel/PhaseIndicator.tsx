import { Phase, PHASE_LABELS, PHASE_ORDER } from "../../../../shared/gameTypes";

interface PhaseIndicatorProps {
  phase: Phase;
  isMyTurn: boolean;
  onAdvance: () => void;
  turnNumber: number;
  activePlayerName: string;
}

export function PhaseIndicator({
  phase,
  isMyTurn,
  onAdvance,
  turnNumber,
  activePlayerName,
}: PhaseIndicatorProps) {
  const currentIndex = PHASE_ORDER.indexOf(phase);

  return (
    <div
      className="flex flex-col items-center rounded"
      style={{
        gap: "clamp(2px, 0.3vw, 5px)",
        padding: "clamp(4px, 0.5vw, 8px) clamp(8px, 1.2vw, 16px)",
        background: "rgba(0,0,0,0.75)",
        border: "1px solid rgba(0,245,255,0.3)",
        boxShadow: "0 0 12px rgba(0,245,255,0.08)",
      }}
    >
      {/* Turn info */}
      <div className="flex items-center" style={{ gap: "clamp(4px, 0.8vw, 10px)" }}>
        <span
          style={{
            fontFamily: "'Orbitron', sans-serif",
            fontSize: "clamp(0.42rem, 0.75vw, 0.65rem)",
            color: "rgba(0,245,255,0.7)",
            letterSpacing: "0.1em",
          }}
        >
          TURN {turnNumber}
        </span>
        <span
          style={{
            fontFamily: "'Orbitron', sans-serif",
            fontSize: "clamp(0.48rem, 0.9vw, 0.78rem)",
            fontWeight: "bold",
            color: isMyTurn ? "var(--neon-cyan)" : "var(--neon-pink)",
            textShadow: isMyTurn
              ? "0 0 10px var(--neon-cyan), 0 0 20px rgba(0,245,255,0.4)"
              : "0 0 10px var(--neon-pink), 0 0 20px rgba(255,45,120,0.4)",
            letterSpacing: "0.12em",
          }}
        >
          {isMyTurn ? "YOUR TURN" : `${activePlayerName.toUpperCase()}'S TURN`}
        </span>
      </div>

      {/* Phase steps */}
      <div className="flex items-center" style={{ gap: "clamp(2px, 0.4vw, 6px)" }}>
        {PHASE_ORDER.map((p, i) => {
          const cls =
            i < currentIndex ? "past" : i === currentIndex ? "active" : "future";
          return (
            <span
              key={p}
              className={`phase-step ${cls}`}
              style={{ fontSize: "clamp(0.36rem, 0.6vw, 0.52rem)" }}
            >
              {PHASE_LABELS[p].replace(" Phase", "").replace("Main Phase ", "MP")}
            </span>
          );
        })}
      </div>

      {/* Bottom row: NEXT button (my turn) or waiting indicator (opponent's turn) — same height always */}
      <div
        style={{
          marginTop: "clamp(1px, 0.2vw, 3px)",
          height: "clamp(14px, 1.8vw, 22px)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        {isMyTurn ? (
          <button
            onClick={onAdvance}
            className="rounded font-bold transition-all"
            style={{
              padding: "clamp(1px, 0.2vw, 3px) clamp(6px, 1vw, 14px)",
              fontFamily: "'Orbitron', sans-serif",
              fontSize: "clamp(0.42rem, 0.75vw, 0.62rem)",
              background: "rgba(0,245,255,0.08)",
              border: "1px solid var(--neon-cyan)",
              color: "var(--neon-cyan)",
              letterSpacing: "0.08em",
              height: "100%",
            }}
            onMouseEnter={(e) => {
              (e.target as HTMLButtonElement).style.background = "rgba(0,245,255,0.2)";
              (e.target as HTMLButtonElement).style.boxShadow = "0 0 12px rgba(0,245,255,0.4)";
            }}
            onMouseLeave={(e) => {
              (e.target as HTMLButtonElement).style.background = "rgba(0,245,255,0.08)";
              (e.target as HTMLButtonElement).style.boxShadow = "none";
            }}
          >
            {phase === "END" ? "END TURN ▶" : `NEXT: ${PHASE_LABELS[PHASE_ORDER[PHASE_ORDER.indexOf(phase) + 1]]?.replace(" Phase", "") ?? "END"} ▶`}
          </button>
        ) : (
          <div
            className="flex items-center"
            style={{ gap: "clamp(3px, 0.4vw, 6px)" }}
          >
            {/* Three pulsing dots */}
            {[0, 1, 2].map((i) => (
              <span
                key={i}
                className="waiting-dot"
                style={{ animationDelay: `${i * 0.22}s` }}
              />
            ))}
            <span
              style={{
                fontFamily: "'Orbitron', sans-serif",
                fontSize: "clamp(0.52rem, 0.85vw, 0.72rem)",
                fontWeight: "bold",
                color: "var(--neon-pink)",
                letterSpacing: "0.16em",
                textShadow: "0 0 10px var(--neon-pink), 0 0 20px rgba(255,45,120,0.5)",
              }}
            >
              WAITING
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
