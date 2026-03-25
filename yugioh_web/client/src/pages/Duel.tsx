import { DuelBoard } from "../components/duel/DuelBoard";
import { useSelfPlayDuel } from "../hooks/useSelfPlayDuel";

export default function Duel() {
  const { state, viewSide, status, sendAction, switchSide, restart } = useSelfPlayDuel();

  // ─── Loading ───────────────────────────────────────────────────────────────

  if (status === "loading" || !state) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ background: "var(--bg-void)" }}>
        <div className="text-center">
          {/* Spinner */}
          <div
            className="w-14 h-14 rounded-full mx-auto mb-5"
            style={{
              border: "2px solid rgba(0,245,255,0.15)",
              borderTopColor: "var(--neon-cyan)",
              animation: "spin 0.8s linear infinite",
            }}
          />
          <div
            className="text-sm mb-1"
            style={{
              fontFamily: "'Orbitron', sans-serif",
              color: "var(--neon-cyan)",
              fontSize: "0.7rem",
              letterSpacing: "0.15em",
              textShadow: "0 0 8px var(--neon-cyan)",
            }}
          >
            LOADING DUEL
          </div>
          <div
            className="text-xs opacity-40"
            style={{ color: "var(--text-secondary)", fontFamily: "'Share Tech Mono', monospace", fontSize: "0.55rem" }}
          >
            Fetching card data...
          </div>
        </div>
        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      </div>
    );
  }

  // ─── Duel board ────────────────────────────────────────────────────────────

  return (
    <div className="fixed inset-0" style={{ background: "var(--bg-void)" }}>
      {/* Side switcher overlay — shown above the board */}
      <div
        className="absolute top-0 left-1/2 z-30 flex items-center gap-2"
        style={{ transform: "translateX(-50%)", paddingTop: "4px" }}
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={switchSide}
          className="flex items-center gap-1.5 px-3 py-1 rounded transition-all"
          style={{
            background: "rgba(0,0,0,0.7)",
            border: "1px solid rgba(0,245,255,0.25)",
            color: "var(--neon-cyan)",
            fontFamily: "'Orbitron', sans-serif",
            fontSize: "clamp(0.45rem, 0.8vw, 0.7rem)",
            letterSpacing: "0.1em",
            backdropFilter: "blur(4px)",
          }}
          title="Switch to opponent's side"
        >
          <span style={{ fontSize: "0.7rem" }}>⇄</span>
          <span>
            VIEWING: <span style={{ color: "var(--neon-yellow)" }}>
              {viewSide === "player1" ? state.player1.name.toUpperCase() : state.player2.name.toUpperCase()}
            </span>
          </span>
        </button>

        <button
          onClick={restart}
          className="px-2 py-1 rounded transition-all opacity-40 hover:opacity-80"
          style={{
            background: "rgba(0,0,0,0.7)",
            border: "1px solid rgba(255,45,120,0.25)",
            color: "var(--neon-pink)",
            fontFamily: "'Orbitron', sans-serif",
            fontSize: "clamp(0.42rem, 0.75vw, 0.65rem)",
            letterSpacing: "0.1em",
            backdropFilter: "blur(4px)",
          }}
          title="Restart duel"
        >
          ↺ RESTART
        </button>
      </div>

      <DuelBoard
        state={state}
        mySide={viewSide}
        onAction={sendAction}
      />
    </div>
  );
}
