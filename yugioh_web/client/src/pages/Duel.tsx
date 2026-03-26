import { useState } from "react";
import { DuelBoard } from "../components/duel/DuelBoard";
import { useSelfPlayDuel } from "../hooks/useSelfPlayDuel";
import { useRealEngine } from "../hooks/useRealEngine";

type DuelMode = "fun" | "real";

const MODE_BUTTON_BASE = {
  fontFamily: "'Orbitron', sans-serif",
  fontSize: "clamp(0.42rem, 0.75vw, 0.65rem)",
  letterSpacing: "0.1em",
  backdropFilter: "blur(4px)",
} as const;

// ─── Fun Mode wrapper ────────────────────────────────────────────────────────

function FunModeDuel() {
  const { state, viewSide, status, sendAction, switchSide, restart } = useSelfPlayDuel();

  if (status === "loading" || !state) {
    return <LoadingSpinner message="Fetching card data..." />;
  }

  return (
    <>
      <OverlayBar>
        <button
          onClick={switchSide}
          className="flex items-center gap-1.5 px-3 py-1 rounded transition-all"
          style={{
            ...MODE_BUTTON_BASE,
            background: "rgba(0,0,0,0.7)",
            border: "1px solid rgba(0,245,255,0.25)",
            color: "var(--neon-cyan)",
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
        <RestartButton onClick={restart} />
      </OverlayBar>
      <DuelBoard state={state} mySide={viewSide} onAction={sendAction} />
    </>
  );
}

// ─── Real Mode wrapper ───────────────────────────────────────────────────────

function RealModeDuel() {
  const { state, engineActions, status, error, reset, submitAction } = useRealEngine();

  // Auto-reset on first mount
  const [hasStarted, setHasStarted] = useState(false);
  if (!hasStarted) {
    setHasStarted(true);
    reset(Math.floor(Math.random() * 100000));
  }

  if (status === "loading" || (status === "idle" && !state)) {
    return <LoadingSpinner message="Connecting to engine..." />;
  }

  if (status === "error" || !state) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ background: "var(--bg-void)" }}>
        <div className="text-center">
          <div style={{ color: "var(--neon-pink)", fontFamily: "'Orbitron', sans-serif", fontSize: "0.8rem", marginBottom: "1rem" }}>
            CONNECTION ERROR
          </div>
          <div style={{ color: "var(--text-secondary)", fontFamily: "'Share Tech Mono', monospace", fontSize: "0.6rem", marginBottom: "1.5rem", maxWidth: "30rem" }}>
            {error || "Failed to connect to Python engine. Is the server running?"}
          </div>
          <button
            onClick={() => reset(Math.floor(Math.random() * 100000))}
            className="px-4 py-2 rounded transition-all"
            style={{
              ...MODE_BUTTON_BASE,
              background: "rgba(0,0,0,0.7)",
              border: "1px solid rgba(0,245,255,0.25)",
              color: "var(--neon-cyan)",
            }}
          >
            RETRY
          </button>
        </div>
      </div>
    );
  }

  return (
    <>
      <OverlayBar>
        <RestartButton onClick={() => reset(Math.floor(Math.random() * 100000))} />
      </OverlayBar>
      {/* TODO(Subphase F): pass engineMode, engineActions, onEngineAction */}
      <DuelBoard
        state={state}
        mySide="player1"
        onAction={() => {}}
      />
    </>
  );
}

// ─── Shared components ───────────────────────────────────────────────────────

function LoadingSpinner({ message }: { message: string }) {
  return (
    <div className="min-h-screen flex items-center justify-center" style={{ background: "var(--bg-void)" }}>
      <div className="text-center">
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
          {message}
        </div>
      </div>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}

function OverlayBar({ children }: { children: React.ReactNode }) {
  return (
    <div
      className="absolute top-0 left-1/2 z-30 flex items-center gap-2"
      style={{ transform: "translateX(-50%)", paddingTop: "4px" }}
      onClick={(e) => e.stopPropagation()}
    >
      {children}
    </div>
  );
}

function RestartButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="px-2 py-1 rounded transition-all opacity-40 hover:opacity-80"
      style={{
        ...MODE_BUTTON_BASE,
        background: "rgba(0,0,0,0.7)",
        border: "1px solid rgba(255,45,120,0.25)",
        color: "var(--neon-pink)",
      }}
      title="Restart duel"
    >
      ↺ RESTART
    </button>
  );
}

// ─── Page ────────────────────────────────────────────────────────────────────

export default function Duel() {
  const [mode, setMode] = useState<DuelMode>("fun");

  return (
    <div className="fixed inset-0" style={{ background: "var(--bg-void)" }}>
      {/* Mode toggle — top-right corner */}
      <div
        className="absolute top-0 right-0 z-40 flex items-center gap-1"
        style={{ padding: "4px 8px" }}
        onClick={(e) => e.stopPropagation()}
      >
        {(["fun", "real"] as const).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className="px-2 py-1 rounded transition-all"
            style={{
              ...MODE_BUTTON_BASE,
              background: mode === m ? "rgba(0,245,255,0.15)" : "rgba(0,0,0,0.5)",
              border: `1px solid ${mode === m ? "rgba(0,245,255,0.5)" : "rgba(255,255,255,0.1)"}`,
              color: mode === m ? "var(--neon-cyan)" : "var(--text-secondary)",
              opacity: mode === m ? 1 : 0.5,
            }}
          >
            {m === "fun" ? "FUN" : "REAL"}
          </button>
        ))}
      </div>

      {mode === "fun" ? <FunModeDuel /> : <RealModeDuel />}
    </div>
  );
}
