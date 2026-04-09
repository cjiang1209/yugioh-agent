import { useEffect, useState } from "react";
import { DuelBoard } from "../components/duel/DuelBoard";
import { useAIEngine } from "../hooks/useAIEngine";
import { DeckSelector } from "./DeckSelector";
import type { DeckPayload } from "../../../shared/deckTypes";

const MODE_BUTTON_BASE = {
  fontFamily: "'Orbitron', sans-serif",
  fontSize: "clamp(0.42rem, 0.75vw, 0.65rem)",
  letterSpacing: "0.1em",
  backdropFilter: "blur(4px)",
} as const;

const ENGINE_API_URL = "http://localhost:8000";

// ─── Page ────────────────────────────────────────────────────────────────────

export default function Duel() {
  const [selectedDeck0, setSelectedDeck0] = useState<DeckPayload | null>(null);
  const [selectedDeck1, setSelectedDeck1] = useState<DeckPayload | null>(null);

  // Phase 1: Deck selection — scrollable page, no fixed wrapper
  if (!selectedDeck0 || !selectedDeck1) {
    return (
      <DeckSelector
        apiUrl={ENGINE_API_URL}
        onDeckSelected={(myDeck, oppDeck) => {
          setSelectedDeck0(myDeck);
          setSelectedDeck1(oppDeck);
        }}
      />
    );
  }

  // Phase 2: Duel — fixed viewport
  return (
    <div className="fixed inset-0" style={{ background: "var(--bg-void)" }}>
      <AIModeDuel deck0={selectedDeck0} deck1={selectedDeck1} />
    </div>
  );
}

// ─── AI Mode wrapper ────────────────────────────────────────────────────────

function AIModeDuel({ deck0, deck1 }: { deck0: DeckPayload; deck1: DeckPayload }) {
  const { state, engineActions, enginePrompt, visibleLog, isReplaying, status, error, reset, submitAction } = useAIEngine(ENGINE_API_URL);

  useEffect(() => {
    reset(Math.floor(Math.random() * 100000), deck0, deck1);
  }, [deck0, deck1]);

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
            onClick={() => reset(Math.floor(Math.random() * 100000), deck0, deck1)}
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
    <DuelBoard
      state={state}
      mySide="player1"
      onAction={() => {}}
      engineMode
      engineActions={engineActions}
      enginePrompt={enginePrompt}
      onEngineAction={submitAction}
      onRestart={() => reset(Math.floor(Math.random() * 100000), deck0, deck1)}
      visibleLog={visibleLog}
      isReplaying={isReplaying}
    />
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
    </div>
  );
}

