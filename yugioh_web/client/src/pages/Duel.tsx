import { useEffect, useState } from "react";
import { DuelBoard } from "../components/duel/DuelBoard";
import { CoinFlipOverlay } from "../components/CoinFlipOverlay";
import { useAIEngine } from "../hooks/useAIEngine";
import { DeckSelector } from "./DeckSelector";
import { resolveTurnOrder, type TurnOrder } from "./turnOrder";
import type { DeckPayload } from "../../../shared/deckTypes";

const MODE_BUTTON_BASE = {
  fontFamily: "'Orbitron', sans-serif",
  fontSize: "clamp(0.42rem, 0.75vw, 0.65rem)",
  letterSpacing: "0.1em",
  backdropFilter: "blur(4px)",
} as const;

const randomSeed = () => Math.floor(Math.random() * 100000);

// ─── Page ────────────────────────────────────────────────────────────────────

export default function Duel() {
  const [myDeck, setMyDeck] = useState<DeckPayload | null>(null);
  const [oppDeck, setOppDeck] = useState<DeckPayload | null>(null);
  const [openCards, setOpenCards] = useState(false);
  const [turnOrder, setTurnOrder] = useState<TurnOrder>("random");
  const [agentPlayer, setAgentPlayer] = useState<0 | 1>(0);
  const [pendingCoinFlip, setPendingCoinFlip] = useState<0 | 1 | null>(null);
  const [recommend, setRecommend] = useState(false);

  // Phase 1: Deck selection — scrollable page, no fixed wrapper
  if (!myDeck || !oppDeck) {
    return (
      <DeckSelector
        onDeckSelected={(my, opp, oc, to, ap, animate, rec) => {
          setMyDeck(my);
          setOppDeck(opp);
          setOpenCards(oc);
          setTurnOrder(to);
          setAgentPlayer(ap);
          setRecommend(rec);
          if (animate) setPendingCoinFlip(ap);
        }}
      />
    );
  }

  // Phase 1.5: Coin flip animation
  if (pendingCoinFlip !== null) {
    return (
      <CoinFlipOverlay
        result={pendingCoinFlip}
        onComplete={() => setPendingCoinFlip(null)}
      />
    );
  }

  // Phase 2: Duel — fixed viewport
  return (
    <div className="fixed inset-0" style={{ background: "var(--bg-void)" }}>
      <AIModeDuel
        myDeck={myDeck}
        oppDeck={oppDeck}
        openCards={openCards}
        agentPlayer={agentPlayer}
        recommend={recommend}
        onChangeDecks={() => {
          setMyDeck(null);
          setOppDeck(null);
        }}
        onRestartTurnOrder={() => {
          const { agentPlayer: ap, animateCoinFlip } =
            resolveTurnOrder(turnOrder);
          setAgentPlayer(ap);
          if (animateCoinFlip) setPendingCoinFlip(ap);
          return { agentPlayer: ap, willAnimate: animateCoinFlip };
        }}
      />
    </div>
  );
}

// ─── AI Mode wrapper ────────────────────────────────────────────────────────

function AIModeDuel({
  myDeck,
  oppDeck,
  openCards,
  agentPlayer,
  recommend,
  onChangeDecks,
  onRestartTurnOrder,
}: {
  myDeck: DeckPayload;
  oppDeck: DeckPayload;
  openCards: boolean;
  agentPlayer: 0 | 1;
  recommend: boolean;
  onChangeDecks: () => void;
  onRestartTurnOrder: () => { agentPlayer: 0 | 1; willAnimate: boolean };
}) {
  const {
    state,
    outcome,
    engineActions,
    recommendedActionIndex,
    valueTrace,
    actionProbs,
    autoplay,
    toggleAutoplay,
    enginePrompt,
    visibleLog,
    isReplaying,
    status,
    error,
    reset,
    submitAction,
  } = useAIEngine(openCards, recommend);

  const [inspectorOn, setInspectorOn] = useState(false);

  const deck0 = agentPlayer === 0 ? myDeck : oppDeck;
  const deck1 = agentPlayer === 0 ? oppDeck : myDeck;

  useEffect(() => {
    reset(randomSeed(), deck0, deck1, agentPlayer);
  }, [deck0, deck1, agentPlayer]);

  if (status === "loading" || (status === "idle" && !state)) {
    return <LoadingSpinner message="Connecting to engine..." />;
  }

  if (status === "error" || !state) {
    return (
      <div
        className="min-h-screen flex items-center justify-center"
        style={{ background: "var(--bg-void)" }}
      >
        <div className="text-center">
          <div
            style={{
              color: "var(--neon-pink)",
              fontFamily: "'Orbitron', sans-serif",
              fontSize: "0.8rem",
              marginBottom: "1rem",
            }}
          >
            CONNECTION ERROR
          </div>
          <div
            style={{
              color: "var(--text-secondary)",
              fontFamily: "'Share Tech Mono', monospace",
              fontSize: "0.6rem",
              marginBottom: "1.5rem",
              maxWidth: "30rem",
            }}
          >
            {error ||
              "Failed to connect to Python engine. Is the server running?"}
          </div>
          <button
            onClick={() => reset(randomSeed(), deck0, deck1, agentPlayer)}
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
      recommendedActionIndex={recommendedActionIndex}
      autoplay={autoplay}
      // Withheld when AI Assist is off: with no recommendations arriving there
      // is nothing for autoplay to play.
      onToggleAutoplay={recommend ? toggleAutoplay : undefined}
      inspectorOn={inspectorOn}
      // Withheld when AI Assist is off: with no recommendation arriving there
      // is no readout to inspect.
      onToggleInspector={
        recommend ? () => setInspectorOn(on => !on) : undefined
      }
      valueTrace={valueTrace}
      actionProbs={actionProbs}
      enginePrompt={enginePrompt}
      onEngineAction={submitAction}
      outcome={outcome}
      onChangeDecks={onChangeDecks}
      onRestart={() => {
        const { agentPlayer: ap, willAnimate } = onRestartTurnOrder();
        if (!willAnimate) {
          reset(randomSeed(), deck0, deck1, ap);
        }
      }}
      visibleLog={visibleLog}
      isReplaying={isReplaying}
      openCards={openCards}
    />
  );
}

// ─── Shared components ───────────────────────────────────────────────────────

function LoadingSpinner({ message }: { message: string }) {
  return (
    <div
      className="min-h-screen flex items-center justify-center"
      style={{ background: "var(--bg-void)" }}
    >
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
          style={{
            color: "var(--text-secondary)",
            fontFamily: "'Share Tech Mono', monospace",
            fontSize: "0.55rem",
          }}
        >
          {message}
        </div>
      </div>
    </div>
  );
}
