import { useEffect, useState } from "react";
import type { DeckDefinition, DeckPayload } from "../../../shared/deckTypes";

const DECK_COLORS = ["var(--neon-cyan)", "var(--neon-pink)", "var(--neon-yellow)"];

interface DeckSelectorProps {
  apiUrl?: string;
  onDeckSelected: (myDeck: DeckPayload, oppDeck: DeckPayload) => void;
}

export function DeckSelector({ apiUrl = "http://localhost:8000", onDeckSelected }: DeckSelectorProps) {
  const [decks, setDecks] = useState<DeckDefinition[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [myDeck, setMyDeck] = useState<DeckDefinition | null>(null);
  const [oppDeck, setOppDeck] = useState<DeckDefinition | null>(null);

  useEffect(() => {
    fetch(`${apiUrl}/api/web/decks`)
      .then((res) => {
        if (!res.ok) throw new Error(`Failed to load decks: ${res.status}`);
        return res.json();
      })
      .then((data: DeckDefinition[]) => {
        setDecks(data);
        setLoading(false);
      })
      .catch((e) => {
        setError(e instanceof Error ? e.message : "Failed to load decks");
        setLoading(false);
      });
  }, [apiUrl]);

  if (loading) {
    return (
      <div
        className="min-h-screen flex flex-col items-center justify-center"
        style={{ background: "var(--bg-void)" }}
      >
        <div
          className="w-10 h-10 rounded-full mx-auto mb-4"
          style={{
            border: "2px solid rgba(0,245,255,0.15)",
            borderTopColor: "var(--neon-cyan)",
            animation: "spin 0.8s linear infinite",
          }}
        />
        <div
          className="text-xs"
          style={{ fontFamily: "'Orbitron', sans-serif", color: "var(--neon-cyan)", letterSpacing: "0.15em" }}
        >
          LOADING DECKS...
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div
        className="min-h-screen flex flex-col items-center justify-center"
        style={{ background: "var(--bg-void)" }}
      >
        <div style={{ color: "var(--neon-pink)", fontFamily: "'Orbitron', sans-serif", fontSize: "0.8rem" }}>
          {error}
        </div>
      </div>
    );
  }

  const canStart = myDeck !== null && oppDeck !== null;

  return (
    <div
      className="min-h-screen flex flex-col items-center justify-center p-6"
      style={{ background: "var(--bg-void)" }}
    >
      {/* Title */}
      <div className="text-center mb-8">
        <h1
          className="text-4xl font-black mb-2"
          style={{
            fontFamily: "'Orbitron', sans-serif",
            color: "var(--neon-cyan)",
            textShadow: "0 0 20px var(--neon-cyan), 0 0 60px var(--neon-cyan)",
            letterSpacing: "0.15em",
          }}
        >
          SELECT DECKS
        </h1>
        <p
          className="text-sm opacity-50"
          style={{ color: "var(--text-secondary)", fontFamily: "'Rajdhani', sans-serif" }}
        >
          Choose a deck for each player
        </p>
      </div>

      {/* Two-slot layout */}
      <div className="flex gap-10 mb-8 flex-wrap justify-center">
        {/* My Deck slot */}
        <SlotSection
          label="MY DECK"
          labelColor="var(--neon-cyan)"
          decks={decks}
          deckColors={DECK_COLORS}
          selected={myDeck}
          onSelect={setMyDeck}
        />

        {/* Divider */}
        <div className="hidden md:flex items-center">
          <div
            style={{
              width: "1px",
              height: "200px",
              background: "linear-gradient(180deg, transparent, var(--border-dim), transparent)",
            }}
          />
        </div>

        {/* Opponent Deck slot */}
        <SlotSection
          label="OPPONENT DECK"
          labelColor="var(--neon-pink)"
          decks={decks}
          deckColors={DECK_COLORS}
          selected={oppDeck}
          onSelect={setOppDeck}
        />
      </div>

      {/* Confirm button */}
      <button
        disabled={!canStart}
        onClick={() => {
          if (myDeck && oppDeck) {
            onDeckSelected(
              { main: myDeck.main, extra: myDeck.extra },
              { main: oppDeck.main, extra: oppDeck.extra },
            );
          }
        }}
        className="px-8 py-3 rounded font-bold text-sm transition-all"
        style={{
          fontFamily: "'Orbitron', sans-serif",
          letterSpacing: "0.15em",
          background: canStart ? "rgba(0,245,255,0.1)" : "rgba(255,255,255,0.03)",
          border: `1px solid ${canStart ? "var(--neon-cyan)" : "var(--border-dim)"}`,
          color: canStart ? "var(--neon-cyan)" : "var(--text-muted)",
          boxShadow: canStart ? "0 0 20px rgba(0,245,255,0.3)" : "none",
          cursor: canStart ? "pointer" : "not-allowed",
          fontSize: "0.75rem",
        }}
      >
        {canStart ? "ENTER THE DUEL \u25B6" : "SELECT BOTH DECKS"}
      </button>
    </div>
  );
}

// ─── Slot section ───────────────────────────────────────────────────────────

function SlotSection({
  label,
  labelColor,
  decks,
  deckColors,
  selected,
  onSelect,
}: {
  label: string;
  labelColor: string;
  decks: DeckDefinition[];
  deckColors: string[];
  selected: DeckDefinition | null;
  onSelect: (deck: DeckDefinition) => void;
}) {
  return (
    <div style={{ minWidth: "240px" }}>
      <div
        className="text-xs font-bold mb-4 tracking-widest text-center"
        style={{ fontFamily: "'Orbitron', sans-serif", color: labelColor, fontSize: "0.65rem" }}
      >
        {label}
      </div>
      <div className="flex flex-col gap-3">
        {decks.map((deck, idx) => {
          const color = deckColors[idx % deckColors.length];
          const isSelected = selected?.filename === deck.filename;
          const mainCount = deck.main.length;
          const extraCount = deck.extra.length;
          return (
            <div
              key={deck.filename}
              className="cursor-pointer rounded p-4 transition-all"
              style={{
                background: isSelected ? `${color}11` : "var(--bg-panel)",
                border: `1px solid ${isSelected ? color : "var(--border-dim)"}`,
                boxShadow: isSelected ? `0 0 20px ${color}44` : "none",
                transform: isSelected ? "translateY(-2px)" : "none",
              }}
              onClick={() => onSelect(deck)}
            >
              <div
                className="text-lg font-bold mb-1"
                style={{
                  fontFamily: "'Orbitron', sans-serif",
                  color,
                  fontSize: "0.85rem",
                  letterSpacing: "0.08em",
                }}
              >
                {deck.name}
              </div>
              <div
                className="text-xs"
                style={{ fontFamily: "'Share Tech Mono', monospace", color, opacity: 0.7 }}
              >
                {extraCount > 0 ? `${mainCount} MAIN + ${extraCount} EXTRA` : `${mainCount} CARDS`}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
