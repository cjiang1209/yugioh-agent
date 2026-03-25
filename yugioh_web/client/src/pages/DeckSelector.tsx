import { useEffect, useState } from "react";
import { YgoCard } from "../../../shared/gameTypes";
import { DeckDefinition, STARTER_DECKS } from "../../../shared/starterDecks";
import { useCardApi } from "../hooks/useCardApi";

interface DeckSelectorProps {
  onDeckSelected: (deck: YgoCard[], deckName: string) => void;
}

export function DeckSelector({ onDeckSelected }: DeckSelectorProps) {
  const [selected, setSelected] = useState<DeckDefinition | null>(null);
  const [loadedDeck, setLoadedDeck] = useState<YgoCard[] | null>(null);
  const [loading, setLoading] = useState(false);
  const { fetchCardsByIds } = useCardApi();

  async function loadDeck(def: DeckDefinition) {
    setSelected(def);
    setLoadedDeck(null);
    setLoading(true);
    const cards = await fetchCardsByIds(def.cardIds);
    setLoadedDeck(cards);
    setLoading(false);
  }

  const deckColors = ["var(--neon-pink)", "var(--neon-cyan)", "var(--neon-yellow)"];

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
          SELECT YOUR DECK
        </h1>
        <p className="text-sm opacity-50" style={{ color: "var(--text-secondary)", fontFamily: "'Rajdhani', sans-serif" }}>
          Choose a pre-built deck to enter the duel
        </p>
      </div>

      {/* Deck cards */}
      <div className="flex gap-6 mb-8 flex-wrap justify-center">
        {STARTER_DECKS.map((deck, idx) => {
          const color = deckColors[idx % deckColors.length];
          const isSelected = selected?.name === deck.name;
          return (
            <div
              key={deck.name}
              className="hud-bracket cursor-pointer rounded p-4 transition-all"
              style={{
                width: "240px",
                background: isSelected ? `${color}11` : "var(--bg-panel)",
                border: `1px solid ${isSelected ? color : "var(--border-dim)"}`,
                boxShadow: isSelected ? `0 0 20px ${color}44` : "none",
                transform: isSelected ? "translateY(-4px)" : "none",
              }}
              onClick={() => loadDeck(deck)}
            >
              <div
                className="text-lg font-bold mb-1"
                style={{
                  fontFamily: "'Orbitron', sans-serif",
                  color,
                  fontSize: "0.9rem",
                  letterSpacing: "0.08em",
                }}
              >
                {deck.name}
              </div>
              <p
                className="text-xs opacity-60 mb-3"
                style={{ color: "var(--text-secondary)", fontFamily: "'Rajdhani', sans-serif" }}
              >
                {deck.description}
              </p>
              <div
                className="text-xs"
                style={{ fontFamily: "'Share Tech Mono', monospace", color, opacity: 0.7 }}
              >
                {deck.cardIds.length} CARDS
              </div>
            </div>
          );
        })}
      </div>

      {/* Deck preview */}
      {selected && (
        <div
          className="rounded p-4 mb-6 animate-slide-up"
          style={{
            background: "var(--bg-panel)",
            border: "1px solid var(--border-dim)",
            maxWidth: "600px",
            width: "100%",
          }}
        >
          <div
            className="text-xs font-bold mb-3 tracking-widest"
            style={{ fontFamily: "'Orbitron', sans-serif", color: "var(--neon-cyan)", fontSize: "0.6rem" }}
          >
            DECK PREVIEW — {selected.name.toUpperCase()}
          </div>
          {loading ? (
            <div className="flex items-center justify-center py-8 gap-3">
              <div
                className="w-4 h-4 rounded-full animate-spin"
                style={{ border: "2px solid var(--neon-cyan)", borderTopColor: "transparent" }}
              />
              <span className="text-xs opacity-50" style={{ color: "var(--neon-cyan)", fontFamily: "'Orbitron', sans-serif", fontSize: "0.6rem" }}>
                LOADING CARDS...
              </span>
            </div>
          ) : loadedDeck ? (
            <div className="flex flex-wrap gap-1.5 max-h-40 overflow-y-auto">
              {loadedDeck.map((card, i) => (
                <div
                  key={`${card.id}-${i}`}
                  className="relative flex-shrink-0 rounded overflow-hidden"
                  style={{ width: "36px", height: "50px", border: "1px solid var(--border-dim)" }}
                  title={card.name}
                >
                  <img
                    src={`https://images.ygoprodeck.com/images/cards_small/${card.id}.jpg`}
                    alt={card.name}
                    className="w-full h-full object-cover"
                    onError={(e) => {
                      (e.target as HTMLImageElement).src =
                        "https://images.ygoprodeck.com/images/cards/back_high.jpg";
                    }}
                  />
                </div>
              ))}
            </div>
          ) : null}
        </div>
      )}

      {/* Confirm button */}
      <button
        disabled={!loadedDeck || loading}
        onClick={() => {
          if (loadedDeck && selected) {
            onDeckSelected(loadedDeck, selected.name);
          }
        }}
        className="px-8 py-3 rounded font-bold text-sm transition-all"
        style={{
          fontFamily: "'Orbitron', sans-serif",
          letterSpacing: "0.15em",
          background: loadedDeck ? "rgba(0,245,255,0.1)" : "rgba(255,255,255,0.03)",
          border: `1px solid ${loadedDeck ? "var(--neon-cyan)" : "var(--border-dim)"}`,
          color: loadedDeck ? "var(--neon-cyan)" : "var(--text-muted)",
          boxShadow: loadedDeck ? "0 0 20px rgba(0,245,255,0.3)" : "none",
          cursor: loadedDeck ? "pointer" : "not-allowed",
          fontSize: "0.75rem",
        }}
      >
        {loading ? "LOADING..." : loadedDeck ? "ENTER THE DUEL ▶" : "SELECT A DECK"}
      </button>
    </div>
  );
}
