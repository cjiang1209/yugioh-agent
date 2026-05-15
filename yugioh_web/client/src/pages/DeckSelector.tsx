import { useEffect, useState } from "react";
import type { DeckDefinition, DeckPayload } from "../../../shared/deckTypes";
import { resolveTurnOrder, type TurnOrder } from "./turnOrder";

const DECK_COLORS = ["var(--neon-cyan)", "var(--neon-pink)", "var(--neon-yellow)"];
const CARD_IMAGE_BASE = "https://images.ygoprodeck.com/images/cards_small";
const CARD_BACK = "https://images.ygoprodeck.com/images/cards/back_high.jpg";

const TURN_ORDER_CAPTION: Record<TurnOrder, string> = {
  random: "Coin flip decides who starts",
  first: "You take the first turn",
  second: "Opponent takes the first turn",
};

interface DeckSelectorProps {
  apiUrl?: string;
  onDeckSelected: (
    myDeck: DeckPayload,
    oppDeck: DeckPayload,
    openCards: boolean,
    turnOrder: TurnOrder,
    agentPlayer: 0 | 1,
    animateCoinFlip: boolean,
  ) => void;
}

function toPayload(deck: DeckDefinition): DeckPayload {
  return {
    main: deck.main.map((c) => c.code),
    extra: deck.extra.map((c) => c.code),
  };
}

export function DeckSelector({ apiUrl = "http://localhost:8000", onDeckSelected }: DeckSelectorProps) {
  const [decks, setDecks] = useState<DeckDefinition[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [myDeck, setMyDeck] = useState<DeckDefinition | null>(null);
  const [oppDeck, setOppDeck] = useState<DeckDefinition | null>(null);
  const [openCards, setOpenCards] = useState(false);
  const [turnOrder, setTurnOrder] = useState<TurnOrder>("random");

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
      className="min-h-screen flex flex-col items-center p-6 pb-12"
      style={{ background: "var(--bg-void)" }}
    >
      {/* Title */}
      <div className="text-center mb-8 mt-6">
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
      <div className="flex gap-10 mb-6 flex-wrap justify-center">
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

      {/* Deck previews — side by side */}
      <div className="flex gap-6 mb-8 w-full max-w-5xl flex-wrap justify-center">
        <DeckPreview labelColor="var(--neon-cyan)" deck={myDeck} />
        <DeckPreview labelColor="var(--neon-pink)" deck={oppDeck} />
      </div>

      {/* Turn order */}
      <div className="flex flex-col items-center mb-6">
        <div
          className="text-xs font-bold mb-3 tracking-widest text-center"
          style={{
            fontFamily: "'Orbitron', sans-serif",
            color: "var(--neon-cyan)",
            fontSize: "0.65rem",
            letterSpacing: "0.15em",
          }}
        >
          TURN ORDER
        </div>
        <div className="flex gap-3" role="radiogroup" aria-label="Turn order">
          <TurnOrderButton
            label="RANDOM"
            value="random"
            current={turnOrder}
            onSelect={setTurnOrder}
          />
          <TurnOrderButton
            label="GO FIRST"
            value="first"
            current={turnOrder}
            onSelect={setTurnOrder}
          />
          <TurnOrderButton
            label="GO SECOND"
            value="second"
            current={turnOrder}
            onSelect={setTurnOrder}
          />
        </div>
        <span
          className="mt-1 opacity-40"
          style={{
            color: "var(--text-secondary)",
            fontFamily: "'Share Tech Mono', monospace",
            fontSize: "0.5rem",
          }}
        >
          {TURN_ORDER_CAPTION[turnOrder]}
        </span>
      </div>

      {/* Open cards toggle */}
      <div className="flex flex-col items-center mb-6">
        <button
          onClick={() => setOpenCards((v) => !v)}
          className="flex items-center gap-2 px-4 py-2 rounded transition-all"
          style={{
            fontFamily: "'Orbitron', sans-serif",
            fontSize: "0.6rem",
            letterSpacing: "0.1em",
            background: openCards ? "rgba(255,215,0,0.1)" : "rgba(255,255,255,0.03)",
            border: `1px solid ${openCards ? "rgba(255,215,0,0.8)" : "var(--border-dim)"}`,
            color: openCards ? "rgba(255,215,0,1)" : "var(--text-muted)",
            boxShadow: openCards ? "0 0 12px rgba(255,215,0,0.25)" : "none",
          }}
        >
          <span style={{ fontSize: "0.85rem" }}>{openCards ? "\u{1F441}" : "\u{1F441}\u200D\u{1F5E8}"}</span>
          {openCards ? "OPEN CARDS: ON" : "OPEN CARDS: OFF"}
        </button>
        <span
          className="mt-1 opacity-40"
          style={{ color: "var(--text-secondary)", fontFamily: "'Share Tech Mono', monospace", fontSize: "0.5rem" }}
        >
          Show opponent hidden cards
        </span>
      </div>

      {/* Confirm button */}
      <button
        disabled={!canStart}
        onClick={() => {
          if (myDeck && oppDeck) {
            const { agentPlayer, animateCoinFlip } = resolveTurnOrder(turnOrder);
            onDeckSelected(
              toPayload(myDeck),
              toPayload(oppDeck),
              openCards,
              turnOrder,
              agentPlayer,
              animateCoinFlip,
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

// ─── Turn order button ──────────────────────────────────────────────────────

function TurnOrderButton({
  label,
  value,
  current,
  onSelect,
}: {
  label: string;
  value: TurnOrder;
  current: TurnOrder;
  onSelect: (v: TurnOrder) => void;
}) {
  const selected = current === value;
  return (
    <button
      onClick={() => onSelect(value)}
      role="radio"
      aria-checked={selected}
      className="px-4 py-2 rounded transition-all"
      style={{
        fontFamily: "'Orbitron', sans-serif",
        fontSize: "0.6rem",
        letterSpacing: "0.1em",
        background: selected ? "rgba(0,245,255,0.1)" : "rgba(255,255,255,0.03)",
        border: `1px solid ${selected ? "var(--neon-cyan)" : "var(--border-dim)"}`,
        color: selected ? "var(--neon-cyan)" : "var(--text-muted)",
        boxShadow: selected ? "0 0 12px rgba(0,245,255,0.25)" : "none",
      }}
    >
      {label}
    </button>
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

// ─── Deck preview ───────────────────────────────────────────────────────────

function DeckPreview({
  labelColor,
  deck,
}: {
  labelColor: string;
  deck: DeckDefinition | null;
}) {
  if (!deck) {
    return (
      <div
        className="flex-1 rounded p-4 flex items-center justify-center"
        style={{
          minWidth: "280px",
          minHeight: "200px",
          background: "var(--bg-panel)",
          border: "1px solid var(--border-dim)",
        }}
      >
        <div
          className="text-xs opacity-30"
          style={{ fontFamily: "'Orbitron', sans-serif", color: labelColor, letterSpacing: "0.1em" }}
        >
          SELECT A DECK
        </div>
      </div>
    );
  }

  return (
    <div
      className="flex-1 rounded p-4"
      style={{
        minWidth: "280px",
        background: "var(--bg-panel)",
        border: `1px solid ${labelColor}33`,
      }}
    >
      {/* Main deck section */}
      <CardSection title={`MAIN DECK (${deck.main.length})`} color={labelColor} cards={deck.main} />

      {/* Extra deck section */}
      {deck.extra.length > 0 && (
        <>
          <div
            style={{
              height: "1px",
              background: `linear-gradient(90deg, transparent, ${labelColor}33, transparent)`,
              margin: "12px 0",
            }}
          />
          <CardSection title={`EXTRA DECK (${deck.extra.length})`} color={labelColor} cards={deck.extra} />
        </>
      )}
    </div>
  );
}

// ─── Card section (main or extra) ───────────────────────────────────────────

function CardSection({
  title,
  color,
  cards,
}: {
  title: string;
  color: string;
  cards: { code: number; name: string }[];
}) {
  return (
    <div>
      <div
        className="text-xs font-bold mb-2 tracking-wider"
        style={{ fontFamily: "'Orbitron', sans-serif", color, fontSize: "0.55rem", opacity: 0.7 }}
      >
        {title}
      </div>
      <div className="flex flex-wrap gap-1.5">
        {cards.map((card, idx) => (
          <div key={`${card.code}-${idx}`} className="group relative" style={{ width: "60px" }}>
            <img
              src={`${CARD_IMAGE_BASE}/${card.code}.jpg`}
              alt={card.name}
              title={card.name}
              className="rounded-sm"
              style={{
                width: "60px",
                height: "87px",
                objectFit: "cover",
                border: "1px solid var(--border-dim)",
              }}
              onError={(e) => {
                const img = e.target as HTMLImageElement;
                img.onerror = null;
                img.src = CARD_BACK;
              }}
            />
            {/* Name tooltip on hover */}
            <div
              className="absolute left-1/2 -translate-x-1/2 bottom-full mb-1 px-2 py-1 rounded opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity z-10 whitespace-nowrap"
              style={{
                background: "rgba(0,0,0,0.9)",
                border: `1px solid ${color}44`,
                color: "var(--text-primary)",
                fontFamily: "'Rajdhani', sans-serif",
                fontSize: "0.65rem",
                maxWidth: "160px",
                overflow: "hidden",
                textOverflow: "ellipsis",
              }}
            >
              {card.name}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
