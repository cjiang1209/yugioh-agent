import { useState } from "react";
import { GameCard } from "../../../../shared/gameTypes";

interface ZoneViewerProps {
  graveyard: GameCard[];
  banished: GameCard[];
  extra?: GameCard[];
  playerName: string;
  initialTab?: "graveyard" | "banished" | "extra";
  onClose: () => void;
  onCardSelect?: (card: GameCard) => void;
}

export function ZoneViewer({
  graveyard,
  banished,
  extra = [],
  playerName,
  initialTab = "extra",
  onClose,
  onCardSelect,
}: ZoneViewerProps) {
  const [tab, setTab] = useState<"graveyard" | "banished" | "extra">(initialTab);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const cards = tab === "graveyard" ? graveyard : tab === "banished" ? banished : extra;
  const isGY = tab === "graveyard";
  const isExtra = tab === "extra";
  const accentColor = isGY ? "var(--neon-pink)" : isExtra ? "#ffd700" : "#b44fff";
  const accentRgb = isGY ? "255,45,120" : isExtra ? "255,215,0" : "180,79,255";

  function handleCardClick(card: GameCard) {
    setSelectedId(card.instanceId);
    onCardSelect?.(card);
  }

  return (
    <div
      className="fixed flex items-center justify-center"
      style={{ top: 0, bottom: 0, left: 0, right: "clamp(180px, 20vw, 280px)", background: "rgba(0,0,0,0.85)", zIndex: 2000 }}
      onClick={onClose}
    >
      <div
        className="rounded overflow-hidden animate-slide-up flex flex-col"
        style={{
          background: "var(--bg-panel)",
          border: `1px solid rgba(${accentRgb},0.35)`,
          boxShadow: `0 0 40px rgba(0,0,0,0.8), 0 0 20px rgba(${accentRgb},0.12)`,
          maxWidth: "720px",
          width: "92vw",
          height: "min(85vh, 600px)",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div
          className="flex items-center gap-3 px-4 py-2"
          style={{ borderBottom: "1px solid rgba(255,255,255,0.06)", background: "rgba(0,0,0,0.45)" }}
        >
          {/* Player name */}
          <span
            className="font-bold shrink-0"
            style={{ fontFamily: "'Orbitron', sans-serif", color: accentColor, fontSize: "0.6rem", letterSpacing: "0.1em" }}
          >
            {playerName.toUpperCase()}
          </span>

          {/* Tab switcher — order: Extra → Graveyard → Banished (matches board zone order) */}
          <div className="flex gap-1 flex-1 justify-center">
            <button
              onClick={() => { setTab("extra"); setSelectedId(null); }}
              className="px-3 py-0.5 text-[0.55rem] rounded transition-all"
              style={{
                fontFamily: "'Orbitron', sans-serif",
                background: tab === "extra" ? "rgba(255,215,0,0.15)" : "transparent",
                border: `1px solid ${tab === "extra" ? "#ffd700" : "rgba(255,255,255,0.1)"}`,
                color: tab === "extra" ? "#ffd700" : "rgba(255,255,255,0.35)",
                boxShadow: tab === "extra" ? "0 0 8px rgba(255,215,0,0.3)" : "none",
              }}
            >
              ★ EXTRA ({extra.length})
            </button>
            <button
              onClick={() => { setTab("graveyard"); setSelectedId(null); }}
              className="px-3 py-0.5 text-[0.55rem] rounded transition-all"
              style={{
                fontFamily: "'Orbitron', sans-serif",
                background: tab === "graveyard" ? "rgba(255,45,120,0.15)" : "transparent",
                border: `1px solid ${tab === "graveyard" ? "var(--neon-pink)" : "rgba(255,255,255,0.1)"}`,
                color: tab === "graveyard" ? "var(--neon-pink)" : "rgba(255,255,255,0.35)",
                boxShadow: tab === "graveyard" ? "0 0 8px rgba(255,45,120,0.3)" : "none",
              }}
            >
              ⚰ GRAVEYARD ({graveyard.length})
            </button>
            <button
              onClick={() => { setTab("banished"); setSelectedId(null); }}
              className="px-3 py-0.5 text-[0.55rem] rounded transition-all"
              style={{
                fontFamily: "'Orbitron', sans-serif",
                background: tab === "banished" ? "rgba(180,79,255,0.15)" : "transparent",
                border: `1px solid ${tab === "banished" ? "#b44fff" : "rgba(255,255,255,0.1)"}`,
                color: tab === "banished" ? "#b44fff" : "rgba(255,255,255,0.35)",
                boxShadow: tab === "banished" ? "0 0 8px rgba(180,79,255,0.3)" : "none",
              }}
            >
              ✦ BANISHED ({banished.length})
            </button>
          </div>

          {/* Close */}
          <button
            onClick={onClose}
            className="text-sm opacity-40 hover:opacity-100 transition-opacity shrink-0"
            style={{ color: "var(--neon-pink)", fontFamily: "'Orbitron', sans-serif" }}
          >
            ✕
          </button>
        </div>

        {/* Card grid */}
        <div className="flex-1 overflow-y-auto p-4">
          {cards.length === 0 ? (
            <div
              className="w-full text-center py-8 opacity-25"
              style={{ fontFamily: "'Orbitron', sans-serif", color: accentColor, fontSize: "0.6rem" }}
            >
              {isGY ? "GRAVEYARD IS EMPTY" : isExtra ? "EXTRA DECK IS EMPTY" : "NO BANISHED CARDS"}
            </div>
          ) : (
            <div className="flex flex-wrap gap-2" style={{ alignContent: "flex-start" }}>
              {cards.map((card) => {
                const isSelected = selectedId === card.instanceId;
                return (
                  <div
                    key={card.instanceId}
                    className="relative flex-shrink-0 cursor-pointer rounded overflow-hidden transition-all"
                    style={{
                      width: "100px",
                      height: "140px",
                      border: isSelected
                        ? `2px solid ${accentColor}`
                        : "1px solid var(--border-dim)",
                      boxShadow: isSelected
                        ? `0 0 12px rgba(${accentRgb},0.6)`
                        : "none",
                      transform: isSelected ? "scale(1.08)" : "scale(1)",
                      filter: isGY ? "none" : isExtra ? "none" : "hue-rotate(60deg) brightness(0.8) saturate(1.2)",
                    }}
                    onClick={() => handleCardClick(card)}
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
                );
              })}
            </div>
          )}
        </div>

        {/* Footer hint */}
        <div
          className="px-4 py-1.5 flex-shrink-0 text-center"
          style={{
            borderTop: "1px solid rgba(255,255,255,0.05)",
            fontSize: "0.5rem",
            color: "rgba(255,255,255,0.25)",
            fontFamily: "'Orbitron', sans-serif",
            letterSpacing: "0.08em",
          }}
        >
          CLICK A CARD TO VIEW DETAILS
        </div>
      </div>
    </div>
  );
}
