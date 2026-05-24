import { GameCard, YgoCard } from "../../../../shared/gameTypes";

interface CardTooltipProps {
  card: YgoCard | GameCard | null;
  position?: "left" | "right" | "top";
}

function getFrameColor(frameType: string): string {
  const map: Record<string, string> = {
    normal: "#c8a850",
    effect: "#c06020",
    ritual: "#4060a0",
    fusion: "#8040a0",
    synchro: "#e0e0e0",
    xyz: "#202020",
    link: "#1040c0",
    spell: "#1a9060",
    trap: "#8020a0",
  };
  return map[frameType] ?? "#888";
}

function getAttributeIcon(attr?: string): string {
  const map: Record<string, string> = {
    DARK: "🌑",
    LIGHT: "☀️",
    EARTH: "🌍",
    WATER: "💧",
    FIRE: "🔥",
    WIND: "🌪️",
    DIVINE: "✨",
  };
  return attr ? (map[attr] ?? "?") : "";
}

export function CardTooltip({ card, position = "right" }: CardTooltipProps) {
  if (!card || !card.id) return null;

  const frameColor = getFrameColor(card.frameType ?? "normal");
  const isMonster = card.type?.includes("Monster");

  return (
    <div
      className="rounded overflow-hidden"
      style={{
        width: "200px",
        background: "var(--bg-panel)",
        border: `1px solid ${frameColor}55`,
        boxShadow: `0 0 20px rgba(0,0,0,0.8), 0 0 8px ${frameColor}33`,
        zIndex: 9999,
      }}
    >
      {/* Card image */}
      <div style={{ height: "140px", background: frameColor + "22" }}>
        <img
          src={`https://images.ygoprodeck.com/images/cards/${card.id}.jpg`}
          alt={card.name}
          className="w-full h-full object-contain"
          onError={e => {
            (e.target as HTMLImageElement).style.display = "none";
          }}
        />
      </div>

      {/* Card info */}
      <div className="p-2">
        {/* Name + type bar */}
        <div
          className="px-1.5 py-0.5 mb-1.5 rounded-sm"
          style={{
            background: frameColor + "33",
            borderLeft: `2px solid ${frameColor}`,
          }}
        >
          <div
            className="text-xs font-bold leading-tight"
            style={{
              color: "var(--text-primary)",
              fontFamily: "'Orbitron', sans-serif",
              fontSize: "0.6rem",
            }}
          >
            {card.name}
          </div>
          <div
            className="text-[0.5rem] opacity-60 mt-0.5"
            style={{ color: "var(--text-secondary)" }}
          >
            {getAttributeIcon(card.attribute)} {card.type}
            {isMonster && card.level ? ` ★${card.level}` : ""}
          </div>
        </div>

        {/* ATK/DEF */}
        {isMonster && (
          <div className="flex gap-2 mb-1.5">
            <span
              className="text-[0.6rem] font-bold"
              style={{
                color: "var(--neon-cyan)",
                fontFamily: "'Share Tech Mono', monospace",
              }}
            >
              ATK/{card.atk ?? "?"}
            </span>
            <span
              className="text-[0.6rem] font-bold"
              style={{
                color: "var(--neon-yellow)",
                fontFamily: "'Share Tech Mono', monospace",
              }}
            >
              DEF/{card.def ?? "?"}
            </span>
          </div>
        )}

        {/* Description */}
        <p
          className="text-[0.5rem] leading-relaxed opacity-70"
          style={{
            color: "var(--text-secondary)",
            maxHeight: "60px",
            overflow: "hidden",
          }}
        >
          {card.desc}
        </p>
      </div>
    </div>
  );
}
