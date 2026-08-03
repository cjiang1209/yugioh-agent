import type { CardInfo } from "../../../../shared/engineTypes";
import { CARD_BACK_URL } from "./CardZone";
import { useCardInfo } from "../../hooks/useCardInfo";

interface CardDetailProps {
  /** A board card: `id` picks the face to fetch, `name` is the fallback until
   *  it arrives. Everything printed comes from CardInfo. */
  card: { id: number; name: string };
}

const IMAGE_BASE = "https://images.ygoprodeck.com/images/cards";

const TEXT_STYLE = {
  fontFamily: "'Rajdhani', sans-serif",
  fontSize: "0.82rem",
  color: "#8aaec8",
  lineHeight: 1.5,
  fontWeight: 500,
} as const;

// 3x3 rosette; the centre cell is the card itself.
const ARROW_GRID: (string | null)[] = [
  "TOP_LEFT",
  "TOP",
  "TOP_RIGHT",
  "LEFT",
  null,
  "RIGHT",
  "BOTTOM_LEFT",
  "BOTTOM",
  "BOTTOM_RIGHT",
];

const ARROW_GLYPHS: Record<string, string> = {
  TOP_LEFT: "◤",
  TOP: "▲",
  TOP_RIGHT: "◥",
  LEFT: "◀",
  RIGHT: "▶",
  BOTTOM_LEFT: "◣",
  BOTTOM: "▼",
  BOTTOM_RIGHT: "◢",
};

/** A Link monster's arrows, laid out as they appear on the card. */
function LinkArrowRosette({ arrows }: { arrows: string[] }) {
  return (
    <div
      className="grid"
      style={{
        gridTemplateColumns: "repeat(3, 1rem)",
        gap: "1px",
        fontSize: "0.7rem",
        lineHeight: 1,
      }}
    >
      {ARROW_GRID.map((position, i) => {
        if (position === null) return <span key={i} />;
        const lit = arrows.includes(position);
        return (
          <span
            key={i}
            data-testid={`link-arrow-${position}`}
            data-lit={lit ? "true" : "false"}
            style={{
              color: lit ? "var(--neon-cyan)" : "#3a4a58",
              textAlign: "center",
            }}
          >
            {ARROW_GLYPHS[position]}
          </span>
        );
      })}
    </div>
  );
}

/** cards.cdb encodes "?" ATK/DEF as -2. */
function formatStat(value: number | null): string {
  if (value === null) return "?";
  return value < 0 ? "?" : String(value);
}

function formatLevel(info: CardInfo): string | null {
  if (info.level === null || info.level_kind === null) return null;
  if (info.level_kind === "rank") return `◆${info.level}`;
  if (info.level_kind === "link") return `LINK-${info.level}`;
  return `★${info.level}`;
}

/**
 * Card details as plain content — no header, width, border or scroll container.
 * The host decides sizing and what to show when nothing is selected.
 *
 * Only the image comes from YGOProDeck. Until the face arrives — or if the code
 * isn't in the database — the board's own name is shown, so the panel is never
 * blank.
 */
export function CardDetail({ card }: CardDetailProps) {
  const info = useCardInfo(card.id);

  const statLine = [info?.attribute, info ? formatLevel(info) : null]
    .filter(Boolean)
    .join(" ");

  return (
    <div className="flex flex-col gap-2">
      <div
        className="w-full rounded overflow-hidden"
        style={{ aspectRatio: "0.717", background: "rgba(255,255,255,0.04)" }}
      >
        <img
          src={`${IMAGE_BASE}/${card.id}.jpg`}
          alt={info?.name ?? card.name}
          className="w-full h-full object-contain"
          onError={e => {
            (e.target as HTMLImageElement).src = CARD_BACK_URL;
          }}
        />
      </div>

      <div
        style={{
          fontFamily: "'Orbitron', sans-serif",
          fontSize: "0.75rem",
          fontWeight: 700,
          color: "#e8f4ff",
          lineHeight: 1.4,
          letterSpacing: "0.03em",
        }}
      >
        {info?.name ?? card.name}
      </div>

      {statLine && <div style={TEXT_STYLE}>{statLine}</div>}

      {info && info.typeline.length > 0 && (
        <div style={TEXT_STYLE}>{info.typeline.join(" / ")}</div>
      )}

      {info?.scales && (
        <div style={TEXT_STYLE}>
          ◀ {info.scales.left} &nbsp; {info.scales.right} ▶
        </div>
      )}

      {info?.link_arrows && <LinkArrowRosette arrows={info.link_arrows} />}

      {info && info.card_type === "monster" && (
        <div className="flex gap-3">
          <span
            style={{
              ...TEXT_STYLE,
              fontWeight: 700,
              color: "var(--neon-cyan)",
            }}
          >
            ATK/{formatStat(info.attack)}
          </span>
          {info.defense !== null && (
            <span
              style={{
                ...TEXT_STYLE,
                fontWeight: 700,
                color: "var(--neon-yellow, #ffe066)",
              }}
            >
              DEF/{formatStat(info.defense)}
            </span>
          )}
        </div>
      )}

      {info?.desc && (
        <p
          style={{
            ...TEXT_STYLE,
            fontWeight: 400,
            lineHeight: 1.6,
            // cards.cdb line breaks are meaningful — Pendulum cards carry two
            // labelled sections. Default collapsing would merge them.
            whiteSpace: "pre-line",
          }}
        >
          {info.desc}
        </p>
      )}
    </div>
  );
}
