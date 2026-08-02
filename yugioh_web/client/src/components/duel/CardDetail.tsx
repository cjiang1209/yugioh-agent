import { useEffect, useReducer } from "react";
import type { GameCard } from "../../../../shared/gameTypes";
import { CARD_BACK_URL } from "./CardZone";
import { getCachedDesc, setCachedDesc } from "../../lib/cardDescCache";

interface CardDetailProps {
  card: GameCard;
}

const IMAGE_BASE = "https://images.ygoprodeck.com/images/cards";

/**
 * Card details as plain content — no header, width, border or scroll
 * container, so it can sit in a panel, a modal or a tooltip. The host decides
 * sizing and what to show when nothing is selected.
 */
export function CardDetail({ card }: CardDetailProps) {
  const cached = card.desc || getCachedDesc(card.id);
  const [, bump] = useReducer((n: number) => n + 1, 0);

  useEffect(() => {
    if (!card.id || cached !== undefined) return; // id 0 = face-down/hidden
    const id = card.id;
    let cancelled = false;
    fetch(`https://db.ygoprodeck.com/api/v7/cardinfo.php?id=${id}`)
      .then(r => (r.ok ? r.json() : null))
      .then(data => {
        // Only cache a description we actually got. The cache is checked with
        // `!== undefined`, so caching "" from an error body (YGOProDeck answers
        // 429s with an error JSON) would suppress every later retry and leave
        // the card blank for the rest of the session.
        const fetched = data?.data?.[0]?.desc;
        if (typeof fetched !== "string") return;
        setCachedDesc(id, fetched);
        if (!cancelled) bump();
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [card.id, cached]);

  const desc = cached ?? "";

  return (
    <div className="flex flex-col gap-2">
      <div
        className="w-full rounded overflow-hidden"
        style={{ aspectRatio: "0.717", background: "rgba(255,255,255,0.04)" }}
      >
        <img
          src={`${IMAGE_BASE}/${card.id}.jpg`}
          alt={card.name}
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
        {card.name}
      </div>

      <div
        style={{
          fontFamily: "'Rajdhani', sans-serif",
          fontSize: "0.82rem",
          color: "#8aaec8",
          lineHeight: 1.5,
          fontWeight: 500,
        }}
      >
        {card.type}
        {card.level ? ` · ★${card.level}` : ""}
        {card.attribute ? ` · ${card.attribute}` : ""}
      </div>

      {card.type?.includes("Monster") && (
        <div className="flex gap-3">
          <span
            style={{
              fontFamily: "'Rajdhani', sans-serif",
              fontSize: "0.82rem",
              color: "var(--neon-cyan)",
              fontWeight: 700,
            }}
          >
            ATK/{card.atk ?? "?"}
          </span>
          <span
            style={{
              fontFamily: "'Rajdhani', sans-serif",
              fontSize: "0.82rem",
              color: "var(--neon-yellow, #ffe066)",
              fontWeight: 700,
            }}
          >
            DEF/{card.def ?? "?"}
          </span>
        </div>
      )}

      <p
        style={{
          fontFamily: "'Rajdhani', sans-serif",
          fontSize: "0.82rem",
          color: "#8aaec8",
          lineHeight: 1.6,
          fontWeight: 400,
        }}
      >
        {desc}
      </p>
    </div>
  );
}
