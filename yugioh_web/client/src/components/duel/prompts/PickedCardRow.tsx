import type { PickedCard } from "../../../../../shared/engineTypes";
import { CardThumbnail } from "../CardThumbnail";

interface PickedCardRowProps {
  pickedCards: PickedCard[];
  /** Total slot count. Render dashed placeholders up to this number. Omit to render only picked thumbnails (no placeholders). */
  totalSlots?: number;
  /** Label template for each slot. Rendered as "{label} {n}". */
  label: "Position" | "Pick";
}

export function PickedCardRow({
  pickedCards,
  totalSlots,
  label,
}: PickedCardRowProps) {
  const slotCount = totalSlots ?? pickedCards.length;
  if (slotCount === 0) {
    return null;
  }
  const slots = Array.from(
    { length: slotCount },
    (_, i) => pickedCards[i] ?? null
  );

  return (
    <div
      style={{
        display: "flex",
        gap: "6px",
        padding: "8px",
        borderBottom: "1px solid rgba(255,255,255,0.06)",
        overflowX: "auto",
      }}
    >
      {slots.map((card, i) => (
        <div
          key={i}
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: "2px",
            minWidth: 60,
          }}
        >
          {card ? (
            <CardThumbnail
              cardCode={card.code}
              width={50}
              height={70}
              borderRadius={3}
              borderColor="rgba(0,245,255,0.5)"
              location={card.location}
              badgeSize={14}
              alt={`${label} ${i + 1}`}
            />
          ) : (
            <div
              style={{
                width: 50,
                height: 70,
                borderRadius: 3,
                border: "1px dashed rgba(180,79,255,0.3)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: "rgba(180,79,255,0.5)",
                fontFamily: "'Share Tech Mono', monospace",
                fontSize: "1.0rem",
              }}
            >
              ?
            </div>
          )}
          <span
            style={{
              fontFamily: "'Share Tech Mono', monospace",
              fontSize: "0.4rem",
              color: card ? "var(--neon-cyan)" : "#7a8a9a",
            }}
          >
            {`${label} ${i + 1}`}
          </span>
        </div>
      ))}
    </div>
  );
}
