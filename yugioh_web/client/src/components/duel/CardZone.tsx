import React, { useState } from "react";
import { FieldCard, GameCard } from "../../../../shared/gameTypes";

interface CardZoneProps {
  slot: FieldCard | null;
  label?: string;
  size?: "sm" | "md" | "lg";
  isSelected?: boolean;
  isValidTarget?: boolean;
  canPlace?: boolean;
  isOpponent?: boolean;
  onClick?: () => void;
  onContextMenu?: (e: React.MouseEvent) => void;
  className?: string;
  showLabel?: boolean;
}

const CARD_BACK_URL = "https://images.ygoprodeck.com/images/cards/back_high.jpg";

// Natural card dimensions — always displayed at true size
const CARD_SIZES = {
  sm: {
    width:  "80px",
    height: "112px",
  },
  md: {
    width:  "100px",
    height: "140px",
  },
  lg: {
    width:  "120px",
    height: "168px",
  },
};

export function CardZone({
  slot,
  label,
  size = "md",
  isSelected,
  isValidTarget,
  canPlace,
  isOpponent,
  onClick,
  onContextMenu,
  className = "",
  showLabel = true,
}: CardZoneProps) {
  const dims = CARD_SIZES[size];

  const zoneClass = [
    "card-zone",
    isSelected ? "selected" : "",
    isValidTarget ? "valid-target" : "",
    canPlace ? "can-place" : "",
    className,
  ]
    .filter(Boolean)
    .join(" ");

  const isDefPos = slot?.position === "DEF" || slot?.position === "FACE_DOWN_DEF";
  const isFaceDown = slot?.faceDown;

  // Face-down DEF cards are rotated 90° and displayed at full natural size.
  // They intentionally overflow the zone boundaries — this matches the physical card game look.

  return (
    <div
      className={zoneClass}
      style={{ width: dims.width, height: dims.height, flexShrink: 0, overflow: "visible" }}
      onClick={onClick}
      onContextMenu={onContextMenu}
      title={slot?.card.name ?? label}
    >
      {slot ? (
        <>
          <div
            className="w-full h-full"
            style={{
              transform: isDefPos ? "rotate(90deg)" : "none",
              transition: "transform 0.2s ease",
            }}
          >
            {isFaceDown ? (
              <div className="card-back w-full h-full" />
            ) : (
              <img
                src={
                  slot.card.id
                    ? `https://images.ygoprodeck.com/images/cards_small/${slot.card.id}.jpg`
                    : CARD_BACK_URL
                }
                alt={slot.card.name}
                className="w-full h-full object-cover rounded-sm"
                onError={(e) => {
                  (e.target as HTMLImageElement).src = CARD_BACK_URL;
                }}
              />
            )}
          </div>
          {/* ATK/DEF badge — outside rotated container so it stays at the bottom */}
          {label === "MONSTER" && slot.card.atk !== undefined && !isFaceDown && (
            <div
              className="absolute bottom-0 left-0 right-0 text-center font-bold"
              style={{
                fontSize: "0.65rem",
                paddingBlock: "2px",
                background: "rgba(0,0,0,0.8)",
                color: slot.position === "ATK" ? "var(--neon-cyan)" : "var(--neon-yellow)",
                zIndex: 1,
              }}
            >
              {slot.position === "ATK" ? slot.card.atk : slot.card.def}
            </div>
          )}
        </>
      ) : (
        showLabel && label ? (
          <span
            className="text-center leading-tight"
            style={{
              fontFamily: "'Orbitron', sans-serif",
              fontSize: "0.6rem",
              color: "var(--neon-cyan)",
              opacity: 0.7,
              textShadow: "0 0 4px rgba(0,245,255,0.4)",
            }}
          >
            {label}
          </span>
        ) : null
      )}
    </div>
  );
}

// ─── Hand Card ────────────────────────────────────────────────────────────────

interface HandCardProps {
  card: GameCard;
  index: number;
  isSelected?: boolean;
  isOpponentCard?: boolean;
  pileMode?: boolean;
  pileOffset?: number; // left offset in px for this card in pile
  onClick?: () => void;
  onContextMenu?: (e: React.MouseEvent) => void;
}

export function HandCard({ card, index, isSelected, isOpponentCard, pileMode, pileOffset, onClick, onContextMenu }: HandCardProps) {
  const w = "100px";
  const h = "140px";
  const [hovered, setHovered] = useState(false);

  if (isOpponentCard) {
    if (pileMode) {
      return (
        <div
          className="absolute flex-shrink-0 cursor-default transition-all duration-150"
          style={{
            width: w,
            height: h,
            left: pileOffset ?? 0,
            top: 6,
            zIndex: index,
            borderRadius: "4px",
            boxShadow: "0 2px 8px rgba(0,0,0,0.6)",
          }}
        >
          <div className="card-back w-full h-full" />
        </div>
      );
    }
    return (
      <div
        className="relative flex-shrink-0 cursor-default"
        style={{ width: w, height: h }}
      >
        <div className="card-back w-full h-full" />
      </div>
    );
  }

  if (pileMode) {
    const isLifted = hovered || isSelected;
    return (
      <div
        className="absolute flex-shrink-0 cursor-pointer transition-all duration-150"
        style={{
          width: w,
          height: h,
          left: pileOffset ?? 0,
          top: 8,
          zIndex: isLifted ? 1000 : index,
          transform: isSelected ? "translateY(-10px)" : isLifted ? "translateY(-10px)" : "translateY(0)",
          border: isSelected ? "1px solid var(--neon-pink)" : "1px solid var(--border-dim)",
          borderRadius: "4px",
          boxShadow: isSelected
            ? "0 0 14px rgba(255,45,120,0.7), 0 -4px 14px rgba(255,45,120,0.4)"
            : "0 2px 8px rgba(0,0,0,0.5)",
        }}
        onClick={onClick}
        onContextMenu={onContextMenu}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        title={card.name}
      >
        <img
          src={`https://images.ygoprodeck.com/images/cards_small/${card.id}.jpg`}
          alt={card.name}
          className="w-full h-full object-cover rounded-sm"
          onError={(e) => {
            (e.target as HTMLImageElement).src =
              "https://images.ygoprodeck.com/images/cards/back_high.jpg";
          }}
        />
      </div>
    );
  }

  return (
    <div
      className={`relative flex-shrink-0 cursor-pointer transition-all duration-150`}
      style={{
        width: w,
        height: h,
        transform: isSelected ? "translateY(-10px)" : "translateY(0)",
        border: isSelected ? "1px solid var(--neon-pink)" : "1px solid var(--border-dim)",
        borderRadius: "4px",
        boxShadow: isSelected
          ? "0 0 14px rgba(255,45,120,0.7), 0 -4px 14px rgba(255,45,120,0.4)"
          : "0 2px 8px rgba(0,0,0,0.5)",
      }}
      onClick={onClick}
      onContextMenu={onContextMenu}
      title={card.name}
    >
      <img
        src={`https://images.ygoprodeck.com/images/cards_small/${card.id}.jpg`}
        alt={card.name}
        className="w-full h-full object-cover rounded-sm"
        onError={(e) => {
          (e.target as HTMLImageElement).src =
            "https://images.ygoprodeck.com/images/cards/back_high.jpg";
        }}
      />
    </div>
  );
}
