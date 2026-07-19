import type { EngineAction } from "../../../../../shared/engineTypes";
import { CardThumbnail } from "../CardThumbnail";
import {
  RecommendedBadge,
  RECOMMENDED_BORDER,
  RECOMMENDED_SHADOW,
} from "../RecommendedBadge";

interface SelectableCardTileProps {
  action: EngineAction;
  isRecommended: boolean;
  onSelect: (actionIndex: number) => void;
}

/**
 * A card tile in a 2-column pick grid (used by SelectCardPanel and
 * SortCardPanel): thumbnail + optional AI-assist star badge + name label.
 * When recommended, the button shows the amber ring and the badge; hover keeps
 * the amber border instead of reverting to the default purple.
 */
export function SelectableCardTile({
  action,
  isRecommended,
  onSelect,
}: SelectableCardTileProps) {
  return (
    <button
      onClick={() => onSelect(action.index)}
      className="transition-all"
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: "3px",
        padding: "6px 4px",
        borderRadius: "4px",
        border: isRecommended
          ? RECOMMENDED_BORDER
          : "1px solid rgba(180,79,255,0.3)",
        boxShadow: isRecommended ? RECOMMENDED_SHADOW : undefined,
        background: "rgba(180,79,255,0.06)",
        cursor: "pointer",
      }}
      onMouseEnter={e => {
        e.currentTarget.style.background = "rgba(180,79,255,0.18)";
        if (!isRecommended) {
          e.currentTarget.style.borderColor = "rgba(180,79,255,0.6)";
        }
      }}
      onMouseLeave={e => {
        e.currentTarget.style.background = "rgba(180,79,255,0.06)";
        if (!isRecommended) {
          e.currentTarget.style.borderColor = "rgba(180,79,255,0.3)";
        }
      }}
    >
      <div style={{ position: "relative" }}>
        <CardThumbnail
          cardCode={action.card_code}
          width={80}
          height={112}
          borderRadius={3}
          borderColor={
            action.card_code > 0
              ? "rgba(180,79,255,0.4)"
              : "rgba(180,79,255,0.3)"
          }
          location={action.location}
          badgeSize={18}
          alt={action.card_name}
          fallback={
            <span
              style={{ fontSize: "0.7rem", color: "#b44fff", opacity: 0.5 }}
            >
              ?
            </span>
          }
        />
        {isRecommended && <RecommendedBadge />}
      </div>
      <span
        style={{
          fontFamily: "'Share Tech Mono', monospace",
          fontSize: "0.45rem",
          color: "#c8d8e8",
          lineHeight: 1.2,
          textAlign: "center",
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
          width: "100%",
        }}
        title={action.card_name}
      >
        {action.card_name || action.description}
      </span>
    </button>
  );
}
