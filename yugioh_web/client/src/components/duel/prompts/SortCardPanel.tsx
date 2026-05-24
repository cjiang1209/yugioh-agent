import type {
  EngineAction,
  EnginePrompt,
} from "../../../../../shared/engineTypes";
import { CardThumbnail } from "../CardThumbnail";
import { PickedCardRow } from "./PickedCardRow";

interface SortCardPanelProps {
  actions: EngineAction[];
  prompt: EnginePrompt;
  onAction: (actionIndex: number) => void;
}

export function SortCardPanel({
  actions,
  prompt,
  onAction,
}: SortCardPanelProps) {
  const sortActions = actions.filter(a => a.category === "sort");
  const pickedCards = prompt.picked_cards ?? [];
  const count = prompt.count ?? 0;
  const nextPosition = pickedCards.length + 1;

  return (
    <div className="flex flex-col h-full" style={{ gap: "0" }}>
      {/* Header */}
      <div
        style={{
          padding: "8px 10px",
          borderBottom: "1px solid rgba(255,255,255,0.06)",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <span
          style={{
            fontFamily: "'Share Tech Mono', monospace",
            fontSize: "0.6rem",
            color: "#c8d8e8",
          }}
        >
          Order cards
        </span>
        <span
          style={{
            fontFamily: "'Orbitron', sans-serif",
            fontSize: "0.45rem",
            letterSpacing: "0.05em",
            color: "var(--neon-cyan)",
            opacity: 0.8,
          }}
        >
          {`Position ${nextPosition} of ${count}`}
        </span>
      </div>

      <PickedCardRow
        pickedCards={pickedCards}
        totalSlots={count}
        label="Position"
      />

      {/* Remaining cards grid */}
      <div
        className="flex-1 overflow-y-auto"
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(2, 1fr)",
          gap: "6px",
          padding: "8px",
          alignContent: "start",
          scrollbarWidth: "thin",
          scrollbarColor: "rgba(0,245,255,0.3) transparent",
        }}
      >
        {sortActions.map(action => (
          <button
            key={action.index}
            onClick={() => onAction(action.index)}
            className="transition-all"
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: "3px",
              padding: "6px 4px",
              borderRadius: "4px",
              border: "1px solid rgba(180,79,255,0.3)",
              background: "rgba(180,79,255,0.06)",
              cursor: "pointer",
            }}
            onMouseEnter={e => {
              e.currentTarget.style.background = "rgba(180,79,255,0.18)";
              e.currentTarget.style.borderColor = "rgba(180,79,255,0.6)";
            }}
            onMouseLeave={e => {
              e.currentTarget.style.background = "rgba(180,79,255,0.06)";
              e.currentTarget.style.borderColor = "rgba(180,79,255,0.3)";
            }}
          >
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
        ))}
      </div>
    </div>
  );
}
