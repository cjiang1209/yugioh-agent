import type {
  EngineAction,
  EnginePrompt,
} from "../../../../../shared/engineTypes";
import { PickedCardRow } from "./PickedCardRow";
import { SelectableCardTile } from "./SelectableCardTile";

interface SortCardPanelProps {
  actions: EngineAction[];
  prompt: EnginePrompt;
  onAction: (actionIndex: number) => void;
  recommendedIndex?: number | null;
  /** Policy probabilities for the prompt on screen, read by `action.index`. */
  actionProbs?: number[] | null;
}

export function SortCardPanel({
  actions,
  prompt,
  onAction,
  recommendedIndex,
  actionProbs,
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
          <SelectableCardTile
            key={action.index}
            action={action}
            isRecommended={
              recommendedIndex != null && action.index === recommendedIndex
            }
            probability={actionProbs?.[action.index]}
            onSelect={onAction}
          />
        ))}
      </div>
    </div>
  );
}
