import type { EngineAction, EnginePrompt } from "../../../../../shared/engineTypes";

const CARD_IMAGE_BASE = "https://images.ygoprodeck.com/images/cards_small";

interface SelectCardPanelProps {
  actions: EngineAction[];
  prompt: EnginePrompt;
  onAction: (actionIndex: number) => void;
}

export function SelectCardPanel({ actions, prompt, onAction }: SelectCardPanelProps) {
  const isTribute = prompt.type === "tribute";
  const cardActions = actions.filter(
    (a) => a.category === "select_card" || a.category === "tribute",
  );
  const finishAction = actions.find((a) => a.category === "finish");

  // Progress info
  let progressText = "";
  if (isTribute) {
    const total = prompt.release_total ?? 0;
    const minRelease = prompt.min_release ?? 1;
    const selected = prompt.cards_selected ?? 0;
    progressText = `Release: ${total}/${minRelease} (${selected} card${selected !== 1 ? "s" : ""})`;
  } else if (prompt.selected_count !== undefined) {
    const max = prompt.max ?? 1;
    progressText = `Selected: ${prompt.selected_count}/${max}`;
  } else if (prompt.min !== undefined && prompt.max !== undefined) {
    const { min, max } = prompt;
    progressText = min === max ? `Select ${min}` : `Select ${min}–${max}`;
  }

  const headerText = isTribute ? "Tribute cards" : "Select a card";

  return (
    <div className="flex flex-col h-full" style={{ gap: "0" }}>
      {/* Header with progress */}
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
          {headerText}
        </span>
        {progressText && (
          <span
            style={{
              fontFamily: "'Orbitron', sans-serif",
              fontSize: "0.45rem",
              letterSpacing: "0.05em",
              color: "var(--neon-cyan)",
              opacity: 0.8,
            }}
          >
            {progressText}
          </span>
        )}
      </div>

      {/* Card grid */}
      <div
        className="flex-1 overflow-y-auto"
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: "6px",
          padding: "8px",
          alignContent: "start",
          scrollbarWidth: "thin",
          scrollbarColor: "rgba(0,245,255,0.3) transparent",
        }}
      >
        {cardActions.map((action) => {
          const hasImage = action.card_code > 0;
          return (
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
              onMouseEnter={(e) => {
                e.currentTarget.style.background = "rgba(180,79,255,0.18)";
                e.currentTarget.style.borderColor = "rgba(180,79,255,0.6)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = "rgba(180,79,255,0.06)";
                e.currentTarget.style.borderColor = "rgba(180,79,255,0.3)";
              }}
            >
              {hasImage ? (
                <img
                  src={`${CARD_IMAGE_BASE}/${action.card_code}.jpg`}
                  alt={action.card_name}
                  style={{
                    width: "48px",
                    height: "70px",
                    objectFit: "cover",
                    borderRadius: "3px",
                    border: "1px solid rgba(180,79,255,0.4)",
                  }}
                  onError={(e) => {
                    (e.target as HTMLImageElement).style.display = "none";
                  }}
                />
              ) : (
                <div
                  style={{
                    width: "48px",
                    height: "70px",
                    borderRadius: "3px",
                    background: "rgba(255,255,255,0.04)",
                    border: "1px solid rgba(180,79,255,0.3)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: "0.7rem",
                    color: "#b44fff",
                    opacity: 0.5,
                  }}
                >
                  ?
                </div>
              )}
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
        })}
      </div>

      {/* Finish button */}
      {finishAction && (
        <div style={{ padding: "6px 8px", borderTop: "1px solid rgba(255,255,255,0.06)" }}>
          <button
            onClick={() => onAction(finishAction.index)}
            className="transition-all"
            style={{
              width: "100%",
              padding: "8px 0",
              borderRadius: "4px",
              border: "1px solid rgba(0,245,255,0.5)",
              background: "rgba(0,245,255,0.12)",
              color: "var(--neon-cyan)",
              fontFamily: "'Orbitron', sans-serif",
              fontSize: "0.5rem",
              letterSpacing: "0.1em",
              cursor: "pointer",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = "rgba(0,245,255,0.25)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = "rgba(0,245,255,0.12)";
            }}
          >
            {finishAction.description.toUpperCase()}
          </button>
        </div>
      )}
    </div>
  );
}
