import type { EngineAction } from "../../../../shared/engineTypes";

const CARD_IMAGE_BASE = "https://images.ygoprodeck.com/images/cards_small";

const CATEGORY_COLORS: Record<string, { bg: string; border: string; text: string }> = {
  summon:         { bg: "rgba(0,200,80,0.15)",   border: "rgba(0,200,80,0.5)",   text: "#00d850" },
  special_summon: { bg: "rgba(0,200,80,0.15)",   border: "rgba(0,200,80,0.5)",   text: "#00d850" },
  attack:         { bg: "rgba(255,45,120,0.15)",  border: "rgba(255,45,120,0.5)", text: "#ff2d78" },
  activate:       { bg: "rgba(0,180,255,0.15)",   border: "rgba(0,180,255,0.5)",  text: "#00b4ff" },
  chain:          { bg: "rgba(0,180,255,0.15)",   border: "rgba(0,180,255,0.5)",  text: "#00b4ff" },
  to_battle:      { bg: "rgba(128,128,128,0.15)", border: "rgba(128,128,128,0.5)", text: "#aaa" },
  to_end:         { bg: "rgba(128,128,128,0.15)", border: "rgba(128,128,128,0.5)", text: "#aaa" },
  to_main2:       { bg: "rgba(128,128,128,0.15)", border: "rgba(128,128,128,0.5)", text: "#aaa" },
  pass:           { bg: "rgba(128,128,128,0.15)", border: "rgba(128,128,128,0.5)", text: "#aaa" },
  finish:         { bg: "rgba(128,128,128,0.15)", border: "rgba(128,128,128,0.5)", text: "#aaa" },
  yes:            { bg: "rgba(0,200,80,0.15)",    border: "rgba(0,200,80,0.5)",   text: "#00d850" },
  no:             { bg: "rgba(255,45,120,0.15)",   border: "rgba(255,45,120,0.5)", text: "#ff2d78" },
  monster_set:    { bg: "rgba(245,230,66,0.15)",  border: "rgba(245,230,66,0.5)", text: "#f5e642" },
  spell_set:      { bg: "rgba(245,230,66,0.15)",  border: "rgba(245,230,66,0.5)", text: "#f5e642" },
  reposition:     { bg: "rgba(245,230,66,0.15)",  border: "rgba(245,230,66,0.5)", text: "#f5e642" },
  select_card:    { bg: "rgba(180,79,255,0.15)",  border: "rgba(180,79,255,0.5)", text: "#b44fff" },
  position:       { bg: "rgba(245,230,66,0.15)",  border: "rgba(245,230,66,0.5)", text: "#f5e642" },
  place:          { bg: "rgba(245,230,66,0.15)",  border: "rgba(245,230,66,0.5)", text: "#f5e642" },
  option:         { bg: "rgba(180,79,255,0.15)",  border: "rgba(180,79,255,0.5)", text: "#b44fff" },
};

const DEFAULT_COLOR = { bg: "rgba(128,128,128,0.1)", border: "rgba(128,128,128,0.3)", text: "#888" };

function categoryLabel(cat: string): string {
  return cat.replace(/_/g, " ").toUpperCase();
}

interface EngineActionPanelProps {
  actions: EngineAction[];
  onAction: (actionIndex: number) => void;
}

export function EngineActionPanel({ actions, onAction }: EngineActionPanelProps) {
  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div
        className="px-3 py-2 flex-shrink-0"
        style={{
          borderBottom: "1px solid rgba(0,245,255,0.15)",
          fontFamily: "'Orbitron', sans-serif",
          fontSize: "0.75rem",
          letterSpacing: "0.1em",
          color: "var(--neon-cyan)",
        }}
      >
        ACTIONS
        <span
          style={{
            marginLeft: "0.5em",
            fontSize: "0.6rem",
            opacity: 0.5,
            fontFamily: "'Share Tech Mono', monospace",
          }}
        >
          ({actions.length})
        </span>
      </div>

      {/* Action list */}
      <div
        className="flex-1 overflow-y-auto"
        style={{ scrollbarWidth: "thin", scrollbarColor: "rgba(0,245,255,0.3) transparent" }}
      >
        {actions.length === 0 ? (
          <div
            className="p-4 text-center"
            style={{
              fontFamily: "'Share Tech Mono', monospace",
              fontSize: "0.6rem",
              color: "var(--text-secondary)",
              opacity: 0.5,
            }}
          >
            No actions available
          </div>
        ) : (
          actions.map((action) => {
            const colors = CATEGORY_COLORS[action.category] ?? DEFAULT_COLOR;
            return (
              <button
                key={action.index}
                onClick={() => onAction(action.index)}
                className="w-full text-left transition-all"
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "8px",
                  padding: "6px 10px",
                  borderBottom: "1px solid rgba(255,255,255,0.04)",
                  background: "transparent",
                  cursor: "pointer",
                  border: "none",
                  borderLeft: "none",
                  borderRight: "none",
                  borderTop: "none",
                }}
                onMouseEnter={(e) => {
                  (e.currentTarget as HTMLElement).style.background = colors.bg;
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget as HTMLElement).style.background = "transparent";
                }}
              >
                {/* Card thumbnail */}
                {action.card_code > 0 ? (
                  <img
                    src={`${CARD_IMAGE_BASE}/${action.card_code}.jpg`}
                    alt=""
                    style={{
                      width: "28px",
                      height: "40px",
                      objectFit: "cover",
                      borderRadius: "2px",
                      flexShrink: 0,
                      border: `1px solid ${colors.border}`,
                    }}
                    onError={(e) => {
                      (e.target as HTMLImageElement).style.display = "none";
                    }}
                  />
                ) : (
                  <div
                    style={{
                      width: "28px",
                      height: "40px",
                      flexShrink: 0,
                      borderRadius: "2px",
                      background: "rgba(255,255,255,0.04)",
                      border: `1px solid ${colors.border}`,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: "0.6rem",
                      color: colors.text,
                      opacity: 0.5,
                    }}
                  >
                    ?
                  </div>
                )}

                {/* Text content */}
                <div style={{ flex: 1, minWidth: 0 }}>
                  {/* Category badge */}
                  <div
                    style={{
                      display: "inline-block",
                      padding: "1px 5px",
                      borderRadius: "3px",
                      fontSize: "0.45rem",
                      fontFamily: "'Orbitron', sans-serif",
                      letterSpacing: "0.08em",
                      background: colors.bg,
                      border: `1px solid ${colors.border}`,
                      color: colors.text,
                      marginBottom: "2px",
                    }}
                  >
                    {categoryLabel(action.category)}
                  </div>
                  {/* Description */}
                  <div
                    style={{
                      fontFamily: "'Share Tech Mono', monospace",
                      fontSize: "0.55rem",
                      color: "#c8d8e8",
                      lineHeight: 1.3,
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    }}
                    title={action.description}
                  >
                    {action.description}
                  </div>
                </div>
              </button>
            );
          })
        )}
      </div>
    </div>
  );
}
