import type { EngineAction } from "../../../../shared/engineTypes";
import { CardThumbnail } from "./CardThumbnail";

const CATEGORY_COLORS: Record<
  string,
  { bg: string; border: string; text: string }
> = {
  summon: {
    bg: "rgba(0,200,80,0.15)",
    border: "rgba(0,200,80,0.5)",
    text: "#00d850",
  },
  special_summon: {
    bg: "rgba(0,200,80,0.15)",
    border: "rgba(0,200,80,0.5)",
    text: "#00d850",
  },
  attack: {
    bg: "rgba(255,45,120,0.15)",
    border: "rgba(255,45,120,0.5)",
    text: "#ff2d78",
  },
  activate: {
    bg: "rgba(0,180,255,0.15)",
    border: "rgba(0,180,255,0.5)",
    text: "#00b4ff",
  },
  chain: {
    bg: "rgba(0,180,255,0.15)",
    border: "rgba(0,180,255,0.5)",
    text: "#00b4ff",
  },
  to_battle: {
    bg: "rgba(128,128,128,0.15)",
    border: "rgba(128,128,128,0.5)",
    text: "#aaa",
  },
  to_end: {
    bg: "rgba(128,128,128,0.15)",
    border: "rgba(128,128,128,0.5)",
    text: "#aaa",
  },
  to_main2: {
    bg: "rgba(128,128,128,0.15)",
    border: "rgba(128,128,128,0.5)",
    text: "#aaa",
  },
  pass: {
    bg: "rgba(128,128,128,0.15)",
    border: "rgba(128,128,128,0.5)",
    text: "#aaa",
  },
  finish: {
    bg: "rgba(128,128,128,0.15)",
    border: "rgba(128,128,128,0.5)",
    text: "#aaa",
  },
  yes: {
    bg: "rgba(0,200,80,0.15)",
    border: "rgba(0,200,80,0.5)",
    text: "#00d850",
  },
  no: {
    bg: "rgba(255,45,120,0.15)",
    border: "rgba(255,45,120,0.5)",
    text: "#ff2d78",
  },
  monster_set: {
    bg: "rgba(245,230,66,0.15)",
    border: "rgba(245,230,66,0.5)",
    text: "#f5e642",
  },
  spell_set: {
    bg: "rgba(245,230,66,0.15)",
    border: "rgba(245,230,66,0.5)",
    text: "#f5e642",
  },
  reposition: {
    bg: "rgba(245,230,66,0.15)",
    border: "rgba(245,230,66,0.5)",
    text: "#f5e642",
  },
  select_card: {
    bg: "rgba(180,79,255,0.15)",
    border: "rgba(180,79,255,0.5)",
    text: "#b44fff",
  },
  position: {
    bg: "rgba(245,230,66,0.15)",
    border: "rgba(245,230,66,0.5)",
    text: "#f5e642",
  },
  place: {
    bg: "rgba(245,230,66,0.15)",
    border: "rgba(245,230,66,0.5)",
    text: "#f5e642",
  },
  option: {
    bg: "rgba(180,79,255,0.15)",
    border: "rgba(180,79,255,0.5)",
    text: "#b44fff",
  },
};

const DEFAULT_COLOR = {
  bg: "rgba(128,128,128,0.1)",
  border: "rgba(128,128,128,0.3)",
  text: "#888",
};

function categoryLabel(cat: string): string {
  return cat.replace(/_/g, " ").toUpperCase();
}

interface EngineActionPanelProps {
  actions: EngineAction[];
  onAction: (actionIndex: number) => void;
  recommendedIndex?: number | null;
}

export function EngineActionPanel({
  actions,
  onAction,
  recommendedIndex,
}: EngineActionPanelProps) {
  return (
    <div className="flex flex-col h-full">
      {/* Action list */}
      <div
        className="flex-1 overflow-y-auto"
        style={{
          scrollbarWidth: "thin",
          scrollbarColor: "rgba(0,245,255,0.3) transparent",
        }}
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
          actions.map(action => {
            const colors = CATEGORY_COLORS[action.category] ?? DEFAULT_COLOR;
            const isRecommended =
              recommendedIndex != null && action.index === recommendedIndex;
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
                onMouseEnter={e => {
                  (e.currentTarget as HTMLElement).style.background = colors.bg;
                }}
                onMouseLeave={e => {
                  (e.currentTarget as HTMLElement).style.background =
                    "transparent";
                }}
              >
                {/* Thumbnail with corner recommendation badge */}
                <div style={{ position: "relative", flexShrink: 0 }}>
                  <CardThumbnail
                    cardCode={action.card_code}
                    width={50}
                    height={70}
                    borderColor={colors.border}
                    location={action.location}
                    fallback={
                      <span
                        style={{
                          fontSize: "0.6rem",
                          color: colors.text,
                          opacity: 0.5,
                        }}
                      >
                        ?
                      </span>
                    }
                  />
                  {isRecommended && (
                    <span
                      title="Recommended by AI Assist"
                      style={{
                        position: "absolute",
                        top: "-5px",
                        left: "-5px",
                        width: "18px",
                        height: "18px",
                        borderRadius: "50%",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        fontSize: "0.6rem",
                        lineHeight: 1,
                        background: "#ffb020",
                        color: "#1a1200",
                        border: "1px solid #ffd67a",
                        boxShadow: "0 0 8px rgba(255,176,32,0.9)",
                      }}
                    >
                      ★
                    </span>
                  )}
                </div>

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
