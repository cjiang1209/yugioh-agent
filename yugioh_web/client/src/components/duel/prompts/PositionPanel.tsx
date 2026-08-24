import type {
  EngineAction,
  EnginePrompt,
} from "../../../../../shared/engineTypes";
import { CardThumbnail } from "../CardThumbnail";
import {
  RecommendedBadge,
  RECOMMENDED_BORDER,
  RECOMMENDED_SHADOW,
} from "../RecommendedBadge";
import { ActionProbability } from "../ActionProbability";

const POSITION_ICONS: Record<string, string> = {
  "Face-up Attack": "\u2694\uFE0F",
  "Face-down Attack": "\uD83D\uDDE1\uFE0F",
  "Face-up Defense": "\uD83D\uDEE1\uFE0F",
  "Face-down Defense": "\uD83C\uDCCF",
};

const POSITION_COLORS: Record<
  string,
  { bg: string; border: string; text: string }
> = {
  "Face-up Attack": {
    bg: "rgba(255,45,120,0.12)",
    border: "rgba(255,45,120,0.5)",
    text: "#ff2d78",
  },
  "Face-down Attack": {
    bg: "rgba(255,45,120,0.12)",
    border: "rgba(255,45,120,0.5)",
    text: "#ff2d78",
  },
  "Face-up Defense": {
    bg: "rgba(0,180,255,0.12)",
    border: "rgba(0,180,255,0.5)",
    text: "#00b4ff",
  },
  "Face-down Defense": {
    bg: "rgba(245,230,66,0.12)",
    border: "rgba(245,230,66,0.5)",
    text: "#f5e642",
  },
};

const DEFAULT_POS_COLOR = {
  bg: "rgba(128,128,128,0.12)",
  border: "rgba(128,128,128,0.5)",
  text: "#aaa",
};

interface PositionPanelProps {
  actions: EngineAction[];
  prompt: EnginePrompt;
  onAction: (actionIndex: number) => void;
  recommendedIndex?: number | null;
  /** Policy probabilities for the prompt on screen, read by `action.index`. */
  actionProbs?: number[] | null;
}

export function PositionPanel({
  actions,
  prompt,
  onAction,
  recommendedIndex,
  actionProbs,
}: PositionPanelProps) {
  const cardCode = prompt.card_code ?? 0;
  const cardName = prompt.card_name ?? "";

  const positionActions = actions.filter(a => a.category === "position");

  return (
    <div
      className="flex flex-col h-full"
      style={{ padding: "12px 10px", gap: "10px" }}
    >
      {/* Card image */}
      {cardCode > 0 && (
        <div style={{ display: "flex", justifyContent: "center" }}>
          <CardThumbnail
            cardCode={cardCode}
            width={80}
            height={112}
            borderRadius={4}
            borderColor="rgba(245,230,66,0.4)"
            boxShadow="0 0 12px rgba(245,230,66,0.15)"
            alt={cardName}
          />
        </div>
      )}

      {/* Header */}
      <div
        style={{
          fontFamily: "'Share Tech Mono', monospace",
          fontSize: "0.6rem",
          color: "#c8d8e8",
          textAlign: "center",
        }}
      >
        Choose position{cardName ? ` for ${cardName}` : ""}
      </div>

      {/* Position buttons */}
      <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
        {positionActions.map(action => {
          const posName = action.description.replace(`${cardName}: `, "");
          const colors = POSITION_COLORS[posName] ?? DEFAULT_POS_COLOR;
          const icon = POSITION_ICONS[posName] ?? "";
          const isRecommended =
            recommendedIndex != null && action.index === recommendedIndex;

          return (
            <button
              key={action.index}
              onClick={() => onAction(action.index)}
              className="transition-all"
              style={{
                position: "relative",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: "8px",
                padding: "8px 12px",
                borderRadius: "4px",
                border: isRecommended
                  ? RECOMMENDED_BORDER
                  : `1px solid ${colors.border}`,
                boxShadow: isRecommended ? RECOMMENDED_SHADOW : undefined,
                background: colors.bg,
                color: colors.text,
                fontFamily: "'Orbitron', sans-serif",
                fontSize: "0.5rem",
                letterSpacing: "0.08em",
                cursor: "pointer",
                textAlign: "center",
              }}
              onMouseEnter={e => {
                e.currentTarget.style.background = colors.bg.replace(
                  "0.12",
                  "0.25"
                );
              }}
              onMouseLeave={e => {
                e.currentTarget.style.background = colors.bg;
              }}
            >
              {isRecommended && <RecommendedBadge />}
              <ActionProbability value={actionProbs?.[action.index]} />
              <span style={{ fontSize: "0.8rem" }}>{icon}</span>
              <span>{posName.toUpperCase()}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
