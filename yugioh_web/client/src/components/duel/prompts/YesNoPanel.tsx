import type {
  EngineAction,
  EnginePrompt,
} from "../../../../../shared/engineTypes";
import { CardThumbnail } from "../CardThumbnail";

interface YesNoPanelProps {
  actions: EngineAction[];
  prompt: EnginePrompt;
  onAction: (actionIndex: number) => void;
}

export function YesNoPanel({ actions, prompt, onAction }: YesNoPanelProps) {
  const yesAction = actions.find(a => a.category === "yes");
  const noAction = actions.find(a => a.category === "no");

  const isEffect = prompt.type === "effect_yn";
  const cardCode = prompt.card_code ?? 0;
  const cardName = prompt.card_name ?? "";

  const question =
    prompt.prompt_text ??
    (isEffect
      ? `Activate effect of ${cardName || "this card"}?`
      : "Do you want to proceed?");

  return (
    <div
      className="flex flex-col h-full"
      style={{ padding: "12px 10px", gap: "10px" }}
    >
      {/* Card image for effect_yn */}
      {isEffect && cardCode > 0 && (
        <div style={{ display: "flex", justifyContent: "center" }}>
          <CardThumbnail
            cardCode={cardCode}
            width={80}
            height={116}
            borderRadius={4}
            borderColor="rgba(0,180,255,0.4)"
            boxShadow="0 0 12px rgba(0,180,255,0.15)"
            location={prompt.location}
            badgeSize={18}
            alt={cardName}
          />
        </div>
      )}

      {/* Question text */}
      <div
        style={{
          fontFamily: "'Share Tech Mono', monospace",
          fontSize: "0.6rem",
          color: "#c8d8e8",
          textAlign: "center",
          lineHeight: 1.4,
          padding: "0 4px",
        }}
      >
        {question}
      </div>

      {/* YES / NO buttons */}
      <div style={{ display: "flex", gap: "8px", marginTop: "auto" }}>
        {yesAction && (
          <button
            onClick={() => onAction(yesAction.index)}
            className="transition-all"
            style={{
              flex: 1,
              padding: "8px 0",
              borderRadius: "4px",
              border: "1px solid rgba(0,200,80,0.5)",
              background: "rgba(0,200,80,0.12)",
              color: "#00d850",
              fontFamily: "'Orbitron', sans-serif",
              fontSize: "0.55rem",
              letterSpacing: "0.1em",
              cursor: "pointer",
            }}
            onMouseEnter={e => {
              e.currentTarget.style.background = "rgba(0,200,80,0.25)";
            }}
            onMouseLeave={e => {
              e.currentTarget.style.background = "rgba(0,200,80,0.12)";
            }}
          >
            YES
          </button>
        )}
        {noAction && (
          <button
            onClick={() => onAction(noAction.index)}
            className="transition-all"
            style={{
              flex: 1,
              padding: "8px 0",
              borderRadius: "4px",
              border: "1px solid rgba(255,45,120,0.5)",
              background: "rgba(255,45,120,0.12)",
              color: "#ff2d78",
              fontFamily: "'Orbitron', sans-serif",
              fontSize: "0.55rem",
              letterSpacing: "0.1em",
              cursor: "pointer",
            }}
            onMouseEnter={e => {
              e.currentTarget.style.background = "rgba(255,45,120,0.25)";
            }}
            onMouseLeave={e => {
              e.currentTarget.style.background = "rgba(255,45,120,0.12)";
            }}
          >
            NO
          </button>
        )}
      </div>
    </div>
  );
}
