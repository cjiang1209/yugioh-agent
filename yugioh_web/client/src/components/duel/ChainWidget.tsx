import type { PendingChainEntry } from "../../../../shared/engineTypes";
import { CardThumbnail } from "./CardThumbnail";

interface ChainWidgetProps {
  entries: PendingChainEntry[];
}

export function ChainWidget({ entries }: ChainWidgetProps) {
  if (entries.length === 0) return null;
  return (
    <div
      style={{
        position: "absolute",
        top: "8px",
        right: "8px",
        display: "flex",
        flexDirection: "column",
        gap: "6px",
        padding: "10px 12px",
        background: "rgba(0,0,0,0.82)",
        border: "1px solid rgba(0,245,255,0.35)",
        borderRadius: "6px",
        zIndex: 40,
        fontFamily: "'Share Tech Mono', monospace",
        fontSize: "0.7rem",
        color: "var(--text-primary, #e0f7ff)",
        maxWidth: "260px",
      }}
    >
      <div
        style={{ opacity: 0.7, fontSize: "0.6rem", letterSpacing: "0.05em" }}
      >
        CURRENT CHAIN
      </div>
      {entries.map(e => (
        <div
          key={e.chain_link}
          style={{ display: "flex", alignItems: "center", gap: "8px" }}
        >
          <span style={{ opacity: 0.7 }}>L{e.chain_link}</span>
          <CardThumbnail
            cardCode={e.card_code}
            width={28}
            height={41}
            borderColor="rgba(0,245,255,0.4)"
          />
          <div style={{ display: "flex", flexDirection: "column" }}>
            <span>
              {e.card_name}{" "}
              <span style={{ opacity: 0.6 }}>
                ({e.controller === 0 ? "YOU" : "OPP"})
              </span>
            </span>
            {e.effect_text && (
              <span style={{ opacity: 0.8 }}>{e.effect_text}</span>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
