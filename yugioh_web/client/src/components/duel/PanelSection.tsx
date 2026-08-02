import type { CSSProperties, ReactNode } from "react";

interface PanelSectionProps {
  title: string;
  children: ReactNode;
  /** Sizing and border overrides for this section within its panel. */
  style?: CSSProperties;
}

/**
 * A titled panel section: header plus a bounded body slot.
 *
 * The body does not scroll — children that need it bring their own scroll
 * container, so components like DuelLog keep ownership of their scrolling.
 */
export function PanelSection({ title, children, style }: PanelSectionProps) {
  return (
    <div className="flex flex-col" style={{ minHeight: 0, ...style }}>
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
        {title}
      </div>
      <div style={{ flex: "1 1 0", minHeight: 0 }}>{children}</div>
    </div>
  );
}
