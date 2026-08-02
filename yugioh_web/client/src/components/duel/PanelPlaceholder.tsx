interface PanelPlaceholderProps {
  icon?: string;
  label: string;
}

/** Muted centred empty state for a panel section. */
export function PanelPlaceholder({ icon, label }: PanelPlaceholderProps) {
  return (
    <div
      className="h-full flex flex-col items-center justify-center gap-2 opacity-30"
      style={{
        fontFamily: "'Orbitron', sans-serif",
        fontSize: "0.65rem",
        color: "var(--neon-cyan)",
        letterSpacing: "0.1em",
        textAlign: "center",
      }}
    >
      {icon && <div style={{ fontSize: "1.5rem" }}>{icon}</div>}
      <div>{label}</div>
    </div>
  );
}
