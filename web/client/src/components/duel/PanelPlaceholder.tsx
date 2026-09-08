interface PanelPlaceholderProps {
  icon?: string;
  label: string;
}

/** Muted centred empty state for a panel section. */
export function PanelPlaceholder({ icon, label }: PanelPlaceholderProps) {
  return (
    <div
      className="hud-label h-full flex flex-col items-center justify-center gap-2 opacity-30"
      style={{ textAlign: "center" }}
    >
      {icon && <div style={{ fontSize: "var(--ui-text-2xl)" }}>{icon}</div>}
      <div>{label}</div>
    </div>
  );
}
