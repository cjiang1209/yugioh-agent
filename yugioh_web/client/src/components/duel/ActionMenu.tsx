interface ActionItem {
  label: string;
  action: () => void;
  color?: string;
  disabled?: boolean;
}

interface ActionMenuProps {
  items: ActionItem[];
  x: number;
  y: number;
  onClose: () => void;
}

export function ActionMenu({ items, x, y, onClose }: ActionMenuProps) {
  // Clamp to viewport
  const adjustedX = Math.min(x, window.innerWidth - 180);
  const adjustedY = Math.min(y, window.innerHeight - items.length * 36 - 16);

  return (
    <>
      <div className="fixed inset-0 z-40" onClick={onClose} />
      <div
        className="fixed z-50 rounded overflow-hidden animate-slide-up"
        style={{
          left: adjustedX,
          top: adjustedY,
          background: "var(--bg-panel)",
          border: "1px solid var(--border-dim)",
          boxShadow: "0 0 20px rgba(0,0,0,0.8), 0 0 10px rgba(0,245,255,0.1)",
          minWidth: "160px",
        }}
      >
        {items.map((item, i) => (
          <button
            key={i}
            disabled={item.disabled}
            onClick={() => {
              if (!item.disabled) {
                item.action();
                onClose();
              }
            }}
            className="w-full text-left px-3 py-2 text-xs transition-all flex items-center gap-2"
            style={{
              fontFamily: "'Rajdhani', sans-serif",
              fontWeight: 600,
              color: item.disabled
                ? "rgba(150,200,230,0.2)"
                : item.color ?? "var(--text-primary)",
              background: "transparent",
              borderBottom: i < items.length - 1 ? "1px solid rgba(0,245,255,0.05)" : "none",
              cursor: item.disabled ? "not-allowed" : "pointer",
            }}
            onMouseEnter={(e) => {
              if (!item.disabled) {
                (e.currentTarget as HTMLButtonElement).style.background = "rgba(0,245,255,0.08)";
              }
            }}
            onMouseLeave={(e) => {
              (e.currentTarget as HTMLButtonElement).style.background = "transparent";
            }}
          >
            {item.label}
          </button>
        ))}
      </div>
    </>
  );
}
