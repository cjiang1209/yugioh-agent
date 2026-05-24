interface LifePointsProps {
  name: string;
  lp: number;
  maxLp?: number;
  isActive?: boolean;
  isOpponent?: boolean;
  flash?: boolean;
}

export function LifePoints({
  name,
  lp,
  maxLp = 8000,
  isActive,
  isOpponent,
  flash,
}: LifePointsProps) {
  const pct = Math.max(0, Math.min(100, (lp / maxLp) * 100));
  const color =
    pct > 50
      ? "var(--neon-cyan)"
      : pct > 25
        ? "var(--neon-yellow)"
        : "var(--neon-pink)";

  return (
    <div
      className="hud-bracket rounded"
      style={{
        padding: "clamp(4px, 0.5vw, 8px) clamp(8px, 1.2vw, 16px)",
        background: flash
          ? "rgba(255,45,120,0.18)"
          : isActive
            ? "rgba(0,245,255,0.05)"
            : "rgba(0,0,0,0.6)",
        border: `1px solid ${flash ? "var(--neon-pink)" : isActive ? "var(--neon-cyan)" : "rgba(0,245,255,0.3)"}`,
        boxShadow: flash
          ? "0 0 24px rgba(255,45,120,0.7), inset 0 0 16px rgba(255,45,120,0.15)"
          : isActive
            ? "0 0 16px rgba(0,245,255,0.3), inset 0 0 10px rgba(0,245,255,0.03)"
            : "0 0 6px rgba(0,245,255,0.08)",
        minWidth: "clamp(120px, 18vw, 220px)",
        transition: "background 0.15s, border-color 0.15s, box-shadow 0.15s",
        animation: flash ? "lp-damage-flash 0.5s ease-out" : undefined,
      }}
    >
      <div
        className="flex items-center justify-between"
        style={{ marginBottom: "clamp(2px, 0.3vw, 5px)" }}
      >
        <span
          className="font-bold truncate"
          style={{
            fontFamily: "'Orbitron', sans-serif",
            color: isOpponent ? "var(--neon-pink)" : "var(--neon-cyan)",
            fontSize: "clamp(0.55rem, 1vw, 0.9rem)",
            letterSpacing: "0.1em",
            textShadow: isOpponent
              ? "0 0 6px rgba(255,45,120,0.6)"
              : "0 0 6px rgba(0,245,255,0.6)",
            maxWidth: "clamp(70px, 10vw, 130px)",
          }}
        >
          {name}
        </span>
        <span
          className="font-bold ml-2"
          style={{
            fontFamily: "'Share Tech Mono', monospace",
            color,
            textShadow: `0 0 10px ${color}, 0 0 20px ${color}66`,
            fontSize: "clamp(0.7rem, 1.4vw, 1.2rem)",
          }}
        >
          {lp.toLocaleString()}
        </span>
      </div>
      {/* LP Bar */}
      <div
        className="w-full rounded-full overflow-hidden"
        style={{
          height: "clamp(3px, 0.4vw, 5px)",
          background: "rgba(255,255,255,0.1)",
        }}
      >
        <div
          className="h-full rounded-full lp-bar-fill"
          style={{
            width: `${pct}%`,
            background: `linear-gradient(90deg, ${color}, ${color}cc)`,
            boxShadow: `0 0 8px ${color}, 0 0 16px ${color}66`,
          }}
        />
      </div>
    </div>
  );
}
