import { useEffect, useRef } from "react";

interface DuelLogProps {
  logs: string[];
}

export function DuelLog({ logs }: DuelLogProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  return (
    <div className="flex flex-col h-full">
      {/* Header — matches Card Detail header exactly */}
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
        DUEL LOG
      </div>
      {/* Log entries */}
      <div className="flex-1 overflow-y-auto" style={{ padding: "6px 8px" }}>
        {logs.map((log, i) => (
          <div
            key={i}
            className="leading-relaxed"
            style={{
              fontFamily: "'Rajdhani', sans-serif",
              fontSize: "0.82rem",
              color: i === logs.length - 1 ? "#e8f4ff" : "#8aaec8",
              fontWeight: i === logs.length - 1 ? 600 : 400,
              marginBottom: "2px",
            }}
          >
            <span style={{ color: "var(--neon-cyan)", opacity: 0.6 }}>▸ </span>
            {log}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
