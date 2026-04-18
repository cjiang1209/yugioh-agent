import { useEffect, useRef } from "react";

interface DuelLogProps {
  logs: string[];
  isReplaying?: boolean;
}

export function DuelLog({ logs, isReplaying }: DuelLogProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const prevLenRef = useRef(0);
  const animateFrom = prevLenRef.current;

  useEffect(() => {
    if (logs.length > prevLenRef.current) {
      bottomRef.current?.scrollIntoView({
        behavior: prevLenRef.current === 0 ? "auto" : "smooth",
      });
    }
    prevLenRef.current = logs.length;
  }, [logs]);

  return (
    <div className="flex flex-col h-full">
      {/* Log entries */}
      <div className="flex-1 overflow-y-auto" style={{ padding: "6px 8px" }}>
        {logs.map((log, i) => (
          <div
            key={i}
            className={`leading-relaxed${isReplaying && i >= animateFrom ? " log-entry-reveal" : ""}`}
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
