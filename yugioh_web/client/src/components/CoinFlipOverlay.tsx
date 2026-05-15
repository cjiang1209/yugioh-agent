import { useEffect, useRef } from "react";

const SPIN_MS = 2400;
const REVEAL_MS = 300;
const HOLD_MS = 1200;
const TOTAL_MS = SPIN_MS + REVEAL_MS + HOLD_MS;

const FIRST_COLOR = "var(--neon-cyan)";
const SECOND_COLOR = "var(--neon-pink)";

interface CoinFlipOverlayProps {
  result: 0 | 1;
  onComplete: () => void;
}

export function CoinFlipOverlay({ result, onComplete }: CoinFlipOverlayProps) {
  const onCompleteRef = useRef(onComplete);
  useEffect(() => {
    onCompleteRef.current = onComplete;
  });
  useEffect(() => {
    const id = setTimeout(() => onCompleteRef.current(), TOTAL_MS);
    return () => clearTimeout(id);
  }, []);

  const youGoFirst = result === 0;
  const accent = youGoFirst ? FIRST_COLOR : SECOND_COLOR;
  const label = youGoFirst ? "YOU GO FIRST" : "OPPONENT GOES FIRST";

  return (
    <div
      className="fixed inset-0 flex flex-col items-center justify-center z-50"
      style={{ background: "var(--bg-void)" }}
    >
      <style>{`
        @keyframes coin-spin {
          0%   { transform: rotateY(0deg); }
          100% { transform: rotateY(3600deg); }
        }
        @keyframes coin-flicker {
          0%, 100%   { border-color: ${FIRST_COLOR}; box-shadow: 0 0 30px ${FIRST_COLOR}, inset 0 0 20px ${FIRST_COLOR}; }
          50%        { border-color: ${SECOND_COLOR}; box-shadow: 0 0 30px ${SECOND_COLOR}, inset 0 0 20px ${SECOND_COLOR}; }
        }
        @keyframes coin-reveal {
          from { opacity: 0; transform: translateY(8px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>

      <div
        style={{
          width: 120,
          height: 120,
          borderRadius: "50%",
          border: `3px solid ${accent}`,
          boxShadow: `0 0 30px ${accent}, inset 0 0 20px ${accent}`,
          animation: `coin-spin ${SPIN_MS}ms cubic-bezier(0.2, 0.7, 0.2, 1) forwards, coin-flicker 800ms ease-in-out 3`,
          marginBottom: 32,
        }}
      />

      <div
        style={{
          fontFamily: "'Orbitron', sans-serif",
          fontSize: "1.2rem",
          letterSpacing: "0.2em",
          color: accent,
          textShadow: `0 0 12px ${accent}`,
          opacity: 0,
          animation: `coin-reveal ${REVEAL_MS}ms ease-out ${SPIN_MS}ms forwards`,
        }}
      >
        {label}
      </div>

      <div
        className="mt-2"
        style={{
          fontFamily: "'Share Tech Mono', monospace",
          fontSize: "0.6rem",
          color: "var(--text-secondary)",
          opacity: 0,
          animation: `coin-reveal ${REVEAL_MS}ms ease-out ${SPIN_MS}ms forwards`,
        }}
      >
        COIN FLIP
      </div>
    </div>
  );
}
