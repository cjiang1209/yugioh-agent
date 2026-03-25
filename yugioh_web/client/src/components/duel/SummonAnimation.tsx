import { useEffect, useRef } from "react";

export interface SummonAnimationProps {
  /** Screen-space rect of the zone being summoned to */
  zoneRect: DOMRect;
  /** "normal" = gold/white, "special" = cyan/blue */
  kind: "normal" | "special";
  /** Called when the animation finishes (~650ms) */
  onDone: () => void;
}

function centerOf(rect: DOMRect) {
  return { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 };
}

// Particles that rise upward from the zone
const RISE_PARTICLES = Array.from({ length: 16 }, (_, i) => ({
  id: i,
  x: -40 + Math.random() * 80,
  delay: Math.random() * 200,
  size: 3 + Math.random() * 4,
  rise: 60 + Math.random() * 60,
  drift: -20 + Math.random() * 40,
}));

// Burst particles radiating outward
const BURST_PARTICLES = Array.from({ length: 12 }, (_, i) => ({
  id: i,
  angle: (360 / 12) * i,
  dist: 50 + Math.random() * 40,
  size: 4 + Math.random() * 4,
  delay: Math.random() * 80,
}));

export function SummonAnimation({ zoneRect, kind, onDone }: SummonAnimationProps) {
  const doneRef = useRef(false);
  const center = centerOf(zoneRect);

  const primaryColor = kind === "normal" ? "#ffcc00" : "#00e5ff";
  const secondaryColor = kind === "normal" ? "#ff9900" : "#0088ff";
  const glowColor = kind === "normal" ? "rgba(255,200,0,0.6)" : "rgba(0,229,255,0.6)";

  useEffect(() => {
    const t = setTimeout(() => {
      if (!doneRef.current) {
        doneRef.current = true;
        onDone();
      }
    }, 650);
    return () => clearTimeout(t);
  }, [onDone]);

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 9998,
        pointerEvents: "none",
        overflow: "hidden",
      }}
    >
      {/* Zone glow backdrop */}
      <div
        style={{
          position: "absolute",
          left: zoneRect.left - 10,
          top: zoneRect.top - 10,
          width: zoneRect.width + 20,
          height: zoneRect.height + 20,
          borderRadius: 8,
          background: `radial-gradient(ellipse, ${glowColor} 0%, transparent 70%)`,
          animation: "summon-zone-glow 0.65s ease-out forwards",
        }}
      />

      {/* Outer ring */}
      <div
        style={{
          position: "absolute",
          left: center.x,
          top: center.y,
          width: 20,
          height: 20,
          marginLeft: -10,
          marginTop: -10,
          borderRadius: "50%",
          border: `3px solid ${primaryColor}`,
          boxShadow: `0 0 12px 4px ${glowColor}`,
          animation: "summon-ring 0.65s cubic-bezier(0.1,0.5,0.3,1) forwards",
        }}
      />
      {/* Second ring (delayed) */}
      <div
        style={{
          position: "absolute",
          left: center.x,
          top: center.y,
          width: 20,
          height: 20,
          marginLeft: -10,
          marginTop: -10,
          borderRadius: "50%",
          border: `2px solid ${secondaryColor}88`,
          animation: "summon-ring 0.65s cubic-bezier(0.1,0.5,0.3,1) 100ms forwards",
        }}
      />

      {/* Core flash */}
      <div
        style={{
          position: "absolute",
          left: center.x,
          top: center.y,
          width: 60,
          height: 60,
          marginLeft: -30,
          marginTop: -30,
          borderRadius: "50%",
          background: `radial-gradient(circle, rgba(255,255,255,0.95) 0%, ${primaryColor}cc 40%, transparent 100%)`,
          animation: "summon-core 0.65s ease-out forwards",
        }}
      />

      {/* Vertical energy pillar */}
      <div
        style={{
          position: "absolute",
          left: center.x,
          top: zoneRect.top - 80,
          width: 4,
          height: zoneRect.height + 80,
          marginLeft: -2,
          background: `linear-gradient(180deg, transparent 0%, ${primaryColor} 40%, ${primaryColor} 60%, transparent 100%)`,
          boxShadow: `0 0 12px 4px ${glowColor}`,
          borderRadius: 2,
          animation: "summon-pillar 0.65s ease-out forwards",
        }}
      />

      {/* Burst particles */}
      {BURST_PARTICLES.map((p) => {
        const rad = (p.angle * Math.PI) / 180;
        const tx = Math.cos(rad) * p.dist;
        const ty = Math.sin(rad) * p.dist;
        return (
          <div
            key={p.id}
            style={{
              position: "absolute",
              left: center.x,
              top: center.y,
              width: p.size,
              height: p.size,
              marginLeft: -p.size / 2,
              marginTop: -p.size / 2,
              borderRadius: "50%",
              background: p.id % 2 === 0 ? primaryColor : secondaryColor,
              boxShadow: `0 0 ${p.size * 2}px ${p.size}px ${glowColor}`,
              animation: "summon-burst-particle 0.65s ease-out forwards",
              animationDelay: `${p.delay}ms`,
              ["--tx" as string]: `${tx}px`,
              ["--ty" as string]: `${ty}px`,
            } as React.CSSProperties}
          />
        );
      })}

      {/* Rising particles */}
      {RISE_PARTICLES.map((p) => (
        <div
          key={p.id}
          style={{
            position: "absolute",
            left: center.x + p.x,
            top: center.y,
            width: p.size,
            height: p.size,
            marginLeft: -p.size / 2,
            marginTop: -p.size / 2,
            borderRadius: "50%",
            background: primaryColor,
            boxShadow: `0 0 ${p.size}px ${p.size / 2}px ${glowColor}`,
            animation: "summon-rise-particle 0.65s ease-out forwards",
            animationDelay: `${p.delay}ms`,
            ["--rise" as string]: `-${p.rise}px`,
            ["--drift" as string]: `${p.drift}px`,
          } as React.CSSProperties}
        />
      ))}

      <style>{`
        @keyframes summon-zone-glow {
          0%   { opacity: 0; transform: scale(0.8); }
          30%  { opacity: 1; transform: scale(1.1); }
          100% { opacity: 0; transform: scale(1.3); }
        }
        @keyframes summon-ring {
          from { transform: scale(0.2); opacity: 1; }
          to   { transform: scale(6);   opacity: 0; }
        }
        @keyframes summon-core {
          0%   { transform: scale(0.3); opacity: 1; }
          40%  { transform: scale(1.2); opacity: 0.9; }
          100% { transform: scale(1.8); opacity: 0; }
        }
        @keyframes summon-pillar {
          0%   { opacity: 0; transform: scaleY(0); transform-origin: bottom; }
          20%  { opacity: 1; transform: scaleY(1); transform-origin: bottom; }
          100% { opacity: 0; transform: scaleY(1); transform-origin: bottom; }
        }
        @keyframes summon-burst-particle {
          0%   { transform: translate(0,0) scale(1); opacity: 1; }
          100% { transform: translate(var(--tx,0px), var(--ty,0px)) scale(0); opacity: 0; }
        }
        @keyframes summon-rise-particle {
          0%   { transform: translate(0,0) scale(1); opacity: 1; }
          100% { transform: translate(var(--drift,0px), var(--rise,-60px)) scale(0); opacity: 0; }
        }
      `}</style>
    </div>
  );
}
