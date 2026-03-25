import { useEffect, useRef, useState } from "react";

export interface AttackAnimationProps {
  /** Screen-space rect of the attacker zone */
  from: DOMRect;
  /** Screen-space rect of the target zone, or null for direct attack */
  to: DOMRect | null;
  /** Called when the animation finishes */
  onDone: () => void;
}

function centerOf(rect: DOMRect) {
  return { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 };
}

// Generate random particles for the burst
function makeParticles(count: number) {
  return Array.from({ length: count }, (_, i) => ({
    id: i,
    angle: (360 / count) * i + Math.random() * (360 / count),
    dist: 60 + Math.random() * 80,
    size: 3 + Math.random() * 5,
    delay: Math.random() * 60,
    color: ["#ff2d78", "#ffcc00", "#ffffff", "#00e5ff", "#ff6600"][
      Math.floor(Math.random() * 5)
    ],
  }));
}

const PARTICLES = makeParticles(24);

/**
 * Cinematic full-viewport attack animation:
 *  Phase 1 (0–350ms)  — multi-layer beam travels from attacker to target
 *  Phase 2 (350–700ms) — screen flash + shockwave rings + particle burst
 *  Phase 3 (700–900ms) — fade out
 */
export function AttackAnimation({ from, to, onDone }: AttackAnimationProps) {
  const [phase, setPhase] = useState<"beam" | "impact" | "done">("beam");
  const doneRef = useRef(false);

  const src = centerOf(from);
  const dst = to
    ? centerOf(to)
    : { x: window.innerWidth / 2, y: window.innerHeight / 2 };

  const dx = dst.x - src.x;
  const dy = dst.y - src.y;
  const len = Math.sqrt(dx * dx + dy * dy);
  const angle = Math.atan2(dy, dx) * (180 / Math.PI);

  useEffect(() => {
    const t1 = setTimeout(() => setPhase("impact"), 350);
    const t2 = setTimeout(() => {
      setPhase("done");
      if (!doneRef.current) {
        doneRef.current = true;
        onDone();
      }
    }, 900);
    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
    };
  }, [onDone]);

  if (phase === "done") return null;

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 9999,
        pointerEvents: "none",
        overflow: "hidden",
      }}
    >
      {/* ══════════════════════════════════════════
          BEAM PHASE
      ══════════════════════════════════════════ */}
      {phase === "beam" && (
        <>
          {/* Outer glow halo */}
          <div
            style={{
              position: "absolute",
              left: src.x,
              top: src.y,
              width: len,
              height: 24,
              transformOrigin: "0 50%",
              transform: `rotate(${angle}deg) translateY(-50%)`,
              background:
                "linear-gradient(90deg, transparent 0%, rgba(255,45,120,0.25) 20%, rgba(255,100,0,0.35) 70%, transparent 100%)",
              filter: "blur(8px)",
              borderRadius: 12,
              animation: "atk-beam 0.35s ease-out forwards",
            }}
          />
          {/* Mid beam */}
          <div
            style={{
              position: "absolute",
              left: src.x,
              top: src.y,
              width: len,
              height: 10,
              transformOrigin: "0 50%",
              transform: `rotate(${angle}deg) translateY(-50%)`,
              background:
                "linear-gradient(90deg, rgba(255,45,120,0) 0%, rgba(255,45,120,1) 25%, rgba(255,180,0,1) 65%, rgba(255,255,255,1) 100%)",
              boxShadow:
                "0 0 14px 5px rgba(255,45,120,0.9), 0 0 30px 10px rgba(255,100,0,0.5)",
              borderRadius: 5,
              animation: "atk-beam 0.35s ease-out forwards",
            }}
          />
          {/* Core bright line */}
          <div
            style={{
              position: "absolute",
              left: src.x,
              top: src.y,
              width: len,
              height: 3,
              transformOrigin: "0 50%",
              transform: `rotate(${angle}deg) translateY(-50%)`,
              background:
                "linear-gradient(90deg, rgba(255,200,200,0) 0%, rgba(255,255,255,1) 30%, rgba(255,255,255,1) 100%)",
              borderRadius: 2,
              animation: "atk-beam 0.35s ease-out forwards",
            }}
          />
          {/* Slash lines fanning from origin */}
          {[-28, -14, 0, 14, 28].map((offset, i) => (
            <div
              key={i}
              style={{
                position: "absolute",
                left: src.x,
                top: src.y,
                width: 50 + i * 15,
                height: 2,
                transformOrigin: "0 50%",
                transform: `rotate(${angle + offset}deg) translateY(-50%)`,
                background:
                  "linear-gradient(90deg, rgba(255,200,50,0.95), transparent)",
                boxShadow: "0 0 6px 2px rgba(255,200,50,0.5)",
                borderRadius: 2,
                opacity: 0.85,
                animation: `atk-slash 0.35s ease-out forwards`,
                animationDelay: `${i * 20}ms`,
              }}
            />
          ))}
          {/* Leading energy ball */}
          <div
            style={{
              position: "absolute",
              left: src.x,
              top: src.y,
              width: 18,
              height: 18,
              marginLeft: -9,
              marginTop: -9,
              borderRadius: "50%",
              background:
                "radial-gradient(circle, #fff 0%, #ffcc00 40%, #ff2d78 80%, transparent 100%)",
              boxShadow: "0 0 20px 8px rgba(255,45,120,0.8)",
              animation: `atk-ball 0.35s ease-in forwards`,
              ["--tx" as string]: `${dx}px`,
              ["--ty" as string]: `${dy}px`,
            } as React.CSSProperties}
          />
        </>
      )}

      {/* ══════════════════════════════════════════
          IMPACT PHASE
      ══════════════════════════════════════════ */}
      {phase === "impact" && (
        <>
          {/* Full-screen white flash */}
          <div
            style={{
              position: "fixed",
              inset: 0,
              background: "rgba(255,255,255,0.55)",
              animation: "atk-screen-flash 0.55s ease-out forwards",
            }}
          />

          {/* Shockwave ring 1 */}
          <div
            style={{
              position: "absolute",
              left: dst.x,
              top: dst.y,
              width: 20,
              height: 20,
              marginLeft: -10,
              marginTop: -10,
              borderRadius: "50%",
              border: "4px solid rgba(255,200,50,1)",
              boxShadow: "0 0 16px 6px rgba(255,45,120,0.8)",
              animation: "atk-ring 0.55s cubic-bezier(0.1,0.6,0.4,1) forwards",
            }}
          />
          {/* Shockwave ring 2 (delayed) */}
          <div
            style={{
              position: "absolute",
              left: dst.x,
              top: dst.y,
              width: 20,
              height: 20,
              marginLeft: -10,
              marginTop: -10,
              borderRadius: "50%",
              border: "3px solid rgba(0,229,255,0.85)",
              boxShadow: "0 0 12px 4px rgba(0,229,255,0.6)",
              animation:
                "atk-ring 0.55s cubic-bezier(0.1,0.6,0.4,1) 80ms forwards",
            }}
          />
          {/* Shockwave ring 3 (more delayed) */}
          <div
            style={{
              position: "absolute",
              left: dst.x,
              top: dst.y,
              width: 20,
              height: 20,
              marginLeft: -10,
              marginTop: -10,
              borderRadius: "50%",
              border: "2px solid rgba(255,255,255,0.7)",
              animation:
                "atk-ring 0.55s cubic-bezier(0.1,0.6,0.4,1) 160ms forwards",
            }}
          />

          {/* Bright core burst */}
          <div
            style={{
              position: "absolute",
              left: dst.x,
              top: dst.y,
              width: 80,
              height: 80,
              marginLeft: -40,
              marginTop: -40,
              borderRadius: "50%",
              background:
                "radial-gradient(circle, rgba(255,255,255,1) 0%, rgba(255,200,50,0.9) 35%, rgba(255,45,120,0.5) 65%, transparent 100%)",
              animation: "atk-core 0.55s ease-out forwards",
            }}
          />

          {/* Cross slash marks */}
          {[0, 45, 90, 135].map((deg, i) => (
            <div
              key={i}
              style={{
                position: "absolute",
                left: dst.x,
                top: dst.y,
                width: 100,
                height: 3,
                marginTop: -1.5,
                transformOrigin: "50% 50%",
                transform: `translateX(-50%) rotate(${deg}deg)`,
                background:
                  "linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.95) 30%, rgba(255,200,50,0.9) 50%, rgba(255,255,255,0.95) 70%, transparent 100%)",
                boxShadow: "0 0 6px 2px rgba(255,200,50,0.7)",
                borderRadius: 2,
                animation: `atk-slash-impact 0.55s ease-out forwards`,
                animationDelay: `${i * 30}ms`,
              }}
            />
          ))}

          {/* Particle burst */}
          {PARTICLES.map((p) => {
            const rad = (p.angle * Math.PI) / 180;
            const tx = Math.cos(rad) * p.dist;
            const ty = Math.sin(rad) * p.dist;
            return (
              <div
                key={p.id}
                style={{
                  position: "absolute",
                  left: dst.x,
                  top: dst.y,
                  width: p.size,
                  height: p.size,
                  marginLeft: -p.size / 2,
                  marginTop: -p.size / 2,
                  borderRadius: "50%",
                  background: p.color,
                  boxShadow: `0 0 ${p.size * 2}px ${p.size}px ${p.color}88`,
                  animation: `atk-particle 0.55s ease-out forwards`,
                  animationDelay: `${p.delay}ms`,
                  ["--tx" as string]: `${tx}px`,
                  ["--ty" as string]: `${ty}px`,
                }}
              />
            );
          })}
        </>
      )}

      <style>{`
        @keyframes atk-beam {
          from { clip-path: inset(0 100% 0 0); opacity: 1; }
          to   { clip-path: inset(0 0% 0 0);   opacity: 1; }
        }
        @keyframes atk-slash {
          from { opacity: 0.9; transform: rotate(var(--r,0deg)) translateY(-50%) scaleX(0); }
          to   { opacity: 0;   transform: rotate(var(--r,0deg)) translateY(-50%) scaleX(1); }
        }
        @keyframes atk-ball {
          from { transform: translate(0, 0) scale(1); opacity: 1; }
          to   { transform: translate(var(--tx,0px), var(--ty,0px)) scale(0.5); opacity: 0; }
        }
        @keyframes atk-screen-flash {
          0%   { opacity: 0.55; }
          20%  { opacity: 0.75; }
          100% { opacity: 0; }
        }
        @keyframes atk-ring {
          from { transform: scale(0.15); opacity: 1; }
          to   { transform: scale(8);    opacity: 0; }
        }
        @keyframes atk-core {
          0%   { transform: scale(0.3); opacity: 1; }
          40%  { transform: scale(1.4); opacity: 0.9; }
          100% { transform: scale(2);   opacity: 0; }
        }
        @keyframes atk-slash-impact {
          from { opacity: 1; transform: translateX(-50%) rotate(var(--deg,0deg)) scaleX(0.1); }
          to   { opacity: 0; transform: translateX(-50%) rotate(var(--deg,0deg)) scaleX(1.4); }
        }
        @keyframes atk-particle {
          0%   { transform: translate(0, 0) scale(1); opacity: 1; }
          100% { transform: translate(var(--tx,0px), var(--ty,0px)) scale(0); opacity: 0; }
        }
      `}</style>
    </div>
  );
}
