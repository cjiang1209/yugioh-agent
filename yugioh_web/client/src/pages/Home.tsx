import { Link } from "wouter";

export default function Home() {
  return (
    <div
      className="min-h-screen flex flex-col items-center justify-center relative overflow-hidden"
      style={{ background: "var(--bg-void)" }}
    >
      {/* Scanlines overlay */}
      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          background:
            "repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.04) 2px, rgba(0,0,0,0.04) 4px)",
          zIndex: 1,
        }}
      />

      {/* Grid background */}
      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          backgroundImage: `
            linear-gradient(rgba(0,245,255,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,245,255,0.03) 1px, transparent 1px)
          `,
          backgroundSize: "40px 40px",
          zIndex: 0,
        }}
      />

      {/* Corner decorations */}
      <div
        className="fixed top-4 left-4 pointer-events-none"
        style={{ width: "60px", height: "60px", borderTop: "1px solid var(--neon-cyan)", borderLeft: "1px solid var(--neon-cyan)", opacity: 0.3 }}
      />
      <div
        className="fixed top-4 right-4 pointer-events-none"
        style={{ width: "60px", height: "60px", borderTop: "1px solid var(--neon-cyan)", borderRight: "1px solid var(--neon-cyan)", opacity: 0.3 }}
      />
      <div
        className="fixed bottom-4 left-4 pointer-events-none"
        style={{ width: "60px", height: "60px", borderBottom: "1px solid var(--neon-cyan)", borderLeft: "1px solid var(--neon-cyan)", opacity: 0.3 }}
      />
      <div
        className="fixed bottom-4 right-4 pointer-events-none"
        style={{ width: "60px", height: "60px", borderBottom: "1px solid var(--neon-cyan)", borderRight: "1px solid var(--neon-cyan)", opacity: 0.3 }}
      />

      {/* Main content */}
      <div className="relative z-10 text-center px-6">
        {/* Title */}
        <div className="mb-2">
          <div
            className="text-xs tracking-[0.5em] mb-3 opacity-50"
            style={{ fontFamily: "'Share Tech Mono', monospace", color: "var(--neon-cyan)" }}
          >
            ── ONLINE DUEL SIMULATOR ──
          </div>
          <h1
            className="text-7xl font-black mb-1 animate-flicker"
            style={{
              fontFamily: "'Orbitron', sans-serif",
              color: "var(--neon-pink)",
              textShadow:
                "0 0 20px var(--neon-pink), 0 0 60px var(--neon-pink), 0 0 100px rgba(255,45,120,0.3)",
              letterSpacing: "0.15em",
              lineHeight: 1,
            }}
          >
            YU-GI-OH!
          </h1>
          <h2
            className="text-3xl font-bold tracking-[0.4em]"
            style={{
              fontFamily: "'Orbitron', sans-serif",
              color: "var(--neon-cyan)",
              textShadow: "0 0 15px var(--neon-cyan), 0 0 40px var(--neon-cyan)",
            }}
          >
            DUEL ARENA
          </h2>
        </div>

        {/* Divider */}
        <div className="flex items-center justify-center gap-4 my-6">
          <div style={{ height: "1px", width: "80px", background: "linear-gradient(90deg, transparent, var(--neon-cyan))" }} />
          <div
            className="text-xs"
            style={{ color: "var(--neon-cyan)", fontFamily: "'Share Tech Mono', monospace", opacity: 0.7 }}
          >
            ★ IT'S TIME TO DUEL ★
          </div>
          <div style={{ height: "1px", width: "80px", background: "linear-gradient(90deg, var(--neon-cyan), transparent)" }} />
        </div>

        {/* Feature list */}
        <div className="flex flex-wrap justify-center gap-3 mb-8 max-w-lg mx-auto">
          {[
            "Self-Play Mode",
            "Full Turn Phases",
            "Monster Battles",
            "Spell & Trap Cards",
            "Real Card Data",
            "Live Graveyard",
          ].map((feat) => (
            <span
              key={feat}
              className="px-3 py-1 rounded text-xs"
              style={{
                background: "rgba(0,245,255,0.05)",
                border: "1px solid rgba(0,245,255,0.2)",
                color: "var(--neon-cyan)",
                fontFamily: "'Rajdhani', sans-serif",
                fontWeight: 600,
                letterSpacing: "0.05em",
              }}
            >
              {feat}
            </span>
          ))}
        </div>

        {/* CTA */}
        <Link href="/duel">
          <button
            className="px-12 py-4 rounded font-black text-lg transition-all"
            style={{
              fontFamily: "'Orbitron', sans-serif",
              letterSpacing: "0.2em",
              background: "linear-gradient(135deg, rgba(255,45,120,0.15), rgba(0,245,255,0.15))",
              border: "1px solid var(--neon-cyan)",
              color: "var(--neon-cyan)",
              boxShadow: "0 0 30px rgba(0,245,255,0.3), inset 0 0 30px rgba(0,245,255,0.05)",
              fontSize: "0.9rem",
            }}
            onMouseEnter={(e) => {
              (e.currentTarget as HTMLButtonElement).style.boxShadow =
                "0 0 50px rgba(0,245,255,0.5), inset 0 0 30px rgba(0,245,255,0.1)";
              (e.currentTarget as HTMLButtonElement).style.transform = "translateY(-2px)";
            }}
            onMouseLeave={(e) => {
              (e.currentTarget as HTMLButtonElement).style.boxShadow =
                "0 0 30px rgba(0,245,255,0.3), inset 0 0 30px rgba(0,245,255,0.05)";
              (e.currentTarget as HTMLButtonElement).style.transform = "none";
            }}
          >
            START DUEL ▶
          </button>
        </Link>

        {/* Instructions */}
        <div
          className="mt-8 text-xs opacity-40 max-w-sm mx-auto leading-relaxed"
          style={{ color: "var(--text-secondary)", fontFamily: "'Rajdhani', sans-serif" }}
        >
          Click START DUEL to play both sides — use the ⋄ button to switch between Yugi and Kaiba during the duel.
        </div>
      </div>
    </div>
  );
}
