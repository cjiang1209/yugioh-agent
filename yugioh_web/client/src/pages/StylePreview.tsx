// ─── Style Preview Page ───────────────────────────────────────────────────────
// Shows three visual style options for the duel board so the user can compare.

const SAMPLE_CARDS = [
  { id: 46986414, name: "Dark Magician" },
  { id: 89631139, name: "Blue-Eyes White Dragon" },
  { id: 77585513, name: "Kuriboh" },
];

function CardBack({ style }: { style: React.CSSProperties }) {
  return (
    <div
      style={{
        width: 40,
        height: 56,
        borderRadius: 4,
        background: "linear-gradient(135deg, #1a1a3e 0%, #0d0d1a 100%)",
        border: "1px solid rgba(255,255,255,0.15)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontSize: 10,
        ...style,
      }}
    >
      🃏
    </div>
  );
}

function CardFace({ id, name, style }: { id: number; name: string; style?: React.CSSProperties }) {
  return (
    <div style={{ width: 40, height: 56, borderRadius: 4, overflow: "hidden", ...style }}>
      <img
        src={`https://images.ygoprodeck.com/images/cards_small/${id}.jpg`}
        alt={name}
        style={{ width: "100%", height: "100%", objectFit: "cover" }}
        onError={(e) => { (e.target as HTMLImageElement).src = "https://images.ygoprodeck.com/images/cards/back_high.jpg"; }}
      />
    </div>
  );
}

// ─── Option A: Frosted Glass HUD ─────────────────────────────────────────────

function OptionA() {
  const bg = "#060810";
  const panelBg = "rgba(255,255,255,0.04)";
  const panelBorder = "rgba(255,255,255,0.12)";
  const textPrimary = "rgba(255,255,255,0.9)";
  const textSecondary = "rgba(255,255,255,0.5)";
  const accent = "#60a5fa"; // soft blue
  const accentGlow = "rgba(96,165,250,0.25)";
  const gyColor = "rgba(248,113,113,0.8)";
  const banColor = "rgba(167,139,250,0.8)";

  const zoneStyle: React.CSSProperties = {
    width: 44,
    height: 60,
    borderRadius: 6,
    background: panelBg,
    border: `1px solid ${panelBorder}`,
    backdropFilter: "blur(8px)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: 9,
    color: textSecondary,
    fontFamily: "'Rajdhani', sans-serif",
    fontWeight: 600,
    letterSpacing: "0.05em",
  };

  return (
    <div style={{ background: bg, borderRadius: 12, overflow: "hidden", border: "1px solid rgba(255,255,255,0.08)", flex: 1, minWidth: 0 }}>
      {/* Header label */}
      <div style={{ padding: "8px 12px", borderBottom: "1px solid rgba(255,255,255,0.06)", background: "rgba(255,255,255,0.02)" }}>
        <div style={{ fontFamily: "'Orbitron', sans-serif", fontSize: 10, color: accent, letterSpacing: "0.15em", textShadow: `0 0 8px ${accentGlow}` }}>
          OPTION A — FROSTED GLASS HUD
        </div>
        <div style={{ fontSize: 9, color: textSecondary, marginTop: 2, fontFamily: "sans-serif" }}>
          Soft white text · frosted panels · cool blue accent
        </div>
      </div>

      {/* Board */}
      <div style={{ padding: 10, display: "flex", flexDirection: "column", gap: 6 }}>

        {/* Opponent bar */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "4px 6px", background: "rgba(255,255,255,0.03)", borderRadius: 6, border: "1px solid rgba(255,255,255,0.07)" }}>
          <div>
            <div style={{ fontFamily: "'Orbitron', sans-serif", fontSize: 9, color: textPrimary, letterSpacing: "0.1em" }}>KAIBA</div>
            <div style={{ fontFamily: "'Share Tech Mono', monospace", fontSize: 11, color: accent, textShadow: `0 0 6px ${accentGlow}` }}>8,000 LP</div>
          </div>
          <div style={{ display: "flex", gap: 4 }}>
            <div style={{ textAlign: "center" }}>
              <div style={{ width: 28, height: 38, borderRadius: 3, border: `1px solid ${gyColor}`, background: "rgba(248,113,113,0.06)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10 }}>⚰</div>
              <div style={{ fontSize: 8, color: gyColor, fontFamily: "monospace", marginTop: 1 }}>0</div>
            </div>
            <div style={{ textAlign: "center" }}>
              <div style={{ width: 28, height: 38, borderRadius: 3, border: `1px solid ${banColor}`, background: "rgba(167,139,250,0.06)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10 }}>✦</div>
              <div style={{ fontSize: 8, color: banColor, fontFamily: "monospace", marginTop: 1 }}>0</div>
            </div>
            <CardBack style={{ border: "1px solid rgba(255,255,255,0.15)" }} />
          </div>
        </div>

        {/* Opponent hand */}
        <div style={{ display: "flex", gap: 3, justifyContent: "center" }}>
          {[0,1,2,3].map(i => <CardBack key={i} style={{ width: 28, height: 40, opacity: 0.6 }} />)}
        </div>

        {/* Opponent S/T row */}
        <div style={{ display: "flex", gap: 3, justifyContent: "center" }}>
          {[0,1,2,3,4].map(i => <div key={i} style={{ ...zoneStyle, width: 44, height: 32, fontSize: 7 }}>S/T</div>)}
        </div>

        {/* Opponent monster row */}
        <div style={{ display: "flex", gap: 3, justifyContent: "center" }}>
          {[0,1,2,3,4].map(i => <div key={i} style={zoneStyle}>MON</div>)}
        </div>

        {/* Phase bar */}
        <div style={{ display: "flex", gap: 2, justifyContent: "center", padding: "4px 0", borderTop: "1px solid rgba(255,255,255,0.06)", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
          {["DRAW","STANDBY","MAIN 1","BATTLE","MAIN 2","END"].map((p, i) => (
            <div key={p} style={{
              padding: "2px 6px", borderRadius: 3, fontSize: 7,
              fontFamily: "'Orbitron', sans-serif",
              background: i === 0 ? "rgba(96,165,250,0.15)" : "transparent",
              border: `1px solid ${i === 0 ? accent : "rgba(255,255,255,0.08)"}`,
              color: i === 0 ? accent : "rgba(255,255,255,0.35)",
              boxShadow: i === 0 ? `0 0 6px ${accentGlow}` : "none",
            }}>{p}</div>
          ))}
        </div>

        {/* My monster row */}
        <div style={{ display: "flex", gap: 3, justifyContent: "center" }}>
          <CardFace id={46986414} name="Dark Magician" style={{ border: `1px solid ${accent}`, boxShadow: `0 0 8px ${accentGlow}` }} />
          {[1,2,3,4].map(i => <div key={i} style={zoneStyle}>MON</div>)}
        </div>

        {/* My S/T row */}
        <div style={{ display: "flex", gap: 3, justifyContent: "center" }}>
          {[0,1,2,3,4].map(i => <div key={i} style={{ ...zoneStyle, width: 44, height: 32, fontSize: 7 }}>S/T</div>)}
        </div>

        {/* My hand */}
        <div style={{ display: "flex", gap: 3, justifyContent: "center" }}>
          {SAMPLE_CARDS.map(c => <CardFace key={c.id} id={c.id} name={c.name} style={{ border: "1px solid rgba(255,255,255,0.2)" }} />)}
          <CardBack style={{ border: "1px solid rgba(255,255,255,0.15)" }} />
          <CardBack style={{ border: "1px solid rgba(255,255,255,0.15)" }} />
        </div>

        {/* My bar */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "4px 6px", background: "rgba(255,255,255,0.03)", borderRadius: 6, border: "1px solid rgba(255,255,255,0.07)" }}>
          <div>
            <div style={{ fontFamily: "'Orbitron', sans-serif", fontSize: 9, color: textPrimary, letterSpacing: "0.1em" }}>YUGI</div>
            <div style={{ fontFamily: "'Share Tech Mono', monospace", fontSize: 11, color: accent, textShadow: `0 0 6px ${accentGlow}` }}>8,000 LP</div>
          </div>
          <div style={{ display: "flex", gap: 4 }}>
            <div style={{ textAlign: "center" }}>
              <div style={{ width: 28, height: 38, borderRadius: 3, border: `1px solid ${gyColor}`, background: "rgba(248,113,113,0.06)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10 }}>⚰</div>
              <div style={{ fontSize: 8, color: gyColor, fontFamily: "monospace", marginTop: 1 }}>0</div>
            </div>
            <div style={{ textAlign: "center" }}>
              <div style={{ width: 28, height: 38, borderRadius: 3, border: `1px solid ${banColor}`, background: "rgba(167,139,250,0.06)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10 }}>✦</div>
              <div style={{ fontSize: 8, color: banColor, fontFamily: "monospace", marginTop: 1 }}>0</div>
            </div>
            <CardBack style={{ border: "1px solid rgba(255,255,255,0.15)" }} />
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Option B: High-Contrast Neon ────────────────────────────────────────────

function OptionB() {
  const bg = "#050508";
  const panelBg = "rgba(0,245,255,0.04)";
  const panelBorder = "rgba(0,245,255,0.25)";
  const textPrimary = "#e8f4ff";
  const accent = "#00f5ff";
  const accentGlow = "rgba(0,245,255,0.4)";
  const pink = "#ff2d78";
  const pinkGlow = "rgba(255,45,120,0.4)";
  const purple = "#b44fff";

  const zoneStyle: React.CSSProperties = {
    width: 44,
    height: 60,
    borderRadius: 4,
    background: panelBg,
    border: `1px solid ${panelBorder}`,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: 8,
    color: accent,
    fontFamily: "'Orbitron', sans-serif",
    letterSpacing: "0.05em",
    textShadow: `0 0 4px ${accentGlow}`,
  };

  return (
    <div style={{ background: bg, borderRadius: 12, overflow: "hidden", border: `1px solid ${panelBorder}`, flex: 1, minWidth: 0, boxShadow: `0 0 20px rgba(0,245,255,0.08)` }}>
      {/* Header label */}
      <div style={{ padding: "8px 12px", borderBottom: `1px solid rgba(0,245,255,0.1)`, background: "rgba(0,245,255,0.02)" }}>
        <div style={{ fontFamily: "'Orbitron', sans-serif", fontSize: 10, color: accent, letterSpacing: "0.15em", textShadow: `0 0 10px ${accentGlow}` }}>
          OPTION B — HIGH-CONTRAST NEON
        </div>
        <div style={{ fontSize: 9, color: "rgba(0,245,255,0.5)", marginTop: 2, fontFamily: "monospace" }}>
          Full-brightness neon · glowing labels · cyberpunk
        </div>
      </div>

      {/* Board */}
      <div style={{ padding: 10, display: "flex", flexDirection: "column", gap: 6 }}>

        {/* Opponent bar */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "4px 6px", background: "rgba(255,45,120,0.04)", borderRadius: 4, border: `1px solid rgba(255,45,120,0.2)` }}>
          <div>
            <div style={{ fontFamily: "'Orbitron', sans-serif", fontSize: 9, color: pink, letterSpacing: "0.1em", textShadow: `0 0 6px ${pinkGlow}` }}>KAIBA</div>
            <div style={{ fontFamily: "'Share Tech Mono', monospace", fontSize: 12, color: textPrimary, fontWeight: "bold" }}>8,000 LP</div>
          </div>
          <div style={{ display: "flex", gap: 4 }}>
            <div style={{ textAlign: "center" }}>
              <div style={{ width: 28, height: 38, borderRadius: 3, border: `1px solid ${pink}`, background: "rgba(255,45,120,0.08)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, boxShadow: `0 0 6px rgba(255,45,120,0.3)` }}>⚰</div>
              <div style={{ fontSize: 8, color: pink, fontFamily: "monospace", marginTop: 1, textShadow: `0 0 4px ${pinkGlow}` }}>0</div>
            </div>
            <div style={{ textAlign: "center" }}>
              <div style={{ width: 28, height: 38, borderRadius: 3, border: `1px solid ${purple}`, background: "rgba(180,79,255,0.08)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, boxShadow: `0 0 6px rgba(180,79,255,0.3)` }}>✦</div>
              <div style={{ fontSize: 8, color: purple, fontFamily: "monospace", marginTop: 1 }}>0</div>
            </div>
            <CardBack style={{ border: `1px solid ${panelBorder}`, boxShadow: `0 0 4px rgba(0,245,255,0.15)` }} />
          </div>
        </div>

        {/* Opponent hand */}
        <div style={{ display: "flex", gap: 3, justifyContent: "center" }}>
          {[0,1,2,3].map(i => <CardBack key={i} style={{ width: 28, height: 40, opacity: 0.5, border: `1px solid rgba(255,45,120,0.2)` }} />)}
        </div>

        {/* Opponent S/T row */}
        <div style={{ display: "flex", gap: 3, justifyContent: "center" }}>
          {[0,1,2,3,4].map(i => <div key={i} style={{ ...zoneStyle, width: 44, height: 32, fontSize: 7, color: "rgba(0,245,255,0.6)" }}>S/T</div>)}
        </div>

        {/* Opponent monster row */}
        <div style={{ display: "flex", gap: 3, justifyContent: "center" }}>
          {[0,1,2,3,4].map(i => <div key={i} style={zoneStyle}>MON</div>)}
        </div>

        {/* Phase bar */}
        <div style={{ display: "flex", gap: 2, justifyContent: "center", padding: "4px 0", borderTop: `1px solid rgba(0,245,255,0.08)`, borderBottom: `1px solid rgba(0,245,255,0.08)` }}>
          {["DRAW","STANDBY","MAIN 1","BATTLE","MAIN 2","END"].map((p, i) => (
            <div key={p} style={{
              padding: "2px 5px", borderRadius: 2, fontSize: 7,
              fontFamily: "'Orbitron', sans-serif",
              background: i === 0 ? "rgba(0,245,255,0.12)" : "transparent",
              border: `1px solid ${i === 0 ? accent : "rgba(0,245,255,0.15)"}`,
              color: i === 0 ? accent : "rgba(0,245,255,0.5)",
              boxShadow: i === 0 ? `0 0 8px ${accentGlow}` : "none",
              textShadow: i === 0 ? `0 0 6px ${accentGlow}` : "none",
            }}>{p}</div>
          ))}
        </div>

        {/* My monster row */}
        <div style={{ display: "flex", gap: 3, justifyContent: "center" }}>
          <CardFace id={46986414} name="Dark Magician" style={{ border: `1px solid ${accent}`, boxShadow: `0 0 10px ${accentGlow}` }} />
          {[1,2,3,4].map(i => <div key={i} style={zoneStyle}>MON</div>)}
        </div>

        {/* My S/T row */}
        <div style={{ display: "flex", gap: 3, justifyContent: "center" }}>
          {[0,1,2,3,4].map(i => <div key={i} style={{ ...zoneStyle, width: 44, height: 32, fontSize: 7, color: "rgba(0,245,255,0.6)" }}>S/T</div>)}
        </div>

        {/* My hand */}
        <div style={{ display: "flex", gap: 3, justifyContent: "center" }}>
          {SAMPLE_CARDS.map(c => <CardFace key={c.id} id={c.id} name={c.name} style={{ border: `1px solid ${accent}`, boxShadow: `0 0 6px rgba(0,245,255,0.2)` }} />)}
          <CardBack style={{ border: `1px solid ${panelBorder}` }} />
          <CardBack style={{ border: `1px solid ${panelBorder}` }} />
        </div>

        {/* My bar */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "4px 6px", background: "rgba(0,245,255,0.03)", borderRadius: 4, border: `1px solid rgba(0,245,255,0.15)` }}>
          <div>
            <div style={{ fontFamily: "'Orbitron', sans-serif", fontSize: 9, color: accent, letterSpacing: "0.1em", textShadow: `0 0 6px ${accentGlow}` }}>YUGI</div>
            <div style={{ fontFamily: "'Share Tech Mono', monospace", fontSize: 12, color: textPrimary, fontWeight: "bold" }}>8,000 LP</div>
          </div>
          <div style={{ display: "flex", gap: 4 }}>
            <div style={{ textAlign: "center" }}>
              <div style={{ width: 28, height: 38, borderRadius: 3, border: `1px solid ${pink}`, background: "rgba(255,45,120,0.08)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, boxShadow: `0 0 6px rgba(255,45,120,0.3)` }}>⚰</div>
              <div style={{ fontSize: 8, color: pink, fontFamily: "monospace", marginTop: 1, textShadow: `0 0 4px ${pinkGlow}` }}>0</div>
            </div>
            <div style={{ textAlign: "center" }}>
              <div style={{ width: 28, height: 38, borderRadius: 3, border: `1px solid ${purple}`, background: "rgba(180,79,255,0.08)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, boxShadow: `0 0 6px rgba(180,79,255,0.3)` }}>✦</div>
              <div style={{ fontSize: 8, color: purple, fontFamily: "monospace", marginTop: 1 }}>0</div>
            </div>
            <CardBack style={{ border: `1px solid ${panelBorder}` }} />
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Option C: Warm Amber & Slate ────────────────────────────────────────────

function OptionC() {
  const bg = "#0e0c0a";
  const panelBg = "rgba(255,179,71,0.04)";
  const panelBorder = "rgba(255,179,71,0.2)";
  const textPrimary = "#f0e6d3";
  const textSecondary = "rgba(240,230,211,0.5)";
  const amber = "#ffb347";
  const amberGlow = "rgba(255,179,71,0.35)";
  const rose = "#f87171";
  const roseGlow = "rgba(248,113,113,0.3)";
  const violet = "#c084fc";

  const zoneStyle: React.CSSProperties = {
    width: 44,
    height: 60,
    borderRadius: 4,
    background: panelBg,
    border: `1px solid ${panelBorder}`,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: 8,
    color: textSecondary,
    fontFamily: "'Rajdhani', sans-serif",
    fontWeight: 700,
    letterSpacing: "0.06em",
  };

  return (
    <div style={{ background: bg, borderRadius: 12, overflow: "hidden", border: `1px solid rgba(255,179,71,0.15)`, flex: 1, minWidth: 0, boxShadow: `0 0 20px rgba(255,179,71,0.05)` }}>
      {/* Header label */}
      <div style={{ padding: "8px 12px", borderBottom: `1px solid rgba(255,179,71,0.1)`, background: "rgba(255,179,71,0.02)" }}>
        <div style={{ fontFamily: "'Orbitron', sans-serif", fontSize: 10, color: amber, letterSpacing: "0.15em", textShadow: `0 0 8px ${amberGlow}` }}>
          OPTION C — WARM AMBER & SLATE
        </div>
        <div style={{ fontSize: 9, color: textSecondary, marginTop: 2, fontFamily: "sans-serif" }}>
          Warm off-white text · amber accent · ancient magic feel
        </div>
      </div>

      {/* Board */}
      <div style={{ padding: 10, display: "flex", flexDirection: "column", gap: 6 }}>

        {/* Opponent bar */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "4px 6px", background: "rgba(248,113,113,0.04)", borderRadius: 4, border: `1px solid rgba(248,113,113,0.15)` }}>
          <div>
            <div style={{ fontFamily: "'Orbitron', sans-serif", fontSize: 9, color: rose, letterSpacing: "0.1em", textShadow: `0 0 6px ${roseGlow}` }}>KAIBA</div>
            <div style={{ fontFamily: "'Share Tech Mono', monospace", fontSize: 12, color: textPrimary }}>8,000 LP</div>
          </div>
          <div style={{ display: "flex", gap: 4 }}>
            <div style={{ textAlign: "center" }}>
              <div style={{ width: 28, height: 38, borderRadius: 3, border: `1px solid ${rose}`, background: "rgba(248,113,113,0.06)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10 }}>⚰</div>
              <div style={{ fontSize: 8, color: rose, fontFamily: "monospace", marginTop: 1 }}>0</div>
            </div>
            <div style={{ textAlign: "center" }}>
              <div style={{ width: 28, height: 38, borderRadius: 3, border: `1px solid ${violet}`, background: "rgba(192,132,252,0.06)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10 }}>✦</div>
              <div style={{ fontSize: 8, color: violet, fontFamily: "monospace", marginTop: 1 }}>0</div>
            </div>
            <CardBack style={{ border: `1px solid rgba(255,179,71,0.2)` }} />
          </div>
        </div>

        {/* Opponent hand */}
        <div style={{ display: "flex", gap: 3, justifyContent: "center" }}>
          {[0,1,2,3].map(i => <CardBack key={i} style={{ width: 28, height: 40, opacity: 0.5, border: `1px solid rgba(248,113,113,0.2)` }} />)}
        </div>

        {/* Opponent S/T row */}
        <div style={{ display: "flex", gap: 3, justifyContent: "center" }}>
          {[0,1,2,3,4].map(i => <div key={i} style={{ ...zoneStyle, width: 44, height: 32, fontSize: 7 }}>S/T</div>)}
        </div>

        {/* Opponent monster row */}
        <div style={{ display: "flex", gap: 3, justifyContent: "center" }}>
          {[0,1,2,3,4].map(i => <div key={i} style={zoneStyle}>MON</div>)}
        </div>

        {/* Phase bar */}
        <div style={{ display: "flex", gap: 2, justifyContent: "center", padding: "4px 0", borderTop: `1px solid rgba(255,179,71,0.08)`, borderBottom: `1px solid rgba(255,179,71,0.08)` }}>
          {["DRAW","STANDBY","MAIN 1","BATTLE","MAIN 2","END"].map((p, i) => (
            <div key={p} style={{
              padding: "2px 5px", borderRadius: 3, fontSize: 7,
              fontFamily: "'Orbitron', sans-serif",
              background: i === 0 ? "rgba(255,179,71,0.12)" : "transparent",
              border: `1px solid ${i === 0 ? amber : "rgba(255,179,71,0.12)"}`,
              color: i === 0 ? amber : "rgba(240,230,211,0.35)",
              boxShadow: i === 0 ? `0 0 6px ${amberGlow}` : "none",
              textShadow: i === 0 ? `0 0 5px ${amberGlow}` : "none",
            }}>{p}</div>
          ))}
        </div>

        {/* My monster row */}
        <div style={{ display: "flex", gap: 3, justifyContent: "center" }}>
          <CardFace id={46986414} name="Dark Magician" style={{ border: `1px solid ${amber}`, boxShadow: `0 0 8px ${amberGlow}` }} />
          {[1,2,3,4].map(i => <div key={i} style={zoneStyle}>MON</div>)}
        </div>

        {/* My S/T row */}
        <div style={{ display: "flex", gap: 3, justifyContent: "center" }}>
          {[0,1,2,3,4].map(i => <div key={i} style={{ ...zoneStyle, width: 44, height: 32, fontSize: 7 }}>S/T</div>)}
        </div>

        {/* My hand */}
        <div style={{ display: "flex", gap: 3, justifyContent: "center" }}>
          {SAMPLE_CARDS.map(c => <CardFace key={c.id} id={c.id} name={c.name} style={{ border: `1px solid rgba(255,179,71,0.4)`, boxShadow: `0 0 5px rgba(255,179,71,0.15)` }} />)}
          <CardBack style={{ border: `1px solid rgba(255,179,71,0.2)` }} />
          <CardBack style={{ border: `1px solid rgba(255,179,71,0.2)` }} />
        </div>

        {/* My bar */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "4px 6px", background: "rgba(255,179,71,0.03)", borderRadius: 4, border: `1px solid rgba(255,179,71,0.15)` }}>
          <div>
            <div style={{ fontFamily: "'Orbitron', sans-serif", fontSize: 9, color: amber, letterSpacing: "0.1em", textShadow: `0 0 6px ${amberGlow}` }}>YUGI</div>
            <div style={{ fontFamily: "'Share Tech Mono', monospace", fontSize: 12, color: textPrimary }}>8,000 LP</div>
          </div>
          <div style={{ display: "flex", gap: 4 }}>
            <div style={{ textAlign: "center" }}>
              <div style={{ width: 28, height: 38, borderRadius: 3, border: `1px solid ${rose}`, background: "rgba(248,113,113,0.06)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10 }}>⚰</div>
              <div style={{ fontSize: 8, color: rose, fontFamily: "monospace", marginTop: 1 }}>0</div>
            </div>
            <div style={{ textAlign: "center" }}>
              <div style={{ width: 28, height: 38, borderRadius: 3, border: `1px solid ${violet}`, background: "rgba(192,132,252,0.06)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10 }}>✦</div>
              <div style={{ fontSize: 8, color: violet, fontFamily: "monospace", marginTop: 1 }}>0</div>
            </div>
            <CardBack style={{ border: `1px solid rgba(255,179,71,0.2)` }} />
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function StylePreview() {
  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#020204",
        padding: "24px 16px",
        overflowY: "auto",
        fontFamily: "'Rajdhani', sans-serif",
      }}
    >
      {/* Title */}
      <div style={{ textAlign: "center", marginBottom: 24 }}>
        <div style={{ fontFamily: "'Orbitron', sans-serif", fontSize: 18, color: "#00f5ff", letterSpacing: "0.2em", textShadow: "0 0 20px rgba(0,245,255,0.5)" }}>
          STYLE OPTIONS
        </div>
        <div style={{ fontSize: 12, color: "rgba(255,255,255,0.4)", marginTop: 4, fontFamily: "sans-serif" }}>
          Compare the three visual styles below, then tell me which one you prefer (or mix elements from multiple)
        </div>
      </div>

      {/* Three columns */}
      <div style={{ display: "flex", gap: 12, alignItems: "flex-start", maxWidth: 1200, margin: "0 auto" }}>
        <OptionA />
        <OptionB />
        <OptionC />
      </div>

      {/* Legend */}
      <div style={{ textAlign: "center", marginTop: 20, fontSize: 11, color: "rgba(255,255,255,0.3)", fontFamily: "sans-serif" }}>
        Card images are loaded live from YGOPRODeck API · Zone labels and LP bars reflect the actual board layout
      </div>
    </div>
  );
}
