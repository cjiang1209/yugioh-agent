import { describe, expect, it } from "vitest";
import { applyAction, createDuelState, createPlayerState, makeGameCard } from "./gameEngine";
import { DuelState, YgoCard } from "../shared/gameTypes";

// ─── Fixtures ─────────────────────────────────────────────────────────────────

const mockMonster = (id: number, atk: number, def: number, level = 4): YgoCard => ({
  id,
  name: `Monster ${id}`,
  type: "Normal Monster",
  frameType: "normal",
  desc: "Test monster",
  atk,
  def,
  level,
  race: "Warrior",
  attribute: "EARTH",
  card_images: [],
});

const mockSpell = (id: number): YgoCard => ({
  id,
  name: `Spell ${id}`,
  type: "Spell Card",
  frameType: "spell",
  desc: "Test spell",
  race: "Normal",
  card_images: [],
});

const mockTrap = (id: number): YgoCard => ({
  id,
  name: `Trap ${id}`,
  type: "Trap Card",
  frameType: "trap",
  desc: "Test trap",
  race: "Normal",
  card_images: [],
});

function buildDeck(count = 20): YgoCard[] {
  return Array.from({ length: count }, (_, i) => mockMonster(i + 1, 1000 + i * 100, 800 + i * 50));
}

function createTestState(): DuelState {
  return createDuelState(
    "test-room",
    "p1", "Player1", buildDeck(20),
    "p2", "Player2", buildDeck(20)
  );
}

// ─── Tests ────────────────────────────────────────────────────────────────────

describe("Game Engine - Initial State", () => {
  it("creates state with correct initial LP", () => {
    const state = createTestState();
    expect(state.player1.lifePoints).toBe(8000);
    expect(state.player2.lifePoints).toBe(8000);
  });

  it("deals 5 cards to each player at start", () => {
    const state = createTestState();
    expect(state.player1.hand).toHaveLength(5);
    expect(state.player2.hand).toHaveLength(5);
  });

  it("starts at DRAW phase on turn 1", () => {
    const state = createTestState();
    expect(state.phase).toBe("DRAW");
    expect(state.turnNumber).toBe(1);
    expect(state.activePlayer).toBe("player1");
  });

  it("initializes empty zones", () => {
    const state = createTestState();
    expect(state.player1.monsterZones).toHaveLength(5);
    expect(state.player1.monsterZones.every((z) => z === null)).toBe(true);
    expect(state.player1.spellTrapZones.every((z) => z === null)).toBe(true);
  });
});

describe("Game Engine - Phase Transitions", () => {
  it("advances through all phases in order", () => {
    let state = createTestState();
    const phases = ["DRAW", "STANDBY", "MAIN1", "BATTLE", "MAIN2", "END"];
    for (let i = 0; i < phases.length; i++) {
      expect(state.phase).toBe(phases[i]);
      state = applyAction(state, { type: "ADVANCE_PHASE" }, "player1");
    }
    // After END phase, it should be player2's turn at DRAW
    expect(state.activePlayer).toBe("player2");
    expect(state.phase).toBe("DRAW");
  });

  it("switches active player after END phase", () => {
    let state = createTestState();
    // Advance through all 6 phases
    for (let i = 0; i < 6; i++) {
      state = applyAction(state, { type: "ADVANCE_PHASE" }, "player1");
    }
    expect(state.activePlayer).toBe("player2");
  });

  it("resets normal summon flag on new turn", () => {
    let state = createTestState();
    // Advance to MAIN1
    state = applyAction(state, { type: "ADVANCE_PHASE" }, "player1"); // STANDBY
    state = applyAction(state, { type: "ADVANCE_PHASE" }, "player1"); // MAIN1

    // Summon a monster
    const handIndex = 0;
    state = applyAction(state, { type: "SUMMON_MONSTER", handIndex, zoneIndex: 0 }, "player1");
    expect(state.player1.hasNormalSummoned).toBe(true);

    // Advance through rest of turn
    state = applyAction(state, { type: "ADVANCE_PHASE" }, "player1"); // BATTLE
    state = applyAction(state, { type: "ADVANCE_PHASE" }, "player1"); // MAIN2
    state = applyAction(state, { type: "ADVANCE_PHASE" }, "player1"); // END
    state = applyAction(state, { type: "ADVANCE_PHASE" }, "player1"); // next turn DRAW

    expect(state.player2.hasNormalSummoned).toBe(false);
  });
});

describe("Game Engine - Draw Card", () => {
  it("draws a card from deck to hand", () => {
    let state = createTestState();
    const initialHandSize = state.player1.hand.length;
    const initialDeckSize = state.player1.deck.length;
    state = applyAction(state, { type: "DRAW_CARD" }, "player1");
    expect(state.player1.hand).toHaveLength(initialHandSize + 1);
    expect(state.player1.deck).toHaveLength(initialDeckSize - 1);
  });
});

describe("Game Engine - Normal Summon", () => {
  it("summons a level 4 monster to a zone", () => {
    let state = createTestState();
    // Advance to MAIN1
    state = applyAction(state, { type: "ADVANCE_PHASE" }, "player1");
    state = applyAction(state, { type: "ADVANCE_PHASE" }, "player1");

    const handBefore = state.player1.hand.length;
    state = applyAction(state, { type: "SUMMON_MONSTER", handIndex: 0, zoneIndex: 2 }, "player1");

    expect(state.player1.monsterZones[2]).not.toBeNull();
    expect(state.player1.monsterZones[2]?.position).toBe("ATK");
    expect(state.player1.hand).toHaveLength(handBefore - 1);
    expect(state.player1.hasNormalSummoned).toBe(true);
  });

  it("cannot summon twice in one turn", () => {
    let state = createTestState();
    state = applyAction(state, { type: "ADVANCE_PHASE" }, "player1");
    state = applyAction(state, { type: "ADVANCE_PHASE" }, "player1");

    state = applyAction(state, { type: "SUMMON_MONSTER", handIndex: 0, zoneIndex: 0 }, "player1");
    const stateBefore = state;
    state = applyAction(state, { type: "SUMMON_MONSTER", handIndex: 0, zoneIndex: 1 }, "player1");

    // Zone 1 should still be empty
    expect(state.player1.monsterZones[1]).toBeNull();
  });

  it("cannot summon outside of main phase", () => {
    let state = createTestState();
    // Still in DRAW phase
    state = applyAction(state, { type: "SUMMON_MONSTER", handIndex: 0, zoneIndex: 0 }, "player1");
    expect(state.player1.monsterZones[0]).toBeNull();
  });
});

describe("Game Engine - Set Monster", () => {
  it("sets a monster face-down in DEF position", () => {
    let state = createTestState();
    state = applyAction(state, { type: "ADVANCE_PHASE" }, "player1");
    state = applyAction(state, { type: "ADVANCE_PHASE" }, "player1");

    state = applyAction(state, { type: "SET_MONSTER", handIndex: 0, zoneIndex: 0 }, "player1");

    expect(state.player1.monsterZones[0]).not.toBeNull();
    expect(state.player1.monsterZones[0]?.faceDown).toBe(true);
    expect(state.player1.monsterZones[0]?.position).toBe("FACE_DOWN_DEF");
  });
});

describe("Game Engine - Spell/Trap", () => {
  it("activates a normal spell and sends to graveyard", () => {
    // Build a state with a spell in hand
    const spellCard = mockSpell(999);
    let state = createTestState();
    state = {
      ...state,
      player1: {
        ...state.player1,
        hand: [makeGameCard(spellCard), ...state.player1.hand],
      },
    };

    state = applyAction(state, { type: "ADVANCE_PHASE" }, "player1");
    state = applyAction(state, { type: "ADVANCE_PHASE" }, "player1");

    const gyBefore = state.player1.graveyard.length;
    state = applyAction(state, { type: "ACTIVATE_SPELL", handIndex: 0, zoneIndex: 0 }, "player1");

    expect(state.player1.graveyard).toHaveLength(gyBefore + 1);
  });

  it("sets a trap card face-down", () => {
    const trapCard = mockTrap(998);
    let state = createTestState();
    state = {
      ...state,
      player1: {
        ...state.player1,
        hand: [makeGameCard(trapCard), ...state.player1.hand],
      },
    };

    state = applyAction(state, { type: "ADVANCE_PHASE" }, "player1");
    state = applyAction(state, { type: "ADVANCE_PHASE" }, "player1");

    state = applyAction(state, { type: "SET_SPELL_TRAP", handIndex: 0, zoneIndex: 0 }, "player1");

    expect(state.player1.spellTrapZones[0]).not.toBeNull();
    expect(state.player1.spellTrapZones[0]?.faceDown).toBe(true);
  });
});

describe("Game Engine - Battle System", () => {
  function stateWithMonsters(p1Atk: number, p2Atk: number): DuelState {
    let state = createTestState();
    const m1 = makeGameCard(mockMonster(100, p1Atk, 500));
    const m2 = makeGameCard(mockMonster(200, p2Atk, 500));

    state = {
      ...state,
      phase: "BATTLE",
      player1: {
        ...state.player1,
        monsterZones: [{ card: m1, position: "ATK", faceDown: false }, null, null, null, null],
      },
      player2: {
        ...state.player2,
        monsterZones: [{ card: m2, position: "ATK", faceDown: false }, null, null, null, null],
      },
    };
    return state;
  }

  it("destroys weaker monster and deals damage", () => {
    let state = stateWithMonsters(2000, 1000);
    state = applyAction(state, { type: "DECLARE_ATTACK", attackerZone: 0, targetZone: 0, targetSide: "player2" }, "player1");

    expect(state.player2.monsterZones[0]).toBeNull();
    expect(state.player2.lifePoints).toBe(7000); // 8000 - 1000 damage
    expect(state.player1.monsterZones[0]).not.toBeNull();
  });

  it("destroys attacker when it has lower ATK", () => {
    let state = stateWithMonsters(1000, 2000);
    state = applyAction(state, { type: "DECLARE_ATTACK", attackerZone: 0, targetZone: 0, targetSide: "player2" }, "player1");

    expect(state.player1.monsterZones[0]).toBeNull();
    expect(state.player1.lifePoints).toBe(7000);
    expect(state.player2.monsterZones[0]).not.toBeNull();
  });

  it("destroys both monsters when ATK values are equal", () => {
    let state = stateWithMonsters(1500, 1500);
    state = applyAction(state, { type: "DECLARE_ATTACK", attackerZone: 0, targetZone: 0, targetSide: "player2" }, "player1");

    expect(state.player1.monsterZones[0]).toBeNull();
    expect(state.player2.monsterZones[0]).toBeNull();
    expect(state.player1.lifePoints).toBe(8000);
    expect(state.player2.lifePoints).toBe(8000);
  });

  it("direct attack deals full ATK as damage", () => {
    let state = createTestState();
    const m1 = makeGameCard(mockMonster(100, 1800, 500));
    state = {
      ...state,
      phase: "BATTLE",
      player1: {
        ...state.player1,
        monsterZones: [{ card: m1, position: "ATK", faceDown: false }, null, null, null, null],
      },
    };

    state = applyAction(state, { type: "DIRECT_ATTACK", attackerZone: 0 }, "player1");
    expect(state.player2.lifePoints).toBe(6200);
  });

  it("cannot attack outside battle phase", () => {
    let state = createTestState();
    const m1 = makeGameCard(mockMonster(100, 1800, 500));
    const m2 = makeGameCard(mockMonster(200, 1000, 500));
    state = {
      ...state,
      phase: "MAIN1",
      player1: { ...state.player1, monsterZones: [{ card: m1, position: "ATK", faceDown: false }, null, null, null, null] },
      player2: { ...state.player2, monsterZones: [{ card: m2, position: "ATK", faceDown: false }, null, null, null, null] },
    };

    const lpBefore = state.player2.lifePoints;
    state = applyAction(state, { type: "DECLARE_ATTACK", attackerZone: 0, targetZone: 0, targetSide: "player2" }, "player1");
    expect(state.player2.lifePoints).toBe(lpBefore);
  });
});

describe("Game Engine - Win Conditions", () => {
  it("detects win when LP drops to 0", () => {
    let state = createTestState();
    const bigMonster = makeGameCard(mockMonster(999, 8000, 0));
    state = {
      ...state,
      phase: "BATTLE",
      player1: {
        ...state.player1,
        monsterZones: [{ card: bigMonster, position: "ATK", faceDown: false }, null, null, null, null],
      },
    };

    state = applyAction(state, { type: "DIRECT_ATTACK", attackerZone: 0 }, "player1");
    expect(state.winner).toBe("player1");
    expect(state.player2.lifePoints).toBe(0);
  });

  it("detects surrender", () => {
    let state = createTestState();
    state = applyAction(state, { type: "SURRENDER" }, "player1");
    expect(state.winner).toBe("player2");
  });
});

describe("Game Engine - Position Change", () => {
  it("changes monster from ATK to DEF position", () => {
    let state = createTestState();
    const m = makeGameCard(mockMonster(1, 1000, 1500));
    state = {
      ...state,
      phase: "MAIN1",
      player1: {
        ...state.player1,
        monsterZones: [{ card: m, position: "ATK", faceDown: false }, null, null, null, null],
      },
    };

    state = applyAction(state, { type: "CHANGE_POSITION", zoneIndex: 0 }, "player1");
    expect(state.player1.monsterZones[0]?.position).toBe("DEF");
  });
});
