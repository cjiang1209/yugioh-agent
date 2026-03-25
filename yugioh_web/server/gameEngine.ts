// ─── Yu-Gi-Oh! Duel Engine ────────────────────────────────────────────────────
import { nanoid } from "nanoid";
import {
  BattleStep,
  DuelState,
  FieldCard,
  GameAction,
  GameCard,
  PHASE_ORDER,
  Phase,
  PlayerSide,
  PlayerState,
  YgoCard,
} from "../shared/gameTypes";

// ─── Factory helpers ──────────────────────────────────────────────────────────

export function makeGameCard(card: YgoCard): GameCard {
  return { ...card, instanceId: nanoid(8) };
}

function shuffle<T>(arr: T[]): T[] {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export function createPlayerState(id: string, name: string, deck: YgoCard[]): PlayerState {
  const mainDeck = deck
    .filter((c) => !isExtraDeckCard(c))
    .map(makeGameCard);
  const extraDeck = deck
    .filter((c) => isExtraDeckCard(c))
    .map(makeGameCard);

  const shuffled = shuffle(mainDeck);
  const hand = shuffled.splice(0, 5);

  return {
    id,
    name,
    lifePoints: 8000,
    hand,
    deck: shuffled,
    graveyard: [],
    banished: [],
    extraDeck,
    monsterZones: [null, null, null, null, null],
    spellTrapZones: [null, null, null, null, null],
    fieldZone: null,
    extraMonsterZone: null,
    hasNormalSummoned: false,
    hasDrawn: false,
  };
}

function isExtraDeckCard(card: YgoCard): boolean {
  return (
    card.type.includes("Fusion") ||
    card.type.includes("Synchro") ||
    card.type.includes("XYZ") ||
    card.type.includes("Link")
  );
}

export function createDuelState(
  roomId: string,
  p1Id: string,
  p1Name: string,
  p1Deck: YgoCard[],
  p2Id: string,
  p2Name: string,
  p2Deck: YgoCard[]
): DuelState {
  return {
    roomId,
    phase: "DRAW",
    turnNumber: 1,
    activePlayer: "player1",
    player1: createPlayerState(p1Id, p1Name, p1Deck),
    player2: createPlayerState(p2Id, p2Name, p2Deck),
    winner: null,
    battleStep: null,
    log: ["Duel Start! Player 1 goes first."],
  };
}

// ─── State helpers ────────────────────────────────────────────────────────────

function getActive(state: DuelState): PlayerState {
  return state.activePlayer === "player1" ? state.player1 : state.player2;
}

function getInactive(state: DuelState): PlayerState {
  return state.activePlayer === "player1" ? state.player2 : state.player1;
}

function setActive(state: DuelState, updated: PlayerState): DuelState {
  if (state.activePlayer === "player1") return { ...state, player1: updated };
  return { ...state, player2: updated };
}

function setInactive(state: DuelState, updated: PlayerState): DuelState {
  if (state.activePlayer === "player1") return { ...state, player2: updated };
  return { ...state, player1: updated };
}

function addLog(state: DuelState, msg: string): DuelState {
  return { ...state, log: [...state.log.slice(-49), msg] };
}

function checkWin(state: DuelState): DuelState {
  if (state.player1.lifePoints <= 0) {
    return addLog({ ...state, winner: "player2" }, `${state.player2.name} wins! ${state.player1.name}'s LP reached 0.`);
  }
  if (state.player2.lifePoints <= 0) {
    return addLog({ ...state, winner: "player1" }, `${state.player1.name} wins! ${state.player2.name}'s LP reached 0.`);
  }
  if (state.player1.deck.length === 0 && state.phase === "DRAW") {
    return addLog({ ...state, winner: "player2" }, `${state.player2.name} wins! ${state.player1.name} has no cards to draw.`);
  }
  if (state.player2.deck.length === 0 && state.phase === "DRAW") {
    return addLog({ ...state, winner: "player1" }, `${state.player1.name} wins! ${state.player2.name} has no cards to draw.`);
  }
  return state;
}

// ─── Main reducer ─────────────────────────────────────────────────────────────

export function applyAction(state: DuelState, action: GameAction, actorSide: PlayerSide): DuelState {
  if (state.winner) return state;

  // Only the active player can act (except SURRENDER)
  if (action.type !== "SURRENDER" && actorSide !== state.activePlayer) {
    return addLog(state, "It is not your turn.");
  }

  switch (action.type) {
    case "ADVANCE_PHASE":
      return advancePhase(state);

    case "DRAW_CARD":
      return drawCard(state);

    case "SUMMON_MONSTER":
      return summonMonster(state, action.handIndex, action.zoneIndex, action.tributeZones);

    case "SET_MONSTER":
      return setMonster(state, action.handIndex, action.zoneIndex, action.tributeZones);

    case "CHANGE_POSITION":
      return changePosition(state, action.zoneIndex);

    case "ACTIVATE_SPELL":
      return activateSpell(state, action.handIndex, action.zoneIndex);

    case "SET_SPELL_TRAP":
      return setSpellTrap(state, action.handIndex, action.zoneIndex);

    case "ACTIVATE_SET_CARD":
      return activateSetCard(state, action.zoneIndex);

    case "DECLARE_ATTACK":
      return declareAttack(state, action.attackerZone, action.targetZone, action.targetSide);

    case "DIRECT_ATTACK":
      return directAttack(state, action.attackerZone);

    case "SEND_TO_GRAVEYARD":
      return sendToGraveyard(state, action.zoneIndex, action.zoneType);

    case "BANISH_CARD":
      return banishCard(state, action.zoneIndex, action.zoneType);

    case "PLAY_FIELD_SPELL":
      return playFieldSpell(state, action.handIndex);

    case "SEND_FIELD_TO_GY":
      return sendFieldToGraveyard(state);

    case "SUMMON_TO_EMZ":
      return summonToEMZ(state, action.handIndex);

    case "CHANGE_POSITION_EMZ":
      return changePositionEMZ(state);

    case "SEND_EMZ_TO_GRAVEYARD":
      return sendEMZToGraveyard(state);

    case "BANISH_EMZ_CARD":
      return banishEMZCard(state);

    case "ACTIVATE_MONSTER_EFFECT": {
      const actor = actorSide === "player1" ? state.player1 : state.player2;
      let cardName = "Unknown";
      if (action.zoneType === "emz") {
        cardName = actor.extraMonsterZone?.card.name ?? "Unknown";
      } else {
        cardName = actor.monsterZones[action.zoneIndex]?.card.name ?? "Unknown";
      }
      return addLog(state, `${actor.name} activates the effect of ${cardName}.`);
    }

    case "SURRENDER":
      const winner: PlayerSide = actorSide === "player1" ? "player2" : "player1";
      const loserName = actorSide === "player1" ? state.player1.name : state.player2.name;
      return addLog({ ...state, winner }, `${loserName} has surrendered.`);

    default:
      return state;
  }
}

// ─── Phase logic ──────────────────────────────────────────────────────────────

function advancePhase(state: DuelState): DuelState {
  const currentIndex = PHASE_ORDER.indexOf(state.phase);
  if (currentIndex === PHASE_ORDER.length - 1) {
    // End Phase → switch turns
    const nextPlayer: PlayerSide = state.activePlayer === "player1" ? "player2" : "player1";
    let next: DuelState = {
      ...state,
      phase: "DRAW",
      activePlayer: nextPlayer,
      turnNumber: state.turnNumber + 1,
      battleStep: null,
    };
    // Reset normal summon flag
    const activeP = getActive(next);
    next = setActive(next, { ...activeP, hasNormalSummoned: false, hasDrawn: false });
    next = addLog(next, `Turn ${next.turnNumber}: ${getActive(next).name}'s turn begins.`);
    return next;
  }

  const nextPhase = PHASE_ORDER[currentIndex + 1];
  let next: DuelState = { ...state, phase: nextPhase, battleStep: null };
  next = addLog(next, `${getActive(next).name} enters ${nextPhase.replace("_", " ")}.`);
  return next;
}

function drawCard(state: DuelState): DuelState {
  const active = getActive(state);
  if (active.hasDrawn) {
    return addLog(state, "You have already drawn this turn.");
  }
  if (active.deck.length === 0) {
    return addLog(checkWin({ ...state }), `${active.name} has no cards left to draw!`);
  }
  const [drawn, ...rest] = active.deck;
  const updated = { ...active, deck: rest, hand: [...active.hand, drawn], hasDrawn: true };
  let next = setActive(state, updated);
  next = addLog(next, `${active.name} draws a card. (${rest.length} left in deck)`);
  return next;
}

// ─── Monster summon ───────────────────────────────────────────────────────────

function requiredTributes(level: number): number {
  if (level <= 4) return 0;
  if (level <= 6) return 1;
  return 2;
}

function summonMonster(
  state: DuelState,
  handIndex: number,
  zoneIndex: number,
  tributeZones?: number[]
): DuelState {
  if (state.phase !== "MAIN1" && state.phase !== "MAIN2") {
    return addLog(state, "You can only summon during Main Phase.");
  }
  const active = getActive(state);
  if (active.hasNormalSummoned) {
    return addLog(state, "You have already Normal Summoned this turn.");
  }
  if (zoneIndex < 0 || zoneIndex > 4 || active.monsterZones[zoneIndex] !== null) {
    return addLog(state, "Invalid or occupied monster zone.");
  }

  const card = active.hand[handIndex];
  if (!card) return addLog(state, "No card at that hand index.");
  if (!card.type.includes("Monster")) return addLog(state, "That card is not a monster.");

  const level = card.level ?? 1;
  const needed = requiredTributes(level);

  if (needed > 0) {
    if (!tributeZones || tributeZones.length < needed) {
      return addLog(state, `You need ${needed} tribute(s) to summon this monster.`);
    }
    // Remove tributes
    const newZones = [...active.monsterZones] as (typeof active.monsterZones);
    let graveyard = [...active.graveyard];
    for (const tz of tributeZones) {
      const tribute = newZones[tz];
      if (!tribute) return addLog(state, "Invalid tribute zone.");
      graveyard = [...graveyard, tribute.card];
      newZones[tz] = null;
    }
    const newHand = active.hand.filter((_, i) => i !== handIndex);
    newZones[zoneIndex] = { card, position: "ATK", faceDown: false };
    const updated = { ...active, hand: newHand, monsterZones: newZones, graveyard, hasNormalSummoned: true };
    return addLog(setActive(state, updated), `${active.name} tribute summons ${card.name}!`);
  }

  const newHand = active.hand.filter((_, i) => i !== handIndex);
  const newZones = [...active.monsterZones] as (typeof active.monsterZones);
  newZones[zoneIndex] = { card, position: "ATK", faceDown: false };
  const updated = { ...active, hand: newHand, monsterZones: newZones, hasNormalSummoned: true };
  return addLog(setActive(state, updated), `${active.name} normal summons ${card.name} (ATK: ${card.atk})!`);
}

function setMonster(
  state: DuelState,
  handIndex: number,
  zoneIndex: number,
  tributeZones?: number[]
): DuelState {
  if (state.phase !== "MAIN1" && state.phase !== "MAIN2") {
    return addLog(state, "You can only set during Main Phase.");
  }
  const active = getActive(state);
  if (active.hasNormalSummoned) {
    return addLog(state, "You have already Normal Summoned/Set this turn.");
  }
  if (zoneIndex < 0 || zoneIndex > 4 || active.monsterZones[zoneIndex] !== null) {
    return addLog(state, "Invalid or occupied monster zone.");
  }
  const card = active.hand[handIndex];
  if (!card || !card.type.includes("Monster")) return addLog(state, "Invalid card.");

  const level = card.level ?? 1;
  const needed = requiredTributes(level);
  if (needed > 0) {
    if (!tributeZones || tributeZones.length < needed) {
      return addLog(state, `You need ${needed} tribute(s) to set this monster.`);
    }
    const newZones = [...active.monsterZones] as (typeof active.monsterZones);
    let graveyard = [...active.graveyard];
    for (const tz of tributeZones) {
      const tribute = newZones[tz];
      if (!tribute) return addLog(state, "Invalid tribute zone.");
      graveyard = [...graveyard, tribute.card];
      newZones[tz] = null;
    }
    const newHand = active.hand.filter((_, i) => i !== handIndex);
    newZones[zoneIndex] = { card, position: "FACE_DOWN_DEF", faceDown: true };
    const updated = { ...active, hand: newHand, monsterZones: newZones, graveyard, hasNormalSummoned: true };
    return addLog(setActive(state, updated), `${active.name} sets a monster.`);
  }

  const newHand = active.hand.filter((_, i) => i !== handIndex);
  const newZones = [...active.monsterZones] as (typeof active.monsterZones);
  newZones[zoneIndex] = { card, position: "FACE_DOWN_DEF", faceDown: true };
  const updated = { ...active, hand: newHand, monsterZones: newZones, hasNormalSummoned: true };
  return addLog(setActive(state, updated), `${active.name} sets a monster face-down.`);
}

function changePosition(state: DuelState, zoneIndex: number): DuelState {
  if (state.phase !== "MAIN1" && state.phase !== "MAIN2") {
    return addLog(state, "You can only change position during Main Phase.");
  }
  const active = getActive(state);
  const slot = active.monsterZones[zoneIndex];
  if (!slot) return addLog(state, "No monster in that zone.");

  let newPos: FieldCard["position"];
  let msg: string;
  if (slot.position === "ATK") {
    newPos = "DEF";
    msg = `${active.name} changes ${slot.card.name} to Defense Position.`;
  } else if (slot.position === "DEF") {
    newPos = "ATK";
    msg = `${active.name} changes ${slot.card.name} to Attack Position.`;
  } else if (slot.position === "FACE_DOWN_DEF") {
    // Flip summon
    newPos = "ATK";
    msg = `${active.name} flip summons ${slot.card.name}!`;
  } else {
    return addLog(state, "Cannot change position.");
  }

  const newZones = [...active.monsterZones] as (typeof active.monsterZones);
  newZones[zoneIndex] = { ...slot, position: newPos, faceDown: false };
  const updated = { ...active, monsterZones: newZones };
  return addLog(setActive(state, updated), msg);
}

// ─── Spell / Trap ─────────────────────────────────────────────────────────────

function activateSpell(state: DuelState, handIndex: number, zoneIndex: number): DuelState {
  if (state.phase !== "MAIN1" && state.phase !== "MAIN2") {
    return addLog(state, "You can only activate spells during Main Phase.");
  }
  const active = getActive(state);
  if (zoneIndex < 0 || zoneIndex > 4 || active.spellTrapZones[zoneIndex] !== null) {
    return addLog(state, "Invalid or occupied spell/trap zone.");
  }
  const card = active.hand[handIndex];
  if (!card || !card.type.includes("Spell")) return addLog(state, "That is not a Spell card.");

  const newHand = active.hand.filter((_, i) => i !== handIndex);
  // For simplicity, place it face-up then immediately send to graveyard (unless Continuous/Field/Equip)
  const isContinuous = card.race === "Continuous" || card.race === "Field" || card.race === "Equip";
  if (isContinuous) {
    const newZones = [...active.spellTrapZones] as (typeof active.spellTrapZones);
    newZones[zoneIndex] = { card, position: "ATK", faceDown: false };
    const updated = { ...active, hand: newHand, spellTrapZones: newZones };
    return addLog(setActive(state, updated), `${active.name} activates ${card.name}!`);
  }
  // Normal spell → resolve and send to GY
  const graveyard = [...active.graveyard, card];
  const updated = { ...active, hand: newHand, graveyard };
  return addLog(setActive(state, updated), `${active.name} activates ${card.name}! (sent to GY)`);
}

function setSpellTrap(state: DuelState, handIndex: number, zoneIndex: number): DuelState {
  if (state.phase !== "MAIN1" && state.phase !== "MAIN2") {
    return addLog(state, "You can only set during Main Phase.");
  }
  const active = getActive(state);
  if (zoneIndex < 0 || zoneIndex > 4 || active.spellTrapZones[zoneIndex] !== null) {
    return addLog(state, "Invalid or occupied spell/trap zone.");
  }
  const card = active.hand[handIndex];
  if (!card || (!card.type.includes("Spell") && !card.type.includes("Trap"))) {
    return addLog(state, "That is not a Spell or Trap card.");
  }
  const newHand = active.hand.filter((_, i) => i !== handIndex);
  const newZones = [...active.spellTrapZones] as (typeof active.spellTrapZones);
  newZones[zoneIndex] = { card, position: "ATK", faceDown: true };
  const updated = { ...active, hand: newHand, spellTrapZones: newZones };
  return addLog(setActive(state, updated), `${active.name} sets a card face-down.`);
}

function activateSetCard(state: DuelState, zoneIndex: number): DuelState {
  const active = getActive(state);
  const slot = active.spellTrapZones[zoneIndex];
  if (!slot || !slot.faceDown) return addLog(state, "No face-down card there.");
  const newZones = [...active.spellTrapZones] as (typeof active.spellTrapZones);
  // Flip face-up; for traps, resolve and send to GY
  if (slot.card.type.includes("Trap")) {
    const graveyard = [...active.graveyard, slot.card];
    newZones[zoneIndex] = null;
    const updated = { ...active, spellTrapZones: newZones, graveyard };
    return addLog(setActive(state, updated), `${active.name} activates Trap: ${slot.card.name}!`);
  }
  newZones[zoneIndex] = { ...slot, faceDown: false };
  const updated = { ...active, spellTrapZones: newZones };
  return addLog(setActive(state, updated), `${active.name} activates ${slot.card.name}!`);
}

// ─── Battle system ────────────────────────────────────────────────────────────

function declareAttack(
  state: DuelState,
  attackerZone: number,
  targetZone: number,
  targetSide: PlayerSide
): DuelState {
  if (state.phase !== "BATTLE") {
    return addLog(state, "You can only attack during the Battle Phase.");
  }
  const active = getActive(state);
  const attacker = active.monsterZones[attackerZone];
  if (!attacker || attacker.faceDown) return addLog(state, "No valid attacker in that zone.");
  if (attacker.position !== "ATK") return addLog(state, "Monster must be in ATK position to attack.");

  const defenderPlayer = targetSide === "player1" ? state.player1 : state.player2;
  const defender = defenderPlayer.monsterZones[targetZone];
  if (!defender) return addLog(state, "No monster in that zone.");

  const atkVal = attacker.card.atk ?? 0;

  // Flip face-down monster
  const defCard = defender.faceDown
    ? { ...defender, faceDown: false, position: "DEF" as const }
    : defender;
  const defVal = defCard.position === "DEF" ? (defCard.card.def ?? 0) : (defCard.card.atk ?? 0);

  let s = addLog(
    state,
    `${active.name}'s ${attacker.card.name} (ATK:${atkVal}) attacks ${defCard.card.name} (${defCard.position === "DEF" ? "DEF" : "ATK"}:${defVal})!`
  );

  if (defCard.position === "DEF") {
    // ATK vs DEF
    if (atkVal > defVal) {
      // Destroy defender, no damage
      s = destroyMonster(s, targetSide, targetZone);
      s = addLog(s, `${defCard.card.name} is destroyed!`);
    } else if (atkVal < defVal) {
      // Attacker takes piercing damage? Standard rules: no damage in DEF
      s = addLog(s, `${attacker.card.name} cannot destroy ${defCard.card.name}.`);
    } else {
      s = addLog(s, "Both monsters survive (equal values).");
    }
  } else {
    // ATK vs ATK
    if (atkVal > defVal) {
      const dmg = atkVal - defVal;
      s = destroyMonster(s, targetSide, targetZone);
      s = dealDamage(s, targetSide, dmg);
      s = addLog(s, `${defCard.card.name} is destroyed! ${defenderPlayer.name} takes ${dmg} damage.`);
    } else if (atkVal < defVal) {
      const dmg = defVal - atkVal;
      s = destroyMonster(s, state.activePlayer, attackerZone);
      s = dealDamage(s, state.activePlayer, dmg);
      s = addLog(s, `${attacker.card.name} is destroyed! ${active.name} takes ${dmg} damage.`);
    } else {
      // Both destroyed
      s = destroyMonster(s, targetSide, targetZone);
      s = destroyMonster(s, state.activePlayer, attackerZone);
      s = addLog(s, "Both monsters are destroyed!");
    }
  }

  return checkWin(s);
}

function directAttack(state: DuelState, attackerZone: number): DuelState {
  if (state.phase !== "BATTLE") {
    return addLog(state, "You can only attack during the Battle Phase.");
  }
  const active = getActive(state);
  const inactive = getInactive(state);
  const attacker = active.monsterZones[attackerZone];
  if (!attacker || attacker.faceDown) return addLog(state, "No valid attacker.");
  if (attacker.position !== "ATK") return addLog(state, "Monster must be in ATK position.");

  const hasMonsters = inactive.monsterZones.some((z) => z !== null);
  if (hasMonsters) return addLog(state, "Opponent has monsters; you cannot attack directly.");

  const dmg = attacker.card.atk ?? 0;
  let s = dealDamage(state, state.activePlayer === "player1" ? "player2" : "player1", dmg);
  s = addLog(s, `${active.name}'s ${attacker.card.name} attacks directly! ${inactive.name} takes ${dmg} damage.`);
  return checkWin(s);
}

function destroyMonster(state: DuelState, side: PlayerSide, zoneIndex: number): DuelState {
  const player = side === "player1" ? state.player1 : state.player2;
  const slot = player.monsterZones[zoneIndex];
  if (!slot) return state;
  const newZones = [...player.monsterZones] as (typeof player.monsterZones);
  newZones[zoneIndex] = null;
  const graveyard = [...player.graveyard, slot.card];
  const updated = { ...player, monsterZones: newZones, graveyard };
  if (side === "player1") return { ...state, player1: updated };
  return { ...state, player2: updated };
}

function dealDamage(state: DuelState, side: PlayerSide, amount: number): DuelState {
  if (side === "player1") {
    return { ...state, player1: { ...state.player1, lifePoints: Math.max(0, state.player1.lifePoints - amount) } };
  }
  return { ...state, player2: { ...state.player2, lifePoints: Math.max(0, state.player2.lifePoints - amount) } };
}

function sendToGraveyard(state: DuelState, zoneIndex: number, zoneType: "monster" | "spell_trap"): DuelState {
  const active = getActive(state);
  if (zoneType === "monster") {
    const slot = active.monsterZones[zoneIndex];
    if (!slot) return addLog(state, "No card there.");
    const newZones = [...active.monsterZones] as (typeof active.monsterZones);
    newZones[zoneIndex] = null;
    const updated = { ...active, monsterZones: newZones, graveyard: [...active.graveyard, slot.card] };
    return addLog(setActive(state, updated), `${active.name} sends ${slot.card.name} to the graveyard.`);
  } else {
    const slot = active.spellTrapZones[zoneIndex];
    if (!slot) return addLog(state, "No card there.");
    const newZones = [...active.spellTrapZones] as (typeof active.spellTrapZones);
    newZones[zoneIndex] = null;
    const updated = { ...active, spellTrapZones: newZones, graveyard: [...active.graveyard, slot.card] };
    return addLog(setActive(state, updated), `${active.name} sends ${slot.card.name} to the graveyard.`);
  }
}

// ─── Field Zone ───────────────────────────────────────────────────────────────

function playFieldSpell(state: DuelState, handIndex: number): DuelState {
  if (state.phase !== "MAIN1" && state.phase !== "MAIN2") {
    return addLog(state, "You can only play Field Spells during Main Phase.");
  }
  const active = getActive(state);
  const card = active.hand[handIndex];
  if (!card) return addLog(state, "No card at that hand index.");
  if (!card.type.includes("Spell") || card.race !== "Field") {
    return addLog(state, "That card is not a Field Spell.");
  }
  // If there's already a field spell, send it to GY first
  let graveyard = [...active.graveyard];
  if (active.fieldZone) {
    graveyard = [...graveyard, active.fieldZone.card];
  }
  const newHand = active.hand.filter((_, i) => i !== handIndex);
  const updated = { ...active, hand: newHand, fieldZone: { card, position: "ATK" as const, faceDown: false }, graveyard };
  return addLog(setActive(state, updated), `${active.name} activates Field Spell: ${card.name}!`);
}

function sendFieldToGraveyard(state: DuelState): DuelState {
  const active = getActive(state);
  if (!active.fieldZone) return addLog(state, "No Field Spell in play.");
  const card = active.fieldZone.card;
  const graveyard = [...active.graveyard, card];
  const updated = { ...active, fieldZone: null, graveyard };
  return addLog(setActive(state, updated), `${active.name} sends ${card.name} to the graveyard.`);
}

function banishCard(state: DuelState, zoneIndex: number, zoneType: "monster" | "spell_trap" | "graveyard"): DuelState {
  const active = getActive(state);
  if (zoneType === "monster") {
    const slot = active.monsterZones[zoneIndex];
    if (!slot) return addLog(state, "No card there.");
    const newZones = [...active.monsterZones] as (typeof active.monsterZones);
    newZones[zoneIndex] = null;
    const updated = { ...active, monsterZones: newZones, banished: [...active.banished, slot.card] };
    return addLog(setActive(state, updated), `${active.name} banishes ${slot.card.name}!`);
  } else if (zoneType === "spell_trap") {
    const slot = active.spellTrapZones[zoneIndex];
    if (!slot) return addLog(state, "No card there.");
    const newZones = [...active.spellTrapZones] as (typeof active.spellTrapZones);
    newZones[zoneIndex] = null;
    const updated = { ...active, spellTrapZones: newZones, banished: [...active.banished, slot.card] };
    return addLog(setActive(state, updated), `${active.name} banishes ${slot.card.name}!`);
  } else {
    // Banish from graveyard
    if (zoneIndex < 0 || zoneIndex >= active.graveyard.length) return addLog(state, "Invalid graveyard index.");
    const card = active.graveyard[zoneIndex];
    const newGY = active.graveyard.filter((_, i) => i !== zoneIndex);
    const updated = { ...active, graveyard: newGY, banished: [...active.banished, card] };
    return addLog(setActive(state, updated), `${active.name} banishes ${card.name} from the graveyard!`);
  }
}

// ─── Extra Monster Zone ───────────────────────────────────────────────────────

function summonToEMZ(state: DuelState, handIndex: number): DuelState {
  if (state.phase !== "MAIN1" && state.phase !== "MAIN2") {
    return addLog(state, "You can only Special Summon during Main Phase.");
  }
  const active = getActive(state);
  if (active.extraMonsterZone !== null) {
    return addLog(state, "Your Extra Monster Zone is already occupied.");
  }
  const card = active.hand[handIndex];
  if (!card) return addLog(state, "No card at that hand index.");
  if (!isExtraDeckCard(card)) {
    return addLog(state, "Only Extra Deck monsters (Fusion, Synchro, Xyz, Link) can be placed in the Extra Monster Zone.");
  }
  const newHand = active.hand.filter((_, i) => i !== handIndex);
  const updated = {
    ...active,
    hand: newHand,
    extraMonsterZone: { card, position: "ATK" as const, faceDown: false },
  };
  return addLog(setActive(state, updated), `${active.name} special summons ${card.name} to the Extra Monster Zone!`);
}

function changePositionEMZ(state: DuelState): DuelState {
  if (state.phase !== "MAIN1" && state.phase !== "MAIN2") {
    return addLog(state, "You can only change position during Main Phase.");
  }
  const active = getActive(state);
  const slot = active.extraMonsterZone;
  if (!slot) return addLog(state, "No monster in the Extra Monster Zone.");
  // Link Monsters cannot be in Defense Position
  if (slot.card.type.includes("Link")) {
    return addLog(state, "Link Monsters cannot be in Defense Position.");
  }
  const newPos: FieldCard["position"] = slot.position === "ATK" ? "DEF" : "ATK";
  const updated = { ...active, extraMonsterZone: { ...slot, position: newPos } };
  return addLog(setActive(state, updated), `${active.name} changes ${slot.card.name} to ${newPos} position.`);
}

function sendEMZToGraveyard(state: DuelState): DuelState {
  const active = getActive(state);
  const slot = active.extraMonsterZone;
  if (!slot) return addLog(state, "No card in the Extra Monster Zone.");
  const updated = {
    ...active,
    extraMonsterZone: null,
    graveyard: [...active.graveyard, slot.card],
  };
  return addLog(setActive(state, updated), `${active.name} sends ${slot.card.name} from the Extra Monster Zone to the graveyard.`);
}

function banishEMZCard(state: DuelState): DuelState {
  const active = getActive(state);
  const slot = active.extraMonsterZone;
  if (!slot) return addLog(state, "No card in the Extra Monster Zone.");
  const updated = {
    ...active,
    extraMonsterZone: null,
    banished: [...active.banished, slot.card],
  };
  return addLog(setActive(state, updated), `${active.name} banishes ${slot.card.name} from the Extra Monster Zone!`);
}
