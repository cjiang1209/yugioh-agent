// ─── Engine Types ────────────────────────────────────────────────────────────
// TypeScript types mirroring the Python web API response from /api/web/*.
// These represent the real ygopro-core engine state, as opposed to the toy
// game types in gameTypes.ts.

/** Card in the player's hand (no position info). */
export interface EngineHandCard {
  code: number;
  name: string;
  type: "monster" | "spell" | "trap" | "unknown";
  attack?: number | null;
  defense?: number | null;
  level?: number;
  link_rating?: number;
}

/** Card occupying a field zone (monster, spell/trap, graveyard, banished). */
export interface EngineFieldCard {
  code: number;
  name: string;
  type: "monster" | "spell" | "trap" | "unknown";
  position: "ATK" | "DEF" | "FACE_DOWN_ATK" | "FACE_DOWN_DEF";
  attack: number | null;
  defense: number | null;
  level: number;
  link_rating?: number;
}

/** The human player's board (full information). */
export interface EnginePlayerBoard {
  hand: EngineHandCard[];
  monsters: (EngineFieldCard | null)[]; // 5 slots
  spells_traps: (EngineFieldCard | null)[]; // 5 slots
  field_zone: EngineFieldCard | null;
  graveyard: EngineFieldCard[];
  banished: EngineFieldCard[];
  extra_deck_count: number;
  deck_count: number;
  lp: number;
}

/** The opponent's board (face-down cards hidden). */
export interface EngineOpponentBoard {
  hand_count: number;
  monsters: (EngineFieldCard | null)[]; // 5 slots
  spells_traps: (EngineFieldCard | null)[]; // 5 slots
  field_zone: EngineFieldCard | null;
  graveyard: EngineFieldCard[];
  banished: EngineFieldCard[];
  extra_deck_count: number;
  deck_count: number;
  lp: number;
}

export interface EngineBoard {
  player: EnginePlayerBoard;
  opponent: EngineOpponentBoard;
}

/** Current duel phase and turn info. */
export interface EngineGameState {
  turn: number;
  phase:
    | "draw"
    | "standby"
    | "main1"
    | "battle_start"
    | "battle_step"
    | "damage"
    | "damage_calc"
    | "battle"
    | "main2"
    | "end"
    | "unknown";
  is_my_turn: boolean;
  chain_count: number;
}

/** A single legal action the engine offers. */
export interface EngineAction {
  index: number;
  description: string;
  card_code: number;
  card_name: string;
  category:
    | "summon"
    | "special_summon"
    | "reposition"
    | "monster_set"
    | "spell_set"
    | "activate"
    | "to_battle"
    | "to_end"
    | "attack"
    | "to_main2"
    | "yes"
    | "no"
    | "chain"
    | "pass"
    | "select_card"
    | "position"
    | "place"
    | "finish"
    | "option"
    | "unknown";
}

/** Unified response shape from all /api/web/ endpoints. */
export interface EngineResponse {
  board: EngineBoard;
  game_state: EngineGameState;
  actions: EngineAction[];
  event_log: string[];
  done: boolean;
  reward: number;
}
