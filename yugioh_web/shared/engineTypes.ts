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
  extra_monster_zone: (EngineFieldCard | null)[]; // [seq5, seq6]
  graveyard: EngineFieldCard[];
  banished: EngineFieldCard[];
  extra_deck: EngineHandCard[];
  deck_count: number;
  lp: number;
}

/** The opponent's board. Face-down cards hidden by default; in open-cards mode
 *  the server sends unhidden zones plus ``hand`` and ``extra_deck`` arrays. */
export interface EngineOpponentBoard {
  hand_count: number;
  hand?: EngineHandCard[]; // present in open-cards mode
  monsters: (EngineFieldCard | null)[]; // 5 slots
  spells_traps: (EngineFieldCard | null)[]; // 5 slots
  field_zone: EngineFieldCard | null;
  extra_monster_zone: (EngineFieldCard | null)[]; // [seq5, seq6]
  graveyard: EngineFieldCard[];
  banished: EngineFieldCard[];
  extra_deck_count: number;
  extra_deck?: EngineHandCard[]; // present in open-cards mode
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
  controller?: 0 | 1;
  location?: number; // engine location constant (0x02=HAND, 0x04=MZONE, etc.)
  sequence?: number; // slot index
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
    | "tribute"
    | "sort"
    | "finish"
    | "option"
    | "unknown";
}

// ─── Prompt metadata ──────────────────────────────────────────────────────────

export type PromptType =
  | "idle_cmd"
  | "battle_cmd"
  | "effect_yn"
  | "yes_no"
  | "option"
  | "select_card"
  | "chain"
  | "place"
  | "position"
  | "tribute"
  | "sort_card"
  | "unknown";

/** Prompt metadata describing the current engine decision context. */
export interface EnginePrompt {
  type: PromptType;
  card_code?: number;
  card_name?: string;
  /** Engine location of the card the prompt targets (effect_yn, position).
   *  Same encoding as EngineAction.location. */
  location?: number;
  min?: number;
  max?: number;
  cancelable?: boolean;
  finishable?: boolean;
  forced?: boolean;
  count?: number;
  /** Sort-specific: visual data for each picked card, in pick order. */
  picked_cards?: { code: number; location: number }[];
  selected_count?: number;
  // Tribute-specific fields
  min_release?: number;
  max_cards?: number;
  release_total?: number;
  cards_selected?: number;
  // Yes/no prompt fields (MSG_SELECT_YESNO, MSG_SELECT_EFFECTYN).
  // `desc` is the raw engine u64 (sysstring id or packed card-string ref);
  // `prompt_text` is the server-resolved display string, or null when the
  // engine emitted desc=0 or the resolver couldn't find the string.
  desc?: number;
  prompt_text?: string | null;
}

/** A snapshot of the board + events captured after one engine processing chunk. */
export interface EventFrame {
  events: string[];
  board: EngineBoard;
  game_state: EngineGameState;
}

/** Unified response shape from all /api/web/ endpoints. */
export interface EngineResponse {
  board: EngineBoard;
  game_state: EngineGameState;
  actions: EngineAction[];
  prompt: EnginePrompt | null;
  event_log: string[];
  done: boolean;
  reward: number;
  frames: EventFrame[];
}
