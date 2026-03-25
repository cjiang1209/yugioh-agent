// ─── Yu-Gi-Oh! Duel Simulator – Shared Game Types ───────────────────────────

export type CardType =
  | "Normal Monster"
  | "Effect Monster"
  | "Ritual Monster"
  | "Fusion Monster"
  | "Synchro Monster"
  | "XYZ Monster"
  | "Link Monster"
  | "Spell Card"
  | "Trap Card"
  | string;

export type CardAttribute = "DARK" | "LIGHT" | "EARTH" | "WATER" | "FIRE" | "WIND" | "DIVINE";

export interface YgoCard {
  id: number;
  name: string;
  type: CardType;
  frameType: string;
  desc: string;
  atk?: number;
  def?: number;
  level?: number;
  race?: string;
  attribute?: CardAttribute;
  card_images: { id: number; image_url: string; image_url_small: string }[];
}

export interface GameCard extends YgoCard {
  /** Unique instance ID within this duel (same card can appear multiple times) */
  instanceId: string;
}

export type CardPosition = "ATK" | "DEF" | "FACE_DOWN_DEF" | "FACE_DOWN_ATK";
export type ZoneType = "monster" | "spell_trap" | "field" | "graveyard" | "banished" | "extra" | "deck" | "hand";

export interface FieldCard {
  card: GameCard;
  position: CardPosition;
  faceDown: boolean;
}

export type Phase =
  | "DRAW"
  | "STANDBY"
  | "MAIN1"
  | "BATTLE"
  | "MAIN2"
  | "END";

export const PHASE_ORDER: Phase[] = ["DRAW", "STANDBY", "MAIN1", "BATTLE", "MAIN2", "END"];
export const PHASE_LABELS: Record<Phase, string> = {
  DRAW: "Draw Phase",
  STANDBY: "Standby Phase",
  MAIN1: "Main Phase 1",
  BATTLE: "Battle Phase",
  MAIN2: "Main Phase 2",
  END: "End Phase",
};

export type PlayerSide = "player1" | "player2";

export interface PlayerState {
  id: string;
  name: string;
  lifePoints: number;
  hand: GameCard[];
  deck: GameCard[];
  graveyard: GameCard[];
  banished: GameCard[];
  extraDeck: GameCard[];
  monsterZones: (FieldCard | null)[];   // 5 slots
  spellTrapZones: (FieldCard | null)[]; // 5 slots
  fieldZone: FieldCard | null;
  extraMonsterZone: FieldCard | null;   // 1 EMZ slot per player
  hasNormalSummoned: boolean;
  hasDrawn: boolean;
}

export interface DuelState {
  roomId: string;
  phase: Phase;
  turnNumber: number;
  activePlayer: PlayerSide;
  player1: PlayerState;
  player2: PlayerState;
  winner: PlayerSide | null;
  battleStep: BattleStep | null;
  log: string[];
}

export interface BattleStep {
  attackerSide: PlayerSide;
  attackerZone: number;
  targetSide?: PlayerSide;
  targetZone?: number;
  isDirect: boolean;
}

// ─── Socket Events ────────────────────────────────────────────────────────────

export interface ServerToClientEvents {
  game_state: (state: DuelState) => void;
  room_joined: (data: { roomId: string; side: PlayerSide; state: DuelState }) => void;
  room_error: (msg: string) => void;
  opponent_connected: (name: string) => void;
  opponent_disconnected: () => void;
  duel_log: (msg: string) => void;
}

export interface ClientToServerEvents {
  join_room: (data: { roomId: string; playerName: string }) => void;
  game_action: (action: GameAction) => void;
}

// ─── Game Actions ─────────────────────────────────────────────────────────────

export type GameAction =
  | { type: "ADVANCE_PHASE" }
  | { type: "DRAW_CARD" }
  | { type: "SUMMON_MONSTER"; handIndex: number; zoneIndex: number; tributeZones?: number[] }
  | { type: "SET_MONSTER"; handIndex: number; zoneIndex: number; tributeZones?: number[] }
  | { type: "CHANGE_POSITION"; zoneIndex: number }
  | { type: "CHANGE_POSITION_EMZ" }
  | { type: "ACTIVATE_SPELL"; handIndex: number; zoneIndex: number }
  | { type: "SET_SPELL_TRAP"; handIndex: number; zoneIndex: number }
  | { type: "ACTIVATE_SET_CARD"; zoneIndex: number }
  | { type: "DECLARE_ATTACK"; attackerZone: number; targetZone: number; targetSide: PlayerSide }
  | { type: "DIRECT_ATTACK"; attackerZone: number }
  | { type: "SEND_TO_GRAVEYARD"; zoneIndex: number; zoneType: "monster" | "spell_trap" }
  | { type: "SEND_EMZ_TO_GRAVEYARD" }
  | { type: "BANISH_CARD"; zoneIndex: number; zoneType: "monster" | "spell_trap" | "graveyard" }
  | { type: "BANISH_EMZ_CARD" }
  | { type: "SUMMON_TO_EMZ"; handIndex: number }
  | { type: "PLAY_FIELD_SPELL"; handIndex: number }
  | { type: "SEND_FIELD_TO_GY" }
  | { type: "ACTIVATE_MONSTER_EFFECT"; zoneIndex: number; zoneType: "monster" | "emz" }
  | { type: "SURRENDER" };
