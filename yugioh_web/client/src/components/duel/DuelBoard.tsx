import { useEffect, useRef, useState } from "react";
import { Ghost, Mountain, ScrollText, Skull, Star, Swords } from "lucide-react";
import {
  DuelState,
  FieldCard,
  GameAction,
  GameCard,
  PlayerSide,
  PlayerState,
} from "../../../../shared/gameTypes";
import { ActionMenu } from "./ActionMenu";
import { CardZone, HandCard } from "./CardZone";
import { DuelLog } from "./DuelLog";
import { ZoneViewer } from "./ZoneViewer";
import { LifePoints } from "./LifePoints";
import { AttackAnimation } from "./AttackAnimation";
import { SummonAnimation } from "./SummonAnimation";
import { PhaseIndicator } from "./PhaseIndicator";
import { EnginePromptRouter } from "./EnginePromptRouter";
import { ChainWidget } from "./ChainWidget";
import type {
  EngineAction,
  EnginePrompt,
} from "../../../../shared/engineTypes";

interface DuelBoardProps {
  state: DuelState;
  mySide: PlayerSide;
  onAction: (action: GameAction) => void;
  engineMode?: boolean;
  engineActions?: EngineAction[];
  enginePrompt?: EnginePrompt | null;
  onEngineAction?: (actionIndex: number) => void;
  onRestart?: () => void;
  visibleLog?: string[];
  isReplaying?: boolean;
  openCards?: boolean;
}

type SelectionMode =
  | { type: "none" }
  | { type: "hand"; index: number; card: GameCard }
  | { type: "attacker"; zone: number }
  | { type: "tribute"; zones: number[]; handIndex: number; needed: number };

interface ContextMenuState {
  x: number;
  y: number;
  items: {
    label: string;
    action: () => void;
    color?: string;
    disabled?: boolean;
  }[];
}

/** Stable card locator — matches card.instanceId format from useAIEngine. */
type BoardZone = "hand" | "mzone" | "szone" | "emz" | "field";

function locatorKey(
  cardCode: number,
  side: string,
  zone: string,
  seq: number
): string {
  return `${cardCode}-${side}-${zone}-${seq}`;
}

/** Engine location constants. */
const LOCATION_MZONE = 0x04;
const LOCATION_SZONE = 0x08;

/** Map engine location constants to BoardZone names. */
const LOCATION_TO_ZONE: Record<number, string> = {
  0x02: "hand",
  [LOCATION_MZONE]: "mzone",
  [LOCATION_SZONE]: "szone",
  0x10: "grave",
  0x20: "banished",
  0x40: "extra",
};

// Module-level cache for card descriptions fetched from YGOProDeck API
const descCache = new Map<number, string>();

function isExtraDeckType(cardType: string | undefined): boolean {
  return (
    !!cardType &&
    (cardType.includes("Fusion") ||
      cardType.includes("Synchro") ||
      cardType.includes("XYZ") ||
      cardType.includes("Link"))
  );
}

function requiredTributes(level: number) {
  if (level <= 4) return 0;
  if (level <= 6) return 1;
  return 2;
}

// ─── Empty zone placeholders ─────────────────────────────────────────────────

function ZoneLabel({
  icon,
  children,
  color,
}: {
  icon: React.ReactNode;
  children: React.ReactNode;
  color?: string;
}) {
  return (
    <div className="w-full h-full flex flex-col items-center justify-center gap-0.5">
      <span
        style={{
          fontSize: "clamp(0.7rem, 1.4vw, 1.2rem)",
          opacity: 0.5,
          color: color ?? "var(--neon-cyan)",
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        {icon}
      </span>
      <span
        style={{
          fontFamily: "'Orbitron', sans-serif",
          fontSize: "clamp(0.3rem, 0.55vw, 0.45rem)",
          color: color ?? "var(--neon-cyan)",
          opacity: 0.6,
          letterSpacing: "0.05em",
          textAlign: "center",
          lineHeight: 1.2,
        }}
      >
        {children}
      </span>
    </div>
  );
}

const MONSTER_EMPTY = (
  <ZoneLabel icon={<Swords size={18} strokeWidth={2} />}>MONSTER</ZoneLabel>
);
const SPELL_TRAP_EMPTY = (
  <ZoneLabel icon={<ScrollText size={18} strokeWidth={2} />}>
    SPELL
    <br />
    TRAP
  </ZoneLabel>
);
const FIELD_EMPTY = (
  <ZoneLabel icon={<Mountain size={18} strokeWidth={2} />}>FIELD</ZoneLabel>
);
const EMZ_EMPTY = (
  <ZoneLabel
    icon={<Swords size={18} strokeWidth={2} />}
    color="rgba(255,215,0,0.7)"
  >
    EXTRA
    <br />
    MONSTER
  </ZoneLabel>
);

const EMZ_BADGE = (
  <div
    style={{
      position: "absolute",
      top: "2px",
      right: "2px",
      background: "rgba(255,215,0,0.85)",
      borderRadius: "2px",
      padding: "1px 3px",
      fontSize: "0.35rem",
      fontFamily: "'Orbitron', sans-serif",
      color: "#000",
      fontWeight: 700,
      letterSpacing: "0.03em",
    }}
  >
    EMZ
  </div>
);

export function DuelBoard({
  state,
  mySide,
  onAction,
  engineMode,
  engineActions,
  enginePrompt,
  onEngineAction,
  onRestart,
  visibleLog,
  isReplaying,
  openCards,
}: DuelBoardProps) {
  const [selection, setSelection] = useState<SelectionMode>({ type: "none" });
  const [contextMenu, setContextMenu] = useState<ContextMenuState | null>(null);
  const [bottomTab, setBottomTab] = useState<"actions" | "log">(
    engineMode ? "actions" : "log"
  );
  const [selectedCardDetail, setSelectedCardDetail] = useState<GameCard | null>(
    null
  );
  const [selectedLocator, setSelectedLocator] = useState<string | null>(null);

  /** True when an opponent's card has visible data (face-up, or face-down in open-cards mode). */
  const canShowOppDetail = (slot: FieldCard) =>
    !slot.faceDown || (openCards && slot.card.id > 0);

  function selectCardForDetail(
    card: GameCard,
    side: "mine" | "opp",
    zone: BoardZone,
    seq: number
  ) {
    setSelectedCardDetail(card);
    setSelectedLocator(locatorKey(card.id, side, zone, seq));
  }

  /** Check if a board position still holds the expected card. */
  function isLocatorMatch(
    cardId: number | undefined,
    side: string,
    zone: string,
    seq: number
  ): boolean {
    if (!selectedLocator || cardId === undefined) return false;
    return selectedLocator === locatorKey(cardId, side, zone, seq);
  }

  // Build set of actionable card locator keys from engine actions
  const actionableKeys = new Set<string>();
  const actionableZones = new Set<string>();
  if (engineActions) {
    for (const a of engineActions) {
      if (
        a.card_code &&
        a.controller !== undefined &&
        a.location !== undefined &&
        a.sequence !== undefined
      ) {
        const side = a.controller === 0 ? "mine" : "opp";
        if (a.location === LOCATION_SZONE && a.sequence === 5) {
          actionableKeys.add(locatorKey(a.card_code, side, "field", 0));
          actionableZones.add(`${side}-field`);
        } else if (a.location === LOCATION_MZONE && a.sequence >= 5) {
          actionableKeys.add(
            locatorKey(a.card_code, side, "emz", a.sequence - 5)
          );
          actionableZones.add(`${side}-emz`);
        } else {
          const zone = LOCATION_TO_ZONE[a.location];
          if (zone) {
            actionableKeys.add(locatorKey(a.card_code, side, zone, a.sequence));
            actionableZones.add(`${side}-${zone}`);
          }
        }
      }
    }
  }

  function isActionable(
    cardId: number | undefined,
    side: string,
    zone: string,
    seq: number
  ): boolean {
    if (!cardId || actionableKeys.size === 0) return false;
    return actionableKeys.has(locatorKey(cardId, side, zone, seq));
  }

  const [zoneViewer, setZoneViewer] = useState<{
    side: PlayerSide;
    tab: "graveyard" | "banished" | "extra";
  } | null>(null);
  const [showRestartConfirm, setShowRestartConfirm] = useState(false);

  // Fetch card description from YGOProDeck API when a card is selected
  useEffect(() => {
    if (
      !selectedCardDetail ||
      !selectedCardDetail.id ||
      selectedCardDetail.desc
    )
      return;
    const id = selectedCardDetail.id;
    if (descCache.has(id)) {
      setSelectedCardDetail(prev =>
        prev?.id === id ? { ...prev, desc: descCache.get(id)! } : prev
      );
      return;
    }
    fetch(`https://db.ygoprodeck.com/api/v7/cardinfo.php?id=${id}`)
      .then(r => r.json())
      .then(data => {
        const desc = data.data?.[0]?.desc ?? "";
        descCache.set(id, desc);
        setSelectedCardDetail(prev =>
          prev?.id === id ? { ...prev, desc } : prev
        );
      })
      .catch(() => {});
  }, [selectedCardDetail?.id, selectedCardDetail?.desc]);
  const [attackAnim, setAttackAnim] = useState<{
    fromRect: DOMRect;
    toRect: DOMRect | null;
    pendingAction: GameAction;
  } | null>(null);
  const [summonAnim, setSummonAnim] = useState<{
    zoneRect: DOMRect;
    kind: "normal" | "special";
    pendingAction: GameAction;
  } | null>(null);
  // Damage flash: which LP counters are flashing
  const [lpFlash, setLpFlash] = useState<{ my: boolean; opp: boolean }>({
    my: false,
    opp: false,
  });
  // Track previous LP values to detect damage
  const prevMyLp = useRef<number | null>(null);
  const prevOppLp = useRef<number | null>(null);

  // Refs to the monster zone row containers for getBoundingClientRect calculations
  const myMonsterRowRef = useRef<HTMLDivElement>(null);
  const oppMonsterRowRef = useRef<HTMLDivElement>(null);

  const myPlayer = mySide === "player1" ? state.player1 : state.player2;
  const opponentSide: PlayerSide = mySide === "player1" ? "player2" : "player1";
  const opponentPlayer = mySide === "player1" ? state.player2 : state.player1;
  const isMyTurn = state.activePlayer === mySide;
  const phase = state.phase;

  // In engine mode, disable all click-zone interactions — actions come from the panel
  const canAct = !engineMode && isMyTurn && !state.winner;
  const canSummon =
    canAct &&
    (phase === "MAIN1" || phase === "MAIN2") &&
    !myPlayer.hasNormalSummoned;
  const canAttack = canAct && phase === "BATTLE";
  const canSetSpell = canAct && (phase === "MAIN1" || phase === "MAIN2");

  function closeMenu() {
    setContextMenu(null);
  }

  function clearSelection() {
    setSelection({ type: "none" });
  }

  // ─── Attack animation helper ─────────────────────────────────────────────────
  // Card width = 100px, gap = 44px → each column is 144px wide.
  // Zone i center x = rowLeft + i * 144 + 50.
  // Zone center y = rowTop + rowHeight / 2.

  const ZONE_COL_WIDTH = 144; // 100px card + 44px gap
  const ZONE_CARD_HALF = 50; // half of 100px card width
  const ZONE_CARD_H = 140; // card height

  function zoneRect(
    rowRef: React.RefObject<HTMLDivElement | null>,
    zoneIndex: number
  ): DOMRect {
    const row = rowRef.current;
    if (!row)
      return new DOMRect(
        window.innerWidth / 2,
        window.innerHeight / 2,
        100,
        140
      );
    const rowRect = row.getBoundingClientRect();
    const x = rowRect.left + zoneIndex * ZONE_COL_WIDTH;
    const y = rowRect.top;
    return new DOMRect(x, y, ZONE_CARD_HALF * 2, ZONE_CARD_H);
  }

  function fireAttack(
    attackerZone: number,
    targetZone: number | null,
    action: GameAction
  ) {
    const fromRect = zoneRect(myMonsterRowRef, attackerZone);
    const toRect =
      targetZone !== null ? zoneRect(oppMonsterRowRef, targetZone) : null;
    setAttackAnim({ fromRect, toRect, pendingAction: action });
    clearSelection();
  }

  // ─── LP damage detection ────────────────────────────────────────────────────
  useEffect(() => {
    const myLp = myPlayer.lifePoints;
    const oppLp = opponentPlayer.lifePoints;
    const prevMy = prevMyLp.current;
    const prevOpp = prevOppLp.current;
    if (prevMy !== null && myLp < prevMy) {
      setLpFlash(f => ({ ...f, my: true }));
      setTimeout(() => setLpFlash(f => ({ ...f, my: false })), 600);
    }
    if (prevOpp !== null && oppLp < prevOpp) {
      setLpFlash(f => ({ ...f, opp: true }));
      setTimeout(() => setLpFlash(f => ({ ...f, opp: false })), 600);
    }
    prevMyLp.current = myLp;
    prevOppLp.current = oppLp;
  }, [myPlayer.lifePoints, opponentPlayer.lifePoints]);

  // ─── Summon animation helper ─────────────────────────────────────────────────
  function fireSummon(
    zoneIndex: number,
    rowRef: React.RefObject<HTMLDivElement | null>,
    kind: "normal" | "special",
    action: GameAction
  ) {
    const rect = zoneRect(rowRef, zoneIndex);
    setSummonAnim({ zoneRect: rect, kind, pendingAction: action });
    clearSelection();
  }

  // ─── Hand card click ────────────────────────────────────────────────────────

  function handleHandCardClick(index: number, card: GameCard) {
    selectCardForDetail(card, "mine", "hand", index);
    if (!canAct) return;
    if (selection.type === "hand" && selection.index === index) {
      clearSelection();
      return;
    }
    setSelection({ type: "hand", index, card });
  }

  function handleHandCardContext(
    e: React.MouseEvent,
    index: number,
    card: GameCard
  ) {
    e.preventDefault();
    if (!canAct) return;

    const isMonster = card.type?.includes("Monster");
    const isSpell = card.type?.includes("Spell");
    const isTrap = card.type?.includes("Trap");
    const level = card.level ?? 1;
    const needed = requiredTributes(level);

    const items = [];

    if (isMonster && canSummon) {
      if (needed === 0) {
        items.push({
          label: "⚔ Normal Summon",
          action: () => setSelection({ type: "hand", index, card }),
        });
        items.push({
          label: "🛡 Set Face-Down",
          action: () => setSelection({ type: "hand", index, card }),
        });
      } else {
        items.push({
          label: `⚔ Tribute Summon (${needed} tributes)`,
          action: () => {
            const myMonsters = myPlayer.monsterZones
              .map((z, i) => ({ z, i }))
              .filter(x => x.z !== null);
            if (myMonsters.length < needed) {
              alert(`You need ${needed} monster(s) to tribute.`);
              return;
            }
            setSelection({
              type: "tribute",
              zones: [],
              handIndex: index,
              needed,
            });
          },
        });
      }
    }

    if ((isSpell || isTrap) && canSetSpell) {
      const isFieldSpell = isSpell && card.race === "Field";
      if (isFieldSpell) {
        items.push({
          label: "🌐 Activate Field Spell",
          action: () => {
            onAction({ type: "PLAY_FIELD_SPELL", handIndex: index });
            clearSelection();
          },
        });
      } else if (isSpell) {
        items.push({
          label: "✨ Activate Spell",
          action: () => setSelection({ type: "hand", index, card }),
        });
        items.push({
          label: "🃏 Set Face-Down",
          action: () => setSelection({ type: "hand", index, card }),
        });
      } else {
        items.push({
          label: "🃏 Set Face-Down",
          action: () => setSelection({ type: "hand", index, card }),
        });
      }
    }

    if (items.length > 0) {
      setContextMenu({ x: e.clientX, y: e.clientY, items });
    }
  }

  // ─── Monster zone click ─────────────────────────────────────────────────────

  function handleMyMonsterZoneClick(zoneIndex: number) {
    const slot = myPlayer.monsterZones[zoneIndex];
    if (slot) selectCardForDetail(slot.card, "mine", "mzone", zoneIndex);
    // Tribute selection mode
    if (selection.type === "tribute") {
      if (!slot) return;
      const already = selection.zones.includes(zoneIndex);
      const newZones = already
        ? selection.zones.filter(z => z !== zoneIndex)
        : [...selection.zones, zoneIndex];
      setSelection({ ...selection, zones: newZones });
      return;
    }

    if (!canAct) return;

    // Place card from hand
    if (selection.type === "hand") {
      const card = selection.card;
      const isMonster = card.type?.includes("Monster");

      if (isMonster && canSummon) {
        if (slot !== null) return; // occupied
        // Show summon options
        setContextMenu({
          x: window.innerWidth / 2,
          y: window.innerHeight / 2,
          items: [
            {
              label: "⚔ Normal Summon (ATK)",
              action: () => {
                fireSummon(zoneIndex, myMonsterRowRef, "normal", {
                  type: "SUMMON_MONSTER",
                  handIndex: selection.index,
                  zoneIndex,
                });
              },
            },
            {
              label: "🛡 Set (DEF face-down)",
              action: () => {
                onAction({
                  type: "SET_MONSTER",
                  handIndex: selection.index,
                  zoneIndex,
                });
                clearSelection();
              },
            },
          ],
        });
        return;
      }
      clearSelection();
      return;
    }

    // Attack declaration
    if (canAttack && slot && !slot.faceDown && slot.position === "ATK") {
      setSelection({ type: "attacker", zone: zoneIndex });
      return;
    }

    // Context menu for existing monster
    if (slot && isMyTurn) {
      const items = [];
      if (canAct && (phase === "MAIN1" || phase === "MAIN2") && slot.faceDown) {
        items.push({
          label: "🔄 Flip Summon",
          action: () => onAction({ type: "CHANGE_POSITION", zoneIndex }),
        });
      }
      if (canAttack && !slot.faceDown && slot.position === "ATK") {
        items.push({
          label: "⚔ Declare Attack",
          action: () => setSelection({ type: "attacker", zone: zoneIndex }),
        });
        const hasOpponentMonsters = opponentPlayer.monsterZones.some(
          z => z !== null
        );
        if (!hasOpponentMonsters) {
          items.push({
            label: "⚡ Direct Attack",
            action: () => {
              fireAttack(zoneIndex, null, {
                type: "DIRECT_ATTACK",
                attackerZone: zoneIndex,
              });
            },
          });
        }
      }
      if (
        canAct &&
        (phase === "MAIN1" || phase === "MAIN2") &&
        !slot.faceDown
      ) {
        items.push({
          label: "🔄 Change Position",
          action: () => onAction({ type: "CHANGE_POSITION", zoneIndex }),
        });
      }
      // Activate Effect — available for Effect Monsters and other non-Normal monsters
      if (canAct && !slot.faceDown && slot.card.type !== "Normal Monster") {
        items.push({
          label: "✨ Activate Effect",
          action: () =>
            onAction({
              type: "ACTIVATE_MONSTER_EFFECT",
              zoneIndex,
              zoneType: "monster",
            }),
          color: "var(--neon-cyan)",
        });
      }
      if (canAct) {
        items.push({
          label: "⚰ Send to Graveyard",
          action: () =>
            onAction({
              type: "SEND_TO_GRAVEYARD",
              zoneIndex,
              zoneType: "monster",
            }),
          color: "var(--neon-pink)",
        });
        items.push({
          label: "✦ Banish",
          action: () =>
            onAction({ type: "BANISH_CARD", zoneIndex, zoneType: "monster" }),
          color: "#b44fff",
        });
      }
      if (items.length > 0) {
        setContextMenu({
          x: window.innerWidth / 2 - 80,
          y: window.innerHeight / 2,
          items,
        });
      }
    }
  }

  function handleOpponentMonsterZoneClick(zoneIndex: number) {
    const oppSlot = opponentPlayer.monsterZones[zoneIndex];
    // Allow detail view for face-up cards, or face-down cards in open-cards mode
    if (oppSlot && canShowOppDetail(oppSlot))
      selectCardForDetail(oppSlot.card, "opp", "mzone", zoneIndex);
    if (selection.type === "attacker") {
      fireAttack(selection.zone, zoneIndex, {
        type: "DECLARE_ATTACK",
        attackerZone: selection.zone,
        targetZone: zoneIndex,
        targetSide: opponentSide,
      });
    }
  }

  // ─── Spell/Trap zone click ──────────────────────────────────────────────────

  function handleMySpellTrapZoneClick(zoneIndex: number) {
    const slot = myPlayer.spellTrapZones[zoneIndex];

    // Always show card details for own cards (face-down included — player knows their own set cards)
    if (slot) selectCardForDetail(slot.card, "mine", "szone", zoneIndex);

    if (selection.type === "hand") {
      if (!canAct) return;
      const card = selection.card;
      const isSpell = card.type?.includes("Spell");
      const isTrap = card.type?.includes("Trap");

      if ((isSpell || isTrap) && canSetSpell && slot === null) {
        if (isSpell) {
          setContextMenu({
            x: window.innerWidth / 2,
            y: window.innerHeight / 2,
            items: [
              {
                label: "✨ Activate Spell",
                action: () => {
                  onAction({
                    type: "ACTIVATE_SPELL",
                    handIndex: selection.index,
                    zoneIndex,
                  });
                  clearSelection();
                },
              },
              {
                label: "🃏 Set Face-Down",
                action: () => {
                  onAction({
                    type: "SET_SPELL_TRAP",
                    handIndex: selection.index,
                    zoneIndex,
                  });
                  clearSelection();
                },
              },
            ],
          });
        } else {
          onAction({
            type: "SET_SPELL_TRAP",
            handIndex: selection.index,
            zoneIndex,
          });
          clearSelection();
        }
      }
      return;
    }

    // Context menu for existing spell/trap card
    if (slot && isMyTurn) {
      const items: { label: string; action: () => void; color?: string }[] = [];

      if (canAct) {
        items.push({
          label: slot.faceDown ? "✨ Activate (Flip)" : "✨ Activate",
          action: () => onAction({ type: "ACTIVATE_SET_CARD", zoneIndex }),
        });
      }

      if (canAct) {
        items.push({
          label: "⚰ Send to Graveyard",
          action: () =>
            onAction({
              type: "SEND_TO_GRAVEYARD",
              zoneIndex,
              zoneType: "spell_trap",
            }),
          color: "var(--neon-pink)",
        });
        items.push({
          label: "✦ Banish",
          action: () =>
            onAction({
              type: "BANISH_CARD",
              zoneIndex,
              zoneType: "spell_trap",
            }),
          color: "#b44fff",
        });
      }

      if (items.length > 0) {
        setContextMenu({
          x: window.innerWidth / 2 - 80,
          y: window.innerHeight / 2,
          items,
        });
      }
    }
  }

  // ─── Extra Monster Zone click ────────────────────────────────────────────────

  function handleMyEMZClick(slotIndex: number, e: React.MouseEvent) {
    e.stopPropagation();
    const slot = myPlayer.extraMonsterZones[slotIndex] ?? null;
    if (slot) selectCardForDetail(slot.card, "mine", "emz", slotIndex);

    const anyOccupied = myPlayer.extraMonsterZones.some(s => s !== null);
    if (selection.type === "hand" && !anyOccupied) {
      if (
        isExtraDeckType(selection.card.type) &&
        canAct &&
        (phase === "MAIN1" || phase === "MAIN2")
      ) {
        const animCol = slotIndex === 0 ? 1 : 3;
        fireSummon(animCol, myMonsterRowRef, "special", {
          type: "SUMMON_TO_EMZ",
          handIndex: selection.index,
        });
      }
      return;
    }

    if (slot && isMyTurn && canAct) {
      const items: { label: string; action: () => void; color?: string }[] = [];
      if (phase === "MAIN1" || phase === "MAIN2") {
        if (!slot.card.type?.includes("Link")) {
          items.push({
            label: "🔄 Change Position",
            action: () => onAction({ type: "CHANGE_POSITION_EMZ" }),
          });
        }
      }
      if (canAttack && !slot.faceDown && slot.position === "ATK") {
        items.push({
          label: "⚔ Declare Attack",
          action: () => setSelection({ type: "attacker", zone: -1 }),
        });
        const hasOpponentMonsters =
          opponentPlayer.monsterZones.some(z => z !== null) ||
          opponentPlayer.extraMonsterZones.some(s => s !== null);
        if (!hasOpponentMonsters) {
          items.push({
            label: "⚡ Direct Attack",
            action: () =>
              fireAttack(-1, null, { type: "DIRECT_ATTACK", attackerZone: -1 }),
          });
        }
      }
      items.push({
        label: "⚰ Send to Graveyard",
        action: () => onAction({ type: "SEND_EMZ_TO_GRAVEYARD" }),
        color: "var(--neon-pink)",
      });
      items.push({
        label: "✦ Banish",
        action: () => onAction({ type: "BANISH_EMZ_CARD" }),
        color: "#b44fff",
      });
      if (items.length > 0) {
        setContextMenu({ x: e.clientX, y: e.clientY, items });
      }
    }
  }

  function renderEMZSlot(
    slot: FieldCard | null,
    canPlace: boolean,
    isDetailSelected: boolean,
    isActionableSlot: boolean,
    onClick: (e: React.MouseEvent) => void
  ) {
    return (
      <CardZone
        slot={slot}
        variant="extra-monster"
        isDetailSelected={isDetailSelected}
        isActionable={isActionableSlot}
        canPlace={canPlace}
        onClick={onClick}
        badge={EMZ_BADGE}
        emptyContent={EMZ_EMPTY}
      />
    );
  }

  /** Resolve a physical EMZ slot: who owns it and what locator seq to use.
   *  mySeq/oppSeq are indices into extraMonsterZones (0=seq5, 1=seq6). */
  function resolveEMZ(
    mySeq: number,
    oppSeq: number
  ): { slot: FieldCard | null; side: "mine" | "opp"; seq: number } {
    const mine = myPlayer.extraMonsterZones[mySeq] ?? null;
    if (mine) return { slot: mine, side: "mine", seq: mySeq };
    const opp = opponentPlayer.extraMonsterZones[oppSeq] ?? null;
    if (opp) return { slot: opp, side: "opp", seq: oppSeq };
    return { slot: null, side: "mine", seq: mySeq };
  }

  function renderEMZRow() {
    // Physical EMZ positions (from screen / bottom-player perspective):
    //   col 1 (screen-left):  my seq5 OR opp seq6
    //   col 3 (screen-right): my seq6 OR opp seq5
    const left = resolveEMZ(0, 1);
    const right = resolveEMZ(1, 0);

    const anyMineOccupied = myPlayer.extraMonsterZones.some(s => s !== null);
    const canPlaceMine =
      selection.type === "hand" &&
      !anyMineOccupied &&
      canAct &&
      (phase === "MAIN1" || phase === "MAIN2") &&
      isExtraDeckType(selection.card.type);

    const CARD_W = 100;
    const GAP = 44;

    function handleEMZClick(resolved: typeof left) {
      return (e: React.MouseEvent) => {
        e.stopPropagation();
        if (resolved.side === "mine") {
          handleMyEMZClick(resolved.seq, e);
        } else {
          if (resolved.slot && canShowOppDetail(resolved.slot))
            selectCardForDetail(resolved.slot.card, "opp", "emz", resolved.seq);
          if (selection.type === "attacker") {
            fireAttack(selection.zone, null, {
              type: "DECLARE_ATTACK",
              attackerZone: selection.zone,
              targetZone: -2,
              targetSide: opponentSide,
            });
          }
        }
      };
    }

    return (
      <div className="flex justify-center">
        <div className="board-center-row">
          <div style={{ width: "212px", flexShrink: 0 }} />

          <div className="flex" style={{ gap: `${GAP}px` }}>
            <div style={{ width: `${CARD_W}px`, flexShrink: 0 }} />

            {renderEMZSlot(
              left.slot,
              canPlaceMine,
              isLocatorMatch(left.slot?.card.id, left.side, "emz", left.seq),
              isActionable(left.slot?.card.id, left.side, "emz", left.seq),
              handleEMZClick(left)
            )}

            <div style={{ width: `${CARD_W}px`, flexShrink: 0 }} />

            {renderEMZSlot(
              right.slot,
              false,
              isLocatorMatch(right.slot?.card.id, right.side, "emz", right.seq),
              isActionable(right.slot?.card.id, right.side, "emz", right.seq),
              handleEMZClick(right)
            )}

            <div style={{ width: `${CARD_W}px`, flexShrink: 0 }} />
          </div>

          <div style={{ width: "212px", flexShrink: 0 }} />
        </div>
      </div>
    );
  }

  // ─── Tribute confirm ────────────────────────────────────────────────────────

  function confirmTribute(isSummon: boolean) {
    if (selection.type !== "tribute") return;
    if (selection.zones.length < selection.needed) return;

    if (isSummon) {
      // Find first empty monster zone
      const emptyZone = myPlayer.monsterZones.findIndex(z => z === null);
      if (emptyZone === -1) return;
      fireSummon(emptyZone, myMonsterRowRef, "normal", {
        type: "SUMMON_MONSTER",
        handIndex: selection.handIndex,
        zoneIndex: emptyZone,
        tributeZones: selection.zones,
      });
    } else {
      const emptyZone = myPlayer.monsterZones.findIndex(z => z === null);
      if (emptyZone === -1) return;
      onAction({
        type: "SET_MONSTER",
        handIndex: selection.handIndex,
        zoneIndex: emptyZone,
        tributeZones: selection.zones,
      });
    }
    clearSelection();
  }

  // ─── Render helpers ─────────────────────────────────────────────────────────

  /**
   * Returns the monster row and spell/trap row as separate elements so the
   * parent can wrap each in its own board-center-row with different flanks.
   */
  function renderZoneRows(player: PlayerState, side: "mine" | "opponent") {
    const isMine = side === "mine";

    const monsterRow = (
      // Gap must be ≥ 40px so adjacent face-down DEF cards (140px wide when rotated) don't overlap.
      // 44px gives 4px clearance on each side.
      <div
        ref={isMine ? myMonsterRowRef : oppMonsterRowRef}
        className="flex"
        style={{ gap: "44px" }}
      >
        {player.monsterZones.map((slot, i) => {
          const isAttacker =
            selection.type === "attacker" && isMine && selection.zone === i;
          const isTributeSelected =
            selection.type === "tribute" &&
            isMine &&
            selection.zones.includes(i);
          const isValidTarget =
            selection.type === "attacker" && !isMine && slot !== null;
          const canTribute =
            selection.type === "tribute" && isMine && slot !== null;
          const canPlaceMonster =
            selection.type === "hand" &&
            isMine &&
            slot === null &&
            selection.card.type?.includes("Monster") &&
            canSummon;
          return (
            <CardZone
              key={i}
              slot={slot}
              variant="monster"
              size="md"
              emptyContent={MONSTER_EMPTY}
              isSelected={isAttacker || isTributeSelected}
              isDetailSelected={isLocatorMatch(
                slot?.card.id,
                isMine ? "mine" : "opp",
                "mzone",
                i
              )}
              isActionable={isActionable(
                slot?.card.id,
                isMine ? "mine" : "opp",
                "mzone",
                i
              )}
              isValidTarget={isValidTarget}
              canPlace={canTribute || canPlaceMonster}
              isOpponent={!isMine}
              onClick={() =>
                isMine
                  ? handleMyMonsterZoneClick(i)
                  : handleOpponentMonsterZoneClick(i)
              }
              onContextMenu={e => {
                e.preventDefault();
                if (isMine && slot) handleMyMonsterZoneClick(i);
              }}
            />
          );
        })}
      </div>
    );

    const spellTrapRow = (
      // Gap matches Monster row (44px) so each S/T zone aligns directly below its Monster Zone column.
      <div className="flex" style={{ gap: "44px" }}>
        {player.spellTrapZones.map((slot, i) => {
          const canPlace =
            selection.type === "hand" &&
            isMine &&
            slot === null &&
            (selection.card.type?.includes("Spell") ||
              selection.card.type?.includes("Trap")) &&
            !(
              selection.card.type?.includes("Spell") &&
              selection.card.race === "Field"
            );
          return (
            <CardZone
              key={i}
              slot={slot}
              variant="spell-trap"
              size="md"
              emptyContent={SPELL_TRAP_EMPTY}
              isDetailSelected={isLocatorMatch(
                slot?.card.id,
                isMine ? "mine" : "opp",
                "szone",
                i
              )}
              isActionable={isActionable(
                slot?.card.id,
                isMine ? "mine" : "opp",
                "szone",
                i
              )}
              canPlace={canPlace}
              isOpponent={!isMine}
              onClick={() => {
                if (isMine) {
                  handleMySpellTrapZoneClick(i);
                } else if (slot && canShowOppDetail(slot)) {
                  selectCardForDetail(slot.card, "opp", "szone", i);
                }
              }}
            />
          );
        })}
      </div>
    );

    return { monsterRow, spellTrapRow };
  }

  // ─── Field Zone ────────────────────────────────────────────────────────────

  function handleMyFieldZoneClick(e: React.MouseEvent) {
    e.stopPropagation();
    const fieldCard = myPlayer.fieldZone;
    if (fieldCard) selectCardForDetail(fieldCard.card, "mine", "field", 0);
    if (!canAct) return;

    // Place a Field Spell from hand
    if (selection.type === "hand") {
      const card = selection.card;
      if (
        card.type?.includes("Spell") &&
        card.race === "Field" &&
        canSetSpell
      ) {
        onAction({ type: "PLAY_FIELD_SPELL", handIndex: selection.index });
        clearSelection();
      }
      return;
    }

    // Context menu for existing field spell
    if (fieldCard && canAct) {
      setContextMenu({
        x: e.clientX,
        y: e.clientY,
        items: [
          {
            label: "⚰ Send to Graveyard",
            action: () => onAction({ type: "SEND_FIELD_TO_GY" }),
            color: "var(--neon-pink)",
          },
        ],
      });
    }
  }

  function renderFieldZone(player: PlayerState, side: "mine" | "opponent") {
    const isMine = side === "mine";
    const fieldCard = player.fieldZone;
    const canPlaceField =
      isMine &&
      selection.type === "hand" &&
      selection.card.type?.includes("Spell") &&
      selection.card.race === "Field" &&
      canSetSpell;
    const fieldSide = isMine ? "mine" : "opp";
    const isDetailSel = isLocatorMatch(
      fieldCard?.card.id,
      fieldSide,
      "field",
      0
    );
    const isFieldActionable = isActionable(
      fieldCard?.card.id,
      fieldSide,
      "field",
      0
    );

    const handleFieldClick = isMine
      ? handleMyFieldZoneClick
      : () => {
          if (fieldCard && canShowOppDetail(fieldCard))
            selectCardForDetail(fieldCard.card, "opp", "field", 0);
        };

    return (
      <CardZone
        slot={fieldCard}
        variant="field"
        isDetailSelected={isDetailSel}
        isActionable={isFieldActionable}
        canPlace={canPlaceField}
        onClick={handleFieldClick}
        emptyContent={FIELD_EMPTY}
      />
    );
  }

  function renderDeckZone(player: PlayerState, isMine = false) {
    const showDrawPrompt =
      isMine &&
      isMyTurn &&
      phase === "DRAW" &&
      player.deck.length > 0 &&
      !player.hasDrawn;
    return (
      <div
        className="relative rounded overflow-hidden cursor-pointer"
        title={
          showDrawPrompt
            ? "Click to draw a card"
            : `Deck: ${player.deck.length} cards`
        }
        style={{
          width: "100px",
          height: "140px",
          flexShrink: 0,
          border: showDrawPrompt
            ? "1px solid rgba(0,245,255,0.8)"
            : "1px solid rgba(0,245,255,0.35)",
          boxShadow: showDrawPrompt
            ? "0 0 14px rgba(0,245,255,0.7), 0 0 28px rgba(0,245,255,0.3)"
            : "0 0 6px rgba(0,245,255,0.15)",
          transition: "box-shadow 0.2s ease",
        }}
        onClick={
          showDrawPrompt ? () => onAction({ type: "DRAW_CARD" }) : undefined
        }
      >
        <div
          className="card-back w-full h-full"
          style={{ opacity: player.deck.length > 0 ? 1 : 0.2 }}
        />
        {showDrawPrompt && (
          <div
            className="draw-icon-blink"
            style={{
              position: "absolute",
              inset: 0,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: "clamp(1rem, 2.2vw, 1.6rem)",
              color: "var(--neon-cyan)",
              pointerEvents: "none",
            }}
          >
            ✦
          </div>
        )}
        <div
          className="absolute bottom-0 left-0 right-0 text-center font-bold"
          style={{
            fontSize: "0.65rem",
            paddingBlock: "2px",
            background: "rgba(0,0,0,0.8)",
            color: "var(--neon-cyan)",
          }}
        >
          {player.deck.length}
        </div>
      </div>
    );
  }

  function renderGraveyardZone(player: PlayerState, side: PlayerSide) {
    const relSide = side === mySide ? "mine" : "opp";
    const topCard = player.graveyard[player.graveyard.length - 1];
    const gyActionable = actionableZones.has(`${relSide}-grave`);
    return (
      <div
        className={`relative rounded overflow-hidden cursor-pointer${gyActionable ? " actionable" : ""}`}
        title={`Graveyard: ${player.graveyard.length} cards`}
        onClick={() => setZoneViewer({ side, tab: "graveyard" })}
        style={
          {
            width: "100px",
            height: "140px",
            flexShrink: 0,
            "--hl-rgb": "255 45 120",
            border: gyActionable ? undefined : "1px solid rgba(255,45,120,0.6)",
            background: "rgba(255,45,120,0.08)",
            boxShadow: gyActionable
              ? undefined
              : "0 0 6px rgba(255,45,120,0.2)",
          } as React.CSSProperties
        }
      >
        {topCard ? (
          <img
            src={`https://images.ygoprodeck.com/images/cards_small/${topCard.id}.jpg`}
            alt={topCard.name}
            className="w-full h-full object-cover opacity-70"
            onError={e => {
              (e.target as HTMLImageElement).style.display = "none";
            }}
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <span
              style={{
                color: "var(--neon-pink)",
                opacity: 0.6,
                filter: "drop-shadow(0 0 6px rgba(255,45,120,0.5))",
                display: "inline-flex",
              }}
            >
              <Skull size={22} strokeWidth={2} />
            </span>
          </div>
        )}
        <div
          className="absolute bottom-0 left-0 right-0 text-center font-bold"
          style={{
            fontSize: "0.65rem",
            paddingBlock: "2px",
            background: "rgba(0,0,0,0.8)",
            color: "var(--neon-pink)",
          }}
        >
          {player.graveyard.length}
        </div>
      </div>
    );
  }

  function renderBanishedZone(player: PlayerState, side: PlayerSide) {
    const relSide = side === mySide ? "mine" : "opp";
    const topBanished = player.banished[player.banished.length - 1];
    const banActionable = actionableZones.has(`${relSide}-banished`);
    return (
      <div
        className={`relative rounded overflow-hidden cursor-pointer${banActionable ? " actionable" : ""}`}
        title={`Banished: ${player.banished.length} cards`}
        onClick={() => setZoneViewer({ side, tab: "banished" })}
        style={
          {
            width: "100px",
            height: "140px",
            flexShrink: 0,
            "--hl-rgb": "180 79 255",
            border: banActionable
              ? undefined
              : "1px solid rgba(180,79,255,0.6)",
            background: "rgba(180,79,255,0.08)",
            boxShadow: banActionable
              ? undefined
              : "0 0 6px rgba(180,79,255,0.2)",
          } as React.CSSProperties
        }
      >
        {topBanished ? (
          <img
            src={`https://images.ygoprodeck.com/images/cards_small/${topBanished.id}.jpg`}
            alt={topBanished.name}
            className="w-full h-full object-cover opacity-70"
            style={{ filter: "hue-rotate(60deg) brightness(0.8)" }}
            onError={e => {
              (e.target as HTMLImageElement).style.display = "none";
            }}
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <span
              style={{
                color: "#b44fff",
                opacity: 0.6,
                filter: "drop-shadow(0 0 6px rgba(180,79,255,0.5))",
                display: "inline-flex",
              }}
            >
              <Ghost size={22} strokeWidth={2} />
            </span>
          </div>
        )}
        <div
          className="absolute bottom-0 left-0 right-0 text-center font-bold"
          style={{
            fontSize: "0.65rem",
            paddingBlock: "2px",
            background: "rgba(0,0,0,0.8)",
            color: "#b44fff",
          }}
        >
          {player.banished.length}
        </div>
      </div>
    );
  }

  function renderExtraDeckZone(player: PlayerState, side: PlayerSide) {
    const relSide = side === mySide ? "mine" : "opp";
    const extraActionable = actionableZones.has(`${relSide}-extra`);
    const count = player.extraDeck.length;
    return (
      <div
        className={`relative rounded overflow-hidden cursor-pointer${extraActionable ? " actionable" : ""}`}
        title={`Extra Deck: ${count} cards`}
        onClick={() => setZoneViewer({ side, tab: "extra" })}
        style={
          {
            width: "100px",
            height: "140px",
            flexShrink: 0,
            "--hl-rgb": "255 215 0",
            border: extraActionable
              ? undefined
              : "1px solid rgba(255,215,0,0.6)",
            background: "rgba(255,215,0,0.08)",
            boxShadow: extraActionable
              ? undefined
              : "0 0 6px rgba(255,215,0,0.2)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          } as React.CSSProperties
        }
      >
        <span
          style={{
            color: "#ffd700",
            opacity: count > 0 ? 0.85 : 0.3,
            filter: "drop-shadow(0 0 6px rgba(255,215,0,0.5))",
            display: "inline-flex",
          }}
        >
          <Star size={22} strokeWidth={2} />
        </span>
        <div
          className="absolute bottom-0 left-0 right-0 text-center font-bold"
          style={{
            fontSize: "0.65rem",
            paddingBlock: "2px",
            background: "rgba(0,0,0,0.8)",
            color: "#ffd700",
          }}
        >
          {count}
        </div>
      </div>
    );
  }

  // ─── Win overlay ────────────────────────────────────────────────────────────

  if (state.winner) {
    const iWon = state.winner === mySide;
    return (
      <div
        className="fixed inset-0 z-50 flex items-center justify-center"
        style={{ background: "rgba(0,0,0,0.9)" }}
      >
        <div className="text-center animate-slide-up">
          <div
            className="text-6xl font-black mb-4"
            style={{
              fontFamily: "'Orbitron', sans-serif",
              color: iWon ? "var(--neon-cyan)" : "var(--neon-pink)",
              textShadow: iWon
                ? "0 0 20px var(--neon-cyan), 0 0 60px var(--neon-cyan)"
                : "0 0 20px var(--neon-pink), 0 0 60px var(--neon-pink)",
            }}
          >
            {iWon ? "VICTORY" : "DEFEAT"}
          </div>
          <div
            className="text-lg opacity-60"
            style={{ color: "var(--text-secondary)" }}
          >
            {(() => {
              const log = visibleLog ?? state.log;
              return log[log.length - 1];
            })()}
          </div>
        </div>
      </div>
    );
  }

  // ─── Main render ────────────────────────────────────────────────────────────

  const oppRows = renderZoneRows(opponentPlayer, "opponent");
  const myRows = renderZoneRows(myPlayer, "mine");

  return (
    <div
      className="flex h-full w-full"
      style={{ background: "var(--bg-void)", overflow: "hidden" }}
      onClick={() => {
        if (selection.type !== "none") clearSelection();
      }}
    >
      {/* ── Field ── */}
      <div
        className="flex-1 flex flex-col"
        style={{ minWidth: 0, overflowY: "auto", position: "relative" }}
      >
        {/* Chain stack overlay */}
        <ChainWidget entries={state.pendingChain} />
        {/* Spacer top */}
        <div className="flex-1" />

        {/* Opponent area — compact, no stretching */}
        <div
          className="flex flex-col flex-shrink-0"
          style={{ borderBottom: "1px solid rgba(255,45,120,0.35)" }}
        >
          {/* Opponent info bar (LP only) */}
          <div className="flex items-center px-3 py-1.5 flex-shrink-0">
            <LifePoints
              name={opponentPlayer.name}
              lp={opponentPlayer.lifePoints}
              isActive={!isMyTurn}
              isOpponent
              flash={lpFlash.opp}
            />
          </div>

          {/* Opponent hand */}
          {(() => {
            const oppHand = opponentPlayer.hand;
            const oppPile = oppHand.length >= 10;
            const PILE_W = 932;
            const CARD_W = 100;
            const oppStep =
              oppPile && oppHand.length > 1
                ? Math.floor((PILE_W - CARD_W) / (oppHand.length - 1))
                : CARD_W;
            return (
              <div
                className="flex items-center justify-center px-2 py-1.5 flex-shrink-0"
                style={
                  oppPile
                    ? {
                        position: "relative",
                        height: "152px",
                        width: PILE_W,
                        margin: "0 auto",
                        overflow: "visible",
                      }
                    : { gap: "4px" }
                }
              >
                {oppHand.map((c, i) => (
                  <HandCard
                    key={i}
                    card={c as GameCard}
                    index={i}
                    isOpponentCard
                    pileMode={oppPile}
                    pileOffset={oppPile ? i * oppStep : undefined}
                    onClick={
                      openCards && (c.id || 0) > 0
                        ? () => {
                            setSelectedCardDetail(c as GameCard);
                            setSelectedLocator(null);
                          }
                        : undefined
                    }
                  />
                ))}
              </div>
            );
          })()}

          {/* Opponent zones — official mat layout (mirrored) */}
          <div
            className="py-1.5 flex-shrink-0 flex flex-col items-center"
            style={{ gap: "8px" }}
          >
            {/* S/T row (top for opponent) */}
            <div className="board-center-row">
              <div className="board-flank-left">
                {renderDeckZone(opponentPlayer)}
              </div>
              {oppRows.spellTrapRow}
              <div className="board-flank-right">
                {renderExtraDeckZone(opponentPlayer, opponentSide)}
              </div>
            </div>
            {/* Monster row (bottom, closer to center) */}
            <div className="board-center-row">
              <div className="board-flank-left">
                {renderBanishedZone(opponentPlayer, opponentSide)}
                {renderGraveyardZone(opponentPlayer, opponentSide)}
              </div>
              {oppRows.monsterRow}
              <div className="board-flank-right">
                {renderFieldZone(opponentPlayer, "opponent")}
              </div>
            </div>
          </div>
        </div>

        {/* Center divider: phase indicator + EMZ slots + restart button — all in one row */}
        <div
          className="flex-shrink-0 relative"
          style={{
            background: "rgba(0,0,0,0.5)",
            borderTop: "1px solid rgba(0,245,255,0.3)",
            borderBottom: "1px solid rgba(0,245,255,0.3)",
          }}
          onClick={e => e.stopPropagation()}
        >
          {/*
           * The EMZ slots (100×140px) define the row height.
           * The phase card and restart button are absolutely positioned
           * on the left and right edges, vertically centered.
           * The EMZ grid sits in the center using the same board-center-row
           * structure as the zone grids for pixel-perfect column alignment.
           */}

          {/* EMZ slots — centered, defines row height */}
          <div className="flex justify-center py-1">{renderEMZRow()}</div>

          {/* Phase indicator — absolute left, vertically centered */}
          <div
            className="absolute left-3 top-0 bottom-0 flex items-center"
            style={{ pointerEvents: "auto" }}
          >
            <div className="flex flex-col items-start gap-1">
              <PhaseIndicator
                phase={phase}
                isMyTurn={isMyTurn}
                onAdvance={() => onAction({ type: "ADVANCE_PHASE" })}
                turnNumber={state.turnNumber}
                activePlayerName={
                  state.activePlayer === mySide
                    ? "Your"
                    : opponentPlayer.name + "'s"
                }
              />

              {/* Tribute confirm */}
              {selection.type === "tribute" && (
                <div className="flex items-center gap-1 flex-wrap">
                  <span
                    className="text-[0.5rem]"
                    style={{
                      fontFamily: "'Orbitron', sans-serif",
                      color: "var(--neon-yellow)",
                    }}
                  >
                    SELECT {selection.needed - selection.zones.length} MORE
                  </span>
                  {selection.zones.length >= selection.needed && (
                    <>
                      <button
                        onClick={e => {
                          e.stopPropagation();
                          confirmTribute(true);
                        }}
                        className="px-2 py-0.5 text-[0.5rem] rounded"
                        style={{
                          background: "rgba(0,245,255,0.15)",
                          border: "1px solid var(--neon-cyan)",
                          color: "var(--neon-cyan)",
                          fontFamily: "'Orbitron', sans-serif",
                        }}
                      >
                        SUMMON
                      </button>
                      <button
                        onClick={e => {
                          e.stopPropagation();
                          confirmTribute(false);
                        }}
                        className="px-2 py-0.5 text-[0.5rem] rounded"
                        style={{
                          background: "rgba(245,230,66,0.1)",
                          border: "1px solid var(--neon-yellow)",
                          color: "var(--neon-yellow)",
                          fontFamily: "'Orbitron', sans-serif",
                        }}
                      >
                        SET
                      </button>
                      <button
                        onClick={e => {
                          e.stopPropagation();
                          clearSelection();
                        }}
                        className="px-2 py-0.5 text-[0.5rem] rounded"
                        style={{
                          background: "rgba(255,45,120,0.1)",
                          border: "1px solid var(--neon-pink)",
                          color: "var(--neon-pink)",
                          fontFamily: "'Orbitron', sans-serif",
                        }}
                      >
                        CANCEL
                      </button>
                    </>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Restart button — absolute right, vertically centered */}
          {onRestart && (
            <div className="absolute right-3 top-0 bottom-0 flex items-center">
              <button
                onClick={e => {
                  e.stopPropagation();
                  setShowRestartConfirm(true);
                }}
                className="px-2 py-0.5 text-[0.5rem] rounded opacity-70 hover:opacity-100 transition-opacity"
                style={{
                  border: "1px solid var(--neon-pink)",
                  color: "var(--neon-pink)",
                  fontFamily: "'Orbitron', sans-serif",
                }}
              >
                RESTART
              </button>
            </div>
          )}
        </div>

        {/* My area — compact, no stretching */}
        <div className="flex flex-col flex-shrink-0">
          {/* My zones — official mat layout */}
          <div
            className="py-1.5 flex-shrink-0 flex flex-col items-center"
            style={{ gap: "8px" }}
          >
            {/* Monster row */}
            <div className="board-center-row">
              <div className="board-flank-left">
                {renderFieldZone(myPlayer, "mine")}
              </div>
              {myRows.monsterRow}
              <div className="board-flank-right">
                {renderGraveyardZone(myPlayer, mySide)}
                {renderBanishedZone(myPlayer, mySide)}
              </div>
            </div>
            {/* S/T row */}
            <div className="board-center-row">
              <div className="board-flank-left">
                {renderExtraDeckZone(myPlayer, mySide)}
              </div>
              {myRows.spellTrapRow}
              <div className="board-flank-right">
                {renderDeckZone(myPlayer, true)}
              </div>
            </div>
          </div>

          {/* My hand */}
          {(() => {
            const myHand = myPlayer.hand;
            const myPile = myHand.length >= 10;
            const PILE_W = 932;
            const CARD_W = 100;
            const myStep =
              myPile && myHand.length > 1
                ? Math.floor((PILE_W - CARD_W) / (myHand.length - 1))
                : CARD_W;
            return (
              <div
                className="flex items-center justify-center px-2 py-2 flex-shrink-0"
                style={{
                  borderTop: "1px solid rgba(0,245,255,0.2)",
                  minHeight: "4.5rem",
                  ...(myPile
                    ? {
                        position: "relative",
                        height: "156px",
                        width: PILE_W,
                        margin: "0 auto",
                        overflow: "visible",
                      }
                    : { gap: "4px" }),
                }}
                onClick={e => e.stopPropagation()}
              >
                {myHand.map((card, i) => (
                  <HandCard
                    key={card.instanceId}
                    card={card}
                    index={i}
                    isSelected={
                      selection.type === "hand" && selection.index === i
                    }
                    isDetailSelected={isLocatorMatch(
                      card.id,
                      "mine",
                      "hand",
                      i
                    )}
                    isActionable={isActionable(card.id, "mine", "hand", i)}
                    pileMode={myPile}
                    pileOffset={myPile ? i * myStep : undefined}
                    onClick={() => handleHandCardClick(i, card)}
                    onContextMenu={e => handleHandCardContext(e, i, card)}
                  />
                ))}
                {myHand.length === 0 && (
                  <span
                    className="text-[0.55rem]"
                    style={{
                      fontFamily: "'Orbitron', sans-serif",
                      color: "var(--neon-cyan)",
                      opacity: 0.5,
                      letterSpacing: "0.1em",
                    }}
                  >
                    NO CARDS IN HAND
                  </span>
                )}
              </div>
            );
          })()}

          {/* My info bar (LP only) */}
          <div className="flex items-center px-3 py-1.5 flex-shrink-0">
            <LifePoints
              name={myPlayer.name}
              lp={myPlayer.lifePoints}
              isActive={isMyTurn}
              flash={lpFlash.my}
            />
          </div>
        </div>

        {/* Spacer bottom */}
        <div className="flex-1" />
      </div>

      {/* ── Right panel: card detail + log stacked ── */}
      <div
        className="flex-shrink-0 flex flex-col"
        style={{
          width: "clamp(180px, 20vw, 280px)",
          borderLeft: "1px solid rgba(0,245,255,0.25)",
          background: "rgba(0,0,0,0.4)",
          zIndex: 60,
        }}
        onClick={e => e.stopPropagation()}
      >
        {/* Card detail — top half */}
        <div
          className="flex flex-col"
          style={{
            flex: "0 0 45%",
            minHeight: 0,
            borderBottom: "1px solid rgba(0,245,255,0.25)",
          }}
        >
          {/* Header */}
          <div
            className="px-3 py-2 flex-shrink-0"
            style={{
              borderBottom: "1px solid rgba(0,245,255,0.15)",
              fontFamily: "'Orbitron', sans-serif",
              fontSize: "0.75rem",
              letterSpacing: "0.1em",
              color: "var(--neon-cyan)",
            }}
          >
            CARD DETAIL
          </div>
          {/* Content */}
          <div className="overflow-y-auto p-2" style={{ flex: "1 1 0" }}>
            {selectedCardDetail ? (
              <div className="flex flex-col gap-2">
                {/* Card image */}
                <div
                  className="w-full rounded overflow-hidden"
                  style={{
                    aspectRatio: "0.717",
                    background: "rgba(255,255,255,0.04)",
                  }}
                >
                  <img
                    src={`https://images.ygoprodeck.com/images/cards/${selectedCardDetail.id}.jpg`}
                    alt={selectedCardDetail.name}
                    className="w-full h-full object-contain"
                    onError={e => {
                      (e.target as HTMLImageElement).src =
                        "https://images.ygoprodeck.com/images/cards/back_high.jpg";
                    }}
                  />
                </div>
                {/* Card name */}
                <div
                  style={{
                    fontFamily: "'Orbitron', sans-serif",
                    fontSize: "0.75rem",
                    fontWeight: 700,
                    color: "#e8f4ff",
                    lineHeight: 1.4,
                    letterSpacing: "0.03em",
                  }}
                >
                  {selectedCardDetail.name}
                </div>
                {/* Type + attribute */}
                <div
                  style={{
                    fontFamily: "'Rajdhani', sans-serif",
                    fontSize: "0.82rem",
                    color: "#8aaec8",
                    lineHeight: 1.5,
                    fontWeight: 500,
                  }}
                >
                  {selectedCardDetail.type}
                  {selectedCardDetail.level
                    ? ` · ★${selectedCardDetail.level}`
                    : ""}
                  {selectedCardDetail.attribute
                    ? ` · ${selectedCardDetail.attribute}`
                    : ""}
                </div>
                {/* ATK / DEF */}
                {selectedCardDetail.type?.includes("Monster") && (
                  <div className="flex gap-3">
                    <span
                      style={{
                        fontFamily: "'Rajdhani', sans-serif",
                        fontSize: "0.82rem",
                        color: "var(--neon-cyan)",
                        fontWeight: 700,
                      }}
                    >
                      ATK/{selectedCardDetail.atk ?? "?"}
                    </span>
                    <span
                      style={{
                        fontFamily: "'Rajdhani', sans-serif",
                        fontSize: "0.82rem",
                        color: "var(--neon-yellow, #ffe066)",
                        fontWeight: 700,
                      }}
                    >
                      DEF/{selectedCardDetail.def ?? "?"}
                    </span>
                  </div>
                )}
                {/* Description */}
                <p
                  style={{
                    fontFamily: "'Rajdhani', sans-serif",
                    fontSize: "0.82rem",
                    color: "#8aaec8",
                    lineHeight: 1.6,
                    fontWeight: 400,
                  }}
                >
                  {selectedCardDetail.desc}
                </p>
              </div>
            ) : (
              <div
                className="h-full flex flex-col items-center justify-center gap-2 opacity-30"
                style={{
                  fontFamily: "'Orbitron', sans-serif",
                  fontSize: "0.65rem",
                  color: "var(--neon-cyan)",
                  letterSpacing: "0.1em",
                  textAlign: "center",
                }}
              >
                <div style={{ fontSize: "1.5rem" }}>🃏</div>
                <div>SELECT A CARD</div>
              </div>
            )}
          </div>
        </div>
        {/* Bottom half: tabbed actions / log */}
        <div className="flex-1 flex flex-col" style={{ minHeight: 0 }}>
          {engineMode ? (
            <>
              {/* Tab bar */}
              <div
                className="flex flex-shrink-0"
                style={{ borderBottom: "1px solid rgba(0,245,255,0.15)" }}
              >
                {(["actions", "log"] as const).map(tab => (
                  <button
                    key={tab}
                    onClick={() => setBottomTab(tab)}
                    className="flex-1 py-2 transition-all"
                    style={{
                      fontFamily: "'Orbitron', sans-serif",
                      fontSize: "0.75rem",
                      letterSpacing: "0.1em",
                      color:
                        bottomTab === tab
                          ? "var(--neon-cyan)"
                          : "var(--text-secondary)",
                      background:
                        bottomTab === tab
                          ? "rgba(0,245,255,0.06)"
                          : "transparent",
                      borderBottom:
                        bottomTab === tab
                          ? "2px solid var(--neon-cyan)"
                          : "2px solid transparent",
                      opacity: bottomTab === tab ? 1 : 0.5,
                      cursor: "pointer",
                      border: "none",
                      borderTop: "none",
                      borderLeft: "none",
                      borderRight: "none",
                    }}
                  >
                    {tab === "actions"
                      ? `ACTIONS (${engineActions?.length ?? 0})`
                      : "LOG"}
                  </button>
                ))}
              </div>
              {/* Tab content */}
              <div className="flex-1" style={{ minHeight: 0 }}>
                {bottomTab === "actions" &&
                engineActions &&
                engineActions.length > 0 &&
                onEngineAction ? (
                  <EnginePromptRouter
                    actions={engineActions}
                    prompt={enginePrompt ?? null}
                    onAction={onEngineAction}
                  />
                ) : (
                  <DuelLog
                    logs={visibleLog ?? state.log}
                    isReplaying={isReplaying}
                  />
                )}
              </div>
            </>
          ) : (
            <DuelLog logs={visibleLog ?? state.log} isReplaying={isReplaying} />
          )}
        </div>
      </div>

      {/* ── Overlays ── */}
      {contextMenu && (
        <ActionMenu
          items={contextMenu.items}
          x={contextMenu.x}
          y={contextMenu.y}
          onClose={closeMenu}
        />
      )}

      {zoneViewer && (
        <ZoneViewer
          graveyard={
            zoneViewer.side === mySide
              ? myPlayer.graveyard
              : opponentPlayer.graveyard
          }
          banished={
            zoneViewer.side === mySide
              ? myPlayer.banished
              : opponentPlayer.banished
          }
          extra={
            zoneViewer.side === mySide
              ? myPlayer.extraDeck
              : opponentPlayer.extraDeck
          }
          initialTab={zoneViewer.tab}
          playerName={
            zoneViewer.side === mySide ? myPlayer.name : opponentPlayer.name
          }
          onClose={() => setZoneViewer(null)}
          onCardSelect={card => {
            setSelectedCardDetail(card);
            setSelectedLocator(null);
          }}
          isCardActionable={(cardId, zone, seq) => {
            const side = zoneViewer.side === mySide ? "mine" : "opp";
            return isActionable(cardId, side, zone, seq);
          }}
        />
      )}

      {/* Restart confirmation dialog */}
      {showRestartConfirm && (
        <div
          className="fixed inset-0 flex items-center justify-center"
          style={{ background: "rgba(0,0,0,0.75)", zIndex: 2000 }}
          onClick={() => setShowRestartConfirm(false)}
        >
          <div
            className="rounded animate-slide-up flex flex-col items-center gap-4 px-8 py-6"
            style={{
              background: "var(--bg-panel)",
              border: "1px solid rgba(255,45,120,0.45)",
              boxShadow:
                "0 0 40px rgba(0,0,0,0.8), 0 0 24px rgba(255,45,120,0.15)",
              minWidth: "280px",
              maxWidth: "360px",
            }}
            onClick={e => e.stopPropagation()}
          >
            {/* Icon */}
            <div style={{ fontSize: "2rem", lineHeight: 1 }}>↺</div>

            {/* Title */}
            <div
              style={{
                fontFamily: "'Orbitron', sans-serif",
                fontSize: "0.75rem",
                color: "var(--neon-pink)",
                letterSpacing: "0.12em",
                textShadow: "0 0 10px rgba(255,45,120,0.6)",
                textAlign: "center",
              }}
            >
              RESTART?
            </div>

            {/* Body */}
            <div
              style={{
                fontFamily: "'Rajdhani', sans-serif",
                fontSize: "0.82rem",
                color: "#8aaec8",
                textAlign: "center",
                lineHeight: 1.5,
              }}
            >
              Start a new duel? Current progress will be lost.
            </div>

            {/* Buttons */}
            <div className="flex gap-3 w-full">
              <button
                className="flex-1 py-1.5 rounded transition-all hover:opacity-90"
                style={{
                  fontFamily: "'Orbitron', sans-serif",
                  fontSize: "0.6rem",
                  letterSpacing: "0.1em",
                  background: "rgba(255,45,120,0.15)",
                  border: "1px solid var(--neon-pink)",
                  color: "var(--neon-pink)",
                  boxShadow: "0 0 8px rgba(255,45,120,0.2)",
                }}
                onClick={() => {
                  setShowRestartConfirm(false);
                  onRestart?.();
                }}
              >
                YES, RESTART
              </button>
              <button
                className="flex-1 py-1.5 rounded transition-all hover:opacity-90"
                style={{
                  fontFamily: "'Orbitron', sans-serif",
                  fontSize: "0.6rem",
                  letterSpacing: "0.1em",
                  background: "rgba(0,245,255,0.08)",
                  border: "1px solid rgba(0,245,255,0.4)",
                  color: "var(--neon-cyan)",
                }}
                onClick={() => setShowRestartConfirm(false)}
              >
                KEEP PLAYING
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Card tooltip on hover (future enhancement) */}

      {/* Attack animation overlay */}
      {attackAnim && (
        <AttackAnimation
          from={attackAnim.fromRect}
          to={attackAnim.toRect}
          onDone={() => {
            const action = attackAnim.pendingAction;
            setAttackAnim(null);
            onAction(action);
          }}
        />
      )}

      {/* Summon animation overlay */}
      {summonAnim && (
        <SummonAnimation
          zoneRect={summonAnim.zoneRect}
          kind={summonAnim.kind}
          onDone={() => {
            const action = summonAnim.pendingAction;
            setSummonAnim(null);
            onAction(action);
          }}
        />
      )}
    </div>
  );
}
