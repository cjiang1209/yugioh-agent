import { useCallback, useEffect, useRef, useState } from "react";
import {
  DuelState,
  FieldCard,
  GameAction,
  GameCard,
  Phase,
  PlayerSide,
  PlayerState,
} from "../../../../shared/gameTypes";
import { ActionMenu } from "./ActionMenu";
import { CardTooltip } from "./CardTooltip";
import { CardZone, HandCard } from "./CardZone";
import { DuelLog } from "./DuelLog";
import { GraveyardViewer } from "./GraveyardViewer";
import { LifePoints } from "./LifePoints";
import { AttackAnimation } from "./AttackAnimation";
import { SummonAnimation } from "./SummonAnimation";
import { PhaseIndicator } from "./PhaseIndicator";
import { EngineActionPanel } from "./EngineActionPanel";
import type { EngineAction, EnginePrompt } from "../../../../shared/engineTypes";

interface DuelBoardProps {
  state: DuelState;
  mySide: PlayerSide;
  onAction: (action: GameAction) => void;
  engineMode?: boolean;
  engineActions?: EngineAction[];
  enginePrompt?: EnginePrompt | null;
  onEngineAction?: (actionIndex: number) => void;
}

type SelectionMode =
  | { type: "none" }
  | { type: "hand"; index: number; card: GameCard }
  | { type: "attacker"; zone: number }
  | { type: "tribute"; zones: number[]; handIndex: number; needed: number };

interface ContextMenuState {
  x: number;
  y: number;
  items: { label: string; action: () => void; color?: string; disabled?: boolean }[];
}

// Module-level cache for card descriptions fetched from YGOProDeck API
const descCache = new Map<number, string>();

function requiredTributes(level: number) {
  if (level <= 4) return 0;
  if (level <= 6) return 1;
  return 2;
}

export function DuelBoard({ state, mySide, onAction, engineMode, engineActions, enginePrompt, onEngineAction }: DuelBoardProps) {
  const [selection, setSelection] = useState<SelectionMode>({ type: "none" });
  const [contextMenu, setContextMenu] = useState<ContextMenuState | null>(null);
  const [bottomTab, setBottomTab] = useState<"actions" | "log">(engineMode ? "actions" : "log");
  const [selectedCardDetail, setSelectedCardDetail] = useState<GameCard | null>(null);
  const [graveyardViewer, setGraveyardViewer] = useState<{ side: PlayerSide; tab: "graveyard" | "banished" | "extra" } | null>(null);
  const [showSurrenderConfirm, setShowSurrenderConfirm] = useState(false);

  // Fetch card description from YGOProDeck API when a card is selected
  useEffect(() => {
    if (!selectedCardDetail || !selectedCardDetail.id || selectedCardDetail.desc) return;
    const id = selectedCardDetail.id;
    if (descCache.has(id)) {
      setSelectedCardDetail(prev => prev?.id === id ? { ...prev, desc: descCache.get(id)! } : prev);
      return;
    }
    fetch(`https://db.ygoprodeck.com/api/v7/cardinfo.php?id=${id}`)
      .then(r => r.json())
      .then(data => {
        const desc = data.data?.[0]?.desc ?? "";
        descCache.set(id, desc);
        setSelectedCardDetail(prev => prev?.id === id ? { ...prev, desc } : prev);
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
  const [lpFlash, setLpFlash] = useState<{ my: boolean; opp: boolean }>({ my: false, opp: false });
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
  const canSummon = canAct && (phase === "MAIN1" || phase === "MAIN2") && !myPlayer.hasNormalSummoned;
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
  const ZONE_CARD_HALF = 50;  // half of 100px card width
  const ZONE_CARD_H = 140;    // card height

  function zoneRect(rowRef: React.RefObject<HTMLDivElement | null>, zoneIndex: number): DOMRect {
    const row = rowRef.current;
    if (!row) return new DOMRect(window.innerWidth / 2, window.innerHeight / 2, 100, 140);
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
    const toRect = targetZone !== null ? zoneRect(oppMonsterRowRef, targetZone) : null;
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
      setLpFlash((f) => ({ ...f, my: true }));
      setTimeout(() => setLpFlash((f) => ({ ...f, my: false })), 600);
    }
    if (prevOpp !== null && oppLp < prevOpp) {
      setLpFlash((f) => ({ ...f, opp: true }));
      setTimeout(() => setLpFlash((f) => ({ ...f, opp: false })), 600);
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
    setSelectedCardDetail(card);
    if (!canAct) return;
    if (selection.type === "hand" && selection.index === index) {
      clearSelection();
      return;
    }
    setSelection({ type: "hand", index, card });
  }

  function handleHandCardContext(e: React.MouseEvent, index: number, card: GameCard) {
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
              .filter((x) => x.z !== null);
            if (myMonsters.length < needed) {
              alert(`You need ${needed} monster(s) to tribute.`);
              return;
            }
            setSelection({ type: "tribute", zones: [], handIndex: index, needed });
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
    if (slot) setSelectedCardDetail(slot.card);
    // Tribute selection modee
    if (selection.type === "tribute") {
      if (!slot) return;
      const already = selection.zones.includes(zoneIndex);
      const newZones = already
        ? selection.zones.filter((z) => z !== zoneIndex)
        : [...selection.zones, zoneIndex];
      setSelection({ ...selection, zones: newZones });
      return;
    }

    if (!canAct) return;

    // Place card from hand
    if (selection.type === "hand") {
      const card = selection.card;
      const isMonster = card.type?.includes("Monster");
      const isSpell = card.type?.includes("Spell");
      const isTrap = card.type?.includes("Trap");

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
                fireSummon(zoneIndex, myMonsterRowRef, "normal", { type: "SUMMON_MONSTER", handIndex: selection.index, zoneIndex });
              },
            },
            {
              label: "🛡 Set (DEF face-down)",
              action: () => {
                onAction({ type: "SET_MONSTER", handIndex: selection.index, zoneIndex });
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
        const hasOpponentMonsters = opponentPlayer.monsterZones.some((z) => z !== null);
        if (!hasOpponentMonsters) {
          items.push({
            label: "⚡ Direct Attack",
            action: () => {
              fireAttack(zoneIndex, null, { type: "DIRECT_ATTACK", attackerZone: zoneIndex });
            },
          });
        }
      }
      if (canAct && (phase === "MAIN1" || phase === "MAIN2") && !slot.faceDown) {
        items.push({
          label: "🔄 Change Position",
          action: () => onAction({ type: "CHANGE_POSITION", zoneIndex }),
        });
      }
      // Activate Effect — available for Effect Monsters and other non-Normal monsters
      if (canAct && !slot.faceDown && slot.card.type !== "Normal Monster") {
        items.push({
          label: "✨ Activate Effect",
          action: () => onAction({ type: "ACTIVATE_MONSTER_EFFECT", zoneIndex, zoneType: "monster" }),
          color: "var(--neon-cyan)",
        });
      }
      if (canAct) {
        items.push({
          label: "⚰ Send to Graveyard",
          action: () => onAction({ type: "SEND_TO_GRAVEYARD", zoneIndex, zoneType: "monster" }),
          color: "var(--neon-pink)",
        });
        items.push({
          label: "✦ Banish",
          action: () => onAction({ type: "BANISH_CARD", zoneIndex, zoneType: "monster" }),
          color: "#b44fff",
        });
      }
      if (items.length > 0) {
        setContextMenu({ x: window.innerWidth / 2 - 80, y: window.innerHeight / 2, items });
      }
    }
  }

  function handleOpponentMonsterZoneClick(zoneIndex: number) {
    const oppSlot = opponentPlayer.monsterZones[zoneIndex];
    // Never reveal face-down opponent cards
    if (oppSlot && !oppSlot.faceDown) setSelectedCardDetail(oppSlot.card);
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
    if (slot) setSelectedCardDetail(slot.card);

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
                  onAction({ type: "ACTIVATE_SPELL", handIndex: selection.index, zoneIndex });
                  clearSelection();
                },
              },
              {
                label: "🃏 Set Face-Down",
                action: () => {
                  onAction({ type: "SET_SPELL_TRAP", handIndex: selection.index, zoneIndex });
                  clearSelection();
                },
              },
            ],
          });
        } else {
          onAction({ type: "SET_SPELL_TRAP", handIndex: selection.index, zoneIndex });
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
          action: () => onAction({ type: "SEND_TO_GRAVEYARD", zoneIndex, zoneType: "spell_trap" }),
          color: "var(--neon-pink)",
        });
        items.push({
          label: "✦ Banish",
          action: () => onAction({ type: "BANISH_CARD", zoneIndex, zoneType: "spell_trap" }),
          color: "#b44fff",
        });
      }

      if (items.length > 0) {
        setContextMenu({ x: window.innerWidth / 2 - 80, y: window.innerHeight / 2, items });
      }
    }
  }

  // ─── Extra Monster Zone click ────────────────────────────────────────────────

  function handleMyEMZClick(e: React.MouseEvent) {
    e.stopPropagation();
    const slot = myPlayer.extraMonsterZone;
    if (slot) setSelectedCardDetail(slot.card);

    // Place Extra Deck card from hand
    if (selection.type === "hand" && !slot) {
      const card = selection.card;
      const isExtra =
        card.type?.includes("Fusion") ||
        card.type?.includes("Synchro") ||
        card.type?.includes("XYZ") ||
        card.type?.includes("Link");
      if (isExtra && canAct && (phase === "MAIN1" || phase === "MAIN2")) {
        // Use EMZ rect for the animation (col 1 of my monster row)
        fireSummon(1, myMonsterRowRef, "special", { type: "SUMMON_TO_EMZ", handIndex: selection.index });
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
          action: () => setSelection({ type: "attacker", zone: -1 }), // -1 = EMZ attacker
        });
        const hasOpponentMonsters =
          opponentPlayer.monsterZones.some((z) => z !== null) ||
          opponentPlayer.extraMonsterZone !== null;
        if (!hasOpponentMonsters) {
          items.push({
            label: "⚡ Direct Attack",
            action: () => {
              // EMZ direct attack: send as zone -1
              fireAttack(-1, null, { type: "DIRECT_ATTACK", attackerZone: -1 });
            },
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
    isMine: boolean,
    canPlace: boolean,
    onClick: (e: React.MouseEvent) => void
  ) {
    return (
      <div
        onClick={onClick}
        title={slot ? slot.card.name : "Extra Monster Zone"}
        style={{
          width: "100px",
          height: "140px",
          borderRadius: "0.3rem",
          overflow: "visible",
          border: canPlace
            ? "2px solid var(--neon-cyan)"
            : slot
            ? "2px solid rgba(255,215,0,0.8)"
            : "1px dashed rgba(255,215,0,0.4)",
          background: slot ? "transparent" : "rgba(255,215,0,0.04)",
          boxShadow: slot
            ? "0 0 12px rgba(255,215,0,0.5), inset 0 0 8px rgba(255,215,0,0.1)"
            : canPlace
            ? "0 0 8px rgba(0,245,255,0.5)"
            : "0 0 6px rgba(255,215,0,0.15)",
          cursor: "pointer",
          flexShrink: 0,
          position: "relative",
        }}
      >
        {slot ? (
          <div style={{ width: "100%", height: "100%", position: "relative" }}>
            <img
              src={`https://images.ygoprodeck.com/images/cards_small/${slot.card.id}.jpg`}
              alt={slot.card.name}
              style={{
                width: slot.position === "DEF" ? "140px" : "100%",
                height: slot.position === "DEF" ? "100px" : "100%",
                objectFit: "cover",
                transform: slot.position === "DEF" ? "rotate(90deg) translateX(-20px) translateY(20px)" : "none",
                transformOrigin: "center center",
              }}
              onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
            />
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
          </div>
        ) : (
          <div className="w-full h-full flex flex-col items-center justify-center gap-0.5">
            <span style={{ fontSize: "1.2rem", opacity: 0.5 }}>★</span>
            <span
              style={{
                fontFamily: "'Orbitron', sans-serif",
                fontSize: "0.35rem",
                color: "rgba(255,215,0,0.7)",
                letterSpacing: "0.05em",
                textAlign: "center",
                lineHeight: 1.2,
              }}
            >
              EXTRA
              <br />
              MONSTER
            </span>
          </div>
        )}
      </div>
    );
  }

  function renderEMZRow() {
    const mySlot = myPlayer.extraMonsterZone;
    const oppSlot = opponentPlayer.extraMonsterZone;

    const canPlaceMine =
      selection.type === "hand" &&
      !mySlot &&
      canAct &&
      (phase === "MAIN1" || phase === "MAIN2") &&
      (selection.card.type?.includes("Fusion") ||
        selection.card.type?.includes("Synchro") ||
        selection.card.type?.includes("XYZ") ||
        selection.card.type?.includes("Link"));

    /**
     * Both player zone grids are centered in the same horizontal space.
     * The zone grid is a flex row of 5 cards (100px) with gap 44px.
     * Total grid width = 5*100 + 4*44 = 676px.
     *
     * From the viewer's perspective:
     *   - Opponent's zone grid: col 0 = leftmost, col 4 = rightmost
     *   - My zone grid: col 0 = leftmost (my left = screen left), col 4 = rightmost
     *     BUT my board is oriented so my "second zone from my left" = col 1 from my left
     *     = col 1 from screen left as well (same orientation).
     *
     * The two EMZ slots should be:
     *   - Opponent EMZ: at column 1 of the shared grid (second from left)
     *   - My EMZ: at column 3 of the shared grid (second from right = my col 1 from my right)
     *
     * We use a single 5-column flex row (same structure as the zone grid rows)
     * centered the same way as the zone grids, with the field-zone spacers on each side.
     * This guarantees pixel-perfect alignment at any viewport size.
     */
    const CARD_W = 100;
    const GAP = 44;

    const oppEMZClick = (e: React.MouseEvent) => {
      e.stopPropagation();
      if (oppSlot && !oppSlot.faceDown) setSelectedCardDetail(oppSlot.card);
      if (selection.type === "attacker") {
        fireAttack(selection.zone, null, {
          type: "DECLARE_ATTACK",
          attackerZone: selection.zone,
          targetZone: -2,
          targetSide: opponentSide,
        });
      }
    };

    return (
      // Wrap in the same board-center-row + field spacers as the zone grid rows
      // so horizontal centering is identical.
      <div className="flex justify-center">
        <div className="board-center-row">
          {/* field-left spacer (matches opponent's empty field-left) */}
          <div style={{ width: "100px", flexShrink: 0 }} />

          {/* 5-column grid:
               col 1 = my EMZ (2nd zone from my left = screen col 1)
               col 3 = opponent EMZ (2nd zone from opponent's left = screen col 3, mirrored)
          */}
          <div className="flex" style={{ gap: `${GAP}px` }}>
            {/* col 0: empty spacer */}
            <div style={{ width: `${CARD_W}px`, flexShrink: 0 }} />

            {/* col 1: my EMZ */}
            {renderEMZSlot(mySlot, true, canPlaceMine, handleMyEMZClick)}

            {/* col 2: empty spacer */}
            <div style={{ width: `${CARD_W}px`, flexShrink: 0 }} />

            {/* col 3: opponent EMZ */}
            {renderEMZSlot(oppSlot, false, false, oppEMZClick)}

            {/* col 4: empty spacer */}
            <div style={{ width: `${CARD_W}px`, flexShrink: 0 }} />
          </div>

          {/* field-right spacer (matches my empty field-right) */}
          <div style={{ width: "100px", flexShrink: 0 }} />
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
      const emptyZone = myPlayer.monsterZones.findIndex((z) => z === null);
      if (emptyZone === -1) return;
      fireSummon(emptyZone, myMonsterRowRef, "normal", {
        type: "SUMMON_MONSTER",
        handIndex: selection.handIndex,
        zoneIndex: emptyZone,
        tributeZones: selection.zones,
      });
    } else {
      const emptyZone = myPlayer.monsterZones.findIndex((z) => z === null);
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
   * Renders both monster row and spell/trap row in a single aligned grid,
   * with each column (0-4) sharing the same horizontal position.
   * The field zone is placed outside this grid.
   */
  function renderZoneGrid(player: PlayerState, side: "mine" | "opponent") {
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
          const isAttacker = selection.type === "attacker" && isMine && selection.zone === i;
          const isTributeSelected = selection.type === "tribute" && isMine && selection.zones.includes(i);
          const isValidTarget = selection.type === "attacker" && !isMine && slot !== null;
          const canTribute = selection.type === "tribute" && isMine && slot !== null;
          const canPlaceMonster =
            selection.type === "hand" && isMine && slot === null &&
            selection.card.type?.includes("Monster") &&
            canSummon;
          return (
            <CardZone
              key={i}
              slot={slot}
              label="MONSTER"
              size="md"
              isSelected={isAttacker || isTributeSelected}
              isValidTarget={isValidTarget}
              canPlace={canTribute || canPlaceMonster}
              isOpponent={!isMine}
              onClick={() => isMine ? handleMyMonsterZoneClick(i) : handleOpponentMonsterZoneClick(i)}
              onContextMenu={(e) => { e.preventDefault(); if (isMine && slot) handleMyMonsterZoneClick(i); }}
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
            selection.type === "hand" && isMine && slot === null &&
            (selection.card.type?.includes("Spell") || selection.card.type?.includes("Trap")) &&
            !(selection.card.type?.includes("Spell") && selection.card.race === "Field");
          return (
            <CardZone
              key={i}
              slot={slot}
              label="S/T"
              size="md"
              canPlace={canPlace}
              isOpponent={!isMine}
              onClick={() => {
                if (isMine) {
                  handleMySpellTrapZoneClick(i);
                } else if (slot && !slot.faceDown) {
                  setSelectedCardDetail(slot.card);
                }
              }}
            />
          );
        })}
      </div>
    );

    return (
      <div className="flex flex-col items-center" style={{ gap: "8px" }}>
        {/* Opponent: S/T on top, Monster below. Player: Monster on top, S/T below. */}
        {isMine ? monsterRow : spellTrapRow}
        {isMine ? spellTrapRow : monsterRow}
      </div>
    );
  }

  // ─── Field Zone ────────────────────────────────────────────────────────────

  function handleMyFieldZoneClick(e: React.MouseEvent) {
    e.stopPropagation();
    if (!canAct) return;
    const fieldCard = myPlayer.fieldZone;

    // Place a Field Spell from hand
    if (selection.type === "hand") {
      const card = selection.card;
      if (card.type?.includes("Spell") && card.race === "Field" && canSetSpell) {
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

    return (
      <div
        onClick={isMine ? handleMyFieldZoneClick : undefined}
        title={fieldCard ? fieldCard.card.name : "Field Zone"}
        style={{
          width: "100px",
          height: "140px",
          borderRadius: "0.3rem",
          overflow: "hidden",
          border: canPlaceField
            ? "2px solid var(--neon-cyan)"
            : fieldCard
            ? "2px solid rgba(0,245,255,0.7)"
            : "1px dashed rgba(0,245,255,0.3)",
          background: fieldCard
            ? "transparent"
            : "rgba(0,245,255,0.04)",
          boxShadow: fieldCard
            ? "0 0 10px rgba(0,245,255,0.4), inset 0 0 8px rgba(0,245,255,0.1)"
            : canPlaceField
            ? "0 0 8px rgba(0,245,255,0.5)"
            : "none",
          cursor: isMine ? "pointer" : "default",
          flexShrink: 0,
          position: "relative",
        }}
      >
        {fieldCard ? (
          <img
            src={`https://images.ygoprodeck.com/images/cards_small/${fieldCard.card.id}.jpg`}
            alt={fieldCard.card.name}
            className="w-full h-full object-cover"
            onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
          />
        ) : (
          <div className="w-full h-full flex flex-col items-center justify-center gap-0.5">
            <span style={{ fontSize: "clamp(0.7rem, 1.4vw, 1.2rem)", opacity: 0.5 }}>🌐</span>
            <span
              style={{
                fontFamily: "'Orbitron', sans-serif",
                fontSize: "clamp(0.3rem, 0.55vw, 0.45rem)",
                color: "var(--neon-cyan)",
                opacity: 0.6,
                letterSpacing: "0.05em",
                textAlign: "center",
                lineHeight: 1.2,
              }}
            >
              FIELD
            </span>
          </div>
        )}
      </div>
    );
  }

  function renderDeckZone(player: PlayerState, label: string, isMine = false) {
    const showDrawPrompt = isMine && isMyTurn && phase === "DRAW" && player.deck.length > 0 && !player.hasDrawn;
    return (
      <div
        className="flex flex-col items-center gap-0.5 cursor-pointer"
        title={showDrawPrompt ? "Click to draw a card" : `${label}: ${player.deck.length} cards`}
        style={{ position: "relative" }}
        onClick={showDrawPrompt ? () => onAction({ type: "DRAW_CARD" }) : undefined}
      >
        <div style={{ position: "relative" }}>
          <div
            className="card-back rounded"
            style={{
              width: "71px",
              height: "100px",
              opacity: player.deck.length > 0 ? 1 : 0.2,
              boxShadow: showDrawPrompt ? "0 0 14px rgba(0,245,255,0.7), 0 0 28px rgba(0,245,255,0.3)" : undefined,
              border: showDrawPrompt ? "1px solid rgba(0,245,255,0.8)" : undefined,
              transition: "box-shadow 0.2s ease",
            }}
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
        </div>
        <span
          style={{
            fontFamily: "'Share Tech Mono', monospace",
            fontSize: "clamp(0.55rem, 1.1vw, 0.85rem)",
            fontWeight: 700,
            color: "var(--neon-cyan)",
            textShadow: "0 0 6px rgba(0,245,255,0.8), 0 0 12px rgba(0,245,255,0.4)",
            letterSpacing: "0.05em",
          }}
        >
          {player.deck.length}
        </span>
      </div>
    );
  }

  function renderGraveyardZone(player: PlayerState, side: PlayerSide) {
    const topCard = player.graveyard[player.graveyard.length - 1];
    const topBanished = player.banished[player.banished.length - 1];
    return (
      <div className="flex items-end gap-1">
        {/* Graveyard */}
        <div
          className="flex flex-col items-center gap-0.5 cursor-pointer"
          title={`Graveyard: ${player.graveyard.length} cards`}
          onClick={() => setGraveyardViewer({ side, tab: "graveyard" })}
        >
          <div
            className="rounded overflow-hidden"
            style={{
              width: "71px",
              height: "100px",
              border: "1px solid rgba(255,45,120,0.6)",
              background: "rgba(255,45,120,0.08)",
              boxShadow: "0 0 6px rgba(255,45,120,0.2)",
            }}
          >
            {topCard ? (
              <img
                src={`https://images.ygoprodeck.com/images/cards_small/${topCard.id}.jpg`}
                alt={topCard.name}
                className="w-full h-full object-cover opacity-70"
                onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
              />
            ) : (
              <div className="w-full h-full flex items-center justify-center">
                <span style={{ color: "var(--neon-pink)", fontSize: "1rem", opacity: 0.6, textShadow: "0 0 6px rgba(255,45,120,0.5)" }}>⚰</span>
              </div>
            )}
          </div>
          <span
            style={{
              fontFamily: "'Share Tech Mono', monospace",
              fontSize: "clamp(0.55rem, 1.1vw, 0.85rem)",
              fontWeight: 700,
              color: "var(--neon-pink)",
              textShadow: "0 0 6px rgba(255,45,120,0.8), 0 0 12px rgba(255,45,120,0.4)",
              letterSpacing: "0.05em",
            }}
          >
            {player.graveyard.length}
          </span>
        </div>
        {/* Banished */}
        <div
          className="flex flex-col items-center gap-0.5 cursor-pointer"
          title={`Banished: ${player.banished.length} cards`}
          onClick={() => setGraveyardViewer({ side, tab: "banished" })}
        >
          <div
            className="rounded overflow-hidden"
            style={{
              width: "71px",
              height: "100px",
              border: "1px solid rgba(180,79,255,0.6)",
              background: "rgba(180,79,255,0.08)",
              boxShadow: "0 0 6px rgba(180,79,255,0.2)",
            }}
          >
            {topBanished ? (
              <img
                src={`https://images.ygoprodeck.com/images/cards_small/${topBanished.id}.jpg`}
                alt={topBanished.name}
                className="w-full h-full object-cover opacity-70"
                style={{ filter: "hue-rotate(60deg) brightness(0.8)" }}
                onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
              />
            ) : (
              <div className="w-full h-full flex items-center justify-center">
                <span style={{ color: "#b44fff", fontSize: "0.9rem", opacity: 0.6, textShadow: "0 0 6px rgba(180,79,255,0.5)" }}>✦</span>
              </div>
            )}
          </div>
          <span
            style={{
              fontFamily: "'Share Tech Mono', monospace",
              fontSize: "clamp(0.55rem, 1.1vw, 0.85rem)",
              fontWeight: 700,
              color: "#b44fff",
              textShadow: "0 0 6px rgba(180,79,255,0.8), 0 0 12px rgba(180,79,255,0.4)",
              letterSpacing: "0.05em",
            }}
          >
            {player.banished.length}
          </span>
        </div>
      </div>
    );
  }

  function renderExtraDeckZone(player: PlayerState, side: PlayerSide) {
    const count = player.extraDeck.length;
    return (
      <div
        className="flex flex-col items-center gap-0.5 cursor-pointer"
        title={`Extra Deck: ${count} cards`}
        onClick={() => setGraveyardViewer({ side, tab: "extra" })}
      >
        <div
          className="rounded overflow-hidden"
          style={{
            width: "71px",
            height: "100px",
            border: "1px solid rgba(255,215,0,0.6)",
            background: "rgba(255,215,0,0.08)",
            boxShadow: "0 0 6px rgba(255,215,0,0.2)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <span style={{ color: "#ffd700", fontSize: "1rem", opacity: count > 0 ? 0.85 : 0.3, textShadow: "0 0 6px rgba(255,215,0,0.5)" }}>★</span>
        </div>
        <span
          style={{
            fontFamily: "'Share Tech Mono', monospace",
            fontSize: "clamp(0.55rem, 1.1vw, 0.85rem)",
            fontWeight: 700,
            color: "#ffd700",
            textShadow: "0 0 6px rgba(255,215,0,0.8), 0 0 12px rgba(255,215,0,0.4)",
            letterSpacing: "0.05em",
          }}
        >
          {count}
        </span>
      </div>
    );
  }

  // ─── Win overlay ────────────────────────────────────────────────────────────

  if (state.winner) {
    const iWon = state.winner === mySide;
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center" style={{ background: "rgba(0,0,0,0.9)" }}>
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
          <div className="text-lg opacity-60" style={{ color: "var(--text-secondary)" }}>
            {state.log[state.log.length - 1]}
          </div>
        </div>
      </div>
    );
  }

  // ─── Main render ────────────────────────────────────────────────────────────

  return (
    <div
      className="flex h-full w-full"
      style={{ background: "var(--bg-void)", overflow: "hidden" }}
      onClick={() => {
        if (selection.type !== "none") clearSelection();
      }}
    >

      {/* ── Field ── */}
      <div className="flex-1 flex flex-col" style={{ minWidth: 0, overflowY: "auto" }}>

        {/* Spacer top */}
        <div className="flex-1" />

        {/* Opponent area — compact, no stretching */}
        <div
          className="flex flex-col flex-shrink-0"
          style={{ borderBottom: "1px solid rgba(255,45,120,0.35)" }}
        >
          {/* Opponent info bar */}
          <div className="flex items-center justify-between px-3 py-1.5 flex-shrink-0">
            <LifePoints
              name={opponentPlayer.name}
              lp={opponentPlayer.lifePoints}
              isActive={!isMyTurn}
              isOpponent
              flash={lpFlash.opp}
            />
            <div className="flex items-center gap-2">
              {renderExtraDeckZone(opponentPlayer, opponentSide)}
              {renderGraveyardZone(opponentPlayer, opponentSide)}
              {renderDeckZone(opponentPlayer, "Deck")}
            </div>
          </div>

          {/* Opponent hand */}
          {(() => {
            const oppHand = opponentPlayer.hand;
            const oppPile = oppHand.length >= 10;
            const PILE_W = 932; // fixed container width = 9-card spread (9×100 + 8×4)
            const CARD_W = 100;
            // Maximize spread: fill the full container width, tighten with more cards
            const oppStep = oppPile && oppHand.length > 1
              ? Math.floor((PILE_W - CARD_W) / (oppHand.length - 1))
              : CARD_W;
            return (
              <div
                className="flex items-center justify-center px-2 py-1.5 flex-shrink-0"
                style={oppPile
                  ? { position: "relative", height: "152px", width: PILE_W, margin: "0 auto", overflow: "visible" }
                  : { gap: "4px" }
                }
              >
                {oppHand.map((_, i) => (
                  <HandCard
                    key={i}
                    card={_ as GameCard}
                    index={i}
                    isOpponentCard
                    pileMode={oppPile}
                    pileOffset={oppPile ? i * oppStep : undefined}
                  />
                ))}
              </div>
            );
          })()}


          {/* Opponent zones — field zone RIGHT, grid CENTER */}
          <div className="py-1.5 flex-shrink-0 flex justify-center">
            <div className="board-center-row">
              <div className="board-field-left" />{/* empty spacer to mirror player side */}
              {renderZoneGrid(opponentPlayer, "opponent")}
              <div className="board-field-right">
                {renderFieldZone(opponentPlayer, "opponent")}
              </div>
            </div>
          </div>
        </div>

        {/* Center divider: phase indicator + EMZ slots + surrender button — all in one row */}
        <div
          className="flex-shrink-0 relative"
          style={{
            background: "rgba(0,0,0,0.5)",
            borderTop: "1px solid rgba(0,245,255,0.3)",
            borderBottom: "1px solid rgba(0,245,255,0.3)",
          }}
          onClick={(e) => e.stopPropagation()}
        >
          {/*
           * The EMZ slots (100×140px) define the row height.
           * The phase card and surrender button are absolutely positioned
           * on the left and right edges, vertically centered.
           * The EMZ grid sits in the center using the same board-center-row
           * structure as the zone grids for pixel-perfect column alignment.
           */}

          {/* EMZ slots — centered, defines row height */}
          <div className="flex justify-center py-1">
            {renderEMZRow()}
          </div>

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
                  state.activePlayer === mySide ? "Your" : opponentPlayer.name + "'s"
                }
              />

              {/* Tribute confirm */}
              {selection.type === "tribute" && (
                <div className="flex items-center gap-1 flex-wrap">
                  <span
                    className="text-[0.5rem]"
                    style={{ fontFamily: "'Orbitron', sans-serif", color: "var(--neon-yellow)" }}
                  >
                    SELECT {selection.needed - selection.zones.length} MORE
                  </span>
                  {selection.zones.length >= selection.needed && (
                    <>
                      <button
                        onClick={(e) => { e.stopPropagation(); confirmTribute(true); }}
                        className="px-2 py-0.5 text-[0.5rem] rounded"
                        style={{ background: "rgba(0,245,255,0.15)", border: "1px solid var(--neon-cyan)", color: "var(--neon-cyan)", fontFamily: "'Orbitron', sans-serif" }}
                      >
                        SUMMON
                      </button>
                      <button
                        onClick={(e) => { e.stopPropagation(); confirmTribute(false); }}
                        className="px-2 py-0.5 text-[0.5rem] rounded"
                        style={{ background: "rgba(245,230,66,0.1)", border: "1px solid var(--neon-yellow)", color: "var(--neon-yellow)", fontFamily: "'Orbitron', sans-serif" }}
                      >
                        SET
                      </button>
                      <button
                        onClick={(e) => { e.stopPropagation(); clearSelection(); }}
                        className="px-2 py-0.5 text-[0.5rem] rounded"
                        style={{ background: "rgba(255,45,120,0.1)", border: "1px solid var(--neon-pink)", color: "var(--neon-pink)", fontFamily: "'Orbitron', sans-serif" }}
                      >
                        CANCEL
                      </button>
                    </>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Surrender button — absolute right, vertically centered */}
          <div className="absolute right-3 top-0 bottom-0 flex items-center">
            <button
              onClick={(e) => {
                e.stopPropagation();
                setShowSurrenderConfirm(true);
              }}
              className="px-2 py-0.5 text-[0.5rem] rounded opacity-70 hover:opacity-100 transition-opacity"
              style={{ border: "1px solid var(--neon-pink)", color: "var(--neon-pink)", fontFamily: "'Orbitron', sans-serif" }}
            >
              SURRENDER
            </button>
          </div>
        </div>

        {/* My area — compact, no stretching */}
        <div className="flex flex-col flex-shrink-0">
          {/* My zones — field zone LEFT, grid CENTER */}
          <div className="py-1.5 flex-shrink-0 flex justify-center">
            <div className="board-center-row">
              <div className="board-field-left">
                {renderFieldZone(myPlayer, "mine")}
              </div>
              {renderZoneGrid(myPlayer, "mine")}
              <div className="board-field-right" />{/* empty spacer to mirror opponent side */}
            </div>
          </div>

          {/* My hand */}
          {(() => {
            const myHand = myPlayer.hand;
            const myPile = myHand.length >= 10;
            const PILE_W = 932; // fixed container width = 9-card spread (9×100 + 8×4)
            const CARD_W = 100;
            const myStep = myPile && myHand.length > 1
              ? Math.floor((PILE_W - CARD_W) / (myHand.length - 1))
              : CARD_W;
            return (
              <div
                className="flex items-center justify-center px-2 py-2 flex-shrink-0"
                style={{
                  borderTop: "1px solid rgba(0,245,255,0.2)",
                  minHeight: "4.5rem",
                  ...(myPile
                    ? { position: "relative", height: "156px", width: PILE_W, margin: "0 auto", overflow: "visible" }
                    : { gap: "4px" }
                  ),
                }}
                onClick={(e) => e.stopPropagation()}
              >
                {myHand.map((card, i) => (
                  <HandCard
                    key={card.instanceId}
                    card={card}
                    index={i}
                    isSelected={selection.type === "hand" && selection.index === i}
                    pileMode={myPile}
                    pileOffset={myPile ? i * myStep : undefined}
                    onClick={() => handleHandCardClick(i, card)}
                    onContextMenu={(e) => handleHandCardContext(e, i, card)}
                  />
                ))}
                {myHand.length === 0 && (
                  <span className="text-[0.55rem]" style={{ fontFamily: "'Orbitron', sans-serif", color: "var(--neon-cyan)", opacity: 0.5, letterSpacing: "0.1em" }}>
                    NO CARDS IN HAND
                  </span>
                )}
              </div>
            );
          })()}

          {/* My info bar */}
          <div className="flex items-center justify-between px-3 py-1.5 flex-shrink-0">
            <LifePoints
              name={myPlayer.name}
              lp={myPlayer.lifePoints}
              isActive={isMyTurn}
              flash={lpFlash.my}
            />
            <div className="flex items-center gap-2">
              {renderExtraDeckZone(myPlayer, mySide)}
              {renderGraveyardZone(myPlayer, mySide)}
              {renderDeckZone(myPlayer, "Deck", true)}
            </div>
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
        onClick={(e) => e.stopPropagation()}
      >
        {/* Card detail — top half */}
        <div className="flex flex-col" style={{ flex: "0 0 45%", minHeight: 0, borderBottom: "1px solid rgba(0,245,255,0.25)" }}>
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
                  style={{ aspectRatio: "0.717", background: "rgba(255,255,255,0.04)" }}
                >
                  <img
                    src={`https://images.ygoprodeck.com/images/cards/${selectedCardDetail.id}.jpg`}
                    alt={selectedCardDetail.name}
                    className="w-full h-full object-contain"
                    onError={(e) => {
                      (e.target as HTMLImageElement).src = "https://images.ygoprodeck.com/images/cards/back_high.jpg";
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
                <div style={{ fontFamily: "'Rajdhani', sans-serif", fontSize: "0.82rem", color: "#8aaec8", lineHeight: 1.5, fontWeight: 500 }}>
                  {selectedCardDetail.type}
                  {selectedCardDetail.level ? ` · ★${selectedCardDetail.level}` : ""}
                  {selectedCardDetail.attribute ? ` · ${selectedCardDetail.attribute}` : ""}
                </div>
                {/* ATK / DEF */}
                {selectedCardDetail.type?.includes("Monster") && (
                  <div className="flex gap-3">
                    <span style={{ fontFamily: "'Rajdhani', sans-serif", fontSize: "0.82rem", color: "var(--neon-cyan)", fontWeight: 700 }}>
                      ATK/{selectedCardDetail.atk ?? "?"}
                    </span>
                    <span style={{ fontFamily: "'Rajdhani', sans-serif", fontSize: "0.82rem", color: "var(--neon-yellow, #ffe066)", fontWeight: 700 }}>
                      DEF/{selectedCardDetail.def ?? "?"}
                    </span>
                  </div>
                )}
                {/* Description */}
                <p style={{ fontFamily: "'Rajdhani', sans-serif", fontSize: "0.82rem", color: "#8aaec8", lineHeight: 1.6, fontWeight: 400 }}>
                  {selectedCardDetail.desc}
                </p>
              </div>
            ) : (
              <div
                className="h-full flex flex-col items-center justify-center gap-2 opacity-30"
                style={{ fontFamily: "'Orbitron', sans-serif", fontSize: "0.65rem", color: "var(--neon-cyan)", letterSpacing: "0.1em", textAlign: "center" }}
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
                {(["actions", "log"] as const).map((tab) => (
                  <button
                    key={tab}
                    onClick={() => setBottomTab(tab)}
                    className="flex-1 py-2 transition-all"
                    style={{
                      fontFamily: "'Orbitron', sans-serif",
                      fontSize: "0.75rem",
                      letterSpacing: "0.1em",
                      color: bottomTab === tab ? "var(--neon-cyan)" : "var(--text-secondary)",
                      background: bottomTab === tab ? "rgba(0,245,255,0.06)" : "transparent",
                      borderBottom: bottomTab === tab ? "2px solid var(--neon-cyan)" : "2px solid transparent",
                      opacity: bottomTab === tab ? 1 : 0.5,
                      cursor: "pointer",
                      border: "none",
                      borderTop: "none",
                      borderLeft: "none",
                      borderRight: "none",
                    }}
                  >
                    {tab === "actions" ? `ACTIONS (${engineActions?.length ?? 0})` : "LOG"}
                  </button>
                ))}
              </div>
              {/* Tab content */}
              <div className="flex-1" style={{ minHeight: 0 }}>
                {bottomTab === "actions" && engineActions && onEngineAction ? (
                  <EngineActionPanel actions={engineActions} onAction={onEngineAction} />
                ) : (
                  <DuelLog logs={state.log} />
                )}
              </div>
            </>
          ) : (
            <DuelLog logs={state.log} />
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

      {graveyardViewer && (
        <GraveyardViewer
          graveyard={
            graveyardViewer.side === mySide
              ? myPlayer.graveyard
              : opponentPlayer.graveyard
          }
          banished={
            graveyardViewer.side === mySide
              ? myPlayer.banished
              : opponentPlayer.banished
          }
          extra={
            graveyardViewer.side === mySide
              ? myPlayer.extraDeck
              : opponentPlayer.extraDeck
          }
          initialTab={graveyardViewer.tab}
          playerName={
            graveyardViewer.side === mySide ? myPlayer.name : opponentPlayer.name
          }
          onClose={() => setGraveyardViewer(null)}
          onCardSelect={(card) => setSelectedCardDetail(card)}
        />
      )}

      {/* Surrender confirmation dialog */}
      {showSurrenderConfirm && (
        <div
          className="fixed inset-0 flex items-center justify-center"
          style={{ background: "rgba(0,0,0,0.75)", zIndex: 2000 }}
          onClick={() => setShowSurrenderConfirm(false)}
        >
          <div
            className="rounded animate-slide-up flex flex-col items-center gap-4 px-8 py-6"
            style={{
              background: "var(--bg-panel)",
              border: "1px solid rgba(255,45,120,0.45)",
              boxShadow: "0 0 40px rgba(0,0,0,0.8), 0 0 24px rgba(255,45,120,0.15)",
              minWidth: "280px",
              maxWidth: "360px",
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Icon */}
            <div style={{ fontSize: "2rem", lineHeight: 1 }}>🏳️</div>

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
              SURRENDER?
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
              Are you sure you want to concede this duel? This cannot be undone.
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
                  setShowSurrenderConfirm(false);
                  onAction({ type: "SURRENDER" });
                }}
              >
                YES, SURRENDER
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
                onClick={() => setShowSurrenderConfirm(false)}
              >
                KEEP FIGHTING
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
