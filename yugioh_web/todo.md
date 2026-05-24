# Yu-Gi-Oh! Duel Arena - TODO

## Foundation

- [x] Global cyberpunk styles (dark bg, neon pink/cyan, glow effects, geometric fonts)
- [x] App routing setup (Home lobby → Duel room)
- [x] Database schema: duel_rooms, duel_states tables (in-memory via Socket.IO)
- [x] Shared game types (card, zone, phase, game state)

## WebSocket Duel Server

- [x] Socket.IO server integration
- [x] Room creation and joining logic
- [x] Game state management on server
- [x] Event handlers: join_room, game_action
- [x] Broadcast game state to both players

## Duel Board UI

- [x] Duel board layout (opponent top, player bottom, mirrored)
- [x] Monster zones (5 slots each side)
- [x] Spell/Trap zones (5 slots each side)
- [x] Field zone (1 each side)
- [x] Extra deck zone
- [x] Main deck zone (with card count)
- [x] Graveyard zone (with card count)
- [x] Hand display (fan/row of cards)
- [x] Life point display (8000 LP per player)
- [x] Turn phase indicator bar
- [x] Phase advance button
- [x] Player name / turn indicator

## Card Mechanics

- [x] Draw card from deck
- [x] Play card from hand to field (click-to-play with context menu)
- [x] Monster: ATK/DEF display, attack/defense position toggle
- [x] Normal Summon (1 tribute for Lv5-6, 2 for Lv7+)
- [x] Tribute summon logic
- [x] Set monster face-down
- [x] Spell card activation (face-up)
- [x] Trap card set (face-down) and activation
- [x] Send card to graveyard

## Battle System

- [x] Declare attack (select attacker → select target)
- [x] ATK vs ATK / ATK vs DEF damage calculation
- [x] Destroy monster when ATK is lower
- [x] Direct attack (when opponent has no monsters)
- [x] Life point damage application
- [x] Win condition check (LP ≤ 0 or deck out)

## Card Data & Decks

- [x] YGOPRODeck API integration (fetch card by name/ID)
- [x] Pre-built starter decks (Yugi deck, Kaiba deck)
- [x] Deck selection screen before duel
- [x] Card detail tooltip/modal on hover

## Graveyard & Extra

- [x] Graveyard viewer modal (both players)
- [x] Extra deck viewer

## Real-time Sync

- [x] All game actions synced via WebSocket
- [x] Opponent's moves reflected in real-time
- [x] Connection status indicator
- [x] Reconnect handling

## Tests

- [x] Game engine unit tests
- [x] Battle calculation tests
- [x] Phase transition tests

## UX Changes

- [x] Skip deck selection screen — auto-assign predefined decks and go straight to lobby
- [x] Self-play mode: skip lobby/room, auto-start duel with both sides in same session

## Bug Fixes

- [x] Fix broken duel board UI (CSS variables not loading, layout collapsed, zones show as plain text)
- [x] Fix card zone visual styling (no borders, no backgrounds, no neon glow)
- [x] Fix life points bar not rendering
- [x] Fix cyberpunk fonts not loading
- [x] Add visible Graveyard zone to both sides of the board (with card count + viewer)
- [x] Add visible Banished zone to both sides of the board (with card count + viewer)
- [x] Apply Option B (High-Contrast Neon) styles to full duel board
- [x] Fix duel board layout — remove wild open space, compact both player areas to fill screen
- [x] Make board fully responsive — card zones, fonts, and UI scale with viewport size
- [x] Add Field Zone slot to both sides of the board (play/activate field spells)
- [x] Fix card count numbers on deck/GY/banished zones — too small to read
- [x] Align monster zones vertically with spell/trap zones (same column positions)
- [x] Mirror field zone: left side for current player, right side for opponent
- [x] Align opponent zone grid and player zone grid on the same center column (cross-player vertical alignment)
- [x] Fix monster zones not highlighting when a summonable monster is selected from hand
- [x] Remove auto-draw on turn start from game engine
- [x] Add blinking Draw icon on deck during Draw Phase — player clicks to draw a card
- [x] Arrange Graveyard, Banished, Deck zones in the same order for both players
- [x] Fix monster cards overlapping adjacent zones when set face-down
- [x] Fix face-down monster overlap with zone spacing instead of card scaling
- [x] Show full face-down card picture — remove overflow:hidden so rotated cards can extend beyond zone
- [x] Increase monster zone gap so face-down rotated cards don't overlap neighbours
- [x] Prevent drawing twice per turn — lock deck click after first draw in Draw Phase
- [x] Add Extra Deck zone to the duel board for both players
- [x] Fix GraveyardViewer tooltip causing panel to flash/jump on card hover
- [x] Fully fix GraveyardViewer flicker — still occurring after previous fix
- [x] Fix GraveyardViewer to a fixed height so hovering a card does not stretch the panel
- [x] Add fixed card detail panel on duel board — shows full card info when a card is selected
- [x] Invert opponent zone order: Spell/Trap on top, Monster Zone below (mirror real board)
- [x] Improve card detail panel text readability — better contrast and font sizes
- [x] Increase card detail panel font sizes — text too small to read
- [x] GraveyardViewer: click card to show details in fixed panel instead of inline tooltip
- [x] Card detail panel should not be darkened by GraveyardViewer overlay — raise z-index
- [x] Opponent face-down card protection — do not reveal card details when clicking face-down opponent cards
- [x] Increase duel log text size — too small to read
- [x] GraveyardViewer: remove card scaling — show cards at natural full size
- [x] Apply natural card size (100×140px) to all zones on the main game board
- [x] Stack Card Detail panel on top of Duel Log panel on the right side; remove left panel to give board more space
- [x] Unify font size and color in the right panel (Card Detail + Duel Log) for consistent typography
- [x] Fix face-down monsters overlapping adjacent zones on the board
- [x] Show face-down DEF monsters at full 100×140px rotated 90° — no scale-down, allow overflow beyond zone boundaries
- [x] Save face-down card display rule as project knowledge
- [x] Increase Monster Zone gap so adjacent face-down DEF cards (140px wide) never overlap
- [x] Save S/T zone alignment rule to KNOWLEDGE.md
- [x] Align each Spell/Trap zone directly below its corresponding Monster Zone (same column, same gap)
- [x] Fix phase card to consistent fixed height — NEXT button presence should not change card size
- [x] Add animated waiting indicator in phase card for opponent's turn
- [x] Reorder GraveyardViewer subtabs to Extra → Graveyard → Banished (matching board zone order)
- [x] Replace surrender message box with a styled confirmation dialog to prevent accidental surrenders
- [x] Display hand cards in pile style when hand size >= 10
- [x] Pile hand: fixed 676px container, step = (676-100)/(n-1) to always maximize spread, tightens with more cards
- [x] Pile hand container width = 9-card spread width (9×100 + 8×4 = 932px)
- [x] Pile mode cards must not scale down — always 100×140px, only offset changes
- [x] Pile mode: hovered card lifts up and is fully visible above the stack (raised z-index + upward translation)
- [x] Fix: hovered pile card (z-index 1000) appears above surrender modal backdrop — raise modal z-index above 1000
- [x] Make WAITING indicator in phase card more visible (larger text, brighter pink, bigger dots)
- [x] GraveyardViewer overlay z-index fix — raise above 1000 so hovered pile cards don't bleed through
- [x] Set spell/trap zones: clicking shows card details + available actions (Activate, etc.) same as set monsters
- [x] Set monsters: add Flip Summon action option in context menu (Main Phase, player's turn, not set this turn)
- [x] Add 2 Extra Monster Zones (EMZ): shared zones at top center, one per player, for Extra Deck summons
- [x] EMZ: add extraMonsterZone field to PlayerState in gameTypes.ts
- [x] EMZ: engine support — SUMMON_TO_EMZ action, enforce 1 EMZ per player rule
- [x] EMZ: render 2 EMZ slots in DuelBoard UI between the two players' rows, with click handlers and actions
- [x] EMZ: move row into center divider bar (between phase indicator and surrender button)
- [x] EMZ: align each slot horizontally with the player's second Main Monster Zone (index 1)
- [x] EMZ layout fix: align slots with column 1 of each player's zone grid at all viewport sizes (no pixel drift)
- [x] EMZ: merge into same row as phase card and surrender button (no separate row)
- [x] Monster context menu: add "Activate Effect" option for Effect Monsters and other monsters with effects
- [x] Attack animation: beam/slash from attacker zone to target zone with impact flash on DECLARE_ATTACK / DIRECT_ATTACK
- [x] Enhanced attack animation: multi-layer beam, screen flash, shockwave, particle burst, card shake
- [x] Use Blue-Eyes deck (provided passcodes) for both players
- [x] Summoning animation: glow burst on zone when monster is normal or special summoned
- [x] Damage step shake: shake target card zone + flash LP counter red after attack resolves
