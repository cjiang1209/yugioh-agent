---
name: deck-building
description: Use when adding or modifying .ydk decks in assets/decks/ — verifies every card is summonable, every spell/trap activatable, and the deck doesn't trigger MSG_RETRY under random play
---

# Yu-Gi-Oh! Deck Building & Validation

## Overview

A `.ydk` file in this repo is only "playable" if every card has a real role the engine can execute. Decks fail in subtle, deck-specific ways that the ygopro core only surfaces at runtime: missing summon materials, unpayable costs, searches for cards not in the deck, prompts the random sequencer can't answer.

This skill provides a **construction checklist** for catching those failures up front, plus a **coverage harness** (`scripts/deck_coverage.sh`) that empirically proves each card resolves at least once across many random episodes.

## When to Use

Use when:
- Adding a new `.ydk` to `assets/decks/`
- Swapping cards in an existing deck
- A deck "looks right" but training/eval shows it underperforms or logs MSG_RETRY
- Reviewing a deck someone else wrote before merging

Don't use for:
- Tuning a deck's win-rate against a specific opponent (that's a leaderboard concern, not validity)
- Banlist enforcement (the engine doesn't enforce TCG/OCG lists; only structural legality)

## The failure modes

The first three are real deck-construction bugs. The fourth is a **harness-side issue** the deck happens to expose — the card is legal Yu-Gi-Oh!, but the response-builder or the random sequencer can't produce a valid reply to one of its prompts.

In the harness output, spells/traps show `/` in the summoned column (they aren't summoned, only activated), so a dead spell shows `/ ... MISS 0`. A dead extra-deck monster shows `MISS 0 ... MISS 0`.

| Failure | Symptom in `deck_coverage.sh` | Root cause | Fix location |
|---|---|---|---|
| **Extra-deck card unsummonable** | `MISS 0` summoned, `MISS 0` activated | Main deck lacks materials (e.g. Galaxy-Eyes Cipher in a deck with no Lv8 monsters) | Deck |
| **Cost unpayable** | Spell `MISS 0` activated despite hand being live | Cost requires diversity the deck doesn't have (e.g. Cybernetic Horizon needs 2 different "Cyber" attributes but every Cyber monster is LIGHT) | Deck |
| **Search target absent** | Spell `MISS 0` activated, gets drawn often | Searcher fetches a type/archetype not in the deck (e.g. Reinforcement of the Army with zero Warriors) | Deck |
| **Prompt-incompatible card** | MSG_RETRY in stderr during random eval | Response-builder or random sequencer can't satisfy a prompt the card produces (e.g. Excode Talker's runtime-computed zone-selection) | **Harness, not deck** — see "Handling MSG_RETRY" below |

## Quick reference: construction checklist

Before writing the `.ydk`, **always start with external research** (you will miss core cards if you don't), then work through the rest of the checklist in any order:

- **Study the archetype from external references** *(do this first)*. Don't reconstruct a deck from memory or first principles — you will miss core cards. Search the web for a competitive decklist (e.g. `<archetype> Yu-Gi-Oh decklist`) on sites like `masterduelmeta.com`, `ygoprodeck.com`, `yugiohmeta.com`, `duelingnexus.com`, or `ygorganization.com`. Use `WebSearch` + `WebFetch`. Extract the **archetype skeleton** — the cards every list runs (with counts). This is the floor for your main + extra deck. Generic staples (handtraps, Pot of Desires, etc.) and meta hybrid engines (Fiendsmith, Bystials, etc.) can be added or skipped to taste, but missing an archetype core card is almost always a defect.
- **Pick the archetype's core summoning line.** From the reference list, identify the boss monster(s) the deck wins with, then trace backwards: what materials do they need? Confirm every material is in the main deck.
- **Confirm extra-vs-main classification with the type bits.** A `type` bit of `0x4000000` is Link, `0x800000` is Xyz, `0x2000` is Synchro, `0x40` is Fusion → extra deck. `0x2000000` is `TYPE_SPSUMMON` — these are **main-deck monsters** that must be Special Summoned (e.g. Evil★Twins Ki-sikil & Lil-la). Don't put SPSUMMON main-deck cards in `#extra` based on their boss-monster vibe. Lookup with sqlite:
  ```bash
  sqlite3 -separator '|' assets/cards.cdb "SELECT id, name, type, level, atk, def, race, attribute FROM datas JOIN texts USING(id) WHERE id IN (...);"
  ```
- **For each extra-deck card, verify materials are reachable.** Check `level`, `race`, `attribute`, `type` match what the main deck can produce.
- **Read the script for any card whose cost/effect targets a type or attribute.** Path: `third_party/CardScripts/official/c<passcode>.lua`. Specifically check `s.cost`, `s.target`, `s.filter` functions — these reveal the hidden requirements. The `chk == 0` branch of `s.cost` / `s.target` is the activation-legality check; if its condition (e.g. `Duel.IsExistingMatchingCard(filter, ...)`) is false in this deck, the card is dead.
- **For every searcher (Tenki, RotA, Cynet Mining, Cyber Repair Plant, etc.), confirm the deck actually contains the search targets.**
- **Confirm passcode + script exist:**
  ```bash
  # Card in cards.cdb?
  sqlite3 assets/cards.cdb "SELECT name FROM texts WHERE id = <passcode>;"
  # Script present?
  test -f third_party/CardScripts/official/c<passcode>.lua && echo OK
  ```

## File format

`.ydk` files use one card per line, with `#main` / `#extra` / `!side` section markers. Inline comments after `#` are stripped by `yugioh_env/deck_parser.py`:

```
#created by yugioh-env
#main
89631139 # Blue-Eyes White Dragon
89631139 # Blue-Eyes White Dragon
...
#extra
23995346 # Blue-Eyes Ultimate Dragon
...
!side
```

**Always add inline name comments.** Card IDs are otherwise unreviewable.

## Sizes

- Main deck: **40 is the convention** in this repo. Engine allows 40–60 but stick to 40 unless there's a reason.
- Extra deck: **flexible (0–15).** Only include cards your main deck can actually summon.
- Side deck: usually empty (`!side` line only).

## Validation workflow

After writing the `.ydk`:

### Step 1 — Parse-check

```bash
.venv/bin/python -c "
from yugioh_env.deck_parser import parse_ydk
d = parse_ydk('assets/decks/<name>.ydk')
print(f'main={len(d[\"main\"])} extra={len(d[\"extra\"])} side={len(d[\"side\"])}')"
```

Confirms section counts. Catches off-by-one mistakes.

### Step 2 — Mirror eval (surfaces MSG_RETRY)

```bash
bash scripts/eval.sh --agent random --opponents random \
  --deck-paths assets/decks/<name>.ydk --episodes 50 --seed 1 --workers 8 \
  2>/tmp/m.err 1>/tmp/m.out
cat /tmp/m.out
echo "retries=$(grep -c MSG_RETRY /tmp/m.err)"
```

Repeat on a second seed (`--seed 99`) — some prompts only appear on specific RNG paths.

**`retries=0` is the pass criterion**, but a nonzero count usually means the **response-builder or random sequencer** can't answer a prompt the card produces — not that the card is illegal Yu-Gi-Oh!. See "Handling MSG_RETRY" below before changing the deck.

### Step 3 — Coverage harness (detects dead cards)

```bash
bash scripts/deck_coverage.sh \
  --deck assets/decks/<name>.ydk --episodes 500 --seed 1 --workers 8
```

Reports per-card summon and activation counts across 500 random self-play episodes. Use `--workers 8` (or however many cores you have) — parallelism gives near-linear speedup. Results are byte-equal across worker counts (each worker takes a distinct episode range, addressed by `episode_idx`). **Pass criterion:** the `=== Issues ===` section ends with `No hard issues`.

A monster is flagged as dead only if it was **never summoned AND never activated**. Both signals count as "playing" — hand traps (Maxx "C", Ash Blossom, Effect Veiler) fire `MSG_CHAINING` from the hand without being summoned, and cards like Elder Entity N'tss or Dotscaper that activate from deck/GY/banished often never touch the field but still resolve their effects. If you see `Monsters never summoned and never activated` or `Spells/Traps never activated`, the card is dead — diagnose by reading its script and looking for unsatisfied conditions.

The harness also lists `Monsters summoned but no effect activation observed`. This is **informational only** — vanillas (Blue-Eyes White Dragon), beatsticks (Chimeratech Fortress), and cards whose effect IS the summon itself (Cyber Dragon's self-SS) won't fire `MSG_CHAINING` even though they're working correctly. Verify each by reading the card text before assuming it's a bug.

### Step 4 — Cross-deck eval (optional)

If the deck will be used in training, also run a mixed-pool eval to see how it interacts with existing decks:

```bash
bash scripts/eval.sh --agent random --opponents random \
  --deck-paths assets/decks/<new>.ydk assets/decks/blue_eyes.ydk assets/decks/dark_magician.ydk \
  --episodes 100 --seed 7 --workers 8
```

Win rate well below 50 % vs the starters means the deck is hard to pilot under random play — not a bug, but worth knowing.

## Common mistakes

### Trusting that "the card has a script" means "the card works in this deck"

A script can exist and load cleanly while the card is still dead in your deck. A spell or trap can be Set every game but never activate because its cost or target condition is unsatisfiable in this deck — runtime is the only way to find out. Always run Step 3.

### Treating `Monsters summoned but no effect activation` as a bug

Vanillas, beatsticks, and continuous-effect bodies are *supposed* to show `0` activations. Verify against the card text before "fixing" it.

### Bisecting MSG_RETRY by guessing

When MSG_RETRY shows up, don't guess the culprit and don't immediately blame the deck. Follow the "Handling MSG_RETRY" workflow below — bisect to the trigger card, build a minimal repro, then decide whether to swap the card or escalate to the harness owner.

### Skipping the parse-check on edits

A `sed` or hand-edit that drops a line silently turns a 40-card deck into a 39-card deck. The engine refuses to start a duel with the wrong main count. Always run Step 1 after edits.

## Handling MSG_RETRY

**MSG_RETRY almost always indicates a harness defect, not a deck-building error.** The engine sends MSG_RETRY when the response bytes it received don't match what the prompt expected — that's a contract between the engine and our response-builder / random policy, and the card itself is usually legal Yu-Gi-Oh! that a real human or a trained agent would handle without issue.

**The harness fix is out of scope for deck building.** This skill stops at "isolate the trigger card, work around it, report it." Don't dig into `yugioh_env/duel.py`, `message_parser.py`, or `response_builder.py` — that's a separate task for whoever owns the harness.

### 1. Bisect to find the trigger card

Don't guess. Strip the extra deck to bare basics (e.g. Linkuriboh + 1–2 Honeybots) and re-test. If retries vanish, add cards back one at a time until they reappear. Common triggers:
- **Runtime-computed selection counts** (Excode Talker's "choose unused Main Monster Zones equal to the number of monsters currently in the Extra Monster Zones")
- **Quick-effect summons during opponent turns** (e.g. I:P Masquerena's "during opponent's turn, Link-Summon using this and other monsters")
- **Multi-step costs that branch on field state** (some Fusion / Ritual procs)

Run on 2+ seeds (`--seed 1`, `--seed 99`) — a single seed can hide a bug because the offending prompt never fires.

### 2. Save a minimal reproduction

Once you've isolated the card, save the smallest deck that still triggers MSG_RETRY into `/tmp/repro.ydk`. Confirm it reproduces:

```bash
bash scripts/eval.sh --agent random --opponents random \
  --deck-paths /tmp/repro.ydk --episodes 50 --seed 1 --workers 1 \
  2>/tmp/repro.err 1>/tmp/repro.out
echo "retries=$(grep -c MSG_RETRY /tmp/repro.err)"
```

Use `--workers 1` for the repro — single-process makes traces easier to read if someone else picks it up.

### 3. Work around in the deck and file a follow-up

Swap the trigger card out and leave a comment in the `.ydk` so the next person knows it was a workaround, not a banlist decision:

```
# 40669071 # Excode Talker — removed: triggers MSG_RETRY under random play
#                            (harness issue, not a deck bug; see <task/PR link>)
```

Then file a separate task for the harness owner with:
- The triggering card (passcode + name)
- A path to the minimal `/tmp/repro.ydk` (or paste its contents)
- The seed + worker count that reproduces (`--seed 1 --workers 1`)
- The retry count you observed (e.g. `4 retries / 50 episodes`)

That's enough for whoever owns the response-builder to investigate.

## Coverage harness internals

Source: `cli/deck_coverage.py` (invoked via the `scripts/deck_coverage.sh` wrapper, which activates the venv and sets `PYTHONPATH`).

The harness monkey-patches `yugioh_env.message_parser.parse_messages` to tap every parsed message, records `code` from `MSG_SUMMONING / SPSUMMONING / FLIPSUMMONING` (counted as "summoned") and `MSG_CHAINING` (counted as "activated"), then runs N self-play episodes with a random-vs-random policy via `TrainingEnv`. Spells/traps are never "summoned" — the report shows `/` in the summoned column for them.

## Red flags

If you find yourself thinking:

- *"I know this archetype already, I don't need to look it up"* → the construction checklist starts with external research for a reason. Memory misses Link-4/Rank-4 bosses, archetype-specific trap support, and SPSUMMON-classified main-deck monsters that look like extra-deck bosses. A 60-second WebSearch catches them.
- *"The card text looks like it should work, so I'll trust it"* → run Step 3.
- *"Just one MSG_RETRY across 50 episodes is fine"* → re-run on a different seed; if retries persist, follow "Handling MSG_RETRY" — bisect, build a minimal repro, and decide whether the fix is harness-side or a deck workaround.
- *"500 episodes is overkill"* → with `--workers 8` it runs in minutes (heavy combo decks take a few minutes more than light beatdowns), not hours. That's the cost of catching dead cards before they ship; run it.
- *"I'll skip the harness for spell-only swaps"* → a spell with an unsatisfiable cost or target is still a dead card; the harness is what catches it.

All of these mean: run the validation workflow anyway.
