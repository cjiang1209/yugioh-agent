# Yu-Gi-Oh! Duel Arena — Project Knowledge

This file records design rules and decisions that must be honored in all future changes.

---

## Face-Down Monster Card Display

**Rule:** Always show the full picture of face-down monster cards, even if they stretch out of the monster zones. Do not scale them down in size.

**Implementation:**

- Face-down DEF position cards are rotated 90° (`transform: rotate(90deg)`) at their natural 100×140px size.
- After rotation the card appears 140px wide × 100px tall, overflowing the 100px-wide zone boundary on both sides.
- The zone container must use `overflow: visible` — never `overflow: hidden` — so the card is never clipped.
- Do NOT apply any `scale()` transform to shrink the rotated card.

**Spacing rule:** Always adjust the gap between Monster Zones so that adjacent face-down monster cards (140px wide when rotated) do not overlap each other. The minimum gap between zone centers must be at least 140px (i.e., gap ≥ 40px when zones are 100px wide).

---

## Spell/Trap Zone Alignment

**Rule:** Align each Spell/Trap zone directly below its corresponding Monster Zone. Zone 1 S/T must be in the same column as Zone 1 Monster, Zone 2 S/T below Zone 2 Monster, and so on.

**Implementation:** Both the Monster row and the S/T row must use the same `gap` value so their columns stay aligned. The S/T row must not use a different gap or centering that would shift its columns relative to the Monster row. Currently the Monster Zone gap is 44px — the S/T row must also use 44px.

---
