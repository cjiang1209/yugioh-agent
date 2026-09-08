import { render, within } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { SortCardPanel } from "./SortCardPanel";
import type {
  EngineAction,
  EnginePrompt,
} from "../../../../../shared/engineTypes";

// Indices differ from positions, so a lookup keyed by position instead of
// by `action.index` shows the wrong card's data.
const actions: EngineAction[] = [
  {
    index: 2,
    description: "Blue-Eyes White Dragon",
    card_code: 89631139,
    card_name: "Blue-Eyes White Dragon",
    category: "sort",
  },
  {
    index: 5,
    description: "Dark Magician",
    card_code: 46986414,
    card_name: "Dark Magician",
    category: "sort",
  },
];

const prompt: EnginePrompt = { type: "sort_card", count: 2 };

const BADGE = "Recommended by AI Assist";

describe("SortCardPanel recommended badge", () => {
  it("badges the matching card only", () => {
    const { container } = render(
      <SortCardPanel
        actions={actions}
        prompt={prompt}
        onAction={() => {}}
        recommendedIndex={5}
      />
    );
    expect(within(container).getAllByTitle(BADGE)).toHaveLength(1);
    const card = within(container)
      .getByTitle("Dark Magician")
      .closest("button")!;
    expect(within(card).getByTitle(BADGE)).toBeTruthy();
  });

  it("shows no badge when recommendedIndex is null", () => {
    const { container } = render(
      <SortCardPanel
        actions={actions}
        prompt={prompt}
        onAction={() => {}}
        recommendedIndex={null}
      />
    );
    expect(within(container).queryAllByTitle(BADGE)).toHaveLength(0);
  });

  it("shows no badge when recommendedIndex matches no action", () => {
    const { container } = render(
      <SortCardPanel
        actions={actions}
        prompt={prompt}
        onAction={() => {}}
        recommendedIndex={99}
      />
    );
    expect(within(container).queryAllByTitle(BADGE)).toHaveLength(0);
  });
});

describe("SortCardPanel probabilities", () => {
  it("shows a probability on each card tile", () => {
    const { container } = render(
      <SortCardPanel
        actions={actions}
        prompt={prompt}
        onAction={() => {}}
        actionProbs={[0.9, 0.05, 0.7, 0.04, 0.06, 0.3]}
      />
    );
    expect(container.textContent).toContain("70%");
    expect(container.textContent).toContain("30%");
    // The value at position 0 belongs to an action this panel never renders.
    expect(container.textContent).not.toContain("90%");
  });
});
