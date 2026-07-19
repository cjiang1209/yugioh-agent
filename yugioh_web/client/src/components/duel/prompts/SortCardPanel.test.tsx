import { render, within } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { SortCardPanel } from "./SortCardPanel";
import type {
  EngineAction,
  EnginePrompt,
} from "../../../../../shared/engineTypes";

const actions: EngineAction[] = [
  {
    index: 0,
    description: "Blue-Eyes White Dragon",
    card_code: 89631139,
    card_name: "Blue-Eyes White Dragon",
    category: "sort",
  },
  {
    index: 1,
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
        recommendedIndex={1}
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
