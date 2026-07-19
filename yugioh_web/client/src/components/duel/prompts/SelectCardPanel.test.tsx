import { render, within } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { SelectCardPanel } from "./SelectCardPanel";
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
    category: "select_card",
  },
  {
    index: 1,
    description: "Dark Magician",
    card_code: 46986414,
    card_name: "Dark Magician",
    category: "select_card",
  },
  {
    index: 9,
    description: "Finish",
    card_code: 0,
    card_name: "",
    category: "finish",
  },
];

const prompt: EnginePrompt = { type: "select_card", min: 1, max: 2 };

const BADGE = "Recommended by AI Assist";

describe("SelectCardPanel recommended badge", () => {
  it("badges the matching card, including index 0", () => {
    const { container } = render(
      <SelectCardPanel
        actions={actions}
        prompt={prompt}
        onAction={() => {}}
        recommendedIndex={0}
      />
    );
    expect(within(container).getAllByTitle(BADGE)).toHaveLength(1);
    const card = within(container)
      .getByTitle("Blue-Eyes White Dragon")
      .closest("button")!;
    expect(within(card).getByTitle(BADGE)).toBeTruthy();
  });

  it("shows no badge when recommendedIndex is null", () => {
    const { container } = render(
      <SelectCardPanel
        actions={actions}
        prompt={prompt}
        onAction={() => {}}
        recommendedIndex={null}
      />
    );
    expect(within(container).queryAllByTitle(BADGE)).toHaveLength(0);
  });

  it("shows no badge when recommendedIndex matches no card action", () => {
    const { container } = render(
      <SelectCardPanel
        actions={actions}
        prompt={prompt}
        onAction={() => {}}
        recommendedIndex={99}
      />
    );
    expect(within(container).queryAllByTitle(BADGE)).toHaveLength(0);
  });

  it("badges the finish button when it is the recommendation", () => {
    const { container } = render(
      <SelectCardPanel
        actions={actions}
        prompt={prompt}
        onAction={() => {}}
        recommendedIndex={9}
      />
    );
    // Exactly one badge, and it is on the Finish button (not a card).
    expect(within(container).getAllByTitle(BADGE)).toHaveLength(1);
    const finishButton = within(container)
      .getByText("FINISH")
      .closest("button") as HTMLButtonElement;
    expect(within(finishButton).getByTitle(BADGE)).toBeTruthy();
  });
});
