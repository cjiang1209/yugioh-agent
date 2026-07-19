import { render, within } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { YesNoPanel } from "./YesNoPanel";
import type {
  EngineAction,
  EnginePrompt,
} from "../../../../../shared/engineTypes";

const actions: EngineAction[] = [
  {
    index: 2,
    description: "Yes",
    card_code: 0,
    card_name: "",
    category: "yes",
  },
  {
    index: 4,
    description: "No",
    card_code: 0,
    card_name: "",
    category: "no",
  },
];

const prompt: EnginePrompt = { type: "effect_yn", card_name: "Test Card" };

const BADGE = "Recommended by AI Assist";

describe("YesNoPanel recommended badge", () => {
  it("badges the yes button when recommendedIndex matches it", () => {
    const { container } = render(
      <YesNoPanel
        actions={actions}
        prompt={prompt}
        onAction={() => {}}
        recommendedIndex={2}
      />
    );
    expect(within(container).getAllByTitle(BADGE)).toHaveLength(1);
    const yesButton = within(container).getByText("YES").closest("button")!;
    expect(within(yesButton).getByTitle(BADGE)).toBeTruthy();
  });

  it("badges the no button when recommendedIndex matches it", () => {
    const { container } = render(
      <YesNoPanel
        actions={actions}
        prompt={prompt}
        onAction={() => {}}
        recommendedIndex={4}
      />
    );
    expect(within(container).getAllByTitle(BADGE)).toHaveLength(1);
    const noButton = within(container).getByText("NO").closest("button")!;
    expect(within(noButton).getByTitle(BADGE)).toBeTruthy();
  });

  it("shows no badge when recommendedIndex is null", () => {
    const { container } = render(
      <YesNoPanel
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
      <YesNoPanel
        actions={actions}
        prompt={prompt}
        onAction={() => {}}
        recommendedIndex={99}
      />
    );
    expect(within(container).queryAllByTitle(BADGE)).toHaveLength(0);
  });
});
