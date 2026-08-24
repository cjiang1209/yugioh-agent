import { render, within } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { PositionPanel } from "./PositionPanel";
import type {
  EngineAction,
  EnginePrompt,
} from "../../../../../shared/engineTypes";

const actions: EngineAction[] = [
  {
    index: 1,
    description: "Blue-Eyes: Face-up Attack",
    card_code: 89631139,
    card_name: "Blue-Eyes",
    category: "position",
  },
  {
    index: 3,
    description: "Blue-Eyes: Face-up Defense",
    card_code: 89631139,
    card_name: "Blue-Eyes",
    category: "position",
  },
];

const prompt: EnginePrompt = { type: "position", card_name: "Blue-Eyes" };

const BADGE = "Recommended by AI Assist";

describe("PositionPanel recommended badge", () => {
  it("badges the matching position button only", () => {
    const { container } = render(
      <PositionPanel
        actions={actions}
        prompt={prompt}
        onAction={() => {}}
        recommendedIndex={3}
      />
    );
    expect(within(container).getAllByTitle(BADGE)).toHaveLength(1);
    const defButton = within(container)
      .getByText("FACE-UP DEFENSE")
      .closest("button")!;
    expect(within(defButton).getByTitle(BADGE)).toBeTruthy();
  });

  it("shows no badge when recommendedIndex is null", () => {
    const { container } = render(
      <PositionPanel
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
      <PositionPanel
        actions={actions}
        prompt={prompt}
        onAction={() => {}}
        recommendedIndex={99}
      />
    );
    expect(within(container).queryAllByTitle(BADGE)).toHaveLength(0);
  });
});

describe("PositionPanel probabilities", () => {
  it("shows each position's probability, keyed by its action index", () => {
    const { container } = render(
      <PositionPanel
        actions={actions}
        prompt={prompt}
        onAction={() => {}}
        actionProbs={[0.9, 0.7, 0.05, 0.3]}
      />
    );
    expect(container.textContent).toContain("70%");
    expect(container.textContent).toContain("30%");
    // The value at position 0 belongs to an action this panel never renders.
    expect(container.textContent).not.toContain("90%");
  });
});
