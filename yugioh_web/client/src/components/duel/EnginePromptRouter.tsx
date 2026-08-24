import type {
  EngineAction,
  EnginePrompt,
} from "../../../../shared/engineTypes";
import { EngineActionPanel } from "./EngineActionPanel";
import { YesNoPanel } from "./prompts/YesNoPanel";
import { PositionPanel } from "./prompts/PositionPanel";
import { SelectCardPanel } from "./prompts/SelectCardPanel";
import { SortCardPanel } from "./prompts/SortCardPanel";

interface EnginePromptRouterProps {
  actions: EngineAction[];
  prompt: EnginePrompt | null;
  onAction: (actionIndex: number) => void;
  recommendedIndex?: number | null;
  /** Policy probabilities for the prompt on screen, read by `action.index`. */
  actionProbs?: number[] | null;
}

export function EnginePromptRouter({
  actions,
  prompt,
  onAction,
  recommendedIndex,
  actionProbs,
}: EnginePromptRouterProps) {
  if (!prompt) {
    return (
      <EngineActionPanel
        actions={actions}
        onAction={onAction}
        recommendedIndex={recommendedIndex}
        actionProbs={actionProbs}
      />
    );
  }

  switch (prompt.type) {
    case "effect_yn":
    case "yes_no":
      return (
        <YesNoPanel
          actions={actions}
          prompt={prompt}
          onAction={onAction}
          recommendedIndex={recommendedIndex}
          actionProbs={actionProbs}
        />
      );

    case "position":
      return (
        <PositionPanel
          actions={actions}
          prompt={prompt}
          onAction={onAction}
          recommendedIndex={recommendedIndex}
          actionProbs={actionProbs}
        />
      );

    case "select_card":
    case "tribute":
      return (
        <SelectCardPanel
          actions={actions}
          prompt={prompt}
          onAction={onAction}
          recommendedIndex={recommendedIndex}
          actionProbs={actionProbs}
        />
      );

    case "sort_card":
      return (
        <SortCardPanel
          actions={actions}
          prompt={prompt}
          onAction={onAction}
          recommendedIndex={recommendedIndex}
          actionProbs={actionProbs}
        />
      );

    default:
      return (
        <EngineActionPanel
          actions={actions}
          onAction={onAction}
          recommendedIndex={recommendedIndex}
          actionProbs={actionProbs}
        />
      );
  }
}
