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
}

export function EnginePromptRouter({
  actions,
  prompt,
  onAction,
  recommendedIndex,
}: EnginePromptRouterProps) {
  if (!prompt) {
    return (
      <EngineActionPanel
        actions={actions}
        onAction={onAction}
        recommendedIndex={recommendedIndex}
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
        />
      );

    case "position":
      return (
        <PositionPanel
          actions={actions}
          prompt={prompt}
          onAction={onAction}
          recommendedIndex={recommendedIndex}
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
        />
      );

    case "sort_card":
      return (
        <SortCardPanel
          actions={actions}
          prompt={prompt}
          onAction={onAction}
          recommendedIndex={recommendedIndex}
        />
      );

    default:
      return (
        <EngineActionPanel
          actions={actions}
          onAction={onAction}
          recommendedIndex={recommendedIndex}
        />
      );
  }
}
