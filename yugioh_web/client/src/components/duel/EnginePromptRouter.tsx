import type { EngineAction, EnginePrompt } from "../../../../shared/engineTypes";
import { EngineActionPanel } from "./EngineActionPanel";
import { YesNoPanel } from "./prompts/YesNoPanel";
import { PositionPanel } from "./prompts/PositionPanel";
import { SelectCardPanel } from "./prompts/SelectCardPanel";

interface EnginePromptRouterProps {
  actions: EngineAction[];
  prompt: EnginePrompt | null;
  onAction: (actionIndex: number) => void;
}

export function EnginePromptRouter({ actions, prompt, onAction }: EnginePromptRouterProps) {
  if (!prompt) {
    return <EngineActionPanel actions={actions} onAction={onAction} />;
  }

  switch (prompt.type) {
    case "effect_yn":
    case "yes_no":
      return <YesNoPanel actions={actions} prompt={prompt} onAction={onAction} />;

    case "position":
      return <PositionPanel actions={actions} prompt={prompt} onAction={onAction} />;

    case "select_card":
    case "tribute":
      return <SelectCardPanel actions={actions} prompt={prompt} onAction={onAction} />;

    default:
      return <EngineActionPanel actions={actions} onAction={onAction} />;
  }
}
