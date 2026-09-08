export type TurnOrder = "random" | "first" | "second";
export type AgentPlayer = 0 | 1;

export interface ResolvedTurnOrder {
  agentPlayer: AgentPlayer;
  animateCoinFlip: boolean;
}

export function resolveTurnOrder(choice: TurnOrder): ResolvedTurnOrder {
  if (choice === "first") return { agentPlayer: 0, animateCoinFlip: false };
  if (choice === "second") return { agentPlayer: 1, animateCoinFlip: false };
  return {
    agentPlayer: Math.random() < 0.5 ? 0 : 1,
    animateCoinFlip: true,
  };
}
