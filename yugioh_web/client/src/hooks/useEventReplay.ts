import { useCallback, useEffect, useRef, useState } from "react";
import { EventReplayMachine, type EventFrame } from "../lib/EventReplayMachine";
import type { EngineBoard, EngineGameState } from "../../../shared/engineTypes";

export function useEventReplay() {
  const machineRef = useRef(new EventReplayMachine());
  const [visibleLog, setVisibleLog] = useState<string[]>([]);
  const [currentBoard, setCurrentBoard] = useState<EngineBoard | null>(null);
  const [currentGameState, setCurrentGameState] = useState<EngineGameState | null>(null);
  const [isReplaying, setIsReplaying] = useState(false);

  useEffect(() => () => machineRef.current.dispose(), []);

  const startReplay = useCallback(
    (priorLog: string[], frames: EventFrame[], onComplete: () => void) => {
      setIsReplaying(true);
      machineRef.current.start(priorLog, frames, {
        onFrameChange: (board, gs) => {
          setCurrentBoard(board);
          setCurrentGameState(gs);
        },
        onEventReveal: (log) => setVisibleLog(log),
        onComplete: () => {
          setIsReplaying(false);
          setCurrentBoard(null);
          setCurrentGameState(null);
          onComplete();
        },
      });
    },
    [],
  );

  const resetReplay = useCallback(() => {
    machineRef.current.reset();
    setVisibleLog([]);
    setCurrentBoard(null);
    setCurrentGameState(null);
    setIsReplaying(false);
  }, []);

  return { visibleLog, currentBoard, currentGameState, isReplaying, startReplay, resetReplay };
}
