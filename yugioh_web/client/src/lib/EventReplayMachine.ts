import type { EngineBoard, EngineGameState, EventFrame } from "../../../shared/engineTypes";

export type { EventFrame };

export type ReplayCallbacks = {
  onFrameChange: (board: EngineBoard, gameState: EngineGameState) => void;
  onEventReveal: (visibleLog: string[]) => void;
  onComplete: () => void;
};

const EVENT_DELAY_MS = 300;
const FRAME_PAUSE_MS = 150;

export class EventReplayMachine {
  private frames: EventFrame[] = [];
  private pastLog: string[] = [];
  private frameIndex = 0;
  private revealIndex = 0;
  private timerId: ReturnType<typeof setTimeout> | null = null;
  private callbacks: ReplayCallbacks | null = null;
  private _active = false;

  get active(): boolean {
    return this._active;
  }

  start(priorLog: string[], frames: EventFrame[], callbacks: ReplayCallbacks): void {
    this.clearTimer();
    this.frames = frames;
    this.pastLog = [...priorLog];
    this.frameIndex = 0;
    this.revealIndex = 0;
    this.callbacks = callbacks;

    if (frames.length === 0) {
      this._active = false;
      callbacks.onEventReveal([...this.pastLog]);
      callbacks.onComplete();
      return;
    }

    this._active = true;
    // Don't fire onFrameChange here — the frame's board is a post-event
    // snapshot, so it must be applied after the frame's events are revealed.
    callbacks.onEventReveal([...this.pastLog]);
    this.timerId = setTimeout(() => this.tick(), EVENT_DELAY_MS);
  }

  private tick(): void {
    this.timerId = null;
    if (!this._active || !this.callbacks) return;

    const frame = this.frames[this.frameIndex];
    this.revealIndex++;
    this.callbacks.onEventReveal([...this.pastLog, ...frame.events.slice(0, this.revealIndex)]);

    if (this.revealIndex < frame.events.length) {
      this.timerId = setTimeout(() => this.tick(), EVENT_DELAY_MS);
      return;
    }

    this.callbacks.onFrameChange(frame.board, frame.game_state);
    this.pastLog.push(...frame.events);

    if (this.frameIndex < this.frames.length - 1) {
      this.frameIndex++;
      this.revealIndex = 0;
      // Pause so the board change registers before the next frame's events
      this.timerId = setTimeout(() => this.tick(), FRAME_PAUSE_MS);
    } else {
      this._active = false;
      this.frames = [];
      this.callbacks.onComplete();
    }
  }

  reset(): void {
    this.clearTimer();
    this._active = false;
    this.frames = [];
    this.pastLog = [];
    this.frameIndex = 0;
    this.revealIndex = 0;
    this.callbacks = null;
  }

  dispose(): void {
    this.reset();
  }

  private clearTimer(): void {
    if (this.timerId !== null) {
      clearTimeout(this.timerId);
      this.timerId = null;
    }
  }
}
