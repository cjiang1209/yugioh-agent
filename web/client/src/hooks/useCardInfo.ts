import { useCallback, useEffect, useSyncExternalStore } from "react";
import type { CardInfo } from "../../../shared/engineTypes";
import {
  fetchCardInfo,
  getCachedInfo,
  subscribeCardInfo,
} from "../lib/cardInfoCache";

/**
 * The printed face for a passcode, fetched once and shared across consumers.
 *
 * `undefined` while the fetch is in flight (or for passcode 0, a face-down
 * card), `null` when the server doesn't know the code, otherwise the face.
 *
 * The value is read during render from the id passed this render, so a caller
 * switching cards can never paint the previous card's face.
 */
export function useCardInfo(id: number): CardInfo | null | undefined {
  const subscribe = useCallback(
    (onStoreChange: () => void) =>
      id ? subscribeCardInfo(id, onStoreChange) : () => {},
    [id]
  );

  const info = useSyncExternalStore(subscribe, () =>
    id ? getCachedInfo(id) : undefined
  );

  useEffect(() => {
    if (!id || getCachedInfo(id) !== undefined) return;
    // Transient failures aren't cached, so re-selecting the card retries.
    fetchCardInfo(id).catch(() => {});
  }, [id]);

  return info;
}
