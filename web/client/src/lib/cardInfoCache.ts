import type { CardInfo } from "../../../shared/engineTypes";
import { API_BASE } from "./apiBase";

// Printed card faces fetched from the backend, keyed by passcode. Entries are
// write-once per id, so reading during render is stable.
//   undefined — never fetched, or the last attempt failed transiently
//   null      — the server returned 404; this code isn't in cards.cdb
const cache = new Map<number, CardInfo | null>();

// One promise per in-flight passcode, so concurrent requests for the same card
// share a single fetch. Cleared on failure as well as success, or a retained
// rejected promise would block every later attempt.
const inFlight = new Map<number, Promise<CardInfo | null>>();

// Per-passcode listeners, so a consumer can be told when its card lands rather
// than re-checking the cache itself.
const listeners = new Map<number, Set<() => void>>();

export function getCachedInfo(id: number): CardInfo | null | undefined {
  return cache.get(id);
}

/** Call `listener` whenever the cached face for `id` changes. Returns the
 *  unsubscribe function. */
export function subscribeCardInfo(
  id: number,
  listener: () => void
): () => void {
  const forId = listeners.get(id) ?? new Set();
  listeners.set(id, forId);
  forId.add(listener);
  return () => {
    forId.delete(listener);
    if (forId.size === 0) listeners.delete(id);
  };
}

function store(id: number, info: CardInfo | null): void {
  cache.set(id, info);
  listeners.get(id)?.forEach(listener => listener());
}

/** Resolve a card face, fetching at most once per passcode.
 *
 *  Resolves `null` when the server says 404 (cached, never re-requested).
 *  Rejects on a transient failure without caching, so re-selecting the card
 *  retries. */
export function fetchCardInfo(id: number): Promise<CardInfo | null> {
  const cached = cache.get(id);
  if (cached !== undefined) return Promise.resolve(cached);

  const pending = inFlight.get(id);
  if (pending) return pending;

  const request = fetch(`${API_BASE}/api/web/card/${id}`)
    .then(async res => {
      if (res.status === 404) {
        store(id, null);
        return null;
      }
      if (!res.ok) throw new Error(`card ${id}: HTTP ${res.status}`);
      const info = (await res.json()) as CardInfo;
      store(id, info);
      return info;
    })
    .finally(() => {
      inFlight.delete(id);
    });

  inFlight.set(id, request);
  return request;
}

export function clearInfoCache(): void {
  const cleared: number[] = [];
  cache.forEach((_, id) => cleared.push(id));
  cache.clear();
  inFlight.clear();
  // Subscribers are live components, so they stay subscribed — but their
  // snapshots just became stale, so tell them to re-read.
  cleared.forEach(id => listeners.get(id)?.forEach(listener => listener()));
}
