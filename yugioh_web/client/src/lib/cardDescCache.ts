// Card descriptions fetched from the YGOProDeck API, keyed by card id.
// Entries are write-once per id, so reading during render is stable.
const cache = new Map<number, string>();

export function getCachedDesc(id: number): string | undefined {
  return cache.get(id);
}

export function setCachedDesc(id: number, desc: string): void {
  cache.set(id, desc);
}

export function clearDescCache(): void {
  cache.clear();
}
