// ─── Server-side Deck Loader ──────────────────────────────────────────────────
// Fetches real card data from YGOPRODeck API for predefined deck IDs.
// Results are cached in-memory to avoid redundant network calls.

import { YgoCard } from "../shared/gameTypes";

const cardCache = new Map<number, YgoCard>();

async function fetchCard(id: number): Promise<YgoCard | null> {
  if (cardCache.has(id)) return cardCache.get(id)!;
  try {
    const res = await fetch(`https://db.ygoprodeck.com/api/v7/cardinfo.php?id=${id}`);
    if (!res.ok) return null;
    const data = await res.json();
    const card: YgoCard = data.data?.[0] ?? null;
    if (card) cardCache.set(id, card);
    return card;
  } catch {
    return null;
  }
}

/**
 * Fetches all cards for a deck by their IDs (preserving duplicates).
 * Cards that fail to load are silently skipped.
 */
export async function fetchDeckCards(ids: number[]): Promise<YgoCard[]> {
  const uniqueIds = Array.from(new Set(ids));

  // Fetch all unique cards in parallel
  const fetched = await Promise.allSettled(uniqueIds.map(fetchCard));
  const cardMap = new Map<number, YgoCard>();
  fetched.forEach((result, i) => {
    if (result.status === "fulfilled" && result.value) {
      cardMap.set(uniqueIds[i], result.value);
    }
  });

  // Return in original order (with duplicates), filtering out any that failed
  return ids.map((id) => cardMap.get(id)).filter(Boolean) as YgoCard[];
}
