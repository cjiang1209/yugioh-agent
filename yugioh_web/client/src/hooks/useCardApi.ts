import { useCallback, useState } from "react";
import { YgoCard } from "../../../shared/gameTypes";

const API_BASE = "https://db.ygoprodeck.com/api/v7";

export function useCardApi() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchCardsByIds = useCallback(async (ids: number[]): Promise<YgoCard[]> => {
    setLoading(true);
    setError(null);
    try {
      // Fetch unique IDs to avoid redundant requests
      const uniqueIds = Array.from(new Set(ids));
      const results = await Promise.allSettled(
        uniqueIds.map((id) =>
          fetch(`${API_BASE}/cardinfo.php?id=${id}`)
            .then((r) => r.json())
            .then((data) => (data.data?.[0] as YgoCard) ?? null)
        )
      );
      const cardMap = new Map<number, YgoCard>();
      results.forEach((r) => {
        if (r.status === "fulfilled" && r.value) {
          cardMap.set(r.value.id, r.value);
        }
      });
      // Return in original order (with duplicates)
      return ids.map((id) => cardMap.get(id)).filter(Boolean) as YgoCard[];
    } catch (e) {
      setError("Failed to fetch cards");
      return [];
    } finally {
      setLoading(false);
    }
  }, []);

  const searchCards = useCallback(async (query: string): Promise<YgoCard[]> => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/cardinfo.php?fname=${encodeURIComponent(query)}&num=20&offset=0`);
      const data = await res.json();
      return (data.data ?? []) as YgoCard[];
    } catch (e) {
      setError("Search failed");
      return [];
    } finally {
      setLoading(false);
    }
  }, []);

  return { fetchCardsByIds, searchCards, loading, error };
}
