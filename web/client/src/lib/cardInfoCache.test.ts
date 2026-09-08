import { describe, it, expect, vi, afterEach } from "vitest";
import {
  clearInfoCache,
  fetchCardInfo,
  getCachedInfo,
  subscribeCardInfo,
} from "./cardInfoCache";
import { API_BASE } from "./apiBase";
import type { CardInfo } from "../../../shared/engineTypes";

const ID = 44508094;

const INFO: CardInfo = {
  code: ID,
  name: "Stardust Dragon",
  desc: "1 Tuner + 1+ non-Tuner monsters",
  card_type: "monster",
  typeline: ["Dragon", "Synchro", "Effect"],
  attribute: "WIND",
  race: "Dragon",
  level: 8,
  level_kind: "level",
  attack: 2500,
  defense: 2000,
  scales: null,
  link_arrows: null,
};

const okResponse = () => ({
  ok: true,
  status: 200,
  json: () => Promise.resolve(INFO),
});

const notFoundResponse = () => ({
  ok: false,
  status: 404,
  json: () => Promise.resolve({ detail: "Unknown card code: 1" }),
});

/** A promise plus its resolver, so a request can be held open mid-test. */
function deferred<T>() {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>(r => {
    resolve = r;
  });
  return { promise, resolve };
}

describe("cardInfoCache", () => {
  afterEach(() => {
    clearInfoCache();
    vi.unstubAllGlobals();
  });

  it("fetches once and caches the result", async () => {
    const fetchMock = vi.fn(() => Promise.resolve(okResponse()));
    vi.stubGlobal("fetch", fetchMock);

    expect(await fetchCardInfo(ID)).toEqual(INFO);
    expect(getCachedInfo(ID)).toEqual(INFO);
    expect(await fetchCardInfo(ID)).toEqual(INFO);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("issues a single request for overlapping calls", async () => {
    // Without in-flight dedup this fires twice: selecting A, switching away and
    // back re-runs the effect before A has been cached.
    const first = deferred<ReturnType<typeof okResponse>>();
    const fetchMock = vi.fn(() => first.promise);
    vi.stubGlobal("fetch", fetchMock);

    const a = fetchCardInfo(ID);
    const b = fetchCardInfo(ID);
    expect(fetchMock).toHaveBeenCalledTimes(1);

    first.resolve(okResponse());
    expect(await a).toEqual(INFO);
    expect(await b).toEqual(INFO);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("caches a 404 as null and stops asking", async () => {
    const fetchMock = vi.fn(() => Promise.resolve(notFoundResponse()));
    vi.stubGlobal("fetch", fetchMock);

    expect(await fetchCardInfo(1)).toBeNull();
    expect(getCachedInfo(1)).toBeNull();
    expect(await fetchCardInfo(1)).toBeNull();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("does not cache a transient failure and re-fetches", async () => {
    // A rejected promise left in the in-flight map would block every retry for
    // the rest of the session.
    const fetchMock = vi
      .fn()
      .mockRejectedValueOnce(new Error("network down"))
      .mockResolvedValue(okResponse());
    vi.stubGlobal("fetch", fetchMock);

    await expect(fetchCardInfo(ID)).rejects.toThrow("network down");
    expect(getCachedInfo(ID)).toBeUndefined();

    expect(await fetchCardInfo(ID)).toEqual(INFO);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("requests the card endpoint on the shared API base", async () => {
    const fetchMock = vi.fn(() => Promise.resolve(okResponse()));
    vi.stubGlobal("fetch", fetchMock);

    await fetchCardInfo(ID);
    // Assert against API_BASE rather than a hardcoded host: the base is
    // overridable via VITE_API_BASE, and this test is about the path.
    expect(fetchMock).toHaveBeenCalledWith(`${API_BASE}/api/web/card/${ID}`);
  });

  it("notifies subscribers for its own id when a face lands", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.resolve(okResponse()))
    );
    const mine = vi.fn();
    const other = vi.fn();
    subscribeCardInfo(ID, mine);
    subscribeCardInfo(ID + 1, other);

    await fetchCardInfo(ID);

    expect(mine).toHaveBeenCalledTimes(1);
    expect(other).not.toHaveBeenCalled();
  });

  it("notifies on a 404 too, so a consumer stops waiting", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.resolve(notFoundResponse()))
    );
    const listener = vi.fn();
    subscribeCardInfo(ID, listener);

    await fetchCardInfo(ID);

    expect(listener).toHaveBeenCalledTimes(1);
  });

  it("stops notifying after unsubscribe", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.resolve(okResponse()))
    );
    const listener = vi.fn();
    subscribeCardInfo(ID, listener)();

    await fetchCardInfo(ID);

    expect(listener).not.toHaveBeenCalled();
  });
});
