// ─── Yu-Gi-Oh! Duel Socket.IO Server ─────────────────────────────────────────
import { Server as HttpServer } from "http";
import { Server as SocketServer } from "socket.io";
import { ClientToServerEvents, DuelState, PlayerSide, ServerToClientEvents } from "../shared/gameTypes";
import { applyAction, createDuelState } from "./gameEngine";
import { BLUE_EYES_DECK, KAIBA_DECK, YUGI_DECK } from "../shared/starterDecks";
import { fetchDeckCards } from "./deckLoader";

interface RoomSlot {
  socketId: string;
  playerName: string;
  side: PlayerSide;
}

interface DuelRoom {
  roomId: string;
  slots: RoomSlot[];
  state: DuelState | null;
}

const rooms = new Map<string, DuelRoom>();
const socketToRoom = new Map<string, { roomId: string; side: PlayerSide }>();

export function initDuelSocket(httpServer: HttpServer) {
  const io = new SocketServer<ClientToServerEvents, ServerToClientEvents>(httpServer, {
    cors: { origin: "*", methods: ["GET", "POST"] },
    path: "/socket.io",
  });

  io.on("connection", (socket) => {
    console.log(`[Socket] Connected: ${socket.id}`);

    socket.on("join_room", ({ roomId, playerName }) => {
      handleJoinRoom(socket, io, roomId, playerName).catch((err) => {
        console.error("[Socket] join_room error:", err);
        socket.emit("room_error", "Failed to start duel. Please try again.");
      });
    });

    socket.on("game_action", (action) => {
      const info = socketToRoom.get(socket.id);
      if (!info) return;
      const room = rooms.get(info.roomId);
      if (!room || !room.state) return;

      const newState = applyAction(room.state, action, info.side);
      room.state = newState;

      // Broadcast sanitized state to each player
      for (const slot of room.slots) {
        const targetSocket = io.sockets.sockets.get(slot.socketId);
        if (targetSocket) {
          targetSocket.emit("game_state", sanitizeState(newState, slot.side));
        }
      }
    });

    socket.on("disconnect", () => {
      const info = socketToRoom.get(socket.id);
      if (info) {
        const room = rooms.get(info.roomId);
        if (room) {
          socket.to(info.roomId).emit("opponent_disconnected");
          setTimeout(() => {
            const r = rooms.get(info.roomId);
            if (r) {
              r.slots = r.slots.filter((s) => s.socketId !== socket.id);
              if (r.slots.length === 0) rooms.delete(info.roomId);
            }
          }, 30000);
        }
        socketToRoom.delete(socket.id);
      }
      console.log(`[Socket] Disconnected: ${socket.id}`);
    });
  });

  return io;
}

// ─── Async join handler ───────────────────────────────────────────────────────

async function handleJoinRoom(
  socket: ReturnType<SocketServer["sockets"]["sockets"]["get"]> & object,
  io: SocketServer<ClientToServerEvents, ServerToClientEvents>,
  roomId: string,
  playerName: string
) {
  let room = rooms.get(roomId);

  if (!room) {
    room = { roomId, slots: [], state: null };
    rooms.set(roomId, room);
  }

  if (room.slots.length >= 2) {
    (socket as any).emit("room_error", "Room is full.");
    return;
  }

  const side: PlayerSide = room.slots.length === 0 ? "player1" : "player2";
  room.slots.push({ socketId: (socket as any).id, playerName, side });
  socketToRoom.set((socket as any).id, { roomId, side });

  (socket as any).join(roomId);
  console.log(`[Socket] ${playerName} joined room ${roomId} as ${side}`);

  if (room.slots.length === 2) {
    const [p1, p2] = room.slots;

    // Auto-assign Blue-Eyes deck for both players
    console.log(`[Socket] Loading decks for room ${roomId}...`);
    const [p1Deck, p2Deck] = await Promise.all([
      fetchDeckCards(BLUE_EYES_DECK.cardIds),
      fetchDeckCards(BLUE_EYES_DECK.cardIds),
    ]);
    console.log(`[Socket] Decks loaded: P1(${p1Deck.length}) P2(${p2Deck.length})`);

    room.state = createDuelState(
      roomId,
      p1.socketId, p1.playerName, p1Deck,
      p2.socketId, p2.playerName, p2Deck
    );

    // Notify both players
    for (const slot of room.slots) {
      const targetSocket = io.sockets.sockets.get(slot.socketId);
      if (targetSocket) {
        targetSocket.emit("room_joined", {
          roomId,
          side: slot.side,
          state: sanitizeState(room.state, slot.side),
        });
      }
    }
  } else {
    // First player waits
    (socket as any).emit("room_joined", {
      roomId,
      side,
      state: createWaitingState(roomId),
    });
  }
}

// ─── State sanitization (hide opponent's face-down cards) ─────────────────────

function sanitizeState(state: DuelState, viewer: PlayerSide): DuelState {
  const opponentSide: PlayerSide = viewer === "player1" ? "player2" : "player1";
  const opponent = opponentSide === "player1" ? state.player1 : state.player2;

  const sanitizedOpponent = {
    ...opponent,
    hand: opponent.hand.map(() => ({
      instanceId: "hidden",
      id: 0,
      name: "???",
      type: "Unknown",
      frameType: "normal",
      desc: "",
      card_images: [],
    })),
    monsterZones: opponent.monsterZones.map((slot) =>
      slot && slot.faceDown
        ? { ...slot, card: { ...slot.card, name: "???", id: 0, card_images: [] } }
        : slot
    ),
    spellTrapZones: opponent.spellTrapZones.map((slot) =>
      slot && slot.faceDown
        ? { ...slot, card: { ...slot.card, name: "???", id: 0, card_images: [] } }
        : slot
    ),
  };

  if (opponentSide === "player1") return { ...state, player1: sanitizedOpponent };
  return { ...state, player2: sanitizedOpponent };
}

function createWaitingState(roomId: string): DuelState {
  return {
    roomId,
    phase: "DRAW",
    turnNumber: 0,
    activePlayer: "player1",
    player1: { id: "", name: "Waiting...", lifePoints: 8000, hand: [], deck: [], graveyard: [], banished: [], extraDeck: [], monsterZones: [null,null,null,null,null], spellTrapZones: [null,null,null,null,null], fieldZone: null, extraMonsterZone: null, hasNormalSummoned: false, hasDrawn: false },
    player2: { id: "", name: "Waiting...", lifePoints: 8000, hand: [], deck: [], graveyard: [], banished: [], extraDeck: [], monsterZones: [null,null,null,null,null], spellTrapZones: [null,null,null,null,null], fieldZone: null, extraMonsterZone: null, hasNormalSummoned: false, hasDrawn: false },
    winner: null,
    battleStep: null,
    log: ["Waiting for opponent to join..."],
  };
}
