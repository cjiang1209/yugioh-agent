import { useCallback, useEffect, useRef, useState } from "react";
import { io, Socket } from "socket.io-client";
import {
  ClientToServerEvents,
  DuelState,
  GameAction,
  PlayerSide,
  ServerToClientEvents,
} from "../../../shared/gameTypes";

type DuelSocket = Socket<ServerToClientEvents, ClientToServerEvents>;

export type ConnectionStatus = "disconnected" | "connecting" | "waiting" | "dueling" | "ended";

export interface UseDuelSocketReturn {
  state: DuelState | null;
  mySide: PlayerSide | null;
  status: ConnectionStatus;
  roomId: string | null;
  joinRoom: (roomId: string, playerName: string) => void;
  sendAction: (action: GameAction) => void;
  disconnect: () => void;
  lastLog: string;
}

export function useDuelSocket(): UseDuelSocketReturn {
  const socketRef = useRef<DuelSocket | null>(null);
  const [state, setState] = useState<DuelState | null>(null);
  const [mySide, setMySide] = useState<PlayerSide | null>(null);
  const [status, setStatus] = useState<ConnectionStatus>("disconnected");
  const [roomId, setRoomId] = useState<string | null>(null);
  const [lastLog, setLastLog] = useState<string>("");

  useEffect(() => {
    return () => {
      socketRef.current?.disconnect();
    };
  }, []);

  const joinRoom = useCallback((rid: string, playerName: string) => {
    if (socketRef.current) {
      socketRef.current.disconnect();
    }

    setStatus("connecting");
    setRoomId(rid);

    const socket: DuelSocket = io(window.location.origin, {
      path: "/socket.io",
      transports: ["websocket", "polling"],
    });

    socketRef.current = socket;

    socket.on("connect", () => {
      socket.emit("join_room", { roomId: rid, playerName });
    });

    socket.on("room_joined", ({ side, state: initialState }) => {
      setMySide(side);
      setState(initialState);
      const hasOpponent = initialState.player1.id !== "" && initialState.player2.id !== "";
      setStatus(hasOpponent ? "dueling" : "waiting");
    });

    socket.on("game_state", (newState) => {
      setState(newState);
      if (newState.winner) {
        setStatus("ended");
      } else {
        setStatus("dueling");
      }
      if (newState.log.length > 0) {
        setLastLog(newState.log[newState.log.length - 1]);
      }
    });

    socket.on("opponent_connected", (name) => {
      setStatus("dueling");
      setLastLog(`${name} has joined the duel!`);
    });

    socket.on("opponent_disconnected", () => {
      setLastLog("Opponent disconnected.");
      setStatus("ended");
    });

    socket.on("room_error", (msg) => {
      setLastLog(`Error: ${msg}`);
      setStatus("disconnected");
    });

    socket.on("duel_log", (msg) => {
      setLastLog(msg);
    });

    socket.on("disconnect", () => {
      setStatus("disconnected");
    });
  }, []);

  const sendAction = useCallback((action: GameAction) => {
    socketRef.current?.emit("game_action", action);
  }, []);

  const disconnect = useCallback(() => {
    socketRef.current?.disconnect();
    setStatus("disconnected");
    setState(null);
    setMySide(null);
    setRoomId(null);
  }, []);

  return { state, mySide, status, roomId, joinRoom, sendAction, disconnect, lastLog };
}
