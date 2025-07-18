import asyncio
from typing import Dict, Set
from fastapi import WebSocket


class BroadcastService:
    """WebSocket broadcast service with multi-game support"""
    
    def __init__(self):
        # Dict mapping game_id to set of connected websockets
        self._connections: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, game_id: str) -> None:
        """Connect a websocket to a specific game"""
        async with self._lock:
            if game_id not in self._connections:
                self._connections[game_id] = set()
            self._connections[game_id].add(websocket)
    
    async def disconnect(self, websocket: WebSocket, game_id: str) -> None:
        """Disconnect a websocket from a specific game"""
        async with self._lock:
            if game_id in self._connections:
                self._connections[game_id].discard(websocket)
                # Clean up empty game connections
                if not self._connections[game_id]:
                    del self._connections[game_id]
    
    async def send_to_game(self, message: str, game_id: str) -> int:
        """
        Send message to all websockets connected to a specific game
        
        Returns:
            int: Number of successful sends
        """
        if game_id not in self._connections:
            return 0
        
        connections = self._connections[game_id].copy()  # Copy to avoid modification during iteration
        disconnected = []
        sent_count = 0
        
        for websocket in connections:
            try:
                await websocket.send_text(message)
                sent_count += 1
                print(f"Sent message to WebSocket in game {game_id}: {len(message)} characters")
            except Exception as e:
                print(f"Failed to send to WebSocket in game {game_id}: {e}")
                disconnected.append(websocket)
        
        # Remove disconnected websockets
        if disconnected:
            async with self._lock:
                if game_id in self._connections:
                    for ws in disconnected:
                        self._connections[game_id].discard(ws)
                    # Clean up empty game connections
                    if not self._connections[game_id]:
                        del self._connections[game_id]
        
        return sent_count
    
    def get_connection_count(self, game_id: str) -> int:
        """Get number of connections for a specific game"""
        return len(self._connections.get(game_id, set()))
    
    def get_total_connections(self) -> int:
        """Get total number of connections across all games"""
        return sum(len(connections) for connections in self._connections.values())
    
    def get_active_games(self) -> Set[str]:
        """Get set of game IDs that have active connections"""
        return set(self._connections.keys())


# Global broadcast service instance
broadcast_service = BroadcastService()
