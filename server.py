#!/usr/bin/env python3
import asyncio
import json
import uuid
from typing import Dict, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app = FastAPI()

# Store game state
class GameState:
    def __init__(self):
        self.game_id = str(uuid.uuid4())
        self.blue_score = 0
        self.red_score = 0
        self.blue_sets = 0
        self.red_sets = 0
        self.possession_history: List[str] = []
        
    def reset(self):
        self.game_id = str(uuid.uuid4())
        self.blue_score = 0
        self.red_score = 0
        self.blue_sets = 0
        self.red_sets = 0
        self.possession_history = []

# Global game state
game_state = GameState()

# Connected websockets
connected_websockets: List[WebSocket] = []

async def broadcast_score_update():
    """Broadcast current score to all connected clients"""
    if not connected_websockets:
        return
        
    score_html = f'''
    <div id="score-display" hx-swap-oob="true">
        <div class="w-full p-2 bg-gray-100 text-center mb-2 rounded-xl">
            <div class="flex justify-around">
                <span id="red-score" class="text-red-600 font-bold text-2xl">{game_state.red_score}</span>
                <span class="text-gray-600 font-bold text-2xl">-</span>
                <span id="blue-score" class="text-blue-600 font-bold text-2xl">{game_state.blue_score}</span>
            </div>
            <div class="mt-2 flex justify-around text-sm">
                <span>Sets: <span id="red-sets" class="text-red-600 font-bold">{game_state.red_sets}</span></span>
                <span>Sets: <span id="blue-sets" class="text-blue-600 font-bold">{game_state.blue_sets}</span></span>
            </div>
        </div>
    </div>
    '''
    
    # Game history update
    recent_history = game_state.possession_history[-10:] if len(game_state.possession_history) > 0 else []
    history_text = ', '.join(recent_history) if recent_history else 'Game history will appear here'
    
    history_html = f'''
    <div id="game-history" hx-swap-oob="true" class="text-sm text-gray-600 mb-3 p-2 bg-gray-50 rounded-xl min-h-[25px] flex items-center justify-center">
        <p>{history_text}</p>
    </div>
    '''
    print(f"Broadcasting history: {history_text}")
    
    # Also update the stats tab
    recent_possessions = game_state.possession_history[-10:] if len(game_state.possession_history) > 0 else []
    
    stats_html = f'''
    <div id="stats-content" hx-swap-oob="true" class="p-4 bg-gray-50 rounded-xl">
        <p class="text-lg mb-2">Game ID: {game_state.game_id}</p>
        <p class="text-lg mb-2">Score: Red {game_state.red_score} - {game_state.blue_score} Blue</p>
        <p class="text-lg mb-2">Sets: Red {game_state.red_sets} - {game_state.blue_sets} Blue</p>
        <p class="text-lg mb-2">Total possessions: {len(game_state.possession_history)}</p>
        <div class="mt-4">
            <p class="font-bold mb-2">Recent possessions:</p>
            <p class="text-sm">{', '.join(recent_possessions) if recent_possessions else 'None'}</p>
        </div>
    </div>
    '''
    
    message = score_html + history_html + stats_html
    
    disconnected = []
    for websocket in connected_websockets:
        try:
            await websocket.send_text(message)
            print(f"Sent message to WebSocket: {len(message)} characters")
        except Exception as e:
            print(f"Failed to send to WebSocket: {e}")
            disconnected.append(websocket)
    
    # Remove disconnected websockets
    for ws in disconnected:
        connected_websockets.remove(ws)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_websockets.append(websocket)
    
    # Send initial game ID and score
    await broadcast_score_update()
    
    try:
        while True:
            # Listen for messages from client
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                action = message.get('action')
                
                if action:
                    # Add to possession history
                    game_state.possession_history.append(action)
                    print(f"Action received: {action}, history length: {len(game_state.possession_history)}")
                    
                    # Handle scoring
                    if action == 'g_b':  # Blue goal
                        game_state.blue_score += 1
                        # Reset history and start at Red 5-bar
                        game_state.possession_history = ['r5']
                        if game_state.blue_score >= 5:
                            game_state.blue_sets += 1
                            game_state.blue_score = 0
                            game_state.red_score = 0
                    elif action == 'g_r':  # Red goal
                        game_state.red_score += 1
                        # Reset history and start at Blue 5-bar
                        game_state.possession_history = ['b5']
                        if game_state.red_score >= 5:
                            game_state.red_sets += 1
                            game_state.blue_score = 0
                            game_state.red_score = 0
                    
                    # Broadcast updated score to all clients
                    await broadcast_score_update()
                    
            except json.JSONDecodeError:
                continue
                
    except WebSocketDisconnect:
        connected_websockets.remove(websocket)

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("foosball_layout_htmx.html", "r") as f:
        return HTMLResponse(f.read())

@app.post("/reset")
async def reset_game():
    game_state.reset()
    await broadcast_score_update()
    return ""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)