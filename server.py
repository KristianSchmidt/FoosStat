#!/usr/bin/env python3
import asyncio
import json
import logging
import uuid
import os
import glob
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from analytics import AnalyticsEngine
from markov_analytics import LiveGameTracker, Team, GameState as MarkovGameState
import bleach
from domain.models import GameState, Action, GameMode
from domain.services import broadcast_service

app = FastAPI()

# Store game state

# Global game state
game_state = GameState()

async def broadcast_score_update():
    """Broadcast current score to all connected clients"""
    if broadcast_service.get_connection_count(game_state.game_id) == 0:
        return
        
    red_name = game_state.get_red_display_name()
    blue_name = game_state.get_blue_display_name()
        
    score_html = f'''
    <div id="score-display" hx-swap-oob="true">
        <div class="w-full p-2 bg-gray-100 text-center mb-2 rounded-xl">
            <div class="flex justify-around">
                <span id="red-score" class="text-red-600 font-bold text-2xl">{game_state.red_score}</span>
                <span class="text-gray-600 font-bold text-2xl">-</span>
                <span id="blue-score" class="text-blue-600 font-bold text-2xl">{game_state.blue_score}</span>
            </div>
            <div class="mt-2 flex justify-around text-sm">
                <span>{red_name}: <span id="red-sets" class="text-red-600 font-bold">{game_state.red_sets}</span></span>
                <span>{blue_name}: <span id="blue-sets" class="text-blue-600 font-bold">{game_state.blue_sets}</span></span>
            </div>
        </div>
    </div>
    '''
    
    # Game history update with styled possessions
    recent_history = game_state.possession_history[-10:] if len(game_state.possession_history) > 0 else []
    
    def style_possession(possession):
        """Style a possession string with proper colors and capitalization"""
        if possession.startswith('r'):
            # Red possession
            return f'<span class="possession-red">{possession.upper()}</span>'
        elif possession.startswith('b'):
            # Blue possession  
            return f'<span class="possession-blue">{possession.upper()}</span>'
        else:
            return possession
    
    if recent_history:
        styled_history = [style_possession(p) for p in recent_history]
        history_text = ' - '.join(styled_history)
    else:
        history_text = 'Game history will appear here'
    
    history_html = f'''
    <div id="game-history" hx-swap-oob="true" class="text-sm text-gray-600 mb-3 p-2 bg-gray-50 rounded-xl min-h-[25px] flex items-center justify-center">
        <p>{history_text}</p>
    </div>
    '''
    print(f"Broadcasting history: {history_text}")
    
    # Generate comprehensive stats using complete history
    full_stats = AnalyticsEngine.generate_full_stats(game_state.complete_history)
    
    # Build stats HTML
    stats_rows = []
    stat_names = [
        "Goals",
        "Three bar goals/shots", 
        "Three bar goals/possessions",
        "Five bar passes/attempts",
        "Five bar passes/possessions", 
        "Two bar goals",
        "Five bar steals",
        "Two bar clears"
    ]
    
    for stat_name in stat_names:
        red_stat = full_stats["red"].get(stat_name, "0")
        blue_stat = full_stats["blue"].get(stat_name, "0")
        stats_rows.append(f'''
        <tr class="border-b">
            <td class="py-2 px-4 text-left">{stat_name}</td>
            <td class="py-2 px-4 text-center text-red-600 font-mono">{red_stat}</td>
            <td class="py-2 px-4 text-center text-blue-600 font-mono">{blue_stat}</td>
        </tr>
        ''')
    
    # Get markov probabilities
    markov_probs = game_state.markov_tracker.get_current_probabilities()
    
    stats_html = f'''
    <div id="stats-content" hx-swap-oob="true" class="p-4 bg-gray-50 rounded-xl">
        <div class="mb-4">
            <p class="text-lg mb-2">Game ID: {game_state.game_id}</p>
            <p class="text-lg mb-2">Mode: {game_state.game_mode.title()}</p>
            <p class="text-lg mb-2">Score: {red_name} {game_state.red_score} - {game_state.blue_score} {blue_name}</p>
            <p class="text-lg mb-2">Sets: {red_name} {game_state.red_sets} - {game_state.blue_sets} {blue_name}</p>
            <p class="text-lg mb-2">Total possessions: {len(game_state.complete_history)}</p>
        </div>
        
        <div class="mt-6">
            <h3 class="text-xl font-bold mb-4">Advanced Statistics</h3>
            <div class="overflow-x-auto">
                <table class="w-full border-collapse border border-gray-300 bg-white rounded-lg">
                    <thead class="bg-gray-100">
                        <tr>
                            <th class="py-3 px-4 text-left border-b border-gray-300">Statistic</th>
                            <th class="py-3 px-4 text-center border-b border-gray-300 text-red-600">{red_name}</th>
                            <th class="py-3 px-4 text-center border-b border-gray-300 text-blue-600">{blue_name}</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(stats_rows)}
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="mt-6">
            <p class="font-bold mb-2">Recent possessions:</p>
            <p class="text-sm font-mono bg-white p-2 rounded border">{', '.join(game_state.possession_history[-15:]) if game_state.possession_history else 'None'}</p>
        </div>
    </div>
    '''
    
    # Generate markov stats content
    markov_html = f'''
    <div id="markov-content" hx-swap-oob="true" class="p-4 bg-gray-50 rounded-xl">
        <div class="mb-4">
            <h3 class="text-xl font-bold mb-4">Markov Analytics</h3>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div class="bg-white p-4 rounded-lg border">
                    <h4 class="font-bold mb-2">Current Set Win Probability</h4>
                    <div class="text-center">
                        <div class="flex justify-between items-center mb-2">
                            <span class="text-red-600 font-mono">{red_name}</span>
                            <span class="text-blue-600 font-mono">{blue_name}</span>
                        </div>
                        <div class="bg-blue-500 rounded-full h-6 mb-2 flex">
                            <div class="bg-red-500 h-6 rounded-l-full" style="width: {markov_probs['set_win_prob_red']*100:.1f}%"></div>
                            <div class="bg-blue-500 h-6 rounded-r-full" style="width: {markov_probs['set_win_prob_blue']*100:.1f}%"></div>
                        </div>
                        <div class="flex justify-between text-sm">
                            <span class="text-red-600">{markov_probs['set_win_prob_red']*100:.1f}%</span>
                            <span class="text-blue-600">{markov_probs['set_win_prob_blue']*100:.1f}%</span>
                        </div>
                    </div>
                </div>
                
                <div class="bg-white p-4 rounded-lg border">
                    <h4 class="font-bold mb-2">Match Win Probability</h4>
                    <div class="text-center">
                        <div class="flex justify-between items-center mb-2">
                            <span class="text-red-600 font-mono">{red_name}</span>
                            <span class="text-blue-600 font-mono">{blue_name}</span>
                        </div>
                        <div class="bg-blue-500 rounded-full h-6 mb-2 flex">
                            <div class="bg-red-500 h-6 rounded-l-full" style="width: {markov_probs['game_win_prob_red']*100:.1f}%"></div>
                            <div class="bg-blue-500 h-6 rounded-r-full" style="width: {markov_probs['game_win_prob_blue']*100:.1f}%"></div>
                        </div>
                        <div class="flex justify-between text-sm">
                            <span class="text-red-600">{markov_probs['game_win_prob_red']*100:.1f}%</span>
                            <span class="text-blue-600">{markov_probs['game_win_prob_blue']*100:.1f}%</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    '''
    
    message = score_html + history_html + stats_html + markov_html
    
    # Send to all websockets connected to this game
    sent_count = await broadcast_service.send_to_game(message, game_state.game_id)
    print(f"Broadcasted score update to {sent_count} clients")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await broadcast_service.connect(websocket, game_state.game_id)
    
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
                    try:
                        # Validate action using enum
                        parsed_action = Action(action)
                        
                        # Add to both possession histories
                        game_state.add_possession(parsed_action)
                        print(f"Action received: {action}, complete history length: {len(game_state.complete_history)}")
                        
                        # Convert action to markov state and add to tracker
                        if parsed_action in [Action.B2, Action.B3, Action.B5, Action.R2, Action.R3, Action.R5]:
                            markov_state = MarkovGameState(action)
                            game_state.markov_tracker.add_possession(markov_state)
                        
                        # Handle scoring
                        if parsed_action == Action.BLUE_GOAL:
                            game_state.markov_tracker.score_goal(Team.BLUE)
                            game_state.goal_scored("blue")
                        elif parsed_action == Action.RED_GOAL:
                            game_state.markov_tracker.score_goal(Team.RED)
                            game_state.goal_scored("red")
                            
                    except ValueError:
                        print(f"Invalid action received: {action}")
                        continue
                    
                    # Broadcast updated score to all clients
                    await broadcast_score_update()
                    
            except json.JSONDecodeError:
                continue
                
    except WebSocketDisconnect:
        await broadcast_service.disconnect(websocket, game_state.game_id)

@app.get("/available-games")
async def get_available_games():
    """Get list of available pre-played games"""
    games = []
    try:
        for filepath in glob.glob("games/*.txt"):
            filename = os.path.basename(filepath)
            # Create a display name from the filename
            display_name = filename.replace(".txt", "").replace("_", " ").title()
            games.append({
                "filename": filename,
                "display_name": display_name
            })
    except Exception as e:
        print(f"Error loading games: {e}")
    
    return JSONResponse(games)

@app.post("/load-game")
async def load_game(request: dict):
    """Load a pre-played game from file"""
    try:
        filename = request.get("filename")
        if not filename:
            return JSONResponse({"error": "No filename provided"}, status_code=400)
        
        filepath = os.path.join("games", filename)
        if not os.path.exists(filepath):
            return JSONResponse({"error": "Game file not found"}, status_code=404)
        
        # Parse the game file
        with open(filepath, "r") as f:
            lines = f.readlines()
        
        # Reset game state
        game_state.reset()
        
        # Set default names based on filename patterns
        if "hannibal" in filename.lower():
            game_state.red_player1 = "Sven Wonsyld"
            game_state.blue_player1 = "Hannibal Keblovski"
        elif "mathias" in filename.lower():
            game_state.red_player1 = "Mathias Møller"
            game_state.blue_player1 = "Frederic Collignon"
        else:
            game_state.red_player1 = "Red Team"
            game_state.blue_player1 = "Blue Team"
        
        # Process each line
        for line in lines:
            line = line.strip()
            if not line or line.startswith("set "):
                continue
                
            # Parse possession line
            possessions = line.split(",")
            for possession in possessions:
                possession = possession.strip()
                if possession:
                    game_state.complete_history.append(possession)
                    game_state.possession_history.append(possession)
                    
                    # Update scores based on goals
                    if possession == "g_r":
                        game_state.red_score += 1
                    elif possession == "g_b":
                        game_state.blue_score += 1
                    
                    # Check for set completion (assuming 5 points per set)
                    if game_state.red_score >= 5:
                        game_state.red_sets += 1
                        game_state.red_score = 0
                        game_state.blue_score = 0
                    elif game_state.blue_score >= 5:
                        game_state.blue_sets += 1
                        game_state.red_score = 0
                        game_state.blue_score = 0
        
        # Update markov tracker with complete history
        for possession in game_state.complete_history:
            logging.info(f"Adding possession to Markov tracker: {possession}")
            if possession in ["b2", "b3", "b5", "r2", "r3", "r5"]:
                markov_state = MarkovGameState(possession)
                game_state.markov_tracker.add_possession(markov_state)
            elif possession == "g_b":
                game_state.markov_tracker.score_goal(Team.BLUE)
            elif possession == "g_r":
                game_state.markov_tracker.score_goal(Team.RED)
        
        return JSONResponse({"redirect_url": "/game"})
        
    except Exception as e:
        print(f"Error loading game: {e}")
        return JSONResponse({"error": "Error loading game"}, status_code=500)

@app.get("/", response_class=HTMLResponse)
async def get_splash():
    with open("splash.html", "r") as f:
        return HTMLResponse(f.read())

@app.post("/start-game")
async def start_game(
    game_mode: str = Form(...),
    red_player1: str = Form(...),
    blue_player1: str = Form(...),
    red_player2: str = Form(""),
    blue_player2: str = Form("")
):
    # Update game state with player info
    game_state.game_mode = GameMode(game_mode)
    game_state.red_player1 = bleach.clean(red_player1.strip())
    game_state.blue_player1 = bleach.clean(blue_player1.strip())
    game_state.red_player2 = bleach.clean(red_player2.strip()) if red_player2.strip() else None
    game_state.blue_player2 = bleach.clean(blue_player2.strip()) if blue_player2.strip() else None
    
    # Reset scores for new game
    game_state.reset()
    
    # Redirect to game page to ensure fresh websocket connection
    return RedirectResponse(url="/game", status_code=303)

@app.get("/game", response_class=HTMLResponse)
async def get_game():
    return HTMLResponse(get_game_html())

def get_game_html():
    """Generate game HTML with current player names"""
    with open("foosball_game_template.html", "r") as f:
        template = f.read()
    
    red_name = game_state.get_red_display_name()
    blue_name = game_state.get_blue_display_name()
    
    # Replace placeholders with actual names
    html = template.replace("{{RED_NAME}}", red_name)
    html = html.replace("{{BLUE_NAME}}", blue_name)
    
    return html

@app.post("/reset")
async def reset_game():
    game_state.reset()
    await broadcast_score_update()
    return ""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)