"""Manually test the analytics with game state simulation"""

from server import game_state, broadcast_score_update
import asyncio

async def test_manual_game():
    # Simulate some game actions
    actions = ['r5', 'r3', 'g_r', 'b5', 'b3', 'g_b', 'r5', 'b5', 'r3', 'g_r']
    
    print("Testing manual game simulation...")
    
    for action in actions:
        game_state.possession_history.append(action)
        print(f"Added action: {action}")
        print(f"Current history: {game_state.possession_history}")
        
        # Simulate scoring logic
        if action == 'g_r':
            game_state.red_score += 1
            game_state.possession_history = ['b5']  # Reset as in the server
        elif action == 'g_b':
            game_state.blue_score += 1
            game_state.possession_history = ['r5']  # Reset as in the server
        
        print(f"Score: Red {game_state.red_score} - {game_state.blue_score} Blue")
        print("---")

if __name__ == "__main__":
    asyncio.run(test_manual_game())
