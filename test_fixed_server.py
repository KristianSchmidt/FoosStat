#!/usr/bin/env python3
"""Test the fixed server logic with complete history tracking"""

from server import game_state
from analytics import AnalyticsEngine

def simulate_game_sequence():
    # Reset game state
    game_state.reset()
    
    # Simulate a sequence: r5 -> r3 -> goal, then b5 -> b3 -> goal
    actions = ['r5', 'r3', 'g_r', 'b5', 'b3', 'g_b']
    
    print("Simulating game sequence:", ', '.join(actions))
    print()
    
    for action in actions:
        # Add to both histories (like the server does)
        game_state.possession_history.append(action)
        game_state.complete_history.append(action)
        
        print(f"Action: {action}")
        print(f"Recent history: {game_state.possession_history}")
        print(f"Complete history: {game_state.complete_history}")
        
        # Handle scoring (like the server does)
        if action == 'g_b':
            game_state.blue_score += 1
            game_state.possession_history = ['r5']
            game_state.complete_history.append('r5')
            print("Blue scores! Reset recent history, added kickoff to complete history")
        elif action == 'g_r':
            game_state.red_score += 1
            game_state.possession_history = ['b5']
            game_state.complete_history.append('b5')
            print("Red scores! Reset recent history, added kickoff to complete history")
        
        print("---")
    
    print(f"Final complete history: {game_state.complete_history}")
    print(f"Final recent history: {game_state.possession_history}")
    print()
    
    # Test analytics
    stats = AnalyticsEngine.generate_full_stats(game_state.complete_history)
    
    print("ANALYTICS RESULTS:")
    print("RED TEAM:")
    for stat_name, stat_value in stats["red"].items():
        print(f"  {stat_name}: {stat_value}")
    
    print("\nBLUE TEAM:")
    for stat_name, stat_value in stats["blue"].items():
        print(f"  {stat_name}: {stat_value}")

if __name__ == "__main__":
    simulate_game_sequence()
