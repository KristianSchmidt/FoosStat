"""Test fifth set scoring logic."""
from markov_analytics import SetState, Team

def test_fifth_set_scoring():
    """Test that fifth set scoring works correctly."""
    print("Testing fifth set scoring logic...\n")
    
    # Regular set - first to 5
    regular_set = SetState(red_score=5, blue_score=4, possession=Team.RED, is_fifth_set=False)
    print(f"Regular set 5-4: Finished = {regular_set.is_finished}, Winner = {regular_set.winner.value}")
    
    # Fifth set - various scenarios
    test_cases = [
        (5, 4, False, "5-4 should continue (need win by 2)"),
        (6, 4, True, "6-4 should be finished (win by 2)"),
        (7, 6, False, "7-6 should continue (need win by 2)"),
        (8, 6, True, "8-6 should be finished (win by 2)"),
        (8, 7, True, "8-7 should be finished (maximum 8 goals)"),
        (7, 5, True, "7-5 should be finished (win by 2)"),
        (6, 6, False, "6-6 should continue (tied)"),
        (5, 5, False, "5-5 should continue (tied)"),
    ]
    
    for red, blue, should_finish, description in test_cases:
        fifth_set = SetState(red_score=red, blue_score=blue, possession=Team.RED, is_fifth_set=True)
        finished = fifth_set.is_finished
        winner = fifth_set.winner.value if finished else "None"
        
        status = "✓" if finished == should_finish else "✗"
        print(f"{status} {description}: Finished = {finished}, Winner = {winner}")
    
    print("\nTesting complete!")

if __name__ == "__main__":
    test_fifth_set_scoring()
