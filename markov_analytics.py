"""
Markov Model Analytics for Foosball Game Data

This module implements a Markov chain model to analyze foosball possession transitions
and calculate scoring probabilities from different game states.
"""

import numpy as np
import numpy.typing as npt
from typing import Dict, List, Tuple, TypeAlias, Mapping
from collections import defaultdict, Counter
from enum import Enum
from dataclasses import dataclass


class Team(Enum):
    """Represents the two teams/possession states."""
    BLUE = "blue"
    RED = "red"


class GameState(Enum):
    """Represents all possible game states in the Markov chain."""
    # Transient states (possession states)
    BLUE_TWO = "b2"
    BLUE_THREE = "b3"
    BLUE_FIVE = "b5"
    RED_TWO = "r2"
    RED_THREE = "r3"
    RED_FIVE = "r5"
    # Absorbing states (goal states)
    GOAL_BLUE = "g_b"
    GOAL_RED = "g_r"

    @property
    def is_absorbing(self) -> bool:
        """Check if this state is an absorbing state (goal)."""
        return self in {GameState.GOAL_BLUE, GameState.GOAL_RED}

    @property
    def team(self) -> Team:
        """Get the team associated with this state."""
        if self in {GameState.BLUE_TWO, GameState.BLUE_THREE, GameState.BLUE_FIVE, GameState.GOAL_BLUE}:
            return Team.BLUE
        return Team.RED


@dataclass(frozen=True)
class Score:
    """Represents a game score."""
    red: int
    blue: int


# Type aliases
MatrixF: TypeAlias = npt.NDArray[np.float64]
AbsorptionProb: TypeAlias = Dict[GameState, Dict[GameState, float]]

# Constants
ALL_STATES: Tuple[GameState, ...] = tuple(GameState)
TRANSIENT_STATES: Tuple[GameState, ...] = tuple(s for s in ALL_STATES if not s.is_absorbing)
ABSORBING_STATES: Tuple[GameState, ...] = tuple(s for s in ALL_STATES if s.is_absorbing)
STATE_INDEX: Dict[GameState, int] = {s: i for i, s in enumerate(ALL_STATES)}
INDEX_TO_STATE: Dict[int, GameState] = {i: s for s, i in STATE_INDEX.items()}

# Mapping from string tokens to GameState
TOKEN_TO_STATE: Dict[str, GameState] = {s.value: s for s in GameState}

# Possession weights for team strength analysis
POSSESSION_WEIGHTS: Mapping[GameState, float] = {
    GameState.BLUE_TWO: 0.15,
    GameState.BLUE_THREE: 0.15,
    GameState.BLUE_FIVE: 0.20,
    GameState.RED_TWO: 0.15,
    GameState.RED_THREE: 0.15,
    GameState.RED_FIVE: 0.20,
}


class FoosballMarkovModel:
    """Markov model for foosball possession and scoring analysis."""
    
    def __init__(self):
        # Initialize transition matrix (8x8 for all states)
        self.transition_matrix: MatrixF = np.zeros((len(ALL_STATES), len(ALL_STATES)))
        
        # Track raw transition counts for building the model
        self.transition_counts: Dict[GameState, Counter[GameState]] = defaultdict(Counter)
        
    def parse_game_data(self, filename: str) -> List[List[GameState]]:
        """Parse game data from file and extract possession sequences."""
        sequences: List[List[GameState]] = []
        
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('set ') or not line:
                    continue
                
                # Parse the line - it's just comma-separated states
                tokens = [s.strip() for s in line.split(',')]
                # Filter out invalid states like 'to_b', 'to_r' but keep valid ones
                valid_states = [TOKEN_TO_STATE[t] for t in tokens if t in TOKEN_TO_STATE]
                if len(valid_states) > 1:  # Need at least 2 states for a transition
                    sequences.append(valid_states)
        
        return sequences
    
    def build_transition_matrix(self, sequences: List[List[GameState]]):
        """Build transition matrix from possession sequences."""
        # Count transitions
        for sequence in sequences:
            for from_state, to_state in zip(sequence, sequence[1:]):
                self.transition_counts[from_state][to_state] += 1
        
        # Convert counts to probabilities
        for from_state in TRANSIENT_STATES:
            from_idx = STATE_INDEX[from_state]
            total_transitions = sum(self.transition_counts[from_state].values())
            
            if total_transitions > 0:
                for to_state, count in self.transition_counts[from_state].items():
                    to_idx = STATE_INDEX[to_state]
                    self.transition_matrix[from_idx][to_idx] = count / total_transitions
        
        # Absorbing states stay in themselves
        for absorbing_state in ABSORBING_STATES:
            idx = STATE_INDEX[absorbing_state]
            self.transition_matrix[idx][idx] = 1.0
    
    def get_absorption_probabilities(self) -> AbsorptionProb:
        """Calculate probability of reaching each absorbing state from each transient state."""
        
        # Extract Q (transient to transient) and R (transient to absorbing) matrices
        n_transient = len(TRANSIENT_STATES)
        
        # Q matrix: transitions between transient states
        Q = self.transition_matrix[:n_transient, :n_transient]
        
        # R matrix: transitions from transient to absorbing states
        R = self.transition_matrix[:n_transient, n_transient:]
        
        # Calculate fundamental matrix N = (I - Q)^(-1)
        I = np.eye(n_transient)
        try:
            N = np.linalg.inv(I - Q)
        except np.linalg.LinAlgError:
            # If matrix is singular, use pseudo-inverse
            N = np.linalg.pinv(I - Q)
        
        # Absorption probabilities B = N * R
        B = N @ R
        
        # Convert to dictionary format
        result: AbsorptionProb = {}
        for i, from_state in enumerate(TRANSIENT_STATES):
            result[from_state] = {}
            for j, absorbing_state in enumerate(ABSORBING_STATES):
                result[from_state][absorbing_state] = B[i][j]
        
        return result
    
    def get_expected_steps_to_absorption(self) -> Dict[GameState, float]:
        """Calculate expected number of steps to reach an absorbing state."""
        n_transient = len(TRANSIENT_STATES)
        Q = self.transition_matrix[:n_transient, :n_transient]
        
        I = np.eye(n_transient)
        try:
            N = np.linalg.inv(I - Q)
        except np.linalg.LinAlgError:
            N = np.linalg.pinv(I - Q)
        
        # Expected steps = sum of each row in fundamental matrix
        expected_steps = N.sum(axis=1)
        
        return {state: steps for state, steps in zip(TRANSIENT_STATES, expected_steps)}
    
    def analyze_team_strengths(self) -> Dict[str, float]:
        """Analyze overall team strengths based on scoring probabilities."""
        absorption_probs = self.get_absorption_probabilities()
        
        blue_score_prob = sum(absorption_probs[state][GameState.GOAL_BLUE] * POSSESSION_WEIGHTS[state] 
                             for state in TRANSIENT_STATES)
        red_score_prob = sum(absorption_probs[state][GameState.GOAL_RED] * POSSESSION_WEIGHTS[state] 
                            for state in TRANSIENT_STATES)
        
        return {
            'blue_scoring_probability': blue_score_prob,
            'red_scoring_probability': red_score_prob,
            'blue_advantage': blue_score_prob - red_score_prob
        }

    def calculate_set_winning_probability(self, current_score_red: int, current_score_blue: int, 
                                        current_possession: Team, probabilities: AbsorptionProb) -> float:
        """
        Calculate probability of blue winning the set from current state.
        
        Args:
            current_score_red: Current red team score (0-4)
            current_score_blue: Current blue team score (0-4)  
            current_possession: Team who starts on 5 bar
            probabilities: Dict mapping states to goal probabilities
            
        Returns:
            Probability that blue wins the set (0.0 to 1.0)
        """
        # Memoization for dynamic programming
        memo: Dict[Tuple[int, int, Team], float] = {}
        
        def dp(red_score: int, blue_score: int, possession: Team) -> float:
            # Base cases
            if blue_score >= 5:
                return 1.0  # Blue wins
            if red_score >= 5:
                return 0.0  # Red wins
            
            # Check memo
            state_key = (red_score, blue_score, possession)
            if state_key in memo:
                return memo[state_key]
            
            # Determine possession state (who has ball on 5 bar)
            state = GameState.BLUE_FIVE if possession == Team.BLUE else GameState.RED_FIVE
            
            # Get scoring probabilities from current possession
            blue_goal_prob = probabilities[state][GameState.GOAL_BLUE]
            red_goal_prob = probabilities[state][GameState.GOAL_RED]
            
            # Calculate probability blue wins from this state
            # If blue scores: blue score +1, red gets possession
            # If red scores: red score +1, blue gets possession
            prob_blue_wins = (blue_goal_prob * dp(red_score, blue_score + 1, Team.RED) + 
                             red_goal_prob * dp(red_score + 1, blue_score, Team.BLUE))
            
            memo[state_key] = prob_blue_wins
            return prob_blue_wins
        
        return dp(current_score_red, current_score_blue, current_possession)

    def print_set_probabilities(self, probabilities: AbsorptionProb):
        """Pretty print set winning probabilities for various game states."""
        print("=== Set Winning Probabilities ===\n")
        
        # Common game situations
        scenarios = [
            (0, 0, Team.BLUE, "Start of set (Blue serves)"),
            (0, 0, Team.RED, "Start of set (Red serves)"),
            (2, 2, Team.BLUE, "Tied 2-2 (Blue possession)"),
            (2, 2, Team.RED, "Tied 2-2 (Red possession)"),
            (3, 4, Team.BLUE, "Blue leads 4-3 (Blue possession)"),
            (3, 4, Team.RED, "Blue leads 4-3 (Red possession)"),
            (4, 3, Team.BLUE, "Red leads 4-3 (Blue possession)"),
            (4, 3, Team.RED, "Red leads 4-3 (Red possession)"),
            (4, 4, Team.BLUE, "Tied 4-4 (Blue possession)"),
            (4, 4, Team.RED, "Tied 4-4 (Red possession)")
        ]
        
        print(f"{'Scenario':<35} {'Blue Win %':<12} {'Red Win %':<12}")
        print("-" * 65)
        
        for red_score, blue_score, possession, description in scenarios:
            blue_win_prob = self.calculate_set_winning_probability(
                red_score, blue_score, possession, probabilities
            )
            red_win_prob = 1.0 - blue_win_prob
            
            print(f"{description:<35} {blue_win_prob*100:>8.1f}%    {red_win_prob*100:>8.1f}%")
        
        print()
        
        # Show possession advantage
        print("Possession Advantage Analysis:")
        print("=" * 40)
        
        for red_score in range(5):
            for blue_score in range(5):
                blue_poss_prob = self.calculate_set_winning_probability(
                    red_score, blue_score, Team.BLUE, probabilities
                )
                red_poss_prob = self.calculate_set_winning_probability(
                    red_score, blue_score, Team.RED, probabilities
                )
                advantage = blue_poss_prob - red_poss_prob
                
                print(f"Score {red_score}-{blue_score}: Blue possession advantage = {advantage*100:+.1f}%")
        
        print()

    def print_analysis(self):
        """Print comprehensive analysis of the Markov model."""
        print("=== Foosball Markov Model Analysis ===\n")
        
        # Transition matrix
        print("Transition Matrix:")
        print("States:", [s.value for s in ALL_STATES])
        print(np.round(self.transition_matrix, 3))
        print()
        
        # Absorption probabilities
        absorption_probs = self.get_absorption_probabilities()
        print("Scoring Probabilities from Each State:")
        print(f"{'State':<5} {'Blue Goal':<12} {'Red Goal':<12}")
        print("-" * 35)
        for state in TRANSIENT_STATES:
            blue_prob = absorption_probs[state][GameState.GOAL_BLUE]
            red_prob = absorption_probs[state][GameState.GOAL_RED]
            print(f"{state.value:<5} {blue_prob:<12.3f} {red_prob:<12.3f}")
        print()
        
        # Expected steps to goal
        expected_steps = self.get_expected_steps_to_absorption()
        print("Expected Possessions Until Goal:")
        for state, steps in expected_steps.items():
            print(f"{state.value}: {steps:.2f}")
        print()
        
        # Team strengths
        strengths = self.analyze_team_strengths()
        print("Overall Team Analysis:")
        print(f"Blue scoring probability: {strengths['blue_scoring_probability']:.3f}")
        print(f"Red scoring probability: {strengths['red_scoring_probability']:.3f}")
        print(f"Blue advantage: {strengths['blue_advantage']:+.3f}")
        print()
        
        # Set winning probabilities
        self.print_set_probabilities(absorption_probs)


def main():
    """Example usage with sample data."""
    model = FoosballMarkovModel()
    
    # Parse game data
    sequences = model.parse_game_data('games/mathias_peter_vs_collignon_atha.txt')
    print(f"Parsed {len(sequences)} possession sequences")
    
    # Debug: Show first few sequences
    if sequences:
        print("First few sequences:")
        for i, seq in enumerate(sequences[:3]):
            print(f"  {i+1}: {seq}")
    else:
        # Debug the parsing
        print("Debug: No sequences found. Checking file parsing...")
        with open('games/mathias_peter_vs_collignon_atha.txt', 'r') as f:
            for i, line in enumerate(f):
                if i >= 5:  # Just show first 5 lines
                    break
                line = line.strip()
                print(f"Line {i+1}: '{line}'")
                if ':' in line and not line.startswith('set '):
                    sequence_part = line.split(':', 1)[1].strip()
                    tokens = [s.strip() for s in sequence_part.split(',')]
                    valid_states = [TOKEN_TO_STATE[t] for t in tokens if t in TOKEN_TO_STATE]
                    print(f"  -> Tokens: {tokens}")
                    print(f"  -> Valid: {[s.value for s in valid_states]}")
    
    # Build and analyze model
    model.build_transition_matrix(sequences)
    model.print_analysis()


if __name__ == "__main__":
    main()
