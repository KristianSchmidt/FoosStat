"""
Markov Model Analytics for Foosball Game Data

This module implements a Markov chain model to analyze foosball possession transitions
and calculate scoring probabilities from different game states.
"""

import numpy as np
import numpy.typing as npt
from typing import Dict, List, Tuple, TypeAlias, Mapping, Optional
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


@dataclass(frozen=True)
class SetState:
    """Represents the state of a single set."""
    red_score: int
    blue_score: int
    possession: Team
    is_fifth_set: bool = False
    
    @property
    def is_finished(self) -> bool:
        """Check if the set is complete."""
        if self.is_fifth_set:
            # Fifth set: win by 2 goals with maximum of 8 goals
            if max(self.red_score, self.blue_score) >= 8:
                return True
            if max(self.red_score, self.blue_score) >= 5:
                return abs(self.red_score - self.blue_score) >= 2
            return False
        else:
            # Regular sets: first to 5 goals
            return self.red_score >= 5 or self.blue_score >= 5
    
    @property
    def winner(self) -> Team:
        """Get the winner of the set (only valid if is_finished is True)."""
        if not self.is_finished:
            raise ValueError("Set is not finished")
        return Team.BLUE if self.blue_score > self.red_score else Team.RED


@dataclass(frozen=True)
class FullGameState:
    """Represents the complete game state."""
    sets_won_red: int
    sets_won_blue: int
    current_set: SetState
    
    @property
    def is_finished(self) -> bool:
        """Check if the game is complete (best of 5)."""
        return self.sets_won_red >= 3 or self.sets_won_blue >= 3
    
    @property
    def winner(self) -> Team:
        """Get the winner of the game (only valid if is_finished is True)."""
        if not self.is_finished:
            raise ValueError("Game is not finished")
        return Team.BLUE if self.sets_won_blue >= 3 else Team.RED
    
    @property
    def is_fifth_set(self) -> bool:
        """Check if we're in the fifth set (both teams have won 2 sets)."""
        return self.sets_won_red == 2 and self.sets_won_blue == 2


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
        for from_state in ALL_STATES:
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
                                        current_possession: Team, probabilities: AbsorptionProb, 
                                        is_fifth_set: bool = False) -> float:
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
            if is_fifth_set:
                # Fifth set: win by 2 goals with maximum of 8 goals
                if max(red_score, blue_score) >= 8:
                    return 1.0 if blue_score > red_score else 0.0
                if max(red_score, blue_score) >= 5:
                    if abs(red_score - blue_score) >= 2:
                        return 1.0 if blue_score > red_score else 0.0
                # Game continues
            else:
                # Regular sets: first to 5 goals
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
            (0, 0, Team.BLUE, "Start of set (Blue serves)", False),
            (0, 0, Team.RED, "Start of set (Red serves)", False),
            (2, 2, Team.BLUE, "Tied 2-2 (Blue possession)", False),
            (2, 2, Team.RED, "Tied 2-2 (Red possession)", False),
            (3, 4, Team.BLUE, "Blue leads 4-3 (Blue possession)", False),
            (3, 4, Team.RED, "Blue leads 4-3 (Red possession)", False),
            (4, 3, Team.BLUE, "Red leads 4-3 (Blue possession)", False),
            (4, 3, Team.RED, "Red leads 4-3 (Red possession)", False),
            (4, 4, Team.BLUE, "Tied 4-4 (Blue possession)", False),
            (4, 4, Team.RED, "Tied 4-4 (Red possession)", False),
            (5, 5, Team.BLUE, "Fifth set 5-5 (Blue possession)", True),
            (5, 5, Team.RED, "Fifth set 5-5 (Red possession)", True),
            (6, 6, Team.BLUE, "Fifth set 6-6 (Blue possession)", True),
            (6, 6, Team.RED, "Fifth set 6-6 (Red possession)", True),
            (7, 6, Team.BLUE, "Fifth set 7-6 (Blue possession)", True),
            (7, 6, Team.RED, "Fifth set 7-6 (Red possession)", True),
        ]
        
        print(f"{'Scenario':<35} {'Blue Win %':<12} {'Red Win %':<12}")
        print("-" * 65)
        
        for red_score, blue_score, possession, description, is_fifth_set in scenarios:
            blue_win_prob = self.calculate_set_winning_probability(
                red_score, blue_score, possession, probabilities, is_fifth_set
            )
            red_win_prob = 1.0 - blue_win_prob
            
            print(f"{description:<35} {blue_win_prob*100:>8.1f}%    {red_win_prob*100:>8.1f}%")
        
        print()
        
        # Show possession advantage
        # print("Possession Advantage Analysis:")
        # print("=" * 40)
        
        # for red_score in range(5):
        #     for blue_score in range(5):
        #         blue_poss_prob = self.calculate_set_winning_probability(
        #             red_score, blue_score, Team.BLUE, probabilities
        #         )
        #         red_poss_prob = self.calculate_set_winning_probability(
        #             red_score, blue_score, Team.RED, probabilities
        #         )
        #         advantage = blue_poss_prob - red_poss_prob
                
        #         print(f"Score {red_score}-{blue_score}: Blue possession advantage = {advantage*100:+.1f}%")
        
        # print()

    def print_analysis(self):
        """Print comprehensive analysis of the Markov model."""
        print("=== Foosball Markov Model Analysis ===\n")
        
        # Transition matrix
        print("Transition Matrix:")
        matrix_rounded = np.round(self.transition_matrix, 3)
        
        # Create pretty-printed matrix with row and column labels
        state_labels = [s.value for s in ALL_STATES]
        
        # Print column headers
        print("       " + "".join(f"{label:>8}" for label in state_labels))
        print("     " + "-" * (8 * len(state_labels) + 2))
        
        # Print each row with row label
        for i, row_label in enumerate(state_labels):
            row_values = "".join(f"{matrix_rounded[i][j]:8.3f}" for j in range(len(state_labels)))
            print(f"{row_label:>4} | {row_values}")
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


class LiveGameTracker:
    """Tracks a live game and calculates win probabilities using incremental data."""
    
    def __init__(self):
        self.game_state = FullGameState(
            sets_won_red=0,
            sets_won_blue=0,
            current_set=SetState(red_score=0, blue_score=0, possession=Team.BLUE, is_fifth_set=False)
        )
        # Track ALL possessions throughout the entire game
        self.all_possession_sequences: List[List[GameState]] = []
        self.current_possession_sequence: List[GameState] = []
        self.markov_model = FoosballMarkovModel()
        
        # Performance optimizations
        self._cached_model: Optional[FoosballMarkovModel] = None
        self._cached_model_total_sequences: int = 0
        self._cached_absorption_probs: Optional[AbsorptionProb] = None
        
        # Confidence tracking
        self.total_possessions: int = 0
        self.total_goals: int = 0
        
    def add_possession(self, state: GameState):
        """Add a possession to the current sequence."""
        self.current_possession_sequence.append(state)
        self.total_possessions += 1
        
    def score_goal(self, scoring_team: Team):
        """Record a goal and update game state."""
        # Add goal state to current sequence
        goal_state = GameState.GOAL_BLUE if scoring_team == Team.BLUE else GameState.GOAL_RED
        self.current_possession_sequence.append(goal_state)
        
        # Add current possession sequence to complete history
        if len(self.current_possession_sequence) > 1:
            self.all_possession_sequences.append(self.current_possession_sequence.copy())
        
        # Invalidate cache since we have new data
        self._cached_model = None
        self._cached_absorption_probs = None
        
        # Update counters
        self.total_goals += 1
        
        # Update set score
        if scoring_team == Team.BLUE:
            new_blue_score = self.game_state.current_set.blue_score + 1
            new_red_score = self.game_state.current_set.red_score
        else:
            new_blue_score = self.game_state.current_set.blue_score
            new_red_score = self.game_state.current_set.red_score + 1
        
        # Determine who gets possession after goal (opponent serves)
        next_possession = Team.RED if scoring_team == Team.BLUE else Team.BLUE
        
        # Create new set state
        new_set = SetState(
            red_score=new_red_score,
            blue_score=new_blue_score,
            possession=next_possession,
            is_fifth_set=self.game_state.is_fifth_set
        )
        
        # Check if set is finished
        if new_set.is_finished:
            # Update sets won
            if scoring_team == Team.BLUE:
                new_sets_blue = self.game_state.sets_won_blue + 1
                new_sets_red = self.game_state.sets_won_red
            else:
                new_sets_blue = self.game_state.sets_won_blue
                new_sets_red = self.game_state.sets_won_red + 1
            
            # Start new set (winner of previous set serves)
            # Check if next set will be the fifth set
            is_next_fifth_set = new_sets_red == 2 and new_sets_blue == 2
            new_set = SetState(red_score=0, blue_score=0, possession=scoring_team, is_fifth_set=is_next_fifth_set)
            
            self.game_state = FullGameState(
                sets_won_red=new_sets_red,
                sets_won_blue=new_sets_blue,
                current_set=new_set
            )
        else:
            # Update current set
            self.game_state = FullGameState(
                sets_won_red=self.game_state.sets_won_red,
                sets_won_blue=self.game_state.sets_won_blue,
                current_set=new_set
            )
        
        # Reset possession sequence for next point
        self.current_possession_sequence = []
        
    def build_incremental_model(self) -> FoosballMarkovModel:
        """Build Markov model from ALL possession history so far in the entire game."""
        # Include current in-progress sequence if it has useful data
        all_sequences = self.all_possession_sequences.copy()
        if len(self.current_possession_sequence) > 1:
            all_sequences.append(self.current_possession_sequence.copy())
        
        # Use cached model if available and data hasn't changed
        total_sequences = len(all_sequences)
        if (self._cached_model is not None and 
            self._cached_model_total_sequences == total_sequences):
            return self._cached_model
        
        model = FoosballMarkovModel()
        model.build_transition_matrix(all_sequences)
        
        # Cache the model
        self._cached_model = model
        self._cached_model_total_sequences = total_sequences
        
        return model
        
    def calculate_set_win_probability(self) -> float:
        """Calculate probability of blue winning current set."""
        if self.game_state.current_set.is_finished:
            return 1.0 if self.game_state.current_set.winner == Team.BLUE else 0.0
            
        # Get current scores
        blue_score = self.game_state.current_set.blue_score
        red_score = self.game_state.current_set.red_score
        total_score = blue_score + red_score
        
        # Handle case where we don't have enough data yet
        if len(self.all_possession_sequences) < 3:
            if total_score == 0:
                return 0.5  # No data, assume equal probability
            
            # Simple heuristic: current score proportion with some uncertainty
            print("here1")
            return min(max(blue_score / total_score, 0.1), 0.9)
        
        # Build model from data so far
        model = self.build_incremental_model()
        
        # Check if model has valid data
        if len(model.transition_counts) == 0:
            # No transitions recorded, fall back to score-based
            if total_score == 0:
                return 0.5
            print("here2")
            return min(max(blue_score / total_score, 0.1), 0.9)
        
        # Use Markov model
        try:
            absorption_probs = model.get_absorption_probabilities()
            
            # Check if absorption probabilities are reasonable
            has_valid_probs = any(
                sum(absorption_probs[state].values()) > 0.001
                for state in absorption_probs if state in absorption_probs
            )
            
            if not has_valid_probs:
                # Fall back to score-based heuristic
                if total_score == 0:
                    return 0.5
                print("here3")
                return min(max(blue_score / total_score, 0.1), 0.9)
            
            markov_result = model.calculate_set_winning_probability(
                red_score,
                blue_score,
                self.game_state.current_set.possession,
                absorption_probs,
                self.game_state.is_fifth_set
            )
            
            # Debug: Check if result seems reasonable
            if total_score > 0:
                score_based = blue_score / total_score
                # If Markov result is drastically different from score-based and score difference is significant
                if abs(markov_result - score_based) > 0.6 and abs(blue_score - red_score) >= 2:
                    # Blend the results when there's high disagreement and clear score advantage
                    print("here4")
                    return 0.6 * markov_result + 0.4 * min(max(score_based, 0.1), 0.9)
            
            return markov_result
            
        except Exception:
            # If anything fails, use score-based fallback
            if total_score == 0:
                return 0.5
            print("here5")
            return min(max(blue_score / total_score, 0.1), 0.9)
    
    def calculate_game_win_probability(self) -> float:
        """Calculate probability of blue winning the entire game."""
        if self.game_state.is_finished:
            return 1.0 if self.game_state.winner == Team.BLUE else 0.0
        
        # Get current set probability
        set_prob = self.calculate_set_win_probability()
        
        # Use dynamic programming for game-level calculation
        memo: Dict[Tuple[int, int, float], float] = {}
        
        def dp(blue_sets: int, red_sets: int, current_set_prob: float) -> float:
            # Base cases
            if blue_sets >= 3:
                return 1.0
            if red_sets >= 3:
                return 0.0
            
            # Check memo
            state_key = (blue_sets, red_sets, round(current_set_prob, 3))
            if state_key in memo:
                return memo[state_key]
            
            # Calculate probability
            # If blue wins current set
            prob_blue_wins_set = current_set_prob * dp(blue_sets + 1, red_sets, 0.5)
            # If red wins current set  
            prob_red_wins_set = (1 - current_set_prob) * dp(blue_sets, red_sets + 1, 0.5)
            
            result = prob_blue_wins_set + prob_red_wins_set
            memo[state_key] = result
            return result
        
        return dp(
            self.game_state.sets_won_blue,
            self.game_state.sets_won_red,
            set_prob
        )
    
    def calculate_confidence_metrics(self) -> Dict[str, float]:
        """Calculate sophisticated confidence metrics."""
        # Data quantity confidence
        data_confidence = min(len(self.all_possession_sequences) / 15.0, 1.0)
        
        # Data quality confidence (based on transition diversity)
        quality_confidence = 1.0
        if len(self.all_possession_sequences) > 0:
            model = self.build_incremental_model()
            # Check how many different transitions we've seen
            total_transitions = sum(
                sum(model.transition_counts[from_state].values()) 
                for from_state in model.transition_counts
            )
            unique_transitions = sum(
                len(model.transition_counts[from_state]) 
                for from_state in model.transition_counts
            )
            
            if total_transitions > 0:
                quality_confidence = min(unique_transitions / total_transitions, 1.0)
        
        # Game progress confidence (later in game = more confident)
        game_progress = (self.total_goals) / 20.0  # Assume ~20 goals per game
        progress_confidence = min(game_progress, 1.0)
        
        # Combined confidence
        overall_confidence = (data_confidence * 0.4 + 
                            quality_confidence * 0.3 + 
                            progress_confidence * 0.3)
        
        return {
            'data_confidence': data_confidence,
            'quality_confidence': quality_confidence, 
            'progress_confidence': progress_confidence,
            'overall_confidence': overall_confidence
        }
    
    def get_current_probabilities(self) -> Dict[str, float]:
        """Get current set and game win probabilities."""
        set_prob = self.calculate_set_win_probability()
        game_prob = self.calculate_game_win_probability()
        confidence_metrics = self.calculate_confidence_metrics()
        
        return {
            'set_win_prob_blue': set_prob,
            'set_win_prob_red': 1.0 - set_prob,
            'game_win_prob_blue': game_prob,
            'game_win_prob_red': 1.0 - game_prob,
            'confidence': confidence_metrics['overall_confidence'],
            'confidence_breakdown': confidence_metrics
        }
    
    def format_live_update(self, detailed: bool = False) -> str:
        """Format current state for live display."""
        probs = self.get_current_probabilities()
        
        # Game state summary
        sets_display = f"Sets: {self.game_state.sets_won_red}-{self.game_state.sets_won_blue}"
        score_display = f"Score: {self.game_state.current_set.red_score}-{self.game_state.current_set.blue_score}"
        
        # Probability display with confidence indicators
        conf = probs['confidence']
        conf_indicator = "🔴" if conf < 0.3 else "🟡" if conf < 0.7 else "🟢"
        
        # Format probabilities as percentages
        set_blue_pct = probs['set_win_prob_blue'] * 100
        set_red_pct = probs['set_win_prob_red'] * 100
        game_blue_pct = probs['game_win_prob_blue'] * 100
        game_red_pct = probs['game_win_prob_red'] * 100
        
        # Create progress bars
        def create_bar(blue_pct: float, width: int = 20) -> str:
            blue_width = int(blue_pct * width / 100)
            red_width = width - blue_width
            return "█" * blue_width + "▓" * red_width
        
        set_bar = create_bar(set_blue_pct)
        game_bar = create_bar(game_blue_pct)
        
        # Basic display - match score order (Red-Blue)
        output = [
            f"{conf_indicator} {sets_display} | {score_display}",
            f"Set:  [{set_bar}] R:{set_red_pct:5.1f}% B:{set_blue_pct:5.1f}%",
            f"Game: [{game_bar}] R:{game_red_pct:5.1f}% B:{game_blue_pct:5.1f}%",
            f"Confidence: {conf:.3f}"
        ]
        
        # Detailed display
        if detailed:
            breakdown = probs['confidence_breakdown']
            output.extend([
                "",
                f"Data Quality: {breakdown['data_confidence']:.2f}",
                f"Transition Diversity: {breakdown['quality_confidence']:.2f}",
                f"Game Progress: {breakdown['progress_confidence']:.2f}",
                f"Possession: {self.game_state.current_set.possession.value}",
                f"Total Goals: {self.total_goals}, Possessions: {self.total_possessions}"
            ])
        
        return "\n".join(output)
    
    def get_data_stats(self) -> Dict[str, int]:
        """Get statistics about the data being used for calculations."""
        all_sequences = self.all_possession_sequences.copy()
        if len(self.current_possession_sequence) > 1:
            all_sequences.append(self.current_possession_sequence.copy())
        
        total_possessions_in_sequences = sum(len(seq) for seq in all_sequences)
        
        return {
            'total_sequences': len(all_sequences),
            'completed_sequences': len(self.all_possession_sequences),
            'current_sequence_length': len(self.current_possession_sequence),
            'total_possessions_in_model': total_possessions_in_sequences,
            'total_possessions_tracked': self.total_possessions,
            'total_goals': self.total_goals
        }


def demo_live_tracker():
    """Demonstrate live game tracking with historical data."""
    print("=== Live Game Tracker Demo ===\n")
    
    tracker = LiveGameTracker()
    
    # Simulate a game using historical data
    with open('games/data.txt', 'r') as f:
        current_set = 1
        
        for line in f:
            line = line.strip()
            if line.startswith('set '):
                current_set = int(line.split()[1])
                print(f"\n{'='*50}")
                print(f"SET {current_set}")
                print(f"{'='*50}")
                continue
            elif not line:
                continue
            
            # Parse possession sequence
            tokens = [s.strip() for s in line.split(',')]
            valid_states = [TOKEN_TO_STATE[t] for t in tokens if t in TOKEN_TO_STATE]
            
            if len(valid_states) < 2:
                continue
                
            # Add possessions to tracker
            for i in range(len(valid_states) - 1):
                tracker.add_possession(valid_states[i])
            
            # Determine who scored
            final_state = valid_states[-1]
            if final_state == GameState.GOAL_BLUE:
                scoring_team = Team.BLUE
            elif final_state == GameState.GOAL_RED:
                scoring_team = Team.RED
            else:
                continue  # Not a goal
            
            # Record the goal
            tracker.score_goal(scoring_team)
            
            # Show enhanced live display
            print(tracker.format_live_update(detailed=False))
            
            # Show data stats for the first few goals to demonstrate incremental learning
            if tracker.total_goals <= 5:
                stats = tracker.get_data_stats()
                print(f"  → Based on {stats['total_possessions_in_model']} possessions from {stats['total_sequences']} sequences")
            print()
            
            # Break if game is finished
            if tracker.game_state.is_finished:
                print("="*50)
                print(f"🏆 GAME FINISHED! Winner: {tracker.game_state.winner.value.upper()}")
                print("="*50)
                print(tracker.format_live_update(detailed=True))
                break


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
    
    # Demo live tracker
    print("\n" + "="*50)
    demo_live_tracker()


if __name__ == "__main__":
    main()
