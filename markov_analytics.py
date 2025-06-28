"""
Markov Model Analytics for Foosball Game Data

This module implements a Markov chain model to analyze foosball possession transitions
and calculate scoring probabilities from different game states.
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict, Counter


class FoosballMarkovModel:
    """Markov model for foosball possession and scoring analysis."""
    
    def __init__(self):
        # Define all possible states
        self.states = ['b2', 'b3', 'b5', 'r2', 'r3', 'r5', 'g_b', 'g_r']
        self.transient_states = ['b2', 'b3', 'b5', 'r2', 'r3', 'r5']  # Non-absorbing states
        self.absorbing_states = ['g_b', 'g_r']  # Goal states (sinks)
        
        # Create state-to-index mapping
        self.state_to_idx = {state: i for i, state in enumerate(self.states)}
        self.idx_to_state = {i: state for i, state in enumerate(self.states)}
        
        # Initialize transition matrix (8x8 for all states)
        self.transition_matrix = np.zeros((len(self.states), len(self.states)))
        
        # Track raw transition counts for building the model
        self.transition_counts = defaultdict(Counter)
        
    def parse_game_data(self, filename: str) -> List[List[str]]:
        """Parse game data from file and extract possession sequences."""
        sequences = []
        
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('set ') or not line:
                    continue
                
                # Parse the line - it's just comma-separated states
                states = [s.strip() for s in line.split(',')]
                # Filter out invalid states like 'to_b', 'to_r' but keep valid ones
                valid_states = [s for s in states if s in self.state_to_idx]
                if len(valid_states) > 1:  # Need at least 2 states for a transition
                    sequences.append(valid_states)
        
        return sequences
    
    def build_transition_matrix(self, sequences: List[List[str]]):
        """Build transition matrix from possession sequences."""
        # Count transitions
        for sequence in sequences:
            for i in range(len(sequence) - 1):
                from_state = sequence[i]
                to_state = sequence[i + 1]
                self.transition_counts[from_state][to_state] += 1
        
        # Convert counts to probabilities
        for from_state in self.transient_states:
            from_idx = self.state_to_idx[from_state]
            total_transitions = sum(self.transition_counts[from_state].values())
            
            if total_transitions > 0:
                for to_state, count in self.transition_counts[from_state].items():
                    to_idx = self.state_to_idx[to_state]
                    self.transition_matrix[from_idx][to_idx] = count / total_transitions
        
        # Absorbing states stay in themselves
        for absorbing_state in self.absorbing_states:
            idx = self.state_to_idx[absorbing_state]
            self.transition_matrix[idx][idx] = 1.0
    
    def get_absorption_probabilities(self) -> Dict[str, Dict[str, float]]:
        """Calculate probability of reaching each absorbing state from each transient state."""
        
        # Extract Q (transient to transient) and R (transient to absorbing) matrices
        n_transient = len(self.transient_states)
        
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
        result = {}
        for i, from_state in enumerate(self.transient_states):
            result[from_state] = {}
            for j, absorbing_state in enumerate(self.absorbing_states):
                result[from_state][absorbing_state] = B[i][j]
        
        return result
    
    def get_expected_steps_to_absorption(self) -> Dict[str, float]:
        """Calculate expected number of steps to reach an absorbing state."""
        n_transient = len(self.transient_states)
        Q = self.transition_matrix[:n_transient, :n_transient]
        
        I = np.eye(n_transient)
        try:
            N = np.linalg.inv(I - Q)
        except np.linalg.LinAlgError:
            N = np.linalg.pinv(I - Q)
        
        # Expected steps = sum of each row in fundamental matrix
        expected_steps = N.sum(axis=1)
        
        return {state: steps for state, steps in zip(self.transient_states, expected_steps)}
    
    def analyze_team_strengths(self) -> Dict[str, float]:
        """Analyze overall team strengths based on scoring probabilities."""
        absorption_probs = self.get_absorption_probabilities()
        
        # Weight by typical possession distribution
        possession_weights = {'b2': 0.15, 'b3': 0.15, 'b5': 0.20, 
                             'r2': 0.15, 'r3': 0.15, 'r5': 0.20}
        
        blue_score_prob = sum(absorption_probs[state]['g_b'] * possession_weights[state] 
                             for state in self.transient_states)
        red_score_prob = sum(absorption_probs[state]['g_r'] * possession_weights[state] 
                            for state in self.transient_states)
        
        return {
            'blue_scoring_probability': blue_score_prob,
            'red_scoring_probability': red_score_prob,
            'blue_advantage': blue_score_prob - red_score_prob
        }
    
    def print_analysis(self):
        """Print comprehensive analysis of the Markov model."""
        print("=== Foosball Markov Model Analysis ===\n")
        
        # Transition matrix
        print("Transition Matrix:")
        print("States:", self.states)
        print(np.round(self.transition_matrix, 3))
        print()
        
        # Absorption probabilities
        absorption_probs = self.get_absorption_probabilities()
        print("Scoring Probabilities from Each State:")
        print(f"{'State':<5} {'Blue Goal':<12} {'Red Goal':<12}")
        print("-" * 35)
        for state in self.transient_states:
            blue_prob = absorption_probs[state]['g_b']
            red_prob = absorption_probs[state]['g_r']
            print(f"{state:<5} {blue_prob:<12.3f} {red_prob:<12.3f}")
        print()
        
        # Expected steps to goal
        expected_steps = self.get_expected_steps_to_absorption()
        print("Expected Possessions Until Goal:")
        for state, steps in expected_steps.items():
            print(f"{state}: {steps:.2f}")
        print()
        
        # Team strengths
        strengths = self.analyze_team_strengths()
        print("Overall Team Analysis:")
        print(f"Blue scoring probability: {strengths['blue_scoring_probability']:.3f}")
        print(f"Red scoring probability: {strengths['red_scoring_probability']:.3f}")
        print(f"Blue advantage: {strengths['blue_advantage']:+.3f}")


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
                    states = [s.strip() for s in sequence_part.split(',')]
                    valid_states = [s for s in states if s in model.state_to_idx]
                    print(f"  -> States: {states}")
                    print(f"  -> Valid: {valid_states}")
    
    # Build and analyze model
    model.build_transition_matrix(sequences)
    model.print_analysis()


if __name__ == "__main__":
    main()
