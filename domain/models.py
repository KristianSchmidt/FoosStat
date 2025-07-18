from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
import uuid
from markov_analytics import LiveGameTracker


class Action(str, Enum):
    """All possible foosball actions"""
    # Red possessions
    R2 = "r2"
    R3 = "r3" 
    R5 = "r5"
    
    # Blue possessions
    B2 = "b2"
    B3 = "b3"
    B5 = "b5"
    
    # Goals
    RED_GOAL = "g_r"
    BLUE_GOAL = "g_b"


class GameMode(str, Enum):
    """Game modes"""
    SINGLES = "singles"
    DOUBLES = "doubles"


class GameState(BaseModel):
    """Main game state model"""
    game_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    blue_score: int = 0
    red_score: int = 0
    blue_sets: int = 0
    red_sets: int = 0
    possession_history: List[str] = Field(default_factory=list)  # For UI display (recent possessions)
    complete_history: List[str] = Field(default_factory=list)    # Complete game history for analytics
    game_mode: GameMode = GameMode.SINGLES
    red_player1: str = "Red"
    red_player2: Optional[str] = None
    blue_player1: str = "Blue"
    blue_player2: Optional[str] = None
    markov_tracker: LiveGameTracker = Field(default_factory=LiveGameTracker, exclude=True)
    
    class Config:
        # Allow arbitrary types for the markov_tracker
        arbitrary_types_allowed = True
        
    def reset(self) -> None:
        """Reset game state for new game"""
        self.game_id = str(uuid.uuid4())
        self.blue_score = 0
        self.red_score = 0
        self.blue_sets = 0
        self.red_sets = 0
        self.possession_history = []
        self.complete_history = []
        self.markov_tracker = LiveGameTracker()
        
    def get_red_display_name(self) -> str:
        """Get last name for red team display"""
        if self.game_mode == GameMode.SINGLES:
            return self.red_player1.split()[-1] if self.red_player1 else "Red"
        else:
            names = []
            if self.red_player1:
                names.append(self.red_player1.split()[-1])
            if self.red_player2:
                names.append(self.red_player2.split()[-1])
            return "/".join(names) if names else "Red"
    
    def get_blue_display_name(self) -> str:
        """Get last name for blue team display"""
        if self.game_mode == GameMode.SINGLES:
            return self.blue_player1.split()[-1] if self.blue_player1 else "Blue"
        else:
            names = []
            if self.blue_player1:
                names.append(self.blue_player1.split()[-1])
            if self.blue_player2:
                names.append(self.blue_player2.split()[-1])
            return "/".join(names) if names else "Blue"
    
    def add_possession(self, action: Action) -> None:
        """Add a possession to both histories"""
        action_str = action.value
        self.possession_history.append(action_str)
        self.complete_history.append(action_str)
    
    def goal_scored(self, team: str) -> bool:
        """
        Handle goal scoring logic and return True if set is won
        
        Args:
            team: "red" or "blue"
        
        Returns:
            bool: True if a set was won, False otherwise
        """
        if team == "red":
            self.red_score += 1
            # Reset recent history and start at Blue 5-bar
            self.possession_history = ['b5']
            # Add kickoff to complete history too
            self.complete_history.append('b5')
            
            if self.red_score >= 5:
                self.red_sets += 1
                self.blue_score = 0
                self.red_score = 0
                return True
                
        elif team == "blue":
            self.blue_score += 1
            # Reset recent history and start at Red 5-bar
            self.possession_history = ['r5']
            # Add kickoff to complete history too
            self.complete_history.append('r5')
            
            if self.blue_score >= 5:
                self.blue_sets += 1
                self.blue_score = 0
                self.red_score = 0
                return True
        
        return False
