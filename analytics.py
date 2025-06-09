#!/usr/bin/env python3
"""
Advanced foosball analytics engine translated from F# FoosStat2
Provides comprehensive statistics for foosball games
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class PlayerColor(Enum):
    RED = "red"
    BLUE = "blue"

class Rod(Enum):
    DEFENSE = "defense"  # 2-bar
    MIDFIELD = "midfield"  # 5-bar
    ATTACK = "attack"  # 3-bar

@dataclass
class Stat:
    """Base class for statistics"""
    def __add__(self, other):
        raise NotImplementedError

@dataclass
class NumberStat(Stat):
    """Simple number statistic"""
    value: int
    
    def __add__(self, other):
        if isinstance(other, NumberStat):
            return NumberStat(self.value + other.value)
        raise TypeError("Cannot add different stat types")
    
    def __str__(self):
        return str(self.value)

@dataclass
class TryFailStat(Stat):
    """Success/attempts statistic with percentage"""
    successes: int
    attempts: int
    
    def __add__(self, other):
        if isinstance(other, TryFailStat):
            return TryFailStat(
                self.successes + other.successes,
                self.attempts + other.attempts
            )
        raise TypeError("Cannot add different stat types")
    
    def __str__(self):
        if self.attempts == 0:
            return "0 / 0 (N/A)"
        percentage = 100.0 * self.successes / self.attempts
        return f"{self.successes} / {self.attempts} ({percentage:.0f}%)"
    
    @property
    def percentage(self) -> float:
        return 100.0 * self.successes / self.attempts if self.attempts > 0 else 0.0

class Event:
    """Base class for game events"""
    pass

@dataclass
class Possession(Event):
    """Ball possession on a specific rod"""
    color: PlayerColor
    rod: Rod

@dataclass
class Goal(Event):
    """Goal scored by a color"""
    color: PlayerColor

class AnalyticsEngine:
    """Main analytics engine for foosball statistics"""
    
    @staticmethod
    def parse_possession_history(history: List[str]) -> List[Event]:
        """Convert possession history strings to Event objects"""
        events = []
        
        for pos in history:
            if pos == 'g_r':
                events.append(Goal(PlayerColor.RED))
            elif pos == 'g_b':
                events.append(Goal(PlayerColor.BLUE))
            elif pos.startswith('r'):
                rod = AnalyticsEngine._parse_rod(pos[1:])
                events.append(Possession(PlayerColor.RED, rod))
            elif pos.startswith('b'):
                rod = AnalyticsEngine._parse_rod(pos[1:])
                events.append(Possession(PlayerColor.BLUE, rod))
                
        return events
    
    @staticmethod
    def _parse_rod(rod_str: str) -> Rod:
        """Parse rod number to Rod enum"""
        if rod_str == '2':
            return Rod.DEFENSE
        elif rod_str == '5':
            return Rod.MIDFIELD
        elif rod_str == '3':
            return Rod.ATTACK
        else:
            raise ValueError(f"Unknown rod: {rod_str}")
    
    @staticmethod
    def calculate_goals(events: List[Event], color: PlayerColor) -> NumberStat:
        """Calculate total goals for a color"""
        goals = sum(1 for event in events if isinstance(event, Goal) and event.color == color)
        return NumberStat(goals)
    
    @staticmethod
    def calculate_three_bar_shots(events: List[Event], color: PlayerColor) -> TryFailStat:
        """Calculate three-bar (attack) shots and goals"""
        successes = 0
        attempts = 0
        
        for i in range(len(events) - 1):
            current = events[i]
            next_event = events[i + 1]
            
            # Check if current is attack possession of our color
            if (isinstance(current, Possession) and 
                current.color == color and 
                current.rod == Rod.ATTACK):
                
                attempts += 1
                # Check if next event is our goal
                if isinstance(next_event, Goal) and next_event.color == color:
                    successes += 1
                    
        return TryFailStat(successes, attempts)
    
    @staticmethod
    def calculate_three_bar_possessions(events: List[Event], color: PlayerColor) -> TryFailStat:
        """Calculate three-bar effectiveness per possession (no recatch)"""
        successes = 0
        attempts = 0
        
        for i in range(len(events) - 1):
            current = events[i]
            next_event = events[i + 1]
            
            # Check if current is attack possession of our color
            if (isinstance(current, Possession) and 
                current.color == color and 
                current.rod == Rod.ATTACK):
                
                # Check if next event is the same rod (recatch) - if so, skip
                if (isinstance(next_event, Possession) and 
                    next_event.color == color and 
                    next_event.rod == Rod.ATTACK):
                    continue
                    
                attempts += 1
                # Check if next event is our goal
                if isinstance(next_event, Goal) and next_event.color == color:
                    successes += 1
                    
        return TryFailStat(successes, attempts)
    
    @staticmethod
    def calculate_five_bar_passes(events: List[Event], color: PlayerColor) -> TryFailStat:
        """Calculate five-bar (midfield) passing accuracy"""
        successes = 0
        attempts = 0
        
        for i in range(len(events) - 1):
            current = events[i]
            next_event = events[i + 1]
            
            # Check if current is midfield possession of our color
            if (isinstance(current, Possession) and 
                current.color == color and 
                current.rod == Rod.MIDFIELD):
                
                attempts += 1
                # Success if next event is our attack possession
                if (isinstance(next_event, Possession) and 
                    next_event.color == color and 
                    next_event.rod == Rod.ATTACK):
                    successes += 1
                    
        return TryFailStat(successes, attempts)
    
    @staticmethod
    def calculate_five_bar_passes_possessions(events: List[Event], color: PlayerColor) -> TryFailStat:
        """Calculate five-bar passing per possession (no recatch)"""
        successes = 0
        attempts = 0
        
        for i in range(len(events) - 1):
            current = events[i]
            next_event = events[i + 1]
            
            # Check if current is midfield possession of our color
            if (isinstance(current, Possession) and 
                current.color == color and 
                current.rod == Rod.MIDFIELD):
                
                # Check if next event is the same rod (recatch) - if so, skip
                if (isinstance(next_event, Possession) and 
                    next_event.color == color and 
                    next_event.rod == Rod.MIDFIELD):
                    continue
                    
                attempts += 1
                # Success if next event is our attack possession
                if (isinstance(next_event, Possession) and 
                    next_event.color == color and 
                    next_event.rod == Rod.ATTACK):
                    successes += 1
                    
        return TryFailStat(successes, attempts)
    
    @staticmethod
    def calculate_two_bar_goals(events: List[Event], color: PlayerColor) -> NumberStat:
        """Calculate goals scored directly from two-bar (defense)"""
        goals = 0
        
        for i in range(len(events) - 1):
            current = events[i]
            next_event = events[i + 1]
            
            # Check if current is defense possession and next is our goal
            if (isinstance(current, Possession) and 
                current.color == color and 
                current.rod == Rod.DEFENSE and
                isinstance(next_event, Goal) and 
                next_event.color == color):
                goals += 1
                
        return NumberStat(goals)
    
    @staticmethod
    def calculate_five_bar_steals(events: List[Event], color: PlayerColor) -> TryFailStat:
        """Calculate five-bar steals from opponent"""
        successes = 0
        attempts = 0
        
        for i in range(len(events) - 1):
            current = events[i]
            next_event = events[i + 1]
            
            # Check if current is opponent midfield possession
            if (isinstance(current, Possession) and 
                current.color != color and 
                current.rod == Rod.MIDFIELD):
                
                attempts += 1
                # Success if next event is our midfield possession
                if (isinstance(next_event, Possession) and 
                    next_event.color == color and 
                    next_event.rod == Rod.MIDFIELD):
                    successes += 1
                    
        return TryFailStat(successes, attempts)
    
    @staticmethod
    def calculate_two_bar_clears(events: List[Event], color: PlayerColor) -> TryFailStat:
        """Calculate two-bar clearing effectiveness"""
        successes = 0
        attempts = 0
        
        for i in range(len(events) - 1):
            current = events[i]
            next_event = events[i + 1]
            
            # Check if current is defense possession of our color
            if (isinstance(current, Possession) and 
                current.color == color and 
                current.rod == Rod.DEFENSE):
                
                # Check if next event is the same rod (recatch) - if so, skip
                if (isinstance(next_event, Possession) and 
                    next_event.color == color and 
                    next_event.rod == Rod.DEFENSE):
                    continue
                    
                attempts += 1
                # Success if next event is our midfield/attack or goal
                if ((isinstance(next_event, Possession) and 
                     next_event.color == color and 
                     next_event.rod in [Rod.MIDFIELD, Rod.ATTACK]) or
                    (isinstance(next_event, Goal) and next_event.color == color)):
                    successes += 1
                    
        return TryFailStat(successes, attempts)
    
    @classmethod
    def generate_full_stats(cls, possession_history: List[str]) -> Dict[str, Dict[str, Stat]]:
        """Generate comprehensive statistics for both teams"""
        events = cls.parse_possession_history(possession_history)
        
        stats = {
            "red": {},
            "blue": {}
        }
        
        for color_str, color_enum in [("red", PlayerColor.RED), ("blue", PlayerColor.BLUE)]:
            stats[color_str] = {
                "Goals": cls.calculate_goals(events, color_enum),
                "Three bar goals/shots": cls.calculate_three_bar_shots(events, color_enum),
                "Three bar goals/possessions": cls.calculate_three_bar_possessions(events, color_enum),
                "Five bar passes/attempts": cls.calculate_five_bar_passes(events, color_enum),
                "Five bar passes/possessions": cls.calculate_five_bar_passes_possessions(events, color_enum),
                "Two bar goals": cls.calculate_two_bar_goals(events, color_enum),
                "Five bar steals": cls.calculate_five_bar_steals(events, color_enum),
                "Two bar clears": cls.calculate_two_bar_clears(events, color_enum),
            }
            
        return stats
