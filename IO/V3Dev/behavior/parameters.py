"""
Behavior Parameters and Presets
===============================
MetaParameters (personality) and preset configurations.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class MetaParameters:
    """
    High-level personality controls (0.0 - 1.0).
    These affect how base parameters are calculated.
    
    Core Personality:
    - responsiveness: How quickly the light reacts
    - energy: Overall liveliness and pulse speed  
    - attention_span: How long it stays focused
    - sociability: Eagerness to engage with people
    - exploration: How much it wanders
    - memory: How much anti-repetition affects behavior
    
    Global Multipliers:
    - brightness_global: Master brightness multiplier
    - speed_global: Master speed multiplier
    - etc.
    
    Feature Toggles:
    - gestures_enabled, follow_enabled, etc.
    """
    # Core personality (0-1)
    responsiveness: float = 0.5   # Low=slow/contemplative, High=quick/reactive
    energy: float = 0.5           # Low=calm/gentle, High=lively/dynamic
    attention_span: float = 0.5   # Low=easily distracted, High=focused/loyal
    sociability: float = 0.5      # Low=reserved, High=eager to engage
    exploration: float = 0.5      # Low=stays put, High=wanders widely
    memory: float = 0.5           # Low=forgets quickly, High=avoids repetition
    
    # Global multipliers
    brightness_global: float = 1.0
    speed_global: float = 1.0
    pulse_global: float = 1.0
    follow_speed_global: float = 1.0  # Multiplier for follow tracking speed
    dwell_influence: float = 1.0      # How much dwell time affects behavior
    trend_weight: float = 1.0
    time_of_day_weight: float = 1.0
    anti_repetition_weight: float = 1.0
    idle_trend_weight: float = 1.0    # How much passive zone trends affect IDLE
    
    # Feature toggles
    gestures_enabled: bool = True
    follow_enabled: bool = True
    flow_mode_enabled: bool = True
    dwell_rewards_enabled: bool = True
    entrance_flash_enabled: bool = True
    self_analysis_enabled: bool = True
    status_text_enabled: bool = True
    
    def lerp(self, low: float, high: float, param: float) -> float:
        """Linear interpolation based on a parameter (0-1)."""
        return low + (high - low) * max(0, min(1, param))
    
    def scale(self, base: float, param: float, 
              low_mult: float = 0.5, high_mult: float = 1.5) -> float:
        """
        Scale a base value based on a personality parameter.
        
        Args:
            base: Base value to scale
            param: Personality parameter (0-1)
            low_mult: Multiplier when param=0
            high_mult: Multiplier when param=1
            
        Returns:
            Scaled value
        """
        mult = self.lerp(low_mult, high_mult, param)
        return base * mult
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'responsiveness': self.responsiveness,
            'energy': self.energy,
            'attention_span': self.attention_span,
            'sociability': self.sociability,
            'exploration': self.exploration,
            'memory': self.memory,
            'brightness_global': self.brightness_global,
            'speed_global': self.speed_global,
            'pulse_global': self.pulse_global,
            'follow_speed_global': self.follow_speed_global,
            'dwell_influence': self.dwell_influence,
            'trend_weight': self.trend_weight,
            'time_of_day_weight': self.time_of_day_weight,
            'anti_repetition_weight': self.anti_repetition_weight,
            'idle_trend_weight': self.idle_trend_weight,
            'gestures_enabled': self.gestures_enabled,
            'follow_enabled': self.follow_enabled,
            'flow_mode_enabled': self.flow_mode_enabled,
            'dwell_rewards_enabled': self.dwell_rewards_enabled,
            'entrance_flash_enabled': self.entrance_flash_enabled,
            'self_analysis_enabled': self.self_analysis_enabled,
            'status_text_enabled': self.status_text_enabled,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MetaParameters':
        """Deserialize from dictionary."""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


# =============================================================================
# PERSONALITY PRESETS
# =============================================================================

PRESETS: Dict[str, MetaParameters] = {
    'default': MetaParameters(),
    
    'shy': MetaParameters(
        responsiveness=0.3, 
        energy=0.3, 
        attention_span=0.7,
        sociability=0.2, 
        exploration=0.3, 
        memory=0.6
    ),
    
    'eager': MetaParameters(
        responsiveness=0.8, 
        energy=0.7, 
        attention_span=0.4,
        sociability=0.9, 
        exploration=0.6, 
        memory=0.4
    ),
    
    'zen': MetaParameters(
        responsiveness=0.2, 
        energy=0.2, 
        attention_span=0.9,
        sociability=0.4, 
        exploration=0.4, 
        memory=0.8
    ),
    
    'playful': MetaParameters(
        responsiveness=0.7, 
        energy=0.8, 
        attention_span=0.3,
        sociability=0.7, 
        exploration=0.9, 
        memory=0.3
    ),
    
    'night_owl': MetaParameters(
        responsiveness=0.4, 
        energy=0.3, 
        attention_span=0.6,
        sociability=0.5, 
        exploration=0.2, 
        memory=0.7
    ),
    
    'contemplative': MetaParameters(
        responsiveness=0.3,
        energy=0.4,
        attention_span=0.8,
        sociability=0.5,
        exploration=0.5,
        memory=0.7
    ),
    
    'energetic': MetaParameters(
        responsiveness=0.9,
        energy=0.9,
        attention_span=0.3,
        sociability=0.8,
        exploration=0.8,
        memory=0.2
    ),
}


def load_preset(name: str) -> MetaParameters:
    """
    Load a preset personality by name.
    
    Args:
        name: Preset name (default, shy, eager, zen, playful, night_owl, etc.)
        
    Returns:
        MetaParameters instance (copy of preset)
    """
    preset = PRESETS.get(name, PRESETS['default'])
    # Return a copy to avoid modifying the preset
    return MetaParameters(**preset.to_dict())


def list_presets() -> list:
    """Get list of available preset names."""
    return list(PRESETS.keys())
