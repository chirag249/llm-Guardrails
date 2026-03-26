"""SmartGuard package initialization"""

from .classifier import PromptClassifier, KeywordBaseline, ThreatCategory
from .evaluator import Evaluator
from .red_team_suite import RED_TEAM_SUITE

__version__ = "1.0.0"
__all__ = [
    "PromptClassifier",
    "KeywordBaseline", 
    "ThreatCategory",
    "Evaluator",
    "RED_TEAM_SUITE"
]
