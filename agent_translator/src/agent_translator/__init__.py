# src/agent_translator/__init__.py
"""
CrowdWisdomTrading Document Translation System
AI-powered academic document translation with formatting preservation
"""

__version__ = "2.0.0"
__author__ = "CrowdWisdomTrading"

from .crew import DocumentTranslationCrew

__all__ = [
    "DocumentTranslationCrew",
]