"""
Audio Intelligence Sampler v2 - Plugin System

ALL analysis features are implemented as plugins following the sacred architecture.

Architecture principles:
- Plugins can fail without crashing core engine
- Standard interface for all plugins
- Resource requirement declarations
- Clean separation of concerns
- Easy to add new features without touching core
"""

__version__ = "2.0.0-phase1"