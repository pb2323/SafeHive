"""
SafeHive AI Security Sandbox

A CLI-based demonstration and testing platform that simulates a food-ordering 
workflow where AI assistants interact with potentially malicious vendors, 
payment services, and external APIs.

The system addresses the critical need for organizations to understand and 
protect against novel AI attack vectors in real-world scenarios.
"""

__version__ = "0.1.0"
__author__ = "SafeHive Team"
__email__ = "team@safehive.ai"
__description__ = "AI Security Sandbox for testing and demonstrating AI attack vectors"

# Core modules
from . import agents
from . import guards
from . import config
from . import utils
from . import models
from . import tools

__all__ = [
    "agents",
    "guards", 
    "config",
    "utils",
    "models",
    "tools",
    "__version__",
    "__author__",
    "__email__",
    "__description__"
]
