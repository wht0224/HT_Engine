"""
MCP架构核心模块
提供Model-Controller-Processor三层架构的实现
"""

from .MCPManager import MCPManager
from .Model import Model
from .Controller import Controller
from .Processor import Processor

__all__ = [
    "MCPManager",
    "Model",
    "Controller",
    "Processor"
]