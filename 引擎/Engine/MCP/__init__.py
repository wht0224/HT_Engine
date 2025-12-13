# -*- coding: utf-8 -*-
"""
MCP (Model Context Protocol) 模块
让AI可以通过代码直接建模
"""

from .mcp_server import MCPServer
from .modeling_api import ModelingAPI

__all__ = ['MCPServer', 'ModelingAPI']
