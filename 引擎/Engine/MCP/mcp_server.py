# -*- coding: utf-8 -*-
"""
MCP服务器 - 让AI通过Model Context Protocol操作引擎
MCP Server - AI control via Model Context Protocol
"""

import json
import sys
from typing import Any, Dict, List
from Engine.Logger import get_logger


class MCPServer:
    """
    MCP服务器 - 提供标准的Model Context Protocol接口
    让AI IDE工具可以通过MCP协议操作引擎
    """

    def __init__(self, engine):
        """
        初始化MCP服务器

        Args:
            engine: 引擎实例
        """
        self.engine = engine

        # 创建一个简单的logger（不依赖完整引擎）
        try:
            from Engine.Logger import get_logger
            self.logger = get_logger("MCPServer")
        except:
            import logging
            self.logger = logging.getLogger("MCPServer")
            self.logger.setLevel(logging.WARNING)

        self.modeling_api = None

        # 初始化建模API
        if engine:
            try:
                from .modeling_api import ModelingAPI
                self.modeling_api = ModelingAPI(engine)
            except Exception as e:
                self.logger.warning(f"建模API初始化失败: {e}")

        # 注册的工具（MCP Tools）
        self.tools = self._register_tools()

    def _register_tools(self) -> List[Dict[str, Any]]:
        """
        注册MCP工具

        Returns:
            list: 工具列表
        """
        return [
            {
                "name": "create_cube",
                "description": "创建一个立方体模型",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "size": {
                            "type": "number",
                            "description": "立方体边长",
                            "default": 1.0
                        },
                        "name": {
                            "type": "string",
                            "description": "对象名称",
                            "default": "Cube"
                        },
                        "position": {
                            "type": "array",
                            "description": "位置坐标 [x, y, z]",
                            "items": {"type": "number"},
                            "default": [0, 0, 0]
                        }
                    }
                }
            },
            {
                "name": "create_sphere",
                "description": "创建一个球体模型",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "radius": {
                            "type": "number",
                            "description": "球体半径",
                            "default": 1.0
                        },
                        "segments": {
                            "type": "integer",
                            "description": "经线分段数（越大越光滑）",
                            "default": 32
                        },
                        "rings": {
                            "type": "integer",
                            "description": "纬线分段数（越大越光滑）",
                            "default": 16
                        },
                        "name": {
                            "type": "string",
                            "description": "对象名称",
                            "default": "Sphere"
                        },
                        "position": {
                            "type": "array",
                            "description": "位置坐标 [x, y, z]",
                            "items": {"type": "number"},
                            "default": [0, 0, 0]
                        }
                    }
                }
            },
            {
                "name": "create_cylinder",
                "description": "创建一个圆柱体模型",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "radius": {
                            "type": "number",
                            "description": "圆柱半径",
                            "default": 1.0
                        },
                        "height": {
                            "type": "number",
                            "description": "圆柱高度",
                            "default": 2.0
                        },
                        "segments": {
                            "type": "integer",
                            "description": "圆周分段数（越大越光滑）",
                            "default": 32
                        },
                        "name": {
                            "type": "string",
                            "description": "对象名称",
                            "default": "Cylinder"
                        },
                        "position": {
                            "type": "array",
                            "description": "位置坐标 [x, y, z]",
                            "items": {"type": "number"},
                            "default": [0, 0, 0]
                        }
                    }
                }
            },
            {
                "name": "create_plane",
                "description": "创建一个平面模型",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "width": {
                            "type": "number",
                            "description": "平面宽度",
                            "default": 2.0
                        },
                        "height": {
                            "type": "number",
                            "description": "平面高度",
                            "default": 2.0
                        },
                        "name": {
                            "type": "string",
                            "description": "对象名称",
                            "default": "Plane"
                        },
                        "position": {
                            "type": "array",
                            "description": "位置坐标 [x, y, z]",
                            "items": {"type": "number"},
                            "default": [0, 0, 0]
                        }
                    }
                }
            },
            {
                "name": "create_cone",
                "description": "创建一个圆锥体模型",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "radius": {
                            "type": "number",
                            "description": "底面半径",
                            "default": 1.0
                        },
                        "height": {
                            "type": "number",
                            "description": "圆锥高度",
                            "default": 2.0
                        },
                        "segments": {
                            "type": "integer",
                            "description": "底面分段数（越大越光滑）",
                            "default": 32
                        },
                        "name": {
                            "type": "string",
                            "description": "对象名称",
                            "default": "Cone"
                        },
                        "position": {
                            "type": "array",
                            "description": "位置坐标 [x, y, z]",
                            "items": {"type": "number"},
                            "default": [0, 0, 0]
                        }
                    }
                }
            },
            {
                "name": "create_custom_mesh",
                "description": "从顶点数组创建自定义网格模型",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "vertices": {
                            "type": "array",
                            "description": "顶点坐标列表 [[x, y, z], ...]",
                            "items": {
                                "type": "array",
                                "items": {"type": "number"}
                            }
                        },
                        "indices": {
                            "type": "array",
                            "description": "索引列表（可选）",
                            "items": {"type": "integer"},
                            "default": None
                        },
                        "normals": {
                            "type": "array",
                            "description": "法线列表（可选）",
                            "items": {
                                "type": "array",
                                "items": {"type": "number"}
                            },
                            "default": None
                        },
                        "name": {
                            "type": "string",
                            "description": "对象名称",
                            "default": "CustomMesh"
                        },
                        "position": {
                            "type": "array",
                            "description": "位置坐标 [x, y, z]",
                            "items": {"type": "number"},
                            "default": [0, 0, 0]
                        }
                    },
                    "required": ["vertices"]
                }
            },
            {
                "name": "set_shading_mode",
                "description": "设置视口渲染模式",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "description": "渲染模式",
                            "enum": ["solid", "material", "rendered"],
                            "default": "rendered"
                        }
                    }
                }
            },
            {
                "name": "import_model",
                "description": "从文件导入3D模型",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "模型文件路径（支持OBJ, STL, PLY, GLTF等格式）"
                        },
                        "position": {
                            "type": "array",
                            "description": "导入位置 [x, y, z]",
                            "items": {"type": "number"},
                            "default": [0, 0, 0]
                        }
                    },
                    "required": ["file_path"]
                }
            }
        ]

    def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理工具调用 - 通过HTTP API调用运行中的引擎

        Args:
            tool_name: 工具名称
            arguments: 工具参数

        Returns:
            dict: 执行结果
        """
        try:
            import requests

            # HTTP API地址
            api_url = "http://127.0.0.1:5000/api"

            # 提取位置参数
            position = arguments.get("position", [0, 0, 0])

            # 根据工具名称调用相应的API
            if tool_name == "create_cube":
                response = requests.post(f"{api_url}/create_cube", json={
                    "size": arguments.get("size", 1.0),
                    "name": arguments.get("name", "Cube"),
                    "position": position
                }, timeout=5)
                return response.json()

            elif tool_name == "create_sphere":
                response = requests.post(f"{api_url}/create_sphere", json={
                    "radius": arguments.get("radius", 1.0),
                    "segments": arguments.get("segments", 32),
                    "rings": arguments.get("rings", 16),
                    "name": arguments.get("name", "Sphere"),
                    "position": position
                }, timeout=5)
                return response.json()

            elif tool_name == "create_cylinder":
                response = requests.post(f"{api_url}/create_cylinder", json={
                    "radius": arguments.get("radius", 1.0),
                    "height": arguments.get("height", 2.0),
                    "segments": arguments.get("segments", 32),
                    "name": arguments.get("name", "Cylinder"),
                    "position": position
                }, timeout=5)
                return response.json()

            elif tool_name == "create_plane":
                response = requests.post(f"{api_url}/create_plane", json={
                    "width": arguments.get("width", 2.0),
                    "height": arguments.get("height", 2.0),
                    "name": arguments.get("name", "Plane"),
                    "position": position
                }, timeout=5)
                return response.json()

            elif tool_name == "create_cone":
                response = requests.post(f"{api_url}/create_cone", json={
                    "radius": arguments.get("radius", 1.0),
                    "height": arguments.get("height", 2.0),
                    "segments": arguments.get("segments", 32),
                    "name": arguments.get("name", "Cone"),
                    "position": position
                }, timeout=5)
                return response.json()

            elif tool_name == "set_shading_mode":
                response = requests.post(f"{api_url}/set_shading_mode", json={
                    "mode": arguments.get("mode", "rendered")
                }, timeout=5)
                return response.json()

            elif tool_name == "create_custom_mesh":
                # 自定义网格暂不支持通过HTTP（数据量大）
                return {
                    "success": False,
                    "error": "自定义网格暂不支持HTTP API，请使用Python脚本"
                }

            elif tool_name == "import_model":
                # 模型导入暂不支持
                return {
                    "success": False,
                    "error": "模型导入暂不支持HTTP API，请使用Python脚本"
                }

            else:
                return {
                    "success": False,
                    "error": f"未知的工具: {tool_name}"
                }

        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": "无法连接到引擎HTTP API服务器，请确保引擎正在运行 (python main.py)"
            }
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "引擎响应超时"
            }
        except Exception as e:
            self.logger.error(f"工具调用失败: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

    def get_tools_list(self) -> List[Dict[str, Any]]:
        """
        获取工具列表（MCP格式）

        Returns:
            list: 工具列表
        """
        return self.tools

    def start_stdio_server(self):
        """
        启动标准输入输出MCP服务器
        供AI IDE工具通过stdio协议连接
        """
        self.logger.info("MCP服务器启动中（stdio模式）")
        self.logger.info(f"注册的工具数量: {len(self.tools)}")

        try:
            import json
            import sys

            # MCP协议：通过stdin/stdout进行JSON-RPC通信
            while True:
                # 读取一行JSON-RPC请求
                line = sys.stdin.readline()
                if not line:
                    break

                try:
                    request = json.loads(line.strip())

                    # 处理不同的JSON-RPC方法
                    method = request.get("method")
                    request_id = request.get("id")

                    if method == "tools/list":
                        # 返回工具列表
                        response = {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "result": {
                                "tools": self.tools
                            }
                        }
                        sys.stdout.write(json.dumps(response) + "\n")
                        sys.stdout.flush()

                    elif method == "tools/call":
                        # 调用工具
                        params = request.get("params", {})
                        tool_name = params.get("name")
                        arguments = params.get("arguments", {})

                        result = self.handle_tool_call(tool_name, arguments)

                        response = {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "result": result
                        }
                        sys.stdout.write(json.dumps(response) + "\n")
                        sys.stdout.flush()

                    elif method == "initialize":
                        # 初始化握手
                        response = {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "result": {
                                "protocolVersion": "2024-11-05",
                                "capabilities": {
                                    "tools": {}
                                },
                                "serverInfo": {
                                    "name": "rendering-engine",
                                    "version": "1.0.0"
                                }
                            }
                        }
                        sys.stdout.write(json.dumps(response) + "\n")
                        sys.stdout.flush()

                    elif method == "initialized":
                        # 初始化完成通知（无需响应）
                        pass

                    else:
                        # 未知方法
                        error_response = {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {
                                "code": -32601,
                                "message": f"Method not found: {method}"
                            }
                        }
                        sys.stdout.write(json.dumps(error_response) + "\n")
                        sys.stdout.flush()

                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON解析错误: {e}")
                    continue
                except Exception as e:
                    self.logger.error(f"处理请求失败: {e}", exc_info=True)
                    error_response = {
                        "jsonrpc": "2.0",
                        "id": request_id if 'request_id' in locals() else None,
                        "error": {
                            "code": -32000,
                            "message": str(e)
                        }
                    }
                    sys.stdout.write(json.dumps(error_response) + "\n")
                    sys.stdout.flush()

        except KeyboardInterrupt:
            self.logger.info("MCP服务器停止")
        except Exception as e:
            self.logger.error(f"MCP服务器异常: {e}", exc_info=True)
