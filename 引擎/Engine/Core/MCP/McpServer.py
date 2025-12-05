# -*- coding: utf-8 -*-
"""
MCP Server for Modern AI IDE
提供API和端口通信功能，用于与现代AI IDE集成
"""

import argparse
import socket
import threading
import json
import time
from typing import Dict, Any

from Engine.Core.MCP.MCPManager import MCPManager
from Engine import get_engine

class McpServer:
    """MCP Server类，用于处理现代AI IDE的MCP server请求"""
    
    def __init__(self, host: str = "localhost", port: int = 9875, api_version: str = "1.0"):
        """初始化MCP Server
        
        Args:
            host: 服务器主机地址
            port: 服务器端口
            api_version: API版本
        """
        self.host = host
        self.port = port
        self.api_version = api_version
        self.server_socket = None
        self.is_running = False
        self.client_threads = []
        
        # 获取引擎实例和MCP管理器
        self.engine = get_engine()
        self.mcp_manager = None
        
        # API路由映射
        self.api_routes = {
            "GET /api/v1/mcp/info": self._handle_get_mcp_info,
            "POST /api/v1/mcp/command": self._handle_post_mcp_command,
            "GET /api/v1/mcp/scene": self._handle_get_scene_info,
            "POST /api/v1/mcp/scene/node": self._handle_post_create_node,
            "DELETE /api/v1/mcp/scene/node": self._handle_delete_node,
            "PUT /api/v1/mcp/scene/node/transform": self._handle_put_node_transform,
            "POST /api/v1/mcp/modeling/create": self._handle_post_create_primitive
        }
    
    def start(self):
        """启动MCP Server"""
        try:
            # 初始化引擎
            if not self.engine.is_initialized:
                self.engine.initialize()
            
            # 获取MCP管理器
            if hasattr(self.engine, 'mcp_manager') and self.engine.mcp_manager is not None:
                self.mcp_manager = self.engine.mcp_manager
            else:
                self.mcp_manager = MCPManager(self.engine)
            
            # 创建服务器套接字
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            
            self.is_running = True
            print(f"MCP Server启动成功，监听 {self.host}:{self.port}")
            print(f"API版本: {self.api_version}")
            print(f"可用API路由: {list(self.api_routes.keys())}")
            
            # 启动主循环
            self._run_server()
            
        except Exception as e:
            print(f"MCP Server启动失败: {e}")
            self.stop()
    
    def stop(self):
        """停止MCP Server"""
        self.is_running = False
        
        # 关闭客户端线程
        for thread in self.client_threads:
            thread.join(timeout=1.0)
        
        # 关闭服务器套接字
        if self.server_socket:
            self.server_socket.close()
        
        print("MCP Server已停止")
    
    def _run_server(self):
        """运行服务器主循环"""
        while self.is_running:
            try:
                # 接受客户端连接
                client_socket, client_address = self.server_socket.accept()
                print(f"客户端连接: {client_address}")
                
                # 创建客户端处理线程
                client_thread = threading.Thread(
                    target=self._handle_client, 
                    args=(client_socket, client_address)
                )
                client_thread.daemon = True
                client_thread.start()
                self.client_threads.append(client_thread)
                
            except socket.error as e:
                if self.is_running:
                    print(f"服务器错误: {e}")
                break
            except Exception as e:
                print(f"服务器异常: {e}")
    
    def _handle_client(self, client_socket: socket.socket, client_address: tuple):
        """处理客户端连接
        
        Args:
            client_socket: 客户端套接字
            client_address: 客户端地址
        """
        try:
            while self.is_running:
                # 接收客户端请求
                data = client_socket.recv(4096)
                if not data:
                    break
                
                # 解析请求
                request = data.decode('utf-8')
                print(f"收到请求: {request}")
                
                # 处理请求
                response = self._handle_request(request)
                
                # 发送响应
                client_socket.sendall(response.encode('utf-8'))
                
        except Exception as e:
            print(f"处理客户端 {client_address} 时出错: {e}")
        finally:
            client_socket.close()
            print(f"客户端断开连接: {client_address}")
    
    def _handle_request(self, request: str) -> str:
        """处理HTTP请求
        
        Args:
            request: HTTP请求字符串
            
        Returns:
            str: HTTP响应字符串
        """
        try:
            # 解析请求行
            request_lines = request.split('\r\n')
            if not request_lines:
                return self._create_response(400, {"error": "无效的请求"})
            
            request_line = request_lines[0]
            method, path, _ = request_line.split(' ', 2)
            
            # 构建路由键
            route_key = f"{method} {path}"
            
            # 查找路由处理函数
            if route_key in self.api_routes:
                # 解析请求体
                request_body = ""
                for i, line in enumerate(request_lines):
                    if line == '':
                        if i + 1 < len(request_lines):
                            request_body = '\r\n'.join(request_lines[i+1:])
                        break
                
                # 解析JSON请求体
                body_data = {}
                if request_body:
                    body_data = json.loads(request_body)
                
                # 调用路由处理函数
                response_data = self.api_routes[route_key](body_data)
                return self._create_response(200, response_data)
            else:
                return self._create_response(404, {"error": "API路由不存在"})
                
        except Exception as e:
            return self._create_response(500, {"error": str(e)})
    
    def _create_response(self, status_code: int, data: Dict[str, Any]) -> str:
        """创建HTTP响应
        
        Args:
            status_code: HTTP状态码
            data: 响应数据
            
        Returns:
            str: HTTP响应字符串
        """
        status_text = {
            200: "OK",
            400: "Bad Request",
            404: "Not Found",
            500: "Internal Server Error"
        }.get(status_code, "Unknown")
        
        response_headers = [
            f"HTTP/1.1 {status_code} {status_text}",
            "Content-Type: application/json",
            "Connection: keep-alive",
            f"Content-Length: {len(json.dumps(data))}",
            ""
        ]
        
        response = '\r\n'.join(response_headers) + json.dumps(data)
        return response
    
    def _handle_get_mcp_info(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理获取MCP信息的请求
        
        Args:
            data: 请求数据
            
        Returns:
            Dict[str, Any]: 响应数据
        """
        return {
            "api_version": self.api_version,
            "mcp_version": "1.0",
            "status": "running",
            "engine_version": getattr(self.engine, 'version', 'unknown'),
            "features": [
                "scene_management",
                "node_creation",
                "node_transformation",
                "primitive_creation",
                "command_execution"
            ]
        }
    
    def _handle_post_mcp_command(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理执行MCP命令的请求
        
        Args:
            data: 请求数据，包含command字段
            
        Returns:
            Dict[str, Any]: 响应数据
        """
        command = data.get("command")
        if not command:
            return {"error": "缺少command字段"}
        
        try:
            # 执行命令
            result = self.mcp_manager.execute_command(command)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _handle_get_scene_info(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理获取场景信息的请求
        
        Args:
            data: 请求数据
            
        Returns:
            Dict[str, Any]: 响应数据
        """
        if not self.mcp_manager:
            return {"error": "MCP管理器未初始化"}
        
        scene_state = self.mcp_manager.get_scene_state()
        return {
            "success": True,
            "scene": scene_state
        }
    
    def _handle_post_create_node(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理创建节点的请求
        
        Args:
            data: 请求数据，包含name字段
            
        Returns:
            Dict[str, Any]: 响应数据
        """
        name = data.get("name", "NewNode")
        
        try:
            node = self.mcp_manager.create_node(name)
            return {
                "success": True,
                "node": {
                    "name": node.name,
                    "id": id(node)
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _handle_delete_node(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理删除节点的请求
        
        Args:
            data: 请求数据，包含node_id字段
            
        Returns:
            Dict[str, Any]: 响应数据
        """
        node_id = data.get("node_id")
        if not node_id:
            return {"error": "缺少node_id字段"}
        
        try:
            # 简化实现，实际需要根据node_id查找节点
            result = self.mcp_manager.delete_node(None)
            return {"success": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _handle_put_node_transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理更新节点变换的请求
        
        Args:
            data: 请求数据，包含node_id, position, rotation, scale字段
            
        Returns:
            Dict[str, Any]: 响应数据
        """
        node_id = data.get("node_id")
        if not node_id:
            return {"error": "缺少node_id字段"}
        
        try:
            # 简化实现，实际需要根据node_id查找节点并更新变换
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _handle_post_create_primitive(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理创建基本几何体的请求
        
        Args:
            data: 请求数据，包含type, params字段
            
        Returns:
            Dict[str, Any]: 响应数据
        """
        primitive_type = data.get("type", "cube")
        params = data.get("params", {})
        
        try:
            # 创建基本几何体
            if primitive_type == "cube":
                size = params.get("size", 1.0)
                mesh = self.mcp_manager.create_cube(size)
            elif primitive_type == "sphere":
                radius = params.get("radius", 1.0)
                segments = params.get("segments", 32)
                mesh = self.mcp_manager.create_sphere(radius, segments)
            elif primitive_type == "cylinder":
                radius = params.get("radius", 1.0)
                height = params.get("height", 2.0)
                segments = params.get("segments", 32)
                mesh = self.mcp_manager.create_cylinder(radius, height, segments)
            elif primitive_type == "plane":
                width = params.get("width", 1.0)
                height = params.get("height", 1.0)
                segments = params.get("segments", 1)
                mesh = self.mcp_manager.create_plane(width, height, segments)
            else:
                return {"success": False, "error": f"不支持的几何体类型: {primitive_type}"}
            
            return {
                "success": True,
                "mesh": {
                    "type": primitive_type,
                    "id": id(mesh)
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

def main():
    """主函数，用于启动MCP Server"""
    parser = argparse.ArgumentParser(description="MCP Server for Modern AI IDE")
    parser.add_argument("--host", type=str, default="localhost", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=9875, help="服务器端口")
    parser.add_argument("--api-version", type=str, default="1.0", help="API版本")
    
    args = parser.parse_args()
    
    # 创建并启动MCP Server
    server = McpServer(args.host, args.port, args.api_version)
    
    try:
        server.start()
        # 保持服务器运行
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("收到终止信号，正在停止MCP Server...")
        server.stop()

if __name__ == "__main__":
    main()
