#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HTML UI服务器，负责提供HTML文件和处理WebSocket通信
"""

import asyncio
import os
import threading
import webbrowser
import json
import websockets
from http.server import HTTPServer, SimpleHTTPRequestHandler
from websockets.server import serve

# 导入日志系统
from Engine.Logger import get_logger

class HTMLUIServer:
    """HTML UI服务器类"""
    
    def __init__(self, engine, host="localhost", http_port=8000, websocket_port=8001):
        """
        初始化HTML UI服务器
        
        Args:
            engine: 引擎实例
            host: 服务器主机地址
            http_port: HTTP服务器端口
            websocket_port: WebSocket服务器端口
        """
        # 初始化日志器
        self.logger = get_logger("HTMLUIServer")
        self.engine = engine
        self.host = host
        self.http_port = http_port
        self.websocket_port = websocket_port
        self.http_server = None
        self.websocket_server = None
        self.websocket_clients = set()
        self.running = False
        self.status_update_task = None
        self.heartbeat_task = None
        
        # 客户端连接状态跟踪
        self.client_states = {}
        self.heartbeat_interval = 15  # 心跳间隔（秒），调整为15秒，更频繁地检测连接状态
        self.heartbeat_timeout = 45  # 心跳超时（秒），调整为45秒，避免误判
        
        # 设置HTML文件目录
        self.html_dir = os.path.join(os.path.dirname(__file__), "html")
        
        # 确保HTML目录存在
        if not os.path.exists(self.html_dir):
            self.logger.debug(f"创建HTML目录: {self.html_dir}")
            os.makedirs(self.html_dir)
    
    def _check_port_availability(self, port):
        """
        检查端口是否可用
        
        Args:
            port: 要检查的端口号
            
        Returns:
            bool: 端口是否可用
        """
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((self.host, port))
            return True
        except OSError:
            return False
    
    def start(self):
        """启动HTML UI服务器"""
        if self.running:
            return
        
        self.logger.info("启动HTML UI服务器")
        self.running = True
        
        # 启动HTTP服务器线程
        http_thread = threading.Thread(target=self._start_http_server, daemon=True)
        http_thread.start()
        self.logger.debug("HTTP服务器线程已启动")
        
        # 启动WebSocket服务器线程
        websocket_thread = threading.Thread(target=self._start_websocket_server, daemon=True)
        websocket_thread.start()
        self.logger.debug("WebSocket服务器线程已启动")
        
        # 在单独的线程中打开浏览器
        browser_thread = threading.Thread(target=self._open_browser_sync, daemon=True)
        browser_thread.start()
        self.logger.debug("浏览器线程已启动")
    
    def _open_browser_sync(self):
        """同步打开浏览器"""
        asyncio.run(self._open_browser())
    
    def stop(self):
        """停止HTML UI服务器"""
        if not self.running:
            return
        
        self.logger.info("停止HTML UI服务器")
        self.running = False
        
        # 停止HTTP服务器
        if self.http_server:
            self.logger.debug("停止HTTP服务器")
            self.http_server.shutdown()
            self.http_server = None
        
        # 停止WebSocket服务器
        if self.websocket_server:
            self.logger.debug("停止WebSocket服务器")
            self.websocket_server.close()
            self.websocket_server = None
    
    def _start_http_server(self):
        """启动HTTP服务器"""
        # 切换到HTML目录
        original_dir = os.getcwd()
        os.chdir(self.html_dir)
        
        # 检查并尝试可用端口
        available_port = self.http_port
        max_attempts = 5
        
        # 尝试当前端口和后续4个端口
        for attempt in range(max_attempts):
            if self._check_port_availability(available_port):
                break
            self.logger.warning(f"HTTP端口 {available_port} 已被占用，尝试下一个端口...")
            available_port += 1
        
        # 如果所有端口都被占用，使用默认端口
        if not self._check_port_availability(available_port):
            self.logger.error(f"所有HTTP端口 ({self.http_port}-{available_port}) 都被占用，将使用默认端口 {self.http_port}")
            available_port = self.http_port
        
        try:
            # 自定义HTTP请求处理器，添加缓存控制头
            class NoCacheHTTPRequestHandler(SimpleHTTPRequestHandler):
                def end_headers(self):
                    # 添加缓存控制头，避免浏览器缓存
                    self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                    self.send_header('Pragma', 'no-cache')
                    self.send_header('Expires', '0')
                    super().end_headers()
            
            # 创建HTTP服务器
            self.http_server = HTTPServer((self.host, available_port), NoCacheHTTPRequestHandler)
            self.logger.info(f"HTTP服务器启动在 http://{self.host}:{available_port}")
            
            # 启动服务器
            self.http_server.serve_forever()
        except Exception as e:
            self.logger.error(f"HTTP服务器启动失败: {e}")
        finally:
            # 恢复原始目录
            os.chdir(original_dir)
    
    def _start_websocket_server(self):
        """启动WebSocket服务器"""
        asyncio.run(self._run_websocket_server())
    
    async def _run_websocket_server(self):
        """运行WebSocket服务器"""
        # 创建定期发送状态更新的任务
        self.status_update_task = asyncio.create_task(self._send_status_updates())
        
        # 创建心跳任务
        self.heartbeat_task = asyncio.create_task(self._send_heartbeats())
        
        self.logger.info(f"已创建异步任务: 状态更新任务, 心跳任务")
        
        # 检查并尝试可用端口
        available_port = self.websocket_port
        max_attempts = 5
        
        # 尝试当前端口和后续4个端口
        for attempt in range(max_attempts):
            if self._check_port_availability(available_port):
                break
            self.logger.warning(f"WebSocket端口 {available_port} 已被占用，尝试下一个端口...")
            available_port += 1
        
        # 如果所有端口都被占用，使用默认端口
        if not self._check_port_availability(available_port):
            self.logger.error(f"所有WebSocket端口 ({self.websocket_port}-{available_port}) 都被占用，将使用默认端口 {self.websocket_port}")
            available_port = self.websocket_port
        
        try:
            async with serve(self._handle_websocket, self.host, available_port):
                self.logger.info(f"WebSocket服务器启动在 ws://{self.host}:{available_port}")
                await asyncio.Future()  # 运行永远
        except Exception as e:
            self.logger.error(f"WebSocket服务器启动失败: {e}", exc_info=True)
        finally:
            # 取消所有任务
            self.logger.info("开始清理异步任务")
            tasks_to_cancel = [self.status_update_task, self.heartbeat_task]
            for task in tasks_to_cancel:
                if task and not task.done():
                    self.logger.debug(f"取消异步任务: {task.get_name()}")
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        self.logger.debug(f"异步任务已被取消: {task.get_name()}")
                    except Exception as e:
                        self.logger.error(f"取消异步任务时发生错误: {e}", exc_info=True)
            
            # 清理任务引用
            self.logger.info("清理任务引用")
            self.status_update_task = None
            self.heartbeat_task = None
    
    async def _handle_websocket(self, websocket):
        """处理WebSocket连接
        
        Args:
            websocket: WebSocket连接对象
        """
        client_id = id(websocket)
        
        # 添加客户端到集合
        self.websocket_clients.add(websocket)
        
        # 初始化客户端状态
        import time
        client_ip = websocket.remote_address[0] if hasattr(websocket, 'remote_address') else 'unknown'
        client_port = websocket.remote_address[1] if hasattr(websocket, 'remote_address') else 0
        
        self.client_states[client_id] = {
            'websocket': websocket,
            'last_heartbeat': time.time(),
            'connected_at': time.time(),
            'status': 'connected',
            'ip': client_ip,
            'port': client_port,
            'message_count': 0
        }
        
        self.logger.info(f"新的WebSocket连接已建立 (客户端ID: {client_id}, IP: {client_ip}:{client_port})，当前连接数: {len(self.websocket_clients)}")
        
        try:
            # 发送欢迎消息
            self.logger.debug(f"发送欢迎消息给客户端 (客户端ID: {client_id})")
            await websocket.send(json.dumps({
                "type": "welcome",
                "data": "欢迎连接到HTML UI服务器"
            }))
            
            # 发送初始状态
            self.logger.debug(f"发送初始状态给客户端 (客户端ID: {client_id})")
            await self._send_initial_state(websocket)
            
            async for message in websocket:
                try:
                    # 更新消息计数
                    self.client_states[client_id]['message_count'] += 1
                    
                    # 解析消息
                    data = json.loads(message)
                    message_type = data.get('type')
                    
                    # 处理心跳回复
                    if message_type == 'heartbeat':
                        # 更新客户端最后心跳时间
                        self.client_states[client_id]['last_heartbeat'] = time.time()
                        self.logger.debug(f"收到客户端心跳回复 (客户端ID: {client_id})，当前时间: {time.time()}")
                    else:
                        # 处理其他消息
                        self.logger.debug(f"收到客户端消息 (客户端ID: {client_id})，类型: {message_type}，消息计数: {self.client_states[client_id]['message_count']}")
                        await self._handle_message(message, websocket)
                except json.JSONDecodeError as e:
                    self.logger.error(f"解析WebSocket消息失败 (客户端ID: {client_id}): {e}，消息内容: {message}")
                except Exception as e:
                    self.logger.error(f"处理WebSocket消息失败 (客户端ID: {client_id}): {e}", exc_info=True)
        except websockets.exceptions.ConnectionClosed as e:
            self.logger.info(f"WebSocket连接正常关闭 (客户端ID: {client_id})，关闭原因: {e.code} - {e.reason}")
        except asyncio.TimeoutError:
            self.logger.warning(f"WebSocket连接超时 (客户端ID: {client_id})")
        except Exception as e:
            self.logger.error(f"WebSocket处理错误 (客户端ID: {client_id}): {e}", exc_info=True)
        finally:
            # 清理客户端状态
            if client_id in self.client_states:
                client_state = self.client_states[client_id]
                connected_duration = time.time() - client_state['connected_at']
                message_count = client_state['message_count']
                self.logger.debug(f"清理客户端状态 (客户端ID: {client_id})，连接时长: {connected_duration:.2f}秒，消息数量: {message_count}")
                del self.client_states[client_id]
            
            # 从集合中移除客户端
            if websocket in self.websocket_clients:
                self.logger.debug(f"从客户端集合中移除客户端 (客户端ID: {client_id})")
                self.websocket_clients.remove(websocket)
            
            self.logger.info(f"WebSocket连接已关闭 (客户端ID: {client_id})，当前连接数: {len(self.websocket_clients)}")
    
    async def _send_heartbeats(self):
        """
        定期向客户端发送心跳消息，并检测心跳超时
        """
        import time
        last_heartbeat_time = time.time()
        
        self.logger.info(f"心跳任务已启动，心跳间隔: {self.heartbeat_interval}秒，超时时间: {self.heartbeat_timeout}秒")
        
        while self.running:
            try:
                current_time = time.time()
                
                # 每heartbeat_interval秒发送一次心跳
                if current_time - last_heartbeat_time >= self.heartbeat_interval:
                    last_heartbeat_time = current_time
                    
                    # 发送心跳消息给所有客户端
                    heartbeat_message = {
                        "type": "heartbeat",
                        "timestamp": current_time
                    }
                    
                    self.logger.debug(f"开始发送心跳消息，当前时间: {current_time}，客户端数量: {len(self.client_states)}")
                    
                    # 遍历所有客户端，发送心跳并检查超时
                    clients_to_remove = []
                    for client_id, client_state in list(self.client_states.items()):
                        websocket = client_state['websocket']
                        last_heartbeat = client_state['last_heartbeat']
                        
                        # 检查心跳超时
                        if current_time - last_heartbeat > self.heartbeat_timeout:
                            # 客户端超时，准备关闭连接
                            clients_to_remove.append(client_id)
                            self.logger.warning(f"客户端心跳超时，准备关闭连接 (客户端ID: {client_id})，最后心跳时间: {last_heartbeat}，当前时间: {current_time}，超时时间: {self.heartbeat_timeout}秒")
                        else:
                            # 发送心跳消息
                            try:
                                await websocket.send(json.dumps(heartbeat_message))
                                self.logger.debug(f"发送心跳消息给客户端 (客户端ID: {client_id})，当前时间: {current_time}")
                            except Exception as e:
                                self.logger.error(f"发送心跳消息失败 (客户端ID: {client_id}): {e}")
                                clients_to_remove.append(client_id)
                    
                    # 清理超时或出错的客户端
                    if clients_to_remove:
                        self.logger.info(f"准备清理 {len(clients_to_remove)} 个客户端连接")
                    
                    for client_id in clients_to_remove:
                        try:
                            client_state = self.client_states[client_id]
                            websocket = client_state['websocket']
                            
                            self.logger.debug(f"关闭客户端连接 (客户端ID: {client_id})")
                            # 关闭WebSocket连接
                            await websocket.close()
                            
                            # 从客户端集合中移除
                            if websocket in self.websocket_clients:
                                self.websocket_clients.remove(websocket)
                                self.logger.debug(f"从客户端集合中移除客户端 (客户端ID: {client_id})")
                            
                            # 从客户端状态字典中移除
                            del self.client_states[client_id]
                            self.logger.debug(f"从客户端状态字典中移除客户端 (客户端ID: {client_id})")
                            
                            self.logger.info(f"已关闭超时客户端连接 (客户端ID: {client_id})，当前连接数: {len(self.websocket_clients)}")
                        except Exception as e:
                            self.logger.error(f"关闭客户端连接失败 (客户端ID: {client_id}): {e}")
                
                # 等待1秒
                await asyncio.sleep(1.0)
            except Exception as e:
                self.logger.error(f"心跳任务错误: {e}", exc_info=True)
                await asyncio.sleep(5.0)  # 发生错误时，等待5秒后重试
    
    async def _send_status_updates(self):
        """
        定期向客户端发送状态更新
        
        每秒钟发送一次引擎状态、FPS、渲染性能等信息
        """
        import time
        last_time = time.time()
        frame_count = 0
        frame_send_time = 0
        
        while self.running:
            try:
                # 计算FPS
                current_time = time.time()
                delta_time = current_time - last_time
                frame_count += 1
                
                # 每秒发送一次状态更新
                if delta_time >= 1.0:  # 每秒更新一次
                    fps = frame_count / delta_time
                    frame_count = 0
                    last_time = current_time
                    
                    # 获取渲染性能统计
                    render_stats = {}
                    engine_status = "stopped"
                    scene_objects = 0
                    
                    if self.engine is not None:
                        if hasattr(self.engine, 'renderer') and self.engine.renderer:
                            render_stats = self.engine.renderer.get_performance_stats()
                        
                        # 构建状态更新消息
                        engine_status = "running" if self.engine.is_initialized else "stopped"
                        scene_objects = len(self.engine.scene_manager.root_node.children) if hasattr(self.engine, 'scene_manager') and self.engine.scene_manager else 0
                    
                    status_update = {
                        "type": "update_status",
                        "data": {
                            "fps": round(fps, 1),
                            "render_time_ms": render_stats.get("render_time_ms", 0),
                            "draw_calls": render_stats.get("draw_calls", 0),
                            "triangles": render_stats.get("triangles", 0),
                            "visible_objects": render_stats.get("visible_objects", 0),
                            "culled_objects": render_stats.get("culled_objects", 0),
                            "engine_status": engine_status,
                            "scene_objects": scene_objects,
                            "timestamp": time.time()
                        }
                    }
                    
                    # 发送状态更新给所有客户端
                    for client in list(self.websocket_clients):
                        try:
                            await client.send(json.dumps(status_update))
                        except Exception as e:
                            self.logger.error(f"发送状态更新失败: {e}")
                
                # 定期发送渲染帧（每33毫秒，约30FPS）
                if current_time - frame_send_time >= 0.033:  # 约30FPS
                    frame_send_time = current_time
                    await self._send_render_frame()
                
                # 等待100毫秒
                await asyncio.sleep(0.1)
            except Exception as e:
                self.logger.error(f"状态更新任务错误: {e}")
                await asyncio.sleep(1.0)  # 发生错误时，等待1秒后重试
    
    async def _send_render_frame(self):
        """
        发送渲染帧给所有客户端
        """
        if not self.websocket_clients:
            return
        
        try:
            # 获取渲染结果
            if self.engine is not None and hasattr(self.engine, 'renderer') and self.engine.renderer:
                # 动态调整渲染质量以保持帧率
                self._adjust_render_quality_dynamically()
                
                render_result = self.engine.renderer.get_render_result()
                if render_result:
                    # 构建渲染帧消息
                    frame_message = {
                        "type": "render_frame",
                        "data": render_result
                    }
                    
                    # 发送渲染帧给所有客户端
                    for client in list(self.websocket_clients):
                        try:
                            await client.send(json.dumps(frame_message))
                        except Exception as e:
                            self.logger.error(f"发送渲染帧失败: {e}")
        except Exception as e:
            self.logger.error(f"获取渲染帧失败: {e}", exc_info=True)
    
    def _adjust_render_quality_dynamically(self):
        """
        根据实际帧率动态调整渲染质量
        """
        if self.engine is None or not hasattr(self.engine, 'renderer') or not self.engine.renderer:
            return
        
        try:
            # 获取当前渲染性能统计
            stats = self.engine.renderer.get_performance_stats()
            fps = stats.get('fps', 60)
            render_time_ms = stats.get('render_time_ms', 0)
            
            # 根据渲染时间动态调整质量
            if render_time_ms > 20:  # 如果渲染时间超过20ms（50FPS）
                self._decrease_render_quality()
            elif render_time_ms < 10:  # 如果渲染时间低于10ms（100FPS）
                self._increase_render_quality()
        except Exception as e:
            self.logger.error(f"动态调整渲染质量失败: {e}")
    
    def _increase_render_quality(self):
        """
        提高渲染质量
        """
        from Engine.Renderer.Renderer import RenderQuality
        
        quality_levels = [RenderQuality.ULTRA_LOW, RenderQuality.LOW, RenderQuality.MEDIUM, RenderQuality.HIGH]
        current_index = quality_levels.index(self.engine.renderer.render_quality)
        
        if current_index < len(quality_levels) - 1:
            new_quality = quality_levels[current_index + 1]
            self.logger.info(f"提高渲染质量: {self.engine.renderer.render_quality.value} -> {new_quality.value}")
            self.engine.renderer.set_render_quality(new_quality)
    
    def _decrease_render_quality(self):
        """
        降低渲染质量
        """
        from Engine.Renderer.Renderer import RenderQuality
        
        quality_levels = [RenderQuality.ULTRA_LOW, RenderQuality.LOW, RenderQuality.MEDIUM, RenderQuality.HIGH]
        current_index = quality_levels.index(self.engine.renderer.render_quality)
        
        if current_index > 0:
            new_quality = quality_levels[current_index - 1]
            self.logger.info(f"降低渲染质量: {self.engine.renderer.render_quality.value} -> {new_quality.value}")
            self.engine.renderer.set_render_quality(new_quality)
    
    async def _send_initial_state(self, websocket):
        """向客户端发送初始状态信息
        
        Args:
            websocket: WebSocket连接对象
        """
        try:
            # 构建初始状态
            engine_status = "stopped"
            objects_count = 0
            
            if self.engine is not None:
                engine_status = "running" if self.engine.is_initialized else "stopped"
                objects_count = len(self.engine.scene_manager.root_node.children) if hasattr(self.engine, 'scene_manager') and self.engine.scene_manager else 0
            
            initial_state = {
                "type": "initial_state",
                "data": {
                    "engine_status": engine_status,
                    "scene_info": {
                        "objects_count": objects_count
                    },
                    "renderer_info": {
                        "name": "Forward Renderer",
                        "version": "1.0.0"
                    }
                }
            }
            
            await websocket.send(json.dumps(initial_state))
            self.logger.debug("已发送初始状态信息")
        except Exception as e:
            self.logger.error(f"发送初始状态失败: {e}")
    
    async def _handle_message(self, message, websocket):
        """处理来自客户端的消息
        
        Args:
            message: 消息内容
            websocket: WebSocket连接对象
        """
        self.logger.debug(f"收到消息: {message}")
        
        try:
            # 解析JSON消息
            import json
            message_data = json.loads(message)
            message_type = message_data.get("type", "")
            message_data = message_data.get("data", "")
            
            self.logger.debug(f"处理消息类型: {message_type}, 数据: {message_data}")
            
            # 根据消息类型处理
            if message_type == "create_cube":
                self.logger.info("创建立方体")
                if hasattr(self.engine, '_create_cube'):
                    self.engine._create_cube()
                    await websocket.send(json.dumps({"type": "response", "data": "立方体已创建"}))
                else:
                    self.logger.error("引擎没有 _create_cube 方法")
                    await websocket.send(json.dumps({"type": "error", "data": "引擎没有 _create_cube 方法"}))
            elif message_type == "create_sphere":
                self.logger.info("创建球体")
                if hasattr(self.engine, '_create_sphere'):
                    self.engine._create_sphere()
                    await websocket.send(json.dumps({"type": "response", "data": "球体已创建"}))
                else:
                    self.logger.error("引擎没有 _create_sphere 方法")
                    await websocket.send(json.dumps({"type": "error", "data": "引擎没有 _create_sphere 方法"}))
            elif message_type == "create_cylinder":
                self.logger.info("创建圆柱体")
                if hasattr(self.engine, '_create_cylinder'):
                    self.engine._create_cylinder()
                    await websocket.send(json.dumps({"type": "response", "data": "圆柱体已创建"}))
                else:
                    self.logger.error("引擎没有 _create_cylinder 方法")
                    await websocket.send(json.dumps({"type": "error", "data": "引擎没有 _create_cylinder 方法"}))
            elif message_type == "create_plane":
                self.logger.info("创建平面")
                if hasattr(self.engine, '_create_plane'):
                    self.engine._create_plane()
                    await websocket.send(json.dumps({"type": "response", "data": "平面已创建"}))
                else:
                    self.logger.error("引擎没有 _create_plane 方法")
                    await websocket.send(json.dumps({"type": "error", "data": "引擎没有 _create_plane 方法"}))
            elif message_type == "create_cone":
                self.logger.info("创建圆锥体")
                if hasattr(self.engine, '_create_cone'):
                    self.engine._create_cone()
                    await websocket.send(json.dumps({"type": "response", "data": "圆锥体已创建"}))
                else:
                    self.logger.error("引擎没有 _create_cone 方法")
                    await websocket.send(json.dumps({"type": "error", "data": "引擎没有 _create_cone 方法"}))
            elif message_type == "shutdown":
                self.logger.info("关闭引擎")
                if hasattr(self.engine, 'shutdown'):
                    self.engine.shutdown()
                    await websocket.send(json.dumps({"type": "response", "data": "引擎已关闭"}))
                else:
                    self.logger.error("引擎没有 shutdown 方法")
                    await websocket.send(json.dumps({"type": "error", "data": "引擎没有 shutdown 方法"}))
            elif message_type == "menu_file":
                self.logger.debug("文件菜单被点击")
                await websocket.send(json.dumps({"type": "response", "data": "文件菜单被点击"}))
            elif message_type == "menu_edit":
                self.logger.debug("编辑菜单被点击")
                await websocket.send(json.dumps({"type": "response", "data": "编辑菜单被点击"}))
            elif message_type == "menu_view":
                self.logger.debug("视图菜单被点击")
                await websocket.send(json.dumps({"type": "response", "data": "视图菜单被点击"}))
            elif message_type == "menu_object":
                self.logger.debug("对象菜单被点击")
                await websocket.send(json.dumps({"type": "response", "data": "对象菜单被点击"}))
            elif message_type == "menu_render":
                self.logger.debug("渲染菜单被点击")
                await websocket.send(json.dumps({"type": "response", "data": "渲染菜单被点击"}))
            elif message_type == "tool_select":
                self.logger.debug("选择工具被点击")
                if hasattr(self.engine, 'transform_manipulator'):
                    self.engine.transform_manipulator.set_transform_mode("select")
                await websocket.send(json.dumps({"type": "response", "data": "选择工具被激活"}))
            elif message_type == "tool_move":
                self.logger.debug("移动工具被点击")
                if hasattr(self.engine, 'transform_manipulator'):
                    self.engine.transform_manipulator.set_transform_mode("translate")
                await websocket.send(json.dumps({"type": "response", "data": "移动工具被激活"}))
            elif message_type == "tool_rotate":
                self.logger.debug("旋转工具被点击")
                if hasattr(self.engine, 'transform_manipulator'):
                    self.engine.transform_manipulator.set_transform_mode("rotate")
                await websocket.send(json.dumps({"type": "response", "data": "旋转工具被激活"}))
            elif message_type == "tool_scale":
                self.logger.debug("缩放工具被点击")
                if hasattr(self.engine, 'transform_manipulator'):
                    self.engine.transform_manipulator.set_transform_mode("scale")
                await websocket.send(json.dumps({"type": "response", "data": "缩放工具被激活"}))
            elif message_type == "property_position":
                self.logger.debug(f"位置属性被修改: {message_data}")
                # 解析位置数据
                try:
                    pos = list(map(float, message_data.split(',')))
                    if len(pos) == 3:
                        # 应用位置修改到选中对象
                        if hasattr(self.engine, 'scene_manager') and self.engine.scene_manager:
                            # 这里可以根据实际情况修改选中对象的位置
                            self.logger.debug(f"应用位置修改: {pos}")
                except Exception as e:
                    self.logger.error(f"解析位置数据失败: {e}")
                await websocket.send(json.dumps({"type": "response", "data": f"位置属性被修改: {message_data}"}))
            elif message_type == "property_rotation":
                self.logger.debug(f"旋转属性被修改: {message_data}")
                # 解析旋转数据
                try:
                    rot = list(map(float, message_data.split(',')))
                    if len(rot) == 3:
                        # 应用旋转修改到选中对象
                        if hasattr(self.engine, 'scene_manager') and self.engine.scene_manager:
                            # 这里可以根据实际情况修改选中对象的旋转
                            self.logger.debug(f"应用旋转修改: {rot}")
                except Exception as e:
                    self.logger.error(f"解析旋转数据失败: {e}")
                await websocket.send(json.dumps({"type": "response", "data": f"旋转属性被修改: {message_data}"}))
            elif message_type == "property_scale":
                self.logger.debug(f"缩放属性被修改: {message_data}")
                # 解析缩放数据
                try:
                    scale = list(map(float, message_data.split(',')))
                    if len(scale) == 3:
                        # 应用缩放修改到选中对象
                        if hasattr(self.engine, 'scene_manager') and self.engine.scene_manager:
                            # 这里可以根据实际情况修改选中对象的缩放
                            self.logger.debug(f"应用缩放修改: {scale}")
                except Exception as e:
                    self.logger.error(f"解析缩放数据失败: {e}")
                await websocket.send(json.dumps({"type": "response", "data": f"缩放属性被修改: {message_data}"}))
            elif message_type == "property_color":
                self.logger.debug(f"颜色属性被修改: {message_data}")
                # 解析颜色数据
                try:
                    color = list(map(float, message_data.split(',')))
                    if len(color) == 3:
                        # 应用颜色修改到选中对象
                        if hasattr(self.engine, 'scene_manager') and self.engine.scene_manager:
                            # 这里可以根据实际情况修改选中对象的颜色
                            self.logger.debug(f"应用颜色修改: {color}")
                except Exception as e:
                    self.logger.error(f"解析颜色数据失败: {e}")
                await websocket.send(json.dumps({"type": "response", "data": f"颜色属性被修改: {message_data}"}))
            elif message_type == "select_node":
                self.logger.debug(f"选择场景节点: {message_data}")
                # 处理节点选择
                try:
                    node_id = message_data
                    # 这里可以根据实际情况处理节点选择
                    if hasattr(self.engine, 'scene_manager') and self.engine.scene_manager:
                        # 例如，选择场景中的对应节点
                        self.logger.debug(f"选择场景节点: {node_id}")
                except Exception as e:
                    self.logger.error(f"处理节点选择失败: {e}")
                await websocket.send(json.dumps({"type": "response", "data": f"选择节点: {message_data}"}))
            # 相机控制消息处理
            elif message_type == "camera_orbit":
                self.logger.debug(f"相机轨道控制: {message_data}")
                if hasattr(self.engine, 'camera_controller'):
                    try:
                        deltaX = message_data.get('deltaX', 0)
                        deltaY = message_data.get('deltaY', 0)
                        self.engine.camera_controller.orbit(deltaX, deltaY)
                    except Exception as e:
                        self.logger.error(f"相机轨道控制失败: {e}")
            elif message_type == "camera_pan":
                self.logger.debug(f"相机平移控制: {message_data}")
                if hasattr(self.engine, 'camera_controller'):
                    try:
                        deltaX = message_data.get('deltaX', 0)
                        deltaY = message_data.get('deltaY', 0)
                        self.engine.camera_controller.pan(deltaX, deltaY)
                    except Exception as e:
                        self.logger.error(f"相机平移控制失败: {e}")
            elif message_type == "camera_zoom":
                self.logger.debug(f"相机缩放控制: {message_data}")
                if hasattr(self.engine, 'camera_controller'):
                    try:
                        delta = message_data.get('delta', 0)
                        self.engine.camera_controller.zoom(delta)
                    except Exception as e:
                        self.logger.error(f"相机缩放控制失败: {e}")
            elif message_type == "camera_move":
                self.logger.debug(f"相机移动控制: {message_data}")
                if hasattr(self.engine, 'camera_controller'):
                    try:
                        direction = message_data.get('direction', 'forward')
                        self.engine.camera_controller.move(direction)
                    except Exception as e:
                        self.logger.error(f"相机移动控制失败: {e}")
            # 视图模式切换
            elif message_type == "view_perspective":
                self.logger.debug("切换到透视视图")
                if hasattr(self.engine, 'camera'):
                    try:
                        self.engine.camera.set_perspective()
                    except Exception as e:
                        self.logger.error(f"切换到透视视图失败: {e}")
            elif message_type == "view_orthographic":
                self.logger.debug("切换到正交视图")
                if hasattr(self.engine, 'camera'):
                    try:
                        self.engine.camera.set_orthographic()
                    except Exception as e:
                        self.logger.error(f"切换到正交视图失败: {e}")
            elif message_type == "view_top":
                self.logger.debug("切换到顶视图")
                if hasattr(self.engine, 'camera_controller'):
                    try:
                        self.engine.camera_controller.set_top_view()
                    except Exception as e:
                        self.logger.error(f"切换到顶视图失败: {e}")
            elif message_type == "view_front":
                self.logger.debug("切换到前视图")
                if hasattr(self.engine, 'camera_controller'):
                    try:
                        self.engine.camera_controller.set_front_view()
                    except Exception as e:
                        self.logger.error(f"切换到前视图失败: {e}")
            elif message_type == "view_right":
                self.logger.debug("切换到右视图")
                if hasattr(self.engine, 'camera_controller'):
                    try:
                        self.engine.camera_controller.set_right_view()
                    except Exception as e:
                        self.logger.error(f"切换到右视图失败: {e}")
            # 单个属性修改处理
            elif message_type.startswith("property_"):
                self.logger.debug(f"属性修改: {message_type}, {message_data}")
                # 处理单个属性修改
                try:
                    property_name = message_type.split("_")[1]
                    if property_name in ["position_x", "position_y", "position_z", "rotation_x", "rotation_y", "rotation_z", "scale_x", "scale_y", "scale_z"]:
                        # 处理变换属性修改
                        axis = property_name[-1]
                        prop_type = property_name[:-2]
                        value = float(message_data)
                        self.logger.debug(f"应用{prop_type}属性修改，轴: {axis}，值: {value}")
                    elif property_name in ["color", "metallic", "roughness", "alpha"]:
                        # 处理材质属性修改
                        self.logger.debug(f"应用材质属性修改: {property_name} = {message_data}")
                    elif property_name in ["visible", "cast_shadows", "receive_shadows"]:
                        # 处理渲染属性修改
                        self.logger.debug(f"应用渲染属性修改: {property_name} = {message_data}")
                except Exception as e:
                    self.logger.error(f"处理属性修改失败: {e}")
            else:
                self.logger.warning(f"未知消息类型: {message_type}")
                await websocket.send(json.dumps({"type": "error", "data": f"未知消息类型: {message_type}"}))
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON解析失败: {e}")
            await websocket.send(json.dumps({"type": "error", "data": f"JSON解析失败: {str(e)}"}))
        except Exception as e:
            self.logger.error(f"处理消息失败: {e}", exc_info=True)
            # 发送错误响应，但不关闭连接
            try:
                await websocket.send(json.dumps({"type": "error", "data": f"处理消息失败: {str(e)}"}))
            except Exception as send_error:
                self.logger.error(f"发送错误响应失败: {send_error}")
                # 发送错误响应失败，可能连接已经关闭，不做进一步处理
    
    async def send_message(self, message):
        """发送消息给所有客户端
        
        Args:
            message: 消息内容
        """
        if not self.running:
            return
        
        # 发送消息给所有连接的客户端
        for client in self.websocket_clients:
            try:
                await client.send(message)
                self.logger.debug(f"消息已发送给客户端")
            except Exception as e:
                self.logger.error(f"发送消息失败: {e}")
    
    async def _open_browser(self):
        """打开浏览器"""
        # 等待服务器启动
        await asyncio.sleep(1)
        
        # 打开浏览器 - 注意：由于端口可能动态变化，这里使用初始配置的端口
        # 在实际部署中，应该使用服务器实际绑定的端口
        url = f"http://{self.host}:{self.http_port}"
        self.logger.info(f"打开浏览器: {url}")
        webbrowser.open(url)