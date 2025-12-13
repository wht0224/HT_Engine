#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HTML UI服务器，负责提供HTML文件和处理HTTP+SSE通信
"""

import os
import threading
import webbrowser
import json
import time
import queue
from http.server import HTTPServer, BaseHTTPRequestHandler

# 导入日志系统
from Engine.Logger import get_logger

class HTMLUIServer:
    """HTML UI服务器类"""
    
    def __init__(self, engine, host="localhost", http_port=8000):
        """
        初始化HTML UI服务器
        
        Args:
            engine: 引擎实例
            host: 服务器主机地址
            http_port: HTTP服务器端口
        """
        # 初始化日志器
        self.logger = get_logger("HTMLUIServer")
        self.engine = engine
        self.host = host
        self.http_port = http_port
        self.actual_http_port = http_port  # 实际绑定的HTTP端口
        self.http_server = None
        self.running = False
        
        # SSE客户端管理
        self.sse_clients = set()
        self.sse_lock = threading.Lock()
        self.render_queue = queue.Queue(maxsize=10)  # 渲染帧队列
        
        # 设置HTML文件目录
        self.html_dir = os.path.join(os.path.dirname(__file__), "html")
        
        # 确保HTML目录存在
        if not os.path.exists(self.html_dir):
            self.logger.debug(f"创建HTML目录: {self.html_dir}")
            os.makedirs(self.html_dir)
        
        # 状态更新线程
        self.status_thread = None
        self._status_update_running = False
    
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
        
        # 启动状态更新线程
        self._start_status_update_thread()
        
        # 在单独的线程中打开浏览器
        browser_thread = threading.Thread(target=self._open_browser_sync, daemon=True)
        browser_thread.start()
        self.logger.debug("浏览器线程已启动")
    
    def _open_browser_sync(self):
        """同步打开浏览器"""
        self._open_browser()
    
    def stop(self):
        """停止HTML UI服务器"""
        if not self.running:
            return
        
        self.logger.info("停止HTML UI服务器")
        self.running = False
        
        # 停止状态更新线程
        self._status_update_running = False
        if self.status_thread:
            self.status_thread.join(timeout=2.0)
            self.status_thread = None
        
        # 停止HTTP服务器
        if self.http_server:
            self.logger.debug("停止HTTP服务器")
            try:
                # 使用线程执行shutdown，避免阻塞
                def shutdown_http():
                    try:
                        self.http_server.shutdown()
                        self.logger.debug("HTTP服务器已停止")
                    except Exception as e:
                        self.logger.error(f"HTTP服务器停止失败: {e}")
                
                shutdown_thread = threading.Thread(target=shutdown_http)
                shutdown_thread.daemon = True
                shutdown_thread.start()
                
                # 等待最多2秒，然后直接关闭
                shutdown_thread.join(timeout=2.0)
                
                self.http_server = None
            except Exception as e:
                self.logger.error(f"HTTP服务器停止失败: {e}")
                self.http_server = None
        
        # 清理SSE客户端
        with self.sse_lock:
            self.sse_clients.clear()
    
    def _start_status_update_thread(self):
        """启动状态更新线程"""
        self._status_update_running = True
        self.status_thread = threading.Thread(target=self._status_update_loop, daemon=True)
        self.status_thread.start()
        self.logger.debug("状态更新线程已启动")
    
    def _status_update_loop(self):
        """状态更新循环"""
        import time
        last_time = time.time()
        frame_count = 0
        frame_send_time = 0
        
        while self._status_update_running and self.running:
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
                            "timestamp": current_time
                        }
                    }
                    
                    # 使用SSE发送状态更新
                    self._send_sse_message(json.dumps(status_update))
                
                # 定期发送渲染帧（每33毫秒，约30FPS）
                if current_time - frame_send_time >= 0.033:  # 约30FPS
                    frame_send_time = current_time
                    self._send_render_frame()
                
                # 等待100毫秒
                time.sleep(0.1)
            except Exception as e:
                self.logger.error(f"状态更新循环错误: {e}")
                time.sleep(1.0)  # 发生错误时，等待1秒后重试
    
    def _send_render_frame(self):
        """发送渲染帧给所有客户端"""
        if not self.sse_clients:
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
                    
                    # 使用SSE发送渲染帧
                    self._send_sse_message(json.dumps(frame_message))
        except Exception as e:
            self.logger.error(f"获取渲染帧失败: {e}", exc_info=True)
    
    def _send_sse_message(self, message):
        """发送SSE消息给所有客户端
        
        Args:
            message: 要发送的消息内容
        """
        with self.sse_lock:
            clients_to_remove = []
            for client in self.sse_clients:
                try:
                    # SSE消息格式：data: {message}\n\n
                    client.wfile.write(f"data: {message}\n\n".encode('utf-8'))
                    client.wfile.flush()
                except Exception as e:
                    self.logger.error(f"发送SSE消息失败: {e}")
                    clients_to_remove.append(client)
            
            # 移除失效的客户端
            for client in clients_to_remove:
                if client in self.sse_clients:
                    self.sse_clients.remove(client)
                    try:
                        client.wfile.close()
                    except:
                        pass
    
    def _start_http_server(self):
        """启动HTTP服务器"""
        from http.server import SimpleHTTPRequestHandler
        import mimetypes
        
        # 自定义HTTP请求处理器，支持SSE和POST
        class CustomHTTPRequestHandler(SimpleHTTPRequestHandler):
            def end_headers(self):
                # 添加CORS头
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                # 添加缓存控制头，避免浏览器缓存
                self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                self.send_header('Pragma', 'no-cache')
                self.send_header('Expires', '0')
                super().end_headers()
            
            def do_OPTIONS(self):
                """处理OPTIONS请求"""
                self.send_response(200)
                self.end_headers()
            
            def do_GET(self):
                """处理GET请求"""
                if self.path == '/events':
                    # 处理SSE连接
                    self._handle_sse()
                else:
                    # 处理静态文件请求
                    super().do_GET()
            
            def do_POST(self):
                """处理POST请求"""
                # 处理命令请求
                self._handle_post()
            
            def _handle_sse(self):
                """处理SSE连接"""
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Connection', 'keep-alive')
                self.end_headers()
                
                # 将客户端添加到SSE客户端集合
                with self.server.ui_server.sse_lock:
                    self.server.ui_server.sse_clients.add(self)
                
                # 发送初始状态
                try:
                    initial_state = self._get_initial_state()
                    self.wfile.write(f"data: {json.dumps(initial_state)}\n\n".encode('utf-8'))
                    self.wfile.flush()
                except Exception as e:
                    self.server.ui_server.logger.error(f"发送初始状态失败: {e}")
                    return
                
                # 保持连接打开
                try:
                    while True:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                        import time
                        time.sleep(30)  # 每30秒发送一次ping
                except Exception as e:
                    # 连接关闭
                    pass
                finally:
                    # 从客户端集合中移除
                    with self.server.ui_server.sse_lock:
                        if self in self.server.ui_server.sse_clients:
                            self.server.ui_server.sse_clients.remove(self)
            
            def _handle_post(self):
                """处理POST请求"""
                content_length = int(self.headers.get('Content-Length', 0))
                if content_length > 0:
                    post_data = self.rfile.read(content_length)
                    try:
                        # 解析JSON数据
                        message_data = json.loads(post_data.decode('utf-8'))
                        # 处理消息
                        self.server.ui_server._handle_http_message(message_data, self)
                        
                        # 发送成功响应
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        response = {"status": "success", "message": "Command processed"}
                        self.wfile.write(json.dumps(response).encode('utf-8'))
                    except json.JSONDecodeError as e:
                        self.send_response(400)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        response = {"status": "error", "message": f"JSON parse error: {str(e)}"}
                        self.wfile.write(json.dumps(response).encode('utf-8'))
                    except Exception as e:
                        self.send_response(500)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        response = {"status": "error", "message": f"Command processing error: {str(e)}"}
                        self.wfile.write(json.dumps(response).encode('utf-8'))
                else:
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    response = {"status": "error", "message": "Empty request body"}
                    self.wfile.write(json.dumps(response).encode('utf-8'))
            
            def _get_initial_state(self):
                """获取初始状态"""
                engine_status = "stopped"
                objects_count = 0
                
                if self.server.ui_server.engine is not None:
                    engine_status = "running" if self.server.ui_server.engine.is_initialized else "stopped"
                    objects_count = len(self.server.ui_server.engine.scene_manager.root_node.children) if hasattr(self.server.ui_server.engine, 'scene_manager') and self.server.ui_server.engine.scene_manager else 0
                
                return {
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
        
        # 切换到HTML目录
        original_dir = os.getcwd()
        html_dir = os.path.join(os.path.dirname(__file__), "html")
        os.chdir(html_dir)
        
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
            # 保存实际绑定的HTTP端口
            self.actual_http_port = available_port
            
            # 创建HTTP服务器
            self.http_server = HTTPServer((self.host, available_port), CustomHTTPRequestHandler)
            # 将UI服务器实例传递给HTTP服务器
            self.http_server.ui_server = self
            
            self.logger.info(f"HTTP服务器启动在 http://{self.host}:{available_port}")
            
            # 启动服务器
            self.http_server.serve_forever()
        except Exception as e:
            self.logger.error(f"HTTP服务器启动失败: {e}")
        finally:
            # 恢复原始目录
            os.chdir(original_dir)
    
    def _handle_http_message(self, message_data, client):
        """处理HTTP消息
        
        Args:
            message_data: 消息数据
            client: 客户端请求处理器
        """
        try:
            message_type = message_data.get("type", "")
            data = message_data.get("data", "")
            
            self.logger.debug(f"处理HTTP消息类型: {message_type}, 数据: {data}")
            
            # 根据消息类型处理
            if message_type == "create_cube":
                self.logger.info("创建立方体")
                if hasattr(self.engine, '_create_cube'):
                    self.engine._create_cube()
            elif message_type == "create_sphere":
                self.logger.info("创建球体")
                if hasattr(self.engine, '_create_sphere'):
                    self.engine._create_sphere()
            elif message_type == "create_cylinder":
                self.logger.info("创建圆柱体")
                if hasattr(self.engine, '_create_cylinder'):
                    self.engine._create_cylinder()
            elif message_type == "create_plane":
                self.logger.info("创建平面")
                if hasattr(self.engine, '_create_plane'):
                    self.engine._create_plane()
            elif message_type == "create_cone":
                self.logger.info("创建圆锥体")
                if hasattr(self.engine, '_create_cone'):
                    self.engine._create_cone()
            elif message_type == "shutdown":
                self.logger.info("关闭引擎")
                if hasattr(self.engine, 'shutdown'):
                    self.engine.shutdown()
            elif message_type == "tool_select":
                self.logger.debug("选择工具被点击")
                if hasattr(self.engine, 'transform_manipulator'):
                    self.engine.transform_manipulator.set_transform_mode("select")
            elif message_type == "tool_move":
                self.logger.debug("移动工具被点击")
                if hasattr(self.engine, 'transform_manipulator'):
                    self.engine.transform_manipulator.set_transform_mode("translate")
            elif message_type == "tool_rotate":
                self.logger.debug("旋转工具被点击")
                if hasattr(self.engine, 'transform_manipulator'):
                    self.engine.transform_manipulator.set_transform_mode("rotate")
            elif message_type == "tool_scale":
                self.logger.debug("缩放工具被点击")
                if hasattr(self.engine, 'transform_manipulator'):
                    self.engine.transform_manipulator.set_transform_mode("scale")
            # 相机控制消息处理
            elif message_type == "camera_orbit":
                self.logger.debug(f"相机轨道控制: {data}")
                if hasattr(self.engine, 'camera_controller'):
                    try:
                        deltaX = data.get('deltaX', 0)
                        deltaY = data.get('deltaY', 0)
                        self.engine.camera_controller.orbit(deltaX, deltaY)
                    except Exception as e:
                        self.logger.error(f"相机轨道控制失败: {e}")
            elif message_type == "camera_pan":
                self.logger.debug(f"相机平移控制: {data}")
                if hasattr(self.engine, 'camera_controller'):
                    try:
                        deltaX = data.get('deltaX', 0)
                        deltaY = data.get('deltaY', 0)
                        self.engine.camera_controller.pan(deltaX, deltaY)
                    except Exception as e:
                        self.logger.error(f"相机平移控制失败: {e}")
            elif message_type == "camera_zoom":
                self.logger.debug(f"相机缩放控制: {data}")
                if hasattr(self.engine, 'camera_controller'):
                    try:
                        delta = data.get('delta', 0)
                        self.engine.camera_controller.zoom(delta)
                    except Exception as e:
                        self.logger.error(f"相机缩放控制失败: {e}")
            elif message_type == "camera_move":
                self.logger.debug(f"相机移动控制: {data}")
                if hasattr(self.engine, 'camera_controller'):
                    try:
                        direction = data.get('direction', 'forward')
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
            else:
                self.logger.warning(f"未知消息类型: {message_type}")
        except Exception as e:
            self.logger.error(f"处理HTTP消息失败: {e}", exc_info=True)
    
    def _adjust_render_quality_dynamically(self):
        """
        根据实际帧率动态调整渲染质量
        """
        if self.engine is None or not hasattr(self.engine, 'renderer') or not self.engine.renderer:
            return
        
        try:
            # 检查renderer是否有render_quality属性，没有则跳过
            if not hasattr(self.engine.renderer, 'render_quality'):
                # 只在首次检查时记录日志，避免频繁输出
                if not hasattr(self, '_quality_check_done'):
                    self.logger.info("Renderer没有render_quality属性，跳过动态渲染质量调整")
                    self._quality_check_done = True
                return
            
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
    
    def _open_browser(self):
        """打开浏览器"""
        # 打开浏览器
        url = f"http://{self.host}:{self.actual_http_port}"
        self.logger.info(f"打开浏览器: {url}")
        webbrowser.open(url)