# -*- coding: utf-8 -*-
"""
HTTP API服务器 - 让运行中的引擎接收外部命令
HTTP API Server - Allow running engine to receive external commands
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import logging

# 禁用Flask的日志输出
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


class EngineAPIServer:
    """引擎HTTP API服务器"""

    def __init__(self, engine, host='127.0.0.1', port=5000):
        """
        初始化API服务器

        Args:
            engine: 引擎实例
            host: 监听地址
            port: 监听端口
        """
        self.engine = engine
        self.host = host
        self.port = port

        # 创建Flask应用
        self.app = Flask(__name__)
        CORS(self.app)  # 允许跨域

        # 注册路由
        self._register_routes()

        # 服务器线程
        self.server_thread = None
        self.running = False

    def _register_routes(self):
        """注册API路由"""

        @self.app.route('/api/status', methods=['GET'])
        def get_status():
            """获取引擎状态"""
            return jsonify({
                'status': 'running',
                'engine': 'initialized' if self.engine.is_initialized else 'not_initialized'
            })

        @self.app.route('/api/create_cube', methods=['POST'])
        def create_cube():
            """创建立方体"""
            data = request.json
            size = data.get('size', 1.0)
            name = data.get('name', 'Cube')
            position = data.get('position', [0, 0, 0])

            print(f"[HTTP API] 收到创建立方体请求: size={size}, name={name}, position={position}")

            try:
                from Engine.MCP import ModelingAPI
                modeling = ModelingAPI(self.engine)
                mesh = modeling.create_cube(size, name)
                modeling.add_to_scene(mesh, position, name=name)
                print(f"[HTTP API] 立方体创建成功: {len(mesh.vertices)} 顶点")

                # 在视口中显示（使用Tkinter线程安全的方式）
                if hasattr(self.engine, 'tk_ui') and self.engine.tk_ui:
                    if hasattr(self.engine.tk_ui, 'viewport'):
                        # 使用after方法在主线程中执行
                        self.engine.tk_ui.root.after(0, lambda: self.engine.tk_ui.viewport.set_mesh(mesh))
                        print(f"[HTTP API] 已更新视口显示")

                return jsonify({
                    'success': True,
                    'message': f'成功创建立方体 {name}',
                    'vertices': len(mesh.vertices),
                    'triangles': len(mesh.indices) // 3
                })
            except Exception as e:
                print(f"[HTTP API] 创建立方体失败: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/create_sphere', methods=['POST'])
        def create_sphere():
            """创建球体"""
            data = request.json
            radius = data.get('radius', 1.0)
            segments = data.get('segments', 32)
            rings = data.get('rings', 16)
            name = data.get('name', 'Sphere')
            position = data.get('position', [0, 0, 0])

            print(f"[HTTP API] 收到创建球体请求: radius={radius}, segments={segments}, name={name}")

            try:
                from Engine.MCP import ModelingAPI
                modeling = ModelingAPI(self.engine)
                mesh = modeling.create_sphere(radius, segments, rings, name)
                modeling.add_to_scene(mesh, position, name=name)
                print(f"[HTTP API] 球体创建成功: {len(mesh.vertices)} 顶点")

                if hasattr(self.engine, 'tk_ui') and self.engine.tk_ui:
                    if hasattr(self.engine.tk_ui, 'viewport'):
                        self.engine.tk_ui.root.after(0, lambda: self.engine.tk_ui.viewport.set_mesh(mesh))
                        print(f"[HTTP API] 已更新视口显示")

                return jsonify({
                    'success': True,
                    'message': f'成功创建球体 {name}',
                    'vertices': len(mesh.vertices),
                    'triangles': len(mesh.indices) // 3
                })
            except Exception as e:
                print(f"[HTTP API] 创建球体失败: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/create_cylinder', methods=['POST'])
        def create_cylinder():
            """创建圆柱体"""
            data = request.json
            radius = data.get('radius', 1.0)
            height = data.get('height', 2.0)
            segments = data.get('segments', 32)
            name = data.get('name', 'Cylinder')
            position = data.get('position', [0, 0, 0])

            print(f"[HTTP API] 收到创建圆柱体请求: radius={radius}, height={height}, name={name}")

            try:
                from Engine.MCP import ModelingAPI
                modeling = ModelingAPI(self.engine)
                mesh = modeling.create_cylinder(radius, height, segments, name)
                modeling.add_to_scene(mesh, position, name=name)
                print(f"[HTTP API] 圆柱体创建成功: {len(mesh.vertices)} 顶点")

                if hasattr(self.engine, 'tk_ui') and self.engine.tk_ui:
                    if hasattr(self.engine.tk_ui, 'viewport'):
                        self.engine.tk_ui.root.after(0, lambda: self.engine.tk_ui.viewport.set_mesh(mesh))
                        print(f"[HTTP API] 已更新视口显示")

                return jsonify({
                    'success': True,
                    'message': f'成功创建圆柱体 {name}',
                    'vertices': len(mesh.vertices),
                    'triangles': len(mesh.indices) // 3
                })
            except Exception as e:
                print(f"[HTTP API] 创建圆柱体失败: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/create_plane', methods=['POST'])
        def create_plane():
            """创建平面"""
            data = request.json
            width = data.get('width', 2.0)
            height = data.get('height', 2.0)
            name = data.get('name', 'Plane')
            position = data.get('position', [0, 0, 0])

            print(f"[HTTP API] 收到创建平面请求: width={width}, height={height}, name={name}")

            try:
                from Engine.MCP import ModelingAPI
                modeling = ModelingAPI(self.engine)
                mesh = modeling.create_plane(width, height, name)
                modeling.add_to_scene(mesh, position, name=name)
                print(f"[HTTP API] 平面创建成功: {len(mesh.vertices)} 顶点")

                if hasattr(self.engine, 'tk_ui') and self.engine.tk_ui:
                    if hasattr(self.engine.tk_ui, 'viewport'):
                        self.engine.tk_ui.root.after(0, lambda: self.engine.tk_ui.viewport.set_mesh(mesh))
                        print(f"[HTTP API] 已更新视口显示")

                return jsonify({
                    'success': True,
                    'message': f'成功创建平面 {name}',
                    'vertices': len(mesh.vertices),
                    'triangles': len(mesh.indices) // 3
                })
            except Exception as e:
                print(f"[HTTP API] 创建平面失败: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/create_cone', methods=['POST'])
        def create_cone():
            """创建圆锥体"""
            data = request.json
            radius = data.get('radius', 1.0)
            height = data.get('height', 2.0)
            segments = data.get('segments', 32)
            name = data.get('name', 'Cone')
            position = data.get('position', [0, 0, 0])

            print(f"[HTTP API] 收到创建圆锥体请求: radius={radius}, height={height}, name={name}")

            try:
                from Engine.MCP import ModelingAPI
                modeling = ModelingAPI(self.engine)
                mesh = modeling.create_cone(radius, height, segments, name)
                modeling.add_to_scene(mesh, position, name=name)
                print(f"[HTTP API] 圆锥体创建成功: {len(mesh.vertices)} 顶点")

                if hasattr(self.engine, 'tk_ui') and self.engine.tk_ui:
                    if hasattr(self.engine.tk_ui, 'viewport'):
                        self.engine.tk_ui.root.after(0, lambda: self.engine.tk_ui.viewport.set_mesh(mesh))
                        print(f"[HTTP API] 已更新视口显示")

                return jsonify({
                    'success': True,
                    'message': f'成功创建圆锥体 {name}',
                    'vertices': len(mesh.vertices),
                    'triangles': len(mesh.indices) // 3
                })
            except Exception as e:
                print(f"[HTTP API] 创建圆锥体失败: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/set_shading_mode', methods=['POST'])
        def set_shading_mode():
            """设置渲染模式"""
            data = request.json
            mode = data.get('mode', 'rendered')

            print(f"[HTTP API] 收到设置渲染模式请求: mode={mode}")

            try:
                if hasattr(self.engine, 'tk_ui') and self.engine.tk_ui:
                    if hasattr(self.engine.tk_ui, 'viewport'):
                        self.engine.tk_ui.root.after(0, lambda: self.engine.tk_ui.viewport.set_shading_mode(mode))
                        print(f"[HTTP API] 渲染模式已切换到: {mode}")
                        return jsonify({
                            'success': True,
                            'message': f'已切换到 {mode} 模式'
                        })
                return jsonify({'success': False, 'error': '视口未初始化'}), 400
            except Exception as e:
                print(f"[HTTP API] 设置渲染模式失败: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'success': False, 'error': str(e)}), 500

    def start(self):
        """启动API服务器（后台线程）"""
        if self.running:
            return

        def run_server():
            self.app.run(host=self.host, port=self.port, threaded=True, use_reloader=False)

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        self.running = True

        print(f"✅ HTTP API服务器已启动: http://{self.host}:{self.port}")
        print(f"   AI可以通过HTTP调用引擎了！")

    def stop(self):
        """停止API服务器"""
        self.running = False
