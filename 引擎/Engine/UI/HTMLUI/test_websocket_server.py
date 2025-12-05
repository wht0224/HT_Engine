#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试HTML UI服务器的WebSocket功能
"""

import sys
import os

# 添加引擎根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from Engine.UI.HTMLUI.html_ui_server import HTMLUIServer

class MockEngine:
    """模拟引擎类，用于测试HTML UI服务器"""
    def __init__(self):
        self.is_initialized = False
        self.renderer = None
        self.scene_manager = None
        self.camera_controller = None
        self.transform_manipulator = None

# 创建模拟引擎实例
mock_engine = MockEngine()

# 初始化HTML UI服务器
ui_server = HTMLUIServer(mock_engine)

# 启动服务器
ui_server.start()

print("HTML UI服务器已启动，正在测试WebSocket连接...")
print(f"HTTP服务器: http://localhost:{ui_server.http_port}")
print(f"WebSocket服务器: ws://localhost:{ui_server.websocket_port}")
print("按Ctrl+C停止服务器")

# 保持服务器运行
try:
    import time
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n停止HTML UI服务器...")
    ui_server.stop()
    print("服务器已停止")
