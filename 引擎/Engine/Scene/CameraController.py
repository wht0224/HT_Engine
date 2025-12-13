# -*- coding: utf-8 -*-
"""
相机控制器类，用于处理用户输入并控制相机
"""

from Engine.Math import Vector3, Quaternion
import numpy as np

class CameraController:
    """相机控制器类，用于处理用户输入并控制相机"""
    
    def __init__(self, camera):
        """初始化相机控制器
        
        Args:
            camera: 相机对象
        """
        self.camera = camera
        self.move_speed = 5.0  # 移动速度
        self.rotate_speed = 0.01  # 旋转速度（弧度/像素）
        self.zoom_speed = 1.1  # 缩放速度
        self.is_mouse_down = [False, False, False]  # 鼠标按钮状态
        self.last_mouse_pos = (0, 0)  # 上次鼠标位置
        self.is_ui_active = False  # UI是否激活，激活时不处理相机控制
    
    def update(self, delta_time):
        """更新相机控制器
        
        Args:
            delta_time: 帧时间（秒）
        """
        # 这里将在Platform类中处理具体的输入事件
        pass
    
    def handle_mouse_down(self, button, x, y):
        """处理鼠标按下事件
        
        Args:
            button: 鼠标按钮（0: 左键, 1: 中键, 2: 右键）
            x: 鼠标X坐标
            y: 鼠标Y坐标
        """
        if self.is_ui_active:
            return
        
        self.is_mouse_down[button] = True
        self.last_mouse_pos = (x, y)
    
    def handle_mouse_up(self, button, x, y):
        """处理鼠标释放事件
        
        Args:
            button: 鼠标按钮（0: 左键, 1: 中键, 2: 右键）
            x: 鼠标X坐标
            y: 鼠标Y坐标
        """
        self.is_mouse_down[button] = False
    
    def handle_mouse_move(self, x, y):
        """处理鼠标移动事件
        
        Args:
            x: 鼠标X坐标
            y: 鼠标Y坐标
        """
        if self.is_ui_active:
            return
        
        dx = x - self.last_mouse_pos[0]
        dy = y - self.last_mouse_pos[1]
        
        # 右键拖动：旋转相机
        if self.is_mouse_down[2]:
            self.camera.rotate_yaw(dx * self.rotate_speed)
            self.camera.rotate_pitch(-dy * self.rotate_speed)
        
        # 中键拖动：平移相机
        elif self.is_mouse_down[1]:
            # 计算平移距离
            move_speed = self.move_speed * 0.01
            self.camera.move_right(-dx * move_speed)
            self.camera.move_up(dy * move_speed)
        
        self.last_mouse_pos = (x, y)
    
    def handle_mouse_wheel(self, delta):
        """处理鼠标滚轮事件
        
        Args:
            delta: 滚轮滚动量
        """
        if self.is_ui_active:
            return
        
        # 滚轮：缩放相机
        if delta > 0:
            self.camera.zoom(1.0 / self.zoom_speed)
        else:
            self.camera.zoom(self.zoom_speed)
    
    def handle_key_down(self, key):
        """处理键盘按下事件
        
        Args:
            key: 按键码
        """
        if self.is_ui_active:
            return
        
        # WASD：移动相机
        if key == ord('w') or key == ord('W'):
            self.camera.move_forward(self.move_speed * 0.016)  # 假设60fps
        elif key == ord('s') or key == ord('S'):
            self.camera.move_forward(-self.move_speed * 0.016)
        elif key == ord('a') or key == ord('A'):
            self.camera.move_right(-self.move_speed * 0.016)
        elif key == ord('d') or key == ord('D'):
            self.camera.move_right(self.move_speed * 0.016)
        elif key == ord('q') or key == ord('Q'):
            self.camera.move_up(-self.move_speed * 0.016)
        elif key == ord('e') or key == ord('E'):
            self.camera.move_up(self.move_speed * 0.016)
    
    def set_ui_active(self, active):
        """设置UI是否激活
        
        Args:
            active: UI是否激活
        """
        self.is_ui_active = active
    
    def get_ui_active(self):
        """获取UI是否激活
        
        Returns:
            bool: UI是否激活
        """
        return self.is_ui_active
    
    def set_move_speed(self, speed):
        """设置移动速度
        
        Args:
            speed: 移动速度
        """
        self.move_speed = speed
    
    def set_rotate_speed(self, speed):
        """设置旋转速度
        
        Args:
            speed: 旋转速度（弧度/像素）
        """
        self.rotate_speed = speed
    
    def set_zoom_speed(self, speed):
        """设置缩放速度
        
        Args:
            speed: 缩放速度
        """
        self.zoom_speed = speed
