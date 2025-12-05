# -*- coding: utf-8 -*-
"""
视图端口类，负责渲染3D场景
"""

from Engine.UI.Controls.Control import Control

class Viewport(Control):
    """视图端口类，负责渲染3D场景"""
    
    def __init__(self, x, y, width, height):
        """初始化视图端口
        
        Args:
            x: X坐标
            y: Y坐标
            width: 宽度
            height: 高度
        """
        super().__init__(x, y, width, height)
        self.name = "Viewport"
        self.selected_object = None
        self.transform_manipulator = None
        self.camera_controller = None
        
        # 设置视图端口背景色为深灰色，与主窗口背景色区分
        self.background_color = (0.15, 0.15, 0.15, 1.0)
        # 确保视图端口可见
        self.is_visible = True
        
        # 视图设置
        self.view_type = "perspective"  # 透视视图
        self.show_grid = True
        self.show_axes = True
        self.show_wireframe = False
    
    def set_selected_object(self, obj):
        """设置选中的对象
        
        Args:
            obj: 选中的对象
        """
        self.selected_object = obj
        
        # 更新变换操纵器
        from Engine import get_engine
        engine = get_engine()
        if engine and hasattr(engine, 'transform_manipulator'):
            engine.transform_manipulator.set_selected_node(obj)
    
    def _render_content(self):
        """渲染视图端口内容"""
        # 渲染网格
        if self.show_grid:
            self._render_grid()
        
        # 渲染坐标轴
        if self.show_axes:
            self._render_axes()
    
    def _render_grid(self):
        """渲染网格"""
        from OpenGL.GL import (
            glBegin, glEnd, glColor4f, glVertex2f, GL_LINES, glLineWidth,
            glMatrixMode, GL_PROJECTION, GL_MODELVIEW, glLoadIdentity, glOrtho
        )
        
        # 保存当前矩阵
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.width, 0, self.height, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # 设置网格颜色和线宽
        glColor4f(0.5, 0.5, 0.5, 0.5)
        glLineWidth(1.0)
        
        # 渲染网格线
        grid_size = 20
        grid_step = 10
        
        # 计算网格范围
        center_x = self.width // 2
        center_y = self.height // 2
        
        # 渲染垂直线
        glBegin(GL_LINES)
        for x in range(center_x % grid_step, self.width, grid_step):
            glVertex2f(x, 0)
            glVertex2f(x, self.height)
        
        # 渲染水平线
        for y in range(center_y % grid_step, self.height, grid_step):
            glVertex2f(0, y)
            glVertex2f(self.width, y)
        glEnd()
        
        # 渲染中心十字线
        glColor4f(0.8, 0.8, 0.2, 0.8)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        # 垂直线
        glVertex2f(center_x, 0)
        glVertex2f(center_x, self.height)
        # 水平线
        glVertex2f(0, center_y)
        glVertex2f(self.width, center_y)
        glEnd()
    
    def _render_axes(self):
        """渲染坐标轴"""
        from OpenGL.GL import (
            glBegin, glEnd, glColor4f, glVertex3f, GL_LINES, glLineWidth,
            glMatrixMode, GL_PROJECTION, GL_MODELVIEW, glLoadIdentity, glOrtho
        )
        
        # 保存当前矩阵
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.width, 0, self.height, -10, 10)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # 设置线宽
        glLineWidth(3.0)
        
        # 计算坐标轴位置（右下角）
        origin_x = self.width - 100
        origin_y = 100
        axis_length = 50
        
        # 渲染X轴（红色）
        glColor4f(1.0, 0.0, 0.0, 1.0)
        glBegin(GL_LINES)
        glVertex3f(origin_x, origin_y, 0.0)
        glVertex3f(origin_x + axis_length, origin_y, 0.0)
        glEnd()
        
        # 渲染Y轴（绿色）
        glColor4f(0.0, 1.0, 0.0, 1.0)
        glBegin(GL_LINES)
        glVertex3f(origin_x, origin_y, 0.0)
        glVertex3f(origin_x, origin_y + axis_length, 0.0)
        glEnd()
        
        # 渲染Z轴（蓝色）
        glColor4f(0.0, 0.0, 1.0, 1.0)
        glBegin(GL_LINES)
        glVertex3f(origin_x, origin_y, 0.0)
        glVertex3f(origin_x, origin_y, axis_length)
        glEnd()
    
    def set_transform_manipulator(self, manipulator):
        """设置变换操纵器
        
        Args:
            manipulator: 变换操纵器对象
        """
        self.transform_manipulator = manipulator
    
    def set_camera_controller(self, controller):
        """设置相机控制器
        
        Args:
            controller: 相机控制器对象
        """
        self.camera_controller = controller
    
    def handle_resize(self, width, height):
        """处理视图大小变化
        
        Args:
            width: 新宽度
            height: 新高度
        """
        self.set_size(width, height)
        
        # 更新相机投影
        from Engine import get_engine
        engine = get_engine()
        if engine and engine.scene_manager and engine.scene_manager.active_camera:
            engine.scene_manager.active_camera.set_viewport(width, height)
    
    def toggle_grid(self):
        """切换网格显示"""
        self.show_grid = not self.show_grid
    
    def toggle_axes(self):
        """切换坐标轴显示"""
        self.show_axes = not self.show_axes
    
    def toggle_wireframe(self):
        """切换线框显示"""
        self.show_wireframe = not self.show_wireframe
    
    def set_view_type(self, view_type):
        """设置视图类型
        
        Args:
            view_type: 视图类型 (perspective/orthographic)
        """
        self.view_type = view_type
        
        # 更新相机投影
        from Engine import get_engine
        engine = get_engine()
        if engine and engine.scene_manager and engine.scene_manager.active_camera:
            camera = engine.scene_manager.active_camera
            if view_type == "perspective":
                camera.set_perspective(60, self.width / self.height, 0.1, 1000.0)
            else:
                camera.set_orthographic(-5, 5, -5, 5, 0.1, 1000.0)