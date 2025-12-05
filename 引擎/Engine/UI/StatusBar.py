# -*- coding: utf-8 -*-
"""
状态栏类，实现底部状态栏
"""

from Engine.UI.Controls.Control import Control

class StatusBar(Control):
    """状态栏类，实现底部状态栏"""
    
    def __init__(self, x, y, width, height):
        """初始化状态栏
        
        Args:
            x: X坐标
            y: Y坐标
            width: 宽度
            height: 高度
        """
        super().__init__(x, y, width, height)
        self.name = "StatusBar"
        self.background_color = (0.15, 0.15, 0.15, 0.9)
        self.foreground_color = (0.8, 0.8, 0.8, 1.0)
        # 确保状态栏可见
        self.is_visible = True
        
        # 状态栏信息
        self.status_info = {
            "mode": "编辑模式",
            "selection": "未选择对象",
            "fps": "FPS: 0",
            "view": "透视视图"
        }
    
    def update_status(self):
        """更新状态栏信息"""
        # 简化更新逻辑，只保留必要的状态更新
        pass
    
    def _render_content(self):
        """渲染状态栏内容"""
        from OpenGL.GL import glColor4f, glRasterPos2f, glBegin, glEnd, glVertex2f, GL_LINES
        from OpenGL.GLUT import glutBitmapCharacter, GLUT_BITMAP_HELVETICA_10
        
        # 设置文本颜色
        glColor4f(*self.foreground_color)
        
        # 渲染状态栏文本
        text_y = self.y + self.height - 8
        
        # 渲染模式信息
        mode_text = self.status_info["mode"]
        glRasterPos2f(self.x + 10, text_y)
        for char in mode_text:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, ord(char))
        
        # 渲染分隔线
        glBegin(GL_LINES)
        glVertex2f(self.x + 120, self.y + 2)
        glVertex2f(self.x + 120, self.y + self.height - 2)
        glEnd()
        
        # 渲染选择信息
        selection_text = self.status_info["selection"]
        glRasterPos2f(self.x + 130, text_y)
        for char in selection_text:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, ord(char))
        
        # 渲染分隔线
        glBegin(GL_LINES)
        glVertex2f(self.x + 400, self.y + 2)
        glVertex2f(self.x + 400, self.y + self.height - 2)
        glEnd()
        
        # 渲染视图信息
        view_text = self.status_info["view"]
        glRasterPos2f(self.x + 410, text_y)
        for char in view_text:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, ord(char))
        
        # 渲染分隔线
        glBegin(GL_LINES)
        glVertex2f(self.x + 550, self.y + 2)
        glVertex2f(self.x + 550, self.y + self.height - 2)
        glEnd()
        
        # 渲染FPS信息
        fps_text = self.status_info["fps"]
        glRasterPos2f(self.x + 560, text_y)
        for char in fps_text:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, ord(char))
    
    def set_mode(self, mode):
        """设置当前模式
        
        Args:
            mode: 当前模式
        """
        self.status_info["mode"] = mode
    
    def set_selection(self, selection):
        """设置选择信息
        
        Args:
            selection: 选择信息
        """
        self.status_info["selection"] = selection
    
    def set_fps(self, fps):
        """设置FPS信息
        
        Args:
            fps: FPS值
        """
        self.status_info["fps"] = f"FPS: {fps:.1f}"
    
    def set_view(self, view_type):
        """设置视图类型
        
        Args:
            view_type: 视图类型
        """
        self.status_info["view"] = f"{view_type}视图"
    
