# -*- coding: utf-8 -*-
"""
按钮控件类
"""

from Engine.UI.Controls.Control import Control
from Engine.UI.Event import EventType

class Button(Control):
    """按钮控件类"""
    
    def __init__(self, x, y, width, height, text="Button"):
        """初始化按钮
        
        Args:
            x: 按钮X坐标
            y: 按钮Y坐标
            width: 按钮宽度
            height: 按钮高度
            text: 按钮文本
        """
        super().__init__(x, y, width, height)
        self.text = text
        self.name = "Button"
        self.is_hovered = False
        self.is_pressed = False
        self.disabled_color = (0.3, 0.3, 0.3, 0.5)
        self.hover_color = (0.3, 0.3, 0.3, 0.9)
        self.pressed_color = (0.1, 0.1, 0.1, 0.9)
        self.text_color = (1.0, 1.0, 1.0, 1.0)
    
    def _render_content(self):
        """渲染按钮内容"""
        from OpenGL.GL import glColor4f, glRasterPos2f
        from OpenGL.GLUT import glutBitmapCharacter, GLUT_BITMAP_HELVETICA_12
        
        # 设置文本颜色
        glColor4f(*self.text_color)
        
        # 计算文本位置（居中）
        text_width = len(self.text) * self.font_size * 0.6
        text_height = self.font_size
        text_x = self.x + (self.width - text_width) / 2
        text_y = self.y + (self.height + text_height) / 2
        
        # 设置文本位置
        glRasterPos2f(text_x, text_y)
        
        # 渲染文本
        for char in self.text:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(char))
    
    def _render_background(self):
        """渲染按钮背景"""
        from OpenGL.GL import glBegin, glEnd, glColor4f, glVertex2f, GL_QUADS
        
        # 根据按钮状态设置背景颜色
        if not self.is_enabled:
            color = self.disabled_color
        elif self.is_pressed:
            color = self.pressed_color
        elif self.is_hovered:
            color = self.hover_color
        else:
            color = self.background_color
        
        glColor4f(*color)
        glBegin(GL_QUADS)
        glVertex2f(self.x, self.y)
        glVertex2f(self.x + self.width, self.y)
        glVertex2f(self.x + self.width, self.y + self.height)
        glVertex2f(self.x, self.y + self.height)
        glEnd()
    
    def on_mouse_enter(self):
        """鼠标进入事件"""
        self.is_hovered = True
        self._trigger_event(EventType.MOUSE_ENTER)
    
    def on_mouse_leave(self):
        """鼠标离开事件"""
        self.is_hovered = False
        self._trigger_event(EventType.MOUSE_LEAVE)
    
    def on_mouse_down(self, button):
        """鼠标按下事件
        
        Args:
            button: 鼠标按钮（0: 左键, 1: 中键, 2: 右键）
        """
        if button == 0 and self.is_enabled:
            self.is_pressed = True
            self._trigger_event(EventType.MOUSE_DOWN, button=button)
    
    def on_mouse_up(self, button):
        """鼠标释放事件
        
        Args:
            button: 鼠标按钮（0: 左键, 1: 中键, 2: 右键）
        """
        if button == 0 and self.is_enabled:
            was_pressed = self.is_pressed
            self.is_pressed = False
            self._trigger_event(EventType.MOUSE_UP, button=button)
            if was_pressed and self.is_hovered:
                self._trigger_event(EventType.MOUSE_CLICK, button=button)
    
    def set_text(self, text):
        """设置按钮文本
        
        Args:
            text: 按钮文本
        """
        self.text = text
    
    def get_text(self):
        """获取按钮文本
        
        Returns:
            str: 按钮文本
        """
        return self.text
