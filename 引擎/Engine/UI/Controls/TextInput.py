# -*- coding: utf-8 -*-
"""
文本输入控件类
"""

from Engine.UI.Controls.Control import Control
from Engine.UI.Event import EventType

class TextInput(Control):
    """文本输入控件类"""
    
    def __init__(self, x, y, width, height, text=""):
        """初始化文本输入控件
        
        Args:
            x: 控件X坐标
            y: 控件Y坐标
            width: 控件宽度
            height: 控件高度
            text: 初始文本
        """
        super().__init__(x, y, width, height)
        self.text = text
        self.name = "TextInput"
        self.is_focused = False
        self.cursor_position = len(text)
        self.cursor_visible = True
        self.cursor_blink_time = 0.5  # 光标闪烁时间（秒）
        self.cursor_blink_timer = 0.0
        self.text_color = (1.0, 1.0, 1.0, 1.0)
        self.cursor_color = (1.0, 1.0, 1.0, 1.0)
        self.focused_background = (0.3, 0.3, 0.3, 0.8)
        self.unfocused_background = (0.2, 0.2, 0.2, 0.8)
        self.border_color = (0.4, 0.4, 0.4, 1.0)
        self.border_width = 1
        self.is_password = False  # 是否为密码框
    
    def update(self, delta_time):
        """更新控件
        
        Args:
            delta_time: 帧时间（秒）
        """
        super().update(delta_time)
        
        # 更新光标闪烁
        if self.is_focused:
            self.cursor_blink_timer += delta_time
            if self.cursor_blink_timer >= self.cursor_blink_time:
                self.cursor_visible = not self.cursor_visible
                self.cursor_blink_timer = 0.0
    
    def _render_content(self):
        """渲染文本输入内容"""
        from OpenGL.GL import glColor4f, glRasterPos2f
        from OpenGL.GLUT import glutBitmapCharacter, GLUT_BITMAP_HELVETICA_12
        
        # 设置文本颜色
        glColor4f(*self.text_color)
        
        # 计算文本位置
        text_x = self.x + self.padding[3]
        text_y = self.y + (self.height + self.font_size) / 2
        
        # 设置文本位置
        glRasterPos2f(text_x, text_y)
        
        # 渲染文本
        display_text = "*" * len(self.text) if self.is_password else self.text
        for char in display_text:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(char))
        
        # 渲染光标
        if self.is_focused and self.cursor_visible:
            # 计算光标位置
            cursor_x = text_x + self.cursor_position * self.font_size * 0.6
            cursor_y = self.y + self.padding[2]
            cursor_height = self.height - self.padding[0] - self.padding[2]
            
            # 设置光标颜色
            glColor4f(*self.cursor_color)
            
            # 渲染光标
            from OpenGL.GL import glBegin, glEnd, glVertex2f, GL_LINES
            glBegin(GL_LINES)
            glVertex2f(cursor_x, cursor_y)
            glVertex2f(cursor_x, cursor_y + cursor_height)
            glEnd()
    
    def _render_background(self):
        """渲染背景"""
        from OpenGL.GL import glBegin, glEnd, glColor4f, glVertex2f, GL_QUADS
        
        # 根据焦点状态设置背景颜色
        color = self.focused_background if self.is_focused else self.unfocused_background
        
        glColor4f(*color)
        glBegin(GL_QUADS)
        glVertex2f(self.x, self.y)
        glVertex2f(self.x + self.width, self.y)
        glVertex2f(self.x + self.width, self.y + self.height)
        glVertex2f(self.x, self.y + self.height)
        glEnd()
    
    def on_mouse_down(self, button):
        """鼠标按下事件
        
        Args:
            button: 鼠标按钮（0: 左键, 1: 中键, 2: 右键）
        """
        if button == 0 and self.is_enabled:
            # 获取焦点
            self.is_focused = True
            self.cursor_visible = True
            self.cursor_blink_timer = 0.0
            self._trigger_event(EventType.FOCUS_GAINED)
    
    def on_key_down(self, key):
        """键盘按下事件
        
        Args:
            key: 按键码
        """
        if not self.is_focused:
            return
        
        if key == 8:  # 退格键
            if self.cursor_position > 0:
                self.text = self.text[:self.cursor_position - 1] + self.text[self.cursor_position:]
                self.cursor_position -= 1
                self._trigger_event(EventType.VALUE_CHANGED, text=self.text)
        elif key == 13:  # 回车键
            self.is_focused = False
            self._trigger_event(EventType.VALUE_CHANGED, text=self.text)
            self._trigger_event(EventType.FOCUS_LOST)
        elif key == 27:  # ESC键
            self.is_focused = False
            self._trigger_event(EventType.FOCUS_LOST)
        elif key == 127:  # 删除键
            if self.cursor_position < len(self.text):
                self.text = self.text[:self.cursor_position] + self.text[self.cursor_position + 1:]
                self._trigger_event(EventType.VALUE_CHANGED, text=self.text)
        elif key == 20:  # 大写锁定键
            pass  # 忽略
        elif 32 <= key <= 126:  # 可打印字符
            char = chr(key)
            self.text = self.text[:self.cursor_position] + char + self.text[self.cursor_position:]
            self.cursor_position += 1
            self._trigger_event(EventType.VALUE_CHANGED, text=self.text)
    
    def on_focus_gained(self):
        """获得焦点事件"""
        self.is_focused = True
        self.cursor_visible = True
        self.cursor_blink_timer = 0.0
        self._trigger_event(EventType.FOCUS_GAINED)
    
    def on_focus_lost(self):
        """失去焦点事件"""
        self.is_focused = False
        self.cursor_visible = False
        self._trigger_event(EventType.FOCUS_LOST)
    
    def set_text(self, text):
        """设置文本
        
        Args:
            text: 文本内容
        """
        self.text = text
        self.cursor_position = len(text)
    
    def get_text(self):
        """获取文本
        
        Returns:
            str: 文本内容
        """
        return self.text
    
    def set_password(self, is_password):
        """设置是否为密码框
        
        Args:
            is_password: 是否为密码框
        """
        self.is_password = is_password
    
    def focus(self):
        """获取焦点"""
        self.is_focused = True
        self.cursor_visible = True
        self.cursor_blink_timer = 0.0
    
    def unfocus(self):
        """失去焦点"""
        self.is_focused = False
        self.cursor_visible = False
