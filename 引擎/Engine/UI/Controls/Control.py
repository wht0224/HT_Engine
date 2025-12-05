# -*- coding: utf-8 -*-
"""
UI控件基类
"""

from Engine.UI.Event import Event
from Engine.UI.Event import EventType

class Control:
    """UI控件基类"""
    
    def __init__(self, x, y, width, height):
        """初始化控件
        
        Args:
            x: 控件X坐标
            y: 控件Y坐标
            width: 控件宽度
            height: 控件高度
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.is_visible = True
        self.is_enabled = True
        self.is_hovered = False
        self.is_focused = False
        self.is_pressed = False
        self.parent = None
        self.children = []
        self.event_handlers = {}
        self.background_color = (0.3, 0.3, 0.3, 0.9)
        self.foreground_color = (1.0, 1.0, 1.0, 1.0)
        self.border_color = (0.5, 0.5, 0.5, 1.0)
        self.border_width = 1
        self.shadow_color = (0.0, 0.0, 0.0, 0.5)
        self.shadow_offset = (2, 2)
        self.padding = (5, 5, 5, 5)  # 上、右、下、左
        self.margin = (0, 0, 0, 0)  # 上、右、下、左
        self.font_size = 12
        self.name = "Control"
        self.tag = None
    
    def add_child(self, child):
        """添加子控件
        
        Args:
            child: 子控件
        """
        child.parent = self
        self.children.append(child)
    
    def remove_child(self, child):
        """移除子控件
        
        Args:
            child: 子控件
        """
        if child in self.children:
            child.parent = None
            self.children.remove(child)
    
    def contains_point(self, x, y):
        """检查点是否在控件内
        
        Args:
            x: 点的X坐标
            y: 点的Y坐标
            
        Returns:
            bool: 点是否在控件内
        """
        return (self.x <= x <= self.x + self.width and 
                self.y <= y <= self.y + self.height)
    
    def update(self, delta_time):
        """更新控件
        
        Args:
            delta_time: 帧时间（秒）
        """
        for child in self.children:
            child.update(delta_time)
    
    def render(self):
        """渲染控件"""
        if not self.is_visible:
            return
        
        # 渲染阴影
        self._render_shadow()
        
        # 渲染背景
        self._render_background()
        
        # 渲染内容
        self._render_content()
        
        # 渲染边框
        self._render_border()
        
        # 渲染子控件
        for child in self.children:
            child.render()
    
    def _render_shadow(self):
        """渲染阴影"""
        from OpenGL.GL import glBegin, glEnd, glColor4f, glVertex2f, GL_QUADS
        
        if self.shadow_offset[0] == 0 and self.shadow_offset[1] == 0:
            return
        
        shadow_x = self.x + self.shadow_offset[0]
        shadow_y = self.y - self.shadow_offset[1]
        
        glColor4f(*self.shadow_color)
        glBegin(GL_QUADS)
        glVertex2f(shadow_x, shadow_y)
        glVertex2f(shadow_x + self.width, shadow_y)
        glVertex2f(shadow_x + self.width, shadow_y + self.height)
        glVertex2f(shadow_x, shadow_y + self.height)
        glEnd()
    
    def _render_background(self):
        """渲染背景"""
        from OpenGL.GL import glBegin, glEnd, glColor4f, glVertex2f, GL_QUADS
        
        glColor4f(*self.background_color)
        glBegin(GL_QUADS)
        glVertex2f(self.x, self.y)
        glVertex2f(self.x + self.width, self.y)
        glVertex2f(self.x + self.width, self.y + self.height)
        glVertex2f(self.x, self.y + self.height)
        glEnd()
    
    def _render_content(self):
        """渲染内容，由子类实现"""
        pass
    
    def _render_border(self):
        """渲染边框"""
        from OpenGL.GL import glBegin, glEnd, glColor4f, glVertex2f, GL_LINES
        
        if self.border_width > 0:
            glColor4f(*self.border_color)
            glBegin(GL_LINES)
            # 上边框
            glVertex2f(self.x, self.y)
            glVertex2f(self.x + self.width, self.y)
            # 右边框
            glVertex2f(self.x + self.width, self.y)
            glVertex2f(self.x + self.width, self.y + self.height)
            # 下边框
            glVertex2f(self.x + self.width, self.y + self.height)
            glVertex2f(self.x, self.y + self.height)
            # 左边框
            glVertex2f(self.x, self.y + self.height)
            glVertex2f(self.x, self.y)
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
        self.is_pressed = True
        self._trigger_event(EventType.MOUSE_DOWN, button=button)
    
    def on_mouse_up(self, button):
        """鼠标释放事件
        
        Args:
            button: 鼠标按钮（0: 左键, 1: 中键, 2: 右键）
        """
        self.is_pressed = False
        self._trigger_event(EventType.MOUSE_UP, button=button)
        if self.is_hovered:
            self._trigger_event(EventType.MOUSE_CLICK, button=button)
    
    def on_mouse_click(self, button):
        """鼠标点击事件
        
        Args:
            button: 鼠标按钮（0: 左键, 1: 中键, 2: 右键）
        """
        self._trigger_event(EventType.MOUSE_CLICK, button=button)
    
    def on_key_down(self, key):
        """键盘按下事件
        
        Args:
            key: 按键码
        """
        self._trigger_event(EventType.KEY_DOWN, key=key)
    
    def on_key_up(self, key):
        """键盘释放事件
        
        Args:
            key: 按键码
        """
        self._trigger_event(EventType.KEY_UP, key=key)
    
    def on_focus_gained(self):
        """获得焦点事件"""
        self.is_focused = True
        self._trigger_event(EventType.FOCUS_GAINED)
    
    def on_focus_lost(self):
        """失去焦点事件"""
        self.is_focused = False
        self._trigger_event(EventType.FOCUS_LOST)
    
    def add_event_handler(self, event_type, callback):
        """添加事件处理器
        
        Args:
            event_type: 事件类型
            callback: 回调函数
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(callback)
    
    def remove_event_handler(self, event_type, callback):
        """移除事件处理器
        
        Args:
            event_type: 事件类型
            callback: 回调函数
        """
        if event_type in self.event_handlers:
            if callback in self.event_handlers[event_type]:
                self.event_handlers[event_type].remove(callback)
    
    def _trigger_event(self, event_type, **kwargs):
        """触发事件
        
        Args:
            event_type: 事件类型
            **kwargs: 事件参数
        """
        event = Event(event_type, self, **kwargs)
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                handler(event)
                if event.handled:
                    break
    
    def set_position(self, x, y):
        """设置控件位置
        
        Args:
            x: X坐标
            y: Y坐标
        """
        self.x = x
        self.y = y
    
    def set_size(self, width, height):
        """设置控件大小
        
        Args:
            width: 宽度
            height: 高度
        """
        self.width = width
        self.height = height
    
    def get_absolute_position(self):
        """获取控件的绝对位置
        
        Returns:
            tuple: 绝对位置 (x, y)
        """
        if self.parent:
            parent_pos = self.parent.get_absolute_position()
            return (self.x + parent_pos[0], self.y + parent_pos[1])
        return (self.x, self.y)
    
    def focus(self):
        """获取焦点"""
        # 这里将在UIManager中实现
        pass
    
    def unfocus(self):
        """失去焦点"""
        # 这里将在UIManager中实现
        pass
    
    def on_mouse_wheel(self, direction):
        """鼠标滚轮事件
        
        Args:
            direction: 滚轮方向（1: 向上, -1: 向下）
        """
        self._trigger_event(EventType.MOUSE_WHEEL, direction=direction)
