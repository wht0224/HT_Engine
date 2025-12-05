# -*- coding: utf-8 -*-
"""
滑块控件类，用于调整数值
"""

from Engine.UI.Controls.Control import Control
from Engine.UI.Event import EventType

class Slider(Control):
    """滑块控件类，用于调整数值"""
    
    def __init__(self, x, y, width, height, min_value=0.0, max_value=1.0, value=0.5, orientation="horizontal"):
        """初始化滑块
        
        Args:
            x: 滑块X坐标
            y: 滑块Y坐标
            width: 滑块宽度
            height: 滑块高度
            min_value: 最小值
            max_value: 最大值
            value: 当前值
            orientation: 方向：horizontal, vertical
        """
        super().__init__(x, y, width, height)
        self.min_value = min_value
        self.max_value = max_value
        self.value = value
        self.orientation = orientation
        self.name = "Slider"
        self.is_dragging = False
        self.thumb_color = (0.4, 0.4, 0.4, 1.0)
        self.track_color = (0.2, 0.2, 0.2, 1.0)
        self.active_color = (0.6, 0.6, 0.6, 1.0)
        self.thumb_size = (10, 20) if orientation == "horizontal" else (20, 10)
    
    def _render_content(self):
        """渲染滑块内容"""
        from OpenGL.GL import glBegin, glEnd, glColor4f, glVertex2f, GL_QUADS
        
        # 渲染轨道
        glColor4f(*self.track_color)
        if self.orientation == "horizontal":
            # 水平滑块
            track_y = self.y + (self.height - 4) / 2
            glBegin(GL_QUADS)
            glVertex2f(self.x, track_y)
            glVertex2f(self.x + self.width, track_y)
            glVertex2f(self.x + self.width, track_y + 4)
            glVertex2f(self.x, track_y + 4)
            glEnd()
        else:
            # 垂直滑块
            track_x = self.x + (self.width - 4) / 2
            glBegin(GL_QUADS)
            glVertex2f(track_x, self.y)
            glVertex2f(track_x + 4, self.y)
            glVertex2f(track_x + 4, self.y + self.height)
            glVertex2f(track_x, self.y + self.height)
            glEnd()
        
        # 渲染滑块拇指
        if self.is_dragging:
            thumb_color = self.active_color
        else:
            thumb_color = self.thumb_color
        
        glColor4f(*thumb_color)
        
        # 计算拇指位置
        normalized_value = (self.value - self.min_value) / (self.max_value - self.min_value)
        
        if self.orientation == "horizontal":
            # 水平滑块
            thumb_x = self.x + normalized_value * (self.width - self.thumb_size[0])
            thumb_y = self.y + (self.height - self.thumb_size[1]) / 2
        else:
            # 垂直滑块
            thumb_x = self.x + (self.width - self.thumb_size[0]) / 2
            thumb_y = self.y + (1.0 - normalized_value) * (self.height - self.thumb_size[1])
        
        # 渲染拇指
        glBegin(GL_QUADS)
        glVertex2f(thumb_x, thumb_y)
        glVertex2f(thumb_x + self.thumb_size[0], thumb_y)
        glVertex2f(thumb_x + self.thumb_size[0], thumb_y + self.thumb_size[1])
        glVertex2f(thumb_x, thumb_y + self.thumb_size[1])
        glEnd()
    
    def on_mouse_down(self, button):
        """鼠标按下事件
        
        Args:
            button: 鼠标按钮（0: 左键, 1: 中键, 2: 右键）
        """
        if button == 0 and self.is_enabled:
            # 检查是否点击了拇指
            normalized_value = (self.value - self.min_value) / (self.max_value - self.min_value)
            
            if self.orientation == "horizontal":
                # 水平滑块
                thumb_x = self.x + normalized_value * (self.width - self.thumb_size[0])
                thumb_y = self.y + (self.height - self.thumb_size[1]) / 2
            else:
                # 垂直滑块
                thumb_x = self.x + (self.width - self.thumb_size[0]) / 2
                thumb_y = self.y + (1.0 - normalized_value) * (self.height - self.thumb_size[1])
            
            # 检查鼠标是否在拇指内
            if (thumb_x <= self.mouse_position[0] <= thumb_x + self.thumb_size[0] and 
                thumb_y <= self.mouse_position[1] <= thumb_y + self.thumb_size[1]):
                self.is_dragging = True
    
    def on_mouse_up(self, button):
        """鼠标释放事件
        
        Args:
            button: 鼠标按钮（0: 左键, 1: 中键, 2: 右键）
        """
        if button == 0:
            self.is_dragging = False
    
    def on_mouse_move(self, x, y):
        """鼠标移动事件
        
        Args:
            x: 鼠标X坐标
            y: 鼠标Y坐标
        """
        if self.is_dragging:
            # 计算新值
            if self.orientation == "horizontal":
                # 水平滑块
                new_value = ((x - self.x) / self.width) * (self.max_value - self.min_value) + self.min_value
            else:
                # 垂直滑块
                new_value = ((self.y + self.height - y) / self.height) * (self.max_value - self.min_value) + self.min_value
            
            # 限制值在范围内
            new_value = max(self.min_value, min(self.max_value, new_value))
            
            # 如果值改变，触发事件
            if new_value != self.value:
                old_value = self.value
                self.value = new_value
                self._trigger_event(EventType.VALUE_CHANGED, old_value=old_value, new_value=new_value)
    
    def set_value(self, value):
        """设置滑块值
        
        Args:
            value: 滑块值
        """
        old_value = self.value
        self.value = max(self.min_value, min(self.max_value, value))
        if self.value != old_value:
            self._trigger_event(EventType.VALUE_CHANGED, old_value=old_value, new_value=self.value)
    
    def get_value(self):
        """获取滑块值
        
        Returns:
            float: 滑块值
        """
        return self.value
    
    def set_range(self, min_value, max_value):
        """设置滑块范围
        
        Args:
            min_value: 最小值
            max_value: 最大值
        """
        self.min_value = min_value
        self.max_value = max_value
        # 确保当前值在新范围内
        self.set_value(self.value)
