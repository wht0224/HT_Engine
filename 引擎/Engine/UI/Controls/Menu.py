# -*- coding: utf-8 -*-
"""
菜单控件类
"""

from Engine.UI.Controls.Control import Control
from Engine.UI.Controls.Button import Button
from Engine.UI.Event import EventType
from OpenGL.GL import glBegin, glEnd, glColor4f, glVertex2f, GL_TRIANGLES

class Menu(Control):
    """菜单控件类"""
    
    def __init__(self, x, y, width, height, items=None, text="Menu"):
        """
        初始化菜单
        
        Args:
            x: 菜单X坐标
            y: 菜单Y坐标
            width: 菜单宽度
            height: 菜单项高度
            items: 菜单项列表，格式：[{"text": "Item 1", "callback": callback_func}, ...]
            text: 菜单标题文本
        """
        super().__init__(x, y, width, height)
        self.items = items or []
        self.name = "Menu"
        self.is_open = False
        self.selected_item = -1
        self.item_height = height
        self.background_color = (0.2, 0.2, 0.2, 0.9)
        self.hover_color = (0.3, 0.3, 0.3, 0.9)
        self.selected_color = (0.4, 0.4, 0.4, 0.9)
        self.text_color = (1.0, 1.0, 1.0, 1.0)
        self.border_color = (0.4, 0.4, 0.4, 1.0)
        self.border_width = 1
        self.text = text
        
        # 创建菜单项按钮
        self._create_menu_items()
    
    def _create_menu_items(self):
        """创建菜单项按钮"""
        self.menu_buttons = []
        for i, item in enumerate(self.items):
            button = Button(
                self.x, 
                self.y + (i + 1) * self.item_height, 
                self.width, 
                self.item_height, 
                item["text"]
            )
            button.background_color = self.background_color
            button.hover_color = self.hover_color
            button.pressed_color = self.selected_color
            button.text_color = self.text_color
            button.add_event_handler(EventType.MOUSE_CLICK, lambda event, i=i: self._on_item_click(i))
            self.menu_buttons.append(button)
    
    def _on_item_click(self, index):
        """菜单项点击事件
        
        Args:
            index: 菜单项索引
        """
        self.selected_item = index
        self.is_open = False
        if "callback" in self.items[index]:
            self.items[index]["callback"]()
        self._trigger_event(EventType.VALUE_CHANGED, index=index, item=self.items[index])
    
    def _render_content(self):
        """渲染菜单内容"""
        from OpenGL.GL import glColor4f, glRasterPos2f
        from OpenGL.GLUT import glutBitmapCharacter, GLUT_BITMAP_HELVETICA_12
        
        # 渲染菜单标题
        glColor4f(*self.text_color)
        
        # 计算文本位置（居中）
        title = self.text
        text_width = len(title) * self.font_size * 0.6
        text_height = self.font_size
        text_x = self.x + (self.width - text_width) / 2
        text_y = self.y + (self.height + text_height) / 2
        
        # 设置文本位置
        glRasterPos2f(text_x, text_y)
        
        # 渲染文本
        for char in title:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(char))
        
        # 渲染下拉箭头
        arrow_x = self.x + self.width - 20
        arrow_y = self.y + self.height / 2
        glBegin(GL_TRIANGLES)
        glVertex2f(arrow_x, arrow_y - 3)
        glVertex2f(arrow_x + 6, arrow_y + 3)
        glVertex2f(arrow_x + 12, arrow_y - 3)
        glEnd()
    
    def render(self):
        """渲染菜单"""
        super().render()
        
        # 如果菜单打开，渲染菜单项
        if self.is_open:
            for button in self.menu_buttons:
                button.render()
    
    def update(self, delta_time):
        """更新菜单
        
        Args:
            delta_time: 帧时间（秒）
        """
        super().update(delta_time)
        
        for button in self.menu_buttons:
            button.update(delta_time)
    
    def on_mouse_down(self, button):
        """鼠标按下事件
        
        Args:
            button: 鼠标按钮（0: 左键, 1: 中键, 2: 右键）
        """
        if button == 0:
            # 检查是否点击了菜单标题
            if self.contains_point(*self.mouse_position):
                self.is_open = not self.is_open
            else:
                # 检查是否点击了菜单项
                if self.is_open:
                    clicked = False
                    for button in self.menu_buttons:
                        if button.contains_point(*self.mouse_position):
                            button.on_mouse_down(button)
                            clicked = True
                            break
                    if not clicked:
                        self.is_open = False
    
    def on_mouse_move(self, x, y):
        """鼠标移动事件
        
        Args:
            x: 鼠标X坐标
            y: 鼠标Y坐标
        """
        self.mouse_position = (x, y)
        
        # 更新菜单项的悬停状态
        if self.is_open:
            for button in self.menu_buttons:
                if button.contains_point(x, y):
                    if not button.is_hovered:
                        button.on_mouse_enter()
                else:
                    if button.is_hovered:
                        button.on_mouse_leave()
    
    def on_mouse_up(self, button):
        """鼠标释放事件
        
        Args:
            button: 鼠标按钮（0: 左键, 1: 中键, 2: 右键）
        """
        if button == 0 and self.is_open:
            for button in self.menu_buttons:
                if button.contains_point(*self.mouse_position):
                    button.on_mouse_up(button)
    
    def add_item(self, text, callback=None):
        """添加菜单项
        
        Args:
            text: 菜单项文本
            callback: 点击回调函数
        """
        self.items.append({"text": text, "callback": callback})
        self._create_menu_items()
    
    def remove_item(self, index):
        """移除菜单项
        
        Args:
            index: 菜单项索引
        """
        if 0 <= index < len(self.items):
            self.items.pop(index)
            self._create_menu_items()
    
    def clear_items(self):
        """清除所有菜单项"""
        self.items.clear()
        self.menu_buttons.clear()
    
    def get_selected_item(self):
        """获取选中的菜单项
        
        Returns:
            dict: 选中的菜单项
        """
        if 0 <= self.selected_item < len(self.items):
            return self.items[self.selected_item]
        return None
