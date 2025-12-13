"""
游戏世界的控制台指挥官！
负责管理所有UI按钮、滑块、菜单等交互元素
让玩家能够轻松操控游戏世界
"""

from Engine.UI.Controls.Button import Button
from Engine.UI.Controls.Label import Label
from Engine.UI.Controls.Slider import Slider
from Engine.UI.Controls.Menu import Menu
from Engine.UI.Controls.TextInput import TextInput
from Engine.UI.Event import Event
from Engine.UI.Event import EventType

class UIManager:
    """游戏世界的交互指挥官！
    负责管理所有UI按钮、滑块、菜单等交互元素
    让玩家能够轻松操控游戏世界
    """
    
    def __init__(self, platform):
        """初始化UI管理器
        
        Args:
            platform: 平台对象，用于获取窗口信息和事件
        """
        self.platform = platform
        self.controls = []
        self.mouse_position = (0, 0)
        self.mouse_buttons = [False, False, False]  # 左键、中键、右键
        self.key_states = {}
        self.active_control = None
        self.hovered_control = None
        # 确保UI管理器激活，以便渲染UI
        self.is_ui_active = True
        
        # 注册事件处理器
        self._register_event_handlers()
    
    def _register_event_handlers(self):
        """注册事件处理器"""
        # 这里将在Platform类中注册具体的事件处理函数
        pass
    
    def add_control(self, control):
        """添加UI控件
        
        Args:
            control: UI控件对象
        """
        self.controls.append(control)
    
    def remove_control(self, control):
        """移除UI控件
        
        Args:
            control: UI控件对象
        """
        if control in self.controls:
            self.controls.remove(control)
    
    def update(self, delta_time):
        """更新UI
        
        Args:
            delta_time: 帧时间（秒）
        """
        for control in self.controls:
            control.update(delta_time)
    
    def render(self):
        """渲染UI"""
        if not self.is_ui_active:
            return
        
        for control in self.controls:
            control.render()
    
    def handle_mouse_move(self, x, y):
        """处理鼠标移动事件
        
        Args:
            x: 鼠标X坐标
            y: 鼠标Y坐标
            
        Returns:
            bool: 是否处理了事件
        """
        self.mouse_position = (x, y)
        
        # 更新悬停状态
        new_hovered = None
        for control in reversed(self.controls):
            if control.is_visible and control.contains_point(x, y):
                new_hovered = control
                break
        
        if self.hovered_control != new_hovered:
            if self.hovered_control:
                self.hovered_control.on_mouse_leave()
            if new_hovered:
                new_hovered.on_mouse_enter()
            self.hovered_control = new_hovered
        
        # 如果鼠标悬停在控件上，返回True表示事件已处理
        return self.hovered_control is not None
    
    def handle_mouse_down(self, button):
        """处理鼠标按下事件
        
        Args:
            button: 鼠标按钮（0: 左键, 1: 中键, 2: 右键）
            
        Returns:
            bool: 是否处理了事件
        """
        self.mouse_buttons[button] = True
        
        # 检查是否点击了某个控件
        for control in reversed(self.controls):
            if control.is_visible and control.contains_point(*self.mouse_position):
                self.active_control = control
                control.on_mouse_down(button)
                return True
        
        return False
    
    def handle_mouse_up(self, button):
        """处理鼠标释放事件
        
        Args:
            button: 鼠标按钮（0: 左键, 1: 中键, 2: 右键）
            
        Returns:
            bool: 是否处理了事件
        """
        self.mouse_buttons[button] = False
        
        if self.active_control:
            self.active_control.on_mouse_up(button)
            self.active_control = None
            return True
        
        return False
    
    def handle_key_down(self, key):
        """处理键盘按下事件
        
        Args:
            key: 按键码
            
        Returns:
            bool: 是否处理了事件
        """
        self.key_states[key] = True
        
        if self.active_control:
            self.active_control.on_key_down(key)
            return True
        
        # 检查是否有控件处理键盘事件
        for control in self.controls:
            if control.is_visible and control.is_enabled:
                control.on_key_down(key)
                return True
        
        return False
    
    def handle_key_up(self, key):
        """处理键盘释放事件
        
        Args:
            key: 按键码
            
        Returns:
            bool: 是否处理了事件
        """
        if key in self.key_states:
            self.key_states[key] = False
        
        if self.active_control:
            self.active_control.on_key_up(key)
            return True
        
        # 检查是否有控件处理键盘释放事件
        for control in self.controls:
            if control.is_visible and control.is_enabled:
                control.on_key_up(key)
                return True
        
        return False
    
    def set_ui_active(self, active):
        """设置UI是否激活
        
        Args:
            active: 是否激活UI
        """
        self.is_ui_active = active
    
    def get_ui_active(self):
        """获取UI是否激活
        
        Returns:
            bool: UI是否激活
        """
        return self.is_ui_active
    
    def clear(self):
        """清除所有UI控件"""
        self.controls.clear()
        self.active_control = None
        self.hovered_control = None
    
    def is_mouse_over_controls(self):
        """检查鼠标是否悬停在任何UI控件上
        
        Returns:
            bool: 如果鼠标悬停在UI控件上，返回True，否则返回False
        """
        return self.hovered_control is not None
    
    def handle_mouse_wheel(self, direction):
        """处理鼠标滚轮事件
        
        Args:
            direction: 滚轮方向（1: 向上, -1: 向下）
        
        Returns:
            bool: 如果UI处理了滚轮事件，返回True，否则返回False
        """
        if self.hovered_control:
            self.hovered_control.on_mouse_wheel(direction)
            return True
        return False
