# -*- coding: utf-8 -*-
"""
工具栏类，实现左侧垂直工具栏
"""

from Engine.UI.Controls.Control import Control
from Engine.UI.Controls.Button import Button
from Engine.UI.Event import EventType

class ToolBar(Control):
    """工具栏类，实现左侧垂直工具栏"""
    
    def __init__(self, x, y, width, height):
        """初始化工具栏
        
        Args:
            x: X坐标
            y: Y坐标
            width: 宽度
            height: 高度
        """
        super().__init__(x, y, width, height)
        self.name = "ToolBar"
        self.tools = []
        self.current_tool = "select"
        # 设置工具栏背景色为深灰色，与主窗口背景色区分
        self.background_color = (0.2, 0.2, 0.2, 1.0)
        # 确保工具栏可见
        self.is_visible = True
        
        # 创建工具按钮
        self._create_tools()
    
    def _create_tools(self):
        """创建工具按钮"""
        # 工具配置
        tool_configs = [
            {"name": "select", "text": "选择", "tooltip": "选择工具 (Shift+Space)", "icon": "□", "callback": self._on_tool_select},
            {"name": "move", "text": "移动", "tooltip": "移动工具 (G)", "icon": "↔", "callback": self._on_tool_move},
            {"name": "rotate", "text": "旋转", "tooltip": "旋转工具 (R)", "icon": "⟳", "callback": self._on_tool_rotate},
            {"name": "scale", "text": "缩放", "tooltip": "缩放工具 (S)", "icon": "⇱", "callback": self._on_tool_scale}
        ]
        
        # 创建工具按钮
        button_width = self.width - 10
        button_height = 30
        button_spacing = 5
        start_y = 5
        
        for i, tool_config in enumerate(tool_configs):
            button = Button(
                self.x + 5,
                self.y + start_y + i * (button_height + button_spacing),
                button_width,
                button_height,
                f"{tool_config['icon']} {tool_config['text']}"
            )
            button.tag = tool_config["name"]
            button.tooltip = tool_config["tooltip"]
            button.add_event_handler(EventType.MOUSE_CLICK, tool_config["callback"])
            
            # 改进按钮样式
            button.background_color = (0.25, 0.25, 0.25, 0.9)
            button.border_color = (0.4, 0.4, 0.4, 1.0)
            button.foreground_color = (1.0, 1.0, 1.0, 1.0)
            button.border_width = 1
            
            self.add_child(button)
            self.tools.append(button)
            
            # 设置选中状态
            if tool_config["name"] == self.current_tool:
                button.is_pressed = True
                button.background_color = (0.4, 0.4, 0.4, 0.9)
    
    def _on_tool_select(self, event):
        """选择工具点击事件"""
        self._set_current_tool("select")
        print("切换到选择工具")
    
    def _on_tool_move(self, event):
        """移动工具点击事件"""
        self._set_current_tool("move")
        print("切换到移动工具")
        # 通知变换操纵器切换到移动模式
        from Engine import get_engine
        engine = get_engine()
        if engine and hasattr(engine, 'transform_manipulator'):
            engine.transform_manipulator.set_transform_mode("translate")
    
    def _on_tool_rotate(self, event):
        """旋转工具点击事件"""
        self._set_current_tool("rotate")
        print("切换到旋转工具")
        # 通知变换操纵器切换到旋转模式
        from Engine import get_engine
        engine = get_engine()
        if engine and hasattr(engine, 'transform_manipulator'):
            engine.transform_manipulator.set_transform_mode("rotate")
    
    def _on_tool_scale(self, event):
        """缩放工具点击事件"""
        self._set_current_tool("scale")
        print("切换到缩放工具")
        # 通知变换操纵器切换到缩放模式
        from Engine import get_engine
        engine = get_engine()
        if engine and hasattr(engine, 'transform_manipulator'):
            engine.transform_manipulator.set_transform_mode("scale")
    
    def _set_current_tool(self, tool_name):
        """设置当前工具
        
        Args:
            tool_name: 工具名称
        """
        self.current_tool = tool_name
        
        # 更新按钮状态
        for button in self.tools:
            button.is_pressed = (button.tag == tool_name)
    
    def get_current_tool(self):
        """获取当前工具
        
        Returns:
            str: 当前工具名称
        """
        return self.current_tool