# -*- coding: utf-8 -*-
"""
主窗口类，负责管理整个界面布局
"""

from Engine.UI.Controls.Control import Control

class MainWindow(Control):
    """主窗口类，负责管理整个界面布局"""
    
    def __init__(self, platform):
        """初始化主窗口
        
        Args:
            platform: 平台对象
        """
        super().__init__(0, 0, platform.width, platform.height)
        self.name = "MainWindow"
        self.platform = platform
        self.menu_bar = None
        self.tool_bar = None
        self.property_panel = None
        self.status_bar = None
        self.viewport = None
        
        # 确保主窗口可见
        self.is_visible = True
        # 设置主窗口背景色为浅灰色，以便区分子组件
        self.background_color = (0.3, 0.3, 0.3, 1.0)
        
        # 布局参数
        self.menu_height = 25
        self.tool_width = 40
        self.property_width = 300
        self.status_height = 20
        
        # 创建界面组件
        self._create_components()
    
    def _create_components(self):
        """创建界面组件"""
        # 创建菜单栏
        from Engine.UI.MenuBar import MenuBar
        self.menu_bar = MenuBar(0, 0, self.width, self.menu_height)
        self.add_child(self.menu_bar)
        
        # 创建工具栏
        from Engine.UI.ToolBar import ToolBar
        self.tool_bar = ToolBar(0, self.menu_height, self.tool_width, self.height - self.menu_height - self.status_height)
        self.add_child(self.tool_bar)
        
        # 创建属性面板
        from Engine.UI.Controls.PropertyPanel import PropertyPanel
        self.property_panel = PropertyPanel(self.width - self.property_width, self.menu_height, self.property_width, self.height - self.menu_height - self.status_height)
        self.add_child(self.property_panel)
        
        # 创建状态栏
        from Engine.UI.StatusBar import StatusBar
        self.status_bar = StatusBar(0, self.height - self.status_height, self.width, self.status_height)
        self.add_child(self.status_bar)
        
        # 创建视图端口
        from Engine.UI.Viewport import Viewport
        self.viewport = Viewport(self.tool_width, self.menu_height, 
                               self.width - self.tool_width - self.property_width, 
                               self.height - self.menu_height - self.status_height)
        self.add_child(self.viewport)
    
    def set_selected_object(self, obj):
        """设置选中的对象
        
        Args:
            obj: 选中的对象
        """
        if self.property_panel:
            self.property_panel.set_selected_object(obj)
        if self.viewport:
            self.viewport.set_selected_object(obj)
    
    def update(self, delta_time):
        """更新主窗口
        
        Args:
            delta_time: 帧时间（秒）
        """
        super().update(delta_time)
        
        # 更新状态栏
        if self.status_bar:
            self.status_bar.update_status()
    
    def handle_resize(self, width, height):
        """处理窗口大小变化
        
        Args:
            width: 新宽度
            height: 新高度
        """
        self.set_size(width, height)
        
        # 更新菜单栏
        if self.menu_bar:
            self.menu_bar.set_size(width, self.menu_height)
        
        # 更新工具栏
        if self.tool_bar:
            self.tool_bar.set_size(self.tool_width, height - self.menu_height - self.status_height)
        
        # 更新属性面板
        if self.property_panel:
            self.property_panel.set_position(width - self.property_width, self.menu_height)
            self.property_panel.set_size(self.property_width, height - self.menu_height - self.status_height)
        
        # 更新状态栏
        if self.status_bar:
            self.status_bar.set_size(width, self.status_height)
            self.status_bar.set_position(0, height - self.status_height)
        
        # 更新视图端口
        if self.viewport:
            self.viewport.set_size(width - self.tool_width - self.property_width, 
                                 height - self.menu_height - self.status_height)
    
    def get_viewport(self):
        """获取视图端口
        
        Returns:
            Viewport: 视图端口对象
        """
        return self.viewport
    
    def get_tool_bar(self):
        """获取工具栏
        
        Returns:
            ToolBar: 工具栏对象
        """
        return self.tool_bar
    
    def get_property_panel(self):
        """获取属性面板
        
        Returns:
            PropertyPanel: 属性面板对象
        """
        return self.property_panel