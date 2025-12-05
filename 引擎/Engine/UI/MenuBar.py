# -*- coding: utf-8 -*-
"""
菜单栏类，实现顶部菜单栏
"""

from Engine.UI.Controls.Control import Control
from Engine.UI.Controls.Menu import Menu
from Engine.UI.Event import EventType

class MenuBar(Control):
    """菜单栏类，实现顶部菜单栏"""
    
    def __init__(self, x, y, width, height):
        """初始化菜单栏
        
        Args:
            x: X坐标
            y: Y坐标
            width: 宽度
            height: 高度
        """
        super().__init__(x, y, width, height)
        self.name = "MenuBar"
        self.menus = []
        self.current_menu = None
        
        # 设置菜单栏样式
        self.background_color = (0.25, 0.25, 0.25, 0.95)
        self.foreground_color = (1.0, 1.0, 1.0, 1.0)
        self.border_color = (0.5, 0.5, 0.5, 1.0)
        self.border_width = 1
        
        # 创建菜单
        self._create_menus()
    
    def _create_menus(self):
        """创建菜单"""
        # 菜单配置
        menu_configs = [
            {
                "name": "文件",
                "items": [
                    {"text": "新建场景", "shortcut": "Ctrl+N", "callback": self._on_new_scene},
                    {"text": "打开场景", "shortcut": "Ctrl+O", "callback": self._on_open_scene},
                    {"text": "保存场景", "shortcut": "Ctrl+S", "callback": self._on_save_scene},
                    {"text": "保存场景为...", "shortcut": "Ctrl+Shift+S", "callback": self._on_save_scene_as},
                    {"text": "退出", "shortcut": "Ctrl+Q", "callback": self._on_exit}
                ]
            },
            {
                "name": "编辑",
                "items": [
                    {"text": "撤销", "shortcut": "Ctrl+Z", "callback": self._on_undo},
                    {"text": "重做", "shortcut": "Ctrl+Y", "callback": self._on_redo},
                    {"text": "复制", "shortcut": "Ctrl+C", "callback": self._on_copy},
                    {"text": "粘贴", "shortcut": "Ctrl+V", "callback": self._on_paste},
                    {"text": "删除", "shortcut": "Delete", "callback": self._on_delete}
                ]
            },
            {
                "name": "视图",
                "items": [
                    {"text": "透视视图", "shortcut": "P", "callback": self._on_perspective_view},
                    {"text": "正交视图", "shortcut": "O", "callback": self._on_orthographic_view},
                    {"text": "前视图", "shortcut": "Num1", "callback": self._on_front_view},
                    {"text": "顶视图", "shortcut": "Num7", "callback": self._on_top_view},
                    {"text": "右视图", "shortcut": "Num3", "callback": self._on_right_view},
                    {"text": "显示网格", "shortcut": "G", "callback": self._on_toggle_grid},
                    {"text": "显示坐标轴", "shortcut": "X", "callback": self._on_toggle_axes}
                ]
            },
            {
                "name": "对象",
                "items": [
                    {"text": "添加立方体", "shortcut": "Shift+A", "callback": self._on_add_cube},
                    {"text": "添加球体", "shortcut": "Shift+S", "callback": self._on_add_sphere},
                    {"text": "添加圆柱体", "shortcut": "Shift+C", "callback": self._on_add_cylinder},
                    {"text": "添加光源", "shortcut": "Shift+L", "callback": self._on_add_light}
                ]
            },
            {
                "name": "渲染",
                "items": [
                    {"text": "渲染图像", "shortcut": "F12", "callback": self._on_render_image},
                    {"text": "渲染动画", "shortcut": "Ctrl+F12", "callback": self._on_render_animation},
                    {"text": "渲染设置", "callback": self._on_render_settings}
                ]
            }
        ]
        
        # 创建菜单
        current_x = self.x + 10
        menu_height = self.height
        
        for menu_config in menu_configs:
            # 计算菜单宽度
            menu_text = menu_config["name"]
            menu_width = len(menu_text) * 10 + 20  # 简单估算菜单宽度
            
            # 创建菜单
            menu = Menu(
                current_x,
                self.y,
                menu_width,
                menu_height,
                items=menu_config["items"],
                text=menu_text,
            )
            # 添加菜单项点击事件监听
            menu.on_item_click = self._on_menu_item_click
            menu.background_color = self.background_color
            menu.foreground_color = self.foreground_color
            menu.border_color = self.border_color
            menu.border_width = self.border_width
            
            self.add_child(menu)
            self.menus.append(menu)
            
            current_x += menu_width
    
    def _on_new_scene(self, event):
        """新建场景"""
        print("新建场景")
        from Engine import get_engine
        engine = get_engine()
        if engine and engine.scene_manager:
            engine.scene_manager.clear_scene()
    
    def _on_open_scene(self, event):
        """打开场景"""
        print("打开场景")
    
    def _on_save_scene(self, event):
        """保存场景"""
        print("保存场景")
    
    def _on_save_scene_as(self, event):
        """保存场景为..."""
        print("保存场景为...")
    
    def _on_exit(self, event):
        """退出"""
        print("退出")
        from Engine import get_engine
        engine = get_engine()
        if engine:
            engine.shutdown()
        import sys
        sys.exit(0)
    
    def _on_undo(self, event):
        """撤销"""
        print("撤销")
    
    def _on_redo(self, event):
        """重做"""
        print("重做")
    
    def _on_copy(self, event):
        """复制"""
        print("复制")
    
    def _on_paste(self, event):
        """粘贴"""
        print("粘贴")
    
    def _on_delete(self, event):
        """删除"""
        print("删除")
        from Engine import get_engine
        engine = get_engine()
        if engine and engine.scene_manager:
            # 删除选中的节点
            pass
    
    def _on_perspective_view(self, event):
        """透视视图"""
        print("透视视图")
        from Engine import get_engine
        engine = get_engine()
        if engine and engine.main_window and engine.main_window.viewport:
            engine.main_window.viewport.set_view_type("perspective")
            engine.main_window.viewport.handle_resize(engine.main_window.viewport.width, engine.main_window.viewport.height)
    
    def _on_orthographic_view(self, event):
        """正交视图"""
        print("正交视图")
        from Engine import get_engine
        engine = get_engine()
        if engine and engine.main_window and engine.main_window.viewport:
            engine.main_window.viewport.set_view_type("orthographic")
            engine.main_window.viewport.handle_resize(engine.main_window.viewport.width, engine.main_window.viewport.height)
    
    def _on_front_view(self, event):
        """前视图"""
        print("前视图")
    
    def _on_top_view(self, event):
        """顶视图"""
        print("顶视图")
    
    def _on_right_view(self, event):
        """右视图"""
        print("右视图")
    
    def _on_toggle_grid(self, event):
        """显示/隐藏网格"""
        print("显示/隐藏网格")
        from Engine import get_engine
        engine = get_engine()
        if engine and engine.main_window and engine.main_window.viewport:
            engine.main_window.viewport.toggle_grid()
    
    def _on_toggle_axes(self, event):
        """显示/隐藏坐标轴"""
        print("显示/隐藏坐标轴")
        from Engine import get_engine
        engine = get_engine()
        if engine and engine.main_window and engine.main_window.viewport:
            engine.main_window.viewport.toggle_axes()

    def _on_menu_item_click(self, event):  # pyright: ignore[reportUnusedParameter]
        """菜单项点击事件处理"""
        pass



    def _on_add_cube(self, event):
        """添加立方体"""
        print("添加立方体")
        from Engine import get_engine
        engine = get_engine()
        if engine and hasattr(engine, 'create_cube'):
            engine.create_cube()
        elif engine and hasattr(engine, '_create_cube'):
            # 兼容旧版本引擎
            engine._create_cube()
    
    def _on_add_sphere(self, event):
        """添加球体"""
        print("添加球体")
        from Engine import get_engine
        engine = get_engine()
        if engine:
            engine._create_sphere()
    
    def _on_add_cylinder(self, event):
        """添加圆柱体"""
        print("添加圆柱体")
        from Engine import get_engine
        engine = get_engine()
        if engine:
            engine._create_cylinder()
    
    def _on_add_light(self, event):
        """添加光源"""
        print("添加光源")
    
    def _on_render_image(self, event):
        """渲染图像"""
        print("渲染图像")
    
    def _on_render_animation(self, event):
        """渲染动画"""
        print("渲染动画")
    
    def _on_render_settings(self, event):
        """渲染设置"""
        print("渲染设置")
    
    def handle_resize(self, width, height):
        """处理大小变化
        
        Args:
            width: 新宽度
            height: 新高度
        """
        self.set_size(width, height)
        # 重新创建菜单，因为宽度变化可能影响菜单布局
        for menu in self.menus:
            self.remove_child(menu)
        self.menus.clear()
        self._create_menus()
