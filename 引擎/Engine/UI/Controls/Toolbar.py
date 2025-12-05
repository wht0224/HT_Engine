# -*- coding: utf-8 -*-
"""
工具栏控件类，提供常用操作快捷方式
"""

from Engine.UI.Controls.Control import Control
from Engine.UI.Controls.Button import Button
from Engine.UI.Event import EventType

class Toolbar(Control):
    """工具栏控件类"""
    
    def __init__(self, x, y, width, height):
        """初始化工具栏
        
        Args:
            x: 工具栏X坐标
            y: 工具栏Y坐标
            width: 工具栏宽度
            height: 工具栏高度
        """
        super().__init__(x, y, width, height)
        self.name = "Toolbar"
        self.buttons = []
        self.button_width = 80
        self.button_height = 30
        self.button_spacing = 5
        self.current_mode = "translate"  # 默认模式：平移
        
        # 创建工具栏按钮
        self._create_tool_buttons()
    
    def _create_tool_buttons(self):
        """创建工具栏按钮"""
        # 变换模式按钮
        translate_button = Button(self.x + self.button_spacing, self.y + (self.height - self.button_height) / 2, 
                                self.button_width, self.button_height, "平移")
        translate_button.add_event_handler(EventType.MOUSE_CLICK, lambda e: self._on_tool_button_click("translate"))
        self.add_child(translate_button)
        self.buttons.append(translate_button)
        
        rotate_button = Button(self.x + self.button_spacing * 2 + self.button_width, self.y + (self.height - self.button_height) / 2, 
                              self.button_width, self.button_height, "旋转")
        rotate_button.add_event_handler(EventType.MOUSE_CLICK, lambda e: self._on_tool_button_click("rotate"))
        self.add_child(rotate_button)
        self.buttons.append(rotate_button)
        
        scale_button = Button(self.x + self.button_spacing * 3 + self.button_width * 2, self.y + (self.height - self.button_height) / 2, 
                            self.button_width, self.button_height, "缩放")
        scale_button.add_event_handler(EventType.MOUSE_CLICK, lambda e: self._on_tool_button_click("scale"))
        self.add_child(scale_button)
        self.buttons.append(scale_button)
        
        # 相机模式按钮
        free_camera_button = Button(self.x + self.button_spacing * 4 + self.button_width * 3, self.y + (self.height - self.button_height) / 2, 
                                  self.button_width, self.button_height, "自由相机")
        free_camera_button.add_event_handler(EventType.MOUSE_CLICK, lambda e: self._on_camera_button_click("free"))
        self.add_child(free_camera_button)
        self.buttons.append(free_camera_button)
        
        orbit_camera_button = Button(self.x + self.button_spacing * 5 + self.button_width * 4, self.y + (self.height - self.button_height) / 2, 
                                   self.button_width, self.button_height, "轨道相机")
        orbit_camera_button.add_event_handler(EventType.MOUSE_CLICK, lambda e: self._on_camera_button_click("orbit"))
        self.add_child(orbit_camera_button)
        self.buttons.append(orbit_camera_button)
        
        fps_camera_button = Button(self.x + self.button_spacing * 6 + self.button_width * 5, self.y + (self.height - self.button_height) / 2, 
                                 self.button_width, self.button_height, "第一人称")
        fps_camera_button.add_event_handler(EventType.MOUSE_CLICK, lambda e: self._on_camera_button_click("fps"))
        self.add_child(fps_camera_button)
        self.buttons.append(fps_camera_button)
        
        # 视图模式按钮
        perspective_button = Button(self.x + self.button_spacing * 7 + self.button_width * 6, self.y + (self.height - self.button_height) / 2, 
                                  self.button_width, self.button_height, "透视")
        perspective_button.add_event_handler(EventType.MOUSE_CLICK, lambda e: self._on_view_button_click("perspective"))
        self.add_child(perspective_button)
        self.buttons.append(perspective_button)
        
        orthographic_button = Button(self.x + self.button_spacing * 8 + self.button_width * 7, self.y + (self.height - self.button_height) / 2, 
                                   self.button_width, self.button_height, "正交")
        orthographic_button.add_event_handler(EventType.MOUSE_CLICK, lambda e: self._on_view_button_click("orthographic"))
        self.add_child(orthographic_button)
        self.buttons.append(orthographic_button)
        
        # 创建立方体按钮
        cube_button = Button(self.x + self.button_spacing * 9 + self.button_width * 8, self.y + (self.height - self.button_height) / 2, 
                           self.button_width, self.button_height, "创建立方体")
        cube_button.add_event_handler(EventType.MOUSE_CLICK, lambda e: self._on_create_button_click("cube"))
        self.add_child(cube_button)
        self.buttons.append(cube_button)
        
        # 创建球体按钮
        sphere_button = Button(self.x + self.button_spacing * 10 + self.button_width * 9, self.y + (self.height - self.button_height) / 2, 
                             self.button_width, self.button_height, "创建球体")
        sphere_button.add_event_handler(EventType.MOUSE_CLICK, lambda e: self._on_create_button_click("sphere"))
        self.add_child(sphere_button)
        self.buttons.append(sphere_button)
    
    def _on_tool_button_click(self, mode):
        """工具按钮点击事件处理
        
        Args:
            mode: 工具模式
        """
        self.current_mode = mode
        print(f"切换到{mode}模式")
        
        # 通知引擎切换变换模式
        from Engine import get_engine
        engine = get_engine()
        if engine and hasattr(engine, 'transform_manipulator'):
            engine.transform_manipulator.set_transform_mode(mode)
    
    def _on_camera_button_click(self, mode):
        """相机按钮点击事件处理
        
        Args:
            mode: 相机模式
        """
        print(f"切换到{mode}相机模式")
        
        # 通知引擎切换相机模式
        from Engine import get_engine
        engine = get_engine()
        if engine and hasattr(engine, 'camera_controller'):
            engine.camera_controller.set_mode(mode)
    
    def _on_view_button_click(self, mode):
        """视图按钮点击事件处理
        
        Args:
            mode: 视图模式
        """
        print(f"切换到{mode}视图")
        
        # 通知引擎切换视图模式
        from Engine import get_engine
        engine = get_engine()
        if engine and engine.scene_manager and engine.scene_manager.active_camera:
            camera = engine.scene_manager.active_camera
            if mode == "perspective":
                camera.set_perspective(60, 16/9, 0.1, 1000.0)
            elif mode == "orthographic":
                camera.set_orthographic(-5, 5, -5, 5, 0.1, 1000.0)
    
    def _on_create_button_click(self, shape):
        """创建按钮点击事件处理
        
        Args:
            shape: 要创建的形状
        """
        print(f"创建{shape}")
        
        # 通知引擎创建形状
        from Engine import get_engine
        engine = get_engine()
        if engine:
            if shape == "cube":
                engine._create_cube()
            elif shape == "sphere":
                engine._create_sphere()