# -*- coding: utf-8 -*-
"""
Blender风格的主窗口
Main window with Blender-style layout
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import sys
import os

# 添加引擎路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from Engine.Logger import get_logger
from Engine.UI.TkUI.theme.dark_theme import DarkTheme
from Engine.UI.TkUI.opengl_viewport import OpenGLViewport


class TkMainWindow:
    """
    主窗口类 - Blender/UE5风格的3D编辑器界面
    Main window class with Blender/UE5 style 3D editor interface
    """

    def __init__(self, engine):
        """
        初始化主窗口
        Initialize main window

        Args:
            engine: 引擎实例
        """
        self.logger = get_logger("TkMainWindow")
        self.engine = engine
        self.running = False

        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("低端GPU优化渲染引擎 - Low-End GPU Rendering Engine")
        self.root.geometry("1920x1080")
        self.root.minsize(1280, 720)

        # 应用暗色主题
        DarkTheme.apply(self.root)

        # 组件引用
        self.viewport = None
        self.scene_tree = None
        self.property_panel = None
        self.status_labels = {}

        # 创建UI组件
        self._create_menu_bar()
        self._create_toolbar()
        self._create_main_layout()
        self._create_status_bar()

        # 绑定窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # 键盘快捷键
        self._bind_shortcuts()

        self.logger.info("Tkinter主窗口创建完成")

    def _create_menu_bar(self):
        """创建菜单栏"""
        self.menu_bar = tk.Menu(self.root)
        self.root.config(menu=self.menu_bar)

        # 配置菜单样式
        DarkTheme.configure_widget(self.menu_bar, 'menu')

        # File菜单
        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        DarkTheme.configure_widget(file_menu, 'menu')
        self.menu_bar.add_cascade(label="文件 File", menu=file_menu)
        file_menu.add_command(label="新建场景 New Scene", command=self._on_new_scene, accelerator="Ctrl+N")
        file_menu.add_command(label="打开场景 Open Scene", command=self._on_open_scene, accelerator="Ctrl+O")
        file_menu.add_command(label="保存场景 Save Scene", command=self._on_save_scene, accelerator="Ctrl+S")
        file_menu.add_command(label="另存为 Save As...", command=self._on_save_scene_as, accelerator="Ctrl+Shift+S")
        file_menu.add_separator()
        file_menu.add_command(label="导入模型 Import Model", command=self._on_import_model, accelerator="Ctrl+I")
        file_menu.add_command(label="导出模型 Export Model", command=self._on_export_model, accelerator="Ctrl+E")
        file_menu.add_separator()
        file_menu.add_command(label="退出 Quit", command=self._on_close, accelerator="Ctrl+Q")

        # Edit菜单
        edit_menu = tk.Menu(self.menu_bar, tearoff=0)
        DarkTheme.configure_widget(edit_menu, 'menu')
        self.menu_bar.add_cascade(label="编辑 Edit", menu=edit_menu)
        edit_menu.add_command(label="撤销 Undo", command=self._on_undo, accelerator="Ctrl+Z")
        edit_menu.add_command(label="重做 Redo", command=self._on_redo, accelerator="Ctrl+Y")
        edit_menu.add_separator()
        edit_menu.add_command(label="复制 Copy", command=self._on_copy, accelerator="Ctrl+C")
        edit_menu.add_command(label="粘贴 Paste", command=self._on_paste, accelerator="Ctrl+V")
        edit_menu.add_command(label="删除 Delete", command=self._on_delete, accelerator="Delete")
        edit_menu.add_separator()
        edit_menu.add_command(label="全选 Select All", command=self._on_select_all, accelerator="Ctrl+A")

        # View菜单
        view_menu = tk.Menu(self.menu_bar, tearoff=0)
        DarkTheme.configure_widget(view_menu, 'menu')
        self.menu_bar.add_cascade(label="视图 View", menu=view_menu)
        view_menu.add_command(label="透视视图 Perspective", command=self._on_perspective_view, accelerator="Numpad 0")
        view_menu.add_command(label="正交视图 Orthographic", command=self._on_orthographic_view, accelerator="Numpad 5")
        view_menu.add_separator()
        view_menu.add_command(label="顶视图 Top View", command=self._on_top_view, accelerator="Numpad 7")
        view_menu.add_command(label="前视图 Front View", command=self._on_front_view, accelerator="Numpad 1")
        view_menu.add_command(label="右视图 Right View", command=self._on_right_view, accelerator="Numpad 3")
        view_menu.add_separator()
        view_menu.add_checkbutton(label="显示网格 Show Grid", command=self._on_toggle_grid)
        view_menu.add_checkbutton(label="显示坐标轴 Show Axes", command=self._on_toggle_axes)
        view_menu.add_checkbutton(label="线框模式 Wireframe", command=self._on_toggle_wireframe)

        # Object菜单
        object_menu = tk.Menu(self.menu_bar, tearoff=0)
        DarkTheme.configure_widget(object_menu, 'menu')
        self.menu_bar.add_cascade(label="对象 Object", menu=object_menu)

        # 添加子菜单 - 创建几何体
        add_menu = tk.Menu(object_menu, tearoff=0)
        DarkTheme.configure_widget(add_menu, 'menu')
        object_menu.add_cascade(label="添加 Add", menu=add_menu)
        add_menu.add_command(label="立方体 Cube", command=self._on_create_cube, accelerator="Shift+A, C")
        add_menu.add_command(label="球体 Sphere", command=self._on_create_sphere, accelerator="Shift+A, S")
        add_menu.add_command(label="圆柱体 Cylinder", command=self._on_create_cylinder)
        add_menu.add_command(label="平面 Plane", command=self._on_create_plane, accelerator="Shift+A, P")
        add_menu.add_command(label="圆锥体 Cone", command=self._on_create_cone)

        object_menu.add_separator()
        object_menu.add_command(label="复制对象 Duplicate", command=self._on_duplicate, accelerator="Shift+D")
        object_menu.add_command(label="删除对象 Delete", command=self._on_delete, accelerator="X")

        # Render菜单
        render_menu = tk.Menu(self.menu_bar, tearoff=0)
        DarkTheme.configure_widget(render_menu, 'menu')
        self.menu_bar.add_cascade(label="渲染 Render", menu=render_menu)
        render_menu.add_command(label="渲染当前视图 Render View", command=self._on_render_current_view, accelerator="F12")
        render_menu.add_command(label="渲染设置 Render Settings", command=self._on_render_settings)
        render_menu.add_separator()
        render_menu.add_checkbutton(label="实时渲染 Realtime Rendering")

        # Help菜单
        help_menu = tk.Menu(self.menu_bar, tearoff=0)
        DarkTheme.configure_widget(help_menu, 'menu')
        self.menu_bar.add_cascade(label="帮助 Help", menu=help_menu)
        help_menu.add_command(label="快捷键 Shortcuts", command=self._on_show_shortcuts, accelerator="F1")
        help_menu.add_command(label="关于 About", command=self._on_about)

    def _create_toolbar(self):
        """创建工具栏 - Blender/UE5风格"""
        self.toolbar = ttk.Frame(self.root, style='Dark.TFrame', padding=(8, 5))
        self.toolbar.pack(side=tk.TOP, fill=tk.X)

        # 当前工具模式指示器 - UE5风格
        mode_indicator = ttk.Frame(self.toolbar, style='Dark.TFrame')
        mode_indicator.pack(side=tk.LEFT, padx=(5, 15))

        ttk.Label(mode_indicator, text="当前模式:", style='Panel.TLabel',
                 font=('Segoe UI', 9, 'bold')).pack(side=tk.LEFT, padx=(0, 5))
        self.current_mode_label = ttk.Label(mode_indicator, text="选择",
                                           foreground=DarkTheme.ACCENT_PRIMARY,
                                           background=DarkTheme.BG_DARK,
                                           font=('Segoe UI', 9, 'bold'))
        self.current_mode_label.pack(side=tk.LEFT)

        # 分隔线
        ttk.Separator(self.toolbar, orient='vertical', style='Light.TSeparator').pack(
            side=tk.LEFT, fill=tk.Y, padx=10)

        # 变换模式按钮组 - 紧凑布局
        transform_frame = ttk.Frame(self.toolbar, style='Dark.TFrame')
        transform_frame.pack(side=tk.LEFT, padx=5)

        ttk.Label(transform_frame, text="变换:", style='Panel.TLabel',
                 font=('Segoe UI', 8)).pack(side=tk.LEFT, padx=(0, 5))

        # 变换按钮 - 更紧凑的设计
        self.transform_buttons = {}
        transform_tools = [
            ("移动", "G", "translate"),
            ("旋转", "R", "rotate"),
            ("缩放", "S", "scale")
        ]

        for label, key, mode in transform_tools:
            btn = ttk.Button(transform_frame, text=f"{label} [{key}]", width=10,
                           style='Toolbar.TButton',
                           command=lambda m=mode, l=label: self._set_transform_mode(m, l))
            btn.pack(side=tk.LEFT, padx=2)
            self.transform_buttons[mode] = btn
            # 添加工具提示
            self._create_tooltip(btn, f"快捷键: {key}")

        # 分隔线
        ttk.Separator(self.toolbar, orient='vertical', style='Light.TSeparator').pack(
            side=tk.LEFT, fill=tk.Y, padx=10)

        # 相机控制按钮组
        camera_frame = ttk.Frame(self.toolbar, style='Dark.TFrame')
        camera_frame.pack(side=tk.LEFT, padx=5)

        ttk.Label(camera_frame, text="相机:", style='Panel.TLabel',
                 font=('Segoe UI', 8)).pack(side=tk.LEFT, padx=(0, 5))

        camera_tools = [
            ("轨道", "鼠标中键", self._on_camera_orbit),
            ("平移", "Shift+中键", self._on_camera_pan),
            ("缩放", "滚轮", self._on_camera_zoom)
        ]

        for label, hint, cmd in camera_tools:
            btn = ttk.Button(camera_frame, text=label, width=8,
                           style='Toolbar.TButton', command=cmd)
            btn.pack(side=tk.LEFT, padx=2)
            self._create_tooltip(btn, hint)

        # 分隔线
        ttk.Separator(self.toolbar, orient='vertical', style='Light.TSeparator').pack(
            side=tk.LEFT, fill=tk.Y, padx=10)

        # 着色模式按钮组 - 使用切换按钮
        shading_frame = ttk.Frame(self.toolbar, style='Dark.TFrame')
        shading_frame.pack(side=tk.LEFT, padx=5)

        ttk.Label(shading_frame, text="着色:", style='Panel.TLabel',
                 font=('Segoe UI', 8)).pack(side=tk.LEFT, padx=(0, 5))

        self.shading_buttons = {}
        shading_modes = [
            ("线框", "wireframe", "1"),
            ("实体", "solid", "2"),
            ("材质", "material", "3"),
            ("渲染", "rendered", "4")
        ]

        for label, mode, key in shading_modes:
            btn = ttk.Button(shading_frame, text=f"{label} [{key}]", width=9,
                           style='Toolbar.TButton',
                           command=lambda m=mode, l=label: self._set_shading_mode(m, l))
            btn.pack(side=tk.LEFT, padx=2)
            self.shading_buttons[mode] = btn
            self._create_tooltip(btn, f"快捷键: {key}")

        # 右侧快捷操作 - UE5风格
        right_tools = ttk.Frame(self.toolbar, style='Dark.TFrame')
        right_tools.pack(side=tk.RIGHT, padx=5)

        # 渲染按钮 - 使用强调色
        render_btn = ttk.Button(right_tools, text="渲染 [F12]", width=12,
                               style='Accent.TButton',
                               command=self._on_render_current_view)
        render_btn.pack(side=tk.LEFT, padx=2)
        self._create_tooltip(render_btn, "渲染当前视图 (F12)")

        # 截图按钮
        screenshot_btn = ttk.Button(right_tools, text="截图", width=8,
                                   style='Toolbar.TButton',
                                   command=self._on_screenshot_viewport)
        screenshot_btn.pack(side=tk.LEFT, padx=2)
        self._create_tooltip(screenshot_btn, "保存当前视口截图")

    def _create_main_layout(self):
        """创建主布局（Blender风格）"""
        # 主容器 - 使用PanedWindow实现可调整大小的分割
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True)

        # 左侧面板（场景树 + 工具）
        self.left_panel = self._create_left_panel()
        self.main_paned.add(self.left_panel, weight=1)

        # 中央3D视口
        self.viewport_frame = self._create_viewport_panel()
        self.main_paned.add(self.viewport_frame, weight=4)

        # 右侧面板（属性）
        self.right_panel = self._create_right_panel()
        self.main_paned.add(self.right_panel, weight=1)

    def _create_left_panel(self):
        """创建左侧面板（场景树）"""
        panel = ttk.Frame(self.root, style='Panel.TFrame', width=300)

        # 标签页
        notebook = ttk.Notebook(panel)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 场景树标签页
        scene_frame = ttk.Frame(notebook)
        notebook.add(scene_frame, text="场景树 Scene")

        # 场景树
        tree_scroll = ttk.Scrollbar(scene_frame)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.scene_tree = ttk.Treeview(scene_frame, yscrollcommand=tree_scroll.set)
        self.scene_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        tree_scroll.config(command=self.scene_tree.yview)

        # 添加示例节点
        self.scene_tree.insert('', 'end', 'scene', text='场景 Scene', open=True)
        self.scene_tree.insert('scene', 'end', text='相机 Camera')
        self.scene_tree.insert('scene', 'end', text='光源 Light')

        # 绑定选择事件
        self.scene_tree.bind('<<TreeviewSelect>>', self._on_scene_tree_select)

        # 对象库标签页
        library_frame = ttk.Frame(notebook)
        notebook.add(library_frame, text="对象库 Library")

        ttk.Label(library_frame, text="基础几何体", style='Title.TLabel').pack(pady=10)

        # 添加几何体按钮
        for name, cmd in [("立方体 Cube", self._on_create_cube),
                         ("球体 Sphere", self._on_create_sphere),
                         ("圆柱体 Cylinder", self._on_create_cylinder),
                         ("平面 Plane", self._on_create_plane),
                         ("圆锥体 Cone", self._on_create_cone)]:
            ttk.Button(library_frame, text=name, command=cmd).pack(fill=tk.X, padx=10, pady=2)

        return panel

    def _create_viewport_panel(self):
        """创建3D视口面板（使用OpenGLViewport）"""
        panel = ttk.Frame(self.root, style='Viewport.TFrame')

        # 视口工具栏
        viewport_toolbar = ttk.Frame(panel, style='Dark.TFrame', padding=(5, 2))
        viewport_toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(viewport_toolbar, text="3D视口 3D Viewport", style='Title.TLabel').pack(side=tk.LEFT, padx=10)

        ttk.Button(viewport_toolbar, text="透视 Perspective", width=12, style='Toolbar.TButton',
                  command=self._on_perspective_view).pack(side=tk.RIGHT, padx=2)
        ttk.Button(viewport_toolbar, text="正交 Orthographic", width=12, style='Toolbar.TButton',
                  command=self._on_orthographic_view).pack(side=tk.RIGHT, padx=2)
        ttk.Button(viewport_toolbar, text="重置相机 Reset Camera", width=12, style='Toolbar.TButton',
                  command=self._on_reset_camera).pack(side=tk.RIGHT, padx=2)

        # OpenGL 3D视口
        try:
            self.viewport = OpenGLViewport(panel, self.engine)
            self.viewport.pack(fill=tk.BOTH, expand=True)
            self.logger.info("OpenGL视口创建成功")
        except Exception as e:
            self.logger.error(f"OpenGL视口创建失败: {e}")
            # 降级方案：使用占位符
            self.viewport = tk.Canvas(panel, background=DarkTheme.VIEWPORT_BG, highlightthickness=0)
            self.viewport.pack(fill=tk.BOTH, expand=True)
            self.viewport.create_text(
                500, 300,
                text=f"OpenGL视口初始化失败\n{str(e)}",
                fill=DarkTheme.TEXT_SECONDARY,
                font=('Segoe UI', 12),
                justify=tk.CENTER
            )

        return panel

    def _create_right_panel(self):
        """创建右侧面板（属性）"""
        panel = ttk.Frame(self.root, style='Panel.TFrame', width=350)

        # 标签页
        notebook = ttk.Notebook(panel)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 属性标签页
        properties_frame = ttk.Frame(notebook)
        notebook.add(properties_frame, text="属性 Properties")

        # 属性内容区域（可滚动）
        properties_canvas = tk.Canvas(properties_frame, background=DarkTheme.BG_SECONDARY, highlightthickness=0)
        properties_scroll = ttk.Scrollbar(properties_frame, orient=tk.VERTICAL, command=properties_canvas.yview)
        self.property_panel = ttk.Frame(properties_canvas)

        properties_canvas.configure(yscrollcommand=properties_scroll.set)
        properties_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        properties_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        canvas_frame = properties_canvas.create_window((0, 0), window=self.property_panel, anchor='nw')

        # 配置滚动区域
        def configure_scroll_region(event):
            properties_canvas.configure(scrollregion=properties_canvas.bbox('all'))

        self.property_panel.bind('<Configure>', configure_scroll_region)

        # 默认属性面板内容
        self._populate_default_properties()

        # 渲染设置标签页
        render_frame = ttk.Frame(notebook)
        notebook.add(render_frame, text="渲染 Render")

        # 渲染设置内容（可滚动）
        render_canvas = tk.Canvas(render_frame, background=DarkTheme.BG_SECONDARY, highlightthickness=0)
        render_scroll = ttk.Scrollbar(render_frame, orient=tk.VERTICAL, command=render_canvas.yview)
        render_content = ttk.Frame(render_canvas)

        render_canvas.configure(yscrollcommand=render_scroll.set)
        render_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        render_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        canvas_frame = render_canvas.create_window((0, 0), window=render_content, anchor='nw')

        def configure_render_scroll(event):
            render_canvas.configure(scrollregion=render_canvas.bbox('all'))

        render_content.bind('<Configure>', configure_render_scroll)

        # 分辨率设置
        resolution_group = ttk.LabelFrame(render_content, text="分辨率 Resolution", padding=10)
        resolution_group.pack(fill=tk.X, padx=5, pady=5)

        # 当前视口分辨率显示
        current_res_frame = ttk.Frame(resolution_group)
        current_res_frame.pack(fill=tk.X, pady=5)
        ttk.Label(current_res_frame, text="当前视口:", style='Panel.TLabel').pack(side=tk.LEFT, padx=5)
        self.current_resolution_label = ttk.Label(current_res_frame, text="1920 x 1080", style='Panel.TLabel')
        self.current_resolution_label.pack(side=tk.LEFT)

        # 预设分辨率
        ttk.Label(resolution_group, text="渲染分辨率:", style='Panel.TLabel').pack(anchor='w', pady=(10, 5))

        self.resolution_var = tk.StringVar(value="viewport")
        resolution_options = [
            ("使用视口分辨率", "viewport"),
            ("1920 x 1080 (Full HD)", "1080p"),
            ("1280 x 720 (HD)", "720p"),
            ("3840 x 2160 (4K)", "4k"),
            ("2560 x 1440 (2K)", "2k"),
        ]

        for text, value in resolution_options:
            ttk.Radiobutton(
                resolution_group,
                text=text,
                variable=self.resolution_var,
                value=value
            ).pack(anchor='w', padx=10, pady=2)

        # 渲染质量设置
        quality_group = ttk.LabelFrame(render_content, text="质量 Quality", padding=10)
        quality_group.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(quality_group, text="抗锯齿:", style='Panel.TLabel').pack(anchor='w', pady=2)
        self.aa_var = tk.StringVar(value="4x")
        for aa_level in ["无", "2x MSAA", "4x MSAA", "8x MSAA"]:
            ttk.Radiobutton(quality_group, text=aa_level, variable=self.aa_var, value=aa_level).pack(anchor='w', padx=10, pady=2)

        # 输出设置
        output_group = ttk.LabelFrame(render_content, text="输出 Output", padding=10)
        output_group.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(output_group, text="文件格式:", style='Panel.TLabel').pack(anchor='w', pady=2)
        self.format_var = tk.StringVar(value="PNG")
        format_frame = ttk.Frame(output_group)
        format_frame.pack(fill=tk.X, padx=10, pady=2)

        ttk.Radiobutton(format_frame, text="PNG", variable=self.format_var, value="PNG").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(format_frame, text="JPEG", variable=self.format_var, value="JPEG").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(format_frame, text="BMP", variable=self.format_var, value="BMP").pack(side=tk.LEFT, padx=5)

        # 渲染按钮
        button_frame = ttk.Frame(render_content)
        button_frame.pack(fill=tk.X, padx=5, pady=10)

        ttk.Button(
            button_frame,
            text="渲染图像 Render Image",
            command=self._on_render_image,
            style='Accent.TButton'
        ).pack(fill=tk.X, pady=2)

        ttk.Button(
            button_frame,
            text="截图视口 Screenshot",
            command=self._on_screenshot_viewport
        ).pack(fill=tk.X, pady=2)

        return panel

    def _populate_default_properties(self):
        """填充默认属性面板 - Blender/UE5风格"""
        # 清空现有内容
        for widget in self.property_panel.winfo_children():
            widget.destroy()

        # 对象信息头 - UE5风格
        info_header = ttk.Frame(self.property_panel, style='Dark.TFrame', padding=10)
        info_header.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(info_header, text="对象属性", style='Title.TLabel',
                 font=('Segoe UI', 11, 'bold')).pack(anchor='w')
        ttk.Label(info_header, text="当前选中: 无", style='Subtitle.TLabel').pack(anchor='w', pady=(5, 0))

        ttk.Separator(self.property_panel, orient='horizontal', style='Light.TSeparator').pack(
            fill=tk.X, padx=5, pady=10)

        # 变换属性 - 可折叠
        transform_group = self._create_collapsible_group(self.property_panel, "变换 Transform", True)

        # 位置
        self._create_vector3_property(transform_group, "位置 Location", ("X", "Y", "Z"),
                                     (0.0, 0.0, 0.0), "m")

        # 旋转
        self._create_vector3_property(transform_group, "旋转 Rotation", ("X", "Y", "Z"),
                                     (0.0, 0.0, 0.0), "°")

        # 缩放
        self._create_vector3_property(transform_group, "缩放 Scale", ("X", "Y", "Z"),
                                     (1.0, 1.0, 1.0), "")

        # 材质属性 - 可折叠
        material_group = self._create_collapsible_group(self.property_panel, "材质 Material", False)

        # 基础颜色
        color_frame = ttk.Frame(material_group)
        color_frame.pack(fill=tk.X, pady=5)

        ttk.Label(color_frame, text="基础颜色:", width=15, style='Panel.TLabel').pack(side=tk.LEFT, padx=5)

        # 颜色预览框
        color_preview = tk.Canvas(color_frame, width=40, height=20,
                                 background='#808080',
                                 highlightthickness=1,
                                 highlightbackground=DarkTheme.BORDER_DEFAULT)
        color_preview.pack(side=tk.LEFT, padx=5)

        ttk.Button(color_frame, text="选择颜色...", style='Toolbar.TButton',
                  width=12).pack(side=tk.LEFT, padx=5)

        # 金属度
        self._create_slider_property(material_group, "金属度 Metallic", 0.0, 0.0, 1.0)

        # 粗糙度
        self._create_slider_property(material_group, "粗糙度 Roughness", 0.5, 0.0, 1.0)

        # 渲染属性 - 可折叠
        render_group = self._create_collapsible_group(self.property_panel, "渲染 Rendering", False)

        # 可见性
        visibility_frame = ttk.Frame(render_group)
        visibility_frame.pack(fill=tk.X, pady=2)

        visible_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(visibility_frame, text="视口可见", variable=visible_var).pack(anchor='w', padx=5)

        cast_shadow_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(visibility_frame, text="投射阴影", variable=cast_shadow_var).pack(anchor='w', padx=5)

        receive_shadow_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(visibility_frame, text="接收阴影", variable=receive_shadow_var).pack(anchor='w', padx=5)

    def _create_collapsible_group(self, parent, title, is_open=True):
        """创建可折叠的属性组"""
        # 容器
        container = ttk.Frame(parent, style='Panel.TFrame')
        container.pack(fill=tk.X, padx=5, pady=2)

        # 标题栏（可点击展开/折叠）
        header = ttk.Frame(container, style='Panel.TFrame')
        header.pack(fill=tk.X)

        # 展开/折叠指示器
        arrow_label = ttk.Label(header, text="▼" if is_open else "▶",
                               style='Panel.TLabel',
                               font=('Segoe UI', 9))
        arrow_label.pack(side=tk.LEFT, padx=(5, 0))

        title_label = ttk.Label(header, text=title, style='Panel.TLabel',
                               font=('Segoe UI', 9, 'bold'))
        title_label.pack(side=tk.LEFT, padx=5)

        # 内容区域
        content = ttk.Frame(container, style='Panel.TFrame', padding=(15, 5, 5, 10))
        if is_open:
            content.pack(fill=tk.X)

        # 点击标题栏切换展开/折叠
        def toggle():
            if content.winfo_viewable():
                content.pack_forget()
                arrow_label.config(text="▶")
            else:
                content.pack(fill=tk.X)
                arrow_label.config(text="▼")

        header.bind('<Button-1>', lambda e: toggle())
        arrow_label.bind('<Button-1>', lambda e: toggle())
        title_label.bind('<Button-1>', lambda e: toggle())

        return content

    def _create_vector3_property(self, parent, label, axes, values, unit):
        """创建Vector3属性输入控件"""
        row_frame = ttk.Frame(parent)
        row_frame.pack(fill=tk.X, pady=3)

        ttk.Label(row_frame, text=label, width=15, style='Panel.TLabel').pack(side=tk.LEFT, padx=5)

        for i, (axis, value) in enumerate(zip(axes, values)):
            axis_frame = ttk.Frame(row_frame)
            axis_frame.pack(side=tk.LEFT, padx=2)

            # 轴标签（带颜色 - X红, Y绿, Z蓝）
            axis_colors = {
                'X': DarkTheme.GIZMO_X,
                'Y': DarkTheme.GIZMO_Y,
                'Z': DarkTheme.GIZMO_Z
            }
            axis_label = tk.Label(axis_frame, text=axis, width=2,
                                 foreground=axis_colors.get(axis, DarkTheme.TEXT_PRIMARY),
                                 background=DarkTheme.BG_SECONDARY,
                                 font=('Segoe UI', 8, 'bold'))
            axis_label.pack(side=tk.LEFT)

            # 数值输入框
            entry = ttk.Entry(axis_frame, width=8)
            entry.insert(0, str(value))
            entry.pack(side=tk.LEFT)

        # 单位标签
        if unit:
            ttk.Label(row_frame, text=unit, style='Subtitle.TLabel',
                     font=('Segoe UI', 8)).pack(side=tk.LEFT, padx=5)

    def _create_slider_property(self, parent, label, value, min_val, max_val):
        """创建滑块属性控件"""
        row_frame = ttk.Frame(parent)
        row_frame.pack(fill=tk.X, pady=3)

        label_frame = ttk.Frame(row_frame)
        label_frame.pack(fill=tk.X)

        ttk.Label(label_frame, text=label, style='Panel.TLabel').pack(side=tk.LEFT, padx=5)

        value_label = ttk.Label(label_frame, text=f"{value:.2f}",
                               style='Panel.TLabel',
                               font=('Segoe UI', 8))
        value_label.pack(side=tk.RIGHT, padx=5)

        # 滑块
        slider_frame = ttk.Frame(row_frame)
        slider_frame.pack(fill=tk.X, padx=5, pady=2)

        slider = ttk.Scale(slider_frame, from_=min_val, to=max_val,
                          orient=tk.HORIZONTAL,
                          value=value)
        slider.pack(fill=tk.X, side=tk.LEFT, expand=True)

        # 更新数值标签
        def update_label(val):
            value_label.config(text=f"{float(val):.2f}")

        slider.config(command=update_label)

    def _create_status_bar(self):
        """创建状态栏"""
        self.status_bar = ttk.Frame(self.root, style='Dark.TFrame', padding=(5, 2))
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # 状态标签
        self.status_labels['engine'] = ttk.Label(self.status_bar, text="引擎状态: 运行中", style='Panel.TLabel')
        self.status_labels['engine'].pack(side=tk.LEFT, padx=10)

        ttk.Separator(self.status_bar, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=5)

        self.status_labels['fps'] = ttk.Label(self.status_bar, text="FPS: 60", style='Panel.TLabel')
        self.status_labels['fps'].pack(side=tk.LEFT, padx=10)

        self.status_labels['triangles'] = ttk.Label(self.status_bar, text="三角形: 0", style='Panel.TLabel')
        self.status_labels['triangles'].pack(side=tk.LEFT, padx=10)

        self.status_labels['memory'] = ttk.Label(self.status_bar, text="显存: 0 MB", style='Panel.TLabel')
        self.status_labels['memory'].pack(side=tk.LEFT, padx=10)

    def _bind_shortcuts(self):
        """绑定键盘快捷键 - Blender/UE5风格"""
        # 文件操作
        self.root.bind('<Control-n>', lambda e: self._on_new_scene())
        self.root.bind('<Control-o>', lambda e: self._on_open_scene())
        self.root.bind('<Control-s>', lambda e: self._on_save_scene())
        self.root.bind('<Control-q>', lambda e: self._on_close())

        # 编辑操作
        self.root.bind('<Control-z>', lambda e: self._on_undo())
        self.root.bind('<Control-y>', lambda e: self._on_redo())
        self.root.bind('<Control-c>', lambda e: self._on_copy())
        self.root.bind('<Control-v>', lambda e: self._on_paste())
        self.root.bind('<Delete>', lambda e: self._on_delete())
        self.root.bind('<Control-a>', lambda e: self._on_select_all())

        # 变换模式 - Blender风格
        self.root.bind('g', lambda e: self._set_transform_mode('translate', '移动'))
        self.root.bind('G', lambda e: self._set_transform_mode('translate', '移动'))
        self.root.bind('r', lambda e: self._set_transform_mode('rotate', '旋转'))
        self.root.bind('R', lambda e: self._set_transform_mode('rotate', '旋转'))
        self.root.bind('s', lambda e: self._set_transform_mode('scale', '缩放'))
        self.root.bind('S', lambda e: self._set_transform_mode('scale', '缩放'))

        # 着色模式切换
        self.root.bind('1', lambda e: self._set_shading_mode('wireframe', '线框'))
        self.root.bind('2', lambda e: self._set_shading_mode('solid', '实体'))
        self.root.bind('3', lambda e: self._set_shading_mode('material', '材质'))
        self.root.bind('4', lambda e: self._set_shading_mode('rendered', '渲染'))

        # 视图操作
        self.root.bind('<F12>', lambda e: self._on_render_current_view())

        # 对象创建 - Shift+A菜单（简化版）
        self.root.bind('<Shift-A>', lambda e: self._show_add_menu())

        self.logger.info("键盘快捷键已绑定")

    def _show_add_menu(self):
        """显示添加对象快捷菜单（Shift+A）- Blender风格"""
        # 创建弹出菜单
        add_menu = tk.Menu(self.root, tearoff=0)
        DarkTheme.configure_widget(add_menu, 'menu')

        # 基础几何体
        add_menu.add_command(label="立方体 Cube", command=self._on_create_cube)
        add_menu.add_command(label="球体 Sphere", command=self._on_create_sphere)
        add_menu.add_command(label="圆柱体 Cylinder", command=self._on_create_cylinder)
        add_menu.add_command(label="平面 Plane", command=self._on_create_plane)
        add_menu.add_command(label="圆锥体 Cone", command=self._on_create_cone)

        # 在鼠标位置显示菜单
        try:
            add_menu.tk_popup(self.root.winfo_pointerx(), self.root.winfo_pointery())
        finally:
            add_menu.grab_release()

    # ========== 菜单回调函数 ==========
    def _on_new_scene(self):
        self.logger.info("新建场景")
        if messagebox.askyesno("新建场景", "确定要新建场景吗？未保存的更改将丢失。"):
            # TODO: 清除当前场景
            pass

    def _on_open_scene(self):
        self.logger.info("打开场景")
        filename = filedialog.askopenfilename(
            title="打开场景",
            filetypes=[("Scene files", "*.scene"), ("All files", "*.*")]
        )
        if filename:
            self.logger.info(f"打开场景: {filename}")

    def _on_save_scene(self):
        self.logger.info("保存场景")
        # TODO: 实现保存逻辑

    def _on_save_scene_as(self):
        self.logger.info("另存为场景")
        filename = filedialog.asksaveasfilename(
            title="另存为",
            filetypes=[("Scene files", "*.scene"), ("All files", "*.*")]
        )
        if filename:
            self.logger.info(f"另存为: {filename}")

    def _on_import_model(self):
        """导入3D模型"""
        self.logger.info("导入模型")

        # 选择模型文件
        from Engine.Renderer.Resources.ModelLoader import ModelLoader

        filetypes = ModelLoader.get_supported_formats()
        filename = filedialog.askopenfilename(
            title="导入3D模型",
            filetypes=filetypes
        )

        if not filename:
            return

        self.logger.info(f"正在导入模型: {filename}")

        # 加载模型（自动检测格式）
        try:
            mesh = ModelLoader.load_model(filename)

            if mesh:
                self.logger.info(f"模型加载成功: {len(mesh.vertices)} 顶点")

                # 将模型添加到视口显示
                if self.viewport and hasattr(self.viewport, 'set_mesh'):
                    self.viewport.set_mesh(mesh)

                    # 获取文件格式
                    import os
                    _, ext = os.path.splitext(filename)

                    messagebox.showinfo(
                        "导入成功",
                        f"模型导入成功！\n\n"
                        f"格式: {ext.upper()}\n"
                        f"顶点数: {len(mesh.vertices)}\n"
                        f"三角形数: {len(mesh.indices) // 3}"
                    )
                else:
                    messagebox.showinfo(
                        "导入成功",
                        f"模型已加载，但视口不支持显示\n\n"
                        f"顶点数: {len(mesh.vertices)}\n"
                        f"三角形数: {len(mesh.indices) // 3}"
                    )
            else:
                messagebox.showerror("导入失败", "无法加载模型文件，请检查文件格式")

        except Exception as e:
            self.logger.error(f"导入模型失败: {e}", exc_info=True)
            messagebox.showerror("导入失败", f"导入模型时出错：\n{str(e)}")

    def _on_export_model(self):
        self.logger.info("导出模型")

    def _on_undo(self):
        self.logger.info("撤销")

    def _on_redo(self):
        self.logger.info("重做")

    def _on_copy(self):
        self.logger.info("复制")

    def _on_paste(self):
        self.logger.info("粘贴")

    def _on_delete(self):
        self.logger.info("删除")

    def _on_select_all(self):
        self.logger.info("全选")

    def _on_perspective_view(self):
        self.logger.info("切换到透视视图")
        if hasattr(self.viewport, 'set_view_mode'):
            self.viewport.set_view_mode('perspective')

    def _on_orthographic_view(self):
        self.logger.info("切换到正交视图")
        if hasattr(self.viewport, 'set_view_mode'):
            self.viewport.set_view_mode('orthographic')

    def _on_top_view(self):
        self.logger.info("顶视图")
        if hasattr(self.viewport, 'set_camera_preset'):
            self.viewport.set_camera_preset('top')

    def _on_front_view(self):
        self.logger.info("前视图")
        if hasattr(self.viewport, 'set_camera_preset'):
            self.viewport.set_camera_preset('front')

    def _on_right_view(self):
        self.logger.info("右视图")
        if hasattr(self.viewport, 'set_camera_preset'):
            self.viewport.set_camera_preset('right')

    def _on_toggle_grid(self):
        self.logger.info("切换网格显示")
        if hasattr(self.viewport, 'toggle_grid'):
            self.viewport.toggle_grid()

    def _on_toggle_axes(self):
        self.logger.info("切换坐标轴显示")
        if hasattr(self.viewport, 'toggle_axes'):
            self.viewport.toggle_axes()

    def _on_reset_camera(self):
        """重置相机"""
        self.logger.info("重置相机")
        if hasattr(self.viewport, 'reset_camera'):
            self.viewport.reset_camera()

    def _on_toggle_wireframe(self):
        self.logger.info("切换线框模式")

    def _on_create_cube(self):
        self.logger.info("创建立方体")
        if self.engine and hasattr(self.engine, 'scene_mgr'):
            # TODO: 调用引擎创建立方体
            pass

    def _on_create_sphere(self):
        self.logger.info("创建球体")

    def _on_create_cylinder(self):
        self.logger.info("创建圆柱体")

    def _on_create_plane(self):
        self.logger.info("创建平面")

    def _on_create_cone(self):
        self.logger.info("创建圆锥体")

    def _on_duplicate(self):
        self.logger.info("复制对象")

    def _on_render_current_view(self):
        self.logger.info("渲染当前视图")

    def _on_render_settings(self):
        self.logger.info("渲染设置")

    def _on_show_shortcuts(self):
        """显示快捷键帮助"""
        shortcuts_text = """
        快捷键列表 Shortcuts:

        文件 File:
        Ctrl+N: 新建场景 New Scene
        Ctrl+O: 打开场景 Open Scene
        Ctrl+S: 保存场景 Save Scene
        Ctrl+Q: 退出 Quit

        编辑 Edit:
        Ctrl+Z: 撤销 Undo
        Ctrl+Y: 重做 Redo
        Delete: 删除 Delete

        变换 Transform:
        G: 移动 Move
        R: 旋转 Rotate
        S: 缩放 Scale

        视图 View:
        F12: 渲染 Render
        """
        messagebox.showinfo("快捷键 Shortcuts", shortcuts_text)

    def _on_about(self):
        """关于对话框"""
        about_text = """
        低端GPU优化渲染引擎
        Low-End GPU Rendering Engine

        版本 Version: 1.0.0
        专为GTX 750Ti和RX 580优化
        Optimized for GTX 750Ti and RX 580

        MIT License
        """
        messagebox.showinfo("关于 About", about_text)

    def _create_tooltip(self, widget, text):
        """创建工具提示"""
        def on_enter(event):
            # 创建提示窗口
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root + 10}+{event.y_root + 10}")

            label = tk.Label(tooltip, text=text,
                           background=DarkTheme.BG_TERTIARY,
                           foreground=DarkTheme.TEXT_PRIMARY,
                           relief='solid',
                           borderwidth=1,
                           font=('Segoe UI', 8),
                           padx=8, pady=4)
            label.pack()

            widget.tooltip = tooltip

        def on_leave(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                del widget.tooltip

        widget.bind('<Enter>', on_enter)
        widget.bind('<Leave>', on_leave)

    def _set_transform_mode(self, mode, label):
        """设置变换模式"""
        self.logger.info(f"变换模式: {mode}")

        # 更新当前模式指示器
        if hasattr(self, 'current_mode_label'):
            self.current_mode_label.config(text=label)

        # 重置所有按钮样式
        for btn in self.transform_buttons.values():
            btn.config(style='Toolbar.TButton')

        # 高亮当前按钮
        if mode in self.transform_buttons:
            self.transform_buttons[mode].config(style='Accent.TButton')

    def _set_shading_mode(self, mode, label):
        """设置着色/渲染模式"""
        self.logger.info(f"着色模式: {mode}")

        # 更新当前模式指示器
        if hasattr(self, 'current_mode_label'):
            self.current_mode_label.config(text=label)

        # 重置所有按钮样式
        if hasattr(self, 'shading_buttons'):
            for btn in self.shading_buttons.values():
                btn.config(style='Toolbar.TButton')

            # 高亮当前按钮
            if mode in self.shading_buttons:
                self.shading_buttons[mode].config(style='Accent.TButton')

        # 通知视口切换着色模式
        if self.viewport and hasattr(self.viewport, 'set_shading_mode'):
            self.viewport.set_shading_mode(mode)
            self.logger.info(f"切换到 {label} 模式")

    def _on_camera_orbit(self):
        self.logger.info("相机轨道模式")

    def _on_camera_pan(self):
        self.logger.info("相机平移模式")

    def _on_camera_zoom(self):
        self.logger.info("相机缩放模式")

    def _on_scene_tree_select(self, event):
        """场景树选择事件"""
        selection = self.scene_tree.selection()
        if selection:
            item = self.scene_tree.item(selection[0])
            self.logger.info(f"选中: {item['text']}")

    def _on_render_image(self):
        """渲染图像到文件"""
        self.logger.info("渲染图像")

        # 获取渲染分辨率
        resolution_map = {
            "viewport": None,  # 使用视口分辨率
            "1080p": (1920, 1080),
            "720p": (1280, 720),
            "4k": (3840, 2160),
            "2k": (2560, 1440),
        }

        resolution = resolution_map.get(self.resolution_var.get())
        if resolution is None and self.viewport:
            resolution = (self.viewport.width, self.viewport.height)

        # 选择保存文件路径
        file_format = self.format_var.get().lower()
        filename = filedialog.asksaveasfilename(
            title="保存渲染图像",
            defaultextension=f".{file_format}",
            filetypes=[(f"{file_format.upper()} files", f"*.{file_format}"), ("All files", "*.*")]
        )

        if filename:
            self.logger.info(f"渲染到: {filename}, 分辨率: {resolution}")
            messagebox.showinfo("渲染", f"渲染功能开发中\n将渲染到: {filename}\n分辨率: {resolution[0]}x{resolution[1]}")

    def _on_screenshot_viewport(self):
        """截图当前视口"""
        self.logger.info("截图视口")

        # 选择保存文件路径
        file_format = self.format_var.get().lower()
        filename = filedialog.asksaveasfilename(
            title="保存截图",
            defaultextension=f".{file_format}",
            filetypes=[(f"{file_format.upper()} files", f"*.{file_format}"), ("All files", "*.*")]
        )

        if filename:
            self.logger.info(f"截图到: {filename}")
            messagebox.showinfo("截图", f"截图功能开发中\n将保存到: {filename}")

    def _on_close(self):
        """窗口关闭事件"""
        if messagebox.askyesno("退出", "确定要退出吗？"):
            self.logger.info("关闭主窗口")
            self.running = False
            self.root.quit()
            self.root.destroy()

    def start(self):
        """启动UI（由引擎调用）"""
        self.logger.info("启动Tkinter UI")
        self.running = True
        # 不在这里调用mainloop，由main.py控制

    def update(self):
        """更新UI状态"""
        if not self.running:
            return

        # 更新视口分辨率显示
        if self.viewport and hasattr(self.viewport, 'width') and hasattr(self.viewport, 'height'):
            resolution_text = f"{self.viewport.width} x {self.viewport.height}"
            if hasattr(self, 'current_resolution_label'):
                self.current_resolution_label.config(text=resolution_text)

        # 更新状态栏
        if self.engine and hasattr(self.engine, 'renderer'):
            # TODO: 从引擎获取实际数据
            pass

    def shutdown(self):
        """关闭UI"""
        if not self.running:
            return

        self.logger.info("关闭Tkinter UI")
        self.running = False
        try:
            self.root.quit()
            self.root.destroy()
        except:
            pass


# 测试代码
if __name__ == "__main__":
    # 模拟引擎
    class MockEngine:
        def __init__(self):
            self.is_initialized = True
            self.renderer = None
            self.scene_mgr = None

    engine = MockEngine()
    ui = TkMainWindow(engine)
    ui.start()
    ui.root.mainloop()
