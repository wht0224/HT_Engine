# -*- coding: utf-8 -*-
"""
Blender 3.x/UE5风格的专业暗色主题
Professional dark theme inspired by Blender 3.x and Unreal Engine 5
"""

import tkinter as tk
from tkinter import ttk


class DarkTheme:
    """
    专业的暗色主题配置 - 参考Blender 3.x和UE5的设计语言
    Professional dark theme configuration inspired by modern 3D software
    """

    # ========== 主色调 - Main Colors (Blender 3.x风格) ==========
    BG_PRIMARY = "#232323"      # 主背景（深灰） - Primary background
    BG_SECONDARY = "#2b2b2b"    # 次级背景（中灰） - Secondary background
    BG_TERTIARY = "#383838"     # 三级背景（浅灰） - Tertiary background
    BG_DARK = "#1a1a1a"         # 更深的背景 - Darkest background
    BG_INPUT = "#1e1e1e"        # 输入框背景 - Input field background
    BG_HOVER = "#404040"        # 悬停背景 - Hover background

    # ========== 强调色 - Accent Colors (UE5启发) ==========
    # 主强调色（蓝色系 - Blender风格）
    ACCENT_PRIMARY = "#3d8bc9"       # 主强调色 - Primary accent (bright blue)
    ACCENT_PRIMARY_HOVER = "#4a9bd9" # 主强调悬停 - Primary accent hover
    ACCENT_PRIMARY_ACTIVE = "#2d7ab9"# 主强调激活 - Primary accent active

    # 次要强调色（橙色系 - UE5风格）
    ACCENT_SECONDARY = "#e87d0d"     # 次要强调 - Secondary accent (orange)
    ACCENT_SECONDARY_HOVER = "#f58b1d"
    ACCENT_SECONDARY_ACTIVE = "#d86d0d"

    # 状态颜色
    SUCCESS = "#5cb85c"         # 成功/添加（绿色）
    WARNING = "#f0ad4e"         # 警告（黄色）
    ERROR = "#d9534f"           # 错误/删除（红色）
    INFO = "#5bc0de"            # 信息（浅蓝色）

    # ========== 文字颜色 - Text Colors ==========
    TEXT_PRIMARY = "#e8e8e8"    # 主文字（亮灰） - Primary text
    TEXT_SECONDARY = "#9d9d9d"  # 次要文字（灰色） - Secondary text
    TEXT_DISABLED = "#595959"   # 禁用文字（深灰） - Disabled text
    TEXT_HIGHLIGHT = "#ffffff"  # 高亮文字（白色） - Highlight text
    TEXT_LINK = "#4da6ff"       # 链接文字（蓝色） - Link text

    # ========== 边框颜色 - Border Colors ==========
    BORDER_DEFAULT = "#1a1a1a"  # 默认边框（深色）
    BORDER_LIGHT = "#404040"    # 浅色边框
    BORDER_FOCUS = "#3d8bc9"    # 焦点边框（蓝色）
    BORDER_HOVER = "#4a9bd9"    # 悬停边框
    BORDER_SEPARATOR = "#181818"# 分隔线

    # ========== 视口颜色 - Viewport Colors ==========
    VIEWPORT_BG = "#1a1a1a"     # 3D视口背景
    VIEWPORT_GRID = "#2d2d2d"   # 网格颜色
    VIEWPORT_GRID_MAJOR = "#404040"  # 主网格线
    VIEWPORT_GRID_AXIS = "#555555"   # 轴线

    # ========== Gizmo/轴颜色 - Gizmo/Axis Colors (标准RGB) ==========
    GIZMO_X = "#ee5855"         # X轴（红色） - X axis red
    GIZMO_Y = "#7db455"         # Y轴（绿色） - Y axis green
    GIZMO_Z = "#5589d6"         # Z轴（蓝色） - Z axis blue
    GIZMO_W = "#ffffff"         # W/通用（白色） - White/universal
    GIZMO_SELECTED = "#ffdd00"  # 选中的Gizmo（亮黄色）
    GIZMO_HOVER = "#ffa726"     # 悬停的Gizmo（橙色）

    # ========== 选择/激活颜色 - Selection/Active Colors ==========
    SELECTION_PRIMARY = "#3d8bc9"    # 主选择色（蓝色）
    SELECTION_SECONDARY = "#e87d0d"  # 次要选择色（橙色）
    ACTIVE_ELEMENT = "#ff9800"       # 激活元素（橙色）
    INACTIVE_ELEMENT = "#4d4d4d"     # 非激活元素

    @staticmethod
    def apply(root):
        """
        应用暗色主题到Tkinter窗口
        Apply dark theme to Tkinter window

        Args:
            root: Tkinter根窗口
        """
        # 设置窗口背景色
        root.configure(background=DarkTheme.BG_PRIMARY)

        # 创建样式对象
        style = ttk.Style(root)
        style.theme_use('clam')  # 使用clam主题作为基础

        # ========== Frame样式 ==========
        style.configure('TFrame',
                       background=DarkTheme.BG_PRIMARY,
                       borderwidth=0)

        style.configure('Panel.TFrame',
                       background=DarkTheme.BG_SECONDARY,
                       borderwidth=1,
                       relief='solid',
                       bordercolor=DarkTheme.BORDER_DEFAULT)

        style.configure('Dark.TFrame',
                       background=DarkTheme.BG_DARK,
                       borderwidth=0)

        style.configure('Viewport.TFrame',
                       background=DarkTheme.VIEWPORT_BG,
                       borderwidth=0)

        # ========== Label样式 ==========
        style.configure('TLabel',
                       background=DarkTheme.BG_PRIMARY,
                       foreground=DarkTheme.TEXT_PRIMARY,
                       font=('Segoe UI', 9))

        style.configure('Title.TLabel',
                       background=DarkTheme.BG_SECONDARY,
                       foreground=DarkTheme.TEXT_HIGHLIGHT,
                       font=('Segoe UI', 10, 'bold'))

        style.configure('Panel.TLabel',
                       background=DarkTheme.BG_SECONDARY,
                       foreground=DarkTheme.TEXT_PRIMARY,
                       font=('Segoe UI', 9))

        style.configure('Subtitle.TLabel',
                       background=DarkTheme.BG_SECONDARY,
                       foreground=DarkTheme.TEXT_SECONDARY,
                       font=('Segoe UI', 8))

        # ========== Button样式 - 参考Blender ==========
        style.configure('TButton',
                       background=DarkTheme.BG_TERTIARY,
                       foreground=DarkTheme.TEXT_PRIMARY,
                       borderwidth=1,
                       relief='flat',
                       padding=(12, 6),
                       font=('Segoe UI', 9))

        style.map('TButton',
                 background=[('active', DarkTheme.BG_HOVER),
                            ('pressed', DarkTheme.ACCENT_PRIMARY_ACTIVE),
                            ('disabled', DarkTheme.BG_DARK)],
                 foreground=[('pressed', DarkTheme.TEXT_HIGHLIGHT),
                            ('disabled', DarkTheme.TEXT_DISABLED)],
                 bordercolor=[('focus', DarkTheme.BORDER_FOCUS),
                             ('active', DarkTheme.BORDER_HOVER)])

        # 工具栏按钮样式 - Blender风格
        style.configure('Toolbar.TButton',
                       background=DarkTheme.BG_DARK,
                       foreground=DarkTheme.TEXT_PRIMARY,
                       borderwidth=0,
                       padding=(10, 5),
                       font=('Segoe UI', 9))

        style.map('Toolbar.TButton',
                 background=[('active', DarkTheme.BG_HOVER),
                            ('pressed', DarkTheme.ACCENT_PRIMARY)],
                 foreground=[('pressed', DarkTheme.TEXT_HIGHLIGHT)])

        # Accent强调按钮样式 - UE5风格
        style.configure('Accent.TButton',
                       background=DarkTheme.ACCENT_PRIMARY,
                       foreground=DarkTheme.TEXT_HIGHLIGHT,
                       borderwidth=0,
                       padding=(12, 6),
                       font=('Segoe UI', 9, 'bold'))

        style.map('Accent.TButton',
                 background=[('active', DarkTheme.ACCENT_PRIMARY_HOVER),
                            ('pressed', DarkTheme.ACCENT_PRIMARY_ACTIVE)],
                 foreground=[('disabled', DarkTheme.TEXT_DISABLED)])

        # Secondary按钮样式
        style.configure('Secondary.TButton',
                       background=DarkTheme.ACCENT_SECONDARY,
                       foreground=DarkTheme.TEXT_HIGHLIGHT,
                       borderwidth=0,
                       padding=(12, 6),
                       font=('Segoe UI', 9, 'bold'))

        style.map('Secondary.TButton',
                 background=[('active', DarkTheme.ACCENT_SECONDARY_HOVER),
                            ('pressed', DarkTheme.ACCENT_SECONDARY_ACTIVE)])

        # ========== Entry样式 ==========
        style.configure('TEntry',
                       fieldbackground=DarkTheme.BG_INPUT,
                       foreground=DarkTheme.TEXT_PRIMARY,
                       bordercolor=DarkTheme.BORDER_DEFAULT,
                       insertcolor=DarkTheme.TEXT_PRIMARY,
                       selectbackground=DarkTheme.ACCENT_PRIMARY,
                       selectforeground=DarkTheme.TEXT_HIGHLIGHT,
                       padding=6)

        style.map('TEntry',
                 fieldbackground=[('focus', DarkTheme.BG_TERTIARY)],
                 bordercolor=[('focus', DarkTheme.BORDER_FOCUS)])

        # ========== Treeview样式（场景树） - Blender风格 ==========
        style.configure('Treeview',
                       background=DarkTheme.BG_SECONDARY,
                       foreground=DarkTheme.TEXT_PRIMARY,
                       fieldbackground=DarkTheme.BG_SECONDARY,
                       borderwidth=0,
                       rowheight=24,
                       font=('Segoe UI', 9))

        style.configure('Treeview.Heading',
                       background=DarkTheme.BG_DARK,
                       foreground=DarkTheme.TEXT_PRIMARY,
                       borderwidth=1,
                       relief='flat',
                       font=('Segoe UI', 9, 'bold'))

        style.map('Treeview',
                 background=[('selected', DarkTheme.SELECTION_PRIMARY)],
                 foreground=[('selected', DarkTheme.TEXT_HIGHLIGHT)])

        style.map('Treeview.Heading',
                 background=[('active', DarkTheme.BG_TERTIARY)])

        # ========== LabelFrame样式 ==========
        style.configure('TLabelframe',
                       background=DarkTheme.BG_SECONDARY,
                       foreground=DarkTheme.TEXT_PRIMARY,
                       borderwidth=1,
                       bordercolor=DarkTheme.BORDER_DEFAULT,
                       relief='flat')

        style.configure('TLabelframe.Label',
                       background=DarkTheme.BG_SECONDARY,
                       foreground=DarkTheme.TEXT_HIGHLIGHT,
                       font=('Segoe UI', 9, 'bold'))

        # ========== Notebook样式（标签页） - UE5风格 ==========
        style.configure('TNotebook',
                       background=DarkTheme.BG_PRIMARY,
                       borderwidth=0,
                       tabmargins=[2, 5, 2, 0])

        style.configure('TNotebook.Tab',
                       background=DarkTheme.BG_DARK,
                       foreground=DarkTheme.TEXT_SECONDARY,
                       borderwidth=0,
                       padding=[14, 8],
                       font=('Segoe UI', 9))

        style.map('TNotebook.Tab',
                 background=[('selected', DarkTheme.BG_TERTIARY),
                            ('active', DarkTheme.BG_HOVER)],
                 foreground=[('selected', DarkTheme.TEXT_HIGHLIGHT),
                            ('active', DarkTheme.TEXT_PRIMARY)],
                 expand=[('selected', [1, 1, 1, 0])])

        # ========== Separator样式 ==========
        style.configure('TSeparator',
                       background=DarkTheme.BORDER_SEPARATOR)

        style.configure('Light.TSeparator',
                       background=DarkTheme.BORDER_LIGHT)

        # ========== Panedwindow样式 ==========
        style.configure('TPanedwindow',
                       background=DarkTheme.BG_PRIMARY,
                       borderwidth=0,
                       sashwidth=6,
                       sashrelief='flat')

        # ========== Scrollbar样式 - 现代化滚动条 ==========
        style.configure('TScrollbar',
                       background=DarkTheme.BG_TERTIARY,
                       troughcolor=DarkTheme.BG_DARK,
                       borderwidth=0,
                       arrowcolor=DarkTheme.TEXT_SECONDARY,
                       width=12)

        style.map('TScrollbar',
                 background=[('active', DarkTheme.BG_HOVER),
                            ('pressed', DarkTheme.ACCENT_PRIMARY)])

        # ========== Scale样式（滑块） ==========
        style.configure('TScale',
                       background=DarkTheme.BG_SECONDARY,
                       troughcolor=DarkTheme.BG_DARK,
                       borderwidth=0,
                       sliderrelief='flat',
                       sliderlength=20)

        style.map('TScale',
                 background=[('active', DarkTheme.ACCENT_PRIMARY)])

        # ========== Checkbutton样式 ==========
        style.configure('TCheckbutton',
                       background=DarkTheme.BG_SECONDARY,
                       foreground=DarkTheme.TEXT_PRIMARY,
                       font=('Segoe UI', 9),
                       indicatorcolor=DarkTheme.BG_INPUT,
                       selectcolor=DarkTheme.ACCENT_PRIMARY)

        style.map('TCheckbutton',
                 background=[('active', DarkTheme.BG_SECONDARY)],
                 foreground=[('disabled', DarkTheme.TEXT_DISABLED)])

        # ========== Radiobutton样式 ==========
        style.configure('TRadiobutton',
                       background=DarkTheme.BG_SECONDARY,
                       foreground=DarkTheme.TEXT_PRIMARY,
                       font=('Segoe UI', 9))

        style.map('TRadiobutton',
                 background=[('active', DarkTheme.BG_SECONDARY)],
                 foreground=[('disabled', DarkTheme.TEXT_DISABLED)])

        # ========== Progressbar样式 ==========
        style.configure('TProgressbar',
                       background=DarkTheme.ACCENT_PRIMARY,
                       troughcolor=DarkTheme.BG_DARK,
                       borderwidth=0,
                       thickness=6)

        # ========== Menu样式 - Tkinter Menu不使用ttk ==========
        root.option_add('*Menu.background', DarkTheme.BG_DARK)
        root.option_add('*Menu.foreground', DarkTheme.TEXT_PRIMARY)
        root.option_add('*Menu.activeBackground', DarkTheme.ACCENT_PRIMARY)
        root.option_add('*Menu.activeForeground', DarkTheme.TEXT_HIGHLIGHT)
        root.option_add('*Menu.borderWidth', 0)
        root.option_add('*Menu.relief', 'flat')

        print("[OK] Blender/UE5风格暗色主题已应用 - Professional dark theme applied")

    @staticmethod
    def configure_widget(widget, widget_type='default'):
        """
        为特定widget配置暗色主题
        Configure dark theme for specific widget types

        Args:
            widget: 要配置的widget
            widget_type: widget类型 ('canvas', 'text', 'listbox', etc.)
        """
        if widget_type == 'canvas':
            widget.configure(
                background=DarkTheme.VIEWPORT_BG,
                highlightthickness=0,
                borderwidth=0
            )
        elif widget_type == 'text':
            widget.configure(
                background=DarkTheme.BG_INPUT,
                foreground=DarkTheme.TEXT_PRIMARY,
                insertbackground=DarkTheme.TEXT_PRIMARY,
                selectbackground=DarkTheme.ACCENT_PRIMARY,
                selectforeground=DarkTheme.TEXT_HIGHLIGHT,
                borderwidth=0,
                font=('Consolas', 9)
            )
        elif widget_type == 'listbox':
            widget.configure(
                background=DarkTheme.BG_SECONDARY,
                foreground=DarkTheme.TEXT_PRIMARY,
                selectbackground=DarkTheme.ACCENT_PRIMARY,
                selectforeground=DarkTheme.TEXT_HIGHLIGHT,
                borderwidth=0,
                highlightthickness=0,
                font=('Segoe UI', 9)
            )
        elif widget_type == 'menu':
            widget.configure(
                background=DarkTheme.BG_DARK,
                foreground=DarkTheme.TEXT_PRIMARY,
                activebackground=DarkTheme.ACCENT_PRIMARY,
                activeforeground=DarkTheme.TEXT_HIGHLIGHT,
                borderwidth=0,
                tearoff=0
            )


# 使用示例 - Usage Example
if __name__ == "__main__":
    # 测试主题
    root = tk.Tk()
    root.title("Blender/UE5 Dark Theme Test")
    root.geometry("800x600")

    # 应用主题
    DarkTheme.apply(root)

    # 创建一些测试组件
    main_frame = ttk.Frame(root, style='Panel.TFrame', padding=20)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # 标题
    ttk.Label(main_frame, text="Blender/UE5风格暗色主题测试",
             style='Title.TLabel').pack(pady=10)

    # 按钮组
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(fill=tk.X, pady=10)

    ttk.Button(button_frame, text="普通按钮").pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="主要操作", style='Accent.TButton').pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="次要操作", style='Secondary.TButton').pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="工具栏", style='Toolbar.TButton').pack(side=tk.LEFT, padx=5)

    # 输入框
    ttk.Label(main_frame, text="输入框测试:", style='Panel.TLabel').pack(anchor='w', pady=(10, 5))
    ttk.Entry(main_frame).pack(fill=tk.X, pady=5)

    # 树形视图
    ttk.Label(main_frame, text="场景树:", style='Panel.TLabel').pack(anchor='w', pady=(10, 5))
    tree = ttk.Treeview(main_frame, height=5)
    tree.pack(fill=tk.BOTH, expand=True, pady=5)
    tree.insert('', 'end', text='场景 Scene', open=True)
    tree.insert('', 'end', text='立方体 Cube')
    tree.insert('', 'end', text='光源 Light')
    tree.insert('', 'end', text='相机 Camera')

    root.mainloop()
