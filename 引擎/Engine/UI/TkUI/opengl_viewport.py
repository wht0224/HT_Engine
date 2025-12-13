# -*- coding: utf-8 -*-
"""
基于pyopengltk的OpenGL 3D视口
OpenGL 3D Viewport using pyopengltk
"""

import tkinter as tk
from tkinter import ttk
import sys
import os

# 添加引擎路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

try:
    from pyopengltk import OpenGLFrame
    HAS_PYOPENGLTK = True
except ImportError:
    print("警告：pyopengltk未安装，OpenGL视口将不可用")
    print("请运行: pip install pyopengltk")
    HAS_PYOPENGLTK = False
    OpenGLFrame = tk.Frame  # 降级方案

from OpenGL.GL import *
from OpenGL.GLU import *

from Engine.Logger import get_logger
from Engine.UI.TkUI.theme.dark_theme import DarkTheme


class OpenGLViewport(OpenGLFrame if HAS_PYOPENGLTK else tk.Frame):
    """
    OpenGL 3D视口类
    使用pyopengltk在Tkinter中嵌入真实的OpenGL渲染
    """

    def __init__(self, master, engine, **kwargs):
        """
        初始化OpenGL视口

        Args:
            master: 父窗口
            engine: 引擎实例
            **kwargs: 传递给OpenGLFrame的额外参数
        """
        # 如果pyopengltk不可用，使用降级方案
        if not HAS_PYOPENGLTK:
            super().__init__(master, **kwargs)
            self.logger = get_logger("OpenGLViewport")
            self.logger.error("pyopengltk未安装，OpenGL视口不可用")
            self._create_fallback_ui()
            return

        # 初始化OpenGLFrame（pyopengltk会自动调用redraw）
        super().__init__(master, **kwargs)

        self.logger = get_logger("OpenGLViewport")
        self.engine = engine
        self.width = 800
        self.height = 600

        # 渲染状态
        self.is_initialized = False
        self.render_grid = True
        self.render_axes = True
        self.view_mode = 'perspective'  # 'perspective' or 'orthographic'
        self.shading_mode = 'rendered'  # 'solid', 'material', 'rendered'

        # 相机控制
        self.camera_distance = 10.0
        self.camera_rotation_x = 30.0
        self.camera_rotation_y = 45.0
        self.camera_pan_x = 0.0
        self.camera_pan_y = 0.0

        # 鼠标交互状态
        self.mouse_down = False
        self.mouse_button = None
        self.last_mouse_x = 0
        self.last_mouse_y = 0

        # 性能统计
        self.frame_count = 0
        self.fps = 0

        # 自定义网格
        self.custom_mesh = None  # 用户导入的模型

        # 绑定事件
        self._bind_events()

        # 设置焦点以接收鼠标事件
        self.bind('<Enter>', lambda e: self.focus_set())

        self.logger.info("OpenGL视口创建完成")

    def _create_fallback_ui(self):
        """创建降级UI（当pyopengltk不可用时）"""
        canvas = tk.Canvas(self, background=DarkTheme.VIEWPORT_BG, highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        canvas.create_text(
            400, 300,
            text="pyopengltk未安装\n请运行: pip install pyopengltk",
            fill=DarkTheme.TEXT_SECONDARY,
            font=('Segoe UI', 14),
            justify=tk.CENTER
        )

    def initgl(self):
        """
        OpenGL初始化回调（pyopengltk自动调用）
        在OpenGL上下文创建后调用，仅调用一次
        """
        self.logger.info("初始化OpenGL上下文...")

        # 设置清除颜色（暗灰色背景）
        glClearColor(0.15, 0.15, 0.15, 1.0)

        # 启用深度测试
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)

        # 启用背面剔除
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glFrontFace(GL_CCW)

        # 启用混合（用于透明度）
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # 启用平滑着色
        glShadeModel(GL_SMOOTH)

        # 启用抗锯齿（改善线条质量）
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_POLYGON_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)

        # 启用光照（光源位置会在每帧渲染时设置）
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)

        # 设置全局材质参数（默认材质）
        mat_ambient = [0.3, 0.3, 0.3, 1.0]
        mat_diffuse = [0.8, 0.8, 0.8, 1.0]
        mat_specular = [1.0, 1.0, 1.0, 1.0]
        mat_shininess = [50.0]

        glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse)
        glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular)
        glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess)

        # 设置视口
        self.width = self.winfo_width()
        self.height = self.winfo_height()
        glViewport(0, 0, self.width, self.height)

        self.is_initialized = True
        self.logger.info(f"OpenGL初始化完成 - 视口大小: {self.width}x{self.height}")

        # 延迟初始化渲染器（现在OpenGL上下文已准备好）
        if self.engine and hasattr(self.engine, 'initialize_renderer_deferred'):
            self.logger.info("触发渲染器延迟初始化...")
            success = self.engine.initialize_renderer_deferred()
            if success:
                self.logger.info("渲染器延迟初始化成功")
            else:
                self.logger.warning("渲染器延迟初始化失败，将使用基础渲染")

        # 启动持续渲染循环
        self._start_render_loop()

        # 立即设置焦点以确保能接收鼠标事件
        self.focus_set()
        self.logger.info("OpenGL视口焦点已设置，可以接收鼠标事件")

    def _start_render_loop(self):
        """启动持续渲染循环"""
        def render_tick():
            if self.is_initialized:
                # 触发重绘
                self.event_generate('<Expose>')
            # 16ms后再次调用（约60 FPS）
            self.after(16, render_tick)

        # 启动循环
        self.after(16, render_tick)

    def redraw(self):
        """
        渲染循环回调（pyopengltk每帧调用）
        """
        if not self.is_initialized:
            return

        # 清除颜色和深度缓冲区
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # 设置投影矩阵
        self._setup_projection()

        # 设置模型视图矩阵
        self._setup_modelview()

        # 在视图空间中设置光源（跟随相机）
        self._setup_lights()

        # 渲染网格
        if self.render_grid:
            self._render_grid()

        # 渲染坐标轴
        if self.render_axes:
            self._render_axes()

        # 渲染场景对象
        self._render_scene()

        # 更新FPS计数
        self.frame_count += 1

    def _setup_projection(self):
        """设置投影矩阵"""
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        aspect = self.width / self.height if self.height > 0 else 1.0

        if self.view_mode == 'perspective':
            # 透视投影
            gluPerspective(60.0, aspect, 0.1, 1000.0)
        else:
            # 正交投影
            size = 5.0
            glOrtho(-size * aspect, size * aspect, -size, size, 0.1, 1000.0)

    def _setup_modelview(self):
        """设置模型视图矩阵（相机变换）"""
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # 相机位置和旋转
        glTranslatef(self.camera_pan_x, self.camera_pan_y, -self.camera_distance)
        glRotatef(self.camera_rotation_x, 1.0, 0.0, 0.0)
        glRotatef(self.camera_rotation_y, 0.0, 1.0, 0.0)

    def _setup_lights(self):
        """设置光源（在视图空间中，跟随相机）"""
        # 主光源（从右上前方照射）
        light_pos = [10.0, 10.0, 10.0, 1.0]
        light_ambient = [0.3, 0.3, 0.3, 1.0]
        light_diffuse = [1.0, 1.0, 1.0, 1.0]
        light_specular = [1.0, 1.0, 1.0, 1.0]

        glLightfv(GL_LIGHT0, GL_POSITION, light_pos)
        glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
        glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular)

        # 补光（从左后方照射）
        fill_light_pos = [-5.0, 5.0, -5.0, 1.0]
        fill_light_ambient = [0.1, 0.1, 0.1, 1.0]
        fill_light_diffuse = [0.4, 0.4, 0.5, 1.0]

        glLightfv(GL_LIGHT1, GL_POSITION, fill_light_pos)
        glLightfv(GL_LIGHT1, GL_AMBIENT, fill_light_ambient)
        glLightfv(GL_LIGHT1, GL_DIFFUSE, fill_light_diffuse)

    def _render_grid(self):
        """渲染网格"""
        glDisable(GL_LIGHTING)  # 网格不需要光照

        # 启用线条平滑
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_BLEND)

        grid_size = 20  # 增大网格范围
        grid_step = 1.0
        grid_color = (0.25, 0.25, 0.25, 0.6)  # 主网格线颜色
        grid_major_color = (0.35, 0.35, 0.35, 0.9)  # 粗网格线颜色
        origin_color = (0.15, 0.15, 0.15, 1.0)  # 原点线颜色

        # 渲染普通网格线
        glLineWidth(1.0)
        glBegin(GL_LINES)
        glColor4f(*grid_color)
        for i in range(-grid_size, grid_size + 1):
            if i % 5 != 0 and i != 0:  # 跳过粗网格线和原点
                # X方向线
                glVertex3f(i * grid_step, 0.0, -grid_size * grid_step)
                glVertex3f(i * grid_step, 0.0, grid_size * grid_step)
                # Z方向线
                glVertex3f(-grid_size * grid_step, 0.0, i * grid_step)
                glVertex3f(grid_size * grid_step, 0.0, i * grid_step)
        glEnd()

        # 渲染粗网格线（每5格）
        glLineWidth(1.5)
        glBegin(GL_LINES)
        glColor4f(*grid_major_color)
        for i in range(-grid_size, grid_size + 1):
            if i % 5 == 0 and i != 0:  # 跳过原点
                # X方向线
                glVertex3f(i * grid_step, 0.0, -grid_size * grid_step)
                glVertex3f(i * grid_step, 0.0, grid_size * grid_step)
                # Z方向线
                glVertex3f(-grid_size * grid_step, 0.0, i * grid_step)
                glVertex3f(grid_size * grid_step, 0.0, i * grid_step)
        glEnd()

        # 渲染原点线
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glColor4f(*origin_color)
        # X方向原点线
        glVertex3f(0.0, 0.0, -grid_size * grid_step)
        glVertex3f(0.0, 0.0, grid_size * grid_step)
        # Z方向原点线
        glVertex3f(-grid_size * grid_step, 0.0, 0.0)
        glVertex3f(grid_size * grid_step, 0.0, 0.0)
        glEnd()

        glLineWidth(1.0)  # 恢复默认线宽
        glEnable(GL_LIGHTING)

    def _render_axes(self):
        """渲染坐标轴"""
        glDisable(GL_LIGHTING)
        glEnable(GL_LINE_SMOOTH)
        glLineWidth(4.0)  # 更粗的坐标轴
        glBegin(GL_LINES)

        axis_length = 2.0  # 更长的坐标轴

        # X轴 - 红色（更鲜艳）
        glColor3f(1.0, 0.2, 0.2)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(axis_length, 0.0, 0.0)

        # Y轴 - 绿色（更鲜艳）
        glColor3f(0.2, 1.0, 0.2)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, axis_length, 0.0)

        # Z轴 - 蓝色（更鲜艳）
        glColor3f(0.2, 0.2, 1.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, axis_length)

        glEnd()
        glLineWidth(1.0)  # 恢复默认线宽
        glEnable(GL_LIGHTING)

    def _render_scene(self):
        """渲染场景对象"""
        # 如果有自定义网格，渲染自定义网格
        if self.custom_mesh:
            self._render_custom_mesh(self.custom_mesh)
        else:
            # 否则渲染默认的示例立方体
            glPushMatrix()
            glTranslatef(0.0, 0.0, 0.0)  # 立方体中心在世界原点 (0, 0, 0)

            # 设置材质颜色（蓝灰色）
            mat_ambient = [0.2, 0.2, 0.3, 1.0]
            mat_diffuse = [0.5, 0.5, 0.7, 1.0]
            mat_specular = [0.8, 0.8, 0.8, 1.0]
            mat_shininess = [50.0]

            glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient)
            glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse)
            glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular)
            glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess)

            # 渲染实心立方体
            self._render_solid_cube(1.0)

            glPopMatrix()

        # 如果引擎有场景管理器，渲染场景中的对象
        if self.engine and hasattr(self.engine, 'scene_mgr'):
            try:
                # TODO: 遍历场景节点并渲染
                pass
            except Exception as e:
                self.logger.error(f"渲染场景对象失败: {e}")

    def _render_custom_mesh(self, mesh):
        """渲染自定义导入的网格"""
        glPushMatrix()

        # 根据着色模式设置渲染状态
        if self.shading_mode == 'solid':
            # 实体模式 - 简单的纯色着色，无光照
            glDisable(GL_LIGHTING)
            glColor3f(0.7, 0.7, 0.7)  # 灰色

        elif self.shading_mode == 'material':
            # 材质模式 - 基础光照
            glEnable(GL_LIGHTING)
            mat_ambient = [0.4, 0.4, 0.4, 1.0]
            mat_diffuse = [0.6, 0.6, 0.6, 1.0]
            mat_specular = [0.2, 0.2, 0.2, 1.0]
            mat_shininess = [10.0]

            glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient)
            glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse)
            glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular)
            glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess)

        else:  # 'rendered' - 完整渲染
            # 渲染模式 - 完整光照和材质
            glEnable(GL_LIGHTING)
            mat_ambient = [0.3, 0.3, 0.3, 1.0]
            mat_diffuse = [0.7, 0.7, 0.7, 1.0]
            mat_specular = [0.5, 0.5, 0.5, 1.0]
            mat_shininess = [32.0]

            glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient)
            glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse)
            glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular)
            glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess)

        # 使用顶点数组渲染
        if mesh.indices:
            # 使用索引渲染
            glBegin(GL_TRIANGLES)
            for i in range(0, len(mesh.indices), 3):
                for j in range(3):
                    idx = mesh.indices[i + j]
                    if self.shading_mode != 'solid' and idx < len(mesh.normals):
                        n = mesh.normals[idx]
                        glNormal3f(n.x, n.y, n.z)
                    if idx < len(mesh.vertices):
                        v = mesh.vertices[idx]
                        glVertex3f(v.x, v.y, v.z)
            glEnd()
        else:
            # 直接渲染顶点
            glBegin(GL_TRIANGLES)
            for i, vertex in enumerate(mesh.vertices):
                if self.shading_mode != 'solid' and i < len(mesh.normals):
                    n = mesh.normals[i]
                    glNormal3f(n.x, n.y, n.z)
                glVertex3f(vertex.x, vertex.y, vertex.z)
            glEnd()

        # 恢复光照状态
        if self.shading_mode == 'solid':
            glEnable(GL_LIGHTING)

        glPopMatrix()

    def _render_solid_cube(self, size):
        """渲染实心立方体（带法线和光照）"""
        s = size / 2.0

        # 根据着色模式设置渲染状态
        if self.shading_mode == 'solid':
            # 实体模式 - 纯色无光照
            glDisable(GL_LIGHTING)
            glColor3f(0.6, 0.6, 0.8)  # 蓝灰色
        elif self.shading_mode == 'material':
            # 材质模式 - 基础光照
            glEnable(GL_LIGHTING)
        else:
            # 渲染模式 - 完整光照
            glEnable(GL_LIGHTING)

        # 立方体的6个面，每个面有正确的法线
        glBegin(GL_QUADS)

        # 前面 (Z+)
        if self.shading_mode != 'solid':
            glNormal3f(0.0, 0.0, 1.0)
        glVertex3f(-s, -s, s)
        glVertex3f(s, -s, s)
        glVertex3f(s, s, s)
        glVertex3f(-s, s, s)

        # 后面 (Z-)
        if self.shading_mode != 'solid':
            glNormal3f(0.0, 0.0, -1.0)
        glVertex3f(-s, -s, -s)
        glVertex3f(-s, s, -s)
        glVertex3f(s, s, -s)
        glVertex3f(s, -s, -s)

        # 顶面 (Y+)
        if self.shading_mode != 'solid':
            glNormal3f(0.0, 1.0, 0.0)
        glVertex3f(-s, s, -s)
        glVertex3f(-s, s, s)
        glVertex3f(s, s, s)
        glVertex3f(s, s, -s)

        # 底面 (Y-)
        if self.shading_mode != 'solid':
            glNormal3f(0.0, -1.0, 0.0)
        glVertex3f(-s, -s, -s)
        glVertex3f(s, -s, -s)
        glVertex3f(s, -s, s)
        glVertex3f(-s, -s, s)

        # 右面 (X+)
        if self.shading_mode != 'solid':
            glNormal3f(1.0, 0.0, 0.0)
        glVertex3f(s, -s, -s)
        glVertex3f(s, s, -s)
        glVertex3f(s, s, s)
        glVertex3f(s, -s, s)

        # 左面 (X-)
        if self.shading_mode != 'solid':
            glNormal3f(-1.0, 0.0, 0.0)
        glVertex3f(-s, -s, -s)
        glVertex3f(-s, -s, s)
        glVertex3f(-s, s, s)
        glVertex3f(-s, s, -s)

        glEnd()

        # 恢复光照状态
        if self.shading_mode == 'solid':
            glEnable(GL_LIGHTING)

    def _render_wireframe_cube(self, size):
        """渲染线框立方体（降级方案）"""
        glDisable(GL_LIGHTING)
        glColor3f(0.8, 0.8, 0.8)

        s = size / 2.0
        vertices = [
            (-s, -s, -s), (s, -s, -s), (s, s, -s), (-s, s, -s),  # 后面
            (-s, -s, s), (s, -s, s), (s, s, s), (-s, s, s)  # 前面
        ]

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # 后面
            (4, 5), (5, 6), (6, 7), (7, 4),  # 前面
            (0, 4), (1, 5), (2, 6), (3, 7)   # 连接线
        ]

        glBegin(GL_LINES)
        for edge in edges:
            for vertex in edge:
                glVertex3fv(vertices[vertex])
        glEnd()

        glEnable(GL_LIGHTING)

    def _bind_events(self):
        """绑定鼠标和键盘事件"""
        # 鼠标事件
        self.bind('<Button-1>', self._on_mouse_down)  # 左键
        self.bind('<Button-2>', self._on_mouse_down)  # 中键
        self.bind('<Button-3>', self._on_mouse_down)  # 右键
        self.bind('<ButtonRelease-1>', self._on_mouse_up)
        self.bind('<ButtonRelease-2>', self._on_mouse_up)
        self.bind('<ButtonRelease-3>', self._on_mouse_up)
        self.bind('<B1-Motion>', self._on_mouse_move)  # 左键拖拽
        self.bind('<B2-Motion>', self._on_mouse_move)  # 中键拖拽
        self.bind('<B3-Motion>', self._on_mouse_move)  # 右键拖拽
        self.bind('<MouseWheel>', self._on_mouse_wheel)  # 滚轮

        # 窗口大小变化
        self.bind('<Configure>', self._on_resize)

    def _on_mouse_down(self, event):
        """鼠标按下事件"""
        self.mouse_down = True
        self.mouse_button = event.num  # 1=左键, 2=中键, 3=右键
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y

    def _on_mouse_up(self, event):
        """鼠标释放事件"""
        self.mouse_down = False
        self.mouse_button = None

    def _on_mouse_move(self, event):
        """鼠标移动事件（拖拽）"""
        if not self.mouse_down:
            return

        dx = event.x - self.last_mouse_x
        dy = event.y - self.last_mouse_y

        # 右键 - 轨道旋转（Blender风格，提高灵敏度）
        if self.mouse_button == 3:
            rotation_speed = 0.3  # 提高旋转速度
            self.camera_rotation_y += dx * rotation_speed
            self.camera_rotation_x += dy * rotation_speed

            # 限制X旋转范围
            self.camera_rotation_x = max(-89.0, min(89.0, self.camera_rotation_x))

        # 中键 - 平移（提高灵敏度）
        elif self.mouse_button == 2:
            pan_speed = 0.02  # 提高平移速度
            self.camera_pan_x += dx * pan_speed
            self.camera_pan_y -= dy * pan_speed

        # 左键 - 对象选择（TODO: 实现射线拾取）
        elif self.mouse_button == 1:
            # 记录拖拽，用于框选
            pass

        self.last_mouse_x = event.x
        self.last_mouse_y = event.y

    def _on_mouse_wheel(self, event):
        """鼠标滚轮事件（缩放）"""
        # Windows: event.delta / 120
        # macOS/Linux: event.num (4=up, 5=down)
        if hasattr(event, 'delta'):
            delta = event.delta / 120
        else:
            delta = 1 if event.num == 4 else -1

        zoom_speed = 0.8  # 提高缩放速度
        self.camera_distance -= delta * zoom_speed
        self.camera_distance = max(2.0, min(50.0, self.camera_distance))  # 调整缩放范围

    def _on_resize(self, event):
        """窗口大小变化事件"""
        if event.width != self.width or event.height != self.height:
            self.width = event.width
            self.height = event.height
            if self.is_initialized:
                glViewport(0, 0, self.width, self.height)
                self.logger.debug(f"视口大小变化: {self.width}x{self.height}")

    # ========== 公共方法 ==========
    def set_view_mode(self, mode):
        """设置视图模式"""
        if mode in ['perspective', 'orthographic']:
            self.view_mode = mode
            self.logger.info(f"视图模式: {mode}")

    def set_camera_preset(self, preset):
        """设置相机预设视图"""
        presets = {
            'top': (90.0, 0.0, 10.0),       # 顶视图
            'bottom': (-90.0, 0.0, 10.0),   # 底视图
            'front': (0.0, 0.0, 10.0),      # 前视图
            'back': (0.0, 180.0, 10.0),     # 后视图
            'right': (0.0, 90.0, 10.0),     # 右视图
            'left': (0.0, -90.0, 10.0),     # 左视图
        }

        if preset in presets:
            self.camera_rotation_x, self.camera_rotation_y, self.camera_distance = presets[preset]
            self.camera_pan_x = 0.0
            self.camera_pan_y = 0.0
            self.logger.info(f"相机预设: {preset}")

    def toggle_grid(self):
        """切换网格显示"""
        self.render_grid = not self.render_grid

    def toggle_axes(self):
        """切换坐标轴显示"""
        self.render_axes = not self.render_axes

    def reset_camera(self):
        """重置相机"""
        self.camera_distance = 10.0
        self.camera_rotation_x = 30.0
        self.camera_rotation_y = 45.0
        self.camera_pan_x = 0.0
        self.camera_pan_y = 0.0

    def set_mesh(self, mesh):
        """
        设置要显示的自定义网格

        Args:
            mesh: Mesh对象
        """
        self.custom_mesh = mesh
        self.logger.info(f"设置自定义网格: {len(mesh.vertices)} 顶点")

        # 自动调整相机距离以适应模型
        if mesh.bounding_box:
            # 计算包围盒大小
            min_pt, max_pt = mesh.bounding_box.get_min_max()
            size_x = max_pt.x - min_pt.x
            size_y = max_pt.y - min_pt.y
            size_z = max_pt.z - min_pt.z

            # 设置相机距离为包围盒对角线长度的1.5倍
            diagonal = (size_x**2 + size_y**2 + size_z**2)**0.5
            self.camera_distance = max(diagonal * 1.5, 2.0)  # 最小距离为2.0

            self.logger.info(f"自动调整相机距离: {self.camera_distance:.2f}")
        else:
            # 如果没有包围盒，使用默认距离
            self.camera_distance = 10.0

    def clear_mesh(self):
        """清除自定义网格，恢复显示默认立方体"""
        self.custom_mesh = None
        self.logger.info("清除自定义网格")

    def set_shading_mode(self, mode):
        """
        设置着色/渲染模式

        Args:
            mode: 着色模式 ('solid', 'material', 'rendered')
        """
        if mode in ['solid', 'material', 'rendered']:
            self.shading_mode = mode
            mode_names = {
                'solid': '实体着色（无光照）',
                'material': '材质着色（基础光照）',
                'rendered': '完整渲染（高级光照）'
            }
            self.logger.info(f"着色模式切换到: {mode_names[mode]}")
        else:
            self.logger.warning(f"未知的着色模式: {mode}")


# 测试代码
if __name__ == "__main__":
    root = tk.Tk()
    root.title("OpenGL Viewport Test")
    root.geometry("800x600")

    DarkTheme.apply(root)

    # 模拟引擎
    class MockEngine:
        pass

    engine = MockEngine()
    viewport = OpenGLViewport(root, engine)
    viewport.pack(fill=tk.BOTH, expand=True)

    root.mainloop()
