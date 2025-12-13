# -*- coding: utf-8 -*-
"""
变换操纵器控件，用于在3D场景中平移、旋转和缩放对象
"""

from Engine.Math import Vector3, Matrix4x4, Quaternion
from Engine.UI.Controls.Control import Control
from Engine.UI.Event import EventType

class TransformManipulator(Control):
    """变换操纵器控件，用于在3D场景中平移、旋转和缩放对象"""
    
    # 变换模式枚举
    class Mode:
        TRANSLATE = 0
        ROTATE = 1
        SCALE = 2
    
    # 轴枚举
    class Axis:
        NONE = -1
        X = 0
        Y = 1
        Z = 2
        ALL = 3
    
    def __init__(self, camera):
        """初始化变换操纵器
        
        Args:
            camera: 相机对象，用于投影和视图变换
        """
        # 变换操纵器不需要传统的2D位置和大小，所以使用默认值
        super().__init__(0, 0, 0, 0)
        self.name = "TransformManipulator"
        self.camera = camera
        self.mode = self.Mode.TRANSLATE
        self.active_axis = self.Axis.NONE
        self.is_dragging = False
        self.start_mouse_pos = (0, 0)
        self.start_object_pos = Vector3(0, 0, 0)
        self.start_object_rot = Quaternion.identity()
        self.start_object_scale = Vector3(1, 1, 1)
        self.selected_node = None
        self.manipulator_size = 1.0
        
        # 颜色设置
        self.axis_colors = {
            self.Axis.X: (1.0, 0.0, 0.0, 1.0),  # 红色 - X轴
            self.Axis.Y: (0.0, 1.0, 0.0, 1.0),  # 绿色 - Y轴
            self.Axis.Z: (0.0, 0.0, 1.0, 1.0),  # 蓝色 - Z轴
            self.Axis.ALL: (0.5, 0.5, 0.5, 1.0)   # 灰色 - 所有轴
        }
        
        # 高亮颜色
        self.highlight_colors = {
            self.Axis.X: (1.0, 0.5, 0.5, 1.0),  # 亮红色 - X轴
            self.Axis.Y: (0.5, 1.0, 0.5, 1.0),  # 亮绿色 - Y轴
            self.Axis.Z: (0.5, 0.5, 1.0, 1.0),  # 亮蓝色 - Z轴
            self.Axis.ALL: (0.8, 0.8, 0.8, 1.0)   # 亮灰色 - 所有轴
        }
    
    def set_selected_node(self, node):
        """设置当前选中的节点
        
        Args:
            node: 选中的场景节点
        """
        self.selected_node = node
    
    def set_mode(self, mode):
        """设置变换模式
        
        Args:
            mode: 变换模式，取值为Mode.TRANSLATE, Mode.ROTATE, Mode.SCALE
        """
        self.mode = mode
    
    def get_mode(self):
        """获取当前变换模式
        
        Returns:
            int: 当前变换模式
        """
        return self.mode
    
    def toggle_mode(self):
        """切换变换模式"""
        self.mode = (self.mode + 1) % 3
    
    def _render_content(self):
        """渲染变换操纵器内容"""
        if not self.selected_node:
            return
        
        # 获取选中节点的世界位置
        pos = self.selected_node.world_position
        
        # 计算操纵器大小（根据相机距离自动调整）
        camera_pos = self.camera.position
        distance = (camera_pos - pos).length()
        self.manipulator_size = distance * 0.1  # 操纵器大小与距离成正比
        
        # 渲染操纵器
        if self.mode == self.Mode.TRANSLATE:
            self._render_translate_manipulator(pos)
        elif self.mode == self.Mode.ROTATE:
            self._render_rotate_manipulator(pos)
        elif self.mode == self.Mode.SCALE:
            self._render_scale_manipulator(pos)
    
    def _render_translate_manipulator(self, pos):
        """渲染平移操纵器"""
        from OpenGL.GL import glBegin, glEnd, glColor4f, glVertex3f, GL_LINES, glLineWidth
        
        # 设置线宽
        glLineWidth(3.0)
        
        # 渲染X轴
        color = self.highlight_colors[self.Axis.X] if self.active_axis == self.Axis.X else self.axis_colors[self.Axis.X]
        glColor4f(*color)
        glBegin(GL_LINES)
        glVertex3f(pos.x, pos.y, pos.z)
        glVertex3f(pos.x + self.manipulator_size, pos.y, pos.z)
        glEnd()
        
        # 渲染Y轴
        color = self.highlight_colors[self.Axis.Y] if self.active_axis == self.Axis.Y else self.axis_colors[self.Axis.Y]
        glColor4f(*color)
        glBegin(GL_LINES)
        glVertex3f(pos.x, pos.y, pos.z)
        glVertex3f(pos.x, pos.y + self.manipulator_size, pos.z)
        glEnd()
        
        # 渲染Z轴
        color = self.highlight_colors[self.Axis.Z] if self.active_axis == self.Axis.Z else self.axis_colors[self.Axis.Z]
        glColor4f(*color)
        glBegin(GL_LINES)
        glVertex3f(pos.x, pos.y, pos.z)
        glVertex3f(pos.x, pos.y, pos.z + self.manipulator_size)
        glEnd()
        
        # 恢复默认线宽
        glLineWidth(1.0)
    
    def _render_rotate_manipulator(self, pos):
        """渲染旋转操纵器"""
        from OpenGL.GL import glBegin, glEnd, glColor4f, glVertex3f, GL_LINE_LOOP, glLineWidth
        import math
        
        # 设置线宽
        glLineWidth(3.0)
        
        segments = 32
        radius = self.manipulator_size
        
        # 渲染X轴旋转环
        color = self.highlight_colors[self.Axis.X] if self.active_axis == self.Axis.X else self.axis_colors[self.Axis.X]
        glColor4f(*color)
        glBegin(GL_LINE_LOOP)
        for i in range(segments):
            angle = i * (2 * math.pi / segments)
            x = pos.x
            y = pos.y + radius * math.cos(angle)
            z = pos.z + radius * math.sin(angle)
            glVertex3f(x, y, z)
        glEnd()
        
        # 渲染Y轴旋转环
        color = self.highlight_colors[self.Axis.Y] if self.active_axis == self.Axis.Y else self.axis_colors[self.Axis.Y]
        glColor4f(*color)
        glBegin(GL_LINE_LOOP)
        for i in range(segments):
            angle = i * (2 * math.pi / segments)
            x = pos.x + radius * math.cos(angle)
            y = pos.y
            z = pos.z + radius * math.sin(angle)
            glVertex3f(x, y, z)
        glEnd()
        
        # 渲染Z轴旋转环
        color = self.highlight_colors[self.Axis.Z] if self.active_axis == self.Axis.Z else self.axis_colors[self.Axis.Z]
        glColor4f(*color)
        glBegin(GL_LINE_LOOP)
        for i in range(segments):
            angle = i * (2 * math.pi / segments)
            x = pos.x + radius * math.cos(angle)
            y = pos.y + radius * math.sin(angle)
            z = pos.z
            glVertex3f(x, y, z)
        glEnd()
        
        # 恢复默认线宽
        glLineWidth(1.0)
    
    def _render_scale_manipulator(self, pos):
        """渲染缩放操纵器"""
        from OpenGL.GL import glBegin, glEnd, glColor4f, glVertex3f, GL_LINES, glLineWidth
        
        # 设置线宽
        glLineWidth(3.0)
        
        # 渲染X轴
        color = self.highlight_colors[self.Axis.X] if self.active_axis == self.Axis.X else self.axis_colors[self.Axis.X]
        glColor4f(*color)
        glBegin(GL_LINES)
        glVertex3f(pos.x, pos.y, pos.z)
        glVertex3f(pos.x + self.manipulator_size, pos.y, pos.z)
        glEnd()
        
        # 渲染Y轴
        color = self.highlight_colors[self.Axis.Y] if self.active_axis == self.Axis.Y else self.axis_colors[self.Axis.Y]
        glColor4f(*color)
        glBegin(GL_LINES)
        glVertex3f(pos.x, pos.y, pos.z)
        glVertex3f(pos.x, pos.y + self.manipulator_size, pos.z)
        glEnd()
        
        # 渲染Z轴
        color = self.highlight_colors[self.Axis.Z] if self.active_axis == self.Axis.Z else self.axis_colors[self.Axis.Z]
        glColor4f(*color)
        glBegin(GL_LINES)
        glVertex3f(pos.x, pos.y, pos.z)
        glVertex3f(pos.x, pos.y, pos.z + self.manipulator_size)
        glEnd()
        
        # 恢复默认线宽
        glLineWidth(1.0)
    
    def on_mouse_down(self, button):
        """处理鼠标按下事件
        
        Args:
            button: 鼠标按钮（0: 左键, 1: 中键, 2: 右键）
        """
        if button == 0 and self.selected_node:
            # 检测鼠标是否点击了操纵器轴
            self.active_axis = self._get_hovered_axis()
            if self.active_axis != self.Axis.NONE:
                self.is_dragging = True
                self.start_mouse_pos = self.parent.mouse_position
                self.start_object_pos = self.selected_node.position.copy()
                self.start_object_rot = self.selected_node.rotation.copy()
                self.start_object_scale = self.selected_node.scale.copy()
                return True
        return False
    
    def on_mouse_up(self, button):
        """处理鼠标释放事件
        
        Args:
            button: 鼠标按钮（0: 左键, 1: 中键, 2: 右键）
        """
        if button == 0:
            self.is_dragging = False
            self.active_axis = self.Axis.NONE
            return True
        return False
    
    def on_mouse_move(self, x, y):
        """处理鼠标移动事件
        
        Args:
            x: 鼠标X坐标
            y: 鼠标Y坐标
        """
        if self.is_dragging and self.selected_node:
            dx = x - self.start_mouse_pos[0]
            dy = y - self.start_mouse_pos[1]
            
            if self.mode == self.Mode.TRANSLATE:
                self._handle_translate(dx, dy)
            elif self.mode == self.Mode.ROTATE:
                self._handle_rotate(dx, dy)
            elif self.mode == self.Mode.SCALE:
                self._handle_scale(dx, dy)
            
            return True
        else:
            # 检测鼠标悬停的轴
            self.active_axis = self._get_hovered_axis()
        
        return False
    
    def _get_hovered_axis(self):
        """获取鼠标悬停的轴
        
        Returns:
            int: 悬停的轴，取值为Axis.X, Axis.Y, Axis.Z, Axis.ALL, Axis.NONE
        """
        if not self.selected_node:
            return self.Axis.NONE
        
        # 获取鼠标位置和相机信息
        mouse_x, mouse_y = self.parent.mouse_position
        viewport_width = self.camera.viewport_width
        viewport_height = self.camera.viewport_height
        
        # 创建射线
        ray_origin, ray_direction = self.camera.screen_to_ray(mouse_x, mouse_y)
        
        # 获取操纵器位置和大小
        manipulator_pos = self.selected_node.world_position
        manipulator_size = self.manipulator_size
        
        # 检测与各轴的相交
        axis_threshold = 0.05 * manipulator_size  # 轴的检测阈值
        
        # 检测X轴
        x_axis_start = manipulator_pos
        x_axis_end = manipulator_pos + Vector3(manipulator_size, 0, 0)
        if self._ray_intersects_line(ray_origin, ray_direction, x_axis_start, x_axis_end, axis_threshold):
            return self.Axis.X
        
        # 检测Y轴
        y_axis_start = manipulator_pos
        y_axis_end = manipulator_pos + Vector3(0, manipulator_size, 0)
        if self._ray_intersects_line(ray_origin, ray_direction, y_axis_start, y_axis_end, axis_threshold):
            return self.Axis.Y
        
        # 检测Z轴
        z_axis_start = manipulator_pos
        z_axis_end = manipulator_pos + Vector3(0, 0, manipulator_size)
        if self._ray_intersects_line(ray_origin, ray_direction, z_axis_start, z_axis_end, axis_threshold):
            return self.Axis.Z
        
        return self.Axis.NONE
    
    def _ray_intersects_line(self, ray_origin, ray_direction, line_start, line_end, threshold):
        """检测射线是否与线段相交
        
        Args:
            ray_origin: 射线原点
            ray_direction: 射线方向（归一化）
            line_start: 线段起点
            line_end: 线段终点
            threshold: 检测阈值
            
        Returns:
            bool: 射线是否与线段相交
        """
        # 计算线段向量
        line_vec = line_end - line_start
        line_length = line_vec.length()
        line_dir = line_vec.normalized()
        
        # 计算射线与线段所在直线的最短距离
        v1 = ray_origin - line_start
        v2 = line_dir
        v3 = Vector3.cross(ray_direction, v2)
        
        # 如果v3的长度为0，射线与线段平行
        v3_length = v3.length()
        if v3_length < 1e-6:
            return False
        
        # 计算射线与线段所在直线的交点
        t1 = Vector3.cross(v1, v2).dot(v3) / (v3_length * v3_length)
        t2 = v1.dot(v3) / (v3_length * v3_length)
        
        # 检查交点是否在线段上和射线上
        if t1 >= 0.0 and t2 >= 0.0 and t2 <= line_length:
            # 计算交点到射线原点的距离
            intersection_point = ray_origin + ray_direction * t1
            distance = (intersection_point - ray_origin).length()
            
            # 检查距离是否在合理范围内
            if distance < 1000.0:  # 最大检测距离
                # 计算交点到线段的最短距离
                closest_point_on_line = line_start + line_dir * t2
                distance_to_line = (intersection_point - closest_point_on_line).length()
                
                # 如果距离小于阈值，认为相交
                return distance_to_line < threshold
        
        return False
    
    def _handle_translate(self, dx, dy):
        """处理平移操作
        
        Args:
            dx: 鼠标X方向移动距离
            dy: 鼠标Y方向移动距离
        """
        if not self.selected_node:
            return
        
        # 计算平移速度
        translate_speed = 0.001 * self.manipulator_size
        
        # 根据活动轴计算平移向量
        if self.active_axis == self.Axis.X:
            # 沿X轴平移
            translation = Vector3(dx * translate_speed, 0, 0)
        elif self.active_axis == self.Axis.Y:
            # 沿Y轴平移
            translation = Vector3(0, -dy * translate_speed, 0)  # 反转Y轴，因为屏幕Y轴向下
        elif self.active_axis == self.Axis.Z:
            # 沿Z轴平移
            # Z轴平移需要考虑相机方向
            camera_forward = self.camera.get_forward()
            camera_forward.y = 0  # 只在水平面上平移
            camera_forward.normalize()
            translation = camera_forward * (dx * translate_speed)
        else:
            # 自由平移
            # 计算相机的右向和上向向量
            camera_right = self.camera.get_right()
            camera_up = self.camera.get_up()
            
            # 计算平移向量
            translation = camera_right * (dx * translate_speed) + camera_up * (-dy * translate_speed)
        
        # 应用平移
        self.selected_node.set_position(self.start_object_pos + translation)
    
    def _handle_rotate(self, dx, dy):
        """处理旋转操作
        
        Args:
            dx: 鼠标X方向移动距离
            dy: 鼠标Y方向移动距离
        """
        if not self.selected_node:
            return
        
        # 计算旋转速度
        rotate_speed = 0.01
        
        # 根据活动轴计算旋转角度
        if self.active_axis == self.Axis.X:
            # 绕X轴旋转
            rotation = Quaternion.from_euler(dy * rotate_speed, 0, 0)
        elif self.active_axis == self.Axis.Y:
            # 绕Y轴旋转
            rotation = Quaternion.from_euler(0, -dx * rotate_speed, 0)  # 反转X轴，因为屏幕X轴向右
        elif self.active_axis == self.Axis.Z:
            # 绕Z轴旋转
            rotation = Quaternion.from_euler(0, 0, -dx * rotate_speed)  # 反转X轴
        else:
            # 自由旋转
            # 计算旋转角度
            yaw = -dx * rotate_speed
            pitch = dy * rotate_speed
            
            # 创建旋转四元数
            rotation = Quaternion.from_euler(pitch, yaw, 0)
        
        # 应用旋转
        self.selected_node.set_rotation(self.start_object_rot * rotation)
    
    def _handle_scale(self, dx, dy):
        """处理缩放操作
        
        Args:
            dx: 鼠标X方向移动距离
            dy: 鼠标Y方向移动距离
        """
        if not self.selected_node:
            return
        
        # 计算缩放速度
        scale_speed = 0.01
        
        # 计算缩放因子
        scale_factor = 1.0
        if self.active_axis == self.Axis.X:
            # 沿X轴缩放
            scale_factor = 1.0 + (dx * scale_speed)
            new_scale = Vector3(self.start_object_scale.x * scale_factor, self.start_object_scale.y, self.start_object_scale.z)
        elif self.active_axis == self.Axis.Y:
            # 沿Y轴缩放
            scale_factor = 1.0 - (dy * scale_speed)  # 反转Y轴
            new_scale = Vector3(self.start_object_scale.x, self.start_object_scale.y * scale_factor, self.start_object_scale.z)
        elif self.active_axis == self.Axis.Z:
            # 沿Z轴缩放
            scale_factor = 1.0 + (dx * scale_speed)
            new_scale = Vector3(self.start_object_scale.x, self.start_object_scale.y, self.start_object_scale.z * scale_factor)
        else:
            # 均匀缩放
            scale_factor = 1.0 + (dx * scale_speed) - (dy * scale_speed)  # 综合X和Y方向的移动
            new_scale = self.start_object_scale * scale_factor
        
        # 应用缩放
        self.selected_node.set_scale(new_scale)
    
    def contains_point(self, x, y):
        """检查点是否在控件内
        
        Args:
            x: 点的X坐标
            y: 点的Y坐标
            
        Returns:
            bool: 点是否在控件内
        """
        # 变换操纵器的碰撞检测比较特殊，需要通过射线检测实现
        # 这里简化处理，返回False，让父类处理事件
        return False
