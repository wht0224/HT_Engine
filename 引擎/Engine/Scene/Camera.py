# -*- coding: utf-8 -*-
"""
低端GPU渲染引擎相机系统
实现高性能相机类，支持透视和正交投影，针对低端GPU优化
"""

import numpy as np
from enum import Enum

# 确保正确导入Math模块
from Engine.Math import Vector3, Quaternion, Matrix4x4, Frustum


class CameraProjectionType(Enum):
    """相机投影类型枚举"""
    PERSPECTIVE = 0  # 透视投影
    ORTHOGRAPHIC = 1  # 正交投影


class Camera:
    """相机类
    针对低端GPU优化的高性能相机，支持透视和正交投影
    """
    
    def __init__(self, name="MainCamera"):
        """初始化相机
        
        Args:
            name: 相机名称
        """
        # 相机标识
        self.name = name
        
        # 投影参数
        self.projection_type = CameraProjectionType.PERSPECTIVE
        self.field_of_view = 60.0  # 视角（度）
        self.aspect_ratio = 16.0 / 9.0  # 宽高比
        self.near_plane = 0.1  # 近裁剪面
        self.far_plane = 1000.0  # 远裁剪面
        self.ortho_size = 5.0  # 正交投影大小
        
        # 视图参数
        self.position = Vector3(0, 0, 0)
        self.rotation = Quaternion.identity()
        self.target = Vector3(0, 0, -1)  # 观察目标点
        self.up = Vector3(0, 1, 0)  # 上方向
        
        # 缓存矩阵
        self.view_matrix = Matrix4x4.identity()
        self.projection_matrix = Matrix4x4.identity()
        self.view_projection_matrix = Matrix4x4.identity()
        self.inverse_view_matrix = Matrix4x4.identity()
        self.inverse_projection_matrix = Matrix4x4.identity()
        
        # 脏标记
        self.view_dirty = True
        self.projection_dirty = True
        
        # 视锥体
        self.frustum = Frustum()
        self.frustum_dirty = True
        
        # 相机属性
        self.enable_backface_culling = True
        self.enable_depth_test = True
        self.clear_color = (0.0, 0.0, 0.0, 1.0)  # RGBA
        self.clear_depth = 1.0
        self.clear_stencil = 0
        
        # 性能优化参数
        self.use_frustum_culling = True  # 是否使用视锥体剔除
        self.use_occlusion_culling = False  # 是否使用遮挡剔除（低端GPU通常关闭以节省性能）
        self.max_render_distance = 500.0  # 最大渲染距离
        
        # 后处理标志
        self.enable_hdr = False  # 高端功能，低端GPU默认关闭
        self.enable_taa = False  # 临时抗锯齿
        self.enable_bloom = False  # 泛光效果
        
        # 曝光控制
        self.exposure = 1.0
        
        # 动态分辨率缩放支持
        self.dynamic_resolution_scale = 1.0  # 1.0表示原始分辨率
        self.min_dynamic_scale = 0.5  # 最小动态缩放比例
        self.max_dynamic_scale = 1.0  # 最大动态缩放比例
        
        # 视口设置
        self.viewport_x = 0
        self.viewport_y = 0
        self.viewport_width = 1.0
        self.viewport_height = 1.0
        
        # 变换引用
        self.transform_reference = None
        
        # LOD相关参数
        self.lod_bias = 1.0  # LOD偏移因子，>1.0使LOD切换更远
        self.lod_distances = [10.0, 25.0, 50.0, 100.0]  # 默认LOD距离
    
    def set_transform_reference(self, scene_node):
        """设置变换引用
        
        Args:
            scene_node: 场景节点，用于获取变换信息
        """
        self.transform_reference = scene_node
    
    def set_projection_type(self, projection_type):
        """设置投影类型
        
        Args:
            projection_type: 投影类型（透视或正交）
        """
        self.projection_type = projection_type
        self.projection_dirty = True
        self.frustum_dirty = True
    
    def set_perspective(self, fov, aspect_ratio, near_plane, far_plane):
        """设置透视投影
        
        Args:
            fov: 视角（度）
            aspect_ratio: 宽高比
            near_plane: 近裁剪面
            far_plane: 远裁剪面
        """
        self.projection_type = CameraProjectionType.PERSPECTIVE
        self.field_of_view = fov
        self.aspect_ratio = aspect_ratio
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.projection_dirty = True
        self.frustum_dirty = True
    
    def set_orthographic(self, ortho_size, aspect_ratio, near_plane, far_plane):
        """设置正交投影
        
        Args:
            ortho_size: 正交投影大小
            aspect_ratio: 宽高比
            near_plane: 近裁剪面
            far_plane: 远裁剪面
        """
        self.projection_type = CameraProjectionType.ORTHOGRAPHIC
        self.ortho_size = ortho_size
        self.aspect_ratio = aspect_ratio
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.projection_dirty = True
        self.frustum_dirty = True
    
    def set_fov(self, fov):
        """设置视角
        
        Args:
            fov: 视角（度）
        """
        self.field_of_view = fov
        if self.projection_type == CameraProjectionType.PERSPECTIVE:
            self.projection_dirty = True
            self.frustum_dirty = True
    
    def set_aspect_ratio(self, aspect_ratio):
        """设置宽高比
        
        Args:
            aspect_ratio: 宽高比
        """
        self.aspect_ratio = aspect_ratio
        self.projection_dirty = True
        self.frustum_dirty = True
    
    def set_near_far_planes(self, near_plane, far_plane):
        """设置近远裁剪面
        
        Args:
            near_plane: 近裁剪面
            far_plane: 远裁剪面
        """
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.projection_dirty = True
        self.frustum_dirty = True
    
    def set_position(self, position):
        """
        设置相机位置
        
        Args:
            position: 相机位置
        """
        self.position = position.copy()
        self.view_dirty = True
        self.frustum_dirty = True
    
    def get_position(self):
        """
        获取相机位置
        
        Returns:
            Vector3: 相机位置
        """
        return self.position.copy()
    
    def move_forward(self, distance):
        """
        向前移动相机
        
        Args:
            distance: 移动距离
        """
        forward = self.get_forward()
        self.position += forward * distance
        self.view_dirty = True
        self.frustum_dirty = True
    
    def move_right(self, distance):
        """
        向右移动相机
        
        Args:
            distance: 移动距离
        """
        right = self.get_right()
        self.position += right * distance
        self.view_dirty = True
        self.frustum_dirty = True
    
    def move_up(self, distance):
        """
        向上移动相机
        
        Args:
            distance: 移动距离
        """
        up = self.get_up()
        self.position += up * distance
        self.view_dirty = True
        self.frustum_dirty = True
    
    def rotate_yaw(self, angle):
        """
        绕Y轴旋转相机
        
        Args:
            angle: 旋转角度（弧度）
        """
        rotation = Quaternion.from_euler(0, angle, 0)
        self.rotation = rotation * self.rotation
        self._update_direction_vectors()
        self.view_dirty = True
        self.frustum_dirty = True
    
    def rotate_pitch(self, angle):
        """
        绕X轴旋转相机
        
        Args:
            angle: 旋转角度（弧度）
        """
        rotation = Quaternion.from_euler(angle, 0, 0)
        self.rotation = rotation * self.rotation
        self._update_direction_vectors()
        self.view_dirty = True
        self.frustum_dirty = True
    
    def rotate_roll(self, angle):
        """
        绕Z轴旋转相机
        
        Args:
            angle: 旋转角度（弧度）
        """
        rotation = Quaternion.from_euler(0, 0, angle)
        self.rotation = rotation * self.rotation
        self._update_direction_vectors()
        self.view_dirty = True
        self.frustum_dirty = True
    
    def zoom(self, factor):
        """
        缩放相机（调整FOV）
        
        Args:
            factor: 缩放因子
        """
        self.field_of_view = max(10.0, min(120.0, self.field_of_view * factor))
        self.projection_dirty = True
        self.frustum_dirty = True
    
    def set_rotation(self, rotation):
        """设置相机旋转
        
        Args:
            rotation: 相机旋转（四元数）
        """
        self.rotation = rotation.copy()
        self._update_direction_vectors()
        self.view_dirty = True
        self.frustum_dirty = True
    
    def set_rotation_euler(self, euler_angles):
        """设置相机旋转（欧拉角）
        
        Args:
            euler_angles: 欧拉角（XYZ，弧度）
        """
        self.rotation = Quaternion.from_euler(euler_angles.x, euler_angles.y, euler_angles.z)
        self._update_direction_vectors()
        self.view_dirty = True
        self.frustum_dirty = True
    
    def set_target(self, target):
        """设置观察目标
        
        Args:
            target: 目标位置
        """
        self.target = target.copy()
        self._update_rotation_from_look_at()
        self.view_dirty = True
        self.frustum_dirty = True
    
    def set_up(self, up):
        """设置上方向
        
        Args:
            up: 上方向向量
        """
        self.up = up.copy()
        self.view_dirty = True
        self.frustum_dirty = True
    
    def look_at(self, target, up=None):
        """使相机朝向目标
        
        Args:
            target: 目标位置
            up: 上方向（可选）
        """
        self.target = target.copy()
        if up:
            self.up = up.copy()
        self._update_rotation_from_look_at()
        self.view_dirty = True
        self.frustum_dirty = True
    
    def _update_rotation_from_look_at(self):
        """根据look_at目标更新旋转"""
        direction = (self.target - self.position).normalized()
        
        # 计算旋转矩阵
        rot_matrix = Matrix4x4.create_look_at(self.position, self.target, self.up)
        
        # 转换为四元数
        self.rotation = Quaternion.from_matrix(rot_matrix)
        
        # 更新方向向量
        self._update_direction_vectors()
    
    def _update_direction_vectors(self):
        """更新方向向量（前、右、上）"""
        # 计算前方向（默认前方向为-Z轴）
        forward = Vector3(0, 0, -1)
        forward = self.rotation * forward
        
        # 更新目标点
        self.target = self.position + forward
        
        # 计算右方向（前方向叉乘上方向）
        right = forward.cross(self.up)
        right.normalize()
        
        # 重新计算上方向（右方向叉乘前方向）以确保正交
        self.up = right.cross(forward)
        self.up.normalize()
    
    def get_forward(self):
        """获取前方向
        
        Returns:
            Vector3: 前方向向量
        """
        # 计算前方向（默认前方向为-Z轴）
        forward = Vector3(0, 0, -1)
        forward = self.rotation * forward
        return forward
    
    def get_right(self):
        """获取右方向
        
        Returns:
            Vector3: 右方向向量
        """
        forward = self.get_forward()
        right = forward.cross(self.up)
        right.normalize()
        return right
    
    def get_up(self):
        """获取上方向
        
        Returns:
            Vector3: 上方向向量
        """
        return self.up.copy()
    
    def update(self, delta_time):
        """更新相机
        
        Args:
            delta_time: 帧时间（秒）
        """
        # 如果有变换引用，从引用更新位置和旋转
        if self.transform_reference:
            self.position = self.transform_reference.get_world_position()
            self.rotation = self.transform_reference.get_world_rotation()
            self._update_direction_vectors()
            self.view_dirty = True
            self.frustum_dirty = True
        
        # 更新视图矩阵
        if self.view_dirty:
            self._update_view_matrix()
        
        # 更新投影矩阵
        if self.projection_dirty:
            self._update_projection_matrix()
        
        # 更新视锥体
        if self.frustum_dirty:
            self._update_frustum()
    
    def _update_view_matrix(self):
        """更新视图矩阵"""
        # 从位置和旋转计算视图矩阵
        # 视图矩阵 = 逆旋转矩阵 * 逆平移矩阵
        
        # 计算旋转矩阵
        rot_matrix = self.rotation.to_matrix()
        
        # 计算平移矩阵的逆（即负平移）
        inv_translation = Matrix4x4.translation(-self.position.x, -self.position.y, -self.position.z)
        
        # 视图矩阵 = 旋转矩阵的转置（因为旋转的逆等于转置） * 逆平移
        self.view_matrix = rot_matrix.transpose() * inv_translation
        
        # 计算逆视图矩阵（世界矩阵）
        self.inverse_view_matrix = Matrix4x4.translation(self.position.x, self.position.y, self.position.z) * rot_matrix
        
        # 清除脏标记
        self.view_dirty = False
    
    def _update_projection_matrix(self):
        """更新投影矩阵"""
        if self.projection_type == CameraProjectionType.PERSPECTIVE:
            # 透视投影
            fov_rad = np.radians(self.field_of_view)
            tan_half_fov = np.tan(fov_rad / 2.0)
            
            # 计算透视投影矩阵
            self.projection_matrix = Matrix4x4.identity()
            
            # 透视矩阵系数
            self.projection_matrix[0, 0] = 1.0 / (self.aspect_ratio * tan_half_fov)
            self.projection_matrix[1, 1] = 1.0 / tan_half_fov
            
            # 深度范围映射
            # 标准OpenGL投影矩阵，将z从[-near, -far]映射到[-1, 1]
            self.projection_matrix[2, 2] = -(self.far_plane + self.near_plane) / (self.far_plane - self.near_plane)
            self.projection_matrix[2, 3] = -2.0 * self.far_plane * self.near_plane / (self.far_plane - self.near_plane)
            
            # 透视除法
            self.projection_matrix[3, 2] = -1.0
            self.projection_matrix[3, 3] = 0.0
        else:
            # 正交投影
            # 计算正交投影的左右上下边界
            right = self.ortho_size * self.aspect_ratio / 2.0
            left = -right
            top = self.ortho_size / 2.0
            bottom = -top
            
            # 计算正交投影矩阵
            self.projection_matrix = Matrix4x4.identity()
            
            # 缩放系数
            self.projection_matrix[0, 0] = 2.0 / (right - left)
            self.projection_matrix[1, 1] = 2.0 / (top - bottom)
            self.projection_matrix[2, 2] = -2.0 / (self.far_plane - self.near_plane)
            
            # 平移系数
            self.projection_matrix[0, 3] = -(right + left) / (right - left)
            self.projection_matrix[1, 3] = -(top + bottom) / (top - bottom)
            self.projection_matrix[2, 3] = -(self.far_plane + self.near_plane) / (self.far_plane - self.near_plane)
        
        # 计算逆投影矩阵
        self.inverse_projection_matrix = self.projection_matrix.inverse()
        
        # 清除脏标记
        self.projection_dirty = False
    
    def _update_frustum(self):
        """更新视锥体
        视锥体用于视锥体剔除
        """
        # 确保视图和投影矩阵已更新
        if self.view_dirty:
            self._update_view_matrix()
        if self.projection_dirty:
            self._update_projection_matrix()
        
        # 计算视图投影矩阵
        self.view_projection_matrix = self.projection_matrix * self.view_matrix
        
        # 从视图投影矩阵提取视锥体平面
        self.frustum.extract_from_matrix(self.view_projection_matrix)
        
        # 清除脏标记
        self.frustum_dirty = False
    
    def get_view_matrix(self):
        """获取视图矩阵
        
        Returns:
            Matrix4x4: 视图矩阵
        """
        if self.view_dirty:
            self._update_view_matrix()
        return self.view_matrix
    
    def get_projection_matrix(self):
        """获取投影矩阵
        
        Returns:
            Matrix4x4: 投影矩阵
        """
        if self.projection_dirty:
            self._update_projection_matrix()
        return self.projection_matrix
    
    def get_view_projection_matrix(self):
        """获取视图投影矩阵
        
        Returns:
            Matrix4x4: 视图投影矩阵
        """
        if self.view_dirty:
            self._update_view_matrix()
        if self.projection_dirty:
            self._update_projection_matrix()
        
        # 重新计算视图投影矩阵
        self.view_projection_matrix = self.projection_matrix * self.view_matrix
        
        return self.view_projection_matrix
    
    def get_inverse_view_matrix(self):
        """获取逆视图矩阵
        
        Returns:
            Matrix4x4: 逆视图矩阵
        """
        if self.view_dirty:
            self._update_view_matrix()
        return self.inverse_view_matrix
    
    def get_inverse_projection_matrix(self):
        """获取逆投影矩阵
        
        Returns:
            Matrix4x4: 逆投影矩阵
        """
        if self.projection_dirty:
            self._update_projection_matrix()
        return self.inverse_projection_matrix
    
    def get_frustum(self):
        """获取视锥体
        
        Returns:
            Frustum: 视锥体
        """
        if self.frustum_dirty:
            self._update_frustum()
        return self.frustum
    
    def is_visible(self, bounding_box):
        """检查包围盒是否在视锥体内
        
        Args:
            bounding_box: 包围盒
            
        Returns:
            bool: 是否可见
        """
        if not self.use_frustum_culling:
            return True
        
        # 确保视锥体已更新
        if self.frustum_dirty:
            self._update_frustum()
        
        # 检查包围盒是否在视锥体内
        return self.frustum.contains_bounding_box(bounding_box)
    
    def is_sphere_visible(self, center, radius):
        """检查球体是否在视锥体内
        
        Args:
            center: 球心
            radius: 半径
            
        Returns:
            bool: 是否可见
        """
        if not self.use_frustum_culling:
            return True
        
        # 确保视锥体已更新
        if self.frustum_dirty:
            self._update_frustum()
        
        # 检查球体是否在视锥体内
        return self.frustum.contains_sphere(center, radius)
    
    def set_frustum_culling(self, enabled):
        """设置是否使用视锥体剔除
        
        Args:
            enabled: 是否启用
        """
        self.use_frustum_culling = enabled
    
    def set_occlusion_culling(self, enabled):
        """设置是否使用遮挡剔除
        注意：低端GPU通常性能受限，建议关闭
        
        Args:
            enabled: 是否启用
        """
        self.use_occlusion_culling = enabled
    
    def set_max_render_distance(self, distance):
        """设置最大渲染距离
        
        Args:
            distance: 最大渲染距离
        """
        self.max_render_distance = distance
    
    def set_dynamic_resolution_scale(self, scale):
        """设置动态分辨率缩放
        
        Args:
            scale: 缩放因子
        """
        # 限制缩放范围
        self.dynamic_resolution_scale = max(self.min_dynamic_scale, min(self.max_dynamic_scale, scale))
    
    def set_viewport(self, x, y, width, height):
        """设置视口
        
        Args:
            x: 视口X坐标
            y: 视口Y坐标
            width: 视口宽度
            height: 视口高度
        """
        self.viewport_x = x
        self.viewport_y = y
        self.viewport_width = width
        self.viewport_height = height
        
        # 更新宽高比
        self.aspect_ratio = width / height
        self.projection_dirty = True
        self.frustum_dirty = True
    
    def set_clear_color(self, color):
        """设置清除颜色
        
        Args:
            color: RGBA颜色元组
        """
        self.clear_color = color
    
    def set_exposure(self, exposure):
        """设置曝光
        
        Args:
            exposure: 曝光值
        """
        self.exposure = exposure
    
    def enable_hdr_rendering(self, enabled):
        """启用/禁用HDR渲染
        注意：低端GPU通常性能受限，建议关闭
        
        Args:
            enabled: 是否启用
        """
        self.enable_hdr = enabled
    
    def enable_temporal_aa(self, enabled):
        """启用/禁用临时抗锯齿
        
        Args:
            enabled: 是否启用
        """
        self.enable_taa = enabled
    
    def enable_bloom_effect(self, enabled):
        """启用/禁用泛光效果
        
        Args:
            enabled: 是否启用
        """
        self.enable_bloom = enabled
    
    def world_to_view(self, world_position):
        """世界空间到视图空间的变换
        
        Args:
            world_position: 世界空间位置
            
        Returns:
            Vector3: 视图空间位置
        """
        if self.view_dirty:
            self._update_view_matrix()
        
        # 应用视图矩阵
        return self.view_matrix.transform_point(world_position)
    
    def view_to_projection(self, view_position):
        """视图空间到投影空间的变换
        
        Args:
            view_position: 视图空间位置
            
        Returns:
            Vector3: 投影空间位置
        """
        if self.projection_dirty:
            self._update_projection_matrix()
        
        # 应用投影矩阵
        return self.projection_matrix.transform_point(view_position)
    
    def world_to_projection(self, world_position):
        """世界空间到投影空间的变换
        
        Args:
            world_position: 世界空间位置
            
        Returns:
            Vector3: 投影空间位置
        """
        # 先转换到视图空间，再转换到投影空间
        view_position = self.world_to_view(world_position)
        return self.view_to_projection(view_position)
    
    def projection_to_view(self, projection_position):
        """投影空间到视图空间的变换
        
        Args:
            projection_position: 投影空间位置
            
        Returns:
            Vector3: 视图空间位置
        """
        if self.projection_dirty:
            self._update_projection_matrix()
        
        # 应用逆投影矩阵
        return self.inverse_projection_matrix.transform_point(projection_position)
    
    def view_to_world(self, view_position):
        """视图空间到世界空间的变换
        
        Args:
            view_position: 视图空间位置
            
        Returns:
            Vector3: 世界空间位置
        """
        if self.view_dirty:
            self._update_view_matrix()
        
        # 应用逆视图矩阵
        return self.inverse_view_matrix.transform_point(view_position)
    
    def projection_to_world(self, projection_position):
        """投影空间到世界空间的变换
        
        Args:
            projection_position: 投影空间位置
            
        Returns:
            Vector3: 世界空间位置
        """
        # 先转换到视图空间，再转换到世界空间
        view_position = self.projection_to_view(projection_position)
        return self.view_to_world(view_position)
    
    def screen_to_world_ray(self, screen_x, screen_y, viewport_width, viewport_height):
        """从屏幕坐标创建世界空间射线
        
        Args:
            screen_x: 屏幕X坐标
            screen_y: 屏幕Y坐标
            viewport_width: 视口宽度
            viewport_height: 视口高度
            
        Returns:
            tuple: (射线起点, 射线方向)
        """
    
    def screen_to_ray(self, screen_x, screen_y):
        """从屏幕坐标创建世界空间射线（使用相机当前视口大小）
        
        Args:
            screen_x: 屏幕X坐标
            screen_y: 屏幕Y坐标
            
        Returns:
            tuple: (射线起点, 射线方向)
        """
        # 使用相机自身的视口大小
        return self.screen_to_world_ray(screen_x, screen_y, self.viewport_width, self.viewport_height)
    
    def screen_to_world_ray(self, screen_x, screen_y, viewport_width, viewport_height):
        """从屏幕坐标创建世界空间射线
        
        Args:
            screen_x: 屏幕X坐标
            screen_y: 屏幕Y坐标
            viewport_width: 视口宽度
            viewport_height: 视口高度
            
        Returns:
            tuple: (射线起点, 射线方向)
        """
        # 将屏幕坐标映射到NDC空间（归一化设备坐标）
        # NDC空间范围为[-1, 1]，Y轴向上
        ndc_x = (2.0 * screen_x) / viewport_width - 1.0
        ndc_y = 1.0 - (2.0 * screen_y) / viewport_height  # 翻转Y轴
        ndc_z = 1.0  # 远裁剪面
        
        # 从NDC空间到世界空间
        # 先到投影空间
        projection_near = Vector3(ndc_x, ndc_y, -1.0)
        projection_far = Vector3(ndc_x, ndc_y, 1.0)
        
        # 再到世界空间
        world_near = self.projection_to_world(projection_near)
        world_far = self.projection_to_world(projection_far)
        
        # 计算射线方向
        direction = (world_far - world_near).normalized()
        
        return world_near, direction
    
    def get_lod_distance(self, level):
        """获取指定LOD级别的切换距离
        
        Args:
            level: LOD级别
            
        Returns:
            float: LOD切换距离
        """
        if 0 <= level < len(self.lod_distances):
            return self.lod_distances[level] * self.lod_bias
        return float('inf')
    
    def set_lod_bias(self, bias):
        """设置LOD偏移因子
        
        Args:
            bias: LOD偏移因子，>1.0使LOD切换更远
        """
        self.lod_bias = bias
    
    def set_lod_distances(self, distances):
        """设置LOD切换距离
        
        Args:
            distances: LOD切换距离列表
        """
        self.lod_distances = distances
    
    def optimize_for_low_end_gpu(self):
        """针对低端GPU优化相机设置"""
        # 减少渲染距离
        self.max_render_distance = 300.0
        
        # 关闭复杂功能
        self.enable_hdr = False
        self.enable_occlusion_culling = False
        
        # 调整LOD参数，使远处模型简化更快
        self.lod_bias = 0.8  # <1.0使LOD切换更近
        
        # 增加近裁剪面，减少需要渲染的物体数量
        self.near_plane = 0.2
        
        # 减少远裁剪面，减少GPU计算负担
        self.far_plane = 500.0
    
    def __str__(self):
        return f"Camera(name='{self.name}', position={self.position}, target={self.target})"