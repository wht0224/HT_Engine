# -*- coding: utf-8 -*-
"""
低端GPU渲染引擎光源系统
实现针对低端GPU优化的高效光源系统，支持基本光源类型
"""

import numpy as np
from enum import Enum

# 确保正确导入Math模块
from Engine.Math import Vector3, Quaternion, Matrix4x4, BoundingSphere


class LightType(Enum):
    """光源类型枚举"""
    DIRECTIONAL = 0  # 方向光（平行光）
    POINT = 1        # 点光源
    SPOT = 2         # 聚光灯
    AREA = 3         # 面光源（性能消耗高，低端GPU谨慎使用）


class Light:
    """光源类
    针对低端GPU优化的高效光源实现
    """
    
    def __init__(self, name="Light", light_type=LightType.DIRECTIONAL):
        """初始化光源
        
        Args:
            name: 光源名称
            light_type: 光源类型
        """
        # 光源标识
        self.name = name
        self.type = light_type
        
        # 光照属性
        self.color = Vector3(1.0, 1.0, 1.0)  # RGB颜色
        self.intensity = 1.0                  # 光强度
        self.energy = 100.0                   # 光能量（用于物理单位）
        self.temperature = 6500.0             # 色温（开尔文）
        
        # 衰减参数（点光源和聚光灯）
        self.radius = 10.0                    # 影响半径
        self.falloff_distance = 5.0           # 衰减距离
        self.constant_attenuation = 1.0       # 常数衰减
        self.linear_attenuation = 0.09        # 线性衰减
        self.quadratic_attenuation = 0.032    # 二次衰减
        
        # 聚光灯参数
        self.spot_angle = 45.0                # 聚光灯角度（度）
        self.spot_inner_angle = 30.0          # 聚光灯内锥角度（度）
        self.spot_falloff = 1.0               # 聚光灯衰减指数
        
        # 位置和方向（世界空间）
        self.position = Vector3(0, 0, 0)      # 光源位置
        self.direction = Vector3(0, 0, -1)    # 光源方向（对于方向光和面光源）
        self.up = Vector3(0, 1, 0)            # 光源上方向
        
        # 阴影参数
        self.cast_shadows = False             # 是否投射阴影
        self.shadow_map_size = 512            # 阴影贴图大小（低端GPU推荐512或1024）
        self.shadow_bias = 0.001              # 阴影偏移，减少阴影粉刺
        self.shadow_normal_bias = 0.01        # 法线偏移，减少阴影粉刺
        self.shadow_cascade_count = 1         # 阴影级联数量（方向光）
        self.shadow_cascade_splits = [0.1, 0.3, 0.6, 1.0]  # 级联分割比例
        
        # 性能优化参数
        self.priority = 0                     # 光源优先级（0最高）
        self.max_shadow_casters = 100         # 最大阴影投射物数量
        self.is_active = True                 # 是否激活
        self.affects_diffuse = True           # 是否影响漫反射
        self.affects_specular = True          # 是否影响镜面反射
        
        # 缓存
        self.light_space_matrix = Matrix4x4.identity()  # 光源空间矩阵
        self.bounding_sphere = BoundingSphere(Vector3(0, 0, 0), 0.0)  # 影响范围
        self.dirty = True                     # 更新脏标记
        
        # 变换引用
        self.transform_reference = None       # 用于获取变换信息的场景节点
        
        # 后处理效果
        self.volumetric_scattering = False    # 是否启用体积散射（低端GPU默认关闭）
        self.volumetric_intensity = 0.1       # 体积散射强度
        
        # 光照贴图
        self.baked = False                    # 是否是烘焙光照
        self.bake_group = 0                   # 烘焙组
    
    def set_transform_reference(self, scene_node):
        """设置变换引用
        
        Args:
            scene_node: 场景节点，用于获取变换信息
        """
        self.transform_reference = scene_node
    
    def set_type(self, light_type):
        """设置光源类型
        
        Args:
            light_type: 光源类型
        """
        self.type = light_type
        self.dirty = True
    
    def set_color(self, color):
        """设置光源颜色
        
        Args:
            color: RGB颜色向量
        """
        self.color = color.copy()
    
    def set_intensity(self, intensity):
        """设置光强度
        
        Args:
            intensity: 光强度
        """
        self.intensity = intensity
    
    def set_energy(self, energy):
        """设置光能量
        
        Args:
            energy: 光能量
        """
        self.energy = energy
        # 根据能量更新强度（简化转换）
        self.intensity = energy / 100.0
    
    def set_temperature(self, temperature):
        """设置色温
        
        Args:
            temperature: 色温（开尔文）
        """
        self.temperature = temperature
        # 根据色温更新颜色（简化的色温到RGB转换）
        self.color = self._temperature_to_rgb(temperature)
    
    def _temperature_to_rgb(self, temperature):
        """将色温转换为RGB颜色
        
        Args:
            temperature: 色温（开尔文）
            
        Returns:
            Vector3: RGB颜色
        """
        # 简化的色温到RGB转换
        # 实际应用中可以使用更精确的转换算法
        temp = temperature / 100.0
        
        # 红色通道
        if temp <= 66.0:
            r = 1.0
        else:
            r = 1.292936186062745 * pow(temp - 60.0, -0.1332047592)
            r = max(0.0, min(1.0, r))
        
        # 绿色通道
        if temp <= 66.0:
            g = 0.3900815787690196 * np.log(temp) - 0.6318414437886279
        else:
            g = 1.1298908608952942 * pow(temp - 60.0, -0.0755148492)
            g = max(0.0, min(1.0, g))
        
        # 蓝色通道
        if temp >= 66.0:
            b = 1.0
        elif temp <= 19.0:
            b = 0.0
        else:
            b = 0.4280627951305381 * np.log(temp - 10.0) - 0.5432040179291534
            b = max(0.0, min(1.0, b))
        
        return Vector3(r, g, b)
    
    def set_position(self, position):
        """设置光源位置
        
        Args:
            position: 光源位置
        """
        self.position = position.copy()
        self.dirty = True
    
    def set_direction(self, direction):
        """设置光源方向
        
        Args:
            direction: 光源方向（归一化）
        """
        self.direction = direction.copy()
        self.direction.normalize()
        self.dirty = True
    
    def set_radius(self, radius):
        """设置光源影响半径
        
        Args:
            radius: 影响半径
        """
        self.radius = max(0.01, radius)
        self.dirty = True
    
    def set_spot_angle(self, angle):
        """设置聚光灯角度
        
        Args:
            angle: 聚光灯角度（度）
        """
        self.spot_angle = max(0.01, min(179.9, angle))
        self.dirty = True
    
    def set_spot_inner_angle(self, angle):
        """设置聚光灯内锥角度
        
        Args:
            angle: 聚光灯内锥角度（度）
        """
        self.spot_inner_angle = max(0.01, min(self.spot_angle, angle))
        self.dirty = True
    
    def enable_shadows(self, enable, shadow_map_size=512):
        """启用/禁用阴影
        
        Args:
            enable: 是否启用阴影
            shadow_map_size: 阴影贴图大小（低端GPU推荐512）
        """
        self.cast_shadows = enable
        if enable:
            # 低端GPU使用较小的阴影贴图大小以节省内存和提高性能
            self.shadow_map_size = shadow_map_size
    
    def set_shadow_bias(self, bias, normal_bias=0.01):
        """设置阴影偏移
        
        Args:
            bias: 阴影偏移
            normal_bias: 法线偏移
        """
        self.shadow_bias = bias
        self.shadow_normal_bias = normal_bias
    
    def set_shadow_cascades(self, count, splits=None):
        """设置阴影级联（仅方向光）
        
        Args:
            count: 级联数量（低端GPU推荐1或2）
            splits: 级联分割比例
        """
        # 低端GPU限制级联数量以提高性能
        self.shadow_cascade_count = max(1, min(2, count))
        if splits:
            self.shadow_cascade_splits = splits[:self.shadow_cascade_count]
    
    def set_priority(self, priority):
        """设置光源优先级
        
        Args:
            priority: 优先级（数字越小优先级越高）
        """
        self.priority = priority
    
    def update(self, delta_time):
        """更新光源
        
        Args:
            delta_time: 帧时间（秒）
        """
        # 如果有变换引用，从引用更新位置和方向
        if self.transform_reference:
            self.position = self.transform_reference.get_world_position()
            self.direction = self.transform_reference.get_forward()
            self.dirty = True
        
        # 更新光源数据
        if self.dirty:
            self._update_bounding_sphere()
            if self.cast_shadows:
                self._update_shadow_matrices()
            self.dirty = False
    
    def _update_bounding_sphere(self):
        """更新光源的影响范围（包围球）"""
        if self.type == LightType.DIRECTIONAL:
            # 方向光没有明确的边界，使用一个很大的包围球
            self.bounding_sphere = BoundingSphere(Vector3(0, 0, 0), float('inf'))
        else:
            # 点光源和聚光灯使用其半径
            self.bounding_sphere = BoundingSphere(self.position, self.radius)
    
    def _update_shadow_matrices(self):
        """更新阴影映射所需的矩阵
        为不同类型的光源计算适当的光源空间矩阵
        """
        if self.type == LightType.DIRECTIONAL:
            # 方向光：使用正交投影
            # 注意：实际应用中需要根据场景边界动态计算投影矩阵
            # 这里使用简化版本，假设场景中心在原点
            size = 100.0  # 投影大小
            near_plane = 0.1
            far_plane = 200.0
            
            # 计算视图矩阵
            # 方向光的位置是虚拟的，用于确定投影方向
            target = self.position + self.direction
            view_matrix = Matrix4x4.look_at(self.position, target, self.up)
            
            # 计算正交投影矩阵
            projection_matrix = Matrix4x4.orthographic(
                -size, size, -size, size, near_plane, far_plane
            )
            
            # 光源空间矩阵 = 投影矩阵 * 视图矩阵
            self.light_space_matrix = projection_matrix * view_matrix
            
        elif self.type == LightType.POINT:
            # 点光源：需要立方体阴影映射（这里简化为单方向）
            # 低端GPU通常不支持点光源的阴影，这里使用简化实现
            size = self.radius
            near_plane = 0.1
            far_plane = self.radius
            
            # 使用默认前方向作为投影方向
            target = self.position + Vector3(0, 0, -1)
            view_matrix = Matrix4x4.look_at(self.position, target, self.up)
            
            # 透视投影矩阵
            projection_matrix = Matrix4x4.perspective(90.0, 1.0, near_plane, far_plane)
            
            # 光源空间矩阵 = 投影矩阵 * 视图矩阵
            self.light_space_matrix = projection_matrix * view_matrix
            
        elif self.type == LightType.SPOT:
            # 聚光灯：使用透视投影
            near_plane = 0.1
            far_plane = self.radius
            
            # 计算视图矩阵
            target = self.position + self.direction
            view_matrix = Matrix4x4.look_at(self.position, target, self.up)
            
            # 计算透视投影矩阵
            projection_matrix = Matrix4x4.perspective(
                self.spot_angle, 1.0, near_plane, far_plane
            )
            
            # 光源空间矩阵 = 投影矩阵 * 视图矩阵
            self.light_space_matrix = projection_matrix * view_matrix
    
    def get_bounding_sphere(self):
        """获取光源的影响范围
        
        Returns:
            BoundingSphere: 光源影响范围的包围球
        """
        if self.dirty:
            self._update_bounding_sphere()
        return self.bounding_sphere
    
    def get_light_space_matrix(self):
        """获取光源空间矩阵
        
        Returns:
            Matrix4x4: 光源空间矩阵
        """
        if self.dirty and self.cast_shadows:
            self._update_shadow_matrices()
            self.dirty = False
        return self.light_space_matrix
    
    def set_attenuation(self, constant=1.0, linear=0.09, quadratic=0.032):
        """设置衰减参数
        
        Args:
            constant: 常数衰减
            linear: 线性衰减
            quadratic: 二次衰减
        """
        self.constant_attenuation = constant
        self.linear_attenuation = linear
        self.quadratic_attenuation = quadratic
    
    def calculate_attenuation(self, distance):
        """计算指定距离处的衰减因子
        
        Args:
            distance: 到光源的距离
            
        Returns:
            float: 衰减因子（0.0-1.0）
        """
        # 检查是否在影响范围内
        if distance > self.radius:
            return 0.0
        
        # 计算标准衰减公式：1.0 / (constant + linear * d + quadratic * d²)
        attenuation = 1.0 / (
            self.constant_attenuation +
            self.linear_attenuation * distance +
            self.quadratic_attenuation * distance * distance
        )
        
        # 归一化衰减，确保在距离为0时衰减为1.0
        if distance == 0.0:
            return 1.0
        
        # 限制最大衰减值
        attenuation = min(attenuation, 1.0)
        
        # 应用范围衰减
        range_attenuation = 1.0 - (distance / self.radius)
        
        return attenuation * range_attenuation
    
    def calculate_spot_factor(self, direction_to_light):
        """计算聚光灯的强度因子
        
        Args:
            direction_to_light: 从表面指向光源的方向
            
        Returns:
            float: 聚光灯强度因子（0.0-1.0）
        """
        if self.type != LightType.SPOT:
            return 1.0
        
        # 计算表面到光源方向与聚光灯方向的夹角
        cos_angle = direction_to_light.dot(self.direction)
        
        # 转换角度到弧度
        outer_angle_rad = np.radians(self.spot_angle / 2.0)
        inner_angle_rad = np.radians(self.spot_inner_angle / 2.0)
        
        # 计算夹角的余弦
        cos_outer = np.cos(outer_angle_rad)
        cos_inner = np.cos(inner_angle_rad)
        
        # 如果不在聚光灯范围内
        if cos_angle < cos_outer:
            return 0.0
        
        # 如果在内锥范围内
        if cos_angle > cos_inner:
            return 1.0
        
        # 在过渡区域内，应用平滑过渡
        t = (cos_angle - cos_outer) / (cos_inner - cos_outer)
        
        # 应用指数衰减以实现更平滑的过渡
        if self.spot_falloff > 1.0:
            t = pow(t, self.spot_falloff)
        
        return t
    
    def set_volumetric_scattering(self, enabled, intensity=0.1):
        """启用/禁用体积散射效果
        
        Args:
            enabled: 是否启用
            intensity: 散射强度
        """
        self.volumetric_scattering = enabled
        self.volumetric_intensity = intensity
    
    def affect_diffuse(self, affect):
        """设置是否影响漫反射
        
        Args:
            affect: 是否影响
        """
        self.affects_diffuse = affect
    
    def affect_specular(self, affect):
        """设置是否影响镜面反射
        
        Args:
            affect: 是否影响
        """
        self.affects_specular = affect
    
    def optimize_for_low_end_gpu(self):
        """针对低端GPU优化光源设置"""
        # 降低阴影贴图大小以节省GPU内存
        if self.cast_shadows:
            self.shadow_map_size = 512  # 低端GPU推荐使用512
        
        # 方向光限制级联数量
        if self.type == LightType.DIRECTIONAL:
            self.shadow_cascade_count = 1  # 低端GPU只使用1级级联
        
        # 关闭体积散射等高级效果
        self.volumetric_scattering = False
        
        # 调整衰减参数，使光源影响范围更小，减少计算量
        self.radius = min(self.radius, 50.0)
        
        # 增加衰减速度
        self.linear_attenuation *= 1.5
        self.quadratic_attenuation *= 1.5
    
    def get_type_string(self):
        """获取光源类型的字符串表示
        
        Returns:
            str: 光源类型
        """
        if self.type == LightType.DIRECTIONAL:
            return "Directional"
        elif self.type == LightType.POINT:
            return "Point"
        elif self.type == LightType.SPOT:
            return "Spot"
        elif self.type == LightType.AREA:
            return "Area"
        return "Unknown"
    
    def __str__(self):
        return f"Light(name='{self.name}', type='{self.get_type_string()}', position={self.position}, intensity={self.intensity})"


class AmbientLight:
    """环境光类
    提供基本的环境光照，针对低端GPU优化
    """
    
    def __init__(self, name="AmbientLight"):
        """初始化环境光
        
        Args:
            name: 环境光名称
        """
        # 环境光属性
        self.name = name
        self.color = Vector3(0.2, 0.2, 0.2)  # 默认暗蓝色调
        self.intensity = 1.0
        self.energy = 10.0
        
        # 环境光遮蔽
        self.ao_strength = 0.5              # AO强度
        self.ao_radius = 1.0                # AO半径
        
        # 全局光照参数
        self.use_ambient_occlusion = False  # 是否使用环境光遮蔽
        self.use_gi = False                 # 是否使用全局光照（低端GPU通常关闭）
        
        # 性能优化参数
        self.is_active = True
        
        # 天空盒反射
        self.use_skybox = False             # 是否使用天空盒
        self.skybox_intensity = 1.0         # 天空盒强度
    
    def set_color(self, color):
        """设置环境光颜色
        
        Args:
            color: RGB颜色向量
        """
        self.color = color.copy()
    
    def set_intensity(self, intensity):
        """设置环境光强度
        
        Args:
            intensity: 强度
        """
        self.intensity = intensity
    
    def set_energy(self, energy):
        """设置环境光能量
        
        Args:
            energy: 能量
        """
        self.energy = energy
        self.intensity = energy / 10.0
    
    def set_ambient_occlusion(self, enabled, strength=0.5, radius=1.0):
        """设置环境光遮蔽
        
        Args:
            enabled: 是否启用
            strength: AO强度
            radius: AO半径
        """
        self.use_ambient_occlusion = enabled
        self.ao_strength = strength
        self.ao_radius = radius
    
    def set_skybox(self, enabled, intensity=1.0):
        """设置天空盒
        
        Args:
            enabled: 是否使用天空盒
            intensity: 天空盒强度
        """
        self.use_skybox = enabled
        self.skybox_intensity = intensity
    
    def optimize_for_low_end_gpu(self):
        """针对低端GPU优化环境光设置"""
        # 关闭全局光照
        self.use_gi = False
        
        # 如果启用了环境光遮蔽，使用较低的强度和半径
        if self.use_ambient_occlusion:
            self.ao_strength = min(self.ao_strength, 0.5)
            self.ao_radius = min(self.ao_radius, 0.5)
        
        # 降低环境光强度以减少计算量
        self.intensity = min(self.intensity, 0.8)
    
    def __str__(self):
        return f"AmbientLight(name='{self.name}', color={self.color}, intensity={self.intensity})"


class DirectionalLight(Light):
    """方向光类
    继承自基础Light类，专门用于方向光（平行光）
    """
    
    def __init__(self, name="DirectionalLight", direction=Vector3(0, -1, 0), color=Vector3(1, 1, 1), intensity=1.0):
        """初始化方向光
        
        Args:
            name: 光源名称
            direction: 光方向
            color: 光颜色
            intensity: 光强度
        """
        super().__init__(name, LightType.DIRECTIONAL)
        self.direction = direction.copy()
        self.direction.normalize()
        self.color = color.copy()
        self.intensity = intensity
        self.dirty = True


class PointLight(Light):
    """点光源类
    继承自基础Light类，专门用于点光源
    """
    
    def __init__(self, name="PointLight", position=Vector3(0, 0, 0), radius=10.0, color=Vector3(1, 1, 1), intensity=1.0):
        """初始化点光源
        
        Args:
            name: 光源名称
            position: 光源位置
            radius: 影响半径
            color: 光颜色
            intensity: 光强度
        """
        super().__init__(name, LightType.POINT)
        self.position = position.copy()
        self.radius = radius
        self.color = color.copy()
        self.intensity = intensity
        self.dirty = True


class SpotLight(Light):
    """聚光灯类
    继承自基础Light类，专门用于聚光灯
    """
    
    def __init__(self, name="SpotLight", position=Vector3(0, 0, 0), direction=Vector3(0, -1, 0), 
                 spot_angle=45.0, radius=10.0, color=Vector3(1, 1, 1), intensity=1.0):
        """初始化聚光灯
        
        Args:
            name: 光源名称
            position: 光源位置
            direction: 光方向
            spot_angle: 聚光灯角度
            radius: 影响半径
            color: 光颜色
            intensity: 光强度
        """
        super().__init__(name, LightType.SPOT)
        self.position = position.copy()
        self.direction = direction.copy()
        self.direction.normalize()
        self.spot_angle = spot_angle
        self.radius = radius
        self.color = color.copy()
        self.intensity = intensity
        self.dirty = True


class LightManager:
    """灯光管理器
    管理场景中的所有光源，包括添加、删除、更新和获取可见光源
    """
    
    def __init__(self):
        """初始化灯光管理器"""
        # 存储所有光源
        self.lights = []
        self.ambient_lights = []
        
        # 可见光源缓存
        self.visible_lights = []
        self.visible_ambient_lights = []
        
        # 性能优化配置
        self.max_visible_lights = 8  # 低端GPU最大可见光源数
        self.max_visible_ambient_lights = 1  # 最多1个环境光
        
        # 光源优先级队列
        self.light_priority_queue = []
        
        # 脏标记
        self.dirty = True
    
    def add_light(self, light):
        """添加光源到场景
        
        Args:
            light: Light对象
            
        Returns:
            bool: 是否添加成功
        """
        if isinstance(light, AmbientLight):
            # 环境光特殊处理
            if len(self.ambient_lights) < self.max_visible_ambient_lights:
                self.ambient_lights.append(light)
                self.dirty = True
                return True
            return False
        else:
            # 普通光源
            if len(self.lights) < self.max_visible_lights:
                self.lights.append(light)
                self.dirty = True
                return True
            return False
    
    def remove_light(self, light):
        """从场景中移除光源
        
        Args:
            light: Light对象
            
        Returns:
            bool: 是否移除成功
        """
        if isinstance(light, AmbientLight):
            if light in self.ambient_lights:
                self.ambient_lights.remove(light)
                self.dirty = True
                return True
        else:
            if light in self.lights:
                self.lights.remove(light)
                self.dirty = True
                return True
        return False
    
    def clear(self):
        """清空所有光源"""
        self.lights.clear()
        self.ambient_lights.clear()
        self.visible_lights.clear()
        self.visible_ambient_lights.clear()
        self.light_priority_queue.clear()
        self.dirty = True
    
    def update(self, delta_time):
        """更新所有光源
        
        Args:
            delta_time: 帧时间（秒）
        """
        # 更新所有光源
        for light in self.lights:
            light.update(delta_time)
        
        for ambient_light in self.ambient_lights:
            # 环境光更新逻辑（如果有的话）
            pass
        
        # 如果有光源状态变化，标记为脏
        if any(light.dirty for light in self.lights):
            self.dirty = True
    
    def get_visible_lights(self, camera):
        """获取相机视锥体内的可见光源
        
        Args:
            camera: Camera对象
            
        Returns:
            tuple: (可见光源列表, 可见环境光列表)
        """
        if self.dirty:
            # 更新可见光源
            self._update_visible_lights(camera)
            self.dirty = False
        
        return self.visible_lights, self.visible_ambient_lights
    
    def _update_visible_lights(self, camera):
        """更新可见光源列表
        
        Args:
            camera: Camera对象
        """
        # 获取相机视锥体
        frustum = camera.get_frustum()
        
        # 清空可见光源列表
        self.visible_lights.clear()
        self.visible_ambient_lights.clear()
        
        # 添加所有环境光（通常只有一个）
        self.visible_ambient_lights = self.ambient_lights[:self.max_visible_ambient_lights]
        
        # 计算每个光源到相机的距离，用于优先级排序
        light_distances = []
        for light in self.lights:
            if light.is_active:
                # 计算光源到相机的距离
                if light.type == LightType.DIRECTIONAL:
                    # 方向光没有位置，使用最大优先级
                    distance = 0.0
                else:
                    distance = (light.position - camera.position).length()
                
                # 根据距离和优先级计算综合得分
                priority_score = distance / (light.priority + 1.0)
                light_distances.append((priority_score, light))
        
        # 按距离排序，优先选择近的光源
        light_distances.sort(key=lambda x: x[0])
        
        # 选择前N个可见光源
        visible_count = 0
        for _, light in light_distances:
            if visible_count >= self.max_visible_lights:
                break
            
            # 检查光源是否在视锥体内
            if self._is_light_visible(light, frustum):
                self.visible_lights.append(light)
                visible_count += 1
    
    def _is_light_visible(self, light, frustum):
        """检查光源是否在视锥体内
        
        Args:
            light: Light对象
            frustum: Frustum对象
            
        Returns:
            bool: 光源是否可见
        """
        if light.type == LightType.DIRECTIONAL:
            # 方向光总是可见
            return True
        else:
            # 检查光源的包围球是否与视锥体相交
            bounding_sphere = light.get_bounding_sphere()
            return frustum.contains_sphere(bounding_sphere.center, bounding_sphere.radius)
    
    def get_directional_lights(self):
        """获取所有方向光
        
        Returns:
            list: 方向光列表
        """
        return [light for light in self.lights if light.type == LightType.DIRECTIONAL]
    
    def get_point_lights(self):
        """获取所有点光源
        
        Returns:
            list: 点光源列表
        """
        return [light for light in self.lights if light.type == LightType.POINT]
    
    def get_spot_lights(self):
        """获取所有聚光灯
        
        Returns:
            list: 聚光灯列表
        """
        return [light for light in self.lights if light.type == LightType.SPOT]
    
    def optimize_for_low_end_gpu(self):
        """针对低端GPU优化光源设置"""
        # 减少最大可见光源数
        self.max_visible_lights = 8
        self.max_visible_ambient_lights = 1
        
        # 优化所有光源
        for light in self.lights:
            light.optimize_for_low_end_gpu()
        
        for ambient_light in self.ambient_lights:
            ambient_light.optimize_for_low_end_gpu()
    
    def __len__(self):
        """获取光源总数
        
        Returns:
            int: 光源总数
        """
        return len(self.lights) + len(self.ambient_lights)
    
    def __iter__(self):
        """迭代所有光源
        
        Returns:
            iterator: 光源迭代器
        """
        # 先返回普通光源，再返回环境光
        for light in self.lights:
            yield light
        for ambient_light in self.ambient_lights:
            yield ambient_light