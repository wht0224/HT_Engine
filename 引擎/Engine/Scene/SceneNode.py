# -*- coding: utf-8 -*-
"""
低端GPU渲染引擎场景节点系统
场景图的基本构建单元，支持层次化变换、组件化设计
"""

import numpy as np
from collections import defaultdict

# 确保正确导入Math模块
from Engine.Math import Vector3, Quaternion, Matrix4x4, BoundingBox


class SceneNode:
    """场景节点类
    场景图的基本构建单元，负责变换、层级关系和组件管理
    """
    
    def __init__(self, name="Node", position=None, rotation=None, scale=None):
        """初始化场景节点
        
        Args:
            name: 节点名称
            position: 局部位置
            rotation: 局部旋转（四元数）
            scale: 局部缩放
        """
        # 节点标识
        self.name = name
        self.uid = id(self)  # 使用Python对象ID作为唯一标识符
        
        # 变换属性（局部空间）
        self.position = position or Vector3(0, 0, 0)
        self.rotation = rotation or Quaternion.identity()
        self.scale = scale or Vector3(1, 1, 1)
        
        # 世界空间属性（缓存）
        self.world_position = Vector3(0, 0, 0)
        self.world_rotation = Quaternion.identity()
        self.world_scale = Vector3(1, 1, 1)
        self.world_matrix = Matrix4x4.identity()
        self.world_matrix_dirty = True  # 变换矩阵脏标记
        
        # 层级关系
        self.parent = None
        self.children = []
        
        # 组件和资源
        self.components = defaultdict(list)  # 组件字典，按类型分组
        self.mesh = None  # 网格引用
        self.material = None  # 材质引用
        self.animation = None  # 动画引用
        
        # 节点状态
        self.visible = True
        self.enabled = True
        self.is_static = False  # 是否为静态对象
        self.use_instancing = False  # 是否使用实例化渲染
        
        # 空间信息
        self.bounding_box = None  # 局部空间包围盒
        self.world_bounding_box = None  # 世界空间包围盒
        self.bounding_radius = 0.0  # 包围球半径
        self.bounding_sphere_dirty = True  # 包围球脏标记
        
        # LOD设置
        self.lod_levels = []  # LOD级别列表
        self.current_lod_level = 0  # 当前LOD级别
        self.lod_distances = [10.0, 25.0, 50.0, 100.0]  # LOD切换距离
        
        # 渲染属性
        self.receive_shadows = True  # 是否接收阴影
        self.cast_shadows = True  # 是否投射阴影
        self.render_priority = 0  # 渲染优先级
        
        # 批处理属性
        self.batch_id = None  # 批处理ID
        self.instance_index = -1  # 实例化索引
    
    def add_child(self, child):
        """添加子节点
        
        Args:
            child: 子节点
            
        Returns:
            bool: 添加是否成功
        """
        if not isinstance(child, SceneNode):
            return False
            
        # 如果子节点已经有父节点，先从原父节点中移除
        if child.parent:
            child.parent.remove_child(child)
            
        # 设置父子关系
        child.parent = self
        self.children.append(child)
        
        # 标记子节点的世界矩阵需要更新
        child.mark_transform_dirty()
        
        return True
    
    def remove_child(self, child):
        """移除子节点
        
        Args:
            child: 要移除的子节点
            
        Returns:
            bool: 移除是否成功
        """
        if isinstance(child, SceneNode):
            if child in self.children:
                self.children.remove(child)
                child.parent = None
                return True
        elif isinstance(child, str):
            # 通过名称查找子节点
            for i, c in enumerate(self.children):
                if c.name == child:
                    self.children[i].parent = None
                    del self.children[i]
                    return True
        return False
    
    def find_child(self, name):
        """查找子节点
        
        Args:
            name: 要查找的子节点名称
            
        Returns:
            SceneNode or None: 找到的子节点或None
        """
        for child in self.children:
            if child.name == name:
                return child
            
            # 递归查找孙节点
            found = child.find_child(name)
            if found:
                return found
        return None
    
    def get_children_count(self):
        """获取子节点数量
        
        Returns:
            int: 子节点数量
        """
        return len(self.children)
    
    def mark_transform_dirty(self):
        """标记变换为脏（需要重新计算）"""
        self.world_matrix_dirty = True
        self.bounding_sphere_dirty = True
        
        # 递归标记所有子节点为脏
        for child in self.children:
            child.mark_transform_dirty()
    
    def set_position(self, position):
        """设置局部位置
        
        Args:
            position: 新位置
        """
        self.position = position.copy() if position else Vector3(0, 0, 0)
        self.mark_transform_dirty()
    
    def set_rotation(self, rotation):
        """设置局部旋转
        
        Args:
            rotation: 新旋转（四元数）
        """
        self.rotation = rotation.copy() if rotation else Quaternion.identity()
        self.mark_transform_dirty()
    
    def set_rotation_euler(self, euler_angles):
        """设置局部旋转（欧拉角）
        
        Args:
            euler_angles: 欧拉角（XYZ，弧度）
        """
        self.rotation = Quaternion.from_euler(euler_angles.x, euler_angles.y, euler_angles.z)
        self.mark_transform_dirty()
    
    def set_scale(self, scale):
        """设置局部缩放
        
        Args:
            scale: 新缩放
        """
        self.scale = scale.copy() if scale else Vector3(1, 1, 1)
        self.mark_transform_dirty()
    
    def translate(self, translation):
        """平移节点
        
        Args:
            translation: 平移量
        """
        self.position += translation
        self.mark_transform_dirty()
    
    def rotate(self, rotation):
        """旋转节点
        
        Args:
            rotation: 旋转量（四元数）
        """
        self.rotation = rotation * self.rotation
        self.mark_transform_dirty()
    
    def rotate_euler(self, euler_angles):
        """旋转节点（欧拉角）
        
        Args:
            euler_angles: 欧拉角（XYZ，弧度）
        """
        rotation = Quaternion.from_euler(euler_angles.x, euler_angles.y, euler_angles.z)
        self.rotation = rotation * self.rotation
        self.mark_transform_dirty()
    
    def scale_by(self, scale_factor):
        """缩放节点
        
        Args:
            scale_factor: 缩放量
        """
        self.scale *= scale_factor
        self.mark_transform_dirty()
    
    def update_transform(self):
        """更新世界空间变换
        如果父节点的变换发生变化，子节点的世界空间变换也需要重新计算
        """
        if not self.world_matrix_dirty:
            return
        
        # 计算局部变换矩阵
        local_matrix = Matrix4x4.from_transform(self.position, self.rotation, self.scale)
        
        # 计算世界空间变换
        if self.parent:
            # 如果有父节点，将局部变换应用到父节点的世界变换上
            self.world_matrix = self.parent.world_matrix * local_matrix
            
            # 计算世界空间位置、旋转和缩放
            # 注意：这是简化计算，对于有缩放和剪切的变换，可能需要更复杂的分解
            self.world_position = self.parent.world_matrix.transform_point(self.position)
            self.world_rotation = self.parent.world_rotation * self.rotation
            self.world_scale = self.parent.world_scale * self.scale
        else:
            # 如果没有父节点，局部变换就是世界变换
            self.world_matrix = local_matrix
            self.world_position = self.position.copy()
            self.world_rotation = self.rotation.copy()
            self.world_scale = self.scale.copy()
        
        # 清除脏标记
        self.world_matrix_dirty = False
        
        # 更新世界空间包围盒
        self._update_world_bounding_box()
    
    def _update_world_bounding_box(self):
        """更新世界空间包围盒"""
        if not self.bounding_box:
            if self.mesh and self.mesh.bounding_box:
                self.bounding_box = self.mesh.bounding_box
            else:
                # 如果没有网格，使用位置作为包围盒
                self.bounding_box = BoundingBox(self.position, self.position)
        
        # 将局部包围盒变换到世界空间
        if self.bounding_box:
            self.world_bounding_box = self.bounding_box.transform(self.world_matrix)
    
    def update_bounding_sphere(self):
        """更新包围球
        包围球用于快速的空间查询和碰撞检测
        """
        if not self.bounding_sphere_dirty:
            return
        
        # 确保世界变换已更新
        self.update_transform()
        
        if self.world_bounding_box:
            # 从包围盒计算包围球
            center = self.world_bounding_box.get_center()
            max_extent = self.world_bounding_box.get_extents().max_component()
            self.bounding_radius = max_extent
        else:
            # 使用位置和估计半径
            self.bounding_radius = max(self.scale.x, self.scale.y, self.scale.z) * 0.5
        
        self.bounding_sphere_dirty = False
    
    def get_bounding_sphere(self):
        """获取包围球
        
        Returns:
            tuple: (球心, 半径)
        """
        self.update_bounding_sphere()
        return self.world_position, self.bounding_radius
    
    def update(self, delta_time, parent_matrix=None):
        """更新节点
        
        Args:
            delta_time: 帧时间（秒）
            parent_matrix: 父节点的世界矩阵（可选）
        """
        if not self.enabled:
            return
        
        # 更新变换
        self.update_transform()
        
        # 更新组件
        for component_type, components in self.components.items():
            for component in components:
                if hasattr(component, 'update'):
                    component.update(delta_time)
        
        # 更新动画
        if self.animation and hasattr(self.animation, 'update'):
            self.animation.update(delta_time)
        
        # 更新子节点
        for child in self.children:
            child.update(delta_time)
    
    def add_component(self, component):
        """添加组件
        
        Args:
            component: 要添加的组件
            
        Returns:
            bool: 添加是否成功
        """
        if not component:
            return False
            
        component_type = component.__class__.__name__
        self.components[component_type].append(component)
        
        # 设置组件的节点引用
        if hasattr(component, 'set_node'):
            component.set_node(self)
        
        return True
    
    def remove_component(self, component):
        """移除组件
        
        Args:
            component: 要移除的组件或组件类型
            
        Returns:
            bool: 移除是否成功
        """
        if isinstance(component, str):
            # 通过类型名称移除
            component_type = component
            if component_type in self.components and len(self.components[component_type]) > 0:
                del self.components[component_type]
                return True
            return False
        else:
            # 通过实例移除
            component_type = component.__class__.__name__
            if component_type in self.components and component in self.components[component_type]:
                self.components[component_type].remove(component)
                if not self.components[component_type]:
                    del self.components[component_type]
                return True
            return False
    
    def get_component(self, component_type):
        """获取组件
        
        Args:
            component_type: 组件类型或类型名称
            
        Returns:
            component or None: 找到的组件或None
        """
        if isinstance(component_type, str):
            type_name = component_type
        else:
            type_name = component_type.__name__
            
        if type_name in self.components and self.components[type_name]:
            return self.components[type_name][0]  # 返回第一个匹配的组件
        return None
    
    def get_components(self, component_type):
        """获取指定类型的所有组件
        
        Args:
            component_type: 组件类型或类型名称
            
        Returns:
            list: 组件列表
        """
        if isinstance(component_type, str):
            type_name = component_type
        else:
            type_name = component_type.__name__
            
        return self.components.get(type_name, [])
    
    def attach_mesh(self, mesh):
        """附加网格到节点
        
        Args:
            mesh: 网格对象
        """
        self.mesh = mesh
        
        # 更新包围盒
        if mesh and hasattr(mesh, 'bounding_box') and mesh.bounding_box:
            self.bounding_box = mesh.bounding_box
            self.mark_transform_dirty()
    
    def attach_material(self, material):
        """附加材质到节点
        
        Args:
            material: 材质对象
        """
        self.material = material
    
    def attach_animation(self, animation):
        """附加动画到节点
        
        Args:
            animation: 动画对象
        """
        self.animation = animation
        
        # 设置动画的目标节点
        if hasattr(animation, 'set_target_node'):
            animation.set_target_node(self)
    
    def attach_camera(self, camera):
        """附加相机到节点
        
        Args:
            camera: 相机对象
        """
        self.add_component(camera)
        
        # 设置相机的变换引用
        if hasattr(camera, 'set_transform_reference'):
            camera.set_transform_reference(self)
    
    def set_lod_levels(self, levels, distances=None):
        """设置LOD级别
        
        Args:
            levels: LOD级别列表（网格对象）
            distances: LOD切换距离列表
        """
        self.lod_levels = levels
        if distances:
            self.lod_distances = distances
    
    def update_lod_level(self, camera_position):
        """根据到相机的距离更新LOD级别
        
        Args:
            camera_position: 相机位置
            
        Returns:
            int: 新的LOD级别
        """
        if not self.lod_levels:
            return 0
        
        # 计算到相机的距离
        distance = (self.world_position - camera_position).length()
        
        # 确定LOD级别
        new_lod_level = 0
        for i, lod_distance in enumerate(self.lod_distances):
            if distance > lod_distance:
                new_lod_level = i + 1
            else:
                break
        
        # 确保LOD级别有效
        new_lod_level = min(new_lod_level, len(self.lod_levels) - 1)
        
        # 如果LOD级别发生变化，更新网格
        if new_lod_level != self.current_lod_level:
            self.current_lod_level = new_lod_level
            if 0 <= new_lod_level < len(self.lod_levels):
                self.attach_mesh(self.lod_levels[new_lod_level])
        
        return self.current_lod_level
    
    def set_static(self, is_static):
        """设置节点是否为静态
        
        Args:
            is_static: 是否为静态
        """
        self.is_static = is_static
    
    def set_visibility(self, visible):
        """设置节点可见性
        
        Args:
            visible: 是否可见
        """
        self.visible = visible
    
    def set_shadow_casting(self, cast_shadows):
        """设置节点是否投射阴影
        
        Args:
            cast_shadows: 是否投射阴影
        """
        self.cast_shadows = cast_shadows
    
    def set_shadow_receiving(self, receive_shadows):
        """设置节点是否接收阴影
        
        Args:
            receive_shadows: 是否接收阴影
        """
        self.receive_shadows = receive_shadows
    
    def set_instancing(self, use_instancing):
        """设置是否使用实例化渲染
        
        Args:
            use_instancing: 是否使用实例化渲染
        """
        self.use_instancing = use_instancing
    
    def get_world_matrix(self):
        """获取世界矩阵
        
        Returns:
            Matrix4x4: 世界矩阵
        """
        self.update_transform()
        return self.world_matrix
    
    def get_world_position(self):
        """获取世界位置
        
        Returns:
            Vector3: 世界位置
        """
        self.update_transform()
        return self.world_position
    
    def get_world_rotation(self):
        """获取世界旋转
        
        Returns:
            Quaternion: 世界旋转
        """
        self.update_transform()
        return self.world_rotation
    
    def get_world_scale(self):
        """获取世界缩放
        
        Returns:
            Vector3: 世界缩放
        """
        self.update_transform()
        return self.world_scale
    
    def get_world_bounding_box(self):
        """获取世界空间包围盒
        
        Returns:
            BoundingBox: 世界空间包围盒
        """
        self.update_transform()
        return self.world_bounding_box
    
    def look_at(self, target_position):
        """使节点朝向目标位置
        
        Args:
            target_position: 目标位置
        """
        # 计算方向向量
        direction = (target_position - self.position).normalized()
        
        # 计算旋转四元数
        # 默认向前为-Z轴，向上为+Y轴
        if abs(direction.dot(Vector3(0, 1, 0))) > 0.999:  # 如果接近垂直，使用不同的向上方向
            up = Vector3(0, 0, 1) if direction.y > 0 else Vector3(0, 0, -1)
        else:
            up = Vector3(0, 1, 0)
        
        # 计算旋转矩阵
        rot_matrix = Matrix4x4.look_at(self.position, target_position, up)
        
        # 转换为四元数
        self.rotation = Quaternion.from_matrix(rot_matrix)
        
        # 标记变换为脏
        self.mark_transform_dirty()
    
    def get_forward(self):
        """获取节点的前方向（在世界空间中）
        
        Returns:
            Vector3: 前方向向量
        """
        # 默认向前为-Z轴
        forward = Vector3(0, 0, -1)
        return self.world_rotation * forward
    
    def get_up(self):
        """获取节点的上方向（在世界空间中）
        
        Returns:
            Vector3: 上方向向量
        """
        # 默认向上为+Y轴
        up = Vector3(0, 1, 0)
        return self.world_rotation * up
    
    def get_right(self):
        """获取节点的右方向（在世界空间中）
        
        Returns:
            Vector3: 右方向向量
        """
        # 默认向右为+X轴
        right = Vector3(1, 0, 0)
        return self.world_rotation * right
    
    def is_in_frustum(self, frustum):
        """检查节点是否在视锥体内
        
        Args:
            frustum: 视锥体
            
        Returns:
            bool: 是否在视锥体内
        """
        # 确保包围盒已更新
        self.update_transform()
        
        if self.world_bounding_box:
            return frustum.contains_bounding_box(self.world_bounding_box)
        else:
            # 如果没有包围盒，使用包围球进行粗略测试
            center, radius = self.get_bounding_sphere()
            return frustum.contains_sphere(center, radius)
    
    def clone(self, clone_children=False, new_name=None):
        """克隆节点
        
        Args:
            clone_children: 是否克隆子节点
            new_name: 新节点名称
            
        Returns:
            SceneNode: 克隆的节点
        """
        # 创建新节点
        new_node = SceneNode(
            new_name or f"{self.name}_clone",
            self.position.copy(),
            self.rotation.copy(),
            self.scale.copy()
        )
        
        # 复制其他属性
        new_node.visible = self.visible
        new_node.enabled = self.enabled
        new_node.is_static = self.is_static
        new_node.use_instancing = self.use_instancing
        new_node.receive_shadows = self.receive_shadows
        new_node.cast_shadows = self.cast_shadows
        new_node.render_priority = self.render_priority
        
        # 复制网格和材质引用
        new_node.mesh = self.mesh
        new_node.material = self.material
        
        # 克隆子节点
        if clone_children:
            for child in self.children:
                cloned_child = child.clone(True)
                new_node.add_child(cloned_child)
        
        return new_node
    
    def get_hierarchy_path(self):
        """获取节点在场景图中的层次路径
        
        Returns:
            str: 层次路径
        """
        path = self.name
        parent = self.parent
        while parent:
            path = f"{parent.name}/{path}"
            parent = parent.parent
        return path
    
    def __str__(self):
        return f"SceneNode(name='{self.name}', position={self.position}, children={len(self.children)})"


# 简单的组件基类示例
class Component:
    """组件基类
    所有附加到场景节点的组件都应该继承自此类
    """
    
    def __init__(self):
        self.node = None
        self.enabled = True
    
    def set_node(self, node):
        """设置组件所属的节点
        
        Args:
            node: 场景节点
        """
        self.node = node
    
    def update(self, delta_time):
        """更新组件
        
        Args:
            delta_time: 帧时间（秒）
        """
        pass
    
    def on_attach(self):
        """当组件附加到节点时调用"""
        pass
    
    def on_detach(self):
        """当组件从节点分离时调用"""
        pass
    
    def enable(self):
        """启用组件"""
        self.enabled = True
    
    def disable(self):
        """禁用组件"""
        self.enabled = False