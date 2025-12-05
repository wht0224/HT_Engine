# -*- coding: utf-8 -*-
"""
低端GPU渲染引擎场景管理系统
负责场景图的组织、更新和渲染优化
"""

import time
import math
from collections import defaultdict
import numpy as np

# 确保正确导入Math模块
from Engine.Math import Vector3, Matrix4x4, Quaternion
from Renderer.Resources import ResourceManager
from Renderer.Shaders import ShaderManager
from Scene.SceneNode import SceneNode
from Scene.Camera import Camera
from Scene.Light import LightManager
from Utils.PerformanceMonitor import PerformanceMonitor


class SceneManager:
    """场景管理器
    管理整个场景图、相机、灯光以及场景更新和渲染流程
    """
    
    def __init__(self, engine=None):
        """初始化场景管理器
        
        Args:
            engine: 渲染引擎实例（可选）
        """
        # 导入日志系统
        from Engine.Logger import get_logger
        self.logger = get_logger("SceneManager")
        
        # 引擎引用
        self.engine = engine
        self.renderer = engine.renderer if engine else None
        self.resource_manager = engine.resource_manager if engine else None
        self.shader_manager = engine.renderer.shader_manager if engine and engine.renderer else None
        
        # 场景根节点
        self.root_node = SceneNode("root")
        
        # 相机管理
        self.active_camera = None
        self.cameras = []
        
        # 灯光管理
        self.light_manager = LightManager()
        
        # 性能监控
        self.perf_monitor = PerformanceMonitor()
        
        # 场景优化配置
        self.optimization_settings = {
            # 视锥体剔除设置
            "frustum_culling": True,
            "culling_distance": 1000.0,  # 最大可见距离
            "use_occlusion_culling": True,
            
            # 批处理设置
            "static_batching": True,
            "dynamic_batching": True,
            "max_batch_size": 1024,  # 最大批处理顶点数
            
            # 实例化渲染设置
            "instancing_enabled": True,
            "min_instances_for_batching": 4,
            
            # 场景分区设置
            "use_octree": True,
            "octree_depth": 6,
            "octree_max_objects_per_node": 16,
            
            # 视距LOD设置
            "lod_enabled": True,
            "lod_distance_steps": [10.0, 25.0, 50.0, 100.0],
            
            # 视口设置
            "viewport_width": 1280,
            "viewport_height": 720,
            "aspect_ratio": 16.0 / 9.0,
            
            # 低端GPU特定优化
            "max_draw_calls": 1000,   # 针对GTX 750Ti优化的最大绘制调用数
            "max_visible_lights": 8,  # 针对低端GPU的最大可见光源数
            "max_shadow_casters": 4,  # 最大阴影投射光源数
            "shadow_map_resolution": 1024,  # 针对GTX 750Ti的阴影贴图分辨率
        }
        
        # 场景状态
        self.is_loaded = False
        self.is_running = False
        self.frame_count = 0
        self.total_time = 0.0
        
        # 场景分区结构（八叉树）
        self.octree = None
        
        # 可见性缓存
        self.visible_nodes = []
        self.last_visible_count = 0
        
        # 批处理数据
        self.static_batches = {}
        self.dynamic_batches = {}
        self.instanced_objects = defaultdict(list)
        
        # 选中节点管理
        self.selected_node = None
        self.selected_nodes = []
        
        # 初始化八叉树（如果启用）
        if self.optimization_settings["use_octree"]:
            try:
                self._initialize_octree()
                self.logger.debug("八叉树初始化完成")
            except Exception as e:
                self.logger.error(f"八叉树初始化失败: {e}", exc_info=True)
                self.optimization_settings["use_octree"] = False
                self.logger.warning("已禁用八叉树优化")
    
    def _initialize_octree(self):
        """初始化场景八叉树分区"""
        # 创建根八叉树节点
        self.octree = OctreeNode(
            center=Vector3(0, 0, 0),
            size=200.0,  # 默认场景大小
            depth=0,
            max_depth=self.optimization_settings["octree_depth"],
            max_objects=self.optimization_settings["octree_max_objects_per_node"]
        )
    
    def create_camera(self, name="MainCamera", position=None, rotation=None):
        """创建新相机
        
        Args:
            name: 相机名称
            position: 相机位置
            rotation: 相机旋转（四元数）
            
        Returns:
            Camera: 相机实例
        """
        position = position or Vector3(0, 0, 5)
        rotation = rotation or Quaternion.identity()
        
        camera = Camera(name, position, rotation, self.optimization_settings)
        self.cameras.append(camera)
        
        # 如果是第一个相机，设为活动相机
        if not self.active_camera:
            self.active_camera = camera
            
        # 将相机添加到场景图
        camera_node = SceneNode(name + "Node", position, rotation)
        camera_node.attach_camera(camera)
        self.root_node.add_child(camera_node)
        
        return camera
    
    def set_active_camera(self, camera):
        """设置活动相机
        
        Args:
            camera: 相机实例或相机名称
        """
        if isinstance(camera, str):
            # 通过名称查找相机
            for cam in self.cameras:
                if cam.name == camera:
                    self.active_camera = cam
                    return True
            return False
        elif isinstance(camera, Camera):
            self.active_camera = camera
            return True
        return False
    
    def add_light(self, light):
        """添加光源到场景
        
        Args:
            light: Light对象
            
        Returns:
            bool: 是否添加成功
        """
        # 检查是否超过最大光源数
        if len(self.light_manager.lights) >= self.optimization_settings["max_visible_lights"]:
            # 只在DEBUG模式下打印警告
            # print(f"警告: 已达到最大光源数限制 ({self.optimization_settings['max_visible_lights']})")
            return False
            
        # 添加光源到灯光管理器
        return self.light_manager.add_light(light)
    
    def create_node(self, name, position=None, rotation=None, scale=None):
        """在场景中创建一个新节点
        
        Args:
            name: 节点名称
            position: 位置
            rotation: 旋转（四元数）
            scale: 缩放
            
        Returns:
            SceneNode: 创建的节点
        """
        position = position or Vector3(0, 0, 0)
        rotation = rotation or Quaternion.identity()
        scale = scale or Vector3(1, 1, 1)
        
        node = SceneNode(name, position, rotation, scale)
        self.root_node.add_child(node)
        
        # 如果启用了八叉树，将节点添加到八叉树
        if self.optimization_settings["use_octree"] and self.octree:
            self.octree.insert(node)
            
        return node
    
    def find_node(self, name):
        """通过名称查找场景节点
        
        Args:
            name: 节点名称
            
        Returns:
            SceneNode or None: 找到的节点或None
        """
        return self.root_node.find_child(name)
    
    def load_scene(self, scene_name):
        """加载场景（这里是基础框架，实际项目中需要实现具体的加载逻辑）
        
        Args:
            scene_name: 场景名称
            
        Returns:
            bool: 是否加载成功
        """
        self.perf_monitor.start("scene_load")
        
        self.logger.info(f"加载场景: {scene_name}")
        
        # 清空当前场景
        self.clear_scene()
        
        try:
            # TODO: 实现实际的场景加载逻辑
            # 这可能包括从文件加载场景数据、创建节点、加载资源等
            
            # 示例：创建一个基础场景结构
            self._create_default_scene()
            
            # 更新场景状态
            self.is_loaded = True
            self.frame_count = 0
            self.total_time = 0.0
            
            self.logger.info(f"场景 '{scene_name}' 加载成功")
            return True
            
        except Exception as e:
            self.logger.error(f"加载场景 '{scene_name}' 失败: {e}", exc_info=True)
            return False
        finally:
            self.perf_monitor.stop("scene_load")
    
    def _create_default_scene(self):
        """创建默认场景（用于测试）"""
        # 创建主相机
        camera = self.create_camera("MainCamera", Vector3(0, 2, -5), Quaternion.from_euler(15, 0, 0))
        camera.set_perspective(60, 16/9, 0.1, 1000.0)
        
        # 创建方向光（太阳光）
        from Scene.Light import DirectionalLight
        sun = DirectionalLight("Sun", direction=Vector3(-1, -1, -1), color=Vector3(1, 0.95, 0.8), intensity=1.0)
        sun.cast_shadows = True
        self.add_light(sun)
        
        # 创建环境光
        from Scene.Light import AmbientLight
        ambient = AmbientLight("Ambient", color=Vector3(0.2, 0.2, 0.2), intensity=0.5)
        self.add_light(ambient)
        
        # 创建地面
        ground_node = self.create_node("Ground", Vector3(0, -1, 0))
        # 注意：这里只是创建节点，实际的网格和材质需要在应用层添加
    
    def clear_scene(self):
        """清空当前场景"""
        # 重置场景图
        self.root_node = SceneNode("root")
        
        # 重置相机列表
        self.cameras = []
        self.active_camera = None
        
        # 重置灯光
        self.light_manager.clear()
        
        # 重置场景状态
        self.is_loaded = False
        self.visible_nodes = []
        self.static_batches = {}
        self.dynamic_batches = {}
        self.instanced_objects = defaultdict(list)
        
        # 重置八叉树
        if self.optimization_settings["use_octree"]:
            self._initialize_octree()
    
    def update(self, delta_time):
        """更新场景
        
        Args:
            delta_time: 帧时间（秒）
        """
        try:
            self.perf_monitor.start("scene_update")
            
            # 更新场景时间
            self.total_time += delta_time
            self.frame_count += 1
            
            # 更新整个场景图
            self.root_node.update(delta_time, Matrix4x4.identity())
            
            # 更新灯光
            self.light_manager.update(delta_time)
            
            # 执行视锥体剔除和可见性计算
            self._update_visibility()
            
            # 执行批处理和实例化优化
            if self.optimization_settings["static_batching"] or self.optimization_settings["dynamic_batching"]:
                self._update_batching()
            
            # 更新实例化对象列表
            if self.optimization_settings["instancing_enabled"]:
                self._update_instancing()
            
            self.perf_monitor.stop("scene_update")
        except Exception as e:
            self.logger.error(f"场景更新失败: {e}", exc_info=True)
            self.perf_monitor.stop("scene_update")
    
    def _update_visibility(self):
        """更新节点可见性（视锥体剔除和遮挡剔除）"""
        self.perf_monitor.start("visibility_update")
        
        # 清空可见节点列表
        self.visible_nodes = []
        
        # 如果没有活动相机，无法执行剔除
        if not self.active_camera:
            self.perf_monitor.stop("visibility_update")
            return
        
        camera = self.active_camera
        frustum = camera.get_frustum()
        
        if self.optimization_settings["use_octree"] and self.octree:
            # 使用八叉树进行加速的视锥体剔除
            visible_objects = self.octree.get_visible_objects(frustum)
            
            # 对八叉树找到的对象进行精确的视锥体测试
            for node in visible_objects:
                if self._is_node_visible(node, frustum, camera):
                    self.visible_nodes.append(node)
        else:
            # 传统的场景图遍历和视锥体剔除
            self._cull_node(self.root_node, frustum, camera)
        
        # 记录最后可见节点数量
        self.last_visible_count = len(self.visible_nodes)
        
        # 对可见节点按照距离相机的远近排序（用于渲染顺序优化）
        if self.optimization_settings["frustum_culling"]:
            self._sort_visible_nodes_by_distance(camera.position)
        
        self.perf_monitor.stop("visibility_update")
    
    def _cull_node(self, node, frustum, camera):
        """递归地对视锥体中的节点进行剔除
        
        Args:
            node: 当前节点
            frustum: 相机视锥体
            camera: 当前相机
        """
        # 检查节点是否可见
        if self._is_node_visible(node, frustum, camera):
            # 将节点添加到可见列表
            self.visible_nodes.append(node)
            
            # 递归检查子节点
            for child in node.children:
                self._cull_node(child, frustum, camera)
    
    def _is_node_visible(self, node, frustum, camera):
        """检查节点是否在视锥体内且可见
        
        Args:
            node: 要检查的节点
            frustum: 相机视锥体
            camera: 当前相机
            
        Returns:
            bool: 节点是否可见
        """
        # 如果节点明确标记为不可见，则直接返回False
        if not node.visible:
            return False
        
        # 检查距离剔除
        if self.optimization_settings["frustum_culling"]:
            # 计算节点到相机的距离
            distance = (node.world_position - camera.position).length()
            
            # 如果超出最大可见距离，剔除
            if distance > self.optimization_settings["culling_distance"]:
                return False
        
        # 检查节点是否有包围盒
        if node.bounding_box:
            # 对视锥体进行精确的包围盒测试
            return frustum.contains_bounding_box(node.bounding_box)
        elif node.mesh:
            # 如果有网格但没有包围盒，使用网格的包围盒
            if node.mesh.bounding_box:
                return frustum.contains_bounding_box(node.mesh.bounding_box.transform(node.world_matrix))
        
        # 默认情况下，假设节点可见
        return True
    
    def _sort_visible_nodes_by_distance(self, camera_position):
        """根据到相机的距离对可见节点进行排序
        
        Args:
            camera_position: 相机位置
        """
        # 按照距离的平方排序（避免开方运算）
        self.visible_nodes.sort(key=lambda node: (node.world_position - camera_position).length_squared())
    
    def _update_batching(self):
        """更新批处理数据"""
        self.perf_monitor.start("batching_update")
        
        # 清空当前批处理数据
        if self.optimization_settings["dynamic_batching"]:
            self.dynamic_batches.clear()
        
        # 对可见节点进行批处理分组
        for node in self.visible_nodes:
            # 只有带有网格和材质的节点才能被批处理
            if not node.mesh or not node.material:
                continue
            
            # 跳过已经在实例化渲染中的对象
            if self.optimization_settings["instancing_enabled"] and node.use_instancing:
                continue
            
            # 确定批处理键（基于材质和网格特性）
            batch_key = self._get_batch_key(node)
            
            # 根据节点是静态还是动态，添加到相应的批处理
            if node.is_static and self.optimization_settings["static_batching"]:
                if batch_key not in self.static_batches:
                    self.static_batches[batch_key] = []
                self.static_batches[batch_key].append(node)
            elif self.optimization_settings["dynamic_batching"]:
                if batch_key not in self.dynamic_batches:
                    self.dynamic_batches[batch_key] = []
                self.dynamic_batches[batch_key].append(node)
        
        # 优化静态批处理（合并网格数据）
        if self.optimization_settings["static_batching"]:
            self._optimize_static_batches()
        
        self.perf_monitor.stop("batching_update")
    
    def _get_batch_key(self, node):
        """为节点生成批处理键
        
        Args:
            node: 场景节点
            
        Returns:
            tuple: 批处理键
        """
        # 使用材质ID作为主要分组依据
        material_id = id(node.material)
        
        # 如果材质相同，还要考虑网格特性（顶点格式、UV通道数量等）
        mesh_format = node.mesh.get_vertex_format_hash() if node.mesh else 0
        
        # 考虑是否使用骨骼动画等特殊属性
        has_skinning = node.animation and node.animation.is_skinned
        
        # 考虑是否接受阴影
        receives_shadows = node.material.receives_shadows
        
        return (material_id, mesh_format, has_skinning, receives_shadows)
    
    def _optimize_static_batches(self):
        """优化静态批处理（合并网格数据）"""
        # 对于每个静态批处理组
        for batch_key, nodes in list(self.static_batches.items()):
            # 如果节点数量太少，不需要合并
            if len(nodes) < 2:
                continue
            
            # 检查是否有资源管理器
            if not self.resource_manager:
                continue
            
            # 使用资源管理器创建静态批处理
            batch_id = self.resource_manager.create_static_batch(nodes)
            
            # 如果创建成功，更新节点的批处理ID
            if batch_id:
                for node in nodes:
                    node.batch_id = batch_id
                    # 标记节点为已批处理
                    node.is_batched = True
            
            # 记录批处理信息
            print(f"优化静态批处理: {batch_key}, 合并了 {len(nodes)} 个节点")
    
    def _update_instancing(self):
        """更新实例化渲染数据"""
        self.perf_monitor.start("instancing_update")
        
        # 清空当前实例化数据
        self.instanced_objects.clear()
        
        # 对使用相同网格和材质的对象进行分组
        for node in self.visible_nodes:
            # 检查节点是否启用了实例化
            if not node.use_instancing or not node.mesh or not node.material:
                continue
            
            # 使用网格和材质ID作为实例化分组依据
            instance_key = (id(node.mesh), id(node.material))
            
            # 将节点添加到实例化组
            self.instanced_objects[instance_key].append(node)
        
        # 过滤掉实例数量不足的组
        min_instances = self.optimization_settings["min_instances_for_batching"]
        self.instanced_objects = {k: v for k, v in self.instanced_objects.items() 
                                if len(v) >= min_instances}
        
        self.perf_monitor.stop("instancing_update")
    
    def render(self):
        """渲染当前场景
        
        Returns:
            dict: 渲染统计信息
        """
        try:
            self.perf_monitor.start("scene_render")
            
            # 如果没有活动相机，无法渲染
            if not self.active_camera:
                self.logger.warning("没有活动相机，无法渲染场景")
                return {"error": "no_active_camera"}
            
            # 准备渲染数据
            render_data = self._prepare_render_data()
            
            # 使用渲染器进行渲染
            if self.renderer:
                # 注意：实际渲染由引擎的render方法调用，这里可能需要调整
                # 这里保留原有的渲染统计逻辑
                render_stats = self._get_mock_render_stats()
            else:
                # 如果没有渲染器实例，返回模拟的渲染统计
                render_stats = self._get_mock_render_stats()
            
            # 添加场景管理的性能统计
            render_stats["scene_stats"] = {
                "total_nodes": self._count_total_nodes(self.root_node),
                "visible_nodes": self.last_visible_count,
                "camera_name": self.active_camera.name,
                "draw_calls": render_stats.get("draw_calls", 0),
                "batches": len(self.static_batches) + len(self.dynamic_batches),
                "instanced_groups": len(self.instanced_objects)
            }
            
            # 添加性能监控数据
            render_stats["performance_data"] = self.perf_monitor.get_averages()
            
            self.perf_monitor.stop("scene_render")
            return render_stats
        except Exception as e:
            self.logger.error(f"场景渲染失败: {e}", exc_info=True)
            self.perf_monitor.stop("scene_render")
            return {"error": str(e), "draw_calls": 0, "triangles": 0, "vertices": 0}
    
    def _prepare_render_data(self):
        """准备渲染数据
        
        Returns:
            dict: 渲染数据
        """
        render_data = {
            "static_batches": self.static_batches,
            "dynamic_batches": self.dynamic_batches,
            "instanced_objects": self.instanced_objects,
            "optimization_settings": self.optimization_settings
        }
        
        return render_data
    
    def _get_mock_render_stats(self):
        """获取模拟的渲染统计（用于演示）
        
        Returns:
            dict: 模拟的渲染统计
        """
        # 根据可见节点数量估算绘制调用
        estimated_draw_calls = min(
            self.last_visible_count, 
            self.optimization_settings["max_draw_calls"]
        )
        
        # 考虑批处理和实例化的优化效果
        batches_reduction = (len(self.static_batches) + len(self.dynamic_batches))
        instances_reduction = sum(len(v) for v in self.instanced_objects.values())
        
        # 估算优化后的绘制调用数
        optimized_draw_calls = max(1, estimated_draw_calls - batches_reduction - instances_reduction)
        
        return {
            "draw_calls": optimized_draw_calls,
            "triangles": estimated_draw_calls * 1000,  # 假设每个对象平均1000个三角形
            "vertices": estimated_draw_calls * 3000,   # 假设每个对象平均3000个顶点
            "visible_lights": min(len(self.light_manager.lights), self.optimization_settings["max_visible_lights"]),
            "shadow_casters": sum(1 for light in self.light_manager.lights if light.cast_shadows),
            "batch_optimization": batches_reduction > 0,
            "instancing_optimization": instances_reduction > 0
        }
    
    def _count_total_nodes(self, node):
        """递归计算场景中节点总数
        
        Args:
            node: 起始节点
            
        Returns:
            int: 节点总数
        """
        count = 1  # 自身
        for child in node.children:
            count += self._count_total_nodes(child)
        return count
    
    def set_viewport_size(self, width, height):
        """设置视口大小
        
        Args:
            width: 宽度
            height: 高度
        """
        # 更新优化设置中的视口大小
        self.optimization_settings["viewport_width"] = width
        self.optimization_settings["viewport_height"] = height
        self.optimization_settings["aspect_ratio"] = width / max(1, height)
        
        # 更新所有相机的视口大小
        for camera in self.cameras:
            camera.set_viewport(width, height)
    
    def get_scene_bounds(self):
        """获取整个场景的边界
        
        Returns:
            tuple: (最小点, 最大点)
        """
        # 递归计算场景边界
        min_point, max_point = self._calculate_node_bounds(self.root_node)
        return min_point, max_point
    
    def _calculate_node_bounds(self, node):
        """计算节点及其子节点的边界
        
        Args:
            node: 场景节点
            
        Returns:
            tuple: (最小点, 最大点)
        """
        # 初始化边界
        min_point = Vector3(float('inf'), float('inf'), float('inf'))
        max_point = Vector3(float('-inf'), float('-inf'), float('-inf'))
        
        # 检查节点自身的边界
        if node.bounding_box:
            box_min, box_max = node.bounding_box.get_min_max()
            min_point = Vector3.min(min_point, box_min)
            max_point = Vector3.max(max_point, box_max)
        elif node.mesh and node.mesh.bounding_box:
            # 使用网格的边界盒
            box_min, box_max = node.mesh.bounding_box.get_min_max()
            # 应用变换
            transformed_min = node.world_matrix.transform_point(box_min)
            transformed_max = node.world_matrix.transform_point(box_max)
            min_point = Vector3.min(min_point, transformed_min)
            max_point = Vector3.max(max_point, transformed_max)
        
        # 递归计算子节点边界
        for child in node.children:
            child_min, child_max = self._calculate_node_bounds(child)
            min_point = Vector3.min(min_point, child_min)
            max_point = Vector3.max(max_point, child_max)
        
        return min_point, max_point
    
    def optimize_scene(self):
        """对场景进行全局优化
        包括：
        - 静态对象批处理
        - 实例化对象识别
        - LOD设置优化
        - 八叉树重新构建
        """
        print("开始场景优化...")
        
        # 重置八叉树
        if self.optimization_settings["use_octree"]:
            self._initialize_octree()
            # 将所有节点重新插入八叉树
            self._rebuild_octree(self.root_node)
        
        # 识别可批处理的静态对象
        static_nodes = []
        self._find_static_nodes(self.root_node, static_nodes)
        
        # 对静态对象进行批处理优化
        if self.optimization_settings["static_batching"] and static_nodes:
            self._optimize_static_node_batching(static_nodes)
        
        # 识别适合实例化的对象
        if self.optimization_settings["instancing_enabled"]:
            self._identify_instances(self.root_node)
        
        # 优化LOD设置
        if self.optimization_settings["lod_enabled"]:
            self._optimize_lod_settings(self.root_node)
        
        print("场景优化完成")
    
    def _rebuild_octree(self, node):
        """将节点及其子节点重新插入八叉树
        
        Args:
            node: 起始节点
        """
        # 如果节点有位置和包围盒，插入八叉树
        if self.octree and node.world_position:
            self.octree.insert(node)
        
        # 递归处理子节点
        for child in node.children:
            self._rebuild_octree(child)
    
    def _find_static_nodes(self, node, static_nodes):
        """查找所有静态节点
        
        Args:
            node: 起始节点
            static_nodes: 收集静态节点的列表
        """
        if node.is_static and node.mesh:
            static_nodes.append(node)
        
        # 递归处理子节点
        for child in node.children:
            self._find_static_nodes(child, static_nodes)
    
    def _optimize_static_node_batching(self, static_nodes):
        """优化静态节点的批处理
        
        Args:
            static_nodes: 静态节点列表
        """
        # 按材质和网格类型对静态节点进行分组
        node_groups = defaultdict(list)
        for node in static_nodes:
            if node.material:
                # 使用材质ID和网格类型作为分组键
                group_key = (id(node.material), node.mesh.__class__.__name__)
                node_groups[group_key].append(node)
        
        # 对每个组进行优化处理
        for group_key, nodes in node_groups.items():
            if len(nodes) >= 2:  # 至少需要2个节点才能批处理
                # 这里可以实现更复杂的批处理逻辑
                # 例如将相似的静态对象合并为一个大的网格
                pass
    
    def _identify_instances(self, node):
        """识别适合实例化的对象
        
        Args:
            node: 起始节点
        """
        # 检查当前节点
        if node.mesh and node.material:
            # 对于使用相同网格和材质的对象，可以启用实例化
            # 这里简化处理，实际项目中可能需要更复杂的检测
            pass
        
        # 递归处理子节点
        for child in node.children:
            self._identify_instances(child)
    
    def _optimize_lod_settings(self, node):
        """优化节点的LOD设置
        
        Args:
            node: 起始节点
        """
        # 检查当前节点
        if node.mesh and hasattr(node.mesh, 'lod_levels'):
            # 根据节点的重要性和复杂度调整LOD设置
            # 例如，对远处的小物体使用更激进的LOD
            pass
        
        # 递归处理子节点
        for child in node.children:
            self._optimize_lod_settings(child)
    
    def get_performance_stats(self):
        """获取性能统计信息
        
        Returns:
            dict: 性能统计数据
        """
        stats = {
            "frame_count": self.frame_count,
            "total_time": self.total_time,
            "avg_frame_time": self.total_time / max(1, self.frame_count),
            "scene_node_count": self._count_total_nodes(self.root_node),
            "visible_node_count": self.last_visible_count,
            "camera_count": len(self.cameras),
            "light_count": len(self.light_manager.lights),
            "batch_count": len(self.static_batches) + len(self.dynamic_batches),
            "instance_group_count": len(self.instanced_objects),
            "octree_enabled": self.optimization_settings["use_octree"],
            "culling_enabled": self.optimization_settings["frustum_culling"]
        }
        
        # 添加详细的性能监控数据
        stats["detailed_timing"] = self.perf_monitor.get_averages()
        
        return stats
    
    def select_node(self, node):
        """选择单个节点
        
        Args:
            node: 要选择的场景节点
        """
        self.selected_node = node
        self.selected_nodes = [node] if node else []
    
    def add_to_selection(self, node):
        """将节点添加到选择列表
        
        Args:
            node: 要添加的场景节点
        """
        if node not in self.selected_nodes:
            self.selected_nodes.append(node)
            self.selected_node = node
    
    def remove_from_selection(self, node):
        """从选择列表中移除节点
        
        Args:
            node: 要移除的场景节点
        """
        if node in self.selected_nodes:
            self.selected_nodes.remove(node)
            if self.selected_node == node:
                self.selected_node = self.selected_nodes[-1] if self.selected_nodes else None
    
    def clear_selection(self):
        """清空选择列表"""
        self.selected_node = None
        self.selected_nodes = []
    
    def get_selected_node(self):
        """获取当前选中的单个节点
        
        Returns:
            SceneNode: 当前选中的节点
        """
        return self.selected_node
    
    def get_selected_nodes(self):
        """获取所有选中的节点
        
        Returns:
            list: 选中的节点列表
        """
        return self.selected_nodes
    
    def raycast(self, origin, direction, max_distance=1000.0):
        """射线检测，返回与射线相交的最近节点
        
        Args:
            origin: 射线原点
            direction: 射线方向（归一化）
            max_distance: 最大检测距离
            
        Returns:
            tuple: (相交节点, 相交点, 相交距离)，如果没有相交则返回(None, None, float('inf'))
        """
        closest_node = None
        closest_point = None
        closest_distance = float('inf')
        
        # 遍历所有可见节点进行射线检测
        for node in self.visible_nodes:
            if node.mesh and node.visible:
                # 检测射线与节点的相交
                intersect_point, distance = self._raycast_node(node, origin, direction, max_distance)
                if intersect_point and distance < closest_distance:
                    closest_node = node
                    closest_point = intersect_point
                    closest_distance = distance
        
        return closest_node, closest_point, closest_distance
    
    def _raycast_node(self, node, origin, direction, max_distance):
        """检测射线与单个节点的相交
        
        Args:
            node: 场景节点
            origin: 射线原点
            direction: 射线方向（归一化）
            max_distance: 最大检测距离
            
        Returns:
            tuple: (相交点, 相交距离)，如果没有相交则返回(None, None)
        """
        # 简化实现：使用节点的包围盒进行初步检测
        if node.bounding_box:
            # 检测射线与包围盒的相交
            intersect, distance = self._raycast_bounding_box(node.bounding_box, origin, direction, max_distance)
            if not intersect:
                return None, None
        
        # TODO: 实现更精确的网格射线检测
        # 这里简化处理，直接返回包围盒的相交点
        # 实际项目中需要实现三角形级别的射线检测
        
        return origin + direction * distance, distance
    
    def _raycast_bounding_box(self, bounding_box, origin, direction, max_distance):
        """检测射线与包围盒的相交
        
        Args:
            bounding_box: 包围盒
            origin: 射线原点
            direction: 射线方向（归一化）
            max_distance: 最大检测距离
            
        Returns:
            tuple: (是否相交, 相交距离)，如果没有相交则返回(False, None)
        """
        # 简化的AABB射线检测算法
        min_point, max_point = bounding_box.get_min_max()
        
        tmin = (min_point.x - origin.x) / direction.x if direction.x != 0 else -float('inf')
        tmax = (max_point.x - origin.x) / direction.x if direction.x != 0 else float('inf')
        
        tymin = (min_point.y - origin.y) / direction.y if direction.y != 0 else -float('inf')
        tymax = (max_point.y - origin.y) / direction.y if direction.y != 0 else float('inf')
        
        if tmin > tymax or tymin > tmax:
            return False, None
        
        if tymin > tmin:
            tmin = tymin
        if tymax < tmax:
            tmax = tymax
        
        tzmin = (min_point.z - origin.z) / direction.z if direction.z != 0 else -float('inf')
        tzmax = (max_point.z - origin.z) / direction.z if direction.z != 0 else float('inf')
        
        if tmin > tzmax or tzmin > tmax:
            return False, None
        
        if tzmin > tmin:
            tmin = tzmin
        if tzmax < tmax:
            tmax = tzmax
        
        # 检查相交距离是否在有效范围内
        if tmin >= 0 and tmin <= max_distance:
            return True, tmin
        
        return False, None
    
    def pick_node(self, screen_x, screen_y):
        """从屏幕坐标选择节点
        
        Args:
            screen_x: 屏幕X坐标
            screen_y: 屏幕Y坐标
            
        Returns:
            SceneNode: 选中的节点，如果没有选中则返回None
        """
        if not self.active_camera:
            return None
        
        # 将屏幕坐标转换为射线
        ray_origin, ray_direction = self.active_camera.screen_to_ray(screen_x, screen_y)
        
        # 执行射线检测
        hit_node, hit_point, hit_distance = self.raycast(ray_origin, ray_direction)
        
        # 如果检测到节点，选择该节点
        if hit_node:
            self.select_node(hit_node)
            return hit_node
        
        return None
    
    def shutdown(self):
        """
        关闭场景管理器，释放资源
        """
        try:
            self.logger.info("关闭场景管理器...")
            
            # 清空场景图
            self.root_node = SceneNode("root")
            
            # 清空相机列表
            self.cameras = []
            self.active_camera = None
            
            # 清空灯光
            self.light_manager.clear()
            
            # 清空批处理数据
            self.static_batches.clear()
            self.dynamic_batches.clear()
            self.instanced_objects.clear()
            
            # 清空八叉树
            self.octree = None
            
            # 清空可见节点列表
            self.visible_nodes = []
            
            self.logger.info("场景管理器关闭完成")
        except Exception as e:
            self.logger.error(f"关闭场景管理器失败: {e}", exc_info=True)


class OctreeNode:
    """八叉树节点（场景空间分区）"""
    
    def __init__(self, center, size, depth, max_depth, max_objects):
        """初始化八叉树节点
        
        Args:
            center: 节点中心点
            size: 节点大小
            depth: 当前深度
            max_depth: 最大深度
            max_objects: 每个节点最多包含的对象数
        """
        self.center = center
        self.size = size
        self.depth = depth
        self.max_depth = max_depth
        self.max_objects = max_objects
        
        # 子节点（八个象限）
        self.children = [None] * 8
        
        # 存储在此节点中的对象
        self.objects = []
        
        # 是否已分割
        self.is_divided = False
    
    def insert(self, obj):
        """将对象插入八叉树
        
        Args:
            obj: 要插入的场景节点
        """
        # 检查对象是否在此节点的边界内
        if not self._contains(obj):
            return False
        
        # 如果节点未分割且对象数未达到限制，直接添加
        if not self.is_divided and len(self.objects) < self.max_objects:
            self.objects.append(obj)
            return True
        
        # 如果节点未分割但达到了容量限制，进行分割
        if not self.is_divided and self.depth < self.max_depth:
            self._divide()
        
        # 将对象插入到适合的子节点
        if self.is_divided:
            for child in self.children:
                if child and child.insert(obj):
                    return True
        
        # 如果无法插入到任何子节点，保留在当前节点
        self.objects.append(obj)
        return True
    
    def _divide(self):
        """将八叉树节点分割为八个子节点"""
        half_size = self.size / 2
        quarter_size = half_size / 2
        
        # 创建八个子节点（八个象限）
        self.children[0] = OctreeNode(
            Vector3(self.center.x - quarter_size, self.center.y + quarter_size, self.center.z - quarter_size),
            half_size, self.depth + 1, self.max_depth, self.max_objects
        )
        self.children[1] = OctreeNode(
            Vector3(self.center.x + quarter_size, self.center.y + quarter_size, self.center.z - quarter_size),
            half_size, self.depth + 1, self.max_depth, self.max_objects
        )
        self.children[2] = OctreeNode(
            Vector3(self.center.x - quarter_size, self.center.y + quarter_size, self.center.z + quarter_size),
            half_size, self.depth + 1, self.max_depth, self.max_objects
        )
        self.children[3] = OctreeNode(
            Vector3(self.center.x + quarter_size, self.center.y + quarter_size, self.center.z + quarter_size),
            half_size, self.depth + 1, self.max_depth, self.max_objects
        )
        self.children[4] = OctreeNode(
            Vector3(self.center.x - quarter_size, self.center.y - quarter_size, self.center.z - quarter_size),
            half_size, self.depth + 1, self.max_depth, self.max_objects
        )
        self.children[5] = OctreeNode(
            Vector3(self.center.x + quarter_size, self.center.y - quarter_size, self.center.z - quarter_size),
            half_size, self.depth + 1, self.max_depth, self.max_objects
        )
        self.children[6] = OctreeNode(
            Vector3(self.center.x - quarter_size, self.center.y - quarter_size, self.center.z + quarter_size),
            half_size, self.depth + 1, self.max_depth, self.max_objects
        )
        self.children[7] = OctreeNode(
            Vector3(self.center.x + quarter_size, self.center.y - quarter_size, self.center.z + quarter_size),
            half_size, self.depth + 1, self.max_depth, self.max_objects
        )
        
        # 标记为已分割
        self.is_divided = True
        
        # 将当前节点中的对象重新分配到子节点
        objects_to_redistribute = self.objects.copy()
        self.objects.clear()
        
        for obj in objects_to_redistribute:
            for child in self.children:
                if child.insert(obj):
                    break
            else:
                # 如果无法插入到任何子节点，重新添加到当前节点
                self.objects.append(obj)
    
    def _contains(self, obj):
        """检查对象是否在此节点的边界内
        
        Args:
            obj: 场景节点
            
        Returns:
            bool: 对象是否在此节点内
        """
        # 获取对象的世界位置
        pos = obj.world_position
        
        # 检查位置是否在节点边界内
        half_size = self.size / 2
        return (abs(pos.x - self.center.x) <= half_size and
                abs(pos.y - self.center.y) <= half_size and
                abs(pos.z - self.center.z) <= half_size)
    
    def get_visible_objects(self, frustum):
        """获取视锥体内的可见对象
        
        Args:
            frustum: 相机视锥体
            
        Returns:
            list: 可见对象列表
        """
        visible_objects = []
        
        # 检查节点是否与视锥体相交
        if not frustum.contains_bounding_box(self._get_bounding_box()):
            return visible_objects
        
        # 添加当前节点中的对象
        visible_objects.extend(self.objects)
        
        # 递归检查子节点
        if self.is_divided:
            for child in self.children:
                if child:
                    visible_objects.extend(child.get_visible_objects(frustum))
        
        return visible_objects
    
    def _get_bounding_box(self):
        """获取节点的边界盒
        
        Returns:
            BoundingBox: 节点的边界盒
        """
        from Engine.Math import BoundingBox
        
        half_size = self.size / 2
        min_point = Vector3(
            self.center.x - half_size,
            self.center.y - half_size,
            self.center.z - half_size
        )
        max_point = Vector3(
            self.center.x + half_size,
            self.center.y + half_size,
            self.center.z + half_size
        )
        
        return BoundingBox(min_point, max_point)