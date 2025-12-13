# -*- coding: utf-8 -*-
"""
MCP架构 - Processor层
负责处理数据处理和计算
"""

from Engine.Core.MCP.Model import Model
from Engine.Core.MCP.Controller import Controller
from Engine.Scene.SceneNode import SceneNode
from Engine.Math.Math import Vector3, Quaternion, Matrix4x4
from Engine.Renderer.Resources.Mesh import Mesh
from Engine.Renderer.Resources.Material import Material

class Processor:
    """MCP架构的Processor层
    负责处理数据处理和计算，包括渲染处理器、物理处理器和动画处理器
    """
    
    def __init__(self, model, controller):
        """初始化Processor层
        
        Args:
            model: Model层实例
            controller: Controller层实例
        """
        self.model = model
        self.controller = controller
        self.engine = model.engine
        
        # 渲染处理器
        self.render_processor = RenderProcessor(self)
        
        # 物理处理器
        self.physics_processor = PhysicsProcessor(self)
        
        # 动画处理器
        self.animation_processor = AnimationProcessor(self)
        
        # 建模处理器
        self.modeling_processor = ModelingProcessor(self)
        
        # 性能监控
        self.performance_stats = {
            "render_time": 0.0,
            "physics_time": 0.0,
            "animation_time": 0.0,
            "modeling_time": 0.0,
            "total_time": 0.0
        }
    
    def get_model(self):
        """获取Model层实例
        
        Returns:
            Model: Model层实例
        """
        return self.model
    
    def get_controller(self):
        """获取Controller层实例
        
        Returns:
            Controller: Controller层实例
        """
        return self.controller
    
    def update(self, delta_time):
        """更新所有处理器
        
        Args:
            delta_time: 帧时间
        """
        import time
        
        # 更新Model
        model_start = time.time()
        self.model.update(delta_time)
        model_time = time.time() - model_start
        
        # 更新渲染处理器
        render_start = time.time()
        self.render_processor.update(delta_time)
        render_time = time.time() - render_start
        
        # 更新物理处理器
        physics_start = time.time()
        self.physics_processor.update(delta_time)
        physics_time = time.time() - physics_start
        
        # 更新动画处理器
        animation_start = time.time()
        self.animation_processor.update(delta_time)
        animation_time = time.time() - animation_start
        
        # 更新建模处理器
        modeling_start = time.time()
        self.modeling_processor.update(delta_time)
        modeling_time = time.time() - modeling_start
        
        # 更新性能统计
        self.performance_stats = {
            "render_time": render_time,
            "physics_time": physics_time,
            "animation_time": animation_time,
            "modeling_time": modeling_time,
            "model_time": model_time,
            "total_time": render_time + physics_time + animation_time + modeling_time + model_time
        }
    
    def render(self):
        """渲染场景
        
        Returns:
            dict: 渲染统计信息
        """
        # 渲染Model
        model_stats = self.model.render()
        
        # 渲染处理器渲染
        render_stats = self.render_processor.render()
        
        # 合并统计信息
        stats = {
            **model_stats,
            **render_stats,
            **self.performance_stats
        }
        
        return stats
    
    def get_performance_stats(self):
        """获取性能统计信息
        
        Returns:
            dict: 性能统计信息
        """
        return self.performance_stats
    
    def get_render_processor(self):
        """获取渲染处理器
        
        Returns:
            RenderProcessor: 渲染处理器实例
        """
        return self.render_processor
    
    def get_physics_processor(self):
        """获取物理处理器
        
        Returns:
            PhysicsProcessor: 物理处理器实例
        """
        return self.physics_processor
    
    def get_animation_processor(self):
        """获取动画处理器
        
        Returns:
            AnimationProcessor: 动画处理器实例
        """
        return self.animation_processor
    
    def get_modeling_processor(self):
        """获取建模处理器
        
        Returns:
            ModelingProcessor: 建模处理器实例
        """
        return self.modeling_processor

class RenderProcessor:
    """渲染处理器
    负责处理渲染相关的计算和优化
    """
    
    def __init__(self, processor):
        """初始化渲染处理器
        
        Args:
            processor: Processor层实例
        """
        self.processor = processor
        self.model = processor.model
        self.controller = processor.controller
        self.engine = processor.engine
        
        # 渲染设置
        self.render_settings = {
            "enable_shadows": True,
            "enable_ambient_occlusion": True,
            "enable_reflections": True,
            "enable_anti_aliasing": False,
            "shadow_map_resolution": 1024,
            "ambient_occlusion_quality": "low"
        }
        
        # 渲染统计
        self.render_stats = {
            "draw_calls": 0,
            "triangles": 0,
            "vertices": 0,
            "visible_lights": 0,
            "shadow_casters": 0
        }
    
    def update(self, delta_time):
        """更新渲染处理器
        
        Args:
            delta_time: 帧时间
        """
        # 更新渲染设置
        if self.engine.renderer:
            # 应用渲染设置到渲染器
            self.engine.renderer.shadow_map_resolution = self.render_settings["shadow_map_resolution"]
    
    def render(self):
        """
        渲染场景
        
        Returns:
            dict: 渲染统计信息
        """
        if self.engine.renderer and self.engine.scene_mgr:
            # 使用引擎的渲染器进行渲染
            stats = self.engine.renderer.render(self.engine.scene_mgr)
            
            # 更新渲染统计
            if stats is not None:
                self.render_stats.update(stats)
            else:
                # 如果渲染器没有返回统计信息，使用渲染器的get_performance_stats方法获取
                if hasattr(self.engine.renderer, 'get_performance_stats'):
                    stats = self.engine.renderer.get_performance_stats()
                    self.render_stats.update(stats)
                else:
                    stats = {}
            
            return stats
        return {}
    
    def set_render_setting(self, setting_name, value):
        """设置渲染设置
        
        Args:
            setting_name: 设置名称
            value: 设置值
        """
        if setting_name in self.render_settings:
            self.render_settings[setting_name] = value
    
    def get_render_setting(self, setting_name):
        """获取渲染设置
        
        Args:
            setting_name: 设置名称
            
        Returns:
            any: 设置值
        """
        return self.render_settings.get(setting_name)
    
    def get_render_stats(self):
        """获取渲染统计信息
        
        Returns:
            dict: 渲染统计信息
        """
        return self.render_stats

class PhysicsProcessor:
    """物理处理器
    负责处理物理相关的计算和模拟
    """
    
    def __init__(self, processor):
        """初始化物理处理器
        
        Args:
            processor: Processor层实例
        """
        self.processor = processor
        self.model = processor.model
        self.controller = processor.controller
        self.engine = processor.engine
        
        # 物理设置
        self.physics_settings = {
            "gravity": Vector3(0, -9.8, 0),
            "enable_collision": True,
            "enable_gravity": True,
            "fixed_time_step": 1.0 / 60.0,
            "max_sub_steps": 4
        }
        
        # 物理统计
        self.physics_stats = {
            "collisions": 0,
            "rigid_bodies": 0,
            "constraints": 0
        }
        
        # 物理世界
        self.physics_world = None
    
    def update(self, delta_time):
        """更新物理处理器
        
        Args:
            delta_time: 帧时间
        """
        # 简化实现，实际项目中需要集成物理引擎
        pass
    
    def get_physics_settings(self):
        """获取物理设置
        
        Returns:
            dict: 物理设置
        """
        return self.physics_settings
    
    def set_physics_setting(self, setting_name, value):
        """设置物理设置
        
        Args:
            setting_name: 设置名称
            value: 设置值
        """
        if setting_name in self.physics_settings:
            self.physics_settings[setting_name] = value
    
    def get_physics_stats(self):
        """获取物理统计信息
        
        Returns:
            dict: 物理统计信息
        """
        return self.physics_stats

class AnimationProcessor:
    """动画处理器
    负责处理动画相关的计算和播放
    """
    
    def __init__(self, processor):
        """初始化动画处理器
        
        Args:
            processor: Processor层实例
        """
        self.processor = processor
        self.model = processor.model
        self.controller = processor.controller
        self.engine = processor.engine
        
        # 动画设置
        self.animation_settings = {
            "enable_animation": True,
            "animation_speed": 1.0,
            "enable_blending": True,
            "blend_duration": 0.3
        }
        
        # 动画统计
        self.animation_stats = {
            "active_animations": 0,
            "blending_animations": 0
        }
    
    def update(self, delta_time):
        """更新动画处理器
        
        Args:
            delta_time: 帧时间
        """
        # 更新场景中的动画
        if hasattr(self.engine, 'scene_mgr') and self.engine.scene_mgr:
            # 遍历所有节点，更新动画
            self._update_node_animations(self.engine.scene_mgr.root_node, delta_time)
    
    def _update_node_animations(self, node, delta_time):
        """更新节点的动画
        
        Args:
            node: 场景节点
            delta_time: 帧时间
        """
        # 更新当前节点的动画
        if node.animation and self.animation_settings["enable_animation"]:
            # 更新动画
            node.animation.update(delta_time * self.animation_settings["animation_speed"])
            
            # 应用动画到节点
            if hasattr(node.animation, "apply_to_node"):
                node.animation.apply_to_node(node)
        
        # 递归更新子节点的动画
        for child in node.children:
            self._update_node_animations(child, delta_time)
    
    def get_animation_settings(self):
        """获取动画设置
        
        Returns:
            dict: 动画设置
        """
        return self.animation_settings
    
    def set_animation_setting(self, setting_name, value):
        """设置动画设置
        
        Args:
            setting_name: 设置名称
            value: 设置值
        """
        if setting_name in self.animation_settings:
            self.animation_settings[setting_name] = value
    
    def get_animation_stats(self):
        """获取动画统计信息
        
        Returns:
            dict: 动画统计信息
        """
        return self.animation_stats

class ModelingProcessor:
    """建模处理器
    负责处理建模相关的计算和操作
    """
    
    def __init__(self, processor):
        """初始化建模处理器
        
        Args:
            processor: Processor层实例
        """
        self.processor = processor
        self.model = processor.model
        self.controller = processor.controller
        self.engine = processor.engine
        
        # 建模设置
        self.modeling_settings = {
            "snap_distance": 0.1,
            "grid_size": 1.0,
            "enable_snap": False,
            "enable_grid": True
        }
        
        # 建模统计
        self.modeling_stats = {
            "vertices_count": 0,
            "edges_count": 0,
            "faces_count": 0,
            "selected_vertices": 0,
            "selected_edges": 0,
            "selected_faces": 0
        }
    
    def update(self, delta_time):
        """更新建模处理器
        
        Args:
            delta_time: 帧时间
        """
        # 更新建模设置
        self.modeling_settings["enable_snap"] = self.model.snap_enabled
        self.modeling_settings["snap_distance"] = self.model.snap_distance
    
    def create_cube(self, size=1.0):
        """创建立方体
        
        Args:
            size: 立方体大小
            
        Returns:
            Mesh: 立方体网格
        """
        from Engine.Renderer.Resources.Mesh import Mesh
        return Mesh.create_cube(size)
    
    def create_sphere(self, radius=1.0, segments=32):
        """创建球体
        
        Args:
            radius: 球体半径
            segments: 球体分段数
            
        Returns:
            Mesh: 球体网格
        """
        from Engine.Renderer.Resources.Mesh import Mesh
        return Mesh.create_sphere(radius, segments, segments)
    
    def create_cylinder(self, radius=1.0, height=2.0, segments=32):
        """创建圆柱体
        
        Args:
            radius: 圆柱体半径
            height: 圆柱体高度
            segments: 圆柱体分段数
            
        Returns:
            Mesh: 圆柱体网格
        """
        from Engine.Renderer.Resources.Mesh import Mesh
        return Mesh.create_cylinder(radius, height, segments)
    
    def create_plane(self, width=1.0, height=1.0, segments=1):
        """创建平面
        
        Args:
            width: 平面宽度
            height: 平面高度
            segments: 平面分段数
            
        Returns:
            Mesh: 平面网格
        """
        from Engine.Renderer.Resources.Mesh import Mesh
        return Mesh.create_plane(width, height, segments, segments)
    
    def subdivide_mesh(self, mesh, levels=1):
        """细分网格
        
        Args:
            mesh: 要细分的网格
            levels: 细分级别
            
        Returns:
            Mesh: 细分后的网格
        """
        # 简化实现，实际项目中需要实现网格细分算法
        return mesh
    
    def merge_meshes(self, meshes):
        """合并网格
        
        Args:
            meshes: 要合并的网格列表
            
        Returns:
            Mesh: 合并后的网格
        """
        # 简化实现，实际项目中需要实现网格合并算法
        return meshes[0] if meshes else None
    
    def get_modeling_settings(self):
        """获取建模设置
        
        Returns:
            dict: 建模设置
        """
        return self.modeling_settings
    
    def set_modeling_setting(self, setting_name, value):
        """设置建模设置
        
        Args:
            setting_name: 设置名称
            value: 设置值
        """
        if setting_name in self.modeling_settings:
            self.modeling_settings[setting_name] = value
    
    def get_modeling_stats(self):
        """获取建模统计信息
        
        Returns:
            dict: 建模统计信息
        """
        return self.modeling_stats
