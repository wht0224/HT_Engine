from Engine.Renderer.Pipelines.ForwardRenderer import ForwardRenderer
from Engine.Renderer.Pipelines.DeferredRenderer import DeferredRenderer
from Engine.Logger import get_logger

class HybridRenderer:
    """
    混合渲染器，结合前向渲染和延迟渲染的优点
    - 不透明物体使用延迟渲染，提高光照性能
    - 透明物体、粒子、复杂材质使用前向渲染，保证渲染质量
    """
    
    def __init__(self, platform, resource_manager, shader_manager):
        """
        初始化混合渲染器
        
        Args:
            platform: 平台实例
            resource_manager: 资源管理器实例
            shader_manager: 着色器管理器实例
        """
        self.logger = get_logger("HybridRenderer")
        self.platform = platform
        self.resource_manager = resource_manager
        self.shader_manager = shader_manager
        
        # 初始化前向渲染器和延迟渲染器
        self.forward_renderer = ForwardRenderer(platform, resource_manager, shader_manager)
        self.deferred_renderer = DeferredRenderer(platform, resource_manager, shader_manager)
        
        # 渲染参数
        self.max_draw_calls = 1000
        self.max_visible_lights = 8
        self.shadow_map_resolution = 1024
        self.use_instancing = True
        self.shadow_cascade_count = 1
        
        # 性能统计
        self.performance_stats = {
            "render_time_ms": 0,
            "draw_calls": 0,
            "triangles": 0,
            "visible_objects": 0,
            "culled_objects": 0,
            "visible_lights": 0
        }
        
        self.logger.info("混合渲染器初始化完成")
    
    def initialize(self, config=None):
        """
        初始化渲染器
        
        Args:
            config: 渲染器配置参数
        """
        self.logger.info("初始化混合渲染器...")
        
        # 初始化两个渲染器
        self.forward_renderer.initialize(config)
        self.deferred_renderer.initialize(config)
        
        # 应用配置
        if config:
            if "max_draw_calls" in config:
                self.max_draw_calls = config["max_draw_calls"]
            if "max_visible_lights" in config:
                self.max_visible_lights = config["max_visible_lights"]
            if "shadow_map_resolution" in config:
                self.shadow_map_resolution = config["shadow_map_resolution"]
        
        # 同步参数到两个渲染器
        self._sync_renderer_parameters()
        
        self.logger.info("混合渲染器初始化完成")
    
    def shutdown(self):
        """
        关闭渲染器，释放资源
        """
        self.logger.info("关闭混合渲染器...")
        self.forward_renderer.shutdown()
        self.deferred_renderer.shutdown()
    
    def render(self, scene):
        """
        渲染场景
        
        Args:
            scene: 要渲染的场景
        
        Returns:
            tuple: (color_buffer, depth_buffer) 渲染结果
        """
        import time
        start_time = time.time()
        
        # 1. 分离场景中的物体：不透明物体和透明物体
        opaque_objects = []
        transparent_objects = []
        
        for node in scene.visible_nodes:
            if not node.is_visible():
                continue
            
            material = node.get_material()
            if material and material.is_transparent():
                transparent_objects.append(node)
            else:
                opaque_objects.append(node)
        
        # 2. 使用延迟渲染器渲染不透明物体
        # 临时替换场景的可见节点，只包含不透明物体
        original_visible_nodes = scene.visible_nodes
        scene.visible_nodes = opaque_objects
        
        deferred_result = self.deferred_renderer.render(scene)
        
        # 恢复原始可见节点
        scene.visible_nodes = original_visible_nodes
        
        if not deferred_result:
            self.logger.error("延迟渲染失败，跳过混合渲染")
            return None
        
        color_buffer, depth_buffer = deferred_result
        
        # 3. 使用前向渲染器渲染透明物体，叠加到延迟渲染结果上
        if transparent_objects:
            # 临时替换场景的可见节点，只包含透明物体
            scene.visible_nodes = transparent_objects
            
            # 渲染透明物体
            forward_result = self.forward_renderer.render(scene)
            
            # 恢复原始可见节点
            scene.visible_nodes = original_visible_nodes
            
            if forward_result:
                # 叠加透明物体到颜色缓冲
                transparent_color_buffer, _ = forward_result
                self._blend_transparent_objects(color_buffer, transparent_color_buffer)
        
        # 4. 渲染粒子和特殊效果（如果有的话）
        self._render_particles_and_effects(scene, color_buffer, depth_buffer)
        
        # 5. 更新性能统计
        end_time = time.time()
        render_time_ms = (end_time - start_time) * 1000
        
        # 合并两个渲染器的性能统计
        self.performance_stats = {
            "render_time_ms": render_time_ms,
            "draw_calls": self.forward_renderer.performance_stats["draw_calls"] + self.deferred_renderer.performance_stats["draw_calls"],
            "triangles": self.forward_renderer.performance_stats["triangles"] + self.deferred_renderer.performance_stats["triangles"],
            "visible_objects": len(opaque_objects) + len(transparent_objects),
            "culled_objects": self.forward_renderer.performance_stats["culled_objects"] + self.deferred_renderer.performance_stats["culled_objects"],
            "visible_lights": self.deferred_renderer.performance_stats["visible_lights"]
        }
        
        return (color_buffer, depth_buffer)
    
    def resize(self, width, height):
        """
        调整渲染器分辨率
        
        Args:
            width: 新宽度
            height: 新高度
        """
        self.forward_renderer.resize(width, height)
        self.deferred_renderer.resize(width, height)
    
    def enable_feature(self, feature_name, enable=True):
        """
        启用或禁用渲染特性
        
        Args:
            feature_name: 特性名称
            enable: 是否启用
        
        Returns:
            bool: 是否成功设置
        """
        # 同时在两个渲染器上启用或禁用特性
        result1 = self.forward_renderer.enable_feature(feature_name, enable)
        result2 = self.deferred_renderer.enable_feature(feature_name, enable)
        return result1 and result2
    
    def set_feature_quality(self, feature_name, quality):
        """
        设置渲染特性质量
        
        Args:
            feature_name: 特性名称
            quality: 质量级别
        """
        if hasattr(self.forward_renderer, "set_feature_quality"):
            self.forward_renderer.set_feature_quality(feature_name, quality)
        if hasattr(self.deferred_renderer, "set_feature_quality"):
            self.deferred_renderer.set_feature_quality(feature_name, quality)
    
    def set_shadow_map_resolution(self, resolution):
        """
        设置阴影贴图分辨率
        
        Args:
            resolution: 阴影贴图分辨率
        """
        self.shadow_map_resolution = resolution
        self.forward_renderer.set_shadow_map_resolution(resolution)
        self.deferred_renderer.set_shadow_map_resolution(resolution)
    
    def get_performance_stats(self):
        """
        获取性能统计信息
        
        Returns:
            dict: 性能统计数据
        """
        return self.performance_stats
    
    def _sync_renderer_parameters(self):
        """
        同步渲染参数到两个渲染器
        """
        # 同步通用参数
        for renderer in [self.forward_renderer, self.deferred_renderer]:
            renderer.max_draw_calls = self.max_draw_calls
            renderer.max_visible_lights = self.max_visible_lights
            renderer.shadow_map_resolution = self.shadow_map_resolution
            renderer.use_instancing = self.use_instancing
            renderer.shadow_cascade_count = self.shadow_cascade_count
    
    def _blend_transparent_objects(self, color_buffer, transparent_color_buffer):
        """
        混合透明物体到颜色缓冲
        
        Args:
            color_buffer: 延迟渲染的颜色缓冲
            transparent_color_buffer: 前向渲染的透明物体颜色缓冲
        """
        # 这里实现透明物体的混合逻辑
        # 简单实现：直接叠加透明颜色缓冲到主颜色缓冲
        # 实际实现中需要考虑透明度混合模式
        pass
    
    def _render_particles_and_effects(self, scene, color_buffer, depth_buffer):
        """
        渲染粒子和特殊效果
        
        Args:
            scene: 场景对象
            color_buffer: 颜色缓冲
            depth_buffer: 深度缓冲
        """
        # 粒子系统通常使用前向渲染
        # 这里可以添加粒子渲染逻辑
        pass
    
    def set_ssr_quality(self, quality):
        """
        设置屏幕空间反射质量
        
        Args:
            quality: 质量级别 (low/medium/high)
        """
        if hasattr(self.forward_renderer, "set_ssr_quality"):
            self.forward_renderer.set_ssr_quality(quality)
        if hasattr(self.deferred_renderer, "set_ssr_quality"):
            self.deferred_renderer.set_ssr_quality(quality)
    
    def set_volumetric_quality(self, quality):
        """
        设置体积光照质量
        
        Args:
            quality: 质量级别 (low/medium/high)
        """
        if hasattr(self.forward_renderer, "set_volumetric_quality"):
            self.forward_renderer.set_volumetric_quality(quality)
        if hasattr(self.deferred_renderer, "set_volumetric_quality"):
            self.deferred_renderer.set_volumetric_quality(quality)
    
    def set_ao_quality(self, quality):
        """
        设置环境光遮蔽质量
        
        Args:
            quality: 质量级别 (low/medium/high)
        """
        if hasattr(self.forward_renderer, "set_ao_quality"):
            self.forward_renderer.set_ao_quality(quality)
        if hasattr(self.deferred_renderer, "set_ao_quality"):
            self.deferred_renderer.set_ao_quality(quality)
