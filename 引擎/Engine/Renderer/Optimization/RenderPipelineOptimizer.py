from Engine.Logger import get_logger
from Engine.Renderer.Renderer import RenderMode, RenderQuality

class RenderPipelineOptimizer:
    """
    渲染管线优化器，根据GPU能力动态调整渲染管线参数
    """
    
    def __init__(self, renderer, gpu_evaluator):
        """
        初始化渲染管线优化器
        
        Args:
            renderer: 渲染器实例
            gpu_evaluator: GPU能力评估器实例
        """
        self.logger = get_logger("RenderPipelineOptimizer")
        self.renderer = renderer
        self.gpu_evaluator = gpu_evaluator
        
        # 获取推荐的渲染设置
        self.recommended_settings = self.gpu_evaluator.get_recommended_settings()
        
        # 初始化优化器
        self.optimize()
    
    def optimize(self):
        """
        优化渲染管线参数
        """
        self.logger.info("开始优化渲染管线参数")
        
        # 1. 调整渲染模式
        self._adjust_render_mode()
        
        # 2. 调整渲染质量
        self._adjust_render_quality()
        
        # 3. 调整渲染管线参数
        self._adjust_pipeline_parameters()
        
        # 4. 调整渲染特性
        self._adjust_render_features()
        
        # 5. 应用架构特定优化
        self._apply_architecture_specific_optimizations()
        
        self.logger.info("渲染管线参数优化完成")
    
    def _adjust_render_mode(self):
        """
        调整渲染模式
        """
        render_mode_str = self.recommended_settings["render_mode"]
        render_mode = RenderMode(render_mode_str)
        
        if self.renderer.render_mode != render_mode:
            self.logger.info(f"切换渲染模式: {self.renderer.render_mode.value} -> {render_mode.value}")
            self.renderer.render_mode = render_mode
            
            # 重新创建渲染管线
            self.renderer._create_render_pipelines()
    
    def _adjust_render_quality(self):
        """
        调整渲染质量
        """
        quality_str = self.recommended_settings["quality_level"]
        quality = RenderQuality(quality_str)
        
        if self.renderer.quality != quality:
            self.logger.info(f"调整渲染质量: {self.renderer.quality.value} -> {quality.value}")
            self.renderer.quality = quality
    
    def _adjust_pipeline_parameters(self):
        """
        调整渲染管线参数
        """
        # 调整最大绘制调用数
        self.renderer.max_draw_calls = self.recommended_settings["max_draw_calls"]
        
        # 调整最大可见光源数
        self.renderer.max_visible_lights = self.recommended_settings["max_visible_lights"]
        
        # 调整阴影贴图分辨率
        self.renderer.shadow_map_resolution = self.recommended_settings["shadow_map_resolution"]
        
        # 调整纹理质量
        self._adjust_texture_quality(self.recommended_settings["texture_quality"])
        
        # 调整LOD参数
        self.renderer.lod_bias = self.recommended_settings["lod_bias"]
        
        self.logger.info(f"调整渲染管线参数: 最大绘制调用={self.renderer.max_draw_calls}, "
                       f"最大可见光源={self.renderer.max_visible_lights}, "
                       f"阴影贴图分辨率={self.renderer.shadow_map_resolution}, "
                       f"LOD偏差={self.renderer.lod_bias}")
    
    def _adjust_texture_quality(self, texture_quality):
        """
        调整纹理质量
        
        Args:
            texture_quality: 纹理质量等级 (low/medium/high/ultra)
        """
        # 调整纹理分辨率和压缩格式
        if texture_quality == "low":
            # 低纹理质量：使用较低分辨率和高压缩
            self.renderer.tex_res_scale = 0.5
            self.renderer.tex_compression = "bc7"
        elif texture_quality == "medium":
            # 中等纹理质量：使用正常分辨率和中等压缩
            self.renderer.tex_res_scale = 1.0
            self.renderer.tex_compression = "bc7"
        elif texture_quality == "high":
            # 高纹理质量：使用高分辨率和低压缩
            self.renderer.tex_res_scale = 1.5
            self.renderer.tex_compression = "bc7"
        else:  # ultra
            # 超高纹理质量：使用最高分辨率和无损压缩
            self.renderer.tex_res_scale = 2.0
            self.renderer.tex_compression = "none"
        
        self.logger.info(f"调整纹理质量: {texture_quality}, 分辨率缩放={self.renderer.tex_res_scale}, "
                       f"压缩格式={self.renderer.tex_compression}")
    
    def _adjust_render_features(self):
        """
        调整渲染特性
        """
        # 调整SSR（屏幕空间反射）
        self.renderer.ssr_enabled = self.recommended_settings["enable_ssr"]
        
        # 调整体积光照
        self.renderer.volumetric_lighting_enabled = self.recommended_settings["enable_volumetric_lighting"]
        
        # 调整环境光遮蔽
        self.renderer.ao_enabled = self.recommended_settings["enable_ao"]
        
        # 调整其他渲染特性
        self._adjust_other_features()
        
        self.logger.info(f"调整渲染特性: SSR={self.renderer.ssr_enabled}, "
                       f"体积光照={self.renderer.volumetric_lighting_enabled}, "
                       f"环境光遮蔽={self.renderer.ao_enabled}")
    
    def _adjust_other_features(self):
        """
        调整其他渲染特性
        """
        hardware_info = self.renderer.hw_info
        render_features = hardware_info.get("render_features", {})
        
        # 根据GPU支持情况调整渲染特性
        
        # 曲面细分支持
        self.renderer.tessellation_enabled = render_features.get("tessellation", False)
        
        # 几何着色器支持
        self.renderer.geometry_shaders_enabled = render_features.get("geometry_shaders", False)
        
        # 计算着色器支持
        self.renderer.compute_shaders_enabled = render_features.get("compute_shaders", False)
        
        # 光线追踪支持
        self.renderer.ray_tracing_enabled = render_features.get("ray_tracing", False)
        
        # 网格着色器支持
        self.renderer.mesh_shaders_enabled = render_features.get("mesh_shaders", False)
        
        # 可变速率着色支持
        self.renderer.variable_rate_shading_enabled = render_features.get("variable_rate_shading", False)
    
    def _apply_architecture_specific_optimizations(self):
        """
        应用架构特定优化
        """
        architecture = self.renderer.hw_info["gpu_architecture"]
        vendor = self.renderer.hw_info["gpu_vendor"]
        
        self.logger.info(f"应用架构特定优化: {vendor} {architecture}")
        
        # NVIDIA架构特定优化
        if vendor == "NVIDIA":
            self._apply_nvidia_optimizations(architecture)
        
        # AMD架构特定优化
        elif vendor == "AMD":
            self._apply_amd_optimizations(architecture)
        
        # Intel架构特定优化
        elif vendor == "Intel":
            self._apply_intel_optimizations(architecture)
    
    def _apply_nvidia_optimizations(self, architecture):
        """
        应用NVIDIA架构特定优化
        
        Args:
            architecture: NVIDIA GPU架构
        """
        # Maxwell架构优化
        if architecture == "maxwell":
            self.renderer.batch_enabled = True
            self.renderer.instancing_enabled = True
            self.renderer.max_instanced_draws = 1024
            self.renderer.shadow_map_res = min(self.renderer.shadow_map_res, 1024)
        
        # Pascal架构优化
        elif architecture == "pascal":
            self.renderer.batch_enabled = True
            self.renderer.instancing_enabled = True
            self.renderer.max_instanced_draws = 2048
            self.renderer.shadow_map_res = min(self.renderer.shadow_map_res, 2048)
        
        # Turing架构优化
        elif architecture == "turing":
            self.renderer.rt_cores_enabled = True
            self.renderer.tensor_cores_enabled = True
            self.renderer.dlss_enabled = True
        
        # Ampere架构优化
        elif architecture == "ampere":
            self.renderer.rt_cores_enabled = True
            self.renderer.tensor_cores_enabled = True
            self.renderer.dlss_enabled = True
            self.renderer.max_draw_calls = min(self.renderer.max_draw_calls, 3000)
        
        # Ada架构优化
        elif architecture == "ada":
            self.renderer.rt_cores_enabled = True
            self.renderer.tensor_cores_enabled = True
            self.renderer.dlss_enabled = True
            self.renderer.frame_gen_enabled = True
    
    def _apply_amd_optimizations(self, architecture):
        """
        应用AMD架构特定优化
        
        Args:
            architecture: AMD GPU架构
        """
        # GCN架构优化
        if architecture == "gcn":
            self.renderer.batch_enabled = True
            self.renderer.instancing_enabled = False  # GCN架构实例化性能较差
            self.renderer.max_draw_calls = min(self.renderer.max_draw_calls, 1000)
            self.renderer.shadow_map_res = min(self.renderer.shadow_map_res, 1024)
        
        # GCN5架构优化
        elif architecture == "gcn5":
            self.renderer.batch_enabled = True
            self.renderer.instancing_enabled = True
            self.renderer.max_instanced_draws = 1024
        
        # RDNA1架构优化
        elif architecture == "rdna1":
            self.renderer.batch_enabled = True
            self.renderer.instancing_enabled = True
            self.renderer.max_instanced_draws = 2048
            self.renderer.fidelityfx_enabled = True
        
        # RDNA2架构优化
        elif architecture == "rdna2":
            self.renderer.batch_enabled = True
            self.renderer.instancing_enabled = True
            self.renderer.max_instanced_draws = 4096
            self.renderer.fidelityfx_enabled = True
            self.renderer.ray_tracing_enabled = True
    
    def _apply_intel_optimizations(self, architecture):
        """
        应用Intel架构特定优化
        
        Args:
            architecture: Intel GPU架构
        """
        # Intel HD架构优化
        if architecture == "intel_hd" or architecture == "intel_hd_500" or architecture == "intel_hd_600":
            self.renderer.batch_enabled = True
            self.renderer.instancing_enabled = False
            self.renderer.max_draw_calls = min(self.renderer.max_draw_calls, 500)
            self.renderer.shadow_map_res = min(self.renderer.shadow_map_res, 512)
            self.renderer.ao_enabled = False
            self.renderer.ssr_enabled = False
        
        # Intel Iris架构优化
        elif architecture == "intel_iris" or architecture == "intel_iris_xe":
            self.renderer.batch_enabled = True
            self.renderer.instancing_enabled = True
            self.renderer.max_instanced_draws = 1024
            self.renderer.shadow_map_res = min(self.renderer.shadow_map_res, 1024)
        
        # Intel Arc架构优化
        elif architecture == "intel_arc":
            self.renderer.batch_enabled = True
            self.renderer.instancing_enabled = True
            self.renderer.max_instanced_draws = 2048
            self.renderer.ray_tracing_enabled = True
            self.renderer.xess_enabled = True
    
    def update(self, delta_time):
        """
        更新优化器状态
        
        Args:
            delta_time: 帧间隔时间
        """
        # 可以在这里添加动态调整逻辑，例如根据帧率动态调整渲染质量
        pass
    
    def get_optimization_report(self):
        """
        获取优化报告
        
        Returns:
            dict: 优化报告
        """
        return {
            "render_mode": self.renderer.render_mode.value,
            "render_quality": self.renderer.quality.value,
            "max_draw_calls": self.renderer.max_draw_calls,
            "max_visible_lights": self.renderer.max_visible_lights,
            "shadow_map_resolution": self.renderer.shadow_map_resolution,
            "texture_quality": self.recommended_settings["texture_quality"],
            "texture_resolution_scale": self.renderer.tex_res_scale,
            "enable_ssr": self.renderer.ssr_enabled,
            "enable_volumetric_lighting": self.renderer.volumetric_lighting_enabled,
            "enable_ao": self.renderer.ao_enabled,
            "enable_tessellation": self.renderer.tessellation_enabled,
            "enable_geometry_shaders": self.renderer.geometry_shaders_enabled,
            "enable_compute_shaders": self.renderer.compute_shaders_enabled,
            "enable_ray_tracing": self.renderer.ray_tracing_enabled
        }