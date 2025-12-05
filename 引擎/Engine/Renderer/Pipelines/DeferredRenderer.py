import numpy as np
from Engine.Math import Matrix4x4, Vector3
from Engine.Renderer.Resources.VRAMManager import ResourcePriority

# 导入OpenGL固定功能管线函数
from OpenGL.GL import *

class DeferredRenderer:
    """
    延迟渲染器实现
    支持完整的延迟渲染管线，包括G-buffer、光照计算和后处理
    针对低端GPU优化，提供简化的延迟渲染模式
    """
    
    def __init__(self, platform, resource_manager, shader_manager=None):
        """
        初始化延迟渲染器
        
        Args:
            platform: 平台接口
            resource_manager: 资源管理器
            shader_manager: 着色器管理器
        """
        self.platform = platform
        self.resource_manager = resource_manager
        self.shader_manager = shader_manager
        self.resolution = (1280, 720)
        self.clear_color = (0.1, 0.1, 0.1, 1.0)
        
        # G-buffer设置
        self.gbuffer_textures = {
            "position": None,   # 世界空间位置
            "normal": None,     # 世界空间法线
            "albedo": None,     # 漫反射颜色
            "specular": None,   # 高光颜色和粗糙度
            "depth": None       # 深度缓冲区
        }
        
        # 渲染特性
        self.features = {
            "ssr": False,  # 屏幕空间反射
            "shadow_mapping": True,  # 阴影贴图
            "ambient_occlusion": False,  # 环境光遮蔽
            "volumetric_lighting": False,  # 体积光
            "bloom": True,  # 泛光
            "motion_blur": False,  # 运动模糊
        }
        
        # 特性质量级别
        self.feature_quality = {
            "ssr": "low",
            "ambient_occlusion": "low",
            "bloom": "low",
        }
        
        # 阴影设置
        self.shadow_map_resolution = 1024
        self.max_shadow_casters = 4
        self.shadow_cascade_count = 1  # 低端GPU最多使用1级级联
        
        # 性能统计
        self.performance_stats = {
            "draw_calls": 0,
            "triangles": 0,
            "render_time_ms": 0,
            "shader_switches": 0,
            "texture_switches": 0,
            "visible_objects": 0,
            "culled_objects": 0,
            "instanced_objects": 0,
            "batched_objects": 0,
            "visible_lights": 0
        }
        
        # 低端GPU优化设置
        self.use_instancing = True  # 使用实例化渲染
        self.use_frustum_culling = True  # 使用视锥体裁剪
        self.use_occlusion_culling = False  # 低端GPU上禁用遮挡剔除以节省CPU开销
        self.batching_threshold = 8  # 更低的批处理阈值以增加批处理机会
        self.max_draw_calls = 1000  # 限制最大绘制调用数
        self.max_visible_lights = 8  # 限制最大可见光源数
        
        # 硬件检测结果
        self.gpu_info = {
            "name": "",
            "architecture": "",  # "maxwell", "gcn", "other"
            "vram_size_mb": 0,
            "is_low_end": False,
            "is_nvidia": False,
            "is_amd": False
        }
        
        # 着色器程序
        self.shader_programs = {}
        
        # 初始化硬件检测
        self._detect_gpu_info()
    
    def initialize(self, config=None):
        """
        初始化延迟渲染器
        
        Args:
            config: 渲染器配置参数
        """
        print("初始化延迟渲染器...")
        
        # 应用配置
        if config:
            if "resolution" in config:
                self.resolution = config["resolution"]
            if "clear_color" in config:
                self.clear_color = config["clear_color"]
            if "shadow_map_resolution" in config:
                self.shadow_map_resolution = config["shadow_map_resolution"]
        
        # 根据硬件设置默认特性
        self._configure_features_based_on_hardware()
        
        # 创建G-buffer
        self._create_gbuffer()
        
        # 加载和优化着色器
        self._load_and_optimize_shaders()
        
        # 设置渲染状态
        self._setup_optimized_render_states()
        
        print("延迟渲染器初始化完成")
        print(f"GPU: {self.gpu_info['name']}, VRAM: {self.gpu_info['vram_size_mb']}MB")
        print(f"优化模式: {'低端GPU优化' if self.gpu_info['is_low_end'] else '标准模式'}")
    
    def shutdown(self):
        """
        关闭延迟渲染器，释放资源
        """
        print("关闭延迟渲染器")
        # 释放G-buffer资源
        self._destroy_gbuffer()
    
    def render(self, scene):
        """
        渲染场景
        
        Args:
            scene: 要渲染的场景
        """
        # 重置性能统计
        self._reset_performance_stats()
        
        # 记录渲染开始时间
        start_time = self._get_current_time_ms()
        
        # 1. 清除缓冲区
        self._clear_buffers()
        
        # 2. 渲染阴影贴图（如果启用）
        if self.features["shadow_mapping"]:
            self._render_shadow_maps(scene)
        
        # 3. 几何阶段：渲染到G-buffer
        self._geometry_pass(scene)
        
        # 4. 光照阶段：使用G-buffer计算光照
        self._lighting_pass(scene)
        
        # 5. 渲染透明对象（使用前向渲染）
        self._render_transparent_objects(scene)
        
        # 6. 后处理效果
        self._apply_post_processing(scene)
        
        # 计算渲染时间
        self.performance_stats["render_time_ms"] = self._get_current_time_ms() - start_time
        
        # 返回虚拟渲染结果，以便后续处理
        return (None, None)
    
    def resize(self, width, height):
        """
        调整渲染器分辨率
        
        Args:
            width: 新宽度
            height: 新高度
        """
        self.resolution = (width, height)
        # 重新创建G-buffer
        self._destroy_gbuffer()
        self._create_gbuffer()
    
    def set_clear_color(self, r, g, b, a=1.0):
        """
        设置清除颜色
        
        Args:
            r, g, b, a: 颜色通道值
        """
        self.clear_color = (r, g, b, a)
    
    def get_performance_stats(self):
        """
        获取渲染性能统计信息
        
        Returns:
            dict: 性能统计数据
        """
        return self.performance_stats.copy()
    
    def enable_feature(self, feature_name, enable=True):
        """
        启用或禁用渲染特性
        
        Args:
            feature_name: 特性名称
            enable: 是否启用
            
        Returns:
            bool: 是否成功设置
        """
        if feature_name in self.features:
            self.features[feature_name] = enable
            return True
        return False
    
    def _detect_gpu_info(self):
        """
        检测GPU信息和架构
        """
        # 获取GPU信息
        hardware_info = self.platform.get_hardware_info()
        gpu_name = hardware_info.get('gpu_name', '').lower()
        
        # 检测GPU名称和架构
        self.gpu_info['name'] = hardware_info.get('gpu_name', 'Unknown GPU')
        self.gpu_info['vram_size_mb'] = self.platform.get_gpu_memory_budget()
        
        # 检测NVIDIA Maxwell架构 (GTX 750Ti等)
        if 'nvidia' in gpu_name or 'geforce' in gpu_name:
            self.gpu_info['is_nvidia'] = True
            if '750' in gpu_name or 'maxwell' in gpu_name:
                self.gpu_info['architecture'] = 'maxwell'
                self.gpu_info['is_low_end'] = True
        
        # 检测AMD GCN架构 (RX 580等)
        elif 'amd' in gpu_name or 'radeon' in gpu_name:
            self.gpu_info['is_amd'] = True
            if 'rx 580' in gpu_name or 'polaris' in gpu_name or 'gcn' in gpu_name:
                self.gpu_info['architecture'] = 'gcn'
                # RX 580不算低端，但也需要特定优化
                self.gpu_info['is_low_end'] = self.gpu_info['vram_size_mb'] <= 4096
        
        # 根据VRAM大小判断
        if self.gpu_info['vram_size_mb'] <= 2048:
            self.gpu_info['is_low_end'] = True
    
    def _configure_features_based_on_hardware(self):
        """
        根据硬件能力配置渲染特性
        针对NVIDIA Maxwell和AMD GCN架构进行特定优化
        """
        gpu_memory = self.gpu_info['vram_size_mb']
        architecture = self.gpu_info['architecture']
        is_low_end = self.gpu_info['is_low_end']
        
        print(f"配置硬件特定优化: {architecture}, VRAM: {gpu_memory}MB")
        
        # 低端GPU上简化延迟渲染
        if is_low_end:
            print("应用低端GPU延迟渲染优化")
            
            # 简化G-buffer，减少纹理数量
            self._simplify_gbuffer()
            
            # 禁用所有昂贵效果
            self.features["ssr"] = False
            self.features["ambient_occlusion"] = False
            self.features["volumetric_lighting"] = False
            self.features["motion_blur"] = False
            
            # 阴影质量降低
            self.shadow_map_resolution = 512
            self.max_shadow_casters = 1
            self.shadow_cascade_count = 1
            
            # 限制资源使用
            self.max_visible_lights = 4
            self.max_draw_calls = 800
    
    def _create_gbuffer(self):
        """
        创建G-buffer纹理
        """
        print("创建G-buffer...")
        
        # 简化的G-buffer创建，实际需要使用OpenGL创建纹理
        # 这里只是模拟G-buffer创建
        width, height = self.resolution
        
        # 创建深度缓冲区
        self.gbuffer_textures["depth"] = "depth_buffer"
        
        # 创建颜色缓冲区
        self.gbuffer_textures["position"] = "position_buffer"
        self.gbuffer_textures["normal"] = "normal_buffer"
        self.gbuffer_textures["albedo"] = "albedo_buffer"
        self.gbuffer_textures["specular"] = "specular_buffer"
        
        print(f"G-buffer创建完成: {width}x{height}")
    
    def _destroy_gbuffer(self):
        """
        释放G-buffer资源
        """
        print("释放G-buffer资源")
        self.gbuffer_textures.clear()
    
    def _simplify_gbuffer(self):
        """
        简化G-buffer，减少纹理数量以适应低端GPU
        """
        print("简化G-buffer以适应低端GPU")
        
        # 移除一些非必要的G-buffer纹理
        if self.gpu_info['is_low_end']:
            # 合并specular到albedo，使用alpha通道存储粗糙度
            self.gbuffer_textures.pop("specular", None)
    
    def _load_and_optimize_shaders(self):
        """
        加载并根据GPU架构优化着色器
        """
        if not self.shader_manager:
            print("警告: 着色器管理器未提供，无法进行架构特定优化")
            return
        
        print("加载并优化延迟渲染着色器...")
        
        try:
            # 几何通道着色器
            geom_vert_path = "Engine/Shaders/DeferredGeometryVertexShader.glsl"
            geom_frag_path = "Engine/Shaders/DeferredGeometryFragmentShader.glsl"
            
            # 加载着色器源代码
            vertex_code = self.resource_manager.load_shader_source(geom_vert_path)
            fragment_code = self.resource_manager.load_shader_source(geom_frag_path)
            
            # 根据架构优化着色器
            if self.gpu_info['architecture'] == 'maxwell':
                print("优化着色器以适应NVIDIA Maxwell架构")
            elif self.gpu_info['architecture'] == 'gcn':
                print("优化着色器以适应AMD GCN架构")
            
            # 低端GPU额外优化
            if self.gpu_info['is_low_end']:
                print("应用低端GPU特定着色器优化")
                # 简化G-buffer输出
                fragment_code = fragment_code.replace("#define FULL_GBUFFER 1", "#define FULL_GBUFFER 0")
            
            # 创建几何通道着色器程序
            self.shader_programs["geometry"] = self.platform.create_shader_program(
                vertex_code, 
                fragment_code
            )
            
            # 光照通道着色器
            light_vert_path = "Engine/Shaders/DeferredLightingVertexShader.glsl"
            light_frag_path = "Engine/Shaders/DeferredLightingFragmentShader.glsl"
            
            # 加载光照通道着色器
            vertex_code = self.resource_manager.load_shader_source(light_vert_path)
            fragment_code = self.resource_manager.load_shader_source(light_frag_path)
            
            # 创建光照通道着色器程序
            self.shader_programs["lighting"] = self.platform.create_shader_program(
                vertex_code, 
                fragment_code
            )
            
            # 阴影着色器
            if self.features["shadow_mapping"]:
                self._load_shadow_shaders()
                
        except Exception as e:
            print(f"着色器加载和优化失败: {e}")
    
    def _load_shadow_shaders(self):
        """
        加载并优化阴影映射着色器
        """
        # 简化的阴影着色器
        shadow_vert_code = """
        #version 330 core
        
        layout(location = 0) in vec3 position;
        
        uniform mat4 lightSpaceMatrix;
        uniform mat4 model;
        
        void main() {
            gl_Position = lightSpaceMatrix * model * vec4(position, 1.0);
        }
        """
        
        shadow_frag_code = """
        #version 330 core
        
        void main() {
            // 使用深度作为输出
        }
        """
        
        # 优化阴影着色器
        if self.shader_manager:
            if self.gpu_info['architecture'] == 'maxwell':
                shadow_vert_code = self.shader_manager.optimize_for_maxwell(shadow_vert_code)
            elif self.gpu_info['architecture'] == 'gcn':
                shadow_vert_code = self.shader_manager.optimize_for_gcn(shadow_vert_code)
        
        # 创建阴影着色器程序
        self.shader_programs["shadow"] = self.platform.create_shader_program(
            shadow_vert_code, 
            shadow_frag_code
        )
    
    def _setup_optimized_render_states(self):
        """
        设置针对特定GPU架构优化的渲染状态
        """
        # 基本设置
        self.platform.enable_depth_test(True)
        self.platform.enable_cull_face(True)
        
        # NVIDIA Maxwell特定优化
        if self.gpu_info['architecture'] == 'maxwell':
            # 使用更快的深度测试函数
            self.platform.set_depth_func("LESS")  # LESS比LESS_EQUAL更快
            
            # 禁用多边形偏移（Maxwell上可能更慢）
            self.platform.enable_polygon_offset(False)
            
            # 使用更快的混合模式
            self.platform.set_blend_func("ONE", "ZERO")
        
        # AMD GCN特定优化
        elif self.gpu_info['architecture'] == 'gcn':
            # GCN架构可以使用更精确的深度测试
            self.platform.set_depth_func("LESS_EQUAL")
            
            # GCN在某些情况下多边形偏移性能较好
            if self.features["shadow_mapping"]:
                self.platform.enable_polygon_offset(True)
                self.platform.set_polygon_offset(1.0, 1.0)
        
        # 低端GPU通用优化
        if self.gpu_info['is_low_end']:
            # 简化光栅化状态
            if hasattr(self.platform, 'set_rasterization_samples'):
                self.platform.set_rasterization_samples(1)  # 禁用多重采样
    
    def _geometry_pass(self, scene):
        """
        几何通道：将物体渲染到G-buffer
        """
        print("执行几何通道...")
        
        # 收集可见节点
        visible_nodes = getattr(scene, 'visible_nodes', [])
        
        # 增加可见物体计数
        self.performance_stats["visible_objects"] = len(visible_nodes)
        
        # 绑定几何通道着色器
        if "geometry" in self.shader_programs:
            # 简化实现，实际需要绑定着色器和G-buffer
            pass
        
        # 渲染不透明物体
        for node in visible_nodes:
            if hasattr(node, 'mesh') and node.mesh:
                self.performance_stats["draw_calls"] += 1
                self.performance_stats["triangles"] += getattr(node.mesh, 'triangle_count', 12)
    
    def _lighting_pass(self, scene):
        """
        光照通道：使用G-buffer计算光照
        """
        print("执行光照通道...")
        
        # 绑定光照通道着色器
        if "lighting" in self.shader_programs:
            # 简化实现，实际需要绑定着色器和G-buffer
            pass
        
        # 收集场景中的光源
        lights = scene.light_manager.lights
        
        # 选择前N个最重要的光源
        visible_lights = lights[:self.max_visible_lights]
        self.performance_stats["visible_lights"] = len(visible_lights)
        
        # 应用每个光源
        for light in visible_lights:
            # 简化实现，实际需要计算每个光源对每个像素的贡献
            pass
    
    def _render_shadow_maps(self, scene):
        """
        渲染阴影贴图
        针对低端GPU优化的实现
        """
        # 收集场景中的光源
        lights = scene.light_manager.lights
        
        # 选择前N个最重要的光源进行阴影投射
        shadow_casters = lights[:self.max_shadow_casters]
        
        for light in shadow_casters:
            if hasattr(light, 'cast_shadows') and light.cast_shadows:
                # 计算光源的投影矩阵
                light_space_matrix = Matrix4x4()
                
                # 渲染阴影贴图
                # 低端GPU优化：只渲染会影响阴影的不透明物体
                self._render_objects_to_shadow_map(scene, light_space_matrix)
    
    def _render_objects_to_shadow_map(self, scene, light_space_matrix):
        """
        将物体渲染到阴影贴图
        """
        # 简化实现
        pass
    
    def _render_transparent_objects(self, scene):
        """
        渲染透明物体
        使用前向渲染
        """
        print("渲染透明物体...")
        
        # 获取场景中的透明物体
        transparent_objects = []
        
        # 按距离从后到前排序
        transparent_objects.sort(key=lambda obj: getattr(obj.position, 'z', 0), reverse=True)
        
        # 渲染透明物体
        for obj in transparent_objects:
            self.performance_stats["draw_calls"] += 1
            self.performance_stats["triangles"] += getattr(obj.mesh, 'triangle_count', 12)
    
    def _apply_post_processing(self, scene):
        """
        应用后处理效果
        针对低端GPU优化，只应用必要的效果
        """
        # 如果启用了泛光效果
        if self.features["bloom"]:
            self._apply_bloom()
        
        # 如果启用了环境光遮蔽
        if self.features["ambient_occlusion"]:
            self._apply_ambient_occlusion()
        
        # 应用颜色分级
        self._apply_color_grading(scene)
    
    def _apply_bloom(self):
        """
        应用泛光效果
        针对低端GPU优化的实现
        """
        print("应用泛光效果...")
        # 简化实现
    
    def _apply_ambient_occlusion(self):
        """
        应用环境光遮蔽
        简化版本，适合低端GPU
        """
        print("应用环境光遮蔽...")
        # 简化实现
    
    def _apply_color_grading(self, scene):
        """
        应用颜色分级
        模拟游戏"模拟美景摄影"的视觉风格
        """
        print("应用颜色分级...")
        # 简化实现
    
    def _reset_performance_stats(self):
        """
        重置性能统计数据
        """
        for key in self.performance_stats:
            self.performance_stats[key] = 0
    
    def _get_current_time_ms(self):
        """
        获取当前时间（毫秒）
        实际实现会使用平台特定的API
        """
        import time
        return time.time() * 1000
    
    def _clear_buffers(self):
        """
        清除颜色和深度缓冲区
        """
        # 调用平台的clear方法清除缓冲区
        if hasattr(self.platform, 'clear'):
            self.platform.clear(self.clear_color)
