# 确保正确导入Math模块
import sys
import os
# 添加引擎根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .Pipelines.ForwardRenderer import ForwardRenderer
from .Pipelines.DeferredRenderer import DeferredRenderer
from .Pipelines.HybridRenderer import HybridRenderer
from .Resources.VRAMManager import VRAMManager
from .Shaders.ShaderManager import ShaderManager
from .Shaders.ShaderOptimizer import ShaderOptimizer
from .Effects.EffectManager import EffectManager
from enum import Enum
import concurrent.futures

# 使用Python标准库的ThreadPoolExecutor替代外部导入
from concurrent.futures import ThreadPoolExecutor as ThreadPool

class RenderMode(Enum):
    """光影魔法的施展方式"""
    FORWARD = "forward"      # 直接魔法：实时计算每个像素的光影
    DEFERRED = "deferred"    # 延迟魔法：先收集信息，再统一处理
    HYBRID = "hybrid"        # 混合魔法：前两种魔法的完美融合
    LOW_END = "low_end"      # 节能魔法：为老显卡量身定制的超高效模式

class RenderQuality(Enum):
    """光影魔法的精细程度"""
    ULTRA_LOW = "ultra_low"  # 魔法学徒模式：720p，最小内存消耗
    LOW = "low"              # 初级魔法师：720p，适合GTX 750Ti
    MEDIUM = "medium"        # 中级魔法师：1080p，适合RX 580
    HIGH = "high"            # 高级魔法师：1080p，适合现代硬件

class Renderer:
    """
    光影魔法师！负责把3D世界变成美丽的画面
    专为让老显卡也能施展光影魔法而设计
    就像给GTX 750Ti和RX 580装上了魔法加速器
    """
    
    def __init__(self, plt, res_mgr):
        self.plt = plt  # 平台接口
        self.res_mgr = res_mgr  # 资源管理器
        self.vram_mgr = VRAMManager(plt)  # VRAM管理器
        
        # 获取硬件信息
        self.hw_info = self.plt.get_hardware_info()
        
        # 初始化着色器管理器，传递硬件信息字典
        self.shader_mgr = ShaderManager(self.hw_info)
        
        # 初始化着色器优化器，传递GPU架构字符串
        self.shader_optimizer = ShaderOptimizer(self.hw_info.get("gpu_architecture", "default"))
        
        self.curr_pipeline = None  # 当前渲染管线
        self.pipelines = {}  # 渲染管线集合
        self.res = (1280, 720)  # 默认分辨率 resolution → res
        self.vp = (0, 0, 1280, 720)  # 视口 viewport → vp
        self.clear_clr = (0.1, 0.1, 0.1, 1.0)  # 清除颜色 clear_color → clear_clr
        self.render_mode = RenderMode.FORWARD
        self.quality = RenderQuality.LOW  # render_quality → quality
        self.max_draw_calls = 1000  # 针对GTX 750Ti优化的最大绘制调用数
        self.max_visible_lights = 8  # 最大可见光源数
        self.shadow_map_res = 1024  # 阴影贴图分辨率 shadow_map_resolution → shadow_map_res
        self.batch_enabled = True  # 批处理 enabled is_batch_rendering_enabled → batch_enabled
        self.instancing_enabled = True  # 实例化 enabled
        self.frame_count = 0
        
        # 渲染特性开关
        self.ssr_enabled = False  # 屏幕空间反射
        self.volumetric_lighting_enabled = False  # 体积光照
        self.ao_enabled = False  # 环境光遮蔽
        self.tessellation_enabled = False  # 曲面细分
        self.geometry_shaders_enabled = False  # 几何着色器
        self.compute_shaders_enabled = False  # 计算着色器
        self.ray_tracing_enabled = False  # 光线追踪
        self.mesh_shaders_enabled = False  # 网格着色器
        self.vrs_enabled = False  # 可变速率着色 variable_rate_shading → vrs
        
        # 纹理设置
        self.tex_res_scale = 1.0  # 纹理分辨率缩放 texture_resolution_scale → tex_res_scale
        self.tex_compression = "bc7"  # 纹理压缩格式 texture_compression → tex_compression
        
        # LOD设置
        self.lod_bias = 0.0  # LOD偏差
        
        # 架构特定特性
        self.rt_cores_enabled = False  # RT核心支持
        self.tensor_cores_enabled = False  # Tensor核心支持
        self.dlss_enabled = False  # DLSS支持
        self.frame_gen_enabled = False  # 帧生成支持
        self.fidelityfx_enabled = False  # AMD FidelityFX支持
        self.xess_enabled = False  # Intel XeSS支持
        
        # 实例化设置
        self.max_instanced_draws = 2048  # 最大实例化绘制数
        
        # 初始化效果管理器
        self.effect_mgr = EffectManager(self)
        
        # 初始化线程池，用于并行处理渲染任务
        self.thread_pool = ThreadPool(max_workers=4)  # 根据硬件自动调整线程数
    
    def initialize(self, config=None):
        """
        初始化渲染器
        
        Args:
            config: 渲染器配置参数
        """
        from Engine.Logger import get_logger
        logger = get_logger("Renderer")
        
        try:
            logger.info("初始化渲染器...")
            logger.info(f"目标硬件: {self.hw_info['gpu_name']}, VRAM: {self.hw_info['vram_mb']}MB")
            
            # 应用配置
            if config:
                if "resolution" in config:
                    self.res = config["resolution"]
                    self.vp = (0, 0, self.res[0], self.res[1])
                if "clear_color" in config:
                    self.clear_clr = config["clear_color"]
                if "render_mode" in config:
                    self.render_mode = RenderMode(config["render_mode"])
                if "render_quality" in config:
                    self.quality = RenderQuality(config["render_quality"])
            
            # 根据硬件自动选择最佳渲染质量
            self._auto_select_render_quality()
            
            # 创建并初始化渲染管道
            self._create_render_pipelines(config)
            
            # 应用硬件特定优化
            self._apply_hardware_specific_optimizations()
            
            # 初始化GPU能力评估器
            from Engine.Renderer.Optimization.GPUCapabilityEvaluator import GPUCapabilityEvaluator
            self.gpu_evaluator = GPUCapabilityEvaluator(self.hw_info)
            
            # 初始化渲染管线优化器
            from Engine.Renderer.Optimization.RenderPipelineOptimizer import RenderPipelineOptimizer
            self.pipeline_optimizer = RenderPipelineOptimizer(self, self.gpu_evaluator)
            
            logger.info(f"渲染器初始化完成，使用 {self.curr_pipeline.__class__.__name__}")
            logger.info(f"渲染模式: {self.render_mode.value}, 质量级别: {self.quality.value}")
            logger.info(f"最大绘制调用: {self.max_draw_calls}, 最大可见光源: {self.max_visible_lights}")
            logger.info(f"GPU能力等级: {self.gpu_evaluator.get_capability_level()}, 性能评分: {self.gpu_evaluator.get_performance_score():.2f}")
        except Exception as e:
            logger.error(f"渲染器初始化失败: {e}", exc_info=True)
            raise
    
    def shutdown(self):
        """
        关闭渲染器，释放资源
        """
        from Engine.Logger import get_logger
        logger = get_logger("Renderer")
        
        try:
            logger.info("关闭渲染器...")
            
            # 关闭线程池
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=True)
            
            # 关闭所有渲染管道
            for pipeline in self.pipelines.values():
                try:
                    pipeline.shutdown()
                except Exception as e:
                    logger.error(f"关闭渲染管道失败: {e}", exc_info=True)
            
            # 关闭VRAM管理器和着色器系统
            try:
                self.vram_mgr.shutdown()
            except Exception as e:
                logger.error(f"关闭VRAM管理器失败: {e}", exc_info=True)
            
            try:
                self.shader_mgr.shutdown()
            except Exception as e:
                logger.error(f"关闭着色器管理器失败: {e}", exc_info=True)
            
            self.pipelines.clear()
            self.curr_pipeline = None
            
            logger.info("渲染器关闭完成")
        except Exception as e:
            logger.error(f"渲染器关闭失败: {e}", exc_info=True)
    
    def render(self, scene):
        from Engine.Logger import get_logger
        logger = get_logger("Renderer")
        
        if not self.curr_pipeline:
            logger.warning("没有可用的渲染管道，跳过渲染")
            return
        
        try:
            self.vram_mgr.frame_begin()
            
            camera = scene.active_camera
            futures = []
            
            if camera:
                def update_lods():
                    self._update_scene_lods(scene, camera)
                futures.append(self.thread_pool.submit(update_lods))
            
            if self.batch_enabled:
                def prepare_batching():
                    self._prepare_batch_rendering(scene)
                futures.append(self.thread_pool.submit(prepare_batching))
            
            def update_physics():
                if hasattr(scene, 'physics_world') and scene.physics_world:
                    scene.physics_world.update(1.0/60.0)
            futures.append(self.thread_pool.submit(update_physics))
            
            if futures:
                concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)
            
            result = self.curr_pipeline.render(scene)
            
            if result is not None:
                color_buffer, depth_buffer = result
                if color_buffer and depth_buffer:
                    final_buffer = self.effect_mgr.apply_effects(
                        color_buffer, 
                        depth_buffer, 
                        scene, 
                        camera,
                        self.quality
                    )
                    
                    # 渲染到屏幕
                    self._render_to_screen(final_buffer)
            
            # 帧结束后的VRAM管理
            self.vram_mgr.frame_end()
            self.frame_count += 1
        except Exception as e:
            logger.error(f"渲染过程中发生错误: {e}", exc_info=True)
            # 确保VRAM管理器在发生错误时也能正确结束帧
            try:
                self.vram_mgr.frame_end()
            except Exception as vram_error:
                logger.error(f"VRAM管理器帧结束时发生错误: {vram_error}", exc_info=True)
    
    def set_pipeline(self, pipeline_name):
        """
        设置当前使用的渲染管道
        
        Args:
            pipeline_name: 管道名称
            
        Returns:
            bool: 是否成功切换管道
        """
        # 导入日志系统
        from Engine.Logger import get_logger
        logger = get_logger("Renderer")
        
        if pipeline_name in self.render_pipelines:
            self.current_pipeline = self.render_pipelines[pipeline_name]
            # 切换管道时重新应用硬件优化
            self._apply_pipeline_optimizations(self.current_pipeline)
            logger.info(f"切换到 {pipeline_name} 渲染管道")
            return True
        return False
    
    def resize(self, width, height):
        """
        调整渲染器分辨率
        
        Args:
            width: 新宽度
            height: 新高度
        """
        self.res = (width, height)
        self.vp = (0, 0, width, height)
        
        # 通知所有渲染管道调整大小
        for pipeline in self.pipelines.values():
            pipeline.resize(width, height)
        
        # 重新计算VRAM预算
        if hasattr(self.vram_mgr, 'update_memory_budget'):
            self.vram_mgr.update_memory_budget()
    
    def set_clear_color(self, r, g, b, a=1.0):
        """
        设置清除颜色
        
        Args:
            r, g, b, a: 颜色通道值
        """
        self.clear_color = (r, g, b, a)
    
    def get_render_result(self):
        """
        获取当前渲染结果，转换为base64编码的图像数据
        
        Returns:
            str: base64编码的图像数据，格式为data:image/png;base64,...
        """
        from Engine.Logger import get_logger
        logger = get_logger("Renderer")
        
        try:
            # 检查OpenGL上下文是否有效
            try:
                from OpenGL.GL import glGetError, GL_NO_ERROR
                # 尝试调用一个简单的OpenGL函数来检查上下文是否有效
                glGetError()  # 这个函数不会修改上下文状态
            except Exception as ctx_err:
                logger.debug(f"OpenGL上下文无效: {ctx_err}")
                return None
            
            # 从当前渲染缓冲区读取像素数据
            from OpenGL.GL import (
                glReadPixels, GL_RGB, GL_UNSIGNED_BYTE, glGetIntegerv, GL_VIEWPORT
            )
            import numpy as np
            import base64
            from io import BytesIO
            from PIL import Image
            
            # 获取当前视口大小
            viewport = glGetIntegerv(GL_VIEWPORT)
            width, height = viewport[2], viewport[3]
            
            # 如果视口大小为0，跳过渲染结果获取
            if width <= 0 or height <= 0:
                logger.debug(f"无效的视口大小: {width}x{height}")
                return None
            
            # 读取像素数据
            pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
            
            # 将像素数据转换为numpy数组
            img_data = np.frombuffer(pixels, dtype=np.uint8)
            img_data = img_data.reshape(height, width, 3)
            
            # 翻转图像，因为OpenGL读取的像素数据是上下颠倒的
            img_data = np.flipud(img_data)
            
            # 转换为RGB格式
            img = Image.fromarray(img_data, 'RGB')
            
            # 将图像转换为base64编码的字符串
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            # 返回base64编码的图像数据
            return f"data:image/png;base64,{img_str}"
        except Exception as e:
            # 只记录一次错误，避免日志大量输出
            logger.error(f"获取渲染结果失败: {e}")
            return None
    
    def get_performance_stats(self):
        """
        获取渲染性能统计信息
        
        Returns:
            dict: 性能统计数据
        """
        if not self.curr_pipeline:
            return {
                "render_time_ms": 0,
                "draw_calls": 0,
                "triangles": 0,
                "visible_objects": 0,
                "culled_objects": 0,
                "vram_usage": 0,
                "vram_budget": 0,
                "visible_lights": 0
            }
        
        try:
            stats = self.curr_pipeline.get_performance_stats()
            # 添加VRAM使用情况
            stats['vram_usage'] = self.vram_mgr.current_vram_usage
            stats['vram_budget'] = self.vram_mgr.current_vram_limit
            stats['draw_calls'] = stats.get('draw_calls', 0)
            stats['visible_objects'] = stats.get('visible_objects', 0)
            stats['visible_lights'] = stats.get('visible_lights', 0)
            
            return stats
        except Exception as e:
            from Engine.Logger import get_logger
            logger = get_logger("Renderer")
            logger.error(f"获取性能统计失败: {e}", exc_info=True)
            return {
                "render_time_ms": 0,
                "draw_calls": 0,
                "triangles": 0,
                "visible_objects": 0,
                "culled_objects": 0,
                "vram_usage": 0,
                "vram_budget": 0,
                "visible_lights": 0
            }
    
    def enable_feature(self, feature_name, enable=True):
        """
        启用或禁用渲染特性
        
        Args:
            feature_name: 特性名称
            enable: 是否启用
            
        Returns:
            bool: 是否成功设置
        """
        # 检查是否是效果管理器管理的特性
        effect_features = ['fagi', 'ssr', 'ambient_occlusion']
        if feature_name in effect_features:
            result = self.effect_mgr.enable_effect(feature_name, enable)
            # 特性变更可能影响内存使用，更新VRAM预算
            if result and hasattr(self.vram_mgr, 'update_memory_budget'):
                self.vram_mgr.update_memory_budget()
            return result
        
        # 其他特性由渲染管道处理
        if not self.curr_pipeline:
            return False
        
        result = self.curr_pipeline.enable_feature(feature_name, enable)
        
        # 特性变更可能影响内存使用，更新VRAM预算
        if result and feature_name in ['ssr', 'shadow_mapping', 'volumetric_lighting'] and hasattr(self.vram_mgr, 'update_memory_budget'):
            self.vram_mgr.update_memory_budget()
        
        return result
    
    def optimize_for_hardware(self):
        """
        根据当前硬件自动优化渲染设置
        """
        # 临时变量，用于调试和日志记录
        # TODO: 这个变量可能需要在某个地方使用，先留着
        debug_opt = False
        
        print("针对当前硬件优化渲染设置...")
        
        # 获取GPU内存预算
        # 这里之前遇到过获取失败的情况，所以加了try-except，后来又去掉了
        # try:
        #     gpu_memory_budget = self.platform.get_gpu_memory_budget()
        # except:
        #     gpu_memory_budget = 2048  # 默认2GB
        gpu_memory_budget = self.plt.get_gpu_memory_budget()
        gpu_name = self.hw_info['gpu_name'].lower()
        
        # 针对NVIDIA Maxwell架构（如GTX 750Ti）的特定优化
        is_maxwell = "750" in gpu_name or "maxwell" in gpu_name
        # 针对AMD GCN架构（如RX 580）的特定优化
        is_gcn = "rx" in gpu_name and "580" in gpu_name or "gcn" in gpu_name
        
        # 低端GPU优化（2GB VRAM）
        if gpu_memory_budget <= 2048:  # 2GB或更少的VRAM
            print("应用低端GPU优化设置（2GB VRAM）")
            
            # 降低纹理分辨率
            # 这里原本有个旧的方法，后来改了，但是注释留着，方便回退
            # self.texture_resolution_scale = 0.5  # 旧的方法，已废弃
            if hasattr(self.res_mgr, 'set_texture_quality'):
                self.res_mgr.set_texture_quality("low")
            if hasattr(self.vram_mgr, 'set_texture_compression_level'):
                self.vram_mgr.set_texture_compression_level(2)  # 最高压缩级别
            
            # 启用更激进的几何简化
            if hasattr(self.res_mgr, 'set_mesh_lod_bias'):
                self.res_mgr.set_mesh_lod_bias(2.0)
            # 这里之前设置的是1000，后来改成800，因为GTX 750Ti处理不了那么多
            self.max_draw_calls = 800  # 更严格的绘制调用限制，之前是1000
            self.max_visible_lights = 4  # 减少光源数量，之前是8
            self.shadow_map_res = 512  # 最低阴影分辨率，之前是1024
            
            # 强制使用低端渲染模式
            # 这里之前用的是FORWARD，后来改成LOW_END
            self.render_mode = RenderMode.LOW_END
            self.quality = RenderQuality.ULTRA_LOW
            
            # 禁用几乎所有高级效果
            if self.curr_pipeline:
                # 这里之前遇到过curr_pipeline为None的情况，所以加了判断
                self.curr_pipeline.enable_feature("ssr", False)
                self.curr_pipeline.enable_feature("volumetric_lighting", False)
                self.curr_pipeline.enable_feature("reflection_probes", False)
                self.curr_pipeline.enable_feature("bloom", False)
                self.curr_pipeline.enable_feature("shadow_mapping", True)
                if hasattr(self.curr_pipeline, 'set_shadow_map_resolution'):
                    self.curr_pipeline.set_shadow_map_resolution(512)
                if hasattr(self.curr_pipeline, 'set_shadow_cascade_count'):
                    self.curr_pipeline.set_shadow_cascade_count(1)
                # 额外的效果禁用，暂时注释掉，万一以后需要
                # self.curr_pipeline.enable_feature("motion_blur", False)
                # self.curr_pipeline.enable_feature("depth_of_field", False)
        
        # 中端GPU优化（4GB VRAM）
        elif gpu_memory_budget <= 4096:  # 4GB VRAM
            print("应用中端GPU优化设置（4GB VRAM）")
            
            # 中等纹理质量
            if hasattr(self.res_mgr, 'set_texture_quality'):
                self.res_mgr.set_texture_quality("medium")
            if hasattr(self.vram_mgr, 'set_texture_compression_level'):
                self.vram_mgr.set_texture_compression_level(1)  # 中等压缩级别
            
            # 适中的几何简化
            if hasattr(self.res_mgr, 'set_mesh_lod_bias'):
                self.res_mgr.set_mesh_lod_bias(1.0)
            self.max_draw_calls = 1200
            self.max_visible_lights = 8
            self.shadow_map_res = 1024
            
            # 选择合适的渲染质量
            if is_gcn:  # RX 580可以处理更高质量
                self.quality = RenderQuality.MEDIUM
            else:
                self.quality = RenderQuality.LOW
            
            # 启用部分高级效果但降低质量
            if self.curr_pipeline:
                self.curr_pipeline.enable_feature("ssr", True)
                if hasattr(self.curr_pipeline, 'set_ssr_quality'):
                    self.curr_pipeline.set_ssr_quality("low")
                self.curr_pipeline.enable_feature("volumetric_lighting", False)
                self.curr_pipeline.enable_feature("bloom", True)
                if hasattr(self.curr_pipeline, 'set_bloom_quality'):
                    self.curr_pipeline.set_bloom_quality("low")
                if hasattr(self.curr_pipeline, 'set_shadow_cascade_count'):
                    self.curr_pipeline.set_shadow_cascade_count(2)
        
        else:  # 高端GPU
            print("应用高端GPU优化设置")
            # 最高纹理质量
            if hasattr(self.res_mgr, 'set_texture_quality'):
                self.res_mgr.set_texture_quality("high")
            if hasattr(self.vram_mgr, 'set_texture_compression_level'):
                self.vram_mgr.set_texture_compression_level(0)  # 最低压缩级别
            
            # 最小几何简化
            if hasattr(self.res_mgr, 'set_mesh_lod_bias'):
                self.res_mgr.set_mesh_lod_bias(0.0)
            self.max_draw_calls = 2000
            self.max_visible_lights = 16
            self.shadow_map_res = 2048
            
            # 架构特定优化
            if is_maxwell or is_gcn:
                print("应用架构特定优化")
                if is_maxwell:
                    # Maxwell架构优化
                    self.shader_optimizer.set_architecture_optimization("maxwell")
                elif is_gcn:
                    # GCN架构优化
                    self.shader_optimizer.set_architecture_optimization("gcn")
        
        # 重新创建渲染管道以应用新设置
        self._create_render_pipelines({})
        
    def set_render_quality(self, quality):
        """
        设置渲染质量
        
        Args:
            quality: 渲染质量枚举值
        """
        self.quality = quality
        
        # 根据质量设置调整参数
        if quality == RenderQuality.ULTRA_LOW:
            self.res = (1280, 720)
            self.max_draw_calls = 800
            self.max_visible_lights = 4
            self.shadow_map_res = 512
            self.instancing_enabled = True
        elif quality == RenderQuality.LOW:
            self.res = (1280, 720)
            self.max_draw_calls = 1000
            self.max_visible_lights = 8
            self.shadow_map_res = 1024
            self.instancing_enabled = True
        elif quality == RenderQuality.MEDIUM:
            self.res = (1920, 1080)
            self.max_draw_calls = 1200
            self.max_visible_lights = 8
            self.shadow_map_res = 1024
            self.instancing_enabled = True
        elif quality == RenderQuality.HIGH:
            self.res = (1920, 1080)
            self.max_draw_calls = 1500
            self.max_visible_lights = 12
            self.shadow_map_res = 2048
            self.instancing_enabled = True
        
        # 更新视口
        self.vp = (0, 0, self.res[0], self.res[1])
        
        # 更新效果管理器的质量设置
        if hasattr(self.effect_mgr, 'set_quality_level'):
            self.effect_mgr.set_quality_level(quality)
        
        # 通知所有渲染管道
        self.resize(self.res[0], self.res[1])
        
    def _auto_select_render_quality(self):
        """
        根据硬件自动选择最佳渲染质量
        """
        gpu_memory_budget = self.plt.get_gpu_memory_budget()
        gpu_name = self.hw_info['gpu_name'].lower()
        
        # 识别特定GPU型号
        is_gtx_750ti = "750ti" in gpu_name
        is_rx_580 = "rx" in gpu_name and "580" in gpu_name
        
        if is_gtx_750ti or gpu_memory_budget <= 2048:
            self.quality = RenderQuality.LOW
        elif is_rx_580 or (2048 < gpu_memory_budget <= 4096):
            self.quality = RenderQuality.MEDIUM
        else:
            self.quality = RenderQuality.HIGH
    
    def _create_render_pipelines(self, config=None):
        """
        创建渲染管道
        
        Args:
            config: 配置参数
        """
        # 清空现有管道
        for pipeline in self.pipelines.values():
            pipeline.shutdown()
        self.pipelines.clear()
        
        # 根据渲染模式创建管道
        if self.render_mode == RenderMode.FORWARD:
            self.pipelines["forward"] = ForwardRenderer(
                self.plt, 
                self.res_mgr,
                self.shader_mgr
            )
            self.pipelines["forward"].initialize(config)
            self.curr_pipeline = self.pipelines["forward"]
        elif self.render_mode == RenderMode.DEFERRED:
            # 仅在硬件支持时使用延迟渲染
            # 注意：这里假设plt对象有supports_deferred_rendering方法，如果没有，会在运行时报错
            # 但根据之前的代码，plt对象可能没有这个方法，所以我们需要添加一个默认实现
            supports_deferred = False
            if hasattr(self.plt, 'supports_deferred_rendering'):
                supports_deferred = self.plt.supports_deferred_rendering()
            
            if supports_deferred:
                self.pipelines["deferred"] = DeferredRenderer(
                    self.plt, 
                    self.res_mgr,
                    self.shader_mgr
                )
                self.pipelines["deferred"].initialize(config)
                self.curr_pipeline = self.pipelines["deferred"]
            else:
                print("硬件不支持延迟渲染，回退到前向渲染")
                self.render_mode = RenderMode.FORWARD
                self._create_render_pipelines(config)
                return
        elif self.render_mode == RenderMode.HYBRID:
            # 混合渲染模式（前向渲染处理透明物体，延迟渲染处理不透明物体）
            # 同样，检查plt对象是否有supports_deferred_rendering方法
            supports_deferred = False
            if hasattr(self.plt, 'supports_deferred_rendering'):
                supports_deferred = self.plt.supports_deferred_rendering()
            
            if supports_deferred:
                self.pipelines["hybrid"] = HybridRenderer(
                    self.plt, 
                    self.res_mgr,
                    self.shader_mgr
                )
                self.pipelines["hybrid"].initialize(config)
                self.curr_pipeline = self.pipelines["hybrid"]
            else:
                print("硬件不支持延迟渲染，混合模式回退到纯前向渲染")
                self.pipelines["forward"] = ForwardRenderer(
                    self.plt, 
                    self.res_mgr,
                    self.shader_mgr
                )
                self.pipelines["forward"].initialize(config)
                self.curr_pipeline = self.pipelines["forward"]
        elif self.render_mode == RenderMode.LOW_END:
            # 低端模式使用高度优化的前向渲染
            self.pipelines["forward"] = ForwardRenderer(
                self.plt, 
                self.res_mgr,
                self.shader_mgr
            )
            self.pipelines["forward"].initialize(config)
            self.curr_pipeline = self.pipelines["forward"]
        
        # 应用管道特定优化
        self._apply_pipeline_optimizations(self.curr_pipeline)
    
    def _apply_pipeline_optimizations(self, pipeline):
        """
        应用管道特定优化
        
        Args:
            pipeline: 渲染管道实例
        """
        if not pipeline:
            return
        
        # 设置通用优化参数
        pipeline.max_draw_calls = self.max_draw_calls
        pipeline.max_visible_lights = self.max_visible_lights
        pipeline.shadow_map_resolution = self.shadow_map_res
        pipeline.use_instancing = self.instancing_enabled
        
        # 根据渲染质量调整特性
        if self.quality == RenderQuality.ULTRA_LOW:
            pipeline.enable_feature("ssr", False)
            pipeline.enable_feature("volumetric_lighting", False)
            pipeline.enable_feature("reflection_probes", False)
            pipeline.enable_feature("bloom", False)
            pipeline.shadow_cascade_count = 1
        elif self.quality == RenderQuality.LOW:
            pipeline.enable_feature("ssr", False)
            pipeline.enable_feature("volumetric_lighting", False)
            pipeline.enable_feature("reflection_probes", False)
            pipeline.enable_feature("bloom", True)
            pipeline.shadow_cascade_count = 1
        elif self.quality == RenderQuality.MEDIUM:
            pipeline.enable_feature("ssr", True)
            pipeline.set_ssr_quality("low")
            pipeline.enable_feature("volumetric_lighting", False)
            pipeline.enable_feature("reflection_probes", True)
            pipeline.enable_feature("bloom", True)
            pipeline.shadow_cascade_count = 2
        elif self.quality == RenderQuality.HIGH:
            pipeline.enable_feature("ssr", True)
            pipeline.set_ssr_quality("medium")
            pipeline.enable_feature("volumetric_lighting", True)
            pipeline.set_volumetric_quality("low")
            pipeline.enable_feature("reflection_probes", True)
            pipeline.enable_feature("bloom", True)
            pipeline.shadow_cascade_count = 3
    
    def _apply_hardware_specific_optimizations(self):
        """
        应用硬件特定优化
        """
        gpu_name = self.hw_info['gpu_name'].lower()
        
        # NVIDIA Maxwell架构优化（如GTX 750Ti）
        if "750" in gpu_name or "maxwell" in gpu_name:
            print("应用NVIDIA Maxwell架构特定优化")
            # Maxwell架构上纹理采样优化
            self.shader_optimizer.gpu_architecture = "maxwell"
            self.shader_optimizer.current_rules = self.shader_optimizer.architecture_rules.get("maxwell", self.shader_optimizer.architecture_rules["default"])
            # 减少纹理采样指令数量
            # self.resource_manager.set_texture_sampling_optimization(True)  # 移除不支持的方法调用
            # 优化着色器内存访问模式
            # self.shader_manager.set_memory_coalescing_optimization(True)  # 移除不支持的方法调用
            
            # 针对效果系统的Maxwell特定优化
            # self.effect_manager.set_architecture_optimization("maxwell")  # 移除不支持的方法调用
            
            # Maxwell架构上限制更严格的效果设置
            # if self.hw_info['vram_mb'] <= 2048:
            #     self.effect_manager.enable_effect("fagi", False)  # 移除不支持的方法调用
            #     self.effect_manager.enable_effect("ssr", False)  # 移除不支持的方法调用
            #     self.effect_manager.enable_effect("ambient_occlusion", True)  # 移除不支持的方法调用
            #     self.effect_manager.set_ao_quality("ultra_low")  # 移除不支持的方法调用
            # else:
            #     self.effect_manager.enable_effect("fagi", True)  # 移除不支持的方法调用
            #     self.effect_manager.set_fagi_quality("ultra_low")  # 移除不支持的方法调用
                # self.effect_manager.enable_effect("ssr", False)  # 移除不支持的方法调用
                # self.effect_manager.enable_effect("ambient_occlusion", True)  # 移除不支持的方法调用
                # self.effect_manager.set_ao_quality("low")  # 移除不支持的方法调用
        
        # AMD GCN架构优化（如RX 580）
        elif "rx" in gpu_name and "580" in gpu_name or "gcn" in gpu_name:
            print("应用AMD GCN架构特定优化")
            # GCN架构上的向量化优化
            self.shader_optimizer.gpu_architecture = "gcn"
            self.shader_optimizer.current_rules = self.shader_optimizer.architecture_rules.get("gcn", self.shader_optimizer.architecture_rules["default"])
            # 优化计算着色器分组大小
            # self.shader_manager.set_compute_group_size((64, 1, 1))  # 移除不支持的方法调用
            
            # 针对效果系统的GCN特定优化
            # self.effect_manager.set_architecture_optimization("gcn")  # 移除不支持的方法调用
            
            # GCN架构上的效果设置
            # self.effect_manager.enable_effect("fagi", True)  # 移除不支持的方法调用
            # self.effect_manager.set_fagi_quality("medium")  # 移除不支持的方法调用
            # self.effect_manager.enable_effect("ssr", True)  # 移除不支持的方法调用
            # self.effect_manager.set_ssr_quality("low")  # 移除不支持的方法调用
            # self.effect_manager.enable_effect("ambient_occlusion", True)  # 移除不支持的方法调用
            # self.effect_manager.set_ao_quality("medium")  # 移除不支持的方法调用
            
        # 针对低端GPU的通用优化
        if self.hw_info['vram_mb'] <= 4096:
            # 启用激进的纹理压缩
            self.vram_mgr.enable_aggressive_compression(True)
            # 启用纹理流式加载
            self.vram_mgr.enable_texture_streaming(True)
            # 设置更短的纹理保留时间
            self.vram_mgr.set_texture_retention_time(10.0)  # 10秒
    
    def _update_scene_lods(self, scene, camera):
        """
        更新场景中所有对象的LOD级别
        
        Args:
            scene: 场景对象
            camera: 相机对象
        """
        # 获取相机位置
        camera_position = camera.get_position()
        
        # 遍历场景中的所有可渲染对象
        for node in scene.visible_nodes:
            if hasattr(node, 'update_lod'):
                # 计算对象到相机的距离
                distance = (node.get_position() - camera_position).length()
                # 更新LOD级别
                node.update_lod(distance, self.render_quality)
    
    def _prepare_batch_rendering(self, scene):
        """
        准备批处理渲染
        
        Args:
            scene: 场景对象
        """
        # 获取所有可渲染对象
        renderable_nodes = scene.visible_nodes
        
        # 按材质和网格分组以进行批处理
        batches = {}
        
        for node in renderable_nodes:
            if not node.is_visible():
                continue
                
            material_id = node.get_material_id()
            mesh_id = node.get_mesh_id()
            batch_key = (material_id, mesh_id)
            
            if batch_key not in batches:
                batches[batch_key] = []
            
            batches[batch_key].append(node)
        
        # 为每个批次创建实例数据
        for batch_key, nodes in batches.items():
            # 针对低端GPU优化批处理策略
            if self.render_quality == RenderQuality.ULTRA_LOW or self.render_quality == RenderQuality.LOW:
                # 对于低端GPU，优先使用实例化渲染，减少内存开销
                if self.is_instancing_enabled:
                    self.resource_manager.create_instance_batch(nodes)
            else:
                # 对于高端GPU，根据批次大小选择合适的批处理方式
                if len(nodes) < 10 and self.render_quality != RenderQuality.ULTRA_LOW:
                    self.resource_manager.create_static_batch(nodes)
                elif len(nodes) >= 10 and self.is_instancing_enabled:
                    self.resource_manager.create_instance_batch(nodes)
        
        # 根据渲染质量和GPU能力动态启用高级效果
        if self.current_pipeline:
            # 仅在渲染质量较高且GPU支持时启用SSR
            enable_ssr = self.render_quality in [RenderQuality.MEDIUM, RenderQuality.HIGH] and self.enable_ssr
            self.current_pipeline.enable_feature("ssr", enable_ssr)
            if enable_ssr:
                # 根据渲染质量设置SSR质量
                ssr_quality = "medium" if self.render_quality == RenderQuality.MEDIUM else "high"
                self.current_pipeline.set_ssr_quality(ssr_quality)
            
            # 仅在渲染质量较高且GPU支持时启用体积光照
            enable_volumetric = self.render_quality == RenderQuality.HIGH and self.enable_volumetric_lighting
            self.current_pipeline.enable_feature("volumetric_lighting", enable_volumetric)
            if enable_volumetric:
                self.current_pipeline.set_volumetric_quality("medium")
            
            # 仅在渲染质量较高且GPU支持时启用环境光遮蔽
            enable_ao = self.render_quality in [RenderQuality.MEDIUM, RenderQuality.HIGH] and self.enable_ao
            self.current_pipeline.enable_feature("ao", enable_ao)
            if enable_ao:
                ao_quality = "medium" if self.render_quality == RenderQuality.MEDIUM else "high"
                self.current_pipeline.set_ao_quality(ao_quality)