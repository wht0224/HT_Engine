import numpy as np
from .EffectBase import EffectBase, GPUArchitecture, EffectQuality

class AmbientOcclusion(EffectBase):
    """针对低端GPU优化的环境光遮蔽效果"""
    
    def __init__(self, gpu_architecture=GPUArchitecture.OTHER, quality_level=EffectQuality.LOW):
        super().__init__(gpu_architecture, quality_level)
        self.performance_cost = 1.2  # 基础性能开销估算
        self.radius = 1.0  # AO采样半径
        self.intensity = 0.8  # AO强度
        self.falloff = 0.5  # 衰减因子
        self.sample_count = 8  # 采样点数量
        self.downsample_factor = 2  # 降采样因子
        
        # 根据质量级别设置参数
        self._update_resolution_and_settings()
    
    def _setup_resources(self):
        """设置AO所需的资源"""
        # 深度和法线纹理
        self.depth_texture = None
        self.normal_texture = None
        
        # 中间渲染纹理
        self.ao_texture = None
        self.blur_texture = None
        
        # 着色器程序
        self.ao_shader = None
        self.blur_shader = None
        self.composite_shader = None
        
        # 采样核
        self.sample_kernel = self._generate_sample_kernel()
        self.noise_texture = self._generate_noise_texture()
    
    def _generate_sample_kernel(self):
        """生成半球采样核"""
        # 生成优化的采样核，减少计算量
        kernel = np.zeros((self.sample_count, 4), dtype=np.float32)
        
        for i in range(self.sample_count):
            # 生成半球内的随机向量
            x = np.random.uniform(-1.0, 1.0)
            y = np.random.uniform(-1.0, 1.0)
            z = np.random.uniform(0.0, 1.0)  # 只取上半球
            
            # 归一化
            vec = np.array([x, y, z])
            vec /= np.linalg.norm(vec)
            
            # 按距离加权，近处采样更密集
            scale = float(i) / float(self.sample_count)
            scale = 0.1 + (scale * scale) * 0.9
            vec *= scale
            
            kernel[i] = np.array([vec[0], vec[1], vec[2], scale])
        
        return kernel
    
    def _generate_noise_texture(self):
        """生成随机旋转噪声纹理"""
        # 生成低分辨率噪声纹理以减少采样成本
        size = 4  # 4x4噪声纹理，足够且高效
        noise = np.zeros((size, size, 3), dtype=np.float32)
        
        for i in range(size):
            for j in range(size):
                # 生成切线空间中的随机向量
                x = np.random.uniform(-1.0, 1.0)
                y = np.random.uniform(-1.0, 1.0)
                z = 0.0  # 2D旋转
                
                vec = np.array([x, y, z])
                vec /= np.linalg.norm(vec)
                
                noise[i, j] = vec
        
        return noise
    
    def _update_resolution_and_settings(self):
        """根据质量级别更新参数"""
        if self.quality_level == EffectQuality.LOW:  # GTX 750Ti级别
            self.sample_count = 4  # 减少采样点
            self.radius = 0.5
            self.intensity = 0.6
            self.downsample_factor = 4  # 更大的降采样
            self.performance_cost = 0.6  # 降低性能开销
        elif self.quality_level == EffectQuality.MEDIUM:  # RX 580级别
            self.sample_count = 8
            self.radius = 1.0
            self.intensity = 0.8
            self.downsample_factor = 2
            self.performance_cost = 1.2
        else:  # 高级别
            self.sample_count = 16
            self.radius = 1.5
            self.intensity = 1.0
            self.downsample_factor = 2
            self.performance_cost = 2.0
        
        # 重新生成采样核
        self.sample_kernel = self._generate_sample_kernel()
    
    def _optimize_for_gpu(self):
        """根据GPU架构优化AO"""
        super()._optimize_for_gpu()
        
        # NVIDIA Maxwell特定优化
        if self.gpu_architecture == GPUArchitecture.NVIDIA_MAXWELL:
            # Maxwell架构对AO的优化
            self._use_horizon_based_ao = True  # 更高效的计算方式
            self._use_voxel_ao = False  # 避免复杂的体素操作
            self._use_half_precision_calculations = True  # 使用半精度计算
        # AMD GCN特定优化
        elif self.gpu_architecture == GPUArchitecture.AMD_GCN:
            # GCN架构对AO的优化
            self._use_horizon_based_ao = False
            self._use_voxel_ao = False
            self._use_compute_shader_optimization = True  # 利用GCN的计算着色器优势
    
    def update(self, delta_time):
        """更新AO参数"""
        # 可以在这里添加动态调整参数的逻辑
        pass
    
    def _apply_effect(self, input_texture, output_texture):
        """应用AO效果"""
        # 模拟AO实现步骤
        # 1. 降采样深度和法线图
        # 2. 执行AO计算
        # 3. 应用高斯模糊减少噪点
        # 4. 上采样并与原始渲染结果混合
        
        # 根据GPU架构选择不同的实现路径
        if self.gpu_architecture == GPUArchitecture.NVIDIA_MAXWELL:
            return self._apply_ao_maxwell(input_texture, output_texture)
        elif self.gpu_architecture == GPUArchitecture.AMD_GCN:
            return self._apply_ao_gcn(input_texture, output_texture)
        else:
            return self._apply_ao_generic(input_texture, output_texture)
    
    def _apply_ao_maxwell(self, input_texture, output_texture):
        """Maxwell架构特定的AO实现"""
        # 在Maxwell架构上使用高度优化的AO实现
        # 1. 使用超低分辨率计算 (1/4 或 1/8 分辨率)
        # 2. 减少采样点数量
        # 3. 使用简化的模糊算法
        return self._generic_ao_implementation(input_texture, output_texture, 
                                             very_low_resolution=True, minimal_samples=True)
    
    def _apply_ao_gcn(self, input_texture, output_texture):
        """GCN架构特定的AO实现"""
        # 在GCN架构上使用优化的AO实现
        # 1. 中等分辨率计算 (1/2 分辨率)
        # 2. 标准采样点数量
        # 3. 更精确的模糊算法
        return self._generic_ao_implementation(input_texture, output_texture, 
                                             very_low_resolution=False, minimal_samples=False)
    
    def _apply_ao_generic(self, input_texture, output_texture):
        """通用的AO实现"""
        # 保守的通用实现
        return self._generic_ao_implementation(input_texture, output_texture, 
                                             very_low_resolution=True, minimal_samples=True)
    
    def _generic_ao_implementation(self, input_texture, output_texture, 
                                  very_low_resolution=True, minimal_samples=True):
        """通用AO实现核心逻辑"""
        # 改进的AO实现，针对低端GPU优化
        
        # 根据参数调整采样点数量和分辨率
        samples = 4 if minimal_samples else self.sample_count
        downsample = 8 if very_low_resolution else self.downsample_factor
        
        # 1. 降采样深度和法线，使用更高效的降采样算法
        # 对于低端GPU，使用更大的降采样因子以提高性能
        if self.gpu_architecture == GPUArchitecture.NVIDIA_MAXWELL or very_low_resolution:
            downsample = 8
        else:
            downsample = 4
        
        # 2. 计算AO值，使用改进的采样算法
        # 使用优化的采样核，减少采样点数量的同时保持效果
        ao_result = self._compute_optimized_ao(samples, downsample)
        
        # 3. 应用改进的模糊算法，减少噪点
        # 使用双边滤波，在减少噪点的同时保留边缘
        blurred_ao = self._apply_optimized_blur(ao_result)
        
        # 4. 上采样并混合，使用高质量的上采样算法
        final_ao = self._upsample_and_blend(blurred_ao, downsample, input_texture)
        
        # 返回处理后的纹理
        return final_ao
    
    def _compute_optimized_ao(self, samples, downsample):
        """优化的AO计算"""
        # 1. 使用改进的采样核，减少采样点数量的同时保持效果
        # 2. 对于低端GPU，使用更高效的采样模式
        # 3. 考虑法线信息，提高AO的准确性
        
        # 简化实现：返回输入纹理，实际应实现优化的AO计算
        return self.depth_texture
    
    def _apply_optimized_blur(self, ao_texture):
        """优化的AO模糊"""
        # 1. 使用双边滤波，在减少噪点的同时保留边缘
        # 2. 对于低端GPU，使用更少的迭代次数
        # 3. 考虑深度和法线信息，提高模糊质量
        
        # 简化实现：返回输入纹理，实际应实现优化的模糊
        return ao_texture
    
    def _upsample_and_blend(self, ao_texture, downsample, original_texture):
        """上采样并混合AO"""
        # 1. 使用高质量的上采样算法，如双线性插值或更复杂的算法
        # 2. 考虑深度和法线信息，提高上采样质量
        # 3. 与原始纹理混合，使用基于物理的混合方式
        
        # 计算AO强度，考虑材质属性
        # 注意：实际实现中需要从材质系统获取这些属性
        roughness = 0.5  # 临时值，实际应从材质获取
        
        # 根据粗糙度调整AO强度：粗糙表面受AO影响更大
        roughness_factor = roughness * 0.7 + 0.3
        final_intensity = self.intensity * roughness_factor
        
        # 简化实现：返回输入纹理，实际应实现高质量的上采样和混合
        return original_texture * (1.0 - final_intensity) + ao_texture * final_intensity
    
    def set_intensity(self, intensity):
        """设置AO强度"""
        self.intensity = max(0.0, min(2.0, intensity))
    
    def set_radius(self, radius):
        """设置采样半径"""
        self.radius = max(0.1, min(5.0, radius))
    
    def __str__(self):
        return f"AO (Samples: {self.sample_count}, Radius: {self.radius}, Intensity: {self.intensity}, Quality: {self.quality_level.name})"