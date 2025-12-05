from Engine.Logger import get_logger

class GPUCapabilityEvaluator:
    """
    GPU能力评估器，用于评估GPU的渲染能力
    """
    
    def __init__(self, hardware_info):
        """
        初始化GPU能力评估器
        
        Args:
            hardware_info: 硬件信息字典
        """
        self.logger = get_logger("GPUCapabilityEvaluator")
        self.hardware_info = hardware_info
        self.capability_level = "medium"  # 默认中等能力
        self.performance_score = 0.0  # 性能评分
        
        # 初始化GPU性能数据库
        self._initialize_gpu_database()
        
        # 评估GPU能力
        self.evaluate()
    
    def _initialize_gpu_database(self):
        """
        初始化GPU性能数据库
        """
        # 常见GPU型号的性能评分
        self.gpu_database = {
            # NVIDIA GPU
            "gtx 750ti": 1.0,
            "gtx 1050": 1.5,
            "gtx 1060": 2.5,
            "rtx 2060": 3.5,
            "rtx 3060": 4.5,
            "rtx 4060": 5.5,
            "rtx 3080": 6.5,
            "rtx 4080": 7.5,
            "rtx 4090": 8.5,
            "h100": 9.5,
            
            # AMD GPU
            "rx 580": 2.0,
            "rx 6600": 3.0,
            "rx 6700 xt": 4.0,
            "rx 6800": 5.0,
            "rx 7600": 4.5,
            "rx 7700 xt": 5.5,
            "rx 7800 xt": 6.5,
            "rx 7900 xt": 7.5,
            "rx 7900 xtx": 8.0,
            
            # Intel GPU
            "hd graphics 630": 0.5,
            "iris xe": 1.0,
            "arc a380": 1.5,
            "arc a750": 2.5,
            "arc a770": 3.0
        }
    
    def evaluate(self):
        """
        评估GPU能力
        """
        self.logger.info(f"开始评估GPU能力: {self.hardware_info['gpu_name']}")
        
        # 基于硬件参数计算性能评分
        self._calculate_performance_score()
        
        # 确定能力等级
        self._determine_capability_level()
        
        self.logger.info(f"GPU能力评估完成: 评分={self.performance_score:.2f}, 等级={self.capability_level}")
    
    def _calculate_performance_score(self):
        """
        计算GPU性能评分
        """
        gpu_name = self.hardware_info['gpu_name'].lower()
        vram = self.hardware_info['vram_mb']
        architecture = self.hardware_info['gpu_architecture']
        vendor = self.hardware_info['gpu_vendor']
        
        # 1. 基于GPU型号的基础评分
        base_score = 1.0  # 默认评分
        
        # 查找GPU数据库中的评分
        for model, score in self.gpu_database.items():
            if model in gpu_name:
                base_score = score
                break
        
        # 2. VRAM大小调整
        vram_factor = min(vram / 2048, 3.0)  # 基于2GB VRAM的调整，最大3倍
        
        # 3. 架构调整
        architecture_factors = {
            # NVIDIA架构
            "fermi": 0.5,
            "kepler": 0.7,
            "maxwell": 0.9,
            "pascal": 1.0,
            "turing": 1.2,
            "ampere": 1.4,
            "ada": 1.6,
            "hopper": 1.8,
            
            # AMD架构
            "gcn": 0.9,
            "gcn5": 1.0,
            "rdna1": 1.2,
            "rdna2": 1.4,
            
            # Intel架构
            "intel_hd": 0.5,
            "intel_hd_500": 0.6,
            "intel_hd_600": 0.7,
            "intel_iris": 0.8,
            "intel_iris_xe": 1.0,
            "intel_arc": 1.2
        }
        
        arch_factor = architecture_factors.get(architecture, 1.0)
        
        # 4. 渲染特性调整
        feature_factor = 1.0
        render_features = self.hardware_info['render_features']
        
        # 曲面细分支持
        if render_features.get('tessellation', False):
            feature_factor += 0.1
        
        # 几何着色器支持
        if render_features.get('geometry_shaders', False):
            feature_factor += 0.1
        
        # 计算着色器支持
        if render_features.get('compute_shaders', False):
            feature_factor += 0.2
        
        # 光线追踪支持
        if render_features.get('ray_tracing', False):
            feature_factor += 0.3
        
        # 网格着色器支持
        if render_features.get('mesh_shaders', False):
            feature_factor += 0.2
        
        # 5. 最终评分计算
        self.performance_score = base_score * vram_factor * arch_factor * feature_factor
        
        # 确保评分在合理范围内
        self.performance_score = max(0.1, min(self.performance_score, 10.0))
    
    def _determine_capability_level(self):
        """
        确定GPU能力等级
        """
        if self.performance_score < 1.5:
            self.capability_level = "low"
        elif self.performance_score < 3.0:
            self.capability_level = "medium"
        elif self.performance_score < 5.0:
            self.capability_level = "high"
        else:
            self.capability_level = "ultra"
    
    def get_capability_level(self):
        """
        获取GPU能力等级
        
        Returns:
            str: GPU能力等级 (low/medium/high/ultra)
        """
        return self.capability_level
    
    def get_performance_score(self):
        """
        获取GPU性能评分
        
        Returns:
            float: GPU性能评分
        """
        return self.performance_score
    
    def supports_feature(self, feature_name):
        """
        检查GPU是否支持特定特性
        
        Args:
            feature_name: 特性名称
            
        Returns:
            bool: 是否支持该特性
        """
        render_features = self.hardware_info['render_features']
        return render_features.get(feature_name, False)
    
    def get_recommended_render_mode(self):
        """
        获取推荐的渲染模式
        
        Returns:
            str: 推荐的渲染模式
        """
        if self.capability_level == "low":
            return "low_end"
        elif self.capability_level == "medium":
            return "forward"
        elif self.capability_level == "high":
            return "deferred"
        else:
            return "hybrid"
    
    def get_recommended_quality_level(self):
        """
        获取推荐的渲染质量等级
        
        Returns:
            str: 推荐的渲染质量等级
        """
        if self.capability_level == "low":
            return "ultra_low"
        elif self.capability_level == "medium":
            return "medium"
        elif self.capability_level == "high":
            return "high"
        else:
            return "high"
    
    def get_recommended_settings(self):
        """
        获取推荐的渲染设置
        
        Returns:
            dict: 推荐的渲染设置
        """
        settings = {
            "render_mode": self.get_recommended_render_mode(),
            "quality_level": self.get_recommended_quality_level(),
            "max_draw_calls": 1000,
            "max_visible_lights": 8,
            "shadow_map_resolution": 1024,
            "texture_quality": "medium",
            "lod_bias": 0.0,
            "enable_ssr": False,
            "enable_volumetric_lighting": False,
            "enable_ao": False
        }
        
        # 根据能力等级调整设置
        if self.capability_level == "low":
            settings.update({
                "max_draw_calls": 600,
                "max_visible_lights": 4,
                "shadow_map_resolution": 512,
                "texture_quality": "low",
                "lod_bias": 1.0
            })
        elif self.capability_level == "medium":
            settings.update({
                "max_draw_calls": 1200,
                "max_visible_lights": 8,
                "shadow_map_resolution": 1024,
                "texture_quality": "medium",
                "enable_ssr": False,
                "enable_ao": True
            })
        elif self.capability_level == "high":
            settings.update({
                "max_draw_calls": 2000,
                "max_visible_lights": 16,
                "shadow_map_resolution": 2048,
                "texture_quality": "high",
                "enable_ssr": True,
                "enable_volumetric_lighting": True,
                "enable_ao": True
            })
        else:  # ultra
            settings.update({
                "max_draw_calls": 3000,
                "max_visible_lights": 32,
                "shadow_map_resolution": 4096,
                "texture_quality": "ultra",
                "enable_ssr": True,
                "enable_volumetric_lighting": True,
                "enable_ao": True
            })
        
        return settings
