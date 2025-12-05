import numpy as np
import time

class TextureCompressor:
    """
    纹理压缩工具
    针对低端GPU（如NVIDIA GTX 750Ti和AMD RX 580）优化的纹理压缩系统
    实现BC7、ETC2等压缩格式支持和质量优化
    """
    
    def __init__(self):
        # 支持的压缩格式
        self.supported_formats = {
            "BC7": self._compress_bc7,
            "BC3": self._compress_bc3,
            "ETC2": self._compress_etc2,
            "ETC2A": self._compress_etc2a,
            "ASTC": self._compress_astc
        }
        
        # 硬件兼容性信息
        self.hardware_compatibility = {
            "NVIDIA_GTX_750Ti": {"supported": ["BC7", "BC3"], "preferred": "BC7"},
            "AMD_RX_580": {"supported": ["BC7", "BC3", "ETC2", "ETC2A"], "preferred": "BC7"},
            "DEFAULT": {"supported": ["BC7", "BC3", "ETC2", "ETC2A"], "preferred": "BC7"}
        }
        
        # 质量级别设置
        self.quality_settings = {
            "high": {"compression_level": 9, "color_quantization": 256, "dithering": True},
            "medium": {"compression_level": 7, "color_quantization": 128, "dithering": False},
            "low": {"compression_level": 5, "color_quantization": 64, "dithering": False}
        }
        
        # 默认压缩格式
        self.default_format = "BC7"
        
        # 纹理降采样设置
        self.downscale_options = {
            "high": 1.0,    # 原始尺寸
            "medium": 0.5,  # 1/2尺寸
            "low": 0.25     # 1/4尺寸
        }
    
    def compress_texture(self, texture_data, format_name=None, quality="medium", hardware="DEFAULT"):
        """
        压缩纹理
        
        Args:
            texture_data: 纹理数据
            format_name: 压缩格式名称，如果为None则根据硬件自动选择
            quality: 质量级别（high/medium/low）
            hardware: 目标硬件
            
        Returns:
            dict: 压缩后的纹理数据和压缩信息
        """
        start_time = time.time()
        
        # 验证质量设置
        if quality not in self.quality_settings:
            quality = "medium"
            print(f"无效的质量设置，使用默认值: {quality}")
        
        # 获取质量设置
        quality_params = self.quality_settings[quality]
        
        # 确定压缩格式
        if format_name is None:
            # 根据硬件选择最佳格式
            if hardware in self.hardware_compatibility:
                format_name = self.hardware_compatibility[hardware]["preferred"]
            else:
                format_name = self.default_format
        
        # 验证压缩格式
        if format_name not in self.supported_formats:
            print(f"不支持的压缩格式: {format_name}，使用默认格式: {self.default_format}")
            format_name = self.default_format
        
        # 检查硬件兼容性
        is_compatible = self._check_hardware_compatibility(format_name, hardware)
        if not is_compatible:
            print(f"警告: {format_name} 可能与 {hardware} 不完全兼容")
        
        # 应用纹理降采样
        scale_factor = self.downscale_options[quality]
        if scale_factor < 1.0:
            texture_data = self._downscale_texture(texture_data, scale_factor)
        
        # 应用预压缩优化
        texture_data = self._apply_pre_compression_optimizations(texture_data, quality_params)
        
        # 执行压缩
        compression_func = self.supported_formats[format_name]
        compressed_data = compression_func(texture_data, quality_params)
        
        # 计算压缩结果
        original_size = self._calculate_texture_size(texture_data)
        compressed_size = self._calculate_compressed_size(compressed_data, format_name)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
        
        end_time = time.time()
        compression_time = end_time - start_time
        
        # 返回压缩结果和元数据
        result = {
            "data": compressed_data,
            "format": format_name,
            "quality": quality,
            "original_size": original_size,
            "compressed_size": compressed_size,
            "compression_ratio": compression_ratio,
            "compression_time": compression_time,
            "hardware": hardware
        }
        
        print(f"纹理压缩完成: {format_name}, 质量: {quality}, 压缩比: {compression_ratio:.2f}x, 耗时: {compression_time:.2f}秒")
        
        return result
    
    def get_optimal_format(self, hardware, texture_type="color"):
        """
        获取特定硬件和纹理类型的最佳压缩格式
        
        Args:
            hardware: 目标硬件
            texture_type: 纹理类型（color/normal/roughness/metallic等）
            
        Returns:
            str: 最佳压缩格式名称
        """
        # 根据硬件和纹理类型返回最佳格式
        if hardware in self.hardware_compatibility:
            preferred = self.hardware_compatibility[hardware]["preferred"]
            
            # 特殊处理不同纹理类型
            if texture_type == "normal":
                # 法线贴图可能更适合使用BC3格式
                if "BC3" in self.hardware_compatibility[hardware]["supported"]:
                    return "BC3"
            elif texture_type in ["roughness", "metallic", "ao"]:
                # 单通道纹理可以使用特定的压缩格式
                if "BC3" in self.hardware_compatibility[hardware]["supported"]:
                    return "BC3"
            
            return preferred
        else:
            # 默认格式
            return self.default_format
    
    def _compress_bc7(self, texture_data, quality_params):
        """
        使用BC7格式压缩纹理
        BC7提供了最佳的压缩质量，适合大多数纹理类型
        
        Args:
            texture_data: 纹理数据
            quality_params: 质量参数
            
        Returns:
            object: 压缩后的纹理数据
        """
        # 简化实现
        # 实际实现会使用如DirectXTex库或其他BC7压缩实现
        print("使用BC7格式压缩（高质量）")
        
        # 模拟压缩过程
        compressed_data = {
            "format": "BC7",
            "data": texture_data.get("data", None),
            "block_format": "BC7_TYPE_0",  # 最佳质量的块类型
            "quality_level": quality_params["compression_level"]
        }
        
        return compressed_data
    
    def _compress_bc3(self, texture_data, quality_params):
        """
        使用BC3格式压缩纹理
        BC3比BC7稍快但质量略低，适合法线贴图
        
        Args:
            texture_data: 纹理数据
            quality_params: 质量参数
            
        Returns:
            object: 压缩后的纹理数据
        """
        # 简化实现
        print("使用BC3格式压缩（平衡质量和速度）")
        
        compressed_data = {
            "format": "BC3",
            "data": texture_data.get("data", None),
            "quality_level": quality_params["compression_level"]
        }
        
        return compressed_data
    
    def _compress_etc2(self, texture_data, quality_params):
        """
        使用ETC2格式压缩纹理
        ETC2是一种更通用的压缩格式，在多平台上都有良好支持
        
        Args:
            texture_data: 纹理数据
            quality_params: 质量参数
            
        Returns:
            object: 压缩后的纹理数据
        """
        # 简化实现
        print("使用ETC2格式压缩（多平台兼容性）")
        
        compressed_data = {
            "format": "ETC2",
            "data": texture_data.get("data", None),
            "quality_level": quality_params["compression_level"]
        }
        
        return compressed_data
    
    def _compress_etc2a(self, texture_data, quality_params):
        """
        使用ETC2A格式压缩纹理（带Alpha通道）
        
        Args:
            texture_data: 纹理数据
            quality_params: 质量参数
            
        Returns:
            object: 压缩后的纹理数据
        """
        # 简化实现
        print("使用ETC2A格式压缩（带Alpha通道）")
        
        compressed_data = {
            "format": "ETC2A",
            "data": texture_data.get("data", None),
            "quality_level": quality_params["compression_level"]
        }
        
        return compressed_data
    
    def _compress_astc(self, texture_data, quality_params):
        """
        使用ASTC格式压缩纹理
        ASTC提供了更灵活的块大小选项
        
        Args:
            texture_data: 纹理数据
            quality_params: 质量参数
            
        Returns:
            object: 压缩后的纹理数据
        """
        # 简化实现
        print("使用ASTC格式压缩（灵活块大小）")
        
        # 根据质量选择块大小
        if quality_params["compression_level"] >= 8:
            block_size = "4x4"  # 高质量，较大的块大小
        elif quality_params["compression_level"] >= 5:
            block_size = "6x6"  # 中等质量
        else:
            block_size = "8x8"  # 低质量，较小的块大小
        
        compressed_data = {
            "format": "ASTC",
            "block_size": block_size,
            "data": texture_data.get("data", None),
            "quality_level": quality_params["compression_level"]
        }
        
        return compressed_data
    
    def _check_hardware_compatibility(self, format_name, hardware):
        """
        检查压缩格式与硬件的兼容性
        
        Args:
            format_name: 压缩格式名称
            hardware: 硬件名称
            
        Returns:
            bool: 是否兼容
        """
        if hardware in self.hardware_compatibility:
            return format_name in self.hardware_compatibility[hardware]["supported"]
        
        # 默认兼容
        return True
    
    def _downscale_texture(self, texture_data, scale_factor):
        """
        降采样纹理以减少大小
        
        Args:
            texture_data: 原始纹理数据
            scale_factor: 缩放因子（0.0-1.0）
            
        Returns:
            object: 降采样后的纹理数据
        """
        print(f"降采样纹理，缩放因子: {scale_factor}")
        
        # 简化实现
        # 实际实现会使用如双线性或双三次插值进行降采样
        return {
            "data": texture_data.get("data", None),
            "original_size": texture_data.get("size", None),
            "downsampled": True,
            "scale_factor": scale_factor
        }
    
    def _apply_pre_compression_optimizations(self, texture_data, quality_params):
        """
        应用预压缩优化
        
        Args:
            texture_data: 纹理数据
            quality_params: 质量参数
            
        Returns:
            object: 优化后的纹理数据
        """
        # 应用颜色量化
        if "color_quantization" in quality_params:
            texture_data = self._apply_color_quantization(texture_data, quality_params["color_quantization"])
        
        # 应用抖动
        if quality_params.get("dithering", False):
            texture_data = self._apply_dithering(texture_data)
        
        return texture_data
    
    def _apply_color_quantization(self, texture_data, levels):
        """
        应用颜色量化以提高压缩效果
        
        Args:
            texture_data: 纹理数据
            levels: 颜色级别数
            
        Returns:
            object: 处理后的纹理数据
        """
        # 简化实现
        # 实际实现会减少颜色深度以提高压缩率
        return texture_data
    
    def _apply_dithering(self, texture_data):
        """
        应用抖动以减少量化伪影
        
        Args:
            texture_data: 纹理数据
            
        Returns:
            object: 处理后的纹理数据
        """
        # 简化实现
        # 实际实现会应用如Floyd-Steinberg抖动
        return texture_data
    
    def _calculate_texture_size(self, texture_data):
        """
        计算纹理的原始大小（字节）
        
        Args:
            texture_data: 纹理数据
            
        Returns:
            int: 纹理大小（字节）
        """
        # 简化实现，返回估计值
        # 实际实现会根据纹理尺寸和格式计算真实大小
        size = texture_data.get("size", (1024, 1024))  # 默认1024x1024
        return size[0] * size[1] * 4  # 假设RGBA 8位格式
    
    def _calculate_compressed_size(self, compressed_data, format_name):
        """
        计算压缩后纹理的大小（字节）
        
        Args:
            compressed_data: 压缩后的纹理数据
            format_name: 压缩格式名称
            
        Returns:
            int: 压缩后纹理大小（字节）
        """
        # 简化实现，返回估计值
        # 实际实现会根据纹理尺寸和压缩格式计算真实大小
        size = compressed_data.get("original_size", (1024, 1024))  # 默认1024x1024
        
        # 计算块数量
        blocks_x = (size[0] + 3) // 4  # 向上取整到最近的4
        blocks_y = (size[1] + 3) // 4
        total_blocks = blocks_x * blocks_y
        
        # 根据格式计算每块字节数
        if format_name == "BC7":
            bytes_per_block = 16
        elif format_name == "BC3":
            bytes_per_block = 16
        elif format_name in ["ETC2", "ETC2A"]:
            bytes_per_block = 8
        elif format_name == "ASTC":
            block_size = compressed_data.get("block_size", "4x4")
            if block_size == "4x4":
                bytes_per_block = 16
            elif block_size == "6x6":
                bytes_per_block = 16  # 6x6的ASTC块也是16字节
            elif block_size == "8x8":
                bytes_per_block = 16  # 8x8的ASTC块也是16字节
            else:
                bytes_per_block = 16
        else:
            bytes_per_block = 16
        
        return total_blocks * bytes_per_block
    
    def analyze_texture_for_compression(self, texture_data):
        """
        分析纹理以确定最佳压缩策略
        
        Args:
            texture_data: 纹理数据
            
        Returns:
            dict: 压缩建议
        """
        # 简化实现
        # 实际实现会分析纹理的颜色分布、细节水平等，给出最佳压缩格式和质量设置的建议
        return {
            "suggested_format": "BC7",
            "suggested_quality": "medium",
            "estimated_compression_ratio": 4.0,
            "notes": "标准彩色纹理，BC7提供最佳质量/大小比"
        }