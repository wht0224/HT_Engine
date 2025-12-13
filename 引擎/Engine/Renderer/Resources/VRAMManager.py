import time
import threading
import numpy as np
from enum import Enum

class ResourcePriority(Enum):
    """资源优先级枚举"""
    LOW = 1
    MEDIUM = 5
    HIGH = 10
    CRITICAL = 20

class VRAMManager:
    """
    VRAM显存管理器
    针对低端GPU（如NVIDIA GTX 750Ti和AMD RX 580）的显存优化系统
    实现动态显存分配、资源优先级管理和自动回收机制
    """
    
    def __init__(self, platform=None):
        # 默认VRAM限制设置（以MB为单位）
        self.vram_limits = {
            "gtx_750ti": 2048,  # 2GB
            "rx_580": 4096,     # 4GB
            "high_end": 8192    # 高端GPU
        }
        
        # 当前使用的VRAM限制
        self.current_vram_limit = self.vram_limits["gtx_750ti"]  # 默认使用GTX 750Ti的限制
        
        # 安全阈值百分比（保留一部分VRAM作为缓冲区）
        self.safety_margin_percentage = 10  # 保留10%的VRAM
        
        # 当前VRAM使用量（MB）
        self.current_vram_usage = 0
        
        # 资源注册表 - 存储所有已分配的资源
        self.resource_registry = {}
        
        # 资源优先级队列 - 用于确定哪些资源应该优先保留/释放
        self.resource_priority_queue = []
        
        # 资源类型大小估算（字节）
        self.resource_size_estimates = {
            "texture_2d": lambda width, height, format: width * height * self._get_format_bytes_per_pixel(format),
            "texture_cubemap": lambda size, format: size * size * 6 * self._get_format_bytes_per_pixel(format),
            "vertex_buffer": lambda count, stride: count * stride,
            "index_buffer": lambda count, index_size: count * index_size,
            "constant_buffer": lambda size: size,
            "shader_resource_view": lambda: 64,  # 近似值
            "render_target": lambda width, height, format: width * height * self._get_format_bytes_per_pixel(format)
        }
        
        # 资源压缩状态
        self.compressible_resources = set()
        
        # 纹理流设置
        self.texture_streaming_settings = {
            "enabled": True,
            "distance_threshold": 100.0,  # 纹理卸载距离阈值
            "resolution_rates": [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]  # 更细粒度的分辨率级别
        }
        
        # 缓存清理设置
        self.cache_cleanup_settings = {
            "enabled": True,
            "min_usage_duration": 30.0,  # 最小使用持续时间（秒）
            "check_interval": 2.0,       # 更频繁的检查，每2秒一次
            "max_idle_time": 60.0        # 最大空闲时间（秒）
        }
        
        # 资源压缩设置
        self.compression_settings = {
            "enabled": True,
            "compression_threshold": 75,  # 当VRAM使用达到75%时开始压缩，更早开始压缩
            "compression_ratio": 0.5,     # 更激进的压缩比例
            "aggressive_compression": False  # 是否使用更激进的压缩
        }
        
        # 锁，确保线程安全
        self.lock = threading.RLock()
        
        # 初始化缓存清理线程
        self._initialize_cache_cleanup()
    
    def set_gpu_profile(self, gpu_profile):
        """
        设置GPU配置文件，调整VRAM限制
        
        Args:
            gpu_profile: GPU配置文件名称（"gtx_750ti"、"rx_580"、"high_end"）或自定义VRAM大小（MB）
        """
        with self.lock:
            if isinstance(gpu_profile, str) and gpu_profile in self.vram_limits:
                self.current_vram_limit = self.vram_limits[gpu_profile]
                print(f"设置GPU配置文件为: {gpu_profile}，VRAM限制: {self.current_vram_limit} MB")
            elif isinstance(gpu_profile, int) and gpu_profile > 0:
                self.current_vram_limit = gpu_profile
                print(f"设置自定义VRAM限制: {self.current_vram_limit} MB")
            else:
                print(f"无效的GPU配置文件: {gpu_profile}，使用默认配置")
    
    def register_resource(self, resource_id, resource_type, estimated_size_mb, priority=5, compressible=False):
        """
        注册一个新的显存资源
        
        Args:
            resource_id: 资源唯一标识符
            resource_type: 资源类型（"texture"、"buffer"等）
            estimated_size_mb: 估计的内存大小（MB）
            priority: 资源优先级（1-10，10最高）
            compressible: 是否可压缩
            
        Returns:
            bool: 是否成功注册（如果显存不足可能返回False）
        """
        # 临时变量，用于调试和日志记录
        debug_level = 0  # 0: 关闭, 1: 基本, 2: 详细
        
        with self.lock:
            # 检查是否已存在
            # 这里之前遇到过资源ID重复的问题，所以加了这个检查
            if resource_id in self.resource_registry:
                print(f"警告: 资源ID {resource_id} 已存在，跳过注册")
                # 之前这里直接返回False，后来改为返回True，因为重复注册可能是正常的
                # 比如资源被多次注册，但是实际上只需要一个
                return True  # 重复注册，返回True表示已经存在，不需要再次注册
            
            # 检查是否有足够的显存
            # 这里原本有个旧的检查方式，后来改了，但是注释留着
            # old_check = self.current_vram_usage + estimated_size_mb <= self.current_vram_limit
            if not self._check_memory_available(estimated_size_mb):
                # 尝试释放一些资源
                print(f"尝试释放 {estimated_size_mb:.2f} MB 的显存")
                if not self._free_memory(estimated_size_mb):
                    print(f"显存不足，无法注册资源: {resource_id}")
                    return False
            
            # 注册资源
            timestamp = time.time()
            
            # 创建资源信息字典
            # 这里原本有个更复杂的结构，后来简化了，但是注释留着
            resource_info = {
                "resource_id": resource_id,
                "resource_type": resource_type,
                "original_size_mb": estimated_size_mb,
                "current_size_mb": estimated_size_mb,
                "priority": priority,
                "compressible": compressible,
                "created_at": timestamp,
                "last_used": timestamp,
                "compressed": False,
                "compression_ratio": 1.0,
                "streaming_info": {
                    "enabled": resource_type.startswith("texture"),
                    "current_lod": 0,
                    "original_resolution": None,
                    "current_resolution": None,
                    # 这里原本有个额外的字段，后来移除了，但是注释留着
                    # "last_streamed": timestamp  # 上次流式加载时间，暂时不用
                }
            }
            
            # 添加到注册表
            self.resource_registry[resource_id] = resource_info
            
            # 更新优先级队列
            # 这里之前没有更新，后来发现需要，所以加上了
            self._update_priority_queue()
            
            # 更新VRAM使用量
            self.current_vram_usage += estimated_size_mb
            
            # 如果是可压缩资源，添加到可压缩集合
            if compressible:
                # 这里之前遇到过重复添加的问题，所以加了检查
                if resource_id not in self.compressible_resources:
                    self.compressible_resources.add(resource_id)
            
            # 打印日志
            if debug_level >= 1:
                print(f"注册资源: {resource_id}, 大小: {estimated_size_mb:.2f} MB, 优先级: {priority}")
            
            return True
    
    def unregister_resource(self, resource_id):
        """
        注销资源并释放显存
        
        Args:
            resource_id: 资源唯一标识符
            
        Returns:
            bool: 是否成功注销
        """
        with self.lock:
            if resource_id not in self.resource_registry:
                print(f"警告: 找不到资源ID {resource_id}")
                return False
            
            # 获取资源信息
            resource_info = self.resource_registry[resource_id]
            
            # 更新VRAM使用量
            self.current_vram_usage -= resource_info["current_size_mb"]
            
            # 从可压缩集合中移除
            if resource_id in self.compressible_resources:
                self.compressible_resources.remove(resource_id)
            
            # 从注册表中移除
            del self.resource_registry[resource_id]
            
            # 更新优先级队列
            self._update_priority_queue()
            
            print(f"注销资源: {resource_id}, 释放: {resource_info['current_size_mb']:.2f} MB")
            
            return True
    
    def mark_resource_used(self, resource_id):
        """
        标记资源为最近使用
        
        Args:
            resource_id: 资源唯一标识符
        """
        with self.lock:
            if resource_id in self.resource_registry:
                self.resource_registry[resource_id]["last_used"] = time.time()
                self._update_priority_queue()
    
    def estimate_resource_size(self, resource_type, *args, **kwargs):
        """
        估算资源大小
        
        Args:
            resource_type: 资源类型
            *args, **kwargs: 资源特定参数
            
        Returns:
            float: 估计的资源大小（MB）
        """
        if resource_type in self.resource_size_estimates:
            # 调用对应的估算函数
            size_bytes = self.resource_size_estimates[resource_type](*args, **kwargs)
            # 转换为MB
            return size_bytes / (1024 * 1024)
        else:
            print(f"警告: 未知的资源类型: {resource_type}")
            return 0.0
    
    def compress_resources(self, target_percentage=80):
        """
        压缩可压缩的资源以释放显存
        
        Args:
            target_percentage: 目标VRAM使用百分比
            
        Returns:
            float: 释放的显存大小（MB）
        """
        if not self.compression_settings["enabled"]:
            return 0.0
        
        with self.lock:
            # 计算当前VRAM使用百分比
            current_percentage = (self.current_vram_usage / self.current_vram_limit) * 100
            
            # 如果当前使用量低于目标，不需要压缩
            if current_percentage < target_percentage:
                return 0.0
            
            # 计算需要释放的显存大小
            target_usage = (target_percentage / 100) * self.current_vram_limit
            required_free = self.current_vram_usage - target_usage
            
            # 按照优先级排序可压缩资源（低优先级先压缩）
            compressible_resources = []
            for resource_id in self.compressible_resources:
                if resource_id in self.resource_registry and not self.resource_registry[resource_id]["compressed"]:
                    resource_info = self.resource_registry[resource_id]
                    compressible_resources.append((resource_info["priority"], resource_id))
            
            # 按优先级排序（优先级低的排在前面）
            compressible_resources.sort(key=lambda x: (x[0], self.resource_registry[x[1]]["last_used"]))
            
            freed_memory = 0.0
            
            # 压缩资源直到达到目标
            for _, resource_id in compressible_resources:
                if freed_memory >= required_free:
                    break
                
                resource_info = self.resource_registry[resource_id]
                
                # 计算压缩后的大小
                compression_ratio = self.compression_settings["compression_ratio"]
                if self.compression_settings["aggressive_compression"]:
                    compression_ratio *= 0.7  # 更激进的压缩
                
                # 更新资源大小
                compressed_size = resource_info["original_size_mb"] * compression_ratio
                freed = resource_info["current_size_mb"] - compressed_size
                
                # 更新资源信息
                resource_info["current_size_mb"] = compressed_size
                resource_info["compressed"] = True
                resource_info["compression_ratio"] = compression_ratio
                
                # 更新VRAM使用量
                self.current_vram_usage -= freed
                freed_memory += freed
                
                print(f"压缩资源: {resource_id}, 释放: {freed:.2f} MB, 压缩后大小: {compressed_size:.2f} MB")
            
            return freed_memory
    
    def decompress_resource(self, resource_id):
        """
        解压缩资源
        
        Args:
            resource_id: 资源唯一标识符
            
        Returns:
            bool: 是否成功解压缩
        """
        with self.lock:
            if resource_id not in self.resource_registry:
                return False
            
            resource_info = self.resource_registry[resource_id]
            
            if not resource_info["compressed"]:
                return True  # 已经是解压缩状态
            
            # 检查是否有足够的显存
            decompressed_size = resource_info["original_size_mb"]
            current_size = resource_info["current_size_mb"]
            additional_memory_needed = decompressed_size - current_size
            
            if not self._check_memory_available(additional_memory_needed):
                # 尝试释放一些其他资源
                if not self._free_memory(additional_memory_needed):
                    print(f"显存不足，无法解压缩资源: {resource_id}")
                    return False
            
            # 解压缩资源
            resource_info["current_size_mb"] = decompressed_size
            resource_info["compressed"] = False
            
            # 更新VRAM使用量
            self.current_vram_usage += additional_memory_needed
            
            print(f"解压缩资源: {resource_id}, 增加: {additional_memory_needed:.2f} MB")
            
            return True
    
    def adjust_texture_resolution(self, resource_id, resolution_factor):
        """
        调整纹理资源的分辨率
        
        Args:
            resource_id: 资源唯一标识符
            resolution_factor: 分辨率缩放因子（0.1-1.0）
            
        Returns:
            bool: 是否成功调整
        """
        with self.lock:
            if resource_id not in self.resource_registry:
                return False
            
            resource_info = self.resource_registry[resource_id]
            
            # 检查是否是纹理资源
            if not resource_info["resource_type"].startswith("texture"):
                print(f"警告: 资源 {resource_id} 不是纹理资源，无法调整分辨率")
                return False
            
            # 限制分辨率因子范围
            resolution_factor = max(0.1, min(1.0, resolution_factor))
            
            # 计算新的大小（面积按平方比例变化）
            original_size = resource_info["original_size_mb"]
            new_size = original_size * resolution_factor * resolution_factor
            current_size = resource_info["current_size_mb"]
            
            # 计算大小变化
            size_difference = new_size - current_size
            
            # 如果是增加分辨率，需要检查显存
            if size_difference > 0:
                if not self._check_memory_available(size_difference):
                    print(f"显存不足，无法增加纹理分辨率: {resource_id}")
                    return False
            
            # 更新资源信息
            resource_info["current_size_mb"] = new_size
            resource_info["streaming_info"]["current_lod"] = resolution_factor
            
            # 如果原始分辨率未知，假设当前就是原始分辨率
            if resource_info["streaming_info"]["original_resolution"] is None:
                resource_info["streaming_info"]["original_resolution"] = 1.0
            
            resource_info["streaming_info"]["current_resolution"] = resolution_factor
            
            # 更新VRAM使用量
            self.current_vram_usage += size_difference
            
            print(f"调整纹理分辨率: {resource_id}, 缩放因子: {resolution_factor}, 大小变化: {size_difference:+.2f} MB")
            
            return True
    
    def get_memory_stats(self):
        """
        获取显存使用统计信息
        
        Returns:
            dict: 显存统计信息
        """
        with self.lock:
            # 计算安全阈值
            safety_threshold = self.current_vram_limit * (self.safety_margin_percentage / 100.0)
            
            # 计算资源类型分布
            resource_type_distribution = {}
            for resource_info in self.resource_registry.values():
                resource_type = resource_info["resource_type"]
                if resource_type not in resource_type_distribution:
                    resource_type_distribution[resource_type] = 0
                resource_type_distribution[resource_type] += resource_info["current_size_mb"]
            
            # 计算压缩资源数量
            compressed_count = 0
            compressible_count = 0
            for resource_info in self.resource_registry.values():
                if resource_info["compressible"]:
                    compressible_count += 1
                    if resource_info["compressed"]:
                        compressed_count += 1
            
            return {
                "total_vram": self.current_vram_limit,
                "used_vram": self.current_vram_usage,
                "free_vram": self.current_vram_limit - self.current_vram_usage,
                "safety_threshold": safety_threshold,
                "usage_percentage": (self.current_vram_usage / self.current_vram_limit) * 100,
                "resource_count": len(self.resource_registry),
                "compressed_resources": compressed_count,
                "compressible_resources": compressible_count,
                "resource_type_distribution": resource_type_distribution
            }
    
    def print_memory_report(self):
        """
        打印显存使用报告
        """
        stats = self.get_memory_stats()
        
        print("\n=== VRAM使用报告 ===")
        print(f"总VRAM: {stats['total_vram']:.2f} MB")
        print(f"已使用: {stats['used_vram']:.2f} MB ({stats['usage_percentage']:.1f}%)")
        print(f"可用: {stats['free_vram']:.2f} MB")
        print(f"安全阈值: {stats['safety_threshold']:.2f} MB ({self.safety_margin_percentage}%)")
        print(f"资源总数: {stats['resource_count']}")
        print(f"压缩资源: {stats['compressed_resources']}/{stats['compressible_resources']}")
        
        print("\n资源类型分布:")
        for resource_type, size in stats['resource_type_distribution'].items():
            percentage = (size / stats['used_vram']) * 100 if stats['used_vram'] > 0 else 0
            print(f"  {resource_type}: {size:.2f} MB ({percentage:.1f}%)")
        print("==================\n")
    
    def _check_memory_available(self, required_mb):
        """
        检查是否有足够的可用显存
        
        Args:
            required_mb: 需要的显存大小（MB）
            
        Returns:
            bool: 是否有足够的显存
        """
        safety_threshold = self.current_vram_limit * (self.safety_margin_percentage / 100.0)
        available = self.current_vram_limit - self.current_vram_usage - safety_threshold
        return available >= required_mb
    
    def _free_memory(self, required_mb):
        """
        尝试释放显存以满足需求
        
        Args:
            required_mb: 需要释放的显存大小（MB）
            
        Returns:
            bool: 是否成功释放足够的显存
        """
        freed_memory = 0.0
        
        # 1. 首先尝试降低纹理分辨率（如果启用了纹理流）
        if self.texture_streaming_settings["enabled"]:
            freed_by_streaming = self._reduce_texture_resolutions()
            freed_memory += freed_by_streaming
            
            if freed_memory >= required_mb:
                return True
        
        # 2. 然后尝试压缩可压缩资源
        freed_by_compression = self.compress_resources()
        freed_memory += freed_by_compression
        
        if freed_memory >= required_mb:
            return True
        
        # 3. 尝试进一步降低已压缩资源的压缩比
        freed_by_aggressive_compression = self._aggressive_compress_resources()
        freed_memory += freed_by_aggressive_compression
        
        if freed_memory >= required_mb:
            return True
        
        # 4. 最后尝试卸载低优先级、长时间未使用的资源
        # 按优先级和最后使用时间排序（低优先级、长时间未使用的排在前面）
        timestamp = time.time()
        resources_to_consider = []
        
        for resource_id, resource_info in self.resource_registry.items():
            # 跳过不可卸载的资源
            if resource_info["resource_type"] in ["constant_buffer", "shader_resource_view"]:
                continue
            
            # 计算资源分数（低优先级、长时间未使用的资源分数高）
            priority_factor = 11 - resource_info["priority"]  # 优先级1-10变成10-1
            idle_time = timestamp - resource_info["last_used"]
            idle_factor = idle_time / 60.0  # 转换为分钟
            score = priority_factor * (1 + idle_factor)
            
            # 已压缩的资源分数更高
            if resource_info["compressed"]:
                score *= 1.5
            
            resources_to_consider.append((score, resource_id))
        
        # 按分数降序排序
        resources_to_consider.sort(key=lambda x: x[0], reverse=True)
        
        # 尝试卸载资源
        for _, resource_id in resources_to_consider:
            if freed_memory >= required_mb:
                break
            
            resource_info = self.resource_registry[resource_id]
            
            # 实际应用中，这里会调用卸载资源的逻辑
            # 这里只是模拟卸载
            print(f"模拟卸载资源: {resource_id}, 大小: {resource_info['current_size_mb']:.2f} MB")
            
            # 更新VRAM使用量
            freed_memory += resource_info["current_size_mb"]
            self.current_vram_usage -= resource_info["current_size_mb"]
            
            # 从注册表中移除
            if resource_id in self.compressible_resources:
                self.compressible_resources.remove(resource_id)
            
            del self.resource_registry[resource_id]
        
        # 更新优先级队列
        self._update_priority_queue()
        
        return freed_memory >= required_mb
    
    def _reduce_texture_resolutions(self):
        """
        降低纹理分辨率以释放显存
        
        Returns:
            float: 释放的显存大小（MB）
        """
        freed_memory = 0.0
        
        # 按优先级和最后使用时间排序（低优先级、长时间未使用的排在前面）
        timestamp = time.time()
        texture_resources = []
        
        for resource_id, resource_info in self.resource_registry.items():
            if resource_info["resource_type"].startswith("texture") and resource_info["streaming_info"]["enabled"]:
                # 计算资源分数（低优先级、长时间未使用的资源分数高）
                priority_factor = 11 - resource_info["priority"]  # 优先级1-10变成10-1
                idle_time = timestamp - resource_info["last_used"]
                idle_factor = idle_time / 60.0  # 转换为分钟
                current_resolution = resource_info["streaming_info"]["current_resolution"] or 1.0
                resolution_factor = 1.0 / current_resolution  # 低分辨率的资源分数更高
                score = priority_factor * (1 + idle_factor) * resolution_factor
                
                texture_resources.append((score, resource_id))
        
        # 按分数降序排序
        texture_resources.sort(key=lambda x: x[0], reverse=True)
        
        # 尝试降低纹理分辨率
        for _, resource_id in texture_resources:
            resource_info = self.resource_registry[resource_id]
            current_resolution = resource_info["streaming_info"]["current_resolution"] or 1.0
            
            # 找到下一个更低的分辨率
            available_resolutions = sorted(self.texture_streaming_settings["resolution_rates"], reverse=True)
            current_index = available_resolutions.index(current_resolution) if current_resolution in available_resolutions else 0
            
            if current_index < len(available_resolutions) - 1:
                new_resolution = available_resolutions[current_index + 1]
                
                # 计算大小变化
                original_size = resource_info["original_size_mb"]
                old_size = original_size * current_resolution * current_resolution
                new_size = original_size * new_resolution * new_resolution
                freed = old_size - new_size
                
                # 更新资源信息
                resource_info["current_size_mb"] = new_size
                resource_info["streaming_info"]["current_lod"] = new_resolution
                resource_info["streaming_info"]["current_resolution"] = new_resolution
                
                # 更新VRAM使用量
                self.current_vram_usage -= freed
                freed_memory += freed
                
                print(f"降低纹理分辨率: {resource_id}, 从 {current_resolution} 到 {new_resolution}, 释放: {freed:.2f} MB")
        
        return freed_memory
    
    def _aggressive_compress_resources(self):
        """
        对已压缩资源使用更激进的压缩比
        
        Returns:
            float: 释放的显存大小（MB）
        """
        freed_memory = 0.0
        
        # 找到所有已压缩但可以进一步压缩的资源
        for resource_id, resource_info in self.resource_registry.items():
            if resource_info["compressed"] and resource_info["compression_ratio"] > 0.3:  # 只有压缩比大于0.3的资源可以进一步压缩
                # 计算新的压缩比（更激进）
                new_compression_ratio = resource_info["compression_ratio"] * 0.7
                new_compression_ratio = max(0.2, new_compression_ratio)  # 最小压缩比为0.2
                
                # 计算大小变化
                original_size = resource_info["original_size_mb"]
                old_size = original_size * resource_info["compression_ratio"]
                new_size = original_size * new_compression_ratio
                freed = old_size - new_size
                
                # 更新资源信息
                resource_info["current_size_mb"] = new_size
                resource_info["compression_ratio"] = new_compression_ratio
                
                # 更新VRAM使用量
                self.current_vram_usage -= freed
                freed_memory += freed
                
                print(f"激进压缩资源: {resource_id}, 压缩比从 {resource_info['compression_ratio']:.2f} 到 {new_compression_ratio:.2f}, 释放: {freed:.2f} MB")
        
        return freed_memory
    
    def _update_priority_queue(self):
        """
        更新资源优先级队列
        """
        # 实现优先级队列更新
        # 按优先级和最后使用时间排序
        timestamp = time.time()
        self.resource_priority_queue = []
        
        for resource_id, resource_info in self.resource_registry.items():
            # 计算优先级分数
            priority_score = resource_info["priority"] * 1000
            # 增加时间因子（最近使用的优先级更高）
            time_factor = 1000 - min(999, (timestamp - resource_info["last_used"]))
            total_score = priority_score + time_factor
            
            self.resource_priority_queue.append((-total_score, resource_id))  # 负号用于升序排序
        
        # 按分数升序排序（分数越高的资源排在前面）
        self.resource_priority_queue.sort()
    
    def _get_format_bytes_per_pixel(self, format):
        """
        获取纹理格式的每像素字节数
        
        Args:
            format: 纹理格式
            
        Returns:
            int: 每像素字节数
        """
        # 常见纹理格式的每像素字节数
        format_bytes = {
            "rgba8": 4,
            "rgb8": 3,
            "rgba16f": 8,
            "rg16f": 4,
            "r16f": 2,
            "bc7": 0.5,  # 压缩格式，平均每像素0.5字节
            "bc3": 0.5,
            "bc1": 0.25,
            "etc2": 0.5,
            "etc1": 0.25,
            "astc_4x4": 0.5,
            "astc_8x8": 0.125
        }
        
        return format_bytes.get(format.lower(), 4)  # 默认返回4字节
    
    def _initialize_cache_cleanup(self):
        """
        初始化缓存清理线程
        """
        def cleanup_thread():
            while self.cache_cleanup_settings["enabled"]:
                # 休眠指定间隔
                time.sleep(self.cache_cleanup_settings["check_interval"])
                
                # 执行清理
                self._cleanup_unused_resources()
        
        # 启动线程
        self._cleanup_thread = threading.Thread(target=cleanup_thread, daemon=True)
        self._cleanup_thread.start()
        print("缓存清理线程已启动")
    
    def _cleanup_unused_resources(self):
        """
        清理长时间未使用的资源
        """
        with self.lock:
            timestamp = time.time()
            resources_to_remove = []
            
            # 计算当前VRAM使用情况
            current_usage = self.current_vram_usage
            total_vram = self.current_vram_limit
            usage_percentage = (current_usage / total_vram) * 100
            
            # 只有当VRAM使用超过60%时才进行主动清理
            if usage_percentage < 60:
                return
            
            for resource_id, resource_info in self.resource_registry.items():
                # 跳过不可卸载的资源
                if resource_info["resource_type"] in ["constant_buffer", "shader_resource_view"]:
                    continue
                
                # 检查资源年龄
                age = timestamp - resource_info["created_at"]
                if age < self.cache_cleanup_settings["min_usage_duration"]:
                    continue
                
                # 检查空闲时间
                idle_time = timestamp - resource_info["last_used"]
                
                # 根据VRAM使用情况动态调整清理阈值
                # VRAM使用越高，清理阈值越低
                dynamic_threshold = self.cache_cleanup_settings["max_idle_time"] * (1 - (usage_percentage - 60) / 40)
                dynamic_threshold = max(10.0, dynamic_threshold)  # 最低10秒
                
                if idle_time > dynamic_threshold:
                    # 计算清理分数（优先级、空闲时间、资源大小）
                    priority_factor = 11 - resource_info["priority"]  # 低优先级资源更容易被清理
                    idle_factor = idle_time / dynamic_threshold  # 空闲时间越长，分数越高
                    size_factor = resource_info["current_size_mb"] / total_vram  # 资源越大，分数越高
                    
                    # 已压缩的资源分数更高（更容易被清理）
                    compressed_factor = 1.5 if resource_info["compressed"] else 1.0
                    
                    # 纹理资源分数更高（更容易被清理）
                    texture_factor = 1.2 if resource_info["resource_type"].startswith("texture") else 1.0
                    
                    # 计算总分数
                    score = priority_factor * idle_factor * size_factor * compressed_factor * texture_factor
                    
                    # 分数超过阈值则标记为清理
                    if score > 1.0:
                        resources_to_remove.append(resource_id)
            
            # 按大小排序，优先清理大资源
            resources_to_remove.sort(key=lambda id: self.resource_registry[id]["current_size_mb"], reverse=True)
            
            # 限制清理数量，避免一次性清理太多资源
            max_cleanup_count = min(10, len(resources_to_remove))  # 最多清理10个资源
            
            # 移除长时间未使用的资源
            freed_memory = 0.0
            for i in range(max_cleanup_count):
                resource_id = resources_to_remove[i]
                resource_info = self.resource_registry[resource_id]
                freed_memory += resource_info["current_size_mb"]
                print(f"自动清理资源: {resource_id}, 大小: {resource_info['current_size_mb']:.2f} MB, 空闲时间: {timestamp - resource_info['last_used']:.1f}秒")
                self.unregister_resource(resource_id)
            
            if freed_memory > 0:
                print(f"自动清理完成，释放显存: {freed_memory:.2f} MB")
    
    def enable_aggressive_compression(self, enable):
        """
        启用或禁用激进的纹理压缩
        
        Args:
            enable: 是否启用激进的纹理压缩
        """
        with self.lock:
            self.compression_settings["aggressive_compression"] = enable
            print(f"启用激进的纹理压缩: {enable}")
    
    def enable_texture_streaming(self, enable):
        """
        启用或禁用纹理流式加载
        
        Args:
            enable: 是否启用纹理流式加载
        """
        with self.lock:
            self.texture_streaming_settings["enabled"] = enable
            print(f"启用纹理流式加载: {enable}")
    
    def set_texture_retention_time(self, retention_time):
        """
        设置纹理保留时间（仅作为兼容方法，实际使用缓存清理设置）
        
        Args:
            retention_time: 纹理保留时间（秒）
        """
        with self.lock:
            self.cache_cleanup_settings["max_idle_time"] = retention_time
            print(f"设置纹理保留时间: {retention_time}秒")
    
    def frame_begin(self):
        """
        帧开始时的VRAM管理
        """
        with self.lock:
            # 帧开始时的VRAM管理操作
            # 例如：更新资源使用时间、检查内存压力等
            timestamp = time.time()
            
            # 更新资源优先级队列
            self._update_priority_queue()
            
            # 检查VRAM使用情况，如果接近限制则尝试释放一些资源
            safety_threshold = self.current_vram_limit * (self.safety_margin_percentage / 100.0)
            if self.current_vram_usage > self.current_vram_limit - safety_threshold:
                # 尝试释放一些内存
                self._free_memory(safety_threshold)
    
    def frame_end(self):
        """
        帧结束时的VRAM管理
        """
        with self.lock:
            # 帧结束时的VRAM管理操作
            # 例如：更新资源使用时间、执行延迟释放等
            timestamp = time.time()
            
            # 更新资源使用时间
            for resource_id, resource_info in self.resource_registry.items():
                if resource_info["last_used"] == timestamp:
                    # 标记为最近使用
                    resource_info["last_used"] = timestamp
            
            # 执行延迟释放操作（如果有）
            # 这里可以添加一些延迟释放的逻辑
            pass
    
    def shutdown(self):
        """
        关闭VRAM管理器，释放所有资源
        """
        with self.lock:
            # 清空资源注册表
            self.resource_registry.clear()
            
            # 清空资源优先级队列
            self.resource_priority_queue.clear()
            
            # 清空可压缩资源集合
            self.compressible_resources.clear()
            
            # 重置VRAM使用量
            self.current_vram_usage = 0
            
            # 禁用缓存清理
            self.cache_cleanup_settings["enabled"] = False
            
            # 禁用纹理流式加载
            self.texture_streaming_settings["enabled"] = False
            
            print("VRAM管理器已关闭，所有资源已释放")

# 示例用法
if __name__ == "__main__":
    vram_manager = VRAMManager()
    
    # 设置GPU配置文件为GTX 750Ti
    vram_manager.set_gpu_profile("gtx_750ti")
    
    # 注册一些测试资源
    vram_manager.register_resource("texture_001", "texture_2d", 100.0, priority=9, compressible=True)
    vram_manager.register_resource("texture_002", "texture_2d", 150.0, priority=7, compressible=True)
    vram_manager.register_resource("mesh_001", "vertex_buffer", 50.0, priority=8, compressible=False)
    vram_manager.register_resource("constant_buffer_001", "constant_buffer", 2.0, priority=10, compressible=False)
    
    # 打印内存报告
    vram_manager.print_memory_report()
    
    # 尝试压缩资源
    freed = vram_manager.compress_resources()
    print(f"压缩释放的显存: {freed:.2f} MB")
    
    # 再次打印内存报告
    vram_manager.print_memory_report()