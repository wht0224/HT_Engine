# -*- coding: utf-8 -*-
"""
低端GPU渲染引擎性能基准测试框架
专为GTX 750Ti和RX 580等中端硬件设计的性能监控和测试工具
"""

import time
import numpy as np
from Engine.Math import Vector3


class Benchmark:
    """性能基准测试管理器"""
    
    def __init__(self, engine):
        """初始化基准测试框架
        
        Args:
            engine: 渲染引擎实例
        """
        self.engine = engine
        self.platform = engine.platform
        
        # 性能计数器
        self.frame_count = 0
        self.fps_history = []
        self.current_time = time.time()
        self.last_time = self.current_time
        self.frame_time = 0.0
        self.average_fps = 0.0
        
        # 渲染统计
        self.draw_calls = 0
        self.triangle_count = 0
        self.texture_switches = 0
        self.shader_switches = 0
        
        # 内存使用情况
        self.vram_usage = 0  # MB
        self.total_vram = 0  # MB
        
        # 性能测试设置
        self.test_duration = 60  # 秒
        self.test_start_time = 0
        self.is_benchmarking = False
        self.benchmark_results = {}
        
        # 目标硬件性能指标
        self.target_performance = {
            "GTX_750Ti": {
                "target_fps": 30,
                "max_draw_calls": 1000,
                "max_triangles": 500000,
                "max_vram": 1800,  # MB
                "optimal_texture_size": 1024
            },
            "RX_580": {
                "target_fps": 60,
                "max_draw_calls": 2000,
                "max_triangles": 2000000,
                "max_vram": 4000,  # MB
                "optimal_texture_size": 2048
            }
        }
        
        # 检测当前GPU
        self.gpu_type = self._detect_gpu_type()
        self.target_metrics = self.target_performance.get(self.gpu_type, self.target_performance["GTX_750Ti"])
        
        # 性能警告阈值
        self.warning_thresholds = {
            "fps": self.target_metrics["target_fps"] * 0.8,  # 低于目标80%时警告
            "vram": self.target_metrics["max_vram"] * 0.9,   # 使用90%VRAM时警告
            "draw_calls": self.target_metrics["max_draw_calls"] * 0.8,  # 80%时警告
            "triangle_count": self.target_metrics["max_triangles"] * 0.8  # 80%时警告
        }
        
        # 初始化性能日志
        self.log_entries = []
    
    def _detect_gpu_type(self):
        """根据硬件信息检测GPU类型
        
        Returns:
            str: GPU类型标识符
        """
        if self.platform is None:
            return "UNKNOWN"
        
        gpu_name = self.platform.gpu_name.upper()
        if "750TI" in gpu_name:
            return "GTX_750Ti"
        elif "RX 580" in gpu_name or "RX580" in gpu_name:
            return "RX_580"
        else:
            # 如果是其他GPU，根据VRAM大小选择最合适的性能配置
            if self.platform.total_vram < 2048:
                return "GTX_750Ti"  # 小于2GB VRAM，使用750Ti配置
            else:
                return "RX_580"  # 大于等于2GB VRAM，使用580配置
    
    def start_frame(self):
        """开始一帧的计时"""
        self.current_time = time.time()
    
    def end_frame(self):
        """结束一帧并更新性能统计"""
        self.frame_count += 1
        
        # 计算帧率
        new_time = time.time()
        delta_time = new_time - self.current_time
        self.frame_time = delta_time
        
        # 每1秒更新一次FPS
        if new_time - self.last_time >= 1.0:
            current_fps = 1.0 / delta_time if delta_time > 0 else 0
            self.fps_history.append(current_fps)
            
            # 保留最近60帧的FPS历史
            if len(self.fps_history) > 60:
                self.fps_history.pop(0)
            
            # 计算平均FPS
            self.average_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
            
            # 更新VRAM使用情况
            if self.platform:
                self.vram_usage = self.platform.current_vram_usage
                self.total_vram = self.platform.total_vram
            
            # 重置计数器
            self.draw_calls = 0
            self.triangle_count = 0
            self.texture_switches = 0
            self.shader_switches = 0
            
            # 检查性能警告
            self._check_performance_warnings()
            
            # 更新基准测试状态
            if self.is_benchmarking:
                self._update_benchmark()
            
            self.last_time = new_time
    
    def _check_performance_warnings(self):
        """检查性能指标是否达到警告阈值"""
        warnings = []
        
        # 检查FPS
        if self.average_fps < self.warning_thresholds["fps"]:
            warnings.append(f"性能警告: FPS ({self.average_fps:.1f}) 低于目标值")
        
        # 检查VRAM使用
        if self.vram_usage > self.warning_thresholds["vram"]:
            warnings.append(f"性能警告: VRAM使用 ({self.vram_usage:.1f}MB) 接近最大容量")
        
        # 检查绘制调用
        if self.draw_calls > self.warning_thresholds["draw_calls"]:
            warnings.append(f"性能警告: 绘制调用 ({self.draw_calls}) 接近限制")
        
        # 检查三角形数量
        if self.triangle_count > self.warning_thresholds["triangle_count"]:
            warnings.append(f"性能警告: 三角形数量 ({self.triangle_count}) 接近限制")
        
        # 记录警告
        for warning in warnings:
            self.log_performance_warning(warning)
    
    def increment_draw_calls(self, count=1):
        """增加绘制调用计数
        
        Args:
            count: 增加的数量
        """
        self.draw_calls += count
    
    def increment_triangle_count(self, count):
        """增加三角形计数
        
        Args:
            count: 增加的数量
        """
        self.triangle_count += count
    
    def increment_texture_switches(self, count=1):
        """增加纹理切换计数
        
        Args:
            count: 增加的数量
        """
        self.texture_switches += count
    
    def increment_shader_switches(self, count=1):
        """增加着色器切换计数
        
        Args:
            count: 增加的数量
        """
        self.shader_switches += count
    
    def log_performance_warning(self, message):
        """记录性能警告
        
        Args:
            message: 警告消息
        """
        log_entry = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(log_entry)
        self.log_entries.append(log_entry)
    
    def get_performance_stats(self):
        """获取当前性能统计信息
        
        Returns:
            dict: 性能统计数据
        """
        return {
            "fps": self.average_fps,
            "frame_time": self.frame_time * 1000,  # 转换为毫秒
            "draw_calls": self.draw_calls,
            "triangle_count": self.triangle_count,
            "texture_switches": self.texture_switches,
            "shader_switches": self.shader_switches,
            "vram_usage": self.vram_usage,
            "total_vram": self.total_vram,
            "gpu_type": self.gpu_type,
            "target_fps": self.target_metrics["target_fps"]
        }
    
    def start_benchmark(self, duration=60):
        """开始基准测试
        
        Args:
            duration: 测试持续时间（秒）
        """
        self.test_duration = duration
        self.test_start_time = time.time()
        self.is_benchmarking = True
        self.benchmark_results = {
            "start_time": self.test_start_time,
            "duration": duration,
            "gpu_type": self.gpu_type,
            "fps_samples": [],
            "vram_samples": [],
            "frame_time_samples": [],
            "max_fps": 0,
            "min_fps": float('inf'),
            "avg_fps": 0
        }
        
        print(f"开始基准测试，持续时间 {duration} 秒...")
    
    def _update_benchmark(self):
        """更新基准测试数据"""
        elapsed = time.time() - self.test_start_time
        
        # 记录FPS样本
        self.benchmark_results["fps_samples"].append(self.average_fps)
        self.benchmark_results["vram_samples"].append(self.vram_usage)
        self.benchmark_results["frame_time_samples"].append(self.frame_time * 1000)  # 毫秒
        
        # 更新最大/最小FPS
        if self.average_fps > self.benchmark_results["max_fps"]:
            self.benchmark_results["max_fps"] = self.average_fps
        if self.average_fps < self.benchmark_results["min_fps"]:
            self.benchmark_results["min_fps"] = self.average_fps
        
        # 检查是否完成测试
        if elapsed >= self.test_duration:
            self.finish_benchmark()
        else:
            # 显示进度
            progress = (elapsed / self.test_duration) * 100
            print(f"测试进度: {progress:.1f}%, 当前FPS: {self.average_fps:.1f}")
    
    def finish_benchmark(self):
        """完成基准测试并生成报告"""
        self.is_benchmarking = False
        
        # 计算平均FPS
        if self.benchmark_results["fps_samples"]:
            self.benchmark_results["avg_fps"] = sum(self.benchmark_results["fps_samples"]) / len(self.benchmark_results["fps_samples"])
            
            # 计算90/95/99百分位数
            sorted_fps = sorted(self.benchmark_results["fps_samples"])
            n = len(sorted_fps)
            if n > 0:
                self.benchmark_results["p90_fps"] = sorted_fps[int(n * 0.9)]
                self.benchmark_results["p95_fps"] = sorted_fps[int(n * 0.95)]
                self.benchmark_results["p99_fps"] = sorted_fps[int(n * 0.99)]
        
        # 生成报告
        self._generate_benchmark_report()
        
        return self.benchmark_results
    
    def _generate_benchmark_report(self):
        """生成基准测试报告"""
        print("\n===== 基准测试报告 =====")
        print(f"GPU类型: {self.gpu_type}")
        print(f"测试持续时间: {self.test_duration} 秒")
        print(f"平均FPS: {self.benchmark_results['avg_fps']:.2f}")
        print(f"最大FPS: {self.benchmark_results['max_fps']:.2f}")
        print(f"最小FPS: {self.benchmark_results['min_fps']:.2f}")
        
        if "p90_fps" in self.benchmark_results:
            print(f"90% FPS: {self.benchmark_results['p90_fps']:.2f}")
            print(f"95% FPS: {self.benchmark_results['p95_fps']:.2f}")
            print(f"99% FPS: {self.benchmark_results['p99_fps']:.2f}")
        
        # 计算平均VRAM使用
        if self.benchmark_results["vram_samples"]:
            avg_vram = sum(self.benchmark_results["vram_samples"]) / len(self.benchmark_results["vram_samples"])
            print(f"平均VRAM使用: {avg_vram:.1f} MB / {self.total_vram:.1f} MB ({(avg_vram/self.total_vram)*100:.1f}%)")
        
        # 性能评级
        target_fps = self.target_metrics["target_fps"]
        if self.benchmark_results['avg_fps'] >= target_fps * 1.2:
            rating = "优秀"
        elif self.benchmark_results['avg_fps'] >= target_fps:
            rating = "良好"
        elif self.benchmark_results['avg_fps'] >= target_fps * 0.8:
            rating = "一般"
        else:
            rating = "较差"
        
        print(f"\n性能评级: {rating}")
        
        # 优化建议
        print("\n优化建议:")
        if self.benchmark_results['avg_fps'] < target_fps * 0.8:
            print("1. 考虑减少场景复杂度或降低纹理分辨率")
            print("2. 增加LOD级别或降低远处对象的多边形数量")
            print("3. 检查是否存在过多的绘制调用或状态切换")
        
        # VRAM使用过高
        if self.benchmark_results["vram_samples"]:
            max_vram = max(self.benchmark_results["vram_samples"])
            if max_vram > self.total_vram * 0.9:
                print("4. 减少纹理内存使用，考虑使用更激进的压缩")
                print("5. 实现纹理流式加载，只加载视口内可见的高分辨率纹理")
        
        print("========================\n")
    
    def get_optimization_suggestions(self):
        """获取针对当前场景的优化建议
        
        Returns:
            list: 优化建议列表
        """
        suggestions = []
        
        # FPS相关建议
        if self.average_fps < self.target_metrics["target_fps"] * 0.8:
            suggestions.append("降低场景复杂度或启用更激进的LOD设置")
            suggestions.append(f"考虑将纹理分辨率降低到{self.target_metrics['optimal_texture_size']}px")
            suggestions.append("禁用或降低屏幕空间效果（如SSR、SSAO）的质量")
        
        # 绘制调用相关建议
        if self.draw_calls > self.target_metrics["max_draw_calls"] * 0.8:
            suggestions.append("增加实例化渲染使用")
            suggestions.append("合并相似材质的对象")
            suggestions.append("实现几何实例化以减少绘制调用")
        
        # 三角形数量相关建议
        if self.triangle_count > self.target_metrics["max_triangles"] * 0.8:
            suggestions.append("对高多边形模型应用网格简化")
            suggestions.append("增加LOD级别或提高LOD切换距离")
            suggestions.append("使用代理几何体表示远处的复杂物体")
        
        # VRAM相关建议
        vram_percentage = (self.vram_usage / self.total_vram) * 100 if self.total_vram > 0 else 0
        if vram_percentage > 85:
            suggestions.append("使用更高效的纹理压缩格式")
            suggestions.append("实现纹理流式加载系统")
            suggestions.append("释放不可见对象的纹理资源")
        
        # GPU特定建议
        if self.gpu_type == "GTX_750Ti":
            suggestions.append("针对Maxwell架构，考虑使用BC7压缩而非BC3")
            suggestions.append("减少计算着色器的使用，优先使用光栅化管线")
            suggestions.append("限制同时使用的动态光源数量")
        elif self.gpu_type == "RX_580":
            suggestions.append("针对GCN架构，考虑使用更多并行计算着色器")
            suggestions.append("优化内存访问模式以提高带宽利用率")
        
        return suggestions
    
    def compare_with_rtx4090_quality(self, current_scene_complexity):
        """将当前渲染质量与RTX 4090标准进行比较
        
        Args:
            current_scene_complexity: 当前场景复杂度评级 (0-10)
            
        Returns:
            dict: 质量比较结果
        """
        # RTX 4090参考标准（简化模型）
        rtx4090_reference = {
            "max_texture_resolution": 8192,
            "max_triangles": 100000000,  # 1亿
            "max_draw_calls": 100000,
            "max_vram": 24000,  # 24GB
            "target_fps": 120
        }
        
        # 当前硬件能力
        current_capabilities = {
            "max_texture_resolution": self.target_metrics["optimal_texture_size"] * 2,
            "max_triangles": self.target_metrics["max_triangles"] * 1.2,
            "max_draw_calls": self.target_metrics["max_draw_calls"] * 1.2,
            "max_vram": self.target_metrics["max_vram"],
            "target_fps": self.target_metrics["target_fps"]
        }
        
        # 计算相对性能
        relative_performance = {
            "texture_resolution": current_capabilities["max_texture_resolution"] / rtx4090_reference["max_texture_resolution"],
            "triangle_capacity": current_capabilities["max_triangles"] / rtx4090_reference["max_triangles"],
            "draw_call_capacity": current_capabilities["max_draw_calls"] / rtx4090_reference["max_draw_calls"],
            "vram_capacity": current_capabilities["max_vram"] / rtx4090_reference["max_vram"],
            "fps_capacity": current_capabilities["target_fps"] / rtx4090_reference["target_fps"]
        }
        
        # 质量比较分析
        quality_ratio = min(relative_performance.values())
        complexity_factor = current_scene_complexity / 10.0
        
        # 计算等效质量
        equivalent_quality = quality_ratio / complexity_factor if complexity_factor > 0 else 0
        
        # 生成比较报告
        comparison = {
            "quality_ratio": equivalent_quality,
            "relative_performance": relative_performance,
            "quality_evaluation": self._evaluate_quality_equivalence(equivalent_quality),
            "optimization_potential": self._calculate_optimization_potential(relative_performance)
        }
        
        return comparison
    
    def _evaluate_quality_equivalence(self, ratio):
        """评估质量等效性
        
        Args:
            ratio: 质量等效比率
            
        Returns:
            dict: 质量评估结果
        """
        if ratio > 0.8:
            return {
                "rating": "优秀",
                "description": "渲染质量接近RTX 4090水平",
                "feedback": "当前优化非常有效，视觉质量表现优异"
            }
        elif ratio > 0.6:
            return {
                "rating": "良好",
                "description": "渲染质量达到RTX 4090的60-80%",
                "feedback": "优化效果好，但仍有提升空间"
            }
        elif ratio > 0.4:
            return {
                "rating": "中等",
                "description": "渲染质量达到RTX 4090的40-60%",
                "feedback": "需要进一步优化以提升视觉质量"
            }
        else:
            return {
                "rating": "一般",
                "description": "渲染质量低于RTX 4090的40%",
                "feedback": "需要显著优化以提升视觉表现"
            }
    
    def _calculate_optimization_potential(self, relative_perf):
        """计算优化潜力
        
        Args:
            relative_perf: 相对性能指标
            
        Returns:
            dict: 优化潜力分析
        """
        # 找出性能瓶颈
        bottleneck = min(relative_perf.items(), key=lambda x: x[1])
        
        optimization_potential = {
            "bottleneck": bottleneck[0],
            "bottleneck_value": bottleneck[1],
            "improvement_suggestions": []
        }
        
        # 根据瓶颈提供建议
        if bottleneck[0] == "vram_capacity":
            optimization_potential["improvement_suggestions"].extend([
                "实现更高效的纹理压缩算法",
                "优化几何数据存储",
                "开发智能纹理流式加载系统"
            ])
        elif bottleneck[0] == "triangle_capacity":
            optimization_potential["improvement_suggestions"].extend([
                "实现高级LOD系统",
                "使用几何体实例化技术",
                "优化网格简化算法保留视觉重要细节"
            ])
        elif bottleneck[0] == "draw_call_capacity":
            optimization_potential["improvement_suggestions"].extend([
                "实现GPU实例化",
                "合并相似材质对象",
                "使用几何体着色器进行批处理"
            ])
        elif bottleneck[0] == "texture_resolution":
            optimization_potential["improvement_suggestions"].extend([
                "实现纹理压缩与解压缩优化",
                "使用纹理图集减少状态切换",
                "开发智能纹理分辨率缩放系统"
            ])
        
        return optimization_potential
    
    def save_benchmark_results(self, filename="benchmark_results.txt"):
        """保存基准测试结果到文件
        
        Args:
            filename: 文件名
        """
        if not self.benchmark_results:
            print("没有可用的基准测试结果")
            return False
        
        try:
            with open(filename, "w") as f:
                f.write("===== 低端GPU渲染引擎基准测试结果 =====\n")
                f.write(f"日期时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"GPU类型: {self.gpu_type}\n")
                f.write(f"测试持续时间: {self.benchmark_results['duration']} 秒\n\n")
                
                f.write("性能指标:\n")
                f.write(f"平均FPS: {self.benchmark_results['avg_fps']:.2f}\n")
                f.write(f"最大FPS: {self.benchmark_results['max_fps']:.2f}\n")
                f.write(f"最小FPS: {self.benchmark_results['min_fps']:.2f}\n")
                
                if "p90_fps" in self.benchmark_results:
                    f.write(f"90% FPS: {self.benchmark_results['p90_fps']:.2f}\n")
                    f.write(f"95% FPS: {self.benchmark_results['p95_fps']:.2f}\n")
                    f.write(f"99% FPS: {self.benchmark_results['p99_fps']:.2f}\n")
                
                # VRAM统计
                if self.benchmark_results["vram_samples"]:
                    avg_vram = sum(self.benchmark_results["vram_samples"]) / len(self.benchmark_results["vram_samples"])
                    max_vram = max(self.benchmark_results["vram_samples"])
                    f.write(f"\n内存使用:\n")
                    f.write(f"平均VRAM使用: {avg_vram:.1f} MB\n")
                    f.write(f"最大VRAM使用: {max_vram:.1f} MB\n")
                    f.write(f"总VRAM容量: {self.total_vram:.1f} MB\n")
                
                # 优化建议
                suggestions = self.get_optimization_suggestions()
                if suggestions:
                    f.write(f"\n优化建议:\n")
                    for i, suggestion in enumerate(suggestions, 1):
                        f.write(f"{i}. {suggestion}\n")
                
            print(f"基准测试结果已保存至 {filename}")
            return True
        except Exception as e:
            print(f"保存基准测试结果时出错: {e}")
            return False