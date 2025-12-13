import time
import threading
import numpy as np

class PerformanceMonitor:
    """
    性能监控工具
    实时显示帧率、显存使用、绘制调用等性能指标
    针对低端GPU优化，使用轻量级实现
    """
    
    def __init__(self):
        # 帧率监控
        self.frame_times = []
        self.fps = 0.0
        self.frame_count = 0
        self.last_fps_update = time.time()
        self.fps_update_interval = 1.0  # 每秒更新一次帧率
        
        # 渲染性能指标
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
        
        # 内存使用指标
        self.memory_stats = {
            "vram_usage": 0,
            "vram_budget": 0,
            "vram_percentage": 0,
            "system_memory_usage": 0,
            "system_memory_available": 0
        }
        
        # 硬件信息
        self.hardware_info = {
            "gpu_name": "Unknown",
            "gpu_architecture": "Unknown",
            "vram_size": 0,
            "cpu_name": "Unknown",
            "cpu_cores": 0,
            "system_memory": 0
        }
        
        # 监控线程
        self.monitor_thread = None
        self.running = False
        self.monitor_interval = 0.5  # 每500ms更新一次性能指标
        
        # 历史数据，用于绘制图表
        self.history_data = {
            "fps": [],
            "render_time_ms": [],
            "draw_calls": [],
            "vram_usage": []
        }
        self.max_history_size = 60  # 保留60秒的历史数据
    
    def start(self):
        """
        启动性能监控
        """
        if self.running:
            return
        
        self.running = True
        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("性能监控已启动")
    
    def stop(self):
        """
        停止性能监控
        """
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
            self.monitor_thread = None
        print("性能监控已停止")
    
    def _monitor_loop(self):
        """
        监控线程主循环
        """
        while self.running:
            time.sleep(self.monitor_interval)
            # 收集性能指标
            self._collect_performance_stats()
            self._collect_memory_stats()
            
            # 更新历史数据
            self._update_history()
    
    def _collect_performance_stats(self):
        """
        收集渲染性能指标
        """
        # 这里可以添加更多性能数据的收集逻辑
        pass
    
    def _collect_memory_stats(self):
        """
        收集内存使用指标
        """
        # 这里可以添加系统内存和VRAM使用量的收集逻辑
        pass
    
    def _update_history(self):
        """
        更新历史数据
        """
        # 更新帧率历史
        self.history_data["fps"].append(self.fps)
        if len(self.history_data["fps"]) > self.max_history_size:
            self.history_data["fps"].pop(0)
        
        # 更新渲染时间历史
        self.history_data["render_time_ms"].append(self.performance_stats["render_time_ms"])
        if len(self.history_data["render_time_ms"]) > self.max_history_size:
            self.history_data["render_time_ms"].pop(0)
        
        # 更新绘制调用历史
        self.history_data["draw_calls"].append(self.performance_stats["draw_calls"])
        if len(self.history_data["draw_calls"]) > self.max_history_size:
            self.history_data["draw_calls"].pop(0)
        
        # 更新VRAM使用历史
        self.history_data["vram_usage"].append(self.memory_stats["vram_usage"])
        if len(self.history_data["vram_usage"]) > self.max_history_size:
            self.history_data["vram_usage"].pop(0)
    
    def update_frame(self, render_time_ms=0):
        """
        更新帧信息，计算帧率
        
        Args:
            render_time_ms: 当前帧的渲染时间（毫秒）
        """
        # 更新帧计数
        self.frame_count += 1
        
        # 记录当前帧的渲染时间
        self.performance_stats["render_time_ms"] = render_time_ms
        
        # 更新帧率
        current_time = time.time()
        elapsed = current_time - self.last_fps_update
        
        if elapsed >= self.fps_update_interval:
            # 计算帧率
            self.fps = self.frame_count / elapsed
            
            # 重置计数器
            self.frame_count = 0
            self.last_fps_update = current_time
    
    def update_performance_stats(self, stats):
        """
        更新渲染性能统计信息
        
        Args:
            stats: 包含性能统计信息的字典
        """
        for key, value in stats.items():
            if key in self.performance_stats:
                self.performance_stats[key] = value
    
    def update_memory_stats(self, stats):
        """
        更新内存使用统计信息
        
        Args:
            stats: 包含内存统计信息的字典
        """
        for key, value in stats.items():
            if key in self.memory_stats:
                self.memory_stats[key] = value
        
        # 计算VRAM使用率
        if self.memory_stats["vram_budget"] > 0:
            self.memory_stats["vram_percentage"] = (self.memory_stats["vram_usage"] / self.memory_stats["vram_budget"]) * 100
        else:
            self.memory_stats["vram_percentage"] = 0
    
    def set_hardware_info(self, info):
        """
        设置硬件信息
        
        Args:
            info: 包含硬件信息的字典
        """
        for key, value in info.items():
            if key in self.hardware_info:
                self.hardware_info[key] = value
    
    def get_performance_summary(self):
        """
        获取性能摘要
        
        Returns:
            dict: 包含性能摘要的字典
        """
        return {
            "fps": self.fps,
            "performance": self.performance_stats.copy(),
            "memory": self.memory_stats.copy(),
            "hardware": self.hardware_info.copy()
        }
    
    def print_performance_summary(self):
        """
        打印性能摘要
        """
        summary = self.get_performance_summary()
        
        print("\n=== 性能监控摘要 ===")
        print(f"帧率: {summary['fps']:.1f} FPS")
        print("\n渲染性能:")
        print(f"  渲染时间: {summary['performance']['render_time_ms']:.2f} ms")
        print(f"  绘制调用: {summary['performance']['draw_calls']}")
        print(f"  三角形数: {summary['performance']['triangles']:,}")
        print(f"  可见对象: {summary['performance']['visible_objects']}")
        print(f"  裁剪对象: {summary['performance']['culled_objects']}")
        print(f"  实例化对象: {summary['performance']['instanced_objects']}")
        print(f"  批处理对象: {summary['performance']['batched_objects']}")
        print("\n内存使用:")
        print(f"  VRAM使用: {summary['memory']['vram_usage']:.2f} MB / {summary['memory']['vram_budget']} MB ({summary['memory']['vram_percentage']:.1f}%)")
        print("\n硬件信息:")
        print(f"  GPU: {summary['hardware']['gpu_name']} ({summary['hardware']['gpu_architecture']})")
        print(f"  VRAM: {summary['hardware']['vram_size']} MB")
        print("==================\n")
    
    def reset_stats(self):
        """
        重置所有性能统计信息
        """
        # 重置性能统计
        for key in self.performance_stats:
            self.performance_stats[key] = 0
        
        # 重置帧率计数器
        self.frame_count = 0
        self.last_fps_update = time.time()
        
        # 清空历史数据
        for key in self.history_data:
            self.history_data[key] = []

# 全局性能监控实例
performance_monitor = PerformanceMonitor()