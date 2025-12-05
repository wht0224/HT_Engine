# -*- coding: utf-8 -*-
"""
简单的性能监控器
提供start()和stop()方法，用于性能监控
"""

import time

class PerformanceMonitor:
    """
    性能监控器
    用于监控不同阶段的执行时间
    """
    
    def __init__(self):
        """初始化性能监控器"""
        self.timers = {}
        self.averages = {}
    
    def start(self, name):
        """
        开始监控一个阶段
        
        Args:
            name: 阶段名称
        """
        self.timers[name] = time.time()
    
    def stop(self, name):
        """
        停止监控一个阶段并记录时间
        
        Args:
            name: 阶段名称
        """
        if name in self.timers:
            elapsed = time.time() - self.timers[name]
            # 简单的平均值计算
            if name in self.averages:
                self.averages[name] = (self.averages[name] + elapsed) / 2
            else:
                self.averages[name] = elapsed
            del self.timers[name]
    
    def get_averages(self):
        """
        获取各阶段的平均执行时间
        
        Returns:
            dict: 各阶段的平均执行时间
        """
        return self.averages.copy()
    
    def reset(self):
        """
        重置所有计时器
        """
        self.timers.clear()
        self.averages.clear()