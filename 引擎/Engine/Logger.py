# -*- coding: utf-8 -*-
"""
日志系统，支持不同日志级别和格式化输出
"""

import os
import sys
import time
import traceback
from enum import Enum

class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3

class Logger:
    """日志类，支持不同日志级别和格式化输出"""
    
    def __init__(self, name, log_level=LogLevel.INFO, log_file=None):
        """
        初始化日志器
        
        Args:
            name: 日志器名称
            log_level: 日志输出级别
            log_file: 日志文件路径，None表示只输出到控制台
        """
        self.name = name
        self.log_level = log_level
        self.log_file = log_file
        self.log_count = 0
        self.max_log_size = 10 * 1024 * 1024  # 10MB
        self.max_backup_files = 5  # 最大备份文件数
        
        # 确保日志目录存在
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                try:
                    os.makedirs(log_dir)
                except Exception as e:
                    print(f"创建日志目录失败: {e}")
    
    def set_log_level(self, log_level):
        """
        设置日志输出级别
        
        Args:
            log_level: 日志输出级别
        """
        self.log_level = log_level
    
    def _should_log(self, level):
        """
        检查是否应该输出该级别的日志
        
        Args:
            level: 日志级别
            
        Returns:
            bool: 是否应该输出日志
        """
        return level.value >= self.log_level.value
    
    def _format_log(self, level, message):
        """
        格式化日志消息
        
        Args:
            level: 日志级别
            message: 日志消息
            
        Returns:
            str: 格式化后的日志消息
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        level_name = level.name.ljust(7)
        # 添加进程ID和线程ID，便于调试
        import os
        import threading
        pid = os.getpid()
        tid = threading.get_ident()
        return f"[{timestamp}] [{level_name}] [{self.name}] [PID:{pid}] [TID:{tid}] {message}"
    
    def _write_log(self, formatted_log):
        """
        写入日志
        
        Args:
            formatted_log: 格式化后的日志消息
        """
        # 输出到控制台
        try:
            print(formatted_log)
        except Exception as e:
            # 防止控制台输出失败导致程序崩溃
            pass
        
        # 输出到文件
        if self.log_file:
            try:
                # 检查日志文件大小
                if os.path.exists(self.log_file):
                    file_size = os.path.getsize(self.log_file)
                    if file_size > self.max_log_size:
                        # 滚动日志文件
                        self._rotate_logs()
                
                # 写入日志
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(f"{formatted_log}\n")
            except Exception as e:
                # 防止日志文件写入失败导致程序崩溃
                try:
                    print(f"写入日志文件失败: {e}")
                except:
                    pass
    
    def _rotate_logs(self):
        """
        滚动日志文件
        """
        try:
            # 先删除最旧的备份文件
            for i in range(self.max_backup_files - 1, 0, -1):
                old_backup = f"{self.log_file}.{i}"
                new_backup = f"{self.log_file}.{i+1}"
                if os.path.exists(old_backup):
                    if os.path.exists(new_backup):
                        os.remove(new_backup)
                    os.rename(old_backup, new_backup)
            
            # 重命名当前日志文件为第一个备份文件
            backup_file = f"{self.log_file}.1"
            if os.path.exists(backup_file):
                os.remove(backup_file)
            os.rename(self.log_file, backup_file)
        except Exception as e:
            # 防止日志滚动失败导致程序崩溃
            try:
                print(f"滚动日志文件失败: {e}")
            except:
                pass
    
    def debug(self, message):
        """
        输出DEBUG级别日志
        
        Args:
            message: 日志消息
        """
        if self._should_log(LogLevel.DEBUG):
            formatted_log = self._format_log(LogLevel.DEBUG, message)
            self._write_log(formatted_log)
    
    def info(self, message):
        """
        输出INFO级别日志
        
        Args:
            message: 日志消息
        """
        if self._should_log(LogLevel.INFO):
            formatted_log = self._format_log(LogLevel.INFO, message)
            self._write_log(formatted_log)
    
    def warning(self, message):
        """
        输出WARNING级别日志
        
        Args:
            message: 日志消息
        """
        if self._should_log(LogLevel.WARNING):
            formatted_log = self._format_log(LogLevel.WARNING, message)
            self._write_log(formatted_log)
    
    def error(self, message, exc_info=False):
        """
        输出ERROR级别日志
        
        Args:
            message: 日志消息
            exc_info: 是否输出异常信息
        """
        if self._should_log(LogLevel.ERROR):
            formatted_log = self._format_log(LogLevel.ERROR, message)
            self._write_log(formatted_log)
            
            # 输出异常信息
            if exc_info:
                exc_msg = traceback.format_exc()
                self._write_log(f"{exc_msg}")

# 创建全局日志器实例
global_logger = Logger("Engine", LogLevel.INFO)

# 导出日志函数
def get_logger(name=None):
    """
    获取日志器
    
    Args:
        name: 日志器名称，None表示返回全局日志器
        
    Returns:
        Logger: 日志器实例
    """
    if name:
        return Logger(name, global_logger.log_level, global_logger.log_file)
    return global_logger

# 设置全局日志级别
def set_global_log_level(log_level):
    """
    设置全局日志级别
    
    Args:
        log_level: 日志级别
    """
    global_logger.set_log_level(log_level)

# 设置全局日志文件
def set_global_log_file(log_file):
    """
    设置全局日志文件
    
    Args:
        log_file: 日志文件路径
    """
    global_logger.log_file = log_file
