# -*- coding: utf-8 -*-
"""
UI事件系统，用于处理UI控件的事件
"""

from enum import Enum

class EventType(Enum):
    """事件类型枚举"""
    MOUSE_ENTER = "mouse_enter"      # 鼠标进入控件
    MOUSE_LEAVE = "mouse_leave"      # 鼠标离开控件
    MOUSE_DOWN = "mouse_down"        # 鼠标按下
    MOUSE_UP = "mouse_up"          # 鼠标释放
    MOUSE_CLICK = "mouse_click"      # 鼠标点击
    MOUSE_DOUBLE_CLICK = "mouse_double_click"  # 鼠标双击
    MOUSE_DRAG = "mouse_drag"        # 鼠标拖动
    MOUSE_WHEEL = "mouse_wheel"      # 鼠标滚轮
    KEY_DOWN = "key_down"          # 键盘按下
    KEY_UP = "key_up"            # 键盘释放
    TEXT_INPUT = "text_input"        # 文本输入
    VALUE_CHANGED = "value_changed"    # 值改变
    FOCUS_GAINED = "focus_gained"      # 获得焦点
    FOCUS_LOST = "focus_lost"        # 失去焦点

class Event:
    """UI事件类"""
    
    def __init__(self, event_type, source, **kwargs):
        """初始化事件
        
        Args:
            event_type: 事件类型
            source: 事件源
            **kwargs: 事件参数
        """
        self.event_type = event_type
        self.source = source
        self.handled = False
        self.timestamp = 0.0  # 将在事件触发时设置
        self.params = kwargs
    
    def get_param(self, name, default=None):
        """获取事件参数
        
        Args:
            name: 参数名称
            default: 默认值
            
        Returns:
            参数值
        """
        return self.params.get(name, default)
    
    def set_param(self, name, value):
        """设置事件参数
        
        Args:
            name: 参数名称
            value: 参数值
        """
        self.params[name] = value
    
    def stop_propagation(self):
        """停止事件传播"""
        self.handled = True

class EventHandler:
    """事件处理器类"""
    
    def __init__(self, callback):
        """初始化事件处理器
        
        Args:
            callback: 事件回调函数
        """
        self.callback = callback
    
    def handle_event(self, event):
        """处理事件
        
        Args:
            event: 事件对象
        """
        if not event.handled:
            self.callback(event)
