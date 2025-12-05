# Engine包初始化文件
"""
低端GPU渲染引擎主包
"""

__version__ = "1.0.0"
__author__ = "Advanced Diagnostics Team"

# 引擎单例实例
_engine_instance = None

def get_engine():
    """
    获取引擎实例
    """
    global _engine_instance
    if _engine_instance is None:
        from .Engine import Engine
        _engine_instance = Engine()
    return _engine_instance