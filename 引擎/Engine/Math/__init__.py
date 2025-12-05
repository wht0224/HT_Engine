# Math包初始化文件
"""
引擎数学库模块
"""

# 优先导入Cython版本的数学类，如果失败则导入Python版本
# 这样可以获得更好的性能，同时保持兼容性

try:
    from .CythonMath import Vector3, Matrix4x4, Quaternion, BoundingBox, Frustum, BoundingSphere
    # print("使用Cython版本数学库")
except ImportError:
    # 如果Cython版本不可用，回退到Python版本
    from .Math import Vector3, Matrix4x4, Quaternion, BoundingBox, Frustum, BoundingSphere
    # print("使用Python版本数学库")

# 导出所有数学类供外部使用
__all__ = [
    "Vector3",
    "Matrix4x4",
    "Quaternion",
    "BoundingBox",
    "BoundingSphere",
    "Frustum"
]