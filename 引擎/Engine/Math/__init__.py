# Math Package Initialization
"""
Engine Math Module
Optimized math library with Cython acceleration
"""

# Import Cython-optimized classes if available, fallback to Python implementation
try:
    # Import core performance classes from Cython
    from .CythonMath import Vector3, Matrix4x4, Quaternion
    # Import helper classes from Python version (not implemented in Cython)
    from .Math import BoundingBox, Frustum, BoundingSphere, Vector2
    MATH_BACKEND = "Cython+ASM"
except ImportError as e:
    # Fallback to pure Python version
    from .Math import Vector3, Matrix4x4, Quaternion, BoundingBox, Frustum, BoundingSphere, Vector2
    MATH_BACKEND = "Pure Python"

# Export all math classes
__all__ = [
    "Vector2",
    "Vector3",
    "Matrix4x4",
    "Quaternion",
    "BoundingBox",
    "BoundingSphere",
    "Frustum",
    "MATH_BACKEND"
]