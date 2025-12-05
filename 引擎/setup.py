from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# 引擎数学库的构建配置文件
# Build configuration file for engine math library

# 定义Cython扩展模块列表
# Define list of Cython extension modules
extensions = [
    Extension(
        "Engine.Math.CythonMath",  # 模块完全限定名称
        ["Engine/Math/CythonMath.pyx"],  # 源文件相对路径
        include_dirs=[np.get_include()],  # 包含numpy头文件目录
        language="c"  # 使用C语言编译
    )
]

# 设置编译指令，优化性能
# Set compiler directives for performance optimization
compiler_directives = {
    # 使用Python 3语法
    "language_level": "3",
    # 禁用边界检查以提高性能
    "boundscheck": False,
    # 禁用负索引环绕
    "wraparound": False,
    # 启用C风格除法（整数除法结果为浮点数）
    "cdivision": True,
    # 禁用None检查
    "nonecheck": False,
    # 禁用溢出检查
    "overflowcheck": False,
    # 嵌入函数签名到docstring中
    "embedsignature": True,
    # 禁用性能分析
    "profile": False,
    # 禁用行跟踪
    "linetrace": False,
}

setup(
    name="EngineMath",
    version="1.0",
    description="Optimized math library for game engine",
    # 编译Cython扩展模块
    ext_modules=cythonize(
        extensions,
        compiler_directives=compiler_directives
    ),
    # 禁用zip安全，确保扩展模块能正常加载
    zip_safe=False,
)
