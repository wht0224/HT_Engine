# -*- coding: utf-8 -*-
"""
Cython数学库编译配置文件
用于将CythonMath.pyx编译为C扩展模块
"""

import os
import sys
from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 定义Cython扩展
cython_extensions = [
    Extension(
        name="CythonMath",  # 生成的扩展模块名称
        sources=[
            "CythonMath.pyx",  # Cython源文件
            "asm_math.c",  # 汇编优化的C包装文件
        ],
        include_dirs=[current_dir],  # 包含目录
        # 编译优化选项
        extra_compile_args=[
            "/O2",  # 最高优化级别
            "/fp:fast",  # 快速浮点运算
            "/arch:AVX2",  # 使用AVX2指令集
            "/DNDEBUG",  # 禁用调试信息
            "/MT",  # 静态链接MSVC运行时
        ],
        # 链接优化选项
        extra_link_args=[
            "/OPT:REF",  # 移除未引用的函数和数据
            "/OPT:ICF",  # 合并相同的函数和数据
        ],
        # 仅在Windows上使用的选项
        define_macros=[
            ("MS_WIN64", "1"),  # 定义64位Windows宏
        ]
    )
]

# 编译配置
setup(
    name="CythonMath",
    version="1.0",
    description="高性能数学库，使用Cython实现",
    ext_modules=cythonize(
        cython_extensions,
        compiler_directives={
            "language_level": "3",  # 使用Python 3语法
            "boundscheck": False,  # 禁用边界检查
            "wraparound": False,  # 禁用负索引环绕
            "nonecheck": False,  # 禁用None检查
            "cdivision": True,  # 启用C风格除法
            "profile": False,  # 禁用性能分析
            "optimize.use_switch": True,  # 使用switch优化
            "optimize.unpack_method_calls": True,  # 优化方法调用
            "initializedcheck": False,  # 禁用初始化检查
            "overflowcheck": False,  # 禁用溢出检查
            "always_allow_keywords": False,  # 禁用关键字参数
        },
        annotate=False,  # 不生成注释HTML文件
        quiet=False,  # 显示编译输出
    ),
    zip_safe=False,  # 禁用zip安全，提高加载性能
)

# 如果直接运行该脚本，执行编译
if __name__ == "__main__":
    # 编译Cython扩展
    setup(
        name="CythonMath",
        version="1.0",
        description="高性能数学库，使用Cython实现",
        ext_modules=cythonize(
            cython_extensions,
            compiler_directives={
                "language_level": "3",
                "boundscheck": False,
                "wraparound": False,
                "nonecheck": False,
                "cdivision": True,
                "profile": False,
                "optimize.use_switch": True,
                "optimize.unpack_method_calls": True,
                "initializedcheck": False,
                "overflowcheck": False,
                "always_allow_keywords": False,
            },
            annotate=False,
            quiet=False,
        ),
        zip_safe=False,
    )
