from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os
import sys

# 引擎数学库的构建配置文件
# Build configuration file for engine math library

# 检查是否在Windows平台上
is_windows = sys.platform == 'win32'

# 定义当前工作目录
cwd = os.getcwd()

# 定义Cython扩展模块列表
# Define list of Cython extension modules
sources = ["Engine/Math/CythonMath.pyx", "Engine/Math/asm_math.c"]

# 编译选项
# Compiler options
compiler_args = []
linker_args = []

if is_windows:
    # Windows平台下的编译选项
    compiler_args.extend([
        "/O2",  # 最高优化级别
        "/arch:AVX2",  # 启用AVX2指令集
        "/fp:fast",  # 快速浮点运算
        "/MT",  # 静态链接MSVCRT
    ])
    
    # 对于Windows，我们需要手动编译汇编文件
    # 使用MASM编译asm文件
    import subprocess
    import tempfile
    
    # 汇编文件路径
    asm_file = "Engine/Math/asm_math.asm"
    
    # 编译汇编文件
    obj_file = asm_file.replace(".asm", ".obj")
    
    try:
        # 尝试使用MASM编译汇编文件
        # 先尝试直接使用ml64编译
        subprocess.run([
            "ml64", "/c", "/Fo", obj_file, asm_file
        ], check=True, cwd=cwd)
        
        # 将生成的.obj文件添加到扩展模块
        linker_args.append(obj_file)
        print(f"成功编译汇编文件: {asm_file} -> {obj_file}")
    except subprocess.CalledProcessError:
        # 如果直接编译失败，尝试使用vcvarsall.bat设置环境变量
        print("直接编译失败，尝试使用vcvarsall.bat设置环境变量...")
        try:
            subprocess.run([
                "cmd", "/c", 
                '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && ml64 /c /Fo "' + obj_file + '" "' + asm_file + '"'
            ], check=True, cwd=cwd, shell=True)
            
            # 将生成的.obj文件添加到扩展模块
            linker_args.append(obj_file)
            print(f"成功编译汇编文件: {asm_file} -> {obj_file}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"警告: 无法编译汇编文件 {asm_file}: {e}")
            print("将使用C实现的数学函数")
    except FileNotFoundError as e:
        print(f"警告: 无法找到汇编编译器: {e}")
        print("将使用C实现的数学函数")
else:
    # Linux/macOS平台下的编译选项
    compiler_args.extend([
        "-O3",  # 最高优化级别
        "-mavx2",  # 启用AVX2指令集
        "-mfma",  # 启用FMA指令集
        "-ffast-math",  # 快速浮点运算
    ])

extensions = [
    Extension(
        "Engine.Math.CythonMath",  # 模块完全限定名称
        sources,  # 源文件相对路径，包含Cython文件
        include_dirs=[np.get_include(), "Engine/Math"],  # 包含numpy头文件目录和本地头文件目录
        language="c",  # 使用C语言编译
        extra_compile_args=compiler_args,  # 额外的编译选项
        extra_link_args=linker_args,  # 额外的链接选项，包含编译好的.obj文件
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
