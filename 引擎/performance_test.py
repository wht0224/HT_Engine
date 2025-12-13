#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
性能测试脚本，用于测试引擎的向量、四元数和矩阵运算性能
"""

import time
import numpy as np
import sys
import os

# 导入引擎的数学库
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Engine.Math import Vector3, Quaternion, Matrix4x4

def test_vector_performance():
    """测试向量运算性能"""
    print("=== 向量运算性能测试 ===")
    
    # 创建测试数据
    num_operations = 1000000
    vectors1 = [Vector3(np.random.randn(), np.random.randn(), np.random.randn()) for _ in range(num_operations)]
    vectors2 = [Vector3(np.random.randn(), np.random.randn(), np.random.randn()) for _ in range(num_operations)]
    
    # 测试向量加法
    start_time = time.time()
    for v1, v2 in zip(vectors1, vectors2):
        result = v1 + v2
    end_time = time.time()
    print(f"向量加法 ({num_operations}次): {end_time - start_time:.3f}秒, {num_operations / (end_time - start_time):.0f}次/秒")
    
    # 测试向量点积
    start_time = time.time()
    for v1, v2 in zip(vectors1, vectors2):
        result = v1.dot(v2)
    end_time = time.time()
    print(f"向量点积 ({num_operations}次): {end_time - start_time:.3f}秒, {num_operations / (end_time - start_time):.0f}次/秒")
    
    # 测试向量归一化
    start_time = time.time()
    for v in vectors1:
        result = v.normalize()
    end_time = time.time()
    print(f"向量归一化 ({num_operations}次): {end_time - start_time:.3f}秒, {num_operations / (end_time - start_time):.0f}次/秒")
    
    # 测试向量叉积
    start_time = time.time()
    for v1, v2 in zip(vectors1, vectors2):
        result = v1.cross(v2)
    end_time = time.time()
    print(f"向量叉积 ({num_operations}次): {end_time - start_time:.3f}秒, {num_operations / (end_time - start_time):.0f}次/秒")

def test_quaternion_performance():
    """测试四元数运算性能"""
    print("\n=== 四元数运算性能测试 ===")
    
    # 创建测试数据
    num_operations = 1000000
    quats1 = [Quaternion(np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn()) for _ in range(num_operations)]
    quats2 = [Quaternion(np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn()) for _ in range(num_operations)]
    vectors = [Vector3(np.random.randn(), np.random.randn(), np.random.randn()) for _ in range(num_operations)]
    
    # 测试四元数乘法
    start_time = time.time()
    for q1, q2 in zip(quats1, quats2):
        result = q1 * q2
    end_time = time.time()
    print(f"四元数乘法 ({num_operations}次): {end_time - start_time:.3f}秒, {num_operations / (end_time - start_time):.0f}次/秒")
    
    # 测试四元数旋转向量
    start_time = time.time()
    for q, v in zip(quats1, vectors):
        # Cython版本使用 q * v，Python版本使用 q.rotate_vector(v)
        try:
            result = q * v  # Cython API
        except:
            result = q.rotate_vector(v)  # Python API
    end_time = time.time()
    print(f"四元数旋转向量 ({num_operations}次): {end_time - start_time:.3f}秒, {num_operations / (end_time - start_time):.0f}次/秒")
    
    # 测试四元数归一化
    start_time = time.time()
    for q in quats1:
        result = q.normalize()
    end_time = time.time()
    print(f"四元数归一化 ({num_operations}次): {end_time - start_time:.3f}秒, {num_operations / (end_time - start_time):.0f}次/秒")

def test_matrix_performance():
    """测试矩阵运算性能"""
    print("\n=== 矩阵运算性能测试 ===")
    
    # 创建测试数据
    num_operations = 100000
    matrices1 = [Matrix4x4() for _ in range(num_operations)]
    matrices2 = [Matrix4x4() for _ in range(num_operations)]
    vectors = [Vector3(np.random.randn(), np.random.randn(), np.random.randn()) for _ in range(num_operations)]
    
    # 初始化矩阵为随机值
    for i in range(num_operations):
        matrices1[i] = Matrix4x4.from_scale(Vector3(np.random.randn(), np.random.randn(), np.random.randn()))
        matrices2[i] = Matrix4x4.from_scale(Vector3(np.random.randn(), np.random.randn(), np.random.randn()))
    
    # 测试矩阵乘法
    start_time = time.time()
    for m1, m2 in zip(matrices1, matrices2):
        result = m1 * m2
    end_time = time.time()
    print(f"矩阵乘法 ({num_operations}次): {end_time - start_time:.3f}秒, {num_operations / (end_time - start_time):.0f}次/秒")
    
    # 测试矩阵转置（仅Python版本支持）
    if hasattr(matrices1[0], 'transpose'):
        start_time = time.time()
        for m in matrices1:
            result = m.transpose()
        end_time = time.time()
        print(f"矩阵转置 ({num_operations}次): {end_time - start_time:.3f}秒, {num_operations / (end_time - start_time):.0f}次/秒")
    else:
        print(f"矩阵转置: Cython版本暂不支持（待实现）")

    # 测试矩阵求逆（仅Python版本支持）
    if hasattr(matrices1[0], 'inverse'):
        start_time = time.time()
        for m in matrices1:
            result = m.inverse()
        end_time = time.time()
        print(f"矩阵求逆 ({num_operations}次): {end_time - start_time:.3f}秒, {num_operations / (end_time - start_time):.0f}次/秒")
    else:
        print(f"矩阵求逆: Cython版本暂不支持（待实现）")

if __name__ == "__main__":
    import sys
    import os
    
    # 保存测试结果
    results = []
    
    # 重定向print函数，同时输出到控制台和列表
    original_print = print
    def capture_print(*args, **kwargs):
        original_print(*args, **kwargs)
        line = " ".join(str(arg) for arg in args)
        results.append(line)
    
    print = capture_print
    
    print("开始性能测试...")
    print(f"Python版本: {sys.version}")
    print(f"当前目录: {os.getcwd()}")
    print()
    
    # 运行所有性能测试
    test_vector_performance()
    test_quaternion_performance()
    test_matrix_performance()
    
    print("\n性能测试完成！")
    
    # 恢复原始print函数
    print = original_print
    
    # 保存结果到文件
    with open("performance_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(results))
    
    print("\n测试结果已保存到 performance_results.txt 文件")
    print("\n=== 测试结果摘要 ===")
    for line in results:
        if any(keyword in line for keyword in ["秒", "次/秒"]):
            print(line)