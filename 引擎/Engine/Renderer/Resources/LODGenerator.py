import numpy as np
import time

class LODGenerator:
    """
    LOD（细节层次）生成器
    针对低端GPU（如NVIDIA GTX 750Ti和AMD RX 580）优化的网格简化系统
    实现自动LOD生成、边坍缩算法和几何误差控制
    """
    
    def __init__(self):
        # 默认LOD级别设置 - 更细粒度的LOD级别，适合低端GPU
        self.default_lod_levels = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]  # 更细粒度的分辨率级别
        
        # 优化设置 - 针对低端GPU调整
        self.optimization_settings = {
            "preserve_silhouettes": True,    # 保留模型轮廓
            "preserve_boundaries": True,    # 保留边界
            "preserve_texture_seams": True, # 保留纹理接缝
            "aggressive_normals": False,    # 法线优化强度
            "error_threshold": 0.01,        # 针对低端GPU放宽几何误差阈值
            "enable_backface_culling": True, # 启用背面剔除
            "enable_vertex_cache_optimization": True, # 启用顶点缓存优化
            "disable_unused_attributes": True # 禁用未使用的顶点属性
        }
        
        # 性能预算（每个LOD级别的最大三角形数量）
        # 针对低端GPU，增加了更激进的简化比例
        self.performance_budgets = {
            "high": [100, 50, 25, 10],     # 高端设置（每1000三角形的百分比）
            "medium": [80, 40, 20, 8],     # 中端设置
            "low": [50, 20, 10, 3],        # 低端设置 - 更激进的简化
            "ultra_low": [40, 15, 5, 2]     # 超低端设置 - 针对2GB VRAM GPU
        }
        
        # 材质复杂度影响因子
        self.material_complexity_factor = 1.5
    
    def generate_lods(self, mesh_data, level_count=4, quality_preset="medium", material_complexity=1.0):
        """
        为网格生成LOD级别
        
        Args:
            mesh_data: 原始网格数据
            level_count: LOD级别数量（不包括原始网格）
            quality_preset: 质量预设（high/medium/low）
            material_complexity: 材质复杂度因子（1.0-3.0）
            
        Returns:
            dict: 包含多个LOD级别的网格数据
        """
        start_time = time.time()
        
        # 验证质量预设
        if quality_preset not in self.performance_budgets:
            quality_preset = "medium"
            print(f"无效的质量预设，使用默认值: {quality_preset}")
        
        # 计算每个LOD级别的目标三角形百分比
        budgets = self.performance_budgets[quality_preset]
        
        # 调整LOD级别数量
        if level_count > len(budgets):
            level_count = len(budgets)
            print(f"LOD级别数量调整为: {level_count}")
        
        # 计算原始三角形数量
        original_triangle_count = self._count_triangles(mesh_data)
        if original_triangle_count == 0:
            print("警告: 原始网格没有三角形，无法生成LOD")
            return {"lod_0": mesh_data}
        
        print(f"为网格生成LOD，原始三角形数量: {original_triangle_count}")
        
        # 初始化结果字典，包含原始网格（LOD 0）
        lods = {"lod_0": mesh_data}
        
        # 当前处理的网格数据
        current_mesh = mesh_data.copy()
        
        # 生成每个LOD级别
        for i in range(1, level_count + 1):
            # 获取当前LOD级别的预算
            budget_percentage = budgets[i - 1] / 100.0
            
            # 根据材质复杂度调整预算
            adjusted_budget = budget_percentage / (material_complexity * self.material_complexity_factor)
            
            # 计算目标三角形数量
            target_triangles = max(int(original_triangle_count * adjusted_budget), 4)  # 确保至少有4个三角形
            
            print(f"生成LOD {i}: 目标三角形数量 = {target_triangles}")
            
            # 执行网格简化
            simplified_mesh = self._simplify_mesh(
                current_mesh,
                target_triangles,
                self.optimization_settings
            )
            
            # 存储简化后的网格
            lods[f"lod_{i}"] = simplified_mesh
            
            # 更新当前网格
            current_mesh = simplified_mesh.copy()
        
        # 计算生成时间
        end_time = time.time()
        generation_time = end_time - start_time
        
        print(f"LOD生成完成，耗时: {generation_time:.2f}秒")
        
        # 添加LOD元数据
        lods["metadata"] = {
            "original_triangles": original_triangle_count,
            "lod_levels": level_count,
            "quality_preset": quality_preset,
            "generation_time": generation_time,
            "optimization_settings": self.optimization_settings.copy()
        }
        
        return lods
    
    def set_lod_distance_thresholds(self, distances):
        """
        设置LOD切换距离阈值
        
        Args:
            distances: 距离阈值列表 [lod1_distance, lod2_distance, ...]
        """
        if all(isinstance(d, (int, float)) and d > 0 for d in distances):
            self.default_lod_levels = distances
            print(f"LOD切换距离设置为: {distances}")
        else:
            print("警告: 无效的距离值，必须全部为正数")
    
    def get_optimized_lod_settings(self, triangle_count, view_distance, hardware_profile="low"):
        """
        根据三角形数量、视距和硬件配置获取优化的LOD设置
        
        Args:
            triangle_count: 网格的三角形数量
            view_distance: 最大视距
            hardware_profile: 硬件配置文件
            
        Returns:
            dict: 优化的LOD设置
        """
        # 根据三角形数量和硬件配置确定最佳LOD级别数
        if triangle_count < 1000:
            level_count = 2  # 简单模型只需要少量LOD级别
        elif triangle_count < 5000:
            level_count = 3  # 中等复杂度模型
        else:
            level_count = 4  # 复杂模型需要更多LOD级别
        
        # 根据硬件配置调整
        if hardware_profile == "high":
            level_count -= 1  # 高端硬件可以处理更少的LOD级别
        elif hardware_profile == "low":
            pass  # 低端硬件保持默认级别数
        
        # 计算距离阈值
        max_distance = view_distance
        distance_thresholds = []
        for i in range(1, level_count + 1):
            # 指数分布的距离阈值
            distance = max_distance * (i / (level_count + 1)) ** 0.6
            distance_thresholds.append(distance)
        
        return {
            "level_count": level_count,
            "distance_thresholds": distance_thresholds,
            "triangle_reduction": [0.5, 0.25, 0.1][:level_count]  # 三角形减少比例
        }
    
    def _simplify_mesh(self, mesh_data, target_triangles, optimization_settings):
        """
        使用边坍缩算法简化网格
        
        Args:
            mesh_data: 原始网格数据
            target_triangles: 目标三角形数量
            optimization_settings: 优化设置
            
        Returns:
            object: 简化后的网格数据
        """
        # 简化实现
        # 实际实现会使用如边坍缩算法（Edge Collapse）、二次误差度量（QEM）等
        
        # 获取当前三角形数量
        current_triangles = self._count_triangles(mesh_data)
        
        # 计算需要移除的三角形比例
        if current_triangles <= target_triangles:
            # 已经满足目标，无需简化
            return mesh_data
        
        # 计算简化比例
        reduction_ratio = target_triangles / current_triangles
        
        print(f"应用网格简化，目标比例: {reduction_ratio:.2f}")
        
        # 模拟简化过程
        # 在实际实现中，这里会执行实际的网格简化算法
        simplified_mesh = mesh_data.copy()
        simplified_mesh["simplified"] = True
        simplified_mesh["reduction_ratio"] = reduction_ratio
        simplified_mesh["optimization_settings"] = optimization_settings.copy()
        
        # 模拟顶点和索引简化
        # 这里仅作为示例，实际实现会修改顶点和索引数组
        
        return simplified_mesh
    
    def _count_triangles(self, mesh_data):
        """
        计算网格中的三角形数量
        
        Args:
            mesh_data: 网格数据
            
        Returns:
            int: 三角形数量
        """
        # 简化实现
        # 实际实现会检查索引数组并计算三角形数量
        indices = mesh_data.get("indices", [])
        if indices:
            return len(indices) // 3
        else:
            # 如果没有索引，尝试从顶点数量估计
            vertices = mesh_data.get("vertices", [])
            return len(vertices) // 3  # 粗略估计
    
    def _calculate_edge_cost(self, edge, mesh_data):
        """
        计算边坍缩的成本
        使用二次误差度量（QEM）计算坍缩边的几何误差
        
        Args:
            edge: 边对象（两个顶点的索引）
            mesh_data: 网格数据
            
        Returns:
            float: 边坍缩的成本
        """
        # 简化实现
        # 实际实现会使用QEM或其他误差度量方法计算边坍缩成本
        return 1.0
    
    def _preserve_feature_edges(self, mesh_data):
        """
        识别并保护特征边（如轮廓边、边界边和高曲率边）
        
        Args:
            mesh_data: 网格数据
            
        Returns:
            list: 特征边列表
        """
        # 简化实现
        # 实际实现会计算法线变化、纹理坐标变化等来识别特征边
        return []
    
    def _calculate_normal_variation(self, v1_index, v2_index, mesh_data):
        """
        计算两个顶点之间的法线变化
        用于识别高曲率区域
        
        Args:
            v1_index: 第一个顶点索引
            v2_index: 第二个顶点索引
            mesh_data: 网格数据
            
        Returns:
            float: 法线变化角度（弧度）
        """
        # 简化实现
        # 实际实现会计算两个顶点之间的法线差异
        return 0.0
    
    def _optimize_uvs_after_simplification(self, mesh_data):
        """
        在简化后优化UV坐标
        以避免纹理拉伸和扭曲
        
        Args:
            mesh_data: 简化后的网格数据
            
        Returns:
            object: UV优化后的网格数据
        """
        # 简化实现
        # 实际实现会重新计算或调整UV坐标以适应简化后的网格
        return mesh_data
    
    def _calculate_mesh_complexity(self, mesh_data):
        """
        计算网格复杂度指标
        
        Args:
            mesh_data: 网格数据
            
        Returns:
            dict: 复杂度指标
        """
        # 计算基本指标
        triangle_count = self._count_triangles(mesh_data)
        vertex_count = len(mesh_data.get("vertices", []))
        
        # 计算三角形与顶点的比例
        if vertex_count > 0:
            triangles_per_vertex = triangle_count / vertex_count
        else:
            triangles_per_vertex = 0
        
        # 估计内存使用量
        memory_usage = vertex_count * 32 * 4  # 估计每个顶点的数据大小（字节）
        
        return {
            "triangle_count": triangle_count,
            "vertex_count": vertex_count,
            "triangles_per_vertex": triangles_per_vertex,
            "estimated_memory_usage_bytes": memory_usage
        }
    
    def compare_lod_quality(self, original_mesh, simplified_mesh, sample_points=1000):
        """
        比较原始网格和简化网格的质量差异
        
        Args:
            original_mesh: 原始网格
            simplified_mesh: 简化后的网格
            sample_points: 采样点数量
            
        Returns:
            dict: 质量比较结果
        """
        # 简化实现
        # 实际实现会在网格表面采样点，计算几何误差、法线误差等
        
        # 计算基本统计数据
        original_complexity = self._calculate_mesh_complexity(original_mesh)
        simplified_complexity = self._calculate_mesh_complexity(simplified_mesh)
        
        # 计算简化率
        reduction_ratio = simplified_complexity["triangle_count"] / original_complexity["triangle_count"] if original_complexity["triangle_count"] > 0 else 0
        
        # 估计误差（这里是模拟值，实际实现会有真实计算）
        estimated_geometric_error = 0.01 * (1.0 - reduction_ratio)  # 简化率越低，误差越大
        
        return {
            "original_triangles": original_complexity["triangle_count"],
            "simplified_triangles": simplified_complexity["triangle_count"],
            "reduction_ratio": reduction_ratio,
            "estimated_geometric_error": estimated_geometric_error,
            "estimated_memory_savings": original_complexity["estimated_memory_usage_bytes"] - simplified_complexity["estimated_memory_usage_bytes"]
        }