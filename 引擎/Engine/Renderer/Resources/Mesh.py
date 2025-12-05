# -*- coding: utf-8 -*-
"""
网格资源类，用于管理3D模型的顶点、索引和其他几何数据
"""

from Engine.Math.Math import Vector3, Vector2, BoundingBox
import math

# 为了方便使用，将常用的数学函数导入到当前命名空间
sin = math.sin
cos = math.cos

class Mesh:
    """网格资源类，用于管理3D模型的顶点、索引和其他几何数据"""
    
    def __init__(self):
        """初始化网格"""
        self.vertices = []  # 顶点列表
        self.normals = []   # 法线列表
        self.uvs = []       # UV坐标列表
        self.tangents = []  # 切线列表
        self.indices = []   # 索引列表
        self.bounding_box = None  # 包围盒
        self.vertex_format_hash = 0  # 顶点格式哈希值
        
        # LOD设置
        self.lod_levels = []  # LOD级别列表
        self.lod_distances = []  # LOD切换距离
        
        # 实例化数据
        self.instance_count = 1  # 实例数量
        self.instance_data = []  # 实例化数据
        
        # 渲染状态
        self.is_dirty = True  # 数据是否需要更新到GPU
        self.vertex_buffer = None  # 顶点缓冲区
        self.index_buffer = None   # 索引缓冲区
        self.instance_buffer = None  # 实例化缓冲区
    
    @staticmethod
    def create_cube(size=1.0):
        """创建立方体网格
        
        Args:
            size: 立方体大小
            
        Returns:
            Mesh: 立方体网格
        """
        mesh = Mesh()
        
        half_size = size / 2.0
        
        # 立方体顶点数据
        vertices = [
            # 前面
            Vector3(-half_size, -half_size, half_size),
            Vector3(half_size, -half_size, half_size),
            Vector3(half_size, half_size, half_size),
            Vector3(-half_size, half_size, half_size),
            # 后面
            Vector3(-half_size, -half_size, -half_size),
            Vector3(half_size, -half_size, -half_size),
            Vector3(half_size, half_size, -half_size),
            Vector3(-half_size, half_size, -half_size),
            # 左面
            Vector3(-half_size, -half_size, -half_size),
            Vector3(-half_size, -half_size, half_size),
            Vector3(-half_size, half_size, half_size),
            Vector3(-half_size, half_size, -half_size),
            # 右面
            Vector3(half_size, -half_size, half_size),
            Vector3(half_size, -half_size, -half_size),
            Vector3(half_size, half_size, -half_size),
            Vector3(half_size, half_size, half_size),
            # 上面
            Vector3(-half_size, half_size, half_size),
            Vector3(half_size, half_size, half_size),
            Vector3(half_size, half_size, -half_size),
            Vector3(-half_size, half_size, -half_size),
            # 下面
            Vector3(-half_size, -half_size, -half_size),
            Vector3(half_size, -half_size, -half_size),
            Vector3(half_size, -half_size, half_size),
            Vector3(-half_size, -half_size, half_size),
        ]
        
        # 立方体法线数据
        normals = [
            # 前面
            Vector3(0, 0, 1),
            Vector3(0, 0, 1),
            Vector3(0, 0, 1),
            Vector3(0, 0, 1),
            # 后面
            Vector3(0, 0, -1),
            Vector3(0, 0, -1),
            Vector3(0, 0, -1),
            Vector3(0, 0, -1),
            # 左面
            Vector3(-1, 0, 0),
            Vector3(-1, 0, 0),
            Vector3(-1, 0, 0),
            Vector3(-1, 0, 0),
            # 右面
            Vector3(1, 0, 0),
            Vector3(1, 0, 0),
            Vector3(1, 0, 0),
            Vector3(1, 0, 0),
            # 上面
            Vector3(0, 1, 0),
            Vector3(0, 1, 0),
            Vector3(0, 1, 0),
            Vector3(0, 1, 0),
            # 下面
            Vector3(0, -1, 0),
            Vector3(0, -1, 0),
            Vector3(0, -1, 0),
            Vector3(0, -1, 0),
        ]
        
        # 立方体UV数据
        uvs = [
            # 前面
            Vector2(0, 0),
            Vector2(1, 0),
            Vector2(1, 1),
            Vector2(0, 1),
            # 后面
            Vector2(0, 0),
            Vector2(1, 0),
            Vector2(1, 1),
            Vector2(0, 1),
            # 左面
            Vector2(0, 0),
            Vector2(1, 0),
            Vector2(1, 1),
            Vector2(0, 1),
            # 右面
            Vector2(0, 0),
            Vector2(1, 0),
            Vector2(1, 1),
            Vector2(0, 1),
            # 上面
            Vector2(0, 0),
            Vector2(1, 0),
            Vector2(1, 1),
            Vector2(0, 1),
            # 下面
            Vector2(0, 0),
            Vector2(1, 0),
            Vector2(1, 1),
            Vector2(0, 1),
        ]
        
        # 立方体索引数据
        indices = [
            # 前面
            0, 1, 2,
            0, 2, 3,
            # 后面
            4, 5, 6,
            4, 6, 7,
            # 左面
            8, 9, 10,
            8, 10, 11,
            # 右面
            12, 13, 14,
            12, 14, 15,
            # 上面
            16, 17, 18,
            16, 18, 19,
            # 下面
            20, 21, 22,
            20, 22, 23,
        ]
        
        mesh.vertices = vertices
        mesh.normals = normals
        mesh.uvs = uvs
        mesh.indices = indices
        
        # 计算包围盒
        mesh._calculate_bounding_box()
        
        return mesh
    
    @staticmethod
    def create_sphere(radius=1.0, segments=32, rings=16):
        """创建球体网格
        
        Args:
            radius: 球体半径
            segments: 经度方向的分段数
            rings: 纬度方向的分段数
            
        Returns:
            Mesh: 球体网格
        """
        mesh = Mesh()
        
        vertices = []
        normals = []
        uvs = []
        indices = []
        
        # 顶部极点
        top_vertex = Vector3(0, radius, 0)
        vertices.append(top_vertex)
        normals.append(Vector3(0, 1, 0))
        uvs.append(Vector2(0.5, 0.0))
        
        # 中间环
        for ring in range(1, rings):
            theta = ring * (3.141592653589793 / rings)
            sin_theta = sin(theta)
            cos_theta = cos(theta)
            
            for segment in range(segments):
                phi = segment * (2.0 * 3.141592653589793 / segments)
                sin_phi = sin(phi)
                cos_phi = cos(phi)
                
                x = cos_phi * sin_theta
                y = cos_theta
                z = sin_phi * sin_theta
                
                vertex = Vector3(x * radius, y * radius, z * radius)
                vertices.append(vertex)
                normals.append(Vector3(x, y, z))
                uvs.append(Vector2(segment / segments, ring / rings))
        
        # 底部极点
        bottom_vertex = Vector3(0, -radius, 0)
        vertices.append(bottom_vertex)
        normals.append(Vector3(0, -1, 0))
        uvs.append(Vector2(0.5, 1.0))
        
        # 顶部三角形扇
        for segment in range(segments):
            next_segment = (segment + 1) % segments
            indices.append(0)
            indices.append(segment + 1)
            indices.append(next_segment + 1)
        
        # 中间四边形带
        for ring in range(1, rings - 1):
            ring_start = ring * segments + 1
            next_ring_start = (ring + 1) * segments + 1
            
            for segment in range(segments):
                next_segment = (segment + 1) % segments
                
                # 第一个三角形
                indices.append(ring_start + segment)
                indices.append(ring_start + next_segment)
                indices.append(next_ring_start + segment)
                
                # 第二个三角形
                indices.append(ring_start + next_segment)
                indices.append(next_ring_start + next_segment)
                indices.append(next_ring_start + segment)
        
        # 底部三角形扇
        bottom_index = len(vertices) - 1
        for segment in range(segments):
            next_segment = (segment + 1) % segments
            indices.append(bottom_index)
            indices.append(bottom_index - segments + next_segment)
            indices.append(bottom_index - segments + segment)
        
        mesh.vertices = vertices
        mesh.normals = normals
        mesh.uvs = uvs
        mesh.indices = indices
        
        # 计算包围盒
        mesh._calculate_bounding_box()
        
        return mesh
    
    @staticmethod
    def create_cylinder(radius=1.0, height=2.0, segments=32):
        """创建圆柱体网格
        
        Args:
            radius: 圆柱体半径
            height: 圆柱体高度
            segments: 圆周方向的分段数
            
        Returns:
            Mesh: 圆柱体网格
        """
        mesh = Mesh()
        
        vertices = []
        normals = []
        uvs = []
        indices = []
        
        half_height = height / 2.0
        
        # 顶部和底部圆心
        top_center = Vector3(0, half_height, 0)
        bottom_center = Vector3(0, -half_height, 0)
        
        # 生成侧面顶点
        for segment in range(segments):
            angle = segment * (2.0 * 3.141592653589793 / segments)
            cos_angle = cos(angle)
            sin_angle = sin(angle)
            
            # 顶部顶点
            top_vertex = Vector3(cos_angle * radius, half_height, sin_angle * radius)
            vertices.append(top_vertex)
            normals.append(Vector3(cos_angle, 0, sin_angle))
            uvs.append(Vector2(segment / segments, 1.0))
            
            # 底部顶点
            bottom_vertex = Vector3(cos_angle * radius, -half_height, sin_angle * radius)
            vertices.append(bottom_vertex)
            normals.append(Vector3(cos_angle, 0, sin_angle))
            uvs.append(Vector2(segment / segments, 0.0))
        
        # 生成侧面索引
        for segment in range(segments):
            next_segment = (segment + 1) % segments
            
            # 第一个三角形
            indices.append(segment * 2)
            indices.append(next_segment * 2)
            indices.append(segment * 2 + 1)
            
            # 第二个三角形
            indices.append(next_segment * 2)
            indices.append(next_segment * 2 + 1)
            indices.append(segment * 2 + 1)
        
        # 生成顶部和底部圆面
        # 这里简化处理，只添加顶点，不添加索引（实际应用中需要添加圆面的索引）
        vertices.append(top_center)
        normals.append(Vector3(0, 1, 0))
        uvs.append(Vector2(0.5, 0.5))
        
        vertices.append(bottom_center)
        normals.append(Vector3(0, -1, 0))
        uvs.append(Vector2(0.5, 0.5))
        
        mesh.vertices = vertices
        mesh.normals = normals
        mesh.uvs = uvs
        mesh.indices = indices
        
        # 计算包围盒
        mesh._calculate_bounding_box()
        
        return mesh
    
    @staticmethod
    def create_plane(width=1.0, height=1.0, width_segments=1, height_segments=1):
        """创建平面网格
        
        Args:
            width: 平面宽度
            height: 平面高度
            width_segments: 宽度方向的分段数
            height_segments: 高度方向的分段数
            
        Returns:
            Mesh: 平面网格
        """
        mesh = Mesh()
        
        vertices = []
        normals = []
        uvs = []
        indices = []
        
        half_width = width / 2.0
        half_height = height / 2.0
        
        # 生成顶点
        for y in range(height_segments + 1):
            for x in range(width_segments + 1):
                # 计算顶点位置
                px = (x / width_segments) * width - half_width
                py = 0.0
                pz = (y / height_segments) * height - half_height
                
                vertex = Vector3(px, py, pz)
                vertices.append(vertex)
                
                # 平面的法线始终向上
                normals.append(Vector3(0, 1, 0))
                
                # 计算UV坐标
                u = x / width_segments
                v = y / height_segments
                uvs.append(Vector2(u, v))
        
        # 生成索引
        for y in range(height_segments):
            for x in range(width_segments):
                # 当前顶点索引
                current = y * (width_segments + 1) + x
                
                # 下一行的顶点索引
                next_row = (y + 1) * (width_segments + 1) + x
                
                # 第一个三角形
                indices.append(current)
                indices.append(next_row)
                indices.append(current + 1)
                
                # 第二个三角形
                indices.append(next_row)
                indices.append(next_row + 1)
                indices.append(current + 1)
        
        mesh.vertices = vertices
        mesh.normals = normals
        mesh.uvs = uvs
        mesh.indices = indices
        
        # 计算包围盒
        mesh._calculate_bounding_box()
        
        return mesh
    
    def _calculate_bounding_box(self):
        """计算网格的包围盒"""
        if not self.vertices:
            self.bounding_box = BoundingBox(Vector3(-0.5, -0.5, -0.5), Vector3(0.5, 0.5, 0.5))
            return
        
        min_x = min(vertex.x for vertex in self.vertices)
        min_y = min(vertex.y for vertex in self.vertices)
        min_z = min(vertex.z for vertex in self.vertices)
        
        max_x = max(vertex.x for vertex in self.vertices)
        max_y = max(vertex.y for vertex in self.vertices)
        max_z = max(vertex.z for vertex in self.vertices)
        
        min_point = Vector3(min_x, min_y, min_z)
        max_point = Vector3(max_x, max_y, max_z)
        
        self.bounding_box = BoundingBox(min_point, max_point)
    
    def get_vertex_format_hash(self):
        """获取顶点格式哈希值，用于批处理分组
        
        Returns:
            int: 顶点格式哈希值
        """
        # 简化实现，实际应用中需要根据顶点属性计算哈希值
        return hash((len(self.vertices), len(self.normals), len(self.uvs)))
    
    def update(self):
        """更新网格数据，将数据上传到GPU"""
        # TODO: 实现网格数据更新逻辑
        self.is_dirty = False
    
    def draw(self):
        """绘制网格"""
        # TODO: 实现网格绘制逻辑
        pass
    
    def draw_instanced(self, instance_count):
        """实例化绘制网格
        
        Args:
            instance_count: 实例数量
        """
        # TODO: 实现实例化绘制逻辑
        pass
    
    def destroy(self):
        """销毁网格，释放资源"""
        # TODO: 实现资源释放逻辑
        pass
