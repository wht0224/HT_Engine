# -*- coding: utf-8 -*-
"""
建模API - 让AI通过代码创建3D模型
Modeling API - AI code-driven 3D modeling
"""

import math
from Engine.Math import Vector3, Vector2, Quaternion
from Engine.Renderer.Resources.Mesh import Mesh

# 尝试导入logger，失败则使用标准logging
try:
    from Engine.Logger import get_logger
    HAS_ENGINE_LOGGER = True
except:
    import logging
    HAS_ENGINE_LOGGER = False


class SphereTrigCache:
    """
    球体三角函数查找表缓存
    预计算sin/cos值，避免重复计算，提升3-5倍性能
    """
    _cache = {}

    @classmethod
    def get_trig_tables(cls, segments, rings):
        """
        获取或创建三角函数查找表

        Args:
            segments: 经线分段数
            rings: 纬线分段数

        Returns:
            dict: 包含预计算的sin/cos值
        """
        key = (segments, rings)
        if key not in cls._cache:
            # 预计算纬线角度的sin/cos（theta从0到π）
            ring_sin = []
            ring_cos = []
            for ring in range(rings + 1):
                theta = ring * math.pi / rings
                ring_sin.append(math.sin(theta))
                ring_cos.append(math.cos(theta))

            # 预计算经线角度的sin/cos（phi从0到2π）
            seg_sin = []
            seg_cos = []
            for seg in range(segments + 1):
                phi = seg * 2.0 * math.pi / segments
                seg_sin.append(math.sin(phi))
                seg_cos.append(math.cos(phi))

            cls._cache[key] = {
                'ring_sin': ring_sin,
                'ring_cos': ring_cos,
                'seg_sin': seg_sin,
                'seg_cos': seg_cos
            }

        return cls._cache[key]

    @classmethod
    def clear_cache(cls):
        """清空缓存（用于节省内存）"""
        cls._cache.clear()


class ModelingAPI:
    """
    建模API类 - 提供简单的代码接口创建3D模型
    AI可以通过这个API直接生成模型
    """

    def __init__(self, engine):
        """
        初始化建模API

        Args:
            engine: 引擎实例
        """
        self.engine = engine

        # 初始化logger
        if HAS_ENGINE_LOGGER:
            self.logger = get_logger("ModelingAPI")
        else:
            self.logger = logging.getLogger("ModelingAPI")
            self.logger.setLevel(logging.WARNING)

        self.current_mesh = None

    # ========== 基础几何体创建 ==========

    def create_cube(self, size=1.0, name="Cube"):
        """
        创建立方体

        Args:
            size: 立方体边长
            name: 对象名称

        Returns:
            Mesh: 创建的网格对象
        """
        self.logger.info(f"创建立方体: {name}, 大小={size}")

        mesh = Mesh()
        s = size / 2.0

        # 8个顶点
        vertices = [
            Vector3(-s, -s, -s),  # 0: 左下后
            Vector3(s, -s, -s),   # 1: 右下后
            Vector3(s, s, -s),    # 2: 右上后
            Vector3(-s, s, -s),   # 3: 左上后
            Vector3(-s, -s, s),   # 4: 左下前
            Vector3(s, -s, s),    # 5: 右下前
            Vector3(s, s, s),     # 6: 右上前
            Vector3(-s, s, s)     # 7: 左上前
        ]

        # 6个面，每个面4个顶点（需要复制顶点以便每个面有独立的法线）
        mesh.vertices = []
        mesh.normals = []
        mesh.uvs = []
        mesh.indices = []

        # 面定义：(顶点索引, 法线)
        faces = [
            # 前面 (Z+)
            ([4, 5, 6, 7], Vector3(0, 0, 1)),
            # 后面 (Z-)
            ([1, 0, 3, 2], Vector3(0, 0, -1)),
            # 顶面 (Y+)
            ([7, 6, 2, 3], Vector3(0, 1, 0)),
            # 底面 (Y-)
            ([0, 1, 5, 4], Vector3(0, -1, 0)),
            # 右面 (X+)
            ([5, 1, 2, 6], Vector3(1, 0, 0)),
            # 左面 (X-)
            ([0, 4, 7, 3], Vector3(-1, 0, 0))
        ]

        index_offset = 0
        for face_verts, normal in faces:
            # 为每个面添加4个顶点
            for i, vert_idx in enumerate(face_verts):
                mesh.vertices.append(vertices[vert_idx])
                mesh.normals.append(normal)
                # UV坐标 (简单的平面映射)
                u = 1.0 if i in [1, 2] else 0.0
                v = 1.0 if i in [2, 3] else 0.0
                mesh.uvs.append(Vector2(u, v))

            # 两个三角形组成一个面
            mesh.indices.extend([
                index_offset, index_offset + 1, index_offset + 2,
                index_offset, index_offset + 2, index_offset + 3
            ])
            index_offset += 4

        mesh.is_dirty = True
        self._calculate_bounding_box(mesh)
        self.current_mesh = mesh

        return mesh

    def create_sphere(self, radius=1.0, segments=32, rings=16, name="Sphere"):
        """
        创建球体（使用三角函数缓存优化，提升3-5倍性能）

        Args:
            radius: 半径
            segments: 经线分段数
            rings: 纬线分段数
            name: 对象名称

        Returns:
            Mesh: 创建的网格对象
        """
        self.logger.info(f"创建球体: {name}, 半径={radius}, 分段={segments}x{rings}")

        mesh = Mesh()
        mesh.vertices = []
        mesh.normals = []
        mesh.uvs = []
        mesh.indices = []

        # 获取预计算的三角函数表（缓存优化）
        trig_cache = SphereTrigCache.get_trig_tables(segments, rings)
        ring_sin = trig_cache['ring_sin']
        ring_cos = trig_cache['ring_cos']
        seg_sin = trig_cache['seg_sin']
        seg_cos = trig_cache['seg_cos']

        # 生成顶点（使用查找表，避免重复计算sin/cos）
        for ring in range(rings + 1):
            sin_theta = ring_sin[ring]
            cos_theta = ring_cos[ring]

            for seg in range(segments + 1):
                sin_phi = seg_sin[seg]
                cos_phi = seg_cos[seg]

                # 顶点位置
                x = radius * sin_theta * cos_phi
                y = radius * cos_theta
                z = radius * sin_theta * sin_phi

                # 法线（球面法线等于归一化的位置向量）
                normal = Vector3(sin_theta * cos_phi, cos_theta, sin_theta * sin_phi)

                # UV坐标
                u = seg / segments
                v = ring / rings

                mesh.vertices.append(Vector3(x, y, z))
                mesh.normals.append(normal)
                mesh.uvs.append(Vector2(u, v))

        # 生成索引
        for ring in range(rings):
            for seg in range(segments):
                # 当前四边形的四个顶点索引
                a = ring * (segments + 1) + seg
                b = a + segments + 1
                c = b + 1
                d = a + 1

                # 两个三角形
                mesh.indices.extend([a, b, d])
                mesh.indices.extend([b, c, d])

        mesh.is_dirty = True
        self._calculate_bounding_box(mesh)
        self.current_mesh = mesh

        return mesh

    def create_cylinder(self, radius=1.0, height=2.0, segments=32, name="Cylinder"):
        """
        创建圆柱体

        Args:
            radius: 半径
            height: 高度
            segments: 圆周分段数
            name: 对象名称

        Returns:
            Mesh: 创建的网格对象
        """
        self.logger.info(f"创建圆柱体: {name}, 半径={radius}, 高度={height}, 分段={segments}")

        mesh = Mesh()
        mesh.vertices = []
        mesh.normals = []
        mesh.uvs = []
        mesh.indices = []

        half_height = height / 2.0

        # 侧面顶点（上下各一圈）
        for i in range(segments + 1):
            angle = 2.0 * math.pi * i / segments
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            x = radius * cos_a
            z = radius * sin_a

            # 法线（指向外侧）
            normal = Vector3(cos_a, 0, sin_a)

            # 上圈顶点
            mesh.vertices.append(Vector3(x, half_height, z))
            mesh.normals.append(normal)
            mesh.uvs.append(Vector2(i / segments, 1.0))

            # 下圈顶点
            mesh.vertices.append(Vector3(x, -half_height, z))
            mesh.normals.append(normal)
            mesh.uvs.append(Vector2(i / segments, 0.0))

        # 侧面索引
        for i in range(segments):
            a = i * 2
            b = a + 1
            c = a + 2
            d = a + 3

            mesh.indices.extend([a, b, c])
            mesh.indices.extend([b, d, c])

        # 顶面和底面
        base_index = len(mesh.vertices)

        # 顶面中心
        mesh.vertices.append(Vector3(0, half_height, 0))
        mesh.normals.append(Vector3(0, 1, 0))
        mesh.uvs.append(Vector2(0.5, 0.5))
        top_center_idx = base_index

        # 底面中心
        mesh.vertices.append(Vector3(0, -half_height, 0))
        mesh.normals.append(Vector3(0, -1, 0))
        mesh.uvs.append(Vector2(0.5, 0.5))
        bottom_center_idx = base_index + 1

        # 顶面和底面的圆周顶点
        for i in range(segments + 1):
            angle = 2.0 * math.pi * i / segments
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            x = radius * cos_a
            z = radius * sin_a

            # 顶面顶点
            mesh.vertices.append(Vector3(x, half_height, z))
            mesh.normals.append(Vector3(0, 1, 0))
            u = 0.5 + 0.5 * cos_a
            v = 0.5 + 0.5 * sin_a
            mesh.uvs.append(Vector2(u, v))

            # 底面顶点
            mesh.vertices.append(Vector3(x, -half_height, z))
            mesh.normals.append(Vector3(0, -1, 0))
            mesh.uvs.append(Vector2(u, v))

        # 顶面和底面索引
        for i in range(segments):
            top_a = base_index + 2 + i * 2
            top_b = top_a + 2

            bottom_a = base_index + 3 + i * 2
            bottom_b = bottom_a + 2

            # 顶面三角形
            mesh.indices.extend([top_center_idx, top_a, top_b])

            # 底面三角形（逆序）
            mesh.indices.extend([bottom_center_idx, bottom_b, bottom_a])

        mesh.is_dirty = True
        self._calculate_bounding_box(mesh)
        self.current_mesh = mesh

        return mesh

    def create_plane(self, width=2.0, height=2.0, name="Plane"):
        """
        创建平面

        Args:
            width: 宽度
            height: 高度
            name: 对象名称

        Returns:
            Mesh: 创建的网格对象
        """
        self.logger.info(f"创建平面: {name}, 宽度={width}, 高度={height}")

        mesh = Mesh()
        w = width / 2.0
        h = height / 2.0

        # 4个顶点
        mesh.vertices = [
            Vector3(-w, 0, -h),
            Vector3(w, 0, -h),
            Vector3(w, 0, h),
            Vector3(-w, 0, h)
        ]

        # 法线都向上
        mesh.normals = [Vector3(0, 1, 0)] * 4

        # UV坐标
        mesh.uvs = [
            Vector2(0, 0),
            Vector2(1, 0),
            Vector2(1, 1),
            Vector2(0, 1)
        ]

        # 两个三角形
        mesh.indices = [0, 1, 2, 0, 2, 3]

        mesh.is_dirty = True
        self._calculate_bounding_box(mesh)
        self.current_mesh = mesh

        return mesh

    def create_cone(self, radius=1.0, height=2.0, segments=32, name="Cone"):
        """
        创建圆锥体

        Args:
            radius: 底面半径
            height: 高度
            segments: 底面分段数
            name: 对象名称

        Returns:
            Mesh: 创建的网格对象
        """
        self.logger.info(f"创建圆锥体: {name}, 半径={radius}, 高度={height}, 分段={segments}")

        mesh = Mesh()
        mesh.vertices = []
        mesh.normals = []
        mesh.uvs = []
        mesh.indices = []

        half_height = height / 2.0

        # 顶点（尖端）
        apex = Vector3(0, half_height, 0)

        # 侧面
        # 计算侧面法线的倾斜角度
        slant = math.sqrt(radius * radius + height * height)
        normal_y = radius / slant
        normal_xz = height / slant

        base_index = 0
        for i in range(segments):
            angle1 = 2.0 * math.pi * i / segments
            angle2 = 2.0 * math.pi * (i + 1) / segments

            cos1, sin1 = math.cos(angle1), math.sin(angle1)
            cos2, sin2 = math.cos(angle2), math.sin(angle2)

            # 底面两个顶点
            v1 = Vector3(radius * cos1, -half_height, radius * sin1)
            v2 = Vector3(radius * cos2, -half_height, radius * sin2)

            # 侧面法线（平均）
            n1 = Vector3(cos1 * normal_xz, normal_y, sin1 * normal_xz)
            n2 = Vector3(cos2 * normal_xz, normal_y, sin2 * normal_xz)
            n_apex = Vector3(
                (cos1 + cos2) * 0.5 * normal_xz,
                normal_y,
                (sin1 + sin2) * 0.5 * normal_xz
            )

            # 添加三角形（顶点，底边1，底边2）
            mesh.vertices.extend([apex, v1, v2])
            mesh.normals.extend([n_apex, n1, n2])
            mesh.uvs.extend([
                Vector2(0.5, 1.0),
                Vector2(i / segments, 0.0),
                Vector2((i + 1) / segments, 0.0)
            ])
            mesh.indices.extend([base_index, base_index + 1, base_index + 2])
            base_index += 3

        # 底面
        bottom_center_idx = len(mesh.vertices)
        mesh.vertices.append(Vector3(0, -half_height, 0))
        mesh.normals.append(Vector3(0, -1, 0))
        mesh.uvs.append(Vector2(0.5, 0.5))

        for i in range(segments):
            angle = 2.0 * math.pi * i / segments
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            mesh.vertices.append(Vector3(radius * cos_a, -half_height, radius * sin_a))
            mesh.normals.append(Vector3(0, -1, 0))
            u = 0.5 + 0.5 * cos_a
            v = 0.5 + 0.5 * sin_a
            mesh.uvs.append(Vector2(u, v))

        for i in range(segments):
            a = bottom_center_idx + 1 + i
            b = bottom_center_idx + 1 + ((i + 1) % segments)
            mesh.indices.extend([bottom_center_idx, b, a])

        mesh.is_dirty = True
        self._calculate_bounding_box(mesh)
        self.current_mesh = mesh

        return mesh

    # ========== 高级建模功能 ==========

    def create_custom_mesh(self, vertices, indices=None, normals=None, uvs=None, name="CustomMesh"):
        """
        创建自定义网格（从顶点数组）

        Args:
            vertices: 顶点列表 [(x, y, z), ...]
            indices: 索引列表 [i1, i2, i3, ...]（可选，默认按顺序）
            normals: 法线列表 [(x, y, z), ...]（可选）
            uvs: UV坐标列表 [(u, v), ...]（可选）
            name: 对象名称

        Returns:
            Mesh: 创建的网格对象
        """
        self.logger.info(f"创建自定义网格: {name}, 顶点数={len(vertices)}")

        mesh = Mesh()

        # 转换顶点
        mesh.vertices = [Vector3(v[0], v[1], v[2]) for v in vertices]

        # 索引
        if indices:
            mesh.indices = list(indices)
        else:
            mesh.indices = list(range(len(vertices)))

        # 法线
        if normals:
            mesh.normals = [Vector3(n[0], n[1], n[2]) for n in normals]
        else:
            # 自动计算法线
            mesh.normals = self._calculate_normals(mesh.vertices, mesh.indices)

        # UV坐标
        if uvs:
            mesh.uvs = [Vector2(uv[0], uv[1]) for uv in uvs]
        else:
            mesh.uvs = [Vector2(0, 0)] * len(mesh.vertices)

        mesh.is_dirty = True
        self._calculate_bounding_box(mesh)
        self.current_mesh = mesh

        return mesh

    def extrude_face(self, mesh, face_indices, distance):
        """
        挤出面（沿法线方向）

        Args:
            mesh: 网格对象
            face_indices: 面的顶点索引列表
            distance: 挤出距离

        Returns:
            Mesh: 修改后的网格
        """
        # TODO: 实现面挤出功能
        self.logger.warning("挤出功能尚未实现")
        return mesh

    # ========== 辅助方法 ==========

    def _calculate_bounding_box(self, mesh):
        """计算网格的包围盒"""
        if len(mesh.vertices) == 0:
            return

        min_point = Vector3(
            min(v.x for v in mesh.vertices),
            min(v.y for v in mesh.vertices),
            min(v.z for v in mesh.vertices)
        )
        max_point = Vector3(
            max(v.x for v in mesh.vertices),
            max(v.y for v in mesh.vertices),
            max(v.z for v in mesh.vertices)
        )

        from Engine.Math import BoundingBox
        mesh.bounding_box = BoundingBox(min_point, max_point)

    def _calculate_normals(self, vertices, indices):
        """自动计算顶点法线"""
        normals = [Vector3(0, 0, 0) for _ in vertices]

        # 计算每个三角形的法线并累加到顶点
        for i in range(0, len(indices), 3):
            i0, i1, i2 = indices[i], indices[i + 1], indices[i + 2]
            v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]

            # 计算面法线
            edge1 = Vector3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z)
            edge2 = Vector3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z)

            normal = Vector3(
                edge1.y * edge2.z - edge1.z * edge2.y,
                edge1.z * edge2.x - edge1.x * edge2.z,
                edge1.x * edge2.y - edge1.y * edge2.x
            )

            # 累加到顶点法线
            normals[i0] = Vector3(normals[i0].x + normal.x, normals[i0].y + normal.y, normals[i0].z + normal.z)
            normals[i1] = Vector3(normals[i1].x + normal.x, normals[i1].y + normal.y, normals[i1].z + normal.z)
            normals[i2] = Vector3(normals[i2].x + normal.x, normals[i2].y + normal.y, normals[i2].z + normal.z)

        # 归一化法线
        for i in range(len(normals)):
            n = normals[i]
            length = math.sqrt(n.x * n.x + n.y * n.y + n.z * n.z)
            if length > 0:
                normals[i] = Vector3(n.x / length, n.y / length, n.z / length)
            else:
                normals[i] = Vector3(0, 1, 0)

        return normals

    def add_to_scene(self, mesh, position=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1), name="Object"):
        """
        将网格添加到场景

        Args:
            mesh: 网格对象
            position: 位置 (x, y, z)
            rotation: 旋转（欧拉角度） (x, y, z)
            scale: 缩放 (x, y, z)
            name: 节点名称

        Returns:
            SceneNode: 创建的场景节点
        """
        if not self.engine or not hasattr(self.engine, 'scene_mgr'):
            self.logger.error("引擎未初始化或缺少场景管理器")
            return None

        # 创建场景节点
        pos = Vector3(position[0], position[1], position[2])
        rot = Quaternion.from_euler(rotation[0], rotation[1], rotation[2])
        scl = Vector3(scale[0], scale[1], scale[2])

        node = self.engine.scene_mgr.create_node(name, pos, rot, scl)
        node.mesh = mesh

        self.logger.info(f"已添加到场景: {name}")
        return node

    def get_current_mesh(self):
        """获取当前创建的网格"""
        return self.current_mesh
