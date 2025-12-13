# -*- coding: utf-8 -*-
"""
模型加载器 - 支持导入外部3D模型
Model Loader - Import external 3D models
支持格式：OBJ, STL, PLY, GLTF, GLB, DAE, OFF, 3DS等30+种格式
"""

import os
from Engine.Math import Vector3, Vector2
from .Mesh import Mesh
from Engine.Logger import get_logger

# 尝试导入trimesh（通用加载器）
try:
    import trimesh
    import numpy as np
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    print("警告：trimesh未安装，只支持OBJ格式")
    print("安装命令: pip install trimesh")


class ModelLoader:
    """模型加载器类 - 支持多种3D格式"""

    def __init__(self):
        self.logger = get_logger("ModelLoader")

    @staticmethod
    def load_model(file_path):
        """
        通用模型加载方法，自动检测格式

        Args:
            file_path: 模型文件路径

        Returns:
            Mesh: 加载的网格对象，如果失败返回None
        """
        logger = get_logger("ModelLoader")

        if not os.path.exists(file_path):
            logger.error(f"模型文件不存在: {file_path}")
            return None

        # 获取文件扩展名
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        # 如果是OBJ格式，使用原有的加载器（更可靠）
        if ext == '.obj':
            logger.info("使用OBJ专用加载器")
            return ModelLoader.load_obj(file_path)

        # 其他格式使用trimesh加载
        if HAS_TRIMESH:
            logger.info(f"使用trimesh加载 {ext} 格式")
            return ModelLoader._load_with_trimesh(file_path)
        else:
            logger.error(f"不支持的格式 {ext}，且trimesh未安装")
            return None

    @staticmethod
    def _load_with_trimesh(file_path):
        """
        使用trimesh库加载模型

        Args:
            file_path: 模型文件路径

        Returns:
            Mesh: 加载的网格对象，如果失败返回None
        """
        logger = get_logger("ModelLoader")

        try:
            logger.info(f"开始加载模型: {file_path}")

            # 使用trimesh加载模型
            mesh_data = trimesh.load(file_path)

            # trimesh可能返回Scene或Mesh对象
            if isinstance(mesh_data, trimesh.Scene):
                # 如果是场景，合并所有网格
                logger.info("检测到场景对象，合并所有网格...")
                mesh_data = mesh_data.dump(concatenate=True)

            # 确保是Trimesh对象
            if not isinstance(mesh_data, trimesh.Trimesh):
                logger.error("加载的数据不是有效的网格对象")
                return None

            # 转换为我们的Mesh格式
            mesh = Mesh()

            # 提取顶点
            vertices = mesh_data.vertices
            mesh.vertices = [Vector3(v[0], v[1], v[2]) for v in vertices]

            # 提取法线
            if hasattr(mesh_data, 'vertex_normals') and mesh_data.vertex_normals is not None:
                normals = mesh_data.vertex_normals
                mesh.normals = [Vector3(n[0], n[1], n[2]) for n in normals]
            else:
                # 如果没有法线，生成默认法线
                mesh.normals = [Vector3(0, 1, 0)] * len(mesh.vertices)

            # 提取UV坐标（如果有）
            if hasattr(mesh_data.visual, 'uv') and mesh_data.visual.uv is not None:
                uvs = mesh_data.visual.uv
                mesh.uvs = [Vector2(uv[0], uv[1]) for uv in uvs]
            else:
                mesh.uvs = [Vector2(0, 0)] * len(mesh.vertices)

            # 提取面索引
            faces = mesh_data.faces
            mesh.indices = faces.flatten().tolist()

            mesh.is_dirty = True

            # 计算包围盒
            if len(mesh.vertices) > 0:
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

            logger.info(f"模型加载成功: {len(mesh.vertices)} 顶点, {len(mesh.indices)//3} 三角形")
            return mesh

        except Exception as e:
            logger.error(f"使用trimesh加载模型失败: {e}", exc_info=True)
            return None

    @staticmethod
    def load_obj(file_path):
        """
        加载OBJ格式模型文件

        Args:
            file_path: OBJ文件路径

        Returns:
            Mesh: 加载的网格对象，如果失败返回None
        """
        logger = get_logger("ModelLoader")

        if not os.path.exists(file_path):
            logger.error(f"模型文件不存在: {file_path}")
            return None

        logger.info(f"开始加载OBJ模型: {file_path}")

        try:
            # 临时存储
            temp_vertices = []
            temp_normals = []
            temp_uvs = []

            # 最终网格数据
            vertices = []
            normals = []
            uvs = []
            indices = []

            # 顶点索引映射（用于去重）
            vertex_map = {}
            current_index = 0

            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()

                    # 跳过空行和注释
                    if not line or line.startswith('#'):
                        continue

                    parts = line.split()
                    if not parts:
                        continue

                    command = parts[0]

                    # 解析顶点位置 (v x y z)
                    if command == 'v':
                        if len(parts) >= 4:
                            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                            temp_vertices.append(Vector3(x, y, z))

                    # 解析纹理坐标 (vt u v)
                    elif command == 'vt':
                        if len(parts) >= 3:
                            u, v = float(parts[1]), float(parts[2])
                            temp_uvs.append(Vector2(u, v))

                    # 解析法线 (vn x y z)
                    elif command == 'vn':
                        if len(parts) >= 4:
                            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                            temp_normals.append(Vector3(x, y, z))

                    # 解析面 (f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3)
                    elif command == 'f':
                        if len(parts) < 4:
                            continue

                        # OBJ支持三角形和四边形，我们只处理三角形
                        # 如果是四边形，转换为两个三角形
                        face_vertices = parts[1:]

                        # 解析面的顶点索引
                        face_indices = []
                        for vert_str in face_vertices:
                            # 格式可能是: v, v/vt, v/vt/vn, v//vn
                            indices_parts = vert_str.split('/')

                            v_idx = int(indices_parts[0]) - 1  # OBJ索引从1开始
                            vt_idx = -1
                            vn_idx = -1

                            if len(indices_parts) > 1 and indices_parts[1]:
                                vt_idx = int(indices_parts[1]) - 1
                            if len(indices_parts) > 2 and indices_parts[2]:
                                vn_idx = int(indices_parts[2]) - 1

                            # 创建顶点键（用于去重）
                            vertex_key = (v_idx, vt_idx, vn_idx)

                            # 检查是否已经存在这个顶点组合
                            if vertex_key not in vertex_map:
                                # 添加新顶点
                                if v_idx < len(temp_vertices):
                                    vertices.append(temp_vertices[v_idx])
                                else:
                                    vertices.append(Vector3(0, 0, 0))

                                if vt_idx >= 0 and vt_idx < len(temp_uvs):
                                    uvs.append(temp_uvs[vt_idx])
                                else:
                                    uvs.append(Vector2(0, 0))

                                if vn_idx >= 0 and vn_idx < len(temp_normals):
                                    normals.append(temp_normals[vn_idx])
                                else:
                                    normals.append(Vector3(0, 1, 0))

                                vertex_map[vertex_key] = current_index
                                face_indices.append(current_index)
                                current_index += 1
                            else:
                                face_indices.append(vertex_map[vertex_key])

                        # 三角形化（如果是四边形）
                        if len(face_indices) == 3:
                            # 三角形
                            indices.extend(face_indices)
                        elif len(face_indices) == 4:
                            # 四边形 -> 两个三角形
                            indices.extend([face_indices[0], face_indices[1], face_indices[2]])
                            indices.extend([face_indices[0], face_indices[2], face_indices[3]])
                        elif len(face_indices) > 4:
                            # 多边形 -> 扇形三角化
                            for i in range(1, len(face_indices) - 1):
                                indices.extend([face_indices[0], face_indices[i], face_indices[i + 1]])

            # 创建Mesh对象
            mesh = Mesh()
            mesh.vertices = vertices
            mesh.normals = normals if normals else [Vector3(0, 1, 0)] * len(vertices)
            mesh.uvs = uvs if uvs else [Vector2(0, 0)] * len(vertices)
            mesh.indices = indices
            mesh.is_dirty = True

            # 计算包围盒
            if vertices:
                min_point = Vector3(
                    min(v.x for v in vertices),
                    min(v.y for v in vertices),
                    min(v.z for v in vertices)
                )
                max_point = Vector3(
                    max(v.x for v in vertices),
                    max(v.y for v in vertices),
                    max(v.z for v in vertices)
                )
                from Engine.Math import BoundingBox
                mesh.bounding_box = BoundingBox(min_point, max_point)

            logger.info(f"OBJ模型加载成功: {len(vertices)} 顶点, {len(indices)//3} 三角形")
            return mesh

        except Exception as e:
            logger.error(f"加载OBJ模型失败: {e}", exc_info=True)
            return None

    @staticmethod
    def get_supported_formats():
        """
        获取支持的文件格式

        Returns:
            list: 支持的格式列表（用于文件对话框）
        """
        formats = [
            ("所有支持的格式", "*.obj *.stl *.ply *.gltf *.glb *.dae *.off *.3ds"),
            ("OBJ文件 (推荐)", "*.obj"),
        ]

        # 如果trimesh可用，添加更多格式
        if HAS_TRIMESH:
            formats.extend([
                ("STL文件 (3D打印)", "*.stl"),
                ("PLY文件 (点云)", "*.ply"),
                ("GLTF文件 (现代标准)", "*.gltf *.glb"),
                ("Collada文件", "*.dae"),
                ("OFF文件", "*.off"),
                ("3DS文件", "*.3ds"),
            ])

        formats.append(("所有文件", "*.*"))
        return formats
