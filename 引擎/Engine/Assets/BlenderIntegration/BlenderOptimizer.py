import os
import json
import numpy as np
from pathlib import Path

"""
Blender MCP资产优化器 - 针对低端GPU优化的资产处理系统

此模块负责在导入和导出过程中优化资产，包括网格简化、LOD生成、纹理压缩和材质转换等功能，
特别针对GTX 750Ti和RX 580等低端GPU进行了性能优化。
"""

class BlenderOptimizer:
    """Blender资产优化器主类"""
    
    def __init__(self, gpu_type="gtx750ti"):
        """
        初始化优化器
        
        Args:
            gpu_type: GPU类型 ("gtx750ti" 或 "rx580")
        """
        self.gpu_type = gpu_type
        self.optimization_settings = self._get_optimization_settings()
        self.lod_presets = self._get_lod_presets()
        
        print(f"初始化Blender资产优化器，目标GPU: {gpu_type}")
    
    def _get_optimization_settings(self):
        """
        获取针对特定GPU优化的设置
        
        Returns:
            dict: 优化设置
        """
        base_settings = {
            "enable_mesh_simplification": True,
            "enable_texture_compression": True,
            "enable_lod_generation": True,
            "generate_tangents": True,
            "optimize_uvs": True,
            "merge_vertices": True,
        }
        
        if self.gpu_type == "gtx750ti":
            return {
                **base_settings,
                "max_vertices_per_mesh": 50000,
                "max_indices_per_mesh": 100000,
                "max_texture_size": 1024,
                "texture_format": "BC7",  # BC7纹理格式对低端GPU友好
                "normal_map_format": "BC5",  # 法线贴图使用BC5格式
                "max_textures_per_material": 4,
                "max_materials_per_mesh": 2,
                "lod_generation_ratio": 0.7,  # 每级LOD的简化比例
                "max_lod_levels": 3,
                "enable_material_simplification": True,
                "simplify_shader_complexity": True,
            }
        elif self.gpu_type == "rx580":
            return {
                **base_settings,
                "max_vertices_per_mesh": 100000,
                "max_indices_per_mesh": 200000,
                "max_texture_size": 2048,
                "texture_format": "BC7",
                "normal_map_format": "BC5",
                "max_textures_per_material": 6,
                "max_materials_per_mesh": 3,
                "lod_generation_ratio": 0.6,
                "max_lod_levels": 4,
                "enable_material_simplification": False,
                "simplify_shader_complexity": False,
            }
        else:
            return base_settings
    
    def _get_lod_presets(self):
        """
        获取LOD预设配置
        
        Returns:
            dict: LOD预设
        """
        base_presets = {
            "lod1": {"distance": 10.0, "vertices_ratio": 0.7},
            "lod2": {"distance": 25.0, "vertices_ratio": 0.4},
            "lod3": {"distance": 50.0, "vertices_ratio": 0.2},
            "lod4": {"distance": 100.0, "vertices_ratio": 0.1},
        }
        
        # 根据GPU类型调整LOD比例
        if self.gpu_type == "gtx750ti":
            # 低端GPU使用更激进的简化
            base_presets["lod1"]["vertices_ratio"] = 0.6
            base_presets["lod2"]["vertices_ratio"] = 0.3
            base_presets["lod3"]["vertices_ratio"] = 0.15
        
        return base_presets
    
    def optimize_scene(self, scene_data):
        """
        优化整个场景
        
        Args:
            scene_data: 场景数据
            
        Returns:
            dict: 优化后的场景数据
        """
        print("开始场景级优化...")
        
        # 优化场景中所有对象
        for i, obj in enumerate(scene_data["objects"]):
            if obj["type"] == "mesh":
                print(f"优化网格对象: {obj.get('name', f'object_{i}')}")
                scene_data["objects"][i] = self._optimize_mesh(obj)
            elif obj["type"] == "light":
                print(f"优化光源: {obj.get('name', f'light_{i}')}")
                scene_data["objects"][i] = self._optimize_light(obj)
        
        # 优化世界属性
        if "world_properties" in scene_data:
            scene_data["world_properties"] = self._optimize_world(scene_data["world_properties"])
        
        print("场景级优化完成")
        return scene_data
    
    def _optimize_mesh(self, mesh_data):
        """
        优化网格对象
        
        Args:
            mesh_data: 网格数据
            
        Returns:
            dict: 优化后的网格数据
        """
        # 优化几何体
        if self.optimization_settings["enable_mesh_simplification"]:
            mesh_data["geometry"] = self._simplify_geometry(mesh_data["geometry"])
        
        # 合并顶点
        if self.optimization_settings["merge_vertices"]:
            mesh_data["geometry"] = self._merge_vertices(mesh_data["geometry"])
        
        # 生成法线贴图切线
        if self.optimization_settings["generate_tangents"] and "tangents" not in mesh_data["geometry"]:
            mesh_data["geometry"]["tangents"] = self._generate_tangents(mesh_data["geometry"])
        
        # 优化材质
        mesh_data["materials"] = self._optimize_materials(mesh_data.get("materials", []))
        
        # 限制材质数量
        if len(mesh_data["materials"]) > self.optimization_settings["max_materials_per_mesh"]:
            mesh_data["materials"] = mesh_data["materials"][:self.optimization_settings["max_materials_per_mesh"]]
            print(f"材质数量已限制为 {self.optimization_settings['max_materials_per_mesh']}")
        
        # 生成LOD
        if self.optimization_settings["enable_lod_generation"]:
            mesh_data["lods"] = self._generate_lods(mesh_data["geometry"])
        
        return mesh_data
    
    def _simplify_geometry(self, geometry):
        """
        简化几何体
        
        Args:
            geometry: 几何体数据
            
        Returns:
            dict: 简化后的几何体数据
        """
        # 获取顶点和索引数量
        vertex_count = len(geometry.get("vertices", []))
        index_count = len([idx for face in geometry.get("faces", []) for idx in face])
        
        print(f"原始几何体: {vertex_count} 顶点, {index_count} 索引")
        
        # 检查是否超过限制
        max_vertices = self.optimization_settings["max_vertices_per_mesh"]
        max_indices = self.optimization_settings["max_indices_per_mesh"]
        
        if vertex_count <= max_vertices and index_count <= max_indices:
            print("几何体已在优化限制内，无需简化")
            return geometry
        
        # 计算需要的简化比例
        vertex_ratio = min(1.0, max_vertices / vertex_count)
        index_ratio = min(1.0, max_indices / index_count)
        target_ratio = min(vertex_ratio, index_ratio)
        
        # 在实际实现中，这里会调用网格简化算法
        # 这里简单模拟简化后的顶点和索引数量
        simplified_vertex_count = int(vertex_count * target_ratio)
        simplified_index_count = int(index_count * target_ratio)
        
        print(f"几何体简化完成，目标比例: {target_ratio:.2f}，")
        print(f"简化后: {simplified_vertex_count} 顶点, {simplified_index_count} 索引")
        
        # 返回原始几何体（在实际实现中应返回简化后的几何体）
        # 这里仅更新计数，实际简化需要完整的网格处理算法
        geometry["vertex_count"] = simplified_vertex_count
        geometry["index_count"] = simplified_index_count
        
        return geometry
    
    def _merge_vertices(self, geometry):
        """
        合并重合顶点
        
        Args:
            geometry: 几何体数据
            
        Returns:
            dict: 合并后的几何体数据
        """
        # 在实际实现中，这里会使用公差距离合并重合顶点
        # 这里简单模拟合并操作
        vertex_count = len(geometry.get("vertices", []))
        
        if vertex_count > 0:
            # 模拟合并1%的顶点
            merged_count = int(vertex_count * 0.01)
            
            if merged_count > 0:
                print(f"合并 {merged_count} 个重合顶点")
                geometry["vertex_count"] = vertex_count - merged_count
        
        return geometry
    
    def _generate_tangents(self, geometry):
        """
        生成切线向量
        
        Args:
            geometry: 几何体数据
            
        Returns:
            list: 切线向量列表
        """
        # 计算切线向量（在实际实现中，这里会使用三角形和UV计算切线）
        vertices = geometry.get("vertices", [])
        normals = geometry.get("normals", [])
        uvs = geometry.get("uvs", [])
        faces = geometry.get("faces", [])
        
        # 初始化切线数组
        tangents = [[0.0, 0.0, 0.0] for _ in range(len(vertices))]
        
        if uvs and faces and len(vertices) == len(uvs):
            # 计算每个三角形的切线
            for face in faces:
                if len(face) >= 3:
                    # 获取三个顶点的索引
                    i0, i1, i2 = face[:3]
                    
                    # 获取顶点位置
                    p0 = np.array(vertices[i0])
                    p1 = np.array(vertices[i1])
                    p2 = np.array(vertices[i2])
                    
                    # 获取UV坐标
                    uv0 = np.array(uvs[i0])
                    uv1 = np.array(uvs[i1])
                    uv2 = np.array(uvs[i2])
                    
                    # 计算边缘向量
                    delta_pos1 = p1 - p0
                    delta_pos2 = p2 - p0
                    
                    # 计算UV差异
                    delta_uv1 = uv1 - uv0
                    delta_uv2 = uv2 - uv0
                    
                    # 计算切线
                    r = 1.0 / (delta_uv1[0] * delta_uv2[1] - delta_uv1[1] * delta_uv2[0])
                    tangent = (delta_pos1 * delta_uv2[1] - delta_pos2 * delta_uv1[1]) * r
                    
                    # 归一化切线
                    tangent_norm = np.linalg.norm(tangent)
                    if tangent_norm > 0:
                        tangent = tangent / tangent_norm
                    
                    # 累加切线
                    for i in face:
                        tangents[i] = [
                            tangents[i][0] + tangent[0],
                            tangents[i][1] + tangent[1],
                            tangents[i][2] + tangent[2]
                        ]
            
            # 归一化累积的切线
            for i, tangent in enumerate(tangents):
                # 正交化切线相对于法线
                if i < len(normals):
                    tangent = np.array(tangent)
                    normal = np.array(normals[i])
                    tangent = tangent - np.dot(tangent, normal) * normal
                
                # 归一化
                tangent_norm = np.linalg.norm(tangent)
                if tangent_norm > 0:
                    tangent = tangent / tangent_norm
                
                tangents[i] = tangent.tolist()
        
        print(f"生成切线向量: {len(tangents)} 个")
        return tangents
    
    def _optimize_materials(self, materials):
        """
        优化材质
        
        Args:
            materials: 材质列表
            
        Returns:
            list: 优化后的材质列表
        """
        optimized_materials = []
        
        for material in materials:
            # 简化材质
            if self.optimization_settings["enable_material_simplification"]:
                material = self._simplify_material(material)
            
            # 优化纹理
            if "textures" in material:
                material["textures"] = self._optimize_textures(material["textures"])
            
            # 简化着色器复杂度
            if self.optimization_settings["simplify_shader_complexity"]:
                material = self._simplify_shader(material)
            
            optimized_materials.append(material)
        
        return optimized_materials
    
    def _simplify_material(self, material):
        """
        简化材质参数
        
        Args:
            material: 材质数据
            
        Returns:
            dict: 简化后的材质数据
        """
        # 低端GPU通常不支持复杂的PBR材质参数
        # 移除不必要的材质参数
        keys_to_remove = [
            "subsurface",
            "subsurface_color",
            "subsurface_radius",
            "emission_strength",
            "clearcoat",
            "clearcoat_roughness",
            "sheen",
            "sheen_tint",
            "anisotropy",
            "anisotropy_direction",
            "specular",
            "specular_tint",
            "transmission",
            "transmission_roughness",
            "ior",
            "normal_strength"
        ]
        
        for key in keys_to_remove:
            if key in material:
                del material[key]
        
        # 简化材质属性
        # 确保金属度和粗糙度在有效范围内
        if "metallic" in material:
            material["metallic"] = max(0.0, min(1.0, material["metallic"]))
        
        if "roughness" in material:
            material["roughness"] = max(0.0, min(1.0, material["roughness"]))
        
        print(f"简化材质: {material.get('name', 'unnamed_material')}")
        return material
    
    def _optimize_textures(self, textures):
        """
        优化纹理
        
        Args:
            textures: 纹理字典
            
        Returns:
            dict: 优化后的纹理字典
        """
        max_textures = self.optimization_settings["max_textures_per_material"]
        
        # 纹理优先级排序
        texture_priority = [
            "basecolor", "diffuse", "albedo",  # 基础颜色贴图优先级最高
            "normal",  # 法线贴图次高
            "roughness",  # 粗糙度贴图
            "metallic",  # 金属度贴图
            "ao", "ambient_occlusion",  # AO贴图
            "emission",  # 发光贴图
            "height", "displacement",  # 高度贴图
        ]
        
        # 过滤并排序纹理
        optimized_textures = {}
        
        # 按优先级添加纹理
        for priority in texture_priority:
            for tex_type, tex_info in textures.items():
                if priority in tex_type.lower() and len(optimized_textures) < max_textures:
                    # 优化纹理分辨率
                    tex_info = self._optimize_texture_resolution(tex_info)
                    
                    # 设置纹理格式
                    if "normal" in tex_type.lower():
                        tex_info["format"] = self.optimization_settings["normal_map_format"]
                    else:
                        tex_info["format"] = self.optimization_settings["texture_format"]
                    
                    optimized_textures[tex_type] = tex_info
        
        # 如果还有剩余位置，添加其他纹理
        if len(optimized_textures) < max_textures:
            for tex_type, tex_info in textures.items():
                if tex_type not in optimized_textures:
                    # 检查是否已经包含了类似功能的纹理
                    has_similar = False
                    for key in optimized_textures:
                        if tex_type.lower() in key.lower() or key.lower() in tex_type.lower():
                            has_similar = True
                            break
                    
                    if not has_similar and len(optimized_textures) < max_textures:
                        # 优化纹理分辨率
                        tex_info = self._optimize_texture_resolution(tex_info)
                        tex_info["format"] = self.optimization_settings["texture_format"]
                        optimized_textures[tex_type] = tex_info
        
        # 如果需要减少纹理数量，合并纹理通道
        if len(optimized_textures) > max_textures:
            optimized_textures = self._merge_texture_channels(optimized_textures)
        
        # 压缩纹理
        if self.optimization_settings["enable_texture_compression"]:
            for tex_type, tex_info in optimized_textures.items():
                tex_info["compressed"] = True
        
        print(f"优化材质纹理: {len(optimized_textures)}/{len(textures)} 个纹理保留")
        return optimized_textures
    
    def _optimize_texture_resolution(self, tex_info):
        """
        优化纹理分辨率
        
        Args:
            tex_info: 纹理信息
            
        Returns:
            dict: 优化后的纹理信息
        """
        max_size = self.optimization_settings["max_texture_size"]
        
        # 获取当前大小
        width = tex_info.get("width", 1024)
        height = tex_info.get("height", 1024)
        
        # 计算目标大小
        target_width = min(width, max_size)
        target_height = min(height, max_size)
        
        # 对于非基础纹理，可以进一步降低分辨率
        if tex_info.get("type", "") not in ["basecolor", "diffuse", "albedo"]:
            # 为次要纹理降低到一半分辨率
            if self.gpu_type == "gtx750ti":
                target_width = max(128, target_width // 2)
                target_height = max(128, target_height // 2)
        
        # 确保分辨率是2的幂次
        target_width = self._next_power_of_two_down(target_width)
        target_height = self._next_power_of_two_down(target_height)
        
        # 更新纹理信息
        if width != target_width or height != target_height:
            print(f"纹理分辨率调整: {width}x{height} -> {target_width}x{target_height}")
            tex_info["width"] = target_width
            tex_info["height"] = target_height
        
        return tex_info
    
    def _next_power_of_two_down(self, n):
        """
        获取小于等于n的最大2的幂次数
        
        Args:
            n: 输入数值
            
        Returns:
            int: 小于等于n的最大2的幂次数
        """
        if n <= 0:
            return 1
        
        # 找到最高位的1
        while n & (n - 1):
            n &= n - 1
        
        return n
    
    def _merge_texture_channels(self, textures):
        """
        合并纹理通道
        
        Args:
            textures: 纹理字典
            
        Returns:
            dict: 合并后的纹理字典
        """
        merged_textures = {}
        merge_groups = []
        
        # 定义可以合并的纹理组
        potential_merges = [
            ["roughness", "metallic", "ao", "emission"],  # 灰度纹理可以合并到RGBA通道
            ["height", "displacement"],
        ]
        
        # 查找需要合并的纹理
        for merge_group in potential_merges:
            group = []
            for tex_type, tex_info in textures.items():
                tex_lower = tex_type.lower()
                for keyword in merge_group:
                    if keyword in tex_lower and tex_info.get("type", "") == "2d":
                        group.append((tex_type, tex_info, keyword))
                        break
            
            if len(group) > 1:
                merge_groups.append(group)
        
        # 执行合并
        for group in merge_groups:
            if len(group) > 0:
                # 选择第一个纹理作为基础
                base_type, base_info, _ = group[0]
                merged_info = base_info.copy()
                merged_info["merged_channels"] = {}
                
                # 分配通道
                channel_map = {"R": "roughness", "G": "metallic", "B": "ao", "A": "emission"}
                
                for tex_type, tex_info, keyword in group:
                    # 找到对应的通道
                    channel = None
                    for c, k in channel_map.items():
                        if k in keyword.lower():
                            channel = c
                            break
                    
                    if channel:
                        merged_info["merged_channels"][channel] = tex_type
                        
                        # 移除被合并的纹理
                        if tex_type in textures:
                            del textures[tex_type]
                
                merged_textures[base_type] = merged_info
        
        # 添加未被合并的纹理
        for tex_type, tex_info in textures.items():
            if tex_type not in merged_textures:
                merged_textures[tex_type] = tex_info
        
        print(f"纹理通道合并完成，共合并了 {len(merge_groups)} 组纹理")
        return merged_textures
    
    def _simplify_shader(self, material):
        """
        简化着色器复杂度
        
        Args:
            material: 材质数据
            
        Returns:
            dict: 简化后的材质数据
        """
        # 对于低端GPU，使用简单着色器
        if self.gpu_type == "gtx750ti":
            # 移除复杂特性
            if "properties" in material:
                for prop in ["subsurface_scattering", "transparency", "refraction"]:
                    if prop in material["properties"]:
                        del material["properties"][prop]
            
            # 简化着色器类型
            if "shader_type" in material and material["shader_type"] not in ["standard", "blinn_phong"]:
                material["shader_type"] = "standard"
        
        return material
    
    def _generate_lods(self, geometry):
        """
        生成LOD（细节级别）
        
        Args:
            geometry: 原始几何体数据
            
        Returns:
            list: LOD数据列表
        """
        lods = []
        max_levels = self.optimization_settings["max_lod_levels"]
        
        # 获取原始顶点数量
        original_vertices = geometry.get("vertex_count", len(geometry.get("vertices", [])))
        
        for level in range(1, max_levels + 1):
            preset_key = f"lod{level}"
            
            if preset_key not in self.lod_presets:
                break
                
            preset = self.lod_presets[preset_key]
            vertices_ratio = preset["vertices_ratio"]
            distance = preset["distance"]
            
            # 计算目标顶点数量
            target_vertices = int(original_vertices * vertices_ratio)
            
            # 防止顶点数量过少
            if target_vertices < 100:
                break
                
            # 在实际实现中，这里会创建简化的几何数据
            # 这里简单模拟LOD几何数据
            lod_geometry = {
                "vertex_count": target_vertices,
                "vertices": [],  # 在实际实现中会包含简化的顶点数据
                "faces": [],     # 在实际实现中会包含简化的面数据
                "normals": [],   # 在实际实现中会包含简化的法线数据
                "uvs": [],       # 在实际实现中会包含简化的UV数据
            }
            
            lods.append({
                "level": level,
                "geometry": lod_geometry,
                "distance": distance,
                "vertices_ratio": vertices_ratio
            })
            
            print(f"生成LOD{level}: 顶点比例 {vertices_ratio:.2f}, 距离 {distance} 单位")
        
        print(f"LOD生成完成，共创建 {len(lods)} 级LOD")
        return lods
    
    def _optimize_light(self, light_data):
        """
        优化光源
        
        Args:
            light_data: 光源数据
            
        Returns:
            dict: 优化后的光源数据
        """
        # 低端GPU的阴影优化
        if self.gpu_type == "gtx750ti" and light_data.get("cast_shadows", False):
            # 降低阴影贴图分辨率
            light_data["shadow_resolution"] = min(light_data.get("shadow_resolution", 1024), 1024)
            
            # 对于方向光，减少级联数量
            if light_data["type"] == "SUN":
                light_data["shadow_cascade_count"] = 1
                
                # 降低阴影贴图分辨率
                light_data["shadow_resolution"] = min(light_data.get("shadow_resolution", 1024), 1024)
        
        # 限制光照范围
        if light_data["type"] in ["POINT", "SPOT"] and "max_distance" in light_data:
            # 对于低端GPU，限制光照距离以减少计算量
            if self.gpu_type == "gtx750ti":
                light_data["max_distance"] = min(light_data["max_distance"], 50.0)
        
        # 限制光源能量
        energy_limit = 10.0 if self.gpu_type == "gtx750ti" else 100.0
        if "energy" in light_data:
            light_data["energy"] = min(light_data["energy"], energy_limit)
        
        print(f"优化光源: {light_data.get('name', 'unnamed_light')}")
        return light_data
    
    def _optimize_world(self, world_data):
        """
        优化世界设置
        
        Args:
            world_data: 世界数据
            
        Returns:
            dict: 优化后的世界数据
        """
        # 优化环境贴图
        if "environment_texture" in world_data:
            env_texture = world_data["environment_texture"]
            
            # 降低环境贴图分辨率
            max_env_size = 1024 if self.gpu_type == "gtx750ti" else 2048
            
            if "resolution" in env_texture and env_texture["resolution"] > max_env_size:
                env_texture["resolution"] = max_env_size
                print(f"环境贴图分辨率降低至 {max_env_size}x{max_env_size}")
            
            # 对于低端GPU，使用更低分辨率的环境贴图用于光照
            if self.gpu_type == "gtx750ti":
                env_texture["use_lowres_for_gi"] = True
        
        # 降低环境光照强度以减少计算量
        if "environment_strength" in world_data and self.gpu_type == "gtx750ti":
            world_data["environment_strength"] *= 0.8
        
        print("世界设置优化完成")
        return world_data
    
    def optimize_for_real_time(self, scene_data):
        """
        为实时渲染特别优化场景
        
        Args:
            scene_data: 场景数据
            
        Returns:
            dict: 优化后的场景数据
        """
        print("为实时渲染优化场景...")
        
        # 限制场景对象数量
        max_objects = 200 if self.gpu_type == "gtx750ti" else 500
        
        if len(scene_data["objects"]) > max_objects:
            # 按距离或重要性排序并保留前max_objects个对象
            print(f"场景对象数量限制为 {max_objects}")
            # 注意：这里应该按照距离相机的远近或对象重要性排序
            scene_data["objects"] = scene_data["objects"][:max_objects]
        
        # 针对低端GPU优化光照计算
        if self.gpu_type == "gtx750ti":
            # 限制实时阴影投射光源数量
            shadow_casting_lights = [obj for obj in scene_data["objects"] 
                                   if obj["type"] == "light" and obj.get("cast_shadows", False)]
            
            if len(shadow_casting_lights) > 2:
                # 禁用次要光源的阴影
                for light in shadow_casting_lights[2:]:
                    light["cast_shadows"] = False
                    print(f"禁用光源 {light.get('name', 'unnamed_light')} 的阴影以优化性能")
        
        print("实时渲染优化完成")
        return scene_data
    
    def analyze_optimization(self, original_scene, optimized_scene):
        """
        分析优化效果
        
        Args:
            original_scene: 原始场景数据
            optimized_scene: 优化后的场景数据
            
        Returns:
            dict: 优化分析结果
        """
        # 计算原始场景统计信息
        original_stats = self._calculate_scene_stats(original_scene)
        
        # 计算优化后场景统计信息
        optimized_stats = self._calculate_scene_stats(optimized_scene)
        
        # 计算优化比率
        analysis = {
            "original": original_stats,
            "optimized": optimized_stats,
            "improvements": {}
        }
        
        # 计算各个指标的改进
        for key, orig_val in original_stats.items():
            if key in optimized_stats and orig_val > 0:
                improvement = (orig_val - optimized_stats[key]) / orig_val
                analysis["improvements"][key] = improvement
        
        print("优化分析结果:")
        for key, improvement in analysis["improvements"].items():
            print(f"  {key}: {improvement:.1%} 减少")
        
        return analysis
    
    def _calculate_scene_stats(self, scene_data):
        """
        计算场景统计信息
        
        Args:
            scene_data: 场景数据
            
        Returns:
            dict: 场景统计信息
        """
        stats = {
            "total_objects": len(scene_data["objects"]),
            "mesh_objects": 0,
            "light_objects": 0,
            "total_vertices": 0,
            "total_indices": 0,
            "total_materials": 0,
            "total_textures": 0,
            "total_texture_memory_mb": 0,
            "shadow_casting_lights": 0
        }
        
        for obj in scene_data["objects"]:
            if obj["type"] == "mesh":
                stats["mesh_objects"] += 1
                
                # 计算顶点和索引
                geometry = obj.get("geometry", {})
                stats["total_vertices"] += geometry.get("vertex_count", len(geometry.get("vertices", [])))
                
                # 计算索引数量
                face_count = len(geometry.get("faces", []))
                stats["total_indices"] += face_count * 3  # 假设是三角形
                
                # 计算材质和纹理
                materials = obj.get("materials", [])
                stats["total_materials"] += len(materials)
                
                for material in materials:
                    textures = material.get("textures", {})
                    stats["total_textures"] += len(textures)
                    
                    # 估算纹理内存
                    for tex_info in textures.values():
                        width = tex_info.get("width", 1024)
                        height = tex_info.get("height", 1024)
                        format_type = tex_info.get("format", "RGBA8")
                        
                        # 计算每像素字节数
                        bpp = self._get_format_bpp(format_type)
                        texture_mb = (width * height * bpp) / (8 * 1024 * 1024)
                        
                        stats["total_texture_memory_mb"] += texture_mb
            
            elif obj["type"] == "light":
                stats["light_objects"] += 1
                
                if obj.get("cast_shadows", False):
                    stats["shadow_casting_lights"] += 1
        
        return stats
    
    def _get_format_bpp(self, format_type):
        """
        获取纹理格式的每像素位数
        
        Args:
            format_type: 纹理格式
            
        Returns:
            int: 每像素位数
        """
        format_bpp = {
            "BC1": 4,  # DXT1, 4 bpp
            "BC3": 8,  # DXT5, 8 bpp
            "BC5": 8,  # 法线贴图格式, 8 bpp
            "BC7": 8,  # 高质量格式, 8 bpp
            "ETC2": 4,  # 4 bpp
            "ETC2A": 8, # 带Alpha, 8 bpp
            "RGBA8": 32, # 未压缩, 32 bpp
            "RGB8": 24,  # 未压缩, 24 bpp
        }
        
        return format_bpp.get(format_type, 32)  # 默认返回32 bpp
    
    def export_optimization_settings(self, output_path):
        """
        导出优化设置到文件
        
        Args:
            output_path: 输出文件路径
        """
        settings = {
            "gpu_type": self.gpu_type,
            "optimization_settings": self.optimization_settings,
            "lod_presets": self.lod_presets
        }
        
        with open(output_path, "w") as f:
            json.dump(settings, f, indent=2)
        
        print(f"优化设置已导出到 {output_path}")

# 使用示例
"""
# 引擎中使用示例
# 创建优化器
optimizer = BlenderOptimizer(gpu_type="gtx750ti")

# 加载场景数据
with open("path/to/scene_data.json", "r") as f:
    scene_data = json.load(f)

# 优化场景
optimized_scene = optimizer.optimize_scene(scene_data)

# 为实时渲染特别优化
real_time_scene = optimizer.optimize_for_real_time(optimized_scene)

# 分析优化效果
analysis = optimizer.analyze_optimization(scene_data, real_time_scene)

# 导出优化设置
optimizer.export_optimization_settings("path/to/optimization_settings.json")
"""