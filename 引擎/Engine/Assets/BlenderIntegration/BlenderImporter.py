import os
import json
import numpy as np
from pathlib import Path

"""
Blender MCP资产导入器 - 针对低端GPU优化的资产加载系统

此模块负责从Blender导出的格式加载资产到渲染引擎中，特别针对GTX 750Ti和RX 580等低端GPU进行了优化。
实现了资产流式加载、纹理压缩、实例化管理等核心功能。
"""

class BlenderImporter:
    """Blender资产导入器主类"""
    
    def __init__(self, engine_renderer, gpu_type="gtx750ti"):
        """
        初始化导入器
        
        Args:
            engine_renderer: 引擎渲染器实例
            gpu_type: GPU类型 ("gtx750ti" 或 "rx580")
        """
        self.renderer = engine_renderer
        self.gpu_type = gpu_type
        self.loaded_assets = {}
        self.asset_cache = {}
        self.vram_budget = self._get_vram_budget()
        self.current_vram_usage = 0
        self.optimization_settings = self._get_optimization_settings()
        
        print(f"初始化Blender资产导入器，目标GPU: {gpu_type}, VRAM预算: {self.vram_budget}MB")
    
    def _get_vram_budget(self):
        """
        根据GPU类型确定VRAM预算
        
        Returns:
            int: VRAM预算(MB)
        """
        if self.gpu_type == "gtx750ti":
            return 1800  # 2GB总VRAM中预留1800MB给应用
        elif self.gpu_type == "rx580":
            return 7000  # 8GB总VRAM中预留7000MB给应用
        else:
            return 4000  # 默认4GB
    
    def _get_optimization_settings(self):
        """
        获取针对特定GPU优化的设置
        
        Returns:
            dict: 优化设置
        """
        base_settings = {
            "enable_texture_compression": True,
            "use_texture_atlasing": True,
            "enable_lod": True,
            "max_lod_distance": 200.0,
            "instance_threshold": 5,  # 超过此数量的相同模型自动实例化
            "streaming_distance": 100.0,  # 资产流式加载距离
            "async_loading": True,  # 异步加载
        }
        
        if self.gpu_type == "gtx750ti":
            return {
                **base_settings,
                "max_loaded_textures": 256,
                "texture_resolution_scale": 0.5,  # 缩小纹理分辨率
                "max_batched_objects": 200,  # 每批最大对象数
                "enable_instancing": True,  # 启用实例化渲染
                "preload_priority": "high",  # 预加载优先级高的资产
            }
        elif self.gpu_type == "rx580":
            return {
                **base_settings,
                "max_loaded_textures": 512,
                "texture_resolution_scale": 1.0,
                "max_batched_objects": 400,
                "enable_instancing": True,
                "preload_priority": "medium",
            }
        else:
            return base_settings
    
    def load_scene(self, scene_file):
        """
        加载完整场景
        
        Args:
            scene_file: 场景元数据文件路径
            
        Returns:
            dict: 加载的场景数据
        """
        print(f"开始加载场景: {scene_file}")
        
        # 读取场景元数据
        with open(scene_file, "r") as f:
            scene_metadata = json.load(f)
        
        # 获取导出目录
        export_dir = os.path.dirname(scene_file)
        
        # 加载资产
        scene_objects = []
        
        for asset_info in scene_metadata.get("exported_assets", []):
            asset_path = os.path.join(export_dir, asset_info["path"])
            
            try:
                if asset_info["type"] == "mesh":
                    obj = self._load_mesh(asset_path)
                elif asset_info["type"] == "light":
                    obj = self._load_light(asset_path)
                elif asset_info["type"] == "camera":
                    obj = self._load_camera(asset_path)
                else:
                    print(f"未知资产类型: {asset_info['type']}")
                    continue
                
                scene_objects.append(obj)
                self.loaded_assets[asset_info["name"]] = obj
                
            except Exception as e:
                print(f"加载资产失败 {asset_info['name']}: {str(e)}")
        
        # 创建场景对象
        scene = {
            "name": scene_metadata["name"],
            "objects": scene_objects,
            "world_properties": scene_metadata.get("world_properties", {})
        }
        
        # 应用优化
        self._optimize_scene(scene)
        
        print(f"场景加载完成。共加载 {len(scene_objects)} 个对象")
        return scene
    
    def _load_mesh(self, mesh_dir):
        """
        加载网格资产
        
        Args:
            mesh_dir: 网格目录路径
            
        Returns:
            dict: 加载的网格对象
        """
        print(f"加载网格: {mesh_dir}")
        
        # 加载几何数据
        geometry_file = os.path.join(mesh_dir, "geometry.json")
        with open(geometry_file, "r") as f:
            geometry_data = json.load(f)
        
        # 加载材质数据
        materials = []
        materials_file = os.path.join(mesh_dir, "materials.json")
        if os.path.exists(materials_file):
            with open(materials_file, "r") as f:
                materials = json.load(f)
                # 加载材质纹理
                for material in materials:
                    if "textures" in material:
                        self._load_material_textures(material, os.path.dirname(materials_file))
        
        # 加载变换数据
        transform_file = os.path.join(mesh_dir, "transform.json")
        transform = {}
        if os.path.exists(transform_file):
            with open(transform_file, "r") as f:
                transform = json.load(f)
        
        # 加载LOD数据
        lods = self._load_lods(mesh_dir)
        
        # 创建网格对象
        mesh_obj = {
            "type": "mesh",
            "name": transform.get("name", "unnamed_mesh"),
            "geometry": geometry_data,
            "materials": materials,
            "transform": transform,
            "lods": lods,
            "vram_cost": self._calculate_vram_cost(geometry_data, materials)
        }
        
        # 跟踪VRAM使用
        self.current_vram_usage += mesh_obj["vram_cost"]
        
        return mesh_obj
    
    def _load_lods(self, mesh_dir):
        """
        加载LOD数据
        
        Args:
            mesh_dir: 网格目录路径
            
        Returns:
            list: LOD数据列表
        """
        if not self.optimization_settings["enable_lod"]:
            return []
        
        lods = []
        
        # 查找并加载所有LOD级别
        for lod_level in range(1, 10):  # 最大尝试10级LOD
            lod_dir = os.path.join(mesh_dir, f"lod{lod_level}")
            if not os.path.exists(lod_dir):
                break
            
            geometry_file = os.path.join(lod_dir, "geometry.json")
            if os.path.exists(geometry_file):
                with open(geometry_file, "r") as f:
                    lod_data = json.load(f)
                    
                    # 计算LOD距离 (指数增长)
                    distance = 10.0 * (1.5 ** lod_level)
                    
                    lods.append({
                        "level": lod_level,
                        "geometry": lod_data,
                        "distance": distance,
                        "vram_cost": self._calculate_vram_cost(lod_data, [])
                    })
        
        print(f"加载了 {len(lods)} 级LOD数据")
        return lods
    
    def _load_material_textures(self, material, base_dir):
        """
        加载材质纹理
        
        Args:
            material: 材质数据
            base_dir: 基础目录路径
        """
        textures = material["textures"]
        
        for tex_type, tex_info in textures.items():
            tex_path = os.path.join(base_dir, "..", "..", tex_info["file_path"])
            
            # 检查是否已加载
            if tex_path not in self.asset_cache:
                # 根据GPU优化加载纹理
                tex_data = self._load_texture(tex_path, tex_type)
                self.asset_cache[tex_path] = tex_data
                
                # 更新VRAM使用
                tex_vram = tex_info["width"] * tex_info["height"] * self._get_format_bpp(tex_info["format"]) / (8 * 1024 * 1024)
                self.current_vram_usage += tex_vram
            
            # 更新材质中的纹理引用
            textures[tex_type]["handle"] = tex_path
    
    def _load_texture(self, tex_path, tex_type):
        """
        加载并优化纹理
        
        Args:
            tex_path: 纹理文件路径
            tex_type: 纹理类型
            
        Returns:
            dict: 纹理数据
        """
        # 这里将在实际实现中调用引擎的纹理加载API
        # 模拟纹理加载过程
        print(f"加载纹理: {tex_path}")
        
        # 获取文件扩展名
        ext = os.path.splitext(tex_path)[1].lower()
        
        # 根据GPU类型和优化设置调整纹理参数
        scale = self.optimization_settings["texture_resolution_scale"]
        
        # 返回纹理句柄信息
        return {
            "path": tex_path,
            "type": tex_type,
            "format": "BC7" if tex_type != "normal" else "BC5",
            "scale": scale,
            "compressed": self.optimization_settings["enable_texture_compression"]
        }
    
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
    
    def _load_light(self, light_file):
        """
        加载光源
        
        Args:
            light_file: 光源文件路径
            
        Returns:
            dict: 光源对象
        """
        print(f"加载光源: {light_file}")
        
        with open(light_file, "r") as f:
            light_data = json.load(f)
        
        # 根据GPU类型优化光源设置
        if self.gpu_type == "gtx750ti":
            # 低端GPU优化
            if light_data.get("cast_shadows", False):
                # 限制阴影贴图分辨率
                light_data["shadow_resolution"] = min(light_data.get("shadow_resolution", 1024), 1024)
                
                # 对于点光源，减少阴影更新频率
                if light_data["type"] == "POINT":
                    light_data["shadow_update_frequency"] = 2  # 每2帧更新一次
        
        return light_data
    
    def _load_camera(self, camera_file):
        """
        加载相机
        
        Args:
            camera_file: 相机文件路径
            
        Returns:
            dict: 相机对象
        """
        print(f"加载相机: {camera_file}")
        
        with open(camera_file, "r") as f:
            camera_data = json.load(f)
        
        # 根据GPU类型优化相机设置
        if self.gpu_type == "gtx750ti":
            # 低端GPU关闭景深效果
            camera_data["depth_of_field"] = False
        
        return camera_data
    
    def _calculate_vram_cost(self, geometry, materials):
        """
        计算对象的VRAM成本估计
        
        Args:
            geometry: 几何数据
            materials: 材质数据
            
        Returns:
            float: VRAM成本(MB)
        """
        # 计算顶点数据VRAM成本 (假设每个顶点64字节: 位置+法线+UV+切线)
        vertex_cost = geometry.get("vertex_count", 0) * 64 / (1024 * 1024)
        
        # 计算纹理VRAM成本
        texture_cost = 0.0
        for material in materials:
            if "textures" in material:
                for tex_info in material["textures"].values():
                    format_bpp = self._get_format_bpp(tex_info.get("format", "RGBA8"))
                    tex_width = tex_info.get("width", 1024)
                    tex_height = tex_info.get("height", 1024)
                    texture_cost += (tex_width * tex_height * format_bpp) / (8 * 1024 * 1024)
        
        return vertex_cost + texture_cost
    
    def _optimize_scene(self, scene):
        """
        优化场景以适应低端GPU
        
        Args:
            scene: 场景数据
        """
        print("开始场景优化...")
        
        # 1. 合并相同材质的对象以减少绘制调用
        if self.optimization_settings["use_texture_atlasing"]:
            self._merge_similar_objects(scene)
        
        # 2. 设置实例化渲染
        if self.optimization_settings["enable_instancing"]:
            self._setup_instancing(scene)
        
        # 3. 优化光源数量
        self._optimize_lights(scene)
        
        # 4. 检查VRAM使用并优化
        self._optimize_vram_usage(scene)
        
        print(f"场景优化完成。当前VRAM使用估计: {self.current_vram_usage:.2f}MB")
    
    def _merge_similar_objects(self, scene):
        """
        合并具有相同材质的对象
        
        Args:
            scene: 场景数据
        """
        # 按材质分组对象
        material_groups = {}
        
        for obj in scene["objects"]:
            if obj["type"] != "mesh" or not obj.get("materials"):
                continue
                
            # 创建材质标识符
            material_key = str(obj["materials"])
            
            if material_key not in material_groups:
                material_groups[material_key] = []
            
            material_groups[material_key].append(obj)
        
        # 合并每组中的对象
        merged_count = 0
        for material_key, obj_group in material_groups.items():
            if len(obj_group) >= 2 and len(obj_group) <= self.optimization_settings["max_batched_objects"]:
                # 在实际实现中，这里会进行顶点数据合并
                print(f"将合并 {len(obj_group)} 个使用相同材质的对象")
                merged_count += len(obj_group) - 1  # 每组减少(len-1)个绘制调用
        
        print(f"材质批处理优化完成，估计减少 {merged_count} 个绘制调用")
    
    def _setup_instancing(self, scene):
        """
        设置实例化渲染
        
        Args:
            scene: 场景数据
        """
        # 按网格名称分组对象以进行实例化
        mesh_groups = {}
        
        for obj in scene["objects"]:
            if obj["type"] != "mesh":
                continue
                
            # 使用网格名称作为分组键
            mesh_name = obj["name"]
            
            if mesh_name not in mesh_groups:
                mesh_groups[mesh_name] = []
            
            mesh_groups[mesh_name].append(obj)
        
        # 为大型组设置实例化
        instanced_count = 0
        for mesh_name, obj_group in mesh_groups.items():
            if len(obj_group) >= self.optimization_settings["instance_threshold"]:
                # 标记对象为实例化
                for obj in obj_group:
                    obj["use_instancing"] = True
                
                instanced_count += len(obj_group)
                print(f"为 {mesh_name} 设置实例化渲染，实例数: {len(obj_group)}")
        
        print(f"实例化渲染设置完成，{instanced_count} 个对象将使用实例化渲染")
    
    def _optimize_lights(self, scene):
        """
        优化光源数量和质量
        
        Args:
            scene: 场景数据
        """
        lights = [obj for obj in scene["objects"] if obj["type"] == "light"]
        
        # 低端GPU限制阴影投射光源数量
        shadow_casting_lights = [l for l in lights if l.get("cast_shadows", False)]
        
        if self.gpu_type == "gtx750ti" and len(shadow_casting_lights) > 2:
            # 只保留最重要的2个阴影投射光源
            # 简单地按能量排序
            shadow_casting_lights.sort(key=lambda x: x.get("energy", 1.0), reverse=True)
            
            for i, light in enumerate(shadow_casting_lights):
                if i >= 2:
                    light["cast_shadows"] = False
                    print(f"禁用光源 {light['name']} 的阴影投射以优化性能")
        
        # 限制总光源数量
        if len(lights) > 8:
            # 按能量排序并降低次要光源的影响
            lights.sort(key=lambda x: x.get("energy", 1.0), reverse=True)
            
            for i, light in enumerate(lights):
                if i >= 8:
                    # 降低次要光源的能量
                    light["energy"] *= 0.5
                    print(f"降低光源 {light['name']} 的能量以优化性能")
    
    def _optimize_vram_usage(self, scene):
        """
        优化VRAM使用
        
        Args:
            scene: 场景数据
        """
        # 检查VRAM使用是否超过预算
        if self.current_vram_usage > self.vram_budget:
            over_budget = self.current_vram_usage - self.vram_budget
            reduction_target = over_budget * 1.2  # 多减20%作为缓冲
            
            print(f"VRAM使用超出预算 {over_budget:.2f}MB，需要减少 {reduction_target:.2f}MB")
            
            # 获取所有网格对象并按VRAM成本排序
            meshes = [obj for obj in scene["objects"] if obj["type"] == "mesh"]
            meshes.sort(key=lambda x: x.get("vram_cost", 0), reverse=True)
            
            # 降低大型对象的纹理分辨率
            reduced_vram = 0.0
            for mesh in meshes:
                if reduced_vram >= reduction_target:
                    break
                    
                # 降低材质纹理分辨率
                for material in mesh.get("materials", []):
                    if "textures" in material:
                        for tex_type, tex_info in material["textures"].items():
                            if "handle" in tex_info and tex_info["handle"] in self.asset_cache:
                                # 在实际实现中，这里会重新加载低分辨率纹理
                                # 这里简单模拟VRAM减少
                                original_size = tex_info.get("width", 1024) * tex_info.get("height", 1024)
                                new_size = int(original_size * 0.5)  # 降低50%分辨率
                                
                                format_bpp = self._get_format_bpp(tex_info.get("format", "RGBA8"))
                                saved_vram = (original_size - new_size) * format_bpp / (8 * 1024 * 1024)
                                
                                reduced_vram += saved_vram
                                self.current_vram_usage -= saved_vram
                                
                                # 更新纹理信息
                                tex_info["width"] = int(tex_info.get("width", 1024) * 0.707)  # sqrt(0.5)
                                tex_info["height"] = int(tex_info.get("height", 1024) * 0.707)
                                
                                print(f"降低 {mesh['name']} 的纹理分辨率，节省 {saved_vram:.2f}MB VRAM")
                                
                                if reduced_vram >= reduction_target:
                                    break
                        
                    if reduced_vram >= reduction_target:
                        break
    
    def prepare_for_rendering(self, scene):
        """
        为渲染准备场景数据
        
        Args:
            scene: 场景数据
            
        Returns:
            dict: 渲染就绪的场景数据
        """
        print("准备场景数据以供渲染...")
        
        # 将对象数据转换为渲染器可用的格式
        render_data = {
            "meshes": [],
            "lights": [],
            "cameras": [],
            "instanced_groups": []
        }
        
        # 处理网格
        for obj in scene["objects"]:
            if obj["type"] == "mesh":
                # 转换为渲染器格式
                mesh_data = self._convert_to_renderer_mesh(obj)
                
                if obj.get("use_instancing", False):
                    # 添加到实例化组
                    instance_group = self._find_or_create_instance_group(render_data["instanced_groups"], mesh_data["name"])
                    instance_group["instances"].append({
                        "transform": obj["transform"],
                        "visible": obj["transform"].get("visible", True)
                    })
                else:
                    # 直接添加
                    render_data["meshes"].append(mesh_data)
            
            elif obj["type"] == "light":
                # 转换为渲染器光源格式
                light_data = self._convert_to_renderer_light(obj)
                render_data["lights"].append(light_data)
            
            elif obj["type"] == "camera":
                # 转换为渲染器相机格式
                camera_data = self._convert_to_renderer_camera(obj)
                render_data["cameras"].append(camera_data)
        
        # 优化实例化组
        self._optimize_instance_groups(render_data["instanced_groups"])
        
        print(f"渲染准备完成: {len(render_data['meshes'])} 个网格, {len(render_data['lights'])} 个光源, {len(render_data['instanced_groups'])} 个实例化组")
        return render_data
    
    def _convert_to_renderer_mesh(self, mesh_obj):
        """
        将导入的网格转换为渲染器可用格式
        
        Args:
            mesh_obj: 导入的网格对象
            
        Returns:
            dict: 渲染器格式的网格数据
        """
        # 提取渲染所需的几何数据
        geometry = mesh_obj["geometry"]
        
        # 转换材质数据
        renderer_materials = []
        for material in mesh_obj.get("materials", []):
            renderer_material = {
                "name": material["name"],
                "shader_type": "standard",
                "parameters": {
                    "base_color": material.get("base_color", [1.0, 1.0, 1.0, 1.0]),
                    "metallic": material.get("metallic", 0.0),
                    "roughness": material.get("roughness", 0.5)
                },
                "textures": {}
            }
            
            # 添加纹理引用
            if "textures" in material:
                for tex_type, tex_info in material["textures"].items():
                    if "handle" in tex_info:
                        renderer_material["textures"][tex_type] = tex_info["handle"]
            
            renderer_materials.append(renderer_material)
        
        # 创建渲染网格数据
        renderer_mesh = {
            "name": mesh_obj["name"],
            "vertices": geometry["vertices"],
            "indices": [idx for face in geometry["faces"] for idx in face],
            "normals": geometry["normals"],
            "uvs": geometry["uvs"],
            "tangents": geometry["tangents"],
            "materials": renderer_materials,
            "transform": mesh_obj["transform"],
            "lods": mesh_obj.get("lods", [])
        }
        
        return renderer_mesh
    
    def _convert_to_renderer_light(self, light_obj):
        """
        将导入的光源转换为渲染器可用格式
        
        Args:
            light_obj: 导入的光源对象
            
        Returns:
            dict: 渲染器格式的光源数据
        """
        # 转换光源数据以匹配渲染器需求
        renderer_light = {
            "name": light_obj["name"],
            "type": light_obj["type"],
            "color": light_obj["color"],
            "intensity": light_obj["energy"],
            "position": light_obj["transform"]["location"],
            "rotation": light_obj["transform"]["rotation"],
            "cast_shadows": light_obj.get("cast_shadows", False),
            "shadow_map_size": light_obj.get("shadow_resolution", 1024)
        }
        
        # 添加特定类型参数
        if light_obj["type"] == "SUN":
            renderer_light["direction"] = self._euler_to_direction(light_obj["transform"]["rotation"])
            renderer_light["cascade_count"] = light_obj.get("shadow_cascade_count", 1)
        elif light_obj["type"] == "POINT":
            renderer_light["radius"] = light_obj.get("radius", 1.0)
            renderer_light["max_distance"] = light_obj.get("max_distance", 100.0)
        elif light_obj["type"] == "SPOT":
            renderer_light["direction"] = self._euler_to_direction(light_obj["transform"]["rotation"])
            renderer_light["spot_angle"] = light_obj.get("spot_size", 0.785)  # 默认45度
            renderer_light["spot_blend"] = light_obj.get("spot_blend", 0.1)
            renderer_light["max_distance"] = light_obj.get("max_distance", 100.0)
        
        return renderer_light
    
    def _convert_to_renderer_camera(self, camera_obj):
        """
        将导入的相机转换为渲染器可用格式
        
        Args:
            camera_obj: 导入的相机对象
            
        Returns:
            dict: 渲染器格式的相机数据
        """
        renderer_camera = {
            "name": camera_obj["name"],
            "type": camera_obj["type"],
            "position": camera_obj["transform"]["location"],
            "rotation": camera_obj["transform"]["rotation"],
            "fov": camera_obj["fov"],
            "depth_of_field": camera_obj.get("depth_of_field", False)
        }
        
        # 计算相机方向
        renderer_camera["direction"] = self._euler_to_direction(camera_obj["transform"]["rotation"])
        
        return renderer_camera
    
    def _euler_to_direction(self, euler_rotation):
        """
        将欧拉角转换为方向向量
        
        Args:
            euler_rotation: 欧拉角旋转 [x, y, z]
            
        Returns:
            list: 方向向量 [x, y, z]
        """
        # 简化的欧拉角到方向向量转换
        # 在实际实现中应使用完整的矩阵或四元数计算
        x, y, z = euler_rotation
        
        # 简单计算相机前向方向 (Z轴负方向)
        # 注意: 这里是简化实现，实际应使用完整的旋转矩阵
        direction = [
            -np.sin(y) * np.cos(x),
            np.sin(x),
            -np.cos(y) * np.cos(x)
        ]
        
        # 归一化
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = [d/norm for d in direction]
        
        return direction
    
    def _find_or_create_instance_group(self, instance_groups, mesh_name):
        """
        查找或创建实例化组
        
        Args:
            instance_groups: 现有实例化组列表
            mesh_name: 网格名称
            
        Returns:
            dict: 实例化组
        """
        for group in instance_groups:
            if group["mesh_name"] == mesh_name:
                return group
        
        # 创建新组
        new_group = {
            "mesh_name": mesh_name,
            "instances": []
        }
        instance_groups.append(new_group)
        
        return new_group
    
    def _optimize_instance_groups(self, instance_groups):
        """
        优化实例化组
        
        Args:
            instance_groups: 实例化组列表
        """
        # 对大型实例化组进行空间分区
        for group in instance_groups:
            if len(group["instances"]) > 100:
                # 在实际实现中，这里会进行空间分区（如网格或八叉树）
                print(f"对大型实例化组 {group['mesh_name']} ({len(group['instances'])} 个实例) 应用空间分区优化")

    def unload_unused_assets(self, distance_threshold=None):
        """
        卸载未使用的资产以释放VRAM
        
        Args:
            distance_threshold: 距离阈值，超过此距离的资产将被卸载
            
        Returns:
            int: 卸载的资产数量
        """
        # 在实际实现中，这里会根据相机距离和使用情况卸载资产
        print("资产卸载功能将在实际实现中实现")
        return 0

# 使用示例
"""
# 引擎中使用示例
from Engine.Renderer.Renderer import Renderer

# 创建渲染器实例
renderer = Renderer()

# 创建导入器
importer = BlenderImporter(renderer, gpu_type="gtx750ti")

# 加载场景
scene = importer.load_scene("path/to/scene_metadata.json")

# 准备渲染数据
render_data = importer.prepare_for_rendering(scene)

# 传递给渲染器
renderer.render_scene(render_data)
"""