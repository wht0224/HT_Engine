import bpy
import os
import json
import numpy as np
from pathlib import Path

"""
Blender MCP导出器 - 针对低端GPU优化的资产导出管道

此模块提供从Blender到渲染引擎的高效资产导出功能，特别针对GTX 750Ti和RX 580等低端GPU进行了优化。
实现了多边形优化、纹理压缩、LOD生成等核心功能。
"""

class BlenderExporter:
    """Blender资产导出器主类"""
    
    def __init__(self, output_dir=None, gpu_target="gtx750ti"):
        """
        初始化导出器
        
        Args:
            output_dir: 输出目录路径
            gpu_target: 目标GPU类型 ("gtx750ti" 或 "rx580")
        """
        self.output_dir = output_dir or os.path.join(os.path.dirname(bpy.data.filepath), "exports")
        self.gpu_target = gpu_target
        self.export_settings = self._get_gpu_optimized_settings()
        self.exported_assets = []
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"初始化Blender导出器，目标GPU: {gpu_target}")
    
    def _get_gpu_optimized_settings(self):
        """
        获取针对特定GPU优化的导出设置
        
        Returns:
            dict: 优化后的导出设置
        """
        base_settings = {
            # 通用基础设置
            "max_texture_size": 1024,
            "normal_map_format": "BC5",
            "color_map_format": "BC7",
            "max_lod_levels": 4,
            "texture_atlas_size": 2048,
            "enable_batching": True,
            "export_animation": True,
        }
        
        # 根据目标GPU调整参数
        if self.gpu_target == "gtx750ti":
            # Maxwell架构优化 (GTX 750Ti)
            return {
                **base_settings,
                "max_texture_size": 512,  # 更小的纹理以适应2GB VRAM
                "max_vertices": 50000,    # 每个网格的最大顶点数
                "shader_complexity": "low", # 简单着色器
                "shadow_resolution": 1024,
                "enable_ao_bake": False,  # 禁用环境光遮蔽烘焙以节省时间
                "max_bones_per_model": 32, # 限制骨骼数量
            }
        elif self.gpu_target == "rx580":
            # GCN架构优化 (RX 580)
            return {
                **base_settings,
                "max_texture_size": 1024,
                "max_vertices": 100000,
                "shader_complexity": "medium",
                "shadow_resolution": 2048,
                "enable_ao_bake": True,
                "max_bones_per_model": 48,
            }
        else:
            return base_settings
    
    def export_scene(self, export_selected=False):
        """
        导出整个场景或选定对象
        
        Args:
            export_selected: 仅导出选定对象
            
        Returns:
            dict: 导出结果信息
        """
        print(f"开始导出场景... GPU优化: {self.gpu_target}")
        
        # 选择要导出的对象
        objects = bpy.context.selected_objects if export_selected else bpy.data.objects
        
        # 按类型分组对象
        meshes = [obj for obj in objects if obj.type == 'MESH']
        lights = [obj for obj in objects if obj.type == 'LIGHT']
        cameras = [obj for obj in objects if obj.type == 'CAMERA']
        
        # 导出网格
        for mesh_obj in meshes:
            self._export_mesh(mesh_obj)
        
        # 导出光源
        for light_obj in lights:
            self._export_light(light_obj)
        
        # 导出相机
        for camera_obj in cameras:
            self._export_camera(camera_obj)
        
        # 生成场景元数据
        scene_metadata = self._generate_scene_metadata()
        
        # 保存场景元数据
        with open(os.path.join(self.output_dir, "scene_metadata.json"), "w") as f:
            json.dump(scene_metadata, f, indent=2)
        
        print(f"场景导出完成。共导出 {len(self.exported_assets)} 个资产。")
        return {
            "output_dir": self.output_dir,
            "exported_assets": self.exported_assets,
            "settings": self.export_settings
        }
    
    def _export_mesh(self, obj):
        """
        导出网格对象，包括几何数据、材质、UV和LOD
        
        Args:
            obj: Blender网格对象
        """
        print(f"导出网格: {obj.name}")
        
        # 获取网格数据
        mesh = obj.data
        
        # 创建网格导出目录
        mesh_dir = os.path.join(self.output_dir, "meshes", obj.name)
        os.makedirs(mesh_dir, exist_ok=True)
        
        # 优化网格
        optimized_mesh = self._optimize_mesh(obj)
        
        # 导出几何数据
        geometry_data = self._extract_geometry(optimized_mesh)
        with open(os.path.join(mesh_dir, "geometry.json"), "w") as f:
            json.dump(geometry_data, f, indent=2)
        
        # 生成LOD级别
        self._generate_lods(obj, mesh_dir)
        
        # 导出材质
        if obj.material_slots:
            material_data = self._export_materials(obj)
            with open(os.path.join(mesh_dir, "materials.json"), "w") as f:
                json.dump(material_data, f, indent=2)
        
        # 导出变换信息
        transform_data = self._export_transform(obj)
        with open(os.path.join(mesh_dir, "transform.json"), "w") as f:
            json.dump(transform_data, f, indent=2)
        
        # 添加到导出资产列表
        self.exported_assets.append({
            "type": "mesh",
            "name": obj.name,
            "path": os.path.relpath(mesh_dir, self.output_dir)
        })
    
    def _optimize_mesh(self, obj):
        """
        优化网格以适应低端GPU
        
        Args:
            obj: Blender对象
            
        Returns:
            bpy.types.Mesh: 优化后的网格
        """
        # 复制原始对象以避免修改原始数据
        obj_copy = obj.copy()
        obj_copy.data = obj.data.copy()
        
        # 进入编辑模式
        bpy.context.view_layer.objects.active = obj_copy
        bpy.ops.object.mode_set(mode='EDIT')
        
        # 合并顶点
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.remove_doubles(threshold=0.0001)
        
        # 根据GPU目标设置最大顶点数
        max_vertices = self.export_settings["max_vertices"]
        current_vertices = len(obj_copy.data.vertices)
        
        # 如果顶点数超过限制，执行简化
        if current_vertices > max_vertices:
            decimation_ratio = max_vertices / current_vertices
            bpy.ops.object.mode_set(mode='OBJECT')
            modifier = obj_copy.modifiers.new(name="Decimate", type='DECIMATE')
            modifier.ratio = decimation_ratio
            bpy.context.view_layer.objects.active = obj_copy
            bpy.ops.object.modifier_apply(modifier="Decimate")
        
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # 回到原始对象
        bpy.context.view_layer.objects.active = obj
        
        print(f"网格优化完成: {obj.name}, 顶点数: {len(obj_copy.data.vertices)}")
        return obj_copy.data
    
    def _generate_lods(self, obj, output_dir):
        """
        为网格生成多级LOD
        
        Args:
            obj: Blender对象
            output_dir: 输出目录
        """
        lod_levels = self.export_settings["max_lod_levels"]
        print(f"为 {obj.name} 生成 {lod_levels} 级LOD")
        
        # 为每个LOD级别创建简化网格
        for lod_level in range(1, lod_levels + 1):
            # 计算简化比例 (LOD级别越高，简化越多)
            ratio = 1.0 - (lod_level * 0.2)  # 每级减少20%
            if ratio < 0.1:  # 最低保留10%
                ratio = 0.1
            
            # 创建LOD对象副本
            lod_obj = obj.copy()
            lod_obj.data = obj.data.copy()
            
            # 添加简化修改器
            modifier = lod_obj.modifiers.new(name=f"Decimate_LOD{lod_level}", type='DECIMATE')
            modifier.ratio = ratio
            
            # 应用修改器
            bpy.context.view_layer.objects.active = lod_obj
            bpy.ops.object.modifier_apply(modifier=f"Decimate_LOD{lod_level}")
            
            # 提取几何数据
            geometry_data = self._extract_geometry(lod_obj.data)
            
            # 保存LOD数据
            lod_dir = os.path.join(output_dir, f"lod{lod_level}")
            os.makedirs(lod_dir, exist_ok=True)
            
            with open(os.path.join(lod_dir, "geometry.json"), "w") as f:
                json.dump(geometry_data, f, indent=2)
            
            print(f"LOD{lod_level} 生成完成，简化比例: {ratio:.2f}, 顶点数: {len(lod_obj.data.vertices)}")
    
    def _extract_geometry(self, mesh):
        """
        提取网格的几何数据
        
        Args:
            mesh: Blender网格
            
        Returns:
            dict: 几何数据字典
        """
        # 获取顶点坐标
        vertices = [list(v.co) for v in mesh.vertices]
        
        # 获取面索引
        faces = []
        for face in mesh.polygons:
            face_indices = []
            for loop_idx in face.loop_indices:
                face_indices.append(mesh.loops[loop_idx].vertex_index)
            faces.append(face_indices)
        
        # 获取法线
        mesh.calc_normals()
        normals = [list(v.normal) for v in mesh.vertices]
        
        # 获取UV坐标 (如果存在)
        uvs = []
        if mesh.uv_layers:
            uv_layer = mesh.uv_layers.active.data
            for loop in mesh.loops:
                uvs.append([uv_layer[loop.index].uv[0], 1.0 - uv_layer[loop.index].uv[1]])  # 翻转V坐标
        
        # 获取切线和副法线 (用于法线贴图)
        mesh.calc_tangents()
        tangents = [list(loop.tangent) for loop in mesh.loops]
        
        return {
            "vertices": vertices,
            "faces": faces,
            "normals": normals,
            "uvs": uvs,
            "tangents": tangents,
            "vertex_count": len(vertices),
            "face_count": len(faces)
        }
    
    def _export_materials(self, obj):
        """
        导出材质信息，包括纹理引用和着色器参数
        
        Args:
            obj: 包含材质的Blender对象
            
        Returns:
            dict: 材质数据字典
        """
        materials = []
        
        for material_slot in obj.material_slots:
            if not material_slot.material:
                continue
                
            material = material_slot.material
            mat_data = {
                "name": material.name,
                "type": "standard"  # 默认材质类型
            }
            
            # 如果使用节点材质
            if material.use_nodes:
                nodes = material.node_tree.nodes
                links = material.node_tree.links
                
                # 查找材质输出节点
                output_node = None
                for node in nodes:
                    if node.type == 'OUTPUT_MATERIAL':
                        output_node = node
                        break
                
                # 查找BSDF节点
                bsdf_node = None
                if output_node and output_node.inputs['Surface'].links:
                    bsdf_node = output_node.inputs['Surface'].links[0].from_node
                
                # 提取基本材质参数
                if bsdf_node and bsdf_node.type == 'BSDF_PRINCIPLED':
                    # 提取颜色、金属度、粗糙度等参数
                    mat_data["base_color"] = list(bsdf_node.inputs['Base Color'].default_value)
                    mat_data["metallic"] = bsdf_node.inputs['Metallic'].default_value
                    mat_data["roughness"] = bsdf_node.inputs['Roughness'].default_value
                    mat_data["normal_strength"] = bsdf_node.inputs['Normal'].default_value
                    
                    # 提取纹理引用
                    textures = {}
                    
                    # 检查基础颜色纹理
                    if bsdf_node.inputs['Base Color'].links:
                        texture_node = bsdf_node.inputs['Base Color'].links[0].from_node
                        if texture_node.type == 'TEX_IMAGE' and texture_node.image:
                            textures["base_color"] = self._export_texture(texture_node.image)
                    
                    # 检查法线贴图
                    if bsdf_node.inputs['Normal'].links:
                        normal_link = bsdf_node.inputs['Normal'].links[0]
                        if normal_link.from_node.type == 'NORMAL_MAP':
                            normal_map_node = normal_link.from_node
                            if normal_map_node.inputs['Color'].links:
                                texture_node = normal_map_node.inputs['Color'].links[0].from_node
                                if texture_node.type == 'TEX_IMAGE' and texture_node.image:
                                    textures["normal"] = self._export_texture(texture_node.image)
                    
                    # 检查金属度/粗糙度纹理
                    if bsdf_node.inputs['Metallic'].links:
                        texture_node = bsdf_node.inputs['Metallic'].links[0].from_node
                        if texture_node.type == 'TEX_IMAGE' and texture_node.image:
                            textures["metallic"] = self._export_texture(texture_node.image)
                    
                    if bsdf_node.inputs['Roughness'].links:
                        texture_node = bsdf_node.inputs['Roughness'].links[0].from_node
                        if texture_node.type == 'TEX_IMAGE' and texture_node.image:
                            textures["roughness"] = self._export_texture(texture_node.image)
                    
                    mat_data["textures"] = textures
            
            materials.append(mat_data)
        
        return materials
    
    def _export_texture(self, image):
        """
        导出和优化纹理
        
        Args:
            image: Blender图像对象
            
        Returns:
            dict: 纹理信息
        """
        # 确保图像已保存
        if not image.filepath:
            # 为未保存的图像创建临时路径
            temp_name = f"temp_{image.name}.png"
            temp_path = os.path.join(self.output_dir, "textures", temp_name)
            os.makedirs(os.path.join(self.output_dir, "textures"), exist_ok=True)
            
            # 保存图像
            image.filepath_raw = temp_path
            image.file_format = 'PNG'
            image.save()
            filepath = temp_path
        else:
            filepath = bpy.path.abspath(image.filepath)
        
        # 优化纹理大小
        max_size = self.export_settings["max_texture_size"]
        
        # 确定纹理格式
        if "normal" in image.name.lower():
            format_type = self.export_settings["normal_map_format"]
        else:
            format_type = self.export_settings["color_map_format"]
        
        return {
            "file_path": os.path.relpath(filepath, self.output_dir),
            "width": min(image.size[0], max_size),
            "height": min(image.size[1], max_size),
            "format": format_type,
            "needs_compression": True
        }
    
    def _export_transform(self, obj):
        """
        导出对象的变换信息
        
        Args:
            obj: Blender对象
            
        Returns:
            dict: 变换数据
        """
        return {
            "location": list(obj.location),
            "rotation": list(obj.rotation_euler),
            "scale": list(obj.scale),
            "name": obj.name,
            "visible": obj.visible_get()
        }
    
    def _export_light(self, obj):
        """
        导出光源信息
        
        Args:
            obj: Blender光源对象
        """
        print(f"导出光源: {obj.name}")
        
        light = obj.data
        
        # 创建光源导出目录
        lights_dir = os.path.join(self.output_dir, "lights")
        os.makedirs(lights_dir, exist_ok=True)
        
        # 根据GPU目标优化光源设置
        shadow_resolution = self.export_settings["shadow_resolution"]
        
        # 光源数据
        light_data = {
            "name": obj.name,
            "type": light.type,
            "color": list(light.color),
            "energy": light.energy,
            "transform": self._export_transform(obj)
        }
        
        # 特定类型光源参数
        if light.type == 'SUN':
            light_data["angle"] = light.angle
            light_data["shadow_cascade_count"] = 1  # 低端GPU只使用1级级联阴影
        elif light.type == 'POINT':
            light_data["radius"] = light.shadow_soft_size
            light_data["max_distance"] = light.cutoff_distance if light.use_custom_distance else 100.0
        elif light.type == 'SPOT':
            light_data["spot_size"] = light.spot_size
            light_data["spot_blend"] = light.spot_blend
            light_data["max_distance"] = light.cutoff_distance if light.use_custom_distance else 100.0
        
        # 阴影设置
        light_data["cast_shadows"] = light.cast_shadow
        light_data["shadow_resolution"] = shadow_resolution
        
        # 保存光源数据
        light_file = os.path.join(lights_dir, f"{obj.name}.json")
        with open(light_file, "w") as f:
            json.dump(light_data, f, indent=2)
        
        # 添加到导出资产列表
        self.exported_assets.append({
            "type": "light",
            "name": obj.name,
            "path": os.path.relpath(light_file, self.output_dir)
        })
    
    def _export_camera(self, obj):
        """
        导出相机信息
        
        Args:
            obj: Blender相机对象
        """
        print(f"导出相机: {obj.name}")
        
        camera = obj.data
        
        # 创建相机导出目录
        cameras_dir = os.path.join(self.output_dir, "cameras")
        os.makedirs(cameras_dir, exist_ok=True)
        
        # 相机数据
        camera_data = {
            "name": obj.name,
            "type": "perspective" if camera.type == 'PERSP' else "orthographic",
            "fov": camera.angle,
            "depth_of_field": camera.dof.use_dof,
            "transform": self._export_transform(obj)
        }
        
        # 保存相机数据
        camera_file = os.path.join(cameras_dir, f"{obj.name}.json")
        with open(camera_file, "w") as f:
            json.dump(camera_data, f, indent=2)
        
        # 添加到导出资产列表
        self.exported_assets.append({
            "type": "camera",
            "name": obj.name,
            "path": os.path.relpath(camera_file, self.output_dir)
        })
    
    def _generate_scene_metadata(self):
        """
        生成场景元数据
        
        Returns:
            dict: 场景元数据
        """
        scene = bpy.context.scene
        
        metadata = {
            "name": scene.name,
            "world_properties": {
                "ambient_color": list(scene.world.ambient_color) if scene.world else [0, 0, 0],
                "horizon_color": list(scene.world.color) if scene.world else [0, 0, 0]
            },
            "export_settings": self.export_settings,
            "exported_assets": self.exported_assets,
            "date": str(bpy.context.scene.frame_current)
        }
        
        return metadata
    
    def compress_textures(self):
        """
        压缩所有导出的纹理以优化内存使用
        
        Returns:
            int: 压缩的纹理数量
        """
        # 此函数将在实际实现中调用外部工具进行纹理压缩
        # 例如使用DirectXTex、PVRTexTool或Compressonator
        print("纹理压缩功能将在实际实现中调用外部工具")
        return 0
    
    def generate_atlases(self):
        """
        生成纹理图集以减少绘制调用
        
        Returns:
            dict: 图集信息
        """
        # 此函数将在实际实现中实现纹理图集生成
        print("纹理图集生成功能将在实际实现中实现")
        return {"atlases": []}

def register():
    """注册Blender插件"""
    print("Blender MCP导出器插件已注册")

def unregister():
    """注销Blender插件"""
    print("Blender MCP导出器插件已注销")

if __name__ == "__main__":
    # 直接运行时的测试代码
    exporter = BlenderExporter(gpu_target="gtx750ti")
    result = exporter.export_scene()
    print(f"导出结果: {result}")