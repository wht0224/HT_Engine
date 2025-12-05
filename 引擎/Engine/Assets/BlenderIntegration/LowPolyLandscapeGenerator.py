import bpy
import math
import random
import os

class LowPolyLandscapeGenerator:
    """
    低多边形风景生成器 - 专为低端GPU优化的纯代码风景建模工具
    不依赖外部导入，完全通过代码生成风景元素
    优化几何复杂度，适合引擎实时预览
    """
    
    def __init__(self, 
                 width=20, 
                 depth=20, 
                 height_scale=2.0, 
                 detail_level=1,  # 1-简单，2-中等，3-详细
                 optimize_for_engine=True):
        """
        初始化风景生成器
        
        Args:
            width: 地形宽度分段数
            depth: 地形深度分段数
            height_scale: 地形高度缩放因子
            detail_level: 细节级别
            optimize_for_engine: 是否针对引擎优化
        """
        self.width = width
        self.depth = depth
        self.height_scale = height_scale
        self.detail_level = detail_level
        self.optimize_for_engine = optimize_for_engine
        self.objects = []  # 存储生成的所有对象
        
        # 根据细节级别调整参数
        self._adjust_params_by_detail()
        
        print(f"初始化风景生成器 - 细节级别: {detail_level}, 优化引擎预览: {optimize_for_engine}")
    
    def _adjust_params_by_detail(self):
        """根据细节级别调整参数"""
        if self.detail_level == 1:  # 简单模式 - 最低多边形
            self.mountain_resolution = (16, 16)
            self.tree_count = 15
            self.rock_count = 8
            self.cloud_count = 3
        elif self.detail_level == 2:  # 中等模式
            self.mountain_resolution = (24, 24)
            self.tree_count = 25
            self.rock_count = 15
            self.cloud_count = 5
        else:  # 详细模式
            self.mountain_resolution = (32, 32)
            self.tree_count = 40
            self.rock_count = 25
            self.cloud_count = 7
    
    def clear_scene(self):
        """清空场景中的所有对象"""
        print("清理场景...")
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        self.objects = []
    
    def generate_terrain(self, name="Terrain"):
        """
        生成地形
        
        Args:
            name: 地形对象名称
            
        Returns:
            bpy.types.Object: 生成的地形对象
        """
        print("生成地形...")
        
        # 创建网格
        bpy.ops.mesh.primitive_grid_add(
            x_subdivisions=self.width,
            y_subdivisions=self.depth,
            size=20.0,
            location=(0, 0, 0)
        )
        
        terrain = bpy.context.active_object
        terrain.name = name
        
        # 添加噪波修改器来创建地形起伏
        noise_mod = terrain.modifiers.new(name="TerrainNoise", type='DISPLACE')
        
        # 创建噪波纹理
        noise_texture = bpy.data.textures.new(name="TerrainTexture", type='CLOUDS')
        noise_texture.noise_scale = 0.5
        noise_texture.noise_basis = 'BLENDER_ORIGINAL'
        noise_mod.texture = noise_texture
        noise_mod.strength = self.height_scale
        
        # 添加第二个噪波修改器创建细节
        if self.detail_level > 1:
            detail_mod = terrain.modifiers.new(name="DetailNoise", type='DISPLACE')
            detail_texture = bpy.data.textures.new(name="DetailTexture", type='CLOUDS')
            detail_texture.noise_scale = 2.0
            detail_texture.noise_basis = 'VORONOI_F1'
            detail_mod.texture = detail_texture
            detail_mod.strength = self.height_scale * 0.3
        
        # 应用修改器
        bpy.ops.object.convert(target='MESH')
        
        # 为引擎优化：合并顶点
        if self.optimize_for_engine:
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.remove_doubles(threshold=0.01)
            bpy.ops.object.mode_set(mode='OBJECT')
        
        # 计算法线
        bpy.ops.object.shade_smooth()
        
        # 添加材质
        self._create_terrain_material(terrain)
        
        self.objects.append(terrain)
        return terrain
    
    def _create_terrain_material(self, terrain):
        """为地形创建材质"""
        # 创建材质
        mat = bpy.data.materials.new(name="TerrainMaterial")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        # 清除默认节点
        for node in nodes:
            nodes.remove(node)
        
        # 创建输出节点
        output = nodes.new(type='ShaderNodeOutputMaterial')
        
        # 创建混合着色器（用于混合不同地形区域）
        mix_shader = nodes.new(type='ShaderNodeMixShader')
        links.new(mix_shader.outputs[0], output.inputs[0])
        
        # 基础材质（草地）
        principled1 = nodes.new(type='ShaderNodeBsdfPrincipled')
        principled1.inputs['Base Color'].default_value = (0.2, 0.5, 0.2, 1.0)
        principled1.inputs['Roughness'].default_value = 0.8
        links.new(principled1.outputs[0], mix_shader.inputs[1])
        
        # 次要材质（岩石/泥土）
        principled2 = nodes.new(type='ShaderNodeBsdfPrincipled')
        principled2.inputs['Base Color'].default_value = (0.5, 0.3, 0.2, 1.0)
        principled2.inputs['Roughness'].default_value = 0.9
        links.new(principled2.outputs[0], mix_shader.inputs[2])
        
        # 使用顶点高度作为混合因子
        separate_xyz = nodes.new(type='ShaderNodeSeparateXYZ')
        links.new(separate_xyz.outputs[2], mix_shader.inputs[0])  # Z轴作为混合因子
        
        # 创建几何节点获取顶点位置
        geometry = nodes.new(type='ShaderNodeNewGeometry')
        links.new(geometry.outputs[0], separate_xyz.inputs[0])
        
        # 应用材质
        if terrain.data.materials:
            terrain.data.materials[0] = mat
        else:
            terrain.data.materials.append(mat)
    
    def generate_trees(self, terrain=None):
        """
        在地形上生成树
        
        Args:
            terrain: 地形对象，如果为None则使用场景中的第一个地形
            
        Returns:
            list: 生成的树对象列表
        """
        print(f"生成 {self.tree_count} 棵树...")
        trees = []
        
        # 获取地形
        if not terrain:
            terrain = next((obj for obj in bpy.context.scene.objects if obj.name.startswith("Terrain")), None)
            if not terrain:
                print("未找到地形，无法生成树")
                return trees
        
        for i in range(self.tree_count):
            # 随机位置（在地形范围内）
            x = random.uniform(-10, 10)
            y = random.uniform(-10, 10)
            
            # 获取地形该位置的高度
            z = self._get_terrain_height(terrain, x, y)
            
            # 树的高度（根据位置略有变化，远处的树更小）
            distance = math.sqrt(x**2 + y**2)
            height = random.uniform(1.0, 1.5) * max(0.6, 1.0 - distance * 0.02)
            
            # 创建树干
            bpy.ops.mesh.primitive_cylinder_add(
                radius=0.1 * height,
                depth=height * 0.6,
                location=(x, y, z + height * 0.3)
            )
            trunk = bpy.context.active_object
            trunk.name = f"Tree_Trunk_{i}"
            
            # 创建树冠
            bpy.ops.mesh.primitive_cone_add(
                radius1=0.4 * height,
                depth=0.8 * height,
                location=(x, y, z + height * 0.6)
            )
            leaves = bpy.context.active_object
            leaves.name = f"Tree_Leaves_{i}"
            
            # 树干材质
            trunk_mat = bpy.data.materials.new(name=f"TrunkMaterial_{i}")
            trunk_mat.diffuse_color = (0.3, 0.2, 0.1, 1.0)
            trunk.data.materials.append(trunk_mat)
            
            # 树叶材质
            leaf_mat = bpy.data.materials.new(name=f"LeafMaterial_{i}")
            leaf_color = [0.1 + random.uniform(-0.05, 0.05), 
                         0.4 + random.uniform(-0.05, 0.05), 
                         0.1 + random.uniform(-0.05, 0.05), 1.0]
            leaf_mat.diffuse_color = leaf_color
            leaves.data.materials.append(leaf_mat)
            
            # 合并树干和树叶（简化引擎处理）
            if self.optimize_for_engine:
                bpy.ops.object.select_all(action='DESELECT')
                trunk.select_set(True)
                leaves.select_set(True)
                bpy.context.view_layer.objects.active = trunk
                bpy.ops.object.join()
                tree = trunk
            else:
                tree = leaves
            
            trees.append(tree)
            self.objects.append(tree)
        
        return trees
    
    def _get_terrain_height(self, terrain, x, y):
        """获取地形指定位置的高度"""
        # 简单的采样方法，实际项目中可能需要更精确的计算
        # 这里我们使用一个近似值
        mesh = terrain.data
        min_x, min_y = terrain.location.x - 10, terrain.location.y - 10
        max_x, max_y = terrain.location.x + 10, terrain.location.y + 10
        
        # 归一化坐标到0-1范围
        nx = (x - min_x) / (max_x - min_x)
        ny = (y - min_y) / (max_y - min_y)
        
        # 查找最近的顶点
        closest_z = 0
        min_distance = float('inf')
        
        for vertex in mesh.vertices:
            world_vertex = terrain.matrix_world @ vertex.co
            distance = math.sqrt((world_vertex.x - x)**2 + (world_vertex.y - y)** 2)
            if distance < min_distance:
                min_distance = distance
                closest_z = world_vertex.z
        
        return closest_z
    
    def generate_rocks(self, terrain=None):
        """
        在地形上生成岩石
        
        Args:
            terrain: 地形对象
            
        Returns:
            list: 生成的岩石对象列表
        """
        print(f"生成 {self.rock_count} 块岩石...")
        rocks = []
        
        # 获取地形
        if not terrain:
            terrain = next((obj for obj in bpy.context.scene.objects if obj.name.startswith("Terrain")), None)
            if not terrain:
                print("未找到地形，无法生成岩石")
                return rocks
        
        for i in range(self.rock_count):
            # 随机位置，倾向于在高处生成
            x = random.uniform(-9, 9)
            y = random.uniform(-9, 9)
            
            # 获取地形高度
            z = self._get_terrain_height(terrain, x, y)
            
            # 岩石大小
            size = random.uniform(0.3, 0.6)
            
            # 创建简单的低多边形岩石（使用立方体并变形）
            bpy.ops.mesh.primitive_cube_add(
                size=size,
                location=(x, y, z + size/2)
            )
            rock = bpy.context.active_object
            rock.name = f"Rock_{i}"
            
            # 添加轻微变形
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.transform.vertex_random(offset=0.2, uniform=0.1)
            bpy.ops.object.mode_set(mode='OBJECT')
            
            # 岩石材质
            rock_mat = bpy.data.materials.new(name=f"RockMaterial_{i}")
            rock_color = [0.3 + random.uniform(-0.1, 0.1), 
                         0.3 + random.uniform(-0.1, 0.1), 
                         0.3 + random.uniform(-0.1, 0.1), 1.0]
            rock_mat.diffuse_color = rock_color
            rock.data.materials.append(rock_mat)
            
            rocks.append(rock)
            self.objects.append(rock)
        
        return rocks
    
    def generate_lake(self, terrain=None):
        """
        在地形上生成湖泊
        
        Args:
            terrain: 地形对象
            
        Returns:
            bpy.types.Object: 生成的湖泊对象
        """
        print("生成湖泊...")
        
        # 获取地形
        if not terrain:
            terrain = next((obj for obj in bpy.context.scene.objects if obj.name.startswith("Terrain")), None)
            if not terrain:
                print("未找到地形，无法生成湖泊")
                return None
        
        # 创建湖泊平面
        lake_radius = 3.0
        bpy.ops.mesh.primitive_cylinder_add(
            radius=lake_radius,
            depth=0.1,
            location=(0, 0, 0.1)  # 稍微高于地面
        )
        lake = bpy.context.active_object
        lake.name = "Lake"
        
        # 创建湖水材质
        lake_mat = bpy.data.materials.new(name="LakeMaterial")
        lake_mat.use_nodes = True
        nodes = lake_mat.node_tree.nodes
        links = lake_mat.node_tree.links
        
        # 清除默认节点
        for node in nodes:
            nodes.remove(node)
        
        # 创建输出节点
        output = nodes.new(type='ShaderNodeOutputMaterial')
        
        # 创建透明材质
        principled = nodes.new(type='ShaderNodeBsdfPrincipled')
        principled.inputs['Base Color'].default_value = (0.1, 0.3, 0.5, 1.0)
        principled.inputs['Metallic'].default_value = 0.0
        principled.inputs['Roughness'].default_value = 0.1
        principled.inputs['Alpha'].default_value = 0.7
        
        # 设置材质为半透明
        lake_mat.blend_method = 'BLEND'
        
        links.new(principled.outputs[0], output.inputs[0])
        
        lake.data.materials.append(lake_mat)
        
        self.objects.append(lake)
        return lake
    
    def generate_clouds(self):
        """
        生成云朵
        
        Returns:
            list: 生成的云朵对象列表
        """
        print(f"生成 {self.cloud_count} 朵云...")
        clouds = []
        
        for i in range(self.cloud_count):
            # 随机位置
            x = random.uniform(-15, 15)
            y = random.uniform(-15, 15)
            z = random.uniform(3, 6)
            
            # 云的大小
            size = random.uniform(1.5, 3.0)
            
            # 创建云朵（使用多个球体组合）
            cloud_parts = []
            num_parts = random.randint(3, 5)
            
            for j in range(num_parts):
                part_x = x + random.uniform(-size*0.3, size*0.3)
                part_y = y + random.uniform(-size*0.3, size*0.3)
                part_z = z + random.uniform(-size*0.2, size*0.2)
                part_size = size * random.uniform(0.6, 0.9)
                
                bpy.ops.mesh.primitive_uv_sphere_add(
                    radius=part_size,
                    location=(part_x, part_y, part_z),
                    segments=8,
                    ring_count=4
                )
                part = bpy.context.active_object
                part.name = f"Cloud_Part_{i}_{j}"
                cloud_parts.append(part)
            
            # 合并云朵部分
            bpy.ops.object.select_all(action='DESELECT')
            for part in cloud_parts:
                part.select_set(True)
            
            if cloud_parts:
                bpy.context.view_layer.objects.active = cloud_parts[0]
                bpy.ops.object.join()
                cloud = cloud_parts[0]
                cloud.name = f"Cloud_{i}"
                
                # 云朵材质
                cloud_mat = bpy.data.materials.new(name=f"CloudMaterial_{i}")
                cloud_mat.use_nodes = True
                nodes = cloud_mat.node_tree.nodes
                links = cloud_mat.node_tree.links
                
                # 清除默认节点
                for node in nodes:
                    nodes.remove(node)
                
                # 创建输出节点
                output = nodes.new(type='ShaderNodeOutputMaterial')
                
                # 创建透明材质
                principled = nodes.new(type='ShaderNodeBsdfPrincipled')
                principled.inputs['Base Color'].default_value = (1.0, 1.0, 1.0, 1.0)
                principled.inputs['Metallic'].default_value = 0.0
                principled.inputs['Roughness'].default_value = 0.9
                principled.inputs['Alpha'].default_value = 0.9
                
                # 设置材质为半透明
                cloud_mat.blend_method = 'BLEND'
                
                links.new(principled.outputs[0], output.inputs[0])
                
                cloud.data.materials.append(cloud_mat)
                
                clouds.append(cloud)
                self.objects.append(cloud)
        
        return clouds
    
    def setup_camera(self):
        """设置相机"""
        print("设置相机...")
        
        # 创建相机
        bpy.ops.object.camera_add(
            location=(20, -20, 12),
            rotation=(math.radians(60), 0, math.radians(45))
        )
        camera = bpy.context.active_object
        camera.name = "LandscapeCamera"
        
        # 设置为活动相机
        bpy.context.scene.camera = camera
        
        self.objects.append(camera)
        return camera
    
    def setup_lighting(self):
        """设置光照"""
        print("设置光照...")
        
        # 创建主光源（阳光）
        bpy.ops.object.light_add(
            type='SUN',
            radius=1,
            location=(5, 5, 10)
        )
        sun = bpy.context.active_object
        sun.name = "SunLight"
        sun.data.energy = 2.0
        
        # 创建环境光
        bpy.ops.object.light_add(
            type='AREA',
            radius=1,
            location=(0, 0, 10)
        )
        ambient = bpy.context.active_object
        ambient.name = "AmbientLight"
        ambient.data.energy = 1.0
        
        self.objects.append(sun)
        self.objects.append(ambient)
        return sun, ambient
    
    def setup_sky(self):
        """设置天空环境"""
        print("设置天空环境...")
        
        # 创建世界环境
        world = bpy.context.scene.world
        if not world:
            world = bpy.data.worlds.new("World")
            bpy.context.scene.world = world
        
        world.use_nodes = True
        nodes = world.node_tree.nodes
        links = world.node_tree.links
        
        # 清除默认节点
        for node in nodes:
            nodes.remove(node)
        
        # 创建输出节点
        output = nodes.new(type='ShaderNodeOutputWorld')
        
        # 创建背景节点
        bg = nodes.new(type='ShaderNodeBackground')
        bg.inputs['Color'].default_value = (0.5, 0.7, 1.0, 1.0)  # 淡蓝色天空
        bg.inputs['Strength'].default_value = 1.0
        
        links.new(bg.outputs[0], output.inputs[0])
    
    def generate_complete_scene(self):
        """
        生成完整的风景场景
        
        Returns:
            list: 所有生成的对象
        """
        print("开始生成完整风景场景...")
        
        # 清空场景
        self.clear_scene()
        
        # 设置天空
        self.setup_sky()
        
        # 生成地形
        terrain = self.generate_terrain()
        
        # 生成湖泊
        lake = self.generate_lake(terrain)
        
        # 生成树木
        trees = self.generate_trees(terrain)
        
        # 生成岩石
        rocks = self.generate_rocks(terrain)
        
        # 生成云朵
        clouds = self.generate_clouds()
        
        # 设置光照
        sun, ambient = self.setup_lighting()
        
        # 设置相机
        camera = self.setup_camera()
        
        # 优化场景
        if self.optimize_for_engine:
            self._optimize_scene_for_engine()
        
        print(f"风景场景生成完成！总对象数: {len(self.objects)}")
        return self.objects
    
    def _optimize_scene_for_engine(self):
        """为引擎优化场景"""
        print("为引擎预览优化场景...")
        
        # 减少面数：进一步简化网格
        for obj in self.objects:
            if obj.type == 'MESH':
                # 添加简化修改器
                if len(obj.data.vertices) > 1000:
                    decimate = obj.modifiers.new(name="Decimate", type='DECIMATE')
                    decimate.ratio = min(1.0, 1000 / len(obj.data.vertices))
                    
                    # 应用修改器
                    bpy.context.view_layer.objects.active = obj
                    bpy.ops.object.convert(target='MESH')
        
        # 简化材质：合并相似材质
        self._merge_similar_materials()
        
        # 减少光源数量
        light_objects = [obj for obj in self.objects if obj.type == 'LIGHT']
        if len(light_objects) > 2:
            for light in light_objects[2:]:
                bpy.context.view_layer.objects.active = light
                bpy.ops.object.delete()
                self.objects.remove(light)
        
        print("引擎优化完成！")
    
    def _merge_similar_materials(self):
        """合并相似的材质"""
        # 收集所有材质
        materials = {}
        
        for obj in self.objects:
            if obj.type == 'MESH':
                for material_slot in obj.material_slots:
                    if material_slot.material:
                        # 简化材质类型
                        mat_type = "Unknown"
                        if "Trunk" in material_slot.material.name:
                            mat_type = "Trunk"
                        elif "Leaf" in material_slot.material.name:
                            mat_type = "Leaf"
                        elif "Rock" in material_slot.material.name:
                            mat_type = "Rock"
                        elif "Terrain" in material_slot.material.name:
                            mat_type = "Terrain"
                        elif "Lake" in material_slot.material.name:
                            mat_type = "Lake"
                        elif "Cloud" in material_slot.material.name:
                            mat_type = "Cloud"
                        
                        # 只保留每种类型的第一个材质
                        if mat_type not in materials:
                            materials[mat_type] = material_slot.material
                        else:
                            # 替换为通用材质
                            material_slot.material = materials[mat_type]
    
    def export_for_engine_preview(self, output_file=None):
        """
        导出场景数据用于引擎预览
        
        Args:
            output_file: 输出JSON文件路径，默认为None（只返回数据）
            
        Returns:
            dict: 场景数据字典，包含所有对象的几何信息和材质
        """
        print("开始导出引擎预览数据...")
        
        # 准备场景数据
        scene_data = {
            'metadata': {
                'version': 1.1,
                'detail_level': self.detail_level,
                'optimized_for_engine': self.optimize_for_engine,
                'object_count': len(self.objects)
            },
            'terrain': None,
            'vegetation': [],
            'props': [],
            'environment': []
        }
        
        # 遍历所有对象，分类导出
        for obj in self.objects:
            if obj.name.startswith("Terrain"):
                # 导出地形数据
                scene_data['terrain'] = self._export_mesh_data(obj, 'terrain')
            elif obj.name.startswith("Tree"):
                # 导出树木数据
                tree_data = self._export_mesh_data(obj, 'tree')
                scene_data['vegetation'].append(tree_data)
            elif obj.name.startswith("Rock"):
                # 导出岩石数据
                rock_data = self._export_mesh_data(obj, 'rock')
                scene_data['props'].append(rock_data)
            elif obj.name.startswith("Lake"):
                # 导出水体数据
                water_data = self._export_mesh_data(obj, 'water')
                scene_data['environment'].append(water_data)
            elif obj.name.startswith("Cloud"):
                # 导出云朵数据
                cloud_data = self._export_mesh_data(obj, 'cloud')
                scene_data['environment'].append(cloud_data)
        
        # 如果指定了输出文件，保存为JSON
        if output_file:
            import json
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            
            # 处理大数问题 - 将浮点数精度限制到合理范围
            def float_reducer(obj):
                if isinstance(obj, float):
                    return round(obj, 6)
                raise TypeError
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(scene_data, f, default=float_reducer, indent=2, ensure_ascii=False)
            
            print(f"数据已导出到: {output_file}")
        
        print(f"引擎预览数据导出完成！共导出 {len(self.objects)} 个对象")
        return scene_data
    
    def _export_mesh_data(self, obj, obj_type):
        """
        导出单个网格对象的数据
        
        Args:
            obj: Blender对象
            obj_type: 对象类型标识
            
        Returns:
            dict: 包含顶点、面和材质信息的对象数据
        """
        if obj.type != 'MESH':
            return None
        
        mesh = obj.data
        
        # 导出顶点数据
        vertices = []
        for v in mesh.vertices:
            # 转换到世界坐标系
            world_co = obj.matrix_world @ v.co
            vertices.extend([world_co.x, world_co.y, world_co.z])
        
        # 导出面索引
        faces = []
        for p in mesh.polygons:
            faces.extend(list(p.vertices))
        
        # 导出UV坐标（如果有）
        uvs = []
        if len(mesh.uv_layers) > 0:
            uv_layer = mesh.uv_layers.active.data
            for loop in mesh.loops:
                uv = uv_layer[loop.index].uv
                uvs.extend([uv.x, uv.y])
        
        # 导出材质信息
        material_data = {}
        if len(mesh.materials) > 0 and mesh.materials[0]:
            mat = mesh.materials[0]
            material_data = self._export_material_data(mat)
        
        # 构建对象数据
        obj_data = {
            'name': obj.name,
            'type': obj_type,
            'vertices': vertices,
            'faces': faces,
            'uvs': uvs if uvs else None,
            'material': material_data,
            'position': [obj.location.x, obj.location.y, obj.location.z],
            'rotation': [obj.rotation_euler.x, obj.rotation_euler.y, obj.rotation_euler.z],
            'scale': [obj.scale.x, obj.scale.y, obj.scale.z]
        }
        
        # 为低端GPU优化的特殊标记
        if self.optimize_for_engine:
            # 标记需要实例化的对象（如树和岩石）
            if obj_type in ['tree', 'rock']:
                obj_data['use_instancing'] = True
            
            # 标记需要简化的复杂对象
            if len(vertices) > 1000:
                obj_data['needs_simplification'] = True
        
        return obj_data
    
    def _export_material_data(self, material):
        """
        导出材质数据
        
        Args:
            material: Blender材质对象
            
        Returns:
            dict: 材质数据字典
        """
        material_data = {
            'name': material.name,
            'type': 'principled',
            'base_color': [0.8, 0.8, 0.8, 1.0],
            'roughness': 0.5,
            'metallic': 0.0,
            'alpha': 1.0
        }
        
        # 如果是节点材质，尝试提取参数
        if material.use_nodes:
            for node in material.node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    # 提取基础颜色
                    if 'Base Color' in node.inputs:
                        color = node.inputs['Base Color'].default_value
                        material_data['base_color'] = [color[0], color[1], color[2], color[3]]
                    
                    # 提取粗糙度
                    if 'Roughness' in node.inputs:
                        material_data['roughness'] = node.inputs['Roughness'].default_value
                    
                    # 提取金属度
                    if 'Metallic' in node.inputs:
                        material_data['metallic'] = node.inputs['Metallic'].default_value
                    
                    # 提取透明度
                    if 'Alpha' in node.inputs:
                        material_data['alpha'] = node.inputs['Alpha'].default_value
        else:
            # 旧式材质系统
            color = material.diffuse_color
            material_data['base_color'] = [color[0], color[1], color[2], 1.0]
            material_data['roughness'] = 1.0 - material.specular_intensity
        
        # 添加透明标记
        if material_data['alpha'] < 1.0 or material.blend_method != 'OPAQUE':
            material_data['transparent'] = True
        
        return material_data
    
    def preview_in_engine(self, engine_address="localhost", engine_port=5000):
        """
        通过网络直接预览到引擎
        
        Args:
            engine_address: 引擎地址
            engine_port: 引擎端口
            
        Returns:
            bool: 是否发送成功
        """
        print(f"准备发送数据到引擎 ({engine_address}:{engine_port})...")
        
        try:
            import socket
            import json
            
            # 获取场景数据
            scene_data = self.export_for_engine_preview()
            
            # 将数据转换为JSON
            def float_reducer(obj):
                if isinstance(obj, float):
                    return round(obj, 6)
                raise TypeError
            
            json_data = json.dumps(scene_data, default=float_reducer)
            
            # 连接到引擎并发送数据
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(5.0)
                sock.connect((engine_address, engine_port))
                
                # 发送数据长度
                data_length = len(json_data)
                sock.sendall(data_length.to_bytes(4, 'big'))
                
                # 分块发送数据
                chunk_size = 4096
                for i in range(0, data_length, chunk_size):
                    chunk = json_data[i:i+chunk_size]
                    sock.sendall(chunk.encode('utf-8'))
                
                # 等待确认
                response = sock.recv(1024).decode('utf-8')
                if response == "OK":
                    print("场景数据成功发送到引擎！")
                    return True
                else:
                    print(f"引擎响应: {response}")
                    return False
                    
        except Exception as e:
            print(f"发送数据到引擎失败: {str(e)}")
            return False
    
    def auto_preview(self):
        """
        自动生成场景并预览到引擎
        这是一个快捷方法，用于一键生成并预览
        """
        print("==================================")
        print("自动风景生成与引擎预览")
        print("==================================")
        
        # 清理场景
        self.clear_scene()
        
        # 生成完整场景
        self.generate_complete_scene()
        
        # 优化场景（如果需要）
        if self.optimize_for_engine:
            self._optimize_scene_for_engine()
        
        # 尝试直接预览到引擎
        success = self.preview_in_engine()
        
        if not success:
            # 备选方案：导出为文件
            export_path = os.path.join(os.path.dirname(bpy.data.filepath or ""), "engine_preview.json")
            if not bpy.data.filepath:
                export_path = os.path.join(os.path.expanduser("~"), "engine_preview.json")
            
            self.export_for_engine_preview(export_path)
            print(f"\n提示: 引擎预览服务未启动，已导出数据到 {export_path}")
        
        print("\n==================================")
        print("操作完成！")
        print("==================================")
        return success

def main():
    """
    主函数 - 生成低多边形风景场景
    可以在Blender脚本编辑器中直接运行
    """
    print("==================================")
    print("低多边形风景生成器 - 引擎预览版")
    print("==================================")
    
    # 创建生成器实例（中等细节，针对引擎优化）
    generator = LowPolyLandscapeGenerator(
        width=24,
        depth=24,
        height_scale=2.5,
        detail_level=2,
        optimize_for_engine=True
    )
    
    # 生成完整场景
    generator.generate_complete_scene()
    
    # 准备引擎预览数据
    scene_data = generator.export_for_engine_preview()
    
    print("\n==================================")
    print("风景生成完成！")
    print("==================================")
    print("提示:")
    print("1. 您可以在Blender中预览场景")
    print("2. 场景数据已准备好，可以直接在引擎中使用")
    print("3. 所有对象都已针对低端GPU进行优化")
    print("4. 按F12可以渲染预览图像")
    print("==================================")

if __name__ == "__main__":
    # 直接在Blender中运行
    main()

# 以下是引擎预览集成的示例代码（在Blender外部使用）
"""
# 如何在引擎中使用：
# 1. 在Blender中运行此脚本生成场景
# 2. 在引擎代码中，可以通过以下方式访问场景数据：

# from Engine.Assets.BlenderIntegration.LowPolyLandscapeGenerator import LowPolyLandscapeGenerator

# def load_landscape_into_engine():
#     # 创建生成器（不需要实际生成，只用于数据结构）
#     generator = LowPolyLandscapeGenerator(optimize_for_engine=True)
#     
#     # 从Blender中获取场景数据（这里假设有一种机制可以获取）
#     # 实际项目中，可能需要通过MCP插件或其他方式传输数据
#     scene_data = get_scene_data_from_blender()
#     
#     # 使用引擎的API加载场景
#     engine_scene = engine.create_scene()
#     
#     # 加载对象
#     for obj_data in scene_data["objects"]:
#         if "mesh" in obj_data:
#             mesh = obj_data["mesh"]
#             # 创建引擎中的网格对象
#             engine_mesh = engine.create_mesh(
#                 vertices=mesh["vertices"],
#                 faces=mesh["faces"],
#                 normals=mesh["normals"]
#             )
#             
#             # 创建引擎中的实体
#             entity = engine_scene.create_entity(obj_data["name"])
#             entity.add_component("MeshRenderer", mesh=engine_mesh)
#             entity.transform.position = obj_data["location"]
#             entity.transform.rotation = obj_data["rotation"]
#             entity.transform.scale = obj_data["scale"]
#     
#     # 设置相机
#     if scene_data["camera"]:
#         camera_data = scene_data["camera"]
#         camera = engine_scene.create_camera()
#         camera.transform.position = camera_data["location"]
#         camera.transform.rotation = camera_data["rotation"]
#     
#     # 设置光源
#     for light_data in scene_data["lights"]:
#         light = engine_scene.create_light(light_data["type"])
#         light.transform.position = light_data["location"]
#         light.transform.rotation = light_data["rotation"]
#         light.energy = light_data["energy"]

# # 调用函数加载场景
# load_landscape_into_engine()
"""