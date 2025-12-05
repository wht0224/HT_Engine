import bpy
import os
import math

class LowEndModeling:
    """
    针对低端GPU（如GTX 750Ti和RX 580）优化的Blender建模工具
    提供低多边形建模、材质优化和渲染设置功能
    """
    
    def __init__(self):
        self.optimization_level = 1  # 1-3，1为最高优化（最少资源）
        self.target_resolution = (1280, 720)  # 目标渲染分辨率
        self.max_polygon_count = 5000  # 最大多边形数量
        print("低端GPU优化建模工具已初始化")
    
    def clear_scene(self):
        """清理场景，保留相机和灯光"""
        print("清理场景中...")
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.context.scene.objects:
            if obj.type not in ['CAMERA', 'LIGHT']:
                obj.select_set(True)
        bpy.ops.object.delete()
        print("场景清理完成")
    
    def create_low_poly_model(self, model_type="cube"):
        """创建基本的低多边形模型"""
        print(f"创建低多边形{model_type}模型")
        
        if model_type == "cube":
            bpy.ops.mesh.primitive_cube_add(size=2)
            cube = bpy.context.active_object
            # 简化立方体，虽然已经很简单了
            self.optimize_mesh(cube)
            return cube
            
        elif model_type == "sphere":
            bpy.ops.mesh.primitive_uv_sphere_add(radius=1)
            sphere = bpy.context.active_object
            # 根据优化级别调整细分
            segments = 16 if self.optimization_level == 3 else 8
            rings = 12 if self.optimization_level == 3 else 6
            bpy.context.object.data.vertices_remove_by_distance(distance=0.01)
            return sphere
            
        elif model_type == "environment":
            # 创建简单的环境
            # 地面
            bpy.ops.mesh.primitive_plane_add(size=10)
            ground = bpy.context.active_object
            ground.name = "Ground"
            
            # 几个简单的几何体
            bpy.ops.mesh.primitive_cube_add(size=1, location=(2, 2, 0.5))
            cube1 = bpy.context.active_object
            
            bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=1, location=(-2, 2, 0.5))
            cylinder1 = bpy.context.active_object
            
            # 优化所有对象
            self.optimize_mesh(ground)
            self.optimize_mesh(cube1)
            self.optimize_mesh(cylinder1)
            
            return [ground, cube1, cylinder1]
    
    def optimize_mesh(self, obj, target_polygons=1000):
        """优化网格，减少多边形数量"""
        if obj.type != 'MESH':
            return
            
        # 选择对象
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        
        # 进入编辑模式
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        
        # 应用简化修改器
        decimate_ratio = max(0.1, target_polygons / len(obj.data.polygons))
        bpy.ops.mesh.decimate(ratio=decimate_ratio)
        
        # 合并顶点
        bpy.ops.mesh.remove_doubles(threshold=0.001)
        
        # 返回物体模式
        bpy.ops.object.mode_set(mode='OBJECT')
        
        print(f"优化后的多边形数量: {len(obj.data.polygons)}")
    
    def create_optimized_material(self, obj, base_color=(0.8, 0.2, 0.2, 1.0), roughness=0.5):
        """为对象创建优化的材质"""
        # 创建材质
        mat_name = f"{obj.name}_Material"
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        
        # 简化材质节点
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        # 清除默认节点
        for node in nodes:
            nodes.remove(node)
        
        # 添加必要的节点
        output = nodes.new(type='ShaderNodeOutputMaterial')
        principled = nodes.new(type='ShaderNodeBsdfPrincipled')
        
        # 设置材质参数
        principled.inputs["Base Color"].default_value = base_color
        principled.inputs["Roughness"].default_value = roughness
        
        # 连接节点
        links.new(principled.outputs["BSDF"], output.inputs["Surface"])
        
        # 应用材质
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)
        
        print(f"已为 {obj.name} 应用优化材质")
    
    def setup_optimized_lighting(self):
        """设置优化的光照系统，减少GPU负担"""
        print("设置优化的光照")
        
        # 删除现有灯光
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.context.scene.objects:
            if obj.type == 'LIGHT':
                obj.select_set(True)
        bpy.ops.object.delete()
        
        # 添加主光源（关键光）
        bpy.ops.object.light_add(type='SUN', radius=1, location=(5, -5, 10))
        key_light = bpy.context.active_object
        key_light.name = "KeyLight"
        key_light.data.energy = 2.0
        key_light.data.color = (1.0, 0.95, 0.9)
        
        # 添加环境光遮蔽辅助光源
        bpy.ops.object.light_add(type='AREA', radius=1, location=(-3, 3, 2))
        fill_light = bpy.context.active_object
        fill_light.name = "FillLight"
        fill_light.data.energy = 0.5
        fill_light.data.color = (0.9, 0.95, 1.0)
        fill_light.data.shape = 'RECTANGLE'
        fill_light.data.size = 3
        fill_light.data.size_y = 3
        
        print("优化光照设置完成")
    
    def setup_camera(self, location=(5, -5, 5), rotation=(math.radians(60), 0, math.radians(45))):
        """设置相机位置和参数"""
        # 删除现有相机
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.context.scene.objects:
            if obj.type == 'CAMERA':
                obj.select_set(True)
        bpy.ops.object.delete()
        
        # 创建新相机
        bpy.ops.object.camera_add(location=location, rotation=rotation)
        camera = bpy.context.active_object
        camera.name = "MainCamera"
        
        # 设置为活动相机
        bpy.context.scene.camera = camera
        
        # 优化相机设置
        camera.data.lens = 35  # 标准镜头，适合大多数场景
        
        print("相机设置完成")
    
    def setup_eevee_render(self):
        """设置优化的Eevee渲染参数"""
        scene = bpy.context.scene
        scene.render.engine = 'BLENDER_EEVEE'
        
        # 渲染分辨率
        scene.render.resolution_x = self.target_resolution[0]
        scene.render.resolution_y = self.target_resolution[1]
        scene.render.resolution_percentage = 100
        
        # Eevee特定设置
        eevee = scene.eevee
        
        # 降低采样以提高性能
        eevee.samples = 16 if self.optimization_level < 3 else 32
        
        # 关闭高级功能以减少GPU负载
        eevee.use_ssr = False if self.optimization_level == 1 else True
        eevee.use_ssr_refraction = False
        eevee.use_gtao = True  # 环境光遮蔽，但优化设置
        eevee.gtao_distance = 0.2
        eevee.gtao_factor = 0.5
        
        # 阴影设置
        eevee.shadow_cascade_size = '1024'
        eevee.shadow_cascade_count = 2
        
        # 体积光设置
        eevee.volumetric_samples = 16
        
        print("Eevee渲染设置完成")
    
    def setup_compositing(self):
        """设置基础合成效果，提升视觉质量而不增加太多GPU负担"""
        scene = bpy.context.scene
        scene.use_nodes = True
        
        # 获取节点树
        tree = scene.node_tree
        nodes = tree.nodes
        links = tree.links
        
        # 清除所有节点
        for node in nodes:
            nodes.remove(node)
        
        # 添加基本节点
        render_layers = nodes.new(type='CompositorNodeRLayers')
        composite = nodes.new(type='CompositorNodeComposite')
        
        # 添加简单的颜色校正
        color_correction = nodes.new(type='CompositorNodeColorCorrection')
        color_correction.saturation = 1.1  # 略微提高饱和度
        
        # 连接节点
        links.new(render_layers.outputs["Image"], color_correction.inputs["Image"])
        links.new(color_correction.outputs["Image"], composite.inputs["Image"])
        
        print("合成设置完成")
    
    def create_demo_scene(self):
        """创建一个完整的演示场景"""
        print("创建优化演示场景...")
        
        # 1. 清理场景
        self.clear_scene()
        
        # 2. 设置相机
        self.setup_camera()
        
        # 3. 设置光照
        self.setup_optimized_lighting()
        
        # 4. 创建环境
        objects = self.create_low_poly_model("environment")
        
        # 5. 应用材质
        if isinstance(objects, list):
            self.create_optimized_material(objects[0], base_color=(0.3, 0.5, 0.3, 1.0), roughness=0.8)  # 地面
            self.create_optimized_material(objects[1], base_color=(0.8, 0.2, 0.2, 1.0), roughness=0.5)  # 立方体
            self.create_optimized_material(objects[2], base_color=(0.2, 0.2, 0.8, 1.0), roughness=0.6)  # 圆柱体
        
        # 6. 设置渲染
        self.setup_eevee_render()
        self.setup_compositing()
        
        print("演示场景创建完成，可以在GTX 750Ti和RX 580等低端GPU上流畅运行")

# 使用示例
if __name__ == "__main__":
    # 创建低端GPU优化建模工具实例
    low_end_modeler = LowEndModeling()
    
    # 设置优化级别（1-3，1为最高优化）
    low_end_modeler.optimization_level = 1  # 针对最低端硬件优化
    
    # 创建演示场景
    low_end_modeler.create_demo_scene()
    
    print("\n提示：")
    print("1. 渲染设置已优化，可在低端GPU上流畅运行")
    print("2. 要进一步优化，请减小渲染分辨率或降低优化级别")
    print("3. 可调整low_end_modeler.optimization_level值来平衡性能和质量")