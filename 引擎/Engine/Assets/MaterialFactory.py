import bpy
from engine.renderer.Resources.VRAMManager import VRAMManager
from engine.renderer.Resources.TextureCompressor import TextureCompressor

class EnhancedMaterialFactory:
    """增强版材质创建和管理系统"""
    
    def __init__(self, vram_manager=None, texture_compressor=None):
        """初始化材质工厂"""
        self.vram_manager = vram_manager or VRAMManager()
        self.texture_compressor = texture_compressor or TextureCompressor()
        self.material_registry = {}
        self.material_lod_levels = {
            "high": {"texture_resolution": 1.0, "shader_complexity": "full"},
            "medium": {"texture_resolution": 0.5, "shader_complexity": "medium"},
            "low": {"texture_resolution": 0.25, "shader_complexity": "simple"},
            "ultra_low": {"texture_resolution": 0.125, "shader_complexity": "basic"}
        }
    
    def create_base_material(self, name, color=(0.5, 0.5, 0.5, 1.0), metallic=0.0, roughness=0.5, 
                           emission_strength=0.0, alpha=1.0, ior=1.45, transmission=0.0, 
                           subsurface=0.0, subsurface_color=(1.0, 1.0, 1.0, 1.0), 
                           subsurface_radius=(1.0, 0.2, 0.1), clearcoat=0.0, clearcoat_roughness=0.0,
                           sheen=0.0, sheen_tint=0.0, anisotropy=0.0, anisotropy_rotation=0.0, 
                           lod_level="medium"):
        """创建基础材质 - 增强版，支持完整的PBR参数和LOD系统"""
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        # 清除默认节点
        for node in nodes:
            nodes.remove(node)
        
        # 创建必要的节点
        output = nodes.new(type='ShaderNodeOutputMaterial')
        principled = nodes.new(type='ShaderNodeBsdfPrincipled')
        
        # 根据LOD级别调整材质复杂度
        lod_settings = self.material_lod_levels.get(lod_level, self.material_lod_levels["medium"])
        
        # 设置核心PBR材质属性
        principled.inputs['Base Color'].default_value = color
        principled.inputs['Metallic'].default_value = metallic
        principled.inputs['Roughness'].default_value = roughness
        principled.inputs['Alpha'].default_value = alpha
        principled.inputs['IOR'].default_value = ior
        principled.inputs['Transmission'].default_value = transmission
        
        # 根据LOD级别决定是否启用高级PBR属性
        if lod_settings["shader_complexity"] in ["full", "medium"]:
            principled.inputs['Subsurface'].default_value = subsurface
            principled.inputs['Subsurface Color'].default_value = subsurface_color
            principled.inputs['Subsurface Radius'].default_value = subsurface_radius
            principled.inputs['Clearcoat'].default_value = clearcoat
            principled.inputs['Clearcoat Roughness'].default_value = clearcoat_roughness
        
        if lod_settings["shader_complexity"] == "full":
            principled.inputs['Sheen'].default_value = sheen
            principled.inputs['Sheen Tint'].default_value = sheen_tint
            principled.inputs['Anisotropy'].default_value = anisotropy
            principled.inputs['Anisotropy Rotation'].default_value = anisotropy_rotation
        
        # 连接节点
        links.new(principled.outputs['BSDF'], output.inputs['Surface'])
        
        # 如果需要自发光
        if emission_strength > 0:
            emission = nodes.new(type='ShaderNodeEmission')
            emission.inputs['Color'].default_value = color
            emission.inputs['Strength'].default_value = emission_strength
            mix = nodes.new(type='ShaderNodeMixShader')
            links.new(principled.outputs['BSDF'], mix.inputs[1])
            links.new(emission.outputs['Emission'], mix.inputs[2])
            links.new(mix.outputs['Shader'], output.inputs['Surface'])
        
        # 注册材质
        self._register_material(name, mat, lod_level, {
            "color": color,
            "metallic": metallic,
            "roughness": roughness,
            "emission_strength": emission_strength,
            "alpha": alpha,
            "ior": ior,
            "transmission": transmission
        })
        
        return mat, principled
    
    def set_material_properties(self, mat, blend_mode='OPAQUE', shadow_method='OPAQUE', **kwargs):
        """统一设置材质属性"""
        if hasattr(mat, 'blend_method'):
            mat.blend_method = blend_mode
        if hasattr(mat, 'shadow_method'):
            mat.shadow_method = shadow_method
        
        # 应用其他属性
        for attr, value in kwargs.items():
            if hasattr(mat, attr):
                setattr(mat, attr, value)
    
    def create_water_material(self, color=(0.05, 0.5, 0.65, 0.85), lod_level="medium"):
        """创建优化的水面材质 - 增强版，支持LOD"""
        mat, principled = self.create_base_material(
            "WaterMaterial", 
            color=color,
            metallic=0.92,
            roughness=0.01,
            alpha=0.85,
            ior=1.33,
            transmission=0.9,
            clearcoat=0.1,
            clearcoat_roughness=0.05,
            lod_level=lod_level
        )
        
        # 设置EEVEE特定属性
        self.set_material_properties(
            mat, blend_mode='BLEND', shadow_method='HASHED',
            use_screen_refraction=True
        )
        
        return mat
    
    def create_terrain_material(self, lod_level="medium"):
        """创建优化的地形材质 - 增强版，支持LOD"""
        mat, principled = self.create_base_material(
            "TerrainMaterial", 
            color=(0.3, 0.4, 0.2, 1.0),
            metallic=0.0,
            roughness=0.9,
            lod_level=lod_level
        )
        return mat
    
    def create_rock_material(self, lod_level="medium"):
        """创建优化的岩石材质 - 增强版，支持LOD"""
        mat, principled = self.create_base_material(
            "RockMaterial", 
            color=(0.4, 0.35, 0.3, 1.0),
            metallic=0.1,
            roughness=0.8,
            subsurface=0.1,
            lod_level=lod_level
        )
        return mat
    
    def create_wood_material(self, lod_level="medium"):
        """创建优化的木材材质，支持LOD"""
        mat, principled = self.create_base_material(
            "WoodMaterial", 
            color=(0.6, 0.4, 0.2, 1.0),
            metallic=0.0,
            roughness=0.8,
            subsurface=0.3,
            subsurface_color=(0.8, 0.6, 0.4, 1.0),
            subsurface_radius=(0.5, 0.1, 0.05),
            lod_level=lod_level
        )
        return mat
    
    def create_metal_material(self, color=(0.8, 0.8, 0.8, 1.0), lod_level="medium"):
        """创建优化的金属材质，支持LOD"""
        mat, principled = self.create_base_material(
            "MetalMaterial", 
            color=color,
            metallic=1.0,
            roughness=0.1,
            anisotropy=0.2,
            lod_level=lod_level
        )
        return mat
    
    def create_glass_material(self, color=(1.0, 1.0, 1.0, 0.5), lod_level="medium"):
        """创建优化的玻璃材质，支持LOD"""
        mat, principled = self.create_base_material(
            "GlassMaterial", 
            color=color,
            metallic=0.0,
            roughness=0.0,
            alpha=0.5,
            ior=1.5,
            transmission=1.0,
            lod_level=lod_level
        )
        
        # 设置透明属性
        self.set_material_properties(
            mat, blend_mode='BLEND', shadow_method='HASHED'
        )
        
        return mat
    
    def create_leaves_material(self, lod_level="medium"):
        """创建优化的树叶材质，支持LOD"""
        mat, principled = self.create_base_material(
            "LeavesMaterial", 
            color=(0.2, 0.6, 0.1, 1.0),
            metallic=0.0,
            roughness=0.6,
            alpha=0.9,
            subsurface=0.5,
            subsurface_color=(0.4, 0.8, 0.2, 1.0),
            subsurface_radius=(0.3, 0.5, 0.1),
            lod_level=lod_level
        )
        
        # 设置透明属性
        self.set_material_properties(
            mat, blend_mode='BLEND', shadow_method='HASHED'
        )
        
        return mat
    
    def create_cloud_material(self, lod_level="medium"):
        """创建优化的云朵材质，支持LOD"""
        mat, principled = self.create_base_material(
            "CloudMaterial", 
            color=(1.0, 1.0, 1.0, 0.5),
            metallic=0.0,
            roughness=0.9,
            alpha=0.5,
            subsurface=0.8,
            subsurface_color=(1.0, 1.0, 1.0, 1.0),
            subsurface_radius=(0.5, 0.5, 0.5),
            lod_level=lod_level
        )
        
        # 设置透明属性
        self.set_material_properties(
            mat, blend_mode='BLEND', shadow_method='HASHED'
        )
        
        return mat
    
    def create_emissive_material(self, color=(1.0, 1.0, 0.0, 1.0), strength=1.0, lod_level="medium"):
        """创建自发光材质，支持LOD"""
        mat, principled = self.create_base_material(
            "EmissiveMaterial", 
            color=color,
            metallic=0.0,
            roughness=0.5,
            emission_strength=strength,
            lod_level=lod_level
        )
        return mat
    
    def create_material_with_texture(self, name, texture_path, color=(0.5, 0.5, 0.5, 1.0), metallic=0.0, roughness=0.5, 
                                   lod_level="medium", compress_texture=True):
        """创建带有纹理的材质，支持纹理压缩和LOD"""
        mat, principled = self.create_base_material(
            name, color=color, metallic=metallic, roughness=roughness, lod_level=lod_level
        )
        
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        # 创建纹理节点
        tex_image = nodes.new(type='ShaderNodeTexImage')
        
        # 加载纹理
        try:
            tex_image.image = bpy.data.images.load(texture_path)
            
            # 根据LOD级别调整纹理分辨率
            lod_settings = self.material_lod_levels.get(lod_level, self.material_lod_levels["medium"])
            texture_resolution = lod_settings["texture_resolution"]
            
            # 压缩纹理以节省VRAM
            if compress_texture:
                compressed_texture = self.texture_compressor.compress_texture(
                    tex_image.image, 
                    resolution_scale=texture_resolution,
                    compression_quality="medium"
                )
                if compressed_texture:
                    tex_image.image = compressed_texture
            
            # 注册纹理资源到VRAM管理器
            texture_size = self.vram_manager.estimate_resource_size(
                "texture_2d", 
                tex_image.image.size[0], 
                tex_image.image.size[1], 
                "rgba8"
            )
            self.vram_manager.register_resource(
                f"texture_{name}", 
                "texture_2d", 
                texture_size, 
                priority=7, 
                compressible=True
            )
            
        except Exception as e:
            print(f"无法加载纹理 {texture_path}: {e}")
        
        # 连接纹理到基础颜色
        links.new(tex_image.outputs['Color'], principled.inputs['Base Color'])
        
        # 根据LOD级别决定是否连接纹理到其他属性
        if lod_settings["shader_complexity"] in ["full", "medium"]:
            # 连接纹理到粗糙度和金属度（如果有）
            if hasattr(tex_image.outputs, 'Roughness'):
                links.new(tex_image.outputs['Roughness'], principled.inputs['Roughness'])
            if hasattr(tex_image.outputs, 'Metallic'):
                links.new(tex_image.outputs['Metallic'], principled.inputs['Metallic'])
        
        return mat, principled
    
    def _register_material(self, name, material, lod_level, properties):
        """注册材质到材质注册表"""
        self.material_registry[name] = {
            "material": material,
            "lod_level": lod_level,
            "properties": properties,
            "created_at": bpy.context.scene.frame_current,
            "last_used": bpy.context.scene.frame_current
        }
    
    def update_material_lod(self, material_name, new_lod_level):
        """更新材质的LOD级别"""
        if material_name not in self.material_registry:
            print(f"警告: 材质 {material_name} 未注册")
            return False
        
        material_info = self.material_registry[material_name]
        old_material = material_info["material"]
        properties = material_info["properties"]
        
        # 创建新的LOD材质
        new_material = self.create_base_material(
            f"{material_name}_{new_lod_level}",
            **properties,
            lod_level=new_lod_level
        )[0]
        
        # 更新注册表
        material_info["material"] = new_material
        material_info["lod_level"] = new_lod_level
        material_info["last_used"] = bpy.context.scene.frame_current
        
        return new_material
    
    def get_material_lod_info(self, material_name):
        """获取材质的LOD信息"""
        return self.material_registry.get(material_name, None)
    
    def cleanup_unused_materials(self, max_idle_frames=100):
        """清理长时间未使用的材质"""
        current_frame = bpy.context.scene.frame_current
        materials_to_remove = []
        
        for name, info in self.material_registry.items():
            idle_frames = current_frame - info["last_used"]
            if idle_frames > max_idle_frames:
                materials_to_remove.append(name)
        
        for name in materials_to_remove:
            material = self.material_registry[name]["material"]
            if material.name in bpy.data.materials:
                bpy.data.materials.remove(material)
            del self.material_registry[name]
            print(f"清理未使用的材质: {name}")
    
    # 静态方法兼容旧版API
    @staticmethod
    def create_base_material_static(name, color=(0.5, 0.5, 0.5, 1.0), metallic=0.0, roughness=0.5, 
                                   emission_strength=0.0, alpha=1.0, ior=1.45, transmission=0.0, 
                                   subsurface=0.0, subsurface_color=(1.0, 1.0, 1.0, 1.0), 
                                   subsurface_radius=(1.0, 0.2, 0.1), clearcoat=0.0, clearcoat_roughness=0.0,
                                   sheen=0.0, sheen_tint=0.0, anisotropy=0.0, anisotropy_rotation=0.0):
        """静态方法版本，兼容旧版API"""
        factory = EnhancedMaterialFactory()
        return factory.create_base_material(
            name, color, metallic, roughness, emission_strength, alpha, ior, transmission,
            subsurface, subsurface_color, subsurface_radius, clearcoat, clearcoat_roughness,
            sheen, sheen_tint, anisotropy, anisotropy_rotation
        )