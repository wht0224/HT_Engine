# -*- coding: utf-8 -*-
"""
材质资源类，用于管理3D模型的材质属性
"""

from Engine.Math import Vector3

class Material:
    """材质资源类，用于管理3D模型的材质属性"""
    
    def __init__(self):
        """
        初始化材质
        """
        self.name = "DefaultMaterial"
        self.base_color = Vector3(0.8, 0.8, 0.8)  # 基础颜色
        self.roughness = 0.5  # 粗糙度
        self.metallic = 0.0   # 金属度
        self.ao = 1.0         # 环境光遮蔽
        self.alpha = 1.0       # 透明度
        self.emissive = Vector3(0, 0, 0)  # 自发光颜色
        self.emissive_strength = 0.0  # 自发光强度
        
        # 纹理资源
        self.base_color_texture = None
        self.roughness_texture = None
        self.metallic_texture = None
        self.normal_texture = None
        self.ao_texture = None
        self.emissive_texture = None
        
        # 渲染设置
        self.double_sided = False  # 是否双面渲染
        self.wireframe = False     # 是否以线框模式渲染
        self.blend_mode = "opaque"  # 混合模式："opaque", "transparent", "additive", "multiply"
        self.alpha_test = False    # 是否启用Alpha测试
        self.alpha_cutoff = 0.5     # Alpha测试阈值
        
        # 阴影设置
        self.cast_shadows = True    # 是否投射阴影
        self.receive_shadows = True # 是否接收阴影
        
        # 着色器设置
        self.shader = None          # 关联的着色器程序
        self.shader_name = "pbr"    # 着色器名称
        
        # 渲染状态
        self.is_dirty = True        # 数据是否需要更新到GPU
    
    def set_color(self, color):
        """设置材质的基础颜色
        
        Args:
            color: 颜色向量（Vector3）
        """
        self.base_color = color
        self.is_dirty = True
    
    def set_roughness(self, roughness):
        """设置材质的粗糙度
        
        Args:
            roughness: 粗糙度值（0-1）
        """
        self.roughness = max(0.0, min(1.0, roughness))
        self.is_dirty = True
    
    def set_metallic(self, metallic):
        """设置材质的金属度
        
        Args:
            metallic: 金属度值（0-1）
        """
        self.metallic = max(0.0, min(1.0, metallic))
        self.is_dirty = True
    
    def set_ao(self, ao):
        """设置材质的环境光遮蔽
        
        Args:
            ao: 环境光遮蔽值（0-1）
        """
        self.ao = max(0.0, min(1.0, ao))
        self.is_dirty = True
    
    def set_emissive(self, emissive, strength=1.0):
        """设置材质的自发光颜色和强度
        
        Args:
            emissive: 自发光颜色向量（Vector3）
            strength: 自发光强度
        """
        self.emissive = emissive
        self.emissive_strength = max(0.0, strength)
        self.is_dirty = True
    
    def set_double_sided(self, double_sided):
        """设置是否双面渲染
        
        Args:
            double_sided: 是否双面渲染
        """
        self.double_sided = double_sided
        self.is_dirty = True
    
    def set_wireframe(self, wireframe):
        """设置是否以线框模式渲染
        
        Args:
            wireframe: 是否以线框模式渲染
        """
        self.wireframe = wireframe
        self.is_dirty = True
    
    # 混合模式常量
    BLEND_MODE_OPAQUE = "opaque"
    BLEND_MODE_TRANSPARENT = "transparent"
    BLEND_MODE_ADDITIVE = "additive"
    BLEND_MODE_MULTIPLY = "multiply"
    
    def set_transparency(self, transparency):
        """
        设置材质的透明度
        
        Args:
            transparency: 透明度值（0-1）
        """
        self.alpha = max(0.0, min(1.0, transparency))
        self.is_dirty = True
    
    def set_blend_mode(self, blend_mode):
        """
        设置混合模式
        
        Args:
            blend_mode: 混合模式常量或字符串
        """
        valid_modes = ["opaque", "transparent", "additive", "multiply"]
        if blend_mode in valid_modes:
            self.blend_mode = blend_mode
            self.is_dirty = True
    
    def set_shader(self, shader):
        """设置关联的着色器程序
        
        Args:
            shader: 着色器程序
        """
        self.shader = shader
        self.is_dirty = True
    
    def set_shader_name(self, shader_name):
        """设置着色器名称
        
        Args:
            shader_name: 着色器名称
        """
        self.shader_name = shader_name
        self.is_dirty = True
    
    def update(self):
        """更新材质数据，将数据上传到GPU"""
        if not self.is_dirty:
            return
        
        # 如果有纹理，更新纹理数据
        if self.base_color_texture:
            self.base_color_texture.update()
        if self.roughness_texture:
            self.roughness_texture.update()
        if self.metallic_texture:
            self.metallic_texture.update()
        if self.normal_texture:
            self.normal_texture.update()
        if self.ao_texture:
            self.ao_texture.update()
        if self.emissive_texture:
            self.emissive_texture.update()
        
        # 更新着色器参数
        if self.shader:
            self._update_shader_params()
        
        self.is_dirty = False
    
    def _update_shader_params(self):
        """
        更新着色器参数
        """
        if not self.shader:
            return
        
        # 设置PBR材质参数
        self.shader.set_vec3("u_baseColor", self.base_color)
        self.shader.set_float("u_roughness", self.roughness)
        self.shader.set_float("u_metallic", self.metallic)
        self.shader.set_float("u_ao", self.ao)
        self.shader.set_float("u_alpha", self.alpha)
        self.shader.set_vec3("u_emissive", self.emissive)
        self.shader.set_float("u_emissiveStrength", self.emissive_strength)
        
        # 设置纹理采样器
        texture_unit = 0
        if self.base_color_texture:
            self.shader.set_int("u_baseColorTexture", texture_unit)
            texture_unit += 1
        if self.roughness_texture:
            self.shader.set_int("u_roughnessTexture", texture_unit)
            texture_unit += 1
        if self.metallic_texture:
            self.shader.set_int("u_metallicTexture", texture_unit)
            texture_unit += 1
        if self.normal_texture:
            self.shader.set_int("u_normalTexture", texture_unit)
            texture_unit += 1
        if self.ao_texture:
            self.shader.set_int("u_aoTexture", texture_unit)
            texture_unit += 1
        if self.emissive_texture:
            self.shader.set_int("u_emissiveTexture", texture_unit)
            texture_unit += 1
    
    def bind(self):
        """绑定材质，设置着色器参数"""
        # 更新材质数据
        self.update()
        
        # 绑定着色器
        if self.shader:
            self.shader.bind()
        
        # 绑定纹理
        texture_unit = 0
        if self.base_color_texture:
            self.base_color_texture.bind(texture_unit)
            texture_unit += 1
        if self.roughness_texture:
            self.roughness_texture.bind(texture_unit)
            texture_unit += 1
        if self.metallic_texture:
            self.metallic_texture.bind(texture_unit)
            texture_unit += 1
        if self.normal_texture:
            self.normal_texture.bind(texture_unit)
            texture_unit += 1
        if self.ao_texture:
            self.ao_texture.bind(texture_unit)
            texture_unit += 1
        if self.emissive_texture:
            self.emissive_texture.bind(texture_unit)
            texture_unit += 1
        
        # 设置渲染状态
        self._set_render_state()
    
    def unbind(self):
        """解绑材质"""
        # 解绑纹理
        if self.base_color_texture:
            self.base_color_texture.unbind()
        if self.roughness_texture:
            self.roughness_texture.unbind()
        if self.metallic_texture:
            self.metallic_texture.unbind()
        if self.normal_texture:
            self.normal_texture.unbind()
        if self.ao_texture:
            self.ao_texture.unbind()
        if self.emissive_texture:
            self.emissive_texture.unbind()
        
        # 解绑着色器
        if self.shader:
            self.shader.unbind()
    
    def _set_render_state(self):
        """设置渲染状态"""
        from OpenGL.GL import (
            glEnable, glDisable, glDepthFunc, glCullFace,
            GL_DEPTH_TEST, GL_CULL_FACE, GL_BACK, GL_FRONT,
            GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
            GL_ONE, GL_ZERO, GL_ADD, GL_MULTIPLY
        )
        
        # 深度测试
        if self.blend_mode == "opaque":
            glEnable(GL_DEPTH_TEST)
        else:
            glEnable(GL_DEPTH_TEST)
        
        # 背面剔除
        if self.double_sided:
            glDisable(GL_CULL_FACE)
        else:
            glEnable(GL_CULL_FACE)
            glCullFace(GL_BACK)
        
        # 混合模式
        if self.blend_mode == "transparent":
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        elif self.blend_mode == "additive":
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        elif self.blend_mode == "multiply":
            glEnable(GL_BLEND)
            glBlendFunc(GL_ZERO, GL_SRC_COLOR)
        else:  # opaque
            glDisable(GL_BLEND)
        
        # 线框模式
        if self.wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    
    def destroy(self):
        """销毁材质，释放资源"""
        # 释放纹理资源
        if self.base_color_texture:
            self.base_color_texture.destroy()
        if self.roughness_texture:
            self.roughness_texture.destroy()
        if self.metallic_texture:
            self.metallic_texture.destroy()
        if self.normal_texture:
            self.normal_texture.destroy()
        if self.ao_texture:
            self.ao_texture.destroy()
        if self.emissive_texture:
            self.emissive_texture.destroy()
        
        # 释放着色器资源（如果是材质拥有的着色器）
        # 注意：这里需要判断着色器是否被其他材质共享
        # 简单实现：不释放着色器，由着色器管理器管理
        pass
