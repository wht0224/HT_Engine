import os
import time
import numpy as np
import re

class ShaderManager:
    """
    着色器管理器
    针对低端GPU（如NVIDIA GTX 750Ti和AMD RX 580）优化的着色器系统
    实现着色器加载、编译、缓存和架构特定优化
    """
    
    def __init__(self, platform_info=None):
        # 平台信息
        self.platform_info = platform_info or {}
        
        # 着色器缓存目录
        self.shader_cache_dir = "./shader_cache"
        os.makedirs(self.shader_cache_dir, exist_ok=True)
        
        # 已加载的着色器程序
        self.shader_programs = {}
        
        # 着色器代码缓存
        self.shader_source_cache = {}
        
        # 编译选项
        self.compile_options = {
            "optimization_level": 2,              # 优化级别 (0-3)
            "enable_debug_info": False,           # 是否生成调试信息
            "use_precision_qualifiers": True,     # 使用精度限定符
            "auto_precision_fallback": True,      # 自动精度降级
            "enable_fast_math": True,             # 启用快速数学
            "inline_simple_functions": True,      # 内联简单函数
            "strip_unused_code": True,            # 剥离未使用的代码
            "use_half_precision": False           # 默认不使用半精度浮点数
        }
        
        # 架构特定优化
        self.architecture_optimizations = {
            "maxwell": {
                "prefer_16bit_calculations": True,
                "avoid_dynamic_flow_control": True,
                "optimize_memory_access": True,
                "pack_matrices": True,
                "use_fma_instructions": True
            },
            "gcn": {
                "prefer_16bit_calculations": False,
                "avoid_dynamic_flow_control": False,
                "optimize_memory_access": True,
                "pack_matrices": True,
                "use_fma_instructions": True,
                "use_ls_instructions": True
            },
            "pascal": {
                "prefer_16bit_calculations": True,
                "avoid_dynamic_flow_control": False,
                "optimize_memory_access": True,
                "pack_matrices": True,
                "use_fma_instructions": True,
                "use_shared_memory": True
            },
            "default": {
                "prefer_16bit_calculations": False,
                "avoid_dynamic_flow_control": True,
                "optimize_memory_access": True,
                "pack_matrices": False,
                "use_fma_instructions": False
            }
        }
        
        # 确定当前架构优化设置
        self.current_optimizations = self._determine_optimizations()
        
        # 着色器变体管理
        self.shader_variants = {}
        
        # 常用着色器片段库
        self.shader_library = {
            "common.glsl": self._get_common_shader_library(),
            "lighting.glsl": self._get_lighting_shader_library(),
            "shadow.glsl": self._get_shadow_shader_library(),
            "postprocess.glsl": self._get_postprocess_shader_library()
        }
        
        # 性能计数器
        self.performance_stats = {
            "shaders_compiled": 0,
            "shader_compilation_time": 0,
            "shader_variants_generated": 0,
            "shader_cache_hits": 0,
            "shader_cache_misses": 0
        }
    
    def _determine_optimizations(self):
        """
        根据平台信息确定架构特定的优化设置
        
        Returns:
            dict: 当前架构的优化设置
        """
        gpu_name = self.platform_info.get("gpu_name", "").lower()
        
        # 检测NVIDIA Maxwell架构 (GTX 750Ti)
        if "geforce gtx 750" in gpu_name or "maxwell" in gpu_name:
            print("检测到NVIDIA Maxwell架构，应用特定优化")
            return self.architecture_optimizations["maxwell"]
        
        # 检测AMD GCN架构 (RX 580)
        if "rx 580" in gpu_name or "radeon rx 580" in gpu_name or "gcn" in gpu_name:
            print("检测到AMD GCN架构，应用特定优化")
            return self.architecture_optimizations["gcn"]
        
        # 检测NVIDIA Pascal架构
        if "pascal" in gpu_name:
            print("检测到NVIDIA Pascal架构，应用特定优化")
            return self.architecture_optimizations["pascal"]
        
        print("使用默认优化设置")
        return self.architecture_optimizations["default"]
    
    def load_shader_from_file(self, shader_id, vertex_file, fragment_file, defines=None, include_dirs=None):
        """
        从文件加载着色器程序
        
        Args:
            shader_id: 着色器程序标识符
            vertex_file: 顶点着色器文件路径
            fragment_file: 片段着色器文件路径
            defines: 预定义宏列表
            include_dirs: 包含目录列表
            
        Returns:
            object: 编译后的着色器程序
        """
        # 生成变体键
        variant_key = self._generate_variant_key(shader_id, defines)
        
        # 检查缓存
        if variant_key in self.shader_programs:
            print(f"着色器程序 {variant_key} 已从内存缓存加载")
            self.performance_stats["shader_cache_hits"] += 1
            return self.shader_programs[variant_key]
        
        # 检查磁盘缓存
        cached_program = self._load_from_disk_cache(variant_key)
        if cached_program:
            self.performance_stats["shader_cache_hits"] += 1
            return cached_program
        
        self.performance_stats["shader_cache_misses"] += 1
        
        # 读取着色器源代码
        try:
            with open(vertex_file, 'r') as f:
                vertex_source = f.read()
            
            with open(fragment_file, 'r') as f:
                fragment_source = f.read()
            
            # 处理包含和定义
            vertex_source = self._process_shader_source(vertex_source, defines, include_dirs)
            fragment_source = self._process_shader_source(fragment_source, defines, include_dirs)
            
            # 应用架构特定优化
            vertex_source = self._apply_architecture_optimizations(vertex_source, "vertex")
            fragment_source = self._apply_architecture_optimizations(fragment_source, "fragment")
            
            # 编译着色器程序
            start_time = time.time()
            shader_program = self._compile_shader_program(shader_id, vertex_source, fragment_source, defines)
            compilation_time = time.time() - start_time
            
            # 更新性能统计
            self.performance_stats["shaders_compiled"] += 1
            self.performance_stats["shader_compilation_time"] += compilation_time
            self.performance_stats["shader_variants_generated"] += 1
            
            print(f"编译着色器程序 {variant_key} 耗时: {compilation_time:.2f}秒")
            
            # 缓存编译后的程序
            self.shader_programs[variant_key] = shader_program
            
            # 保存到磁盘缓存
            self._save_to_disk_cache(variant_key, shader_program)
            
            return shader_program
            
        except Exception as e:
            print(f"加载着色器程序失败: {e}")
            return None
    
    def load_shader_from_string(self, shader_id, vertex_source, fragment_source, defines=None):
        """
        从字符串加载着色器程序
        
        Args:
            shader_id: 着色器程序标识符
            vertex_source: 顶点着色器源代码
            fragment_source: 片段着色器源代码
            defines: 预定义宏列表
            
        Returns:
            object: 编译后的着色器程序
        """
        # 生成变体键
        variant_key = self._generate_variant_key(shader_id, defines)
        
        # 检查缓存
        if variant_key in self.shader_programs:
            print(f"着色器程序 {variant_key} 已从内存缓存加载")
            self.performance_stats["shader_cache_hits"] += 1
            return self.shader_programs[variant_key]
        
        # 检查磁盘缓存
        cached_program = self._load_from_disk_cache(variant_key)
        if cached_program:
            self.performance_stats["shader_cache_hits"] += 1
            return cached_program
        
        self.performance_stats["shader_cache_misses"] += 1
        
        # 处理包含和定义
        vertex_source = self._process_shader_source(vertex_source, defines)
        fragment_source = self._process_shader_source(fragment_source, defines)
        
        # 应用架构特定优化
        vertex_source = self._apply_architecture_optimizations(vertex_source, "vertex")
        fragment_source = self._apply_architecture_optimizations(fragment_source, "fragment")
        
        # 编译着色器程序
        start_time = time.time()
        shader_program = self._compile_shader_program(shader_id, vertex_source, fragment_source, defines)
        compilation_time = time.time() - start_time
        
        # 更新性能统计
        self.performance_stats["shaders_compiled"] += 1
        self.performance_stats["shader_compilation_time"] += compilation_time
        self.performance_stats["shader_variants_generated"] += 1
        
        print(f"编译着色器程序 {variant_key} 耗时: {compilation_time:.2f}秒")
        
        # 缓存编译后的程序
        self.shader_programs[variant_key] = shader_program
        
        # 保存到磁盘缓存
        self._save_to_disk_cache(variant_key, shader_program)
        
        return shader_program
    
    def _process_shader_source(self, source, defines=None, include_dirs=None):
        """
        处理着色器源代码，包括应用定义和包含文件
        
        Args:
            source: 原始着色器源代码
            defines: 预定义宏列表
            include_dirs: 包含目录列表
            
        Returns:
            str: 处理后的着色器源代码
        """
        # 添加预定义宏
        if defines:
            defines_code = "\n".join([f"#define {define}" for define in defines])
            source = defines_code + "\n" + source
        
        # 处理包含指令
        source = self._process_includes(source, include_dirs)
        
        # 添加版本声明（如果没有）
        if not source.strip().startswith("#version"):
            source = "#version 330 core\n" + source
        
        return source
    
    def _process_includes(self, source, include_dirs=None):
        """
        处理着色器中的包含指令
        
        Args:
            source: 原始着色器源代码
            include_dirs: 包含目录列表
            
        Returns:
            str: 处理后的着色器源代码
        """
        include_pattern = re.compile(r'#include\s+["<]([^>"\n]+)[">]')
        processed_source = source
        
        def replace_include(match):
            include_file = match.group(1)
            
            # 首先在着色器库中查找
            if include_file in self.shader_library:
                return self.shader_library[include_file]
            
            # 然后在包含目录中查找
            if include_dirs:
                for dir_path in include_dirs:
                    full_path = os.path.join(dir_path, include_file)
                    if os.path.exists(full_path):
                        try:
                            with open(full_path, 'r') as f:
                                return f.read()
                        except Exception:
                            pass
            
            # 如果找不到，返回空
            print(f"警告: 找不到包含文件: {include_file}")
            return f"// 包含文件未找到: {include_file}"
        
        # 处理所有包含指令
        while include_pattern.search(processed_source):
            processed_source = include_pattern.sub(replace_include, processed_source)
        
        return processed_source
    
    def _apply_architecture_optimizations(self, source, shader_type):
        """
        应用架构特定的优化
        
        Args:
            source: 原始着色器源代码
            shader_type: 着色器类型 ("vertex" 或 "fragment")
            
        Returns:
            str: 优化后的着色器源代码
        """
        optimized_source = source
        
        # 添加优化标记
        optimized_source = "// 应用架构特定优化\n" + optimized_source
        
        # 针对不同架构应用不同的优化
        opts = self.current_optimizations
        
        # 优化1: 使用半精度浮点数
        if opts.get("prefer_16bit_calculations", False) and self.compile_options["use_half_precision"]:
            # 替换精度声明
            if self.compile_options["use_precision_qualifiers"]:
                # 为片段着色器添加精度限定符
                if shader_type == "fragment" and "precision" not in optimized_source:
                    optimized_source = "precision mediump float;\n" + optimized_source
            
            # 优化矩阵和向量计算（简化示例）
            optimized_source = re.sub(r'\bvec3\b', 'highp vec3', optimized_source)
            optimized_source = re.sub(r'\bvec4\b', 'highp vec4', optimized_source)
        
        # 优化2: 避免动态流控制
        if opts.get("avoid_dynamic_flow_control", False):
            # 添加注释标记可能需要手动优化的部分
            # 实际实现会更复杂，可能需要更深入的代码分析和转换
            pass
        
        # 优化3: 内存访问优化
        if opts.get("optimize_memory_access", False):
            # 添加优化内存访问的注释标记
            pass
        
        # 优化4: 矩阵打包优化
        if opts.get("pack_matrices", False):
            # 添加矩阵打包优化的注释标记
            pass
        
        # 优化5: 使用FMA指令
        if opts.get("use_fma_instructions", False):
            # 添加FMA指令优化的注释标记
            # 实际实现会尝试将a*b+c模式的代码重写为fma(a,b,c)
            pass
        
        # 优化6: AMD特定的LS指令（仅对GCN架构）
        if opts.get("use_ls_instructions", False):
            # 添加AMD GCN特定优化的注释标记
            pass
        
        # 优化7: 使用共享内存（仅对Pascal架构）
        if opts.get("use_shared_memory", False):
            # 添加共享内存优化的注释标记
            pass
        
        return optimized_source
    
    def _compile_shader_program(self, shader_id, vertex_source, fragment_source, defines=None):
        """
        编译着色器程序
        
        Args:
            shader_id: 着色器程序标识符
            vertex_source: 顶点着色器源代码
            fragment_source: 片段着色器源代码
            defines: 预定义宏列表
            
        Returns:
            object: 编译后的着色器程序
        """
        # 这是一个示例实现
        # 实际实现会使用PyOpenGL或其他图形库进行实际的着色器编译
        
        print(f"编译着色器程序: {shader_id}")
        
        # 模拟编译过程
        # 在实际实现中，这里会调用图形API的着色器编译函数
        
        # 返回一个模拟的着色器程序对象
        shader_program = {
            "id": shader_id,
            "program_id": 1,  # 模拟的程序ID
            "vertex_source": vertex_source,
            "fragment_source": fragment_source,
            "defines": defines or [],
            "uniforms": {},
            "attributes": {},
            "compilation_time": time.time()
        }
        
        # 模拟提取uniform和attribute信息
        # 在实际实现中，这里会查询编译后的程序获取这些信息
        shader_program["uniforms"] = self._extract_uniforms(fragment_source)
        shader_program["attributes"] = self._extract_attributes(vertex_source)
        
        return shader_program
    
    def _extract_uniforms(self, shader_source):
        """
        从着色器源代码中提取uniform变量信息
        
        Args:
            shader_source: 着色器源代码
            
        Returns:
            dict: uniform变量信息
        """
        # 这是一个简化的实现，实际实现会更复杂
        uniforms = {}
        uniform_pattern = re.compile(r'uniform\s+(\w+)\s+(\w+);')
        
        for match in uniform_pattern.finditer(shader_source):
            uniform_type = match.group(1)
            uniform_name = match.group(2)
            uniforms[uniform_name] = uniform_type
        
        return uniforms
    
    def _extract_attributes(self, shader_source):
        """
        从着色器源代码中提取attribute变量信息
        
        Args:
            shader_source: 着色器源代码
            
        Returns:
            dict: attribute变量信息
        """
        # 这是一个简化的实现，实际实现会更复杂
        attributes = {}
        attribute_pattern = re.compile(r'attribute\s+(\w+)\s+(\w+);')
        
        for match in attribute_pattern.finditer(shader_source):
            attribute_type = match.group(1)
            attribute_name = match.group(2)
            attributes[attribute_name] = attribute_type
        
        # 也查找in变量（GLSL 330+）
        in_pattern = re.compile(r'in\s+(\w+)\s+(\w+);')
        
        for match in in_pattern.finditer(shader_source):
            attribute_type = match.group(1)
            attribute_name = match.group(2)
            attributes[attribute_name] = attribute_type
        
        return attributes
    
    def _generate_variant_key(self, shader_id, defines=None):
        """
        生成着色器变体的唯一键
        
        Args:
            shader_id: 着色器程序标识符
            defines: 预定义宏列表
            
        Returns:
            str: 变体键
        """
        base_key = shader_id
        
        if defines:
            # 对定义进行排序以确保一致性
            sorted_defines = sorted(defines)
            define_key = "_".join(sorted_defines)
            base_key += "_" + define_key
        
        # 添加架构标识符
        architecture = "unknown"
        if self.current_optimizations == self.architecture_optimizations["maxwell"]:
            architecture = "maxwell"
        elif self.current_optimizations == self.architecture_optimizations["gcn"]:
            architecture = "gcn"
        elif self.current_optimizations == self.architecture_optimizations["pascal"]:
            architecture = "pascal"
        
        return f"{base_key}_arch_{architecture}"
    
    def _load_from_disk_cache(self, variant_key):
        """
        从磁盘缓存加载着色器程序
        
        Args:
            variant_key: 变体键
            
        Returns:
            object or None: 加载的着色器程序或None
        """
        # 实际实现会从磁盘读取编译后的着色器程序
        # 这里仅作为示例
        return None
    
    def _save_to_disk_cache(self, variant_key, shader_program):
        """
        将着色器程序保存到磁盘缓存
        
        Args:
            variant_key: 变体键
            shader_program: 着色器程序
        """
        # 实际实现会将编译后的着色器程序保存到磁盘
        # 这里仅作为示例
        pass
    
    def set_compile_options(self, options):
        """
        设置编译选项
        
        Args:
            options: 编译选项字典
        """
        for key, value in options.items():
            if key in self.compile_options:
                self.compile_options[key] = value
                print(f"设置编译选项: {key} = {value}")
    
    def get_performance_stats(self):
        """
        获取着色器系统性能统计信息
        
        Returns:
            dict: 性能统计信息
        """
        return self.performance_stats.copy()
    
    def print_performance_report(self):
        """
        打印着色器系统性能报告
        """
        stats = self.performance_stats
        
        print("\n=== 着色器系统性能报告 ===")
        print(f"编译着色器总数: {stats['shaders_compiled']}")
        print(f"编译总时间: {stats['shader_compilation_time']:.2f}秒")
        print(f"生成变体总数: {stats['shader_variants_generated']}")
        print(f"缓存命中次数: {stats['shader_cache_hits']}")
        print(f"缓存未命中次数: {stats['shader_cache_misses']}")
        
        total_accesses = stats['shader_cache_hits'] + stats['shader_cache_misses']
        if total_accesses > 0:
            hit_rate = (stats['shader_cache_hits'] / total_accesses) * 100
            print(f"缓存命中率: {hit_rate:.1f}%")
        
        print("======================\n")
    
    def optimize_for_maxwell(self, shader_source):
        """
        为NVIDIA Maxwell架构优化着色器
        
        Args:
            shader_source: 原始着色器源代码
            
        Returns:
            str: 优化后的着色器源代码
        """
        print("应用Maxwell架构着色器优化")
        # 调用现有的架构优化方法
        return self._apply_architecture_optimizations(shader_source, "fragment")
    
    def optimize_for_gcn(self, shader_source):
        """
        为AMD GCN架构优化着色器
        
        Args:
            shader_source: 原始着色器源代码
            
        Returns:
            str: 优化后的着色器源代码
        """
        print("应用GCN架构着色器优化")
        # 调用现有的架构优化方法
        return self._apply_architecture_optimizations(shader_source, "fragment")
    
    def optimize_for_low_end(self, shader_source):
        """
        为低端GPU优化着色器
        
        Args:
            shader_source: 原始着色器源代码
            
        Returns:
            str: 优化后的着色器源代码
        """
        print("应用低端GPU着色器优化")
        # 调用现有的架构优化方法
        return self._apply_architecture_optimizations(shader_source, "fragment")
    
    def _get_common_shader_library(self):
        """
        获取通用着色器库代码
        
        Returns:
            str: 着色器库代码
        """
        return """
// 通用着色器库
// 针对低端GPU优化的通用功能

#ifndef COMMON_GLSL
#define COMMON_GLSL

// 针对低端GPU优化的数学函数

// 快速归一化函数（使用近似值）
highp vec3 fast_normalize(highp vec3 v) {
    highp float len = dot(v, v);
    // 使用近似的倒数平方根
    #ifdef ENABLE_FAST_MATH
    highp float rlen = inversesqrt(len);
    #else
    highp float rlen = 1.0 / sqrt(len);
    #endif
    return v * rlen;
}

// 快速矩阵-向量乘法
// 针对Maxwell/GCN架构优化的内存访问模式
highp vec4 fast_mat4_vec4(highp mat4 m, highp vec4 v) {
    // 手动展开乘法以提高性能
    return highp vec4(
        dot(m[0], v),
        dot(m[1], v),
        dot(m[2], v),
        dot(m[3], v)
    );
}

// 纹理采样优化（避免不必要的精度转换）
#ifdef PREFER_16BIT_CALCULATIONS
lowp vec4 sampleTexture(sampler2D tex, highp vec2 uv) {
    return texture2D(tex, uv);
}
#else
highp vec4 sampleTexture(sampler2D tex, highp vec2 uv) {
    return texture2D(tex, uv);
}
#endif

// 快速光照衰减
// 使用二次衰减但优化计算
highp float fast_attenuation(highp float distance, highp float range) {
    highp float att = 1.0 - min(distance / range, 1.0);
    return att * att;
}

// 快速菲涅尔效果
highp float fast_fresnel(highp vec3 viewDir, highp vec3 normal, highp float bias, highp float scale, highp float power) {
    highp float cosine = dot(viewDir, normal);
    return bias + scale * pow(1.0 - cosine, power);
}

// 简化的GGX法线分布函数
highp float distribution_ggx(highp vec3 N, highp vec3 H, highp float roughness) {
    highp float a = roughness * roughness;
    highp float a2 = a * a;
    highp float NdotH = max(dot(N, H), 0.0);
    highp float NdotH2 = NdotH * NdotH;
    
    highp float nom = a2;
    highp float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = 3.1415926535 * denom * denom;
    
    return nom / denom;
}

// 简化的几何遮挡函数
highp float geometry_schlick_ggx(highp float NdotV, highp float roughness) {
    highp float r = (roughness + 1.0);
    highp float k = (r * r) / 8.0;
    
    highp float nom = NdotV;
    highp float denom = NdotV * (1.0 - k) + k;
    
    return nom / denom;
}

highp float geometry_smith(highp vec3 N, highp vec3 V, highp vec3 L, highp float roughness) {
    highp float NdotV = max(dot(N, V), 0.0);
    highp float NdotL = max(dot(N, L), 0.0);
    highp float ggx2 = geometry_schlick_ggx(NdotV, roughness);
    highp float ggx1 = geometry_schlick_ggx(NdotL, roughness);
    
    return ggx1 * ggx2;
}

#endif // COMMON_GLSL
"""
    
    def _get_lighting_shader_library(self):
        """
        获取光照着色器库代码
        
        Returns:
            str: 着色器库代码
        """
        return """
// 光照着色器库
// 针对低端GPU优化的光照模型

#ifndef LIGHTING_GLSL
#define LIGHTING_GLSL

#include "common.glsl"

// 点光源数据结构
struct PointLight {
    highp vec3 position;
    highp vec3 color;
    highp float intensity;
    highp float range;
};

// 方向光数据结构
struct DirectionalLight {
    highp vec3 direction;
    highp vec3 color;
    highp float intensity;
};

// 聚光灯数据结构
struct SpotLight {
    highp vec3 position;
    highp vec3 direction;
    highp vec3 color;
    highp float intensity;
    highp float range;
    highp float innerConeAngle;
    highp float outerConeAngle;
};

// PBR材质数据结构
struct PBRMaterial {
    highp float roughness;
    highp float metallic;
    highp vec3 baseColor;
    highp float ao;
};

// 简单的PBR光照计算
// 针对低端GPU优化，减少计算复杂度
highp vec3 calculate_pbr_lighting(
    highp vec3 N,          // 法线
    highp vec3 V,          // 视线方向
    highp vec3 L,          // 光源方向
    highp vec3 lightColor, // 光源颜色
    highp float lightIntensity, // 光源强度
    PBRMaterial material   // 材质参数
) {
    // 计算半程向量
    highp vec3 H = fast_normalize(V + L);
    
    // 计算辐射率
    highp vec3 radiance = lightColor * lightIntensity;
    
    // 计算漫反射和镜面反射分量
    highp vec3 F0 = vec3(0.04);
    F0 = mix(F0, material.baseColor, material.metallic);
    
    // 计算菲涅尔项
    highp vec3 F = F0 + (1.0 - F0) * pow(1.0 - max(dot(H, V), 0.0), 5.0);
    
    // 计算法线分布项和几何遮挡项
    highp float NDF = distribution_ggx(N, H, material.roughness);
    highp float G = geometry_smith(N, V, L, material.roughness);
    
    // 计算镜面反射
    highp vec3 numerator = NDF * G * F;
    highp float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
    highp vec3 specular = numerator / denominator;
    
    // 计算漫反射
    highp vec3 kS = F;
    highp vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - material.metallic;
    
    // 计算光照贡献
    highp float NdotL = max(dot(N, L), 0.0);
    highp vec3 Lo = (kD * material.baseColor / 3.1415926535 + specular) * radiance * NdotL;
    
    return Lo;
}

// 优化版环境光照（使用预计算的辐照度贴图）
highp vec3 calculate_ambient_lighting(
    highp vec3 N,          // 法线
    highp vec3 V,          // 视线方向
    highp samplerCube irradianceMap, // 辐照度贴图
    highp samplerCube prefilterMap,  // 预过滤贴图
    highp sampler2D brdfLUT,         // BRDF查找表
    PBRMaterial material   // 材质参数
) {
    // 计算环境漫反射
    highp vec3 F0 = vec3(0.04);
    F0 = mix(F0, material.baseColor, material.metallic);
    
    highp vec3 kS = F0 + (1.0 - F0) * pow(1.0 - max(dot(N, V), 0.0), 5.0);
    highp vec3 kD = 1.0 - kS;
    kD *= 1.0 - material.metallic;
    
    // 采样辐照度贴图
    highp vec3 irradiance = textureCube(irradianceMap, N).rgb;
    highp vec3 diffuse = irradiance * material.baseColor;
    
    // 采样预过滤环境贴图
    highp vec3 R = reflect(-V, N);
    const float MAX_REFLECTION_LOD = 4.0;
    highp vec3 prefilteredColor = textureLod(prefilterMap, R, material.roughness * MAX_REFLECTION_LOD).rgb;
    
    // 采样BRDF查找表
    highp vec2 brdf = texture2D(brdfLUT, vec2(max(dot(N, V), 0.0), material.roughness)).rg;
    highp vec3 specular = prefilteredColor * (kS * brdf.x + brdf.y);
    
    // 组合环境光照
    highp vec3 ambient = (kD * diffuse + specular) * material.ao;
    
    return ambient;
}

// 点光源光照计算
highp vec3 calculate_point_light(
    PointLight light,
    highp vec3 N,
    highp vec3 V,
    highp vec3 worldPos,
    PBRMaterial material
) {
    // 计算光线方向和距离
    highp vec3 L = light.position - worldPos;
    highp float distance = length(L);
    L = normalize(L);
    
    // 检查是否在光照范围内
    if (distance > light.range) {
        return vec3(0.0);
    }
    
    // 计算衰减
    highp float attenuation = fast_attenuation(distance, light.range);
    
    // 计算PBR光照
    highp vec3 radiance = light.color * light.intensity * attenuation;
    
    // 简化的PBR计算（针对低端GPU）
    highp vec3 F0 = vec3(0.04);
    F0 = mix(F0, material.baseColor, material.metallic);
    
    // 计算半程向量
    highp vec3 H = normalize(V + L);
    
    // 简化的镜面反射计算
    highp float NDF = 0.1 + 0.9 * pow(1.0 - max(dot(N, H), 0.0), 3.0);
    NDF = NDF * NDF * material.roughness * material.roughness;
    
    highp vec3 F = F0 + (1.0 - F0) * pow(1.0 - max(dot(H, V), 0.0), 5.0);
    
    highp vec3 kS = F;
    highp vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - material.metallic;
    
    highp float NdotL = max(dot(N, L), 0.0);
    highp vec3 Lo = (kD * material.baseColor / 3.1415926535 + NDF * F) * radiance * NdotL;
    
    return Lo;
}

// 方向光光照计算
highp vec3 calculate_directional_light(
    DirectionalLight light,
    highp vec3 N,
    highp vec3 V,
    PBRMaterial material
) {
    // 光线方向（从表面指向光源）
    highp vec3 L = -light.direction;
    
    // 计算PBR光照
    highp vec3 radiance = light.color * light.intensity;
    
    // 简化的PBR计算（针对低端GPU）
    highp vec3 F0 = vec3(0.04);
    F0 = mix(F0, material.baseColor, material.metallic);
    
    // 计算半程向量
    highp vec3 H = normalize(V + L);
    
    // 简化的镜面反射计算
    highp float NDF = 0.1 + 0.9 * pow(1.0 - max(dot(N, H), 0.0), 3.0);
    NDF = NDF * NDF * material.roughness * material.roughness;
    
    highp vec3 F = F0 + (1.0 - F0) * pow(1.0 - max(dot(H, V), 0.0), 5.0);
    
    highp vec3 kS = F;
    highp vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - material.metallic;
    
    highp float NdotL = max(dot(N, L), 0.0);
    highp vec3 Lo = (kD * material.baseColor / 3.1415926535 + NDF * F) * radiance * NdotL;
    
    return Lo;
}

#endif // LIGHTING_GLSL
"""
    
    def _get_shadow_shader_library(self):
        """
        获取阴影着色器库代码
        
        Returns:
            str: 着色器库代码
        """
        return """
// 阴影着色器库
// 针对低端GPU优化的阴影计算

#ifndef SHADOW_GLSL
#define SHADOW_GLSL

#include "common.glsl"

// PCF阴影过滤（4x4采样）
highp float calculate_shadow_pcf(
    highp sampler2D shadowMap,
    highp vec4 shadowCoord,
    highp float bias,
    highp float shadowStrength
) {
    // 透视除法
    highp vec3 projCoords = shadowCoord.xyz / shadowCoord.w;
    
    // 检查是否在阴影映射范围内
    if (projCoords.z > 1.0) {
        return 1.0; // 不在阴影中
    }
    
    // 偏移深度值以减少阴影痤疮
    highp float currentDepth = projCoords.z - bias;
    
    // 简化的PCF（性能优化）
    highp float shadow = 0.0;
    highp vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    
    // 2x2采样网格（减少采样点以提高性能）
    for (int x = -1; x <= 1; x += 2) {
        for (int y = -1; y <= 1; y += 2) {
            highp vec2 offset = vec2(x, y) * texelSize;
            highp float closestDepth = texture2D(shadowMap, projCoords.xy + offset).r;
            shadow += (currentDepth > closestDepth) ? 1.0 : 0.0;
        }
    }
    
    // 平均采样结果
    shadow /= 4.0;
    
    // 应用阴影强度
    return 1.0 - (shadow * shadowStrength);
}

// 级联阴影映射
highp float calculate_csm_shadow(
    highp sampler2DArray shadowMap,
    highp vec4[] shadowCoords,
    highp int cascadeIndex,
    highp float bias,
    highp float shadowStrength
) {
    // 如果超出所有级联范围，则不在阴影中
    if (cascadeIndex == -1) {
        return 1.0;
    }
    
    // 确保级联索引有效
    cascadeIndex = min(cascadeIndex, 3); // 最多4级级联
    
    // 使用相应级联的阴影坐标
    highp vec4 shadowCoord = shadowCoords[cascadeIndex];
    
    // 透视除法
    highp vec3 projCoords = shadowCoord.xyz / shadowCoord.w;
    
    // 检查是否在阴影映射范围内
    if (projCoords.z > 1.0 || projCoords.x < 0.0 || projCoords.x > 1.0 || projCoords.y < 0.0 || projCoords.y > 1.0) {
        return 1.0; // 不在阴影中
    }
    
    // 偏移深度值以减少阴影痤疮
    highp float currentDepth = projCoords.z - bias;
    
    // 简化的PCF（针对低端GPU优化）
    highp float shadow = 0.0;
    highp vec2 texelSize = 1.0 / textureSize(shadowMap, 0).xy;
    
    // 1x1采样（仅一个采样点，最高性能）
    highp float closestDepth = texture(shadowMap, vec3(projCoords.xy, cascadeIndex)).r;
    shadow = (currentDepth > closestDepth) ? 1.0 : 0.0;
    
    // 应用阴影强度
    return 1.0 - (shadow * shadowStrength);
}

// 接触阴影计算（基于深度贴图的屏幕空间阴影）
highp float calculate_contact_shadows(
    highp sampler2D depthMap,
    highp vec2 screenUV,
    highp vec3 viewPos,
    highp vec3 viewNormal,
    highp float rayLength,
    highp float stepCount
) {
    // 从视图空间转换到屏幕空间
    highp vec3 viewDir = normalize(viewPos);
    highp vec3 rayDir = normalize(reflect(viewDir, viewNormal));
    
    // 沿反射方向采样深度贴图
    highp float shadow = 1.0;
    highp float stepSize = rayLength / stepCount;
    
    // 减少采样点以提高性能
    stepCount = min(stepCount, 16.0);
    
    for (int i = 0; i < int(stepCount); i++) {
        // 计算采样位置
        highp float t = float(i) * stepSize;
        highp vec3 samplePos = viewPos + rayDir * t;
        
        // 转换到屏幕空间
        highp vec4 clipPos = vec4(samplePos, 1.0);
        highp vec3 ndcPos = clipPos.xyz / clipPos.w;
        highp vec2 sampleUV = ndcPos.xy * 0.5 + 0.5;
        
        // 检查是否在屏幕范围内
        if (sampleUV.x < 0.0 || sampleUV.x > 1.0 || sampleUV.y < 0.0 || sampleUV.y > 1.0) {
            break;
        }
        
        // 采样深度贴图
        highp float sampledDepth = texture2D(depthMap, sampleUV).r;
        
        // 检查是否有遮挡
        highp float currentDepth = -samplePos.z;
        highp float depthDiff = currentDepth - sampledDepth;
        
        // 如果深度差在阈值范围内，则认为有接触阴影
        if (depthDiff > 0.01 && depthDiff < 0.1) {
            shadow *= 0.2;
            break; // 一旦找到遮挡就退出，提高性能
        }
    }
    
    return shadow;
}

#endif // SHADOW_GLSL
"""
    
    def _get_postprocess_shader_library(self):
        """
        获取后处理着色器库代码
        
        Returns:
            str: 着色器库代码
        """
        return """
// 后处理着色器库
// 针对低端GPU优化的后处理效果

#ifndef POSTPROCESS_GLSL
#define POSTPROCESS_GLSL

#include "common.glsl"

// 快速近似抗锯齿（FXAA简化版）
highp vec4 fxaa_simple(highp sampler2D tex, highp vec2 uv, highp vec2 resolution) {
    // 计算纹理采样步长
    highp vec2 inverseResolution = 1.0 / resolution;
    
    // 采样当前像素和相邻像素
    highp vec4 center = texture2D(tex, uv);
    highp vec4 left = texture2D(tex, uv + vec2(-inverseResolution.x, 0.0));
    highp vec4 right = texture2D(tex, uv + vec2(inverseResolution.x, 0.0));
    highp vec4 top = texture2D(tex, uv + vec2(0.0, inverseResolution.y));
    highp vec4 bottom = texture2D(tex, uv + vec2(0.0, -inverseResolution.y));
    
    // 计算边缘强度
    highp float lumaCenter = dot(center.rgb, vec3(0.299, 0.587, 0.114));
    highp float lumaLeft = dot(left.rgb, vec3(0.299, 0.587, 0.114));
    highp float lumaRight = dot(right.rgb, vec3(0.299, 0.587, 0.114));
    highp float lumaTop = dot(top.rgb, vec3(0.299, 0.587, 0.114));
    highp float lumaBottom = dot(bottom.rgb, vec3(0.299, 0.587, 0.114));
    
    // 计算边缘方向的对比度
    highp float lumaMin = min(lumaCenter, min(min(lumaLeft, lumaRight), min(lumaTop, lumaBottom)));
    highp float lumaMax = max(lumaCenter, max(max(lumaLeft, lumaRight), max(lumaTop, lumaBottom)));
    highp float lumaRange = lumaMax - lumaMin;
    
    // 如果对比度太低，不进行抗锯齿
    if (lumaRange < 0.08) {
        return center;
    }
    
    // 简单的边缘模糊
    highp vec4 blur = (center + left + right + top + bottom) * 0.2;
    
    // 混合原始颜色和模糊颜色
    highp float blendFactor = smoothstep(0.08, 0.16, lumaRange);
    
    return mix(center, blur, blendFactor);
}

// 快速景深效果（基于散景盘）
highp vec4 depth_of_field(
    highp sampler2D colorTex,
    highp sampler2D depthTex,
    highp vec2 uv,
    highp vec2 resolution,
    highp float focusDistance,
    highp float focusRange,
    highp float aperture
) {
    // 采样深度
    highp float depth = texture2D(depthTex, uv).r;
    
    // 计算散景大小
    highp float defocus = abs(depth - focusDistance) / focusRange;
    highp float blurRadius = defocus * aperture;
    
    // 如果模糊半径太小，直接返回原始颜色
    if (blurRadius < 0.001) {
        return texture2D(colorTex, uv);
    }
    
    // 简化的景深计算（低端GPU优化版）
    highp vec4 color = vec4(0.0);
    highp float totalWeight = 0.0;
    
    // 减少采样点数以提高性能
    highp int sampleCount = 8;
    highp float stepSize = blurRadius / float(sampleCount);
    
    // 简单的环形采样（模拟散景效果）
    for (int i = 0; i < sampleCount; i++) {
        highp float angle = 6.28318530718 * float(i) / float(sampleCount);
        highp vec2 offset = vec2(cos(angle), sin(angle)) * stepSize * float(i);
        
        // 检查偏移是否在屏幕范围内
        highp vec2 sampleUV = uv + offset * resolution;
        if (sampleUV.x >= 0.0 && sampleUV.x <= 1.0 && sampleUV.y >= 0.0 && sampleUV.y <= 1.0) {
            color += texture2D(colorTex, sampleUV);
            totalWeight += 1.0;
        }
    }
    
    // 确保权重不为零
    if (totalWeight > 0.0) {
        color /= totalWeight;
    }
    
    // 混合原始颜色和模糊颜色（基于模糊强度）
    highp float blendFactor = smoothstep(0.0, 1.0, defocus);
    highp vec4 originalColor = texture2D(colorTex, uv);
    
    return mix(originalColor, color, blendFactor);
}

// 快速环境光遮蔽（SSAO简化版）
highp float ssao_simple(
    highp sampler2D depthTex,
    highp sampler2D normalTex,
    highp vec2 uv,
    highp vec2 resolution,
    highp vec3 viewPos,
    highp mat4 projectionMatrix
) {
    // 采样法线和深度
    highp vec3 normal = texture2D(normalTex, uv).xyz * 2.0 - 1.0;
    highp float depth = texture2D(depthTex, uv).r;
    
    // 从屏幕空间转换到视图空间
    highp vec4 clipPos = vec4(uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    highp vec4 viewPosSampled = inverse(projectionMatrix) * clipPos;
    viewPosSampled /= viewPosSampled.w;
    
    // 简化的SSAO计算（低端GPU优化版）
    highp float ambientOcclusion = 1.0;
    highp float sampleRadius = 0.1;
    
    // 减少采样点数
    highp int sampleCount = 4;
    
    for (int i = 0; i < sampleCount; i++) {
        // 使用简单的采样方向（性能优化）
        highp vec3 sampleDir = normalize(vec3(
            sin(float(i) * 1.5708),
            cos(float(i) * 1.5708),
            0.5
        ));
        
        // 使采样方向与法线对齐
        if (dot(sampleDir, normal) < 0.0) {
            sampleDir = -sampleDir;
        }
        
        // 计算采样位置
        highp vec3 samplePos = viewPosSampled.xyz + sampleDir * sampleRadius;
        
        // 转换回屏幕空间
        highp vec4 sampleClipPos = projectionMatrix * vec4(samplePos, 1.0);
        sampleClipPos /= sampleClipPos.w;
        highp vec2 sampleUV = sampleClipPos.xy * 0.5 + 0.5;
        
        // 检查是否在屏幕范围内
        if (sampleUV.x >= 0.0 && sampleUV.x <= 1.0 && sampleUV.y >= 0.0 && sampleUV.y <= 1.0) {
            // 采样深度
            highp float sampleDepth = texture2D(depthTex, sampleUV).r;
            highp vec4 sampleViewPos = inverse(projectionMatrix) * vec4(sampleUV * 2.0 - 1.0, sampleDepth * 2.0 - 1.0, 1.0);
            sampleViewPos /= sampleViewPos.w;
            
            // 比较深度
            highp float rangeCheck = smoothstep(0.0, 1.0, sampleRadius / abs(viewPosSampled.z - sampleViewPos.z));
            ambientOcclusion -= (sampleViewPos.z < samplePos.z - 0.025 ? 1.0 : 0.0) * rangeCheck * 0.25;
        }
    }
    
    return clamp(ambientOcclusion, 0.0, 1.0);
}

// 快速光晕效果
highp vec4 bloom_simple(
    highp sampler2D colorTex,
    highp sampler2D blurTex,
    highp vec2 uv,
    highp float bloomIntensity,
    highp float threshold
) {
    // 采样原始颜色和模糊颜色
    highp vec4 originalColor = texture2D(colorTex, uv);
    highp vec4 blurColor = texture2D(blurTex, uv);
    
    // 计算亮度
    highp float luminance = dot(originalColor.rgb, vec3(0.299, 0.587, 0.114));
    
    // 应用阈值
    highp float bloomFactor = smoothstep(threshold, threshold * 2.0, luminance);
    
    // 混合原始颜色和光晕颜色
    highp vec4 result = originalColor + blurColor * bloomIntensity * bloomFactor;
    
    return result;
}

// 快速颜色分级
highp vec3 color_grading(highp vec3 color, highp sampler2D lookupTable, highp float intensity) {
    // 简化的3D查找表采样
    highp float blueColor = color.b * 63.0;
    
    highp vec2 quad1;
    quad1.y = floor(floor(blueColor) / 8.0);
    quad1.x = floor(blueColor) - (quad1.y * 8.0);
    
    highp vec2 quad2;
    quad2.y = floor(ceil(blueColor) / 8.0);
    quad2.x = ceil(blueColor) - (quad2.y * 8.0);
    
    highp vec2 texPos1;
    texPos1.x = (quad1.x * 0.125) + 0.5/512.0 + ((0.125 - 1.0/512.0) * color.r);
    texPos1.y = (quad1.y * 0.125) + 0.5/512.0 + ((0.125 - 1.0/512.0) * color.g);
    
    highp vec2 texPos2;
    texPos2.x = (quad2.x * 0.125) + 0.5/512.0 + ((0.125 - 1.0/512.0) * color.r);
    texPos2.y = (quad2.y * 0.125) + 0.5/512.0 + ((0.125 - 1.0/512.0) * color.g);
    
    highp vec4 newColor1 = texture2D(lookupTable, texPos1);
    highp vec4 newColor2 = texture2D(lookupTable, texPos2);
    
    highp vec4 newColor = mix(newColor1, newColor2, fract(blueColor));
    
    // 混合原始颜色和分级颜色
    return mix(color, newColor.rgb, intensity);
}

#endif // POSTPROCESS_GLSL
"""
    
    def shutdown(self):
        """
        关闭着色器管理器，释放资源
        """
        # 清空着色器程序缓存
        self.shader_programs.clear()
        
        # 清空着色器源代码缓存
        self.shader_source_cache.clear()
        
        print("着色器管理器已关闭，所有资源已释放")

# 示例用法
if __name__ == "__main__":
    # 创建着色器管理器实例
    shader_manager = ShaderManager({
        "gpu_name": "NVIDIA GeForce GTX 750Ti"
    })
    
    # 设置编译选项
    shader_manager.set_compile_options({
        "use_half_precision": True,
        "enable_fast_math": True
    })
    
    print("着色器管理器初始化完成")
    print(f"当前架构优化设置: {shader_manager.current_optimizations}")
    print(f"编译选项: {shader_manager.compile_options}")