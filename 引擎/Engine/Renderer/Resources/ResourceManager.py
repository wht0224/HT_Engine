import os
import numpy as np
from collections import OrderedDict
import threading
import queue
import time
from enum import Enum

class ResourceLoadStatus(Enum):
    """
    资源加载状态枚举
    """
    LOADING = "loading"  # 正在加载
    COMPLETED = "completed"  # 加载完成
    FAILED = "failed"  # 加载失败
    PENDING = "pending"  # 等待加载

class ResourceManager:
    """
    资源管理器
    针对低端GPU（如NVIDIA GTX 750Ti和AMD RX 580）优化的资源管理系统
    实现纹理压缩、LOD系统和智能资源加载
    """
    
    def __init__(self):
        # 资源存储字典
        self.textures = {}
        self.meshes = {}
        self.shaders = {}
        self.materials = {}
        
        # 资源引用计数
        self.reference_counts = {}
        
        # 纹理和模型质量设置
        self.texture_quality = "medium"  # low, medium, high
        self.mesh_lod_bias = 1.0  # LOD偏差值，影响何时切换到更低细节的模型
        
        # 内存管理设置
        self.max_vram_usage_mb = 1800  # 最大VRAM使用量（MB），为系统保留一些空间
        self.current_vram_usage_mb = 0  # 当前VRAM使用量
        
        # 纹理压缩设置
        self.texture_compression = "BC7"  # 默认使用BC7压缩
        self.texture_compression_quality = "medium"  # 压缩质量
        
        # 资源优先级队列（用于资源换入换出）
        # 使用更高效的数据结构存储资源优先级信息
        self.resource_priority = {}  # 资源ID -> (last_used_time, resource_type)
        self.resource_last_used = {}  # 资源ID -> 最后使用时间
        self.resource_type_map = {}  # 资源ID -> 资源类型
        
        # 资源加载状态
        self.loading_queue = []
        self.unloading_queue = []
        
        # LOD系统配置
        self.enable_lod_system = True
        self.lod_distances = [10.0, 25.0, 50.0]  # LOD切换距离
        
        # 纹理LOD配置
        self.texture_lod_enabled = True
        self.texture_lod_levels = 4  # 纹理LOD级别数量
        
        # 纹理流配置
        self.texture_streaming_enabled = True
        self.streaming_distance_threshold = 100.0  # 纹理流距离阈值
        
        # 异步加载相关
        self.async_loading_enabled = True  # 是否启用异步加载
        self.loading_threads = []  # 加载线程列表
        self.max_loading_threads = 4  # 最大加载线程数
        self.load_queue = queue.Queue()  # 加载任务队列
        self.load_complete_queue = queue.Queue()  # 加载完成队列
        self.load_status = {}  # 资源加载状态
        self.load_events = {}  # 资源加载完成事件
        self.running = False  # 加载线程运行状态
        self.load_lock = threading.RLock()  # 加载状态锁
        self.resource_lock = threading.RLock()  # 资源访问锁
    
    def initialize(self):
        """
        初始化资源管理器
        """
        print("初始化资源管理器...")
        
        # 设置纹理压缩格式
        self._setup_texture_compression()
        
        # 初始化内存监控
        self._initialize_memory_monitoring()
        
        # 初始化异步加载系统
        self._initialize_async_loading()
        
        print(f"资源管理器初始化完成，最大VRAM使用限制: {self.max_vram_usage_mb}MB")
    
    def _initialize_async_loading(self):
        """
        初始化异步加载系统
        """
        self.running = True
        
        # 创建并启动加载线程
        for i in range(self.max_loading_threads):
            thread = threading.Thread(target=self._loading_thread, name=f"ResourceLoader-{i}", daemon=True)
            self.loading_threads.append(thread)
            thread.start()
        
        print(f"异步资源加载系统已初始化，创建了 {self.max_loading_threads} 个加载线程")
    
    def _loading_thread(self):
        """
        资源加载线程函数
        """
        while self.running:
            try:
                # 从队列中获取加载任务
                task = self.load_queue.get(timeout=1.0)
                
                if task is None:
                    # 结束线程
                    break
                
                # 执行加载任务
                self._process_loading_task(task)
                
                # 标记任务完成
                self.load_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"加载线程错误: {e}")
    
    def _process_loading_task(self, task):
        """
        处理加载任务
        
        Args:
            task: 加载任务字典，包含type, id, path, params等字段
        """
        try:
            resource_type = task["type"]
            resource_id = task["id"]
            
            # 更新加载状态为正在加载
            with self.load_lock:
                self.load_status[resource_id] = ResourceLoadStatus.LOADING
            
            # 执行具体的加载操作
            result = None
            if resource_type == "texture":
                result = self._load_texture_sync(task["path"], resource_id, **task["params"])
            elif resource_type == "mesh":
                result = self._load_mesh_sync(task["path"], resource_id, **task["params"])
            elif resource_type == "shader":
                result = self._load_shader_sync(task["vertex_path"], task["fragment_path"], resource_id, **task["params"])
            elif resource_type == "material":
                result = self._load_material_sync(task["material_data"], resource_id, **task["params"])
            
            # 更新加载状态
            with self.load_lock:
                if result:
                    self.load_status[resource_id] = ResourceLoadStatus.COMPLETED
                    # 将加载完成的资源添加到完成队列
                    self.load_complete_queue.put((resource_type, resource_id))
                else:
                    self.load_status[resource_id] = ResourceLoadStatus.FAILED
            
        except Exception as e:
            print(f"处理加载任务失败: {task}, 错误: {e}")
            with self.load_lock:
                self.load_status[task["id"]] = ResourceLoadStatus.FAILED
    
    def update(self, delta_time):
        """
        更新资源管理器，处理加载完成的资源
        
        Args:
            delta_time: 帧间隔时间（秒）
        """
        # 处理加载完成的资源
        self._process_loaded_resources()
        
        # 处理加载队列
        self._process_loading_queue()
        
        # 处理卸载队列
        self._process_unloading_queue()
    
    def _process_loaded_resources(self):
        """
        处理加载完成的资源
        """
        while True:
            try:
                # 从完成队列中获取已加载的资源
                resource_type, resource_id = self.load_complete_queue.get_nowait()
                
                # 触发资源加载完成事件
                with self.load_lock:
                    if resource_id in self.load_events:
                        for event in self.load_events[resource_id]:
                            try:
                                event(resource_id)
                            except Exception as e:
                                print(f"触发资源加载完成事件失败: {e}")
                        # 移除事件
                        del self.load_events[resource_id]
                
                print(f"资源加载完成: {resource_type} {resource_id}")
            except queue.Empty:
                break
            except Exception as e:
                print(f"处理已加载资源失败: {e}")
    
    def _process_loading_queue(self):
        """
        处理加载队列
        """
        # 这里可以添加队列处理逻辑，如优先级调整等
        pass
    
    def _process_unloading_queue(self):
        """
        处理卸载队列
        """
        # 这里可以添加队列处理逻辑，如优先级调整等
        pass
    
    def shutdown(self):
        """
        关闭资源管理器，释放所有资源
        """
        print("关闭资源管理器...")
        
        # 停止异步加载线程
        self.running = False
        
        # 向加载队列中添加结束信号
        for _ in range(self.max_loading_threads):
            self.load_queue.put(None)
        
        # 等待所有加载线程结束
        for thread in self.loading_threads:
            thread.join()
        
        # 释放所有纹理
        for texture_id in list(self.textures.keys()):
            self._free_texture(texture_id)
        
        # 释放所有网格
        for mesh_id in list(self.meshes.keys()):
            self._free_mesh(mesh_id)
        
        # 释放所有着色器
        for shader_id in list(self.shaders.keys()):
            self._free_shader(shader_id)
        
        # 释放所有材质
        for material_id in list(self.materials.keys()):
            self._free_material(material_id)
        
        # 清空所有字典
        self.textures.clear()
        self.meshes.clear()
        self.shaders.clear()
        self.materials.clear()
        self.reference_counts.clear()
        self.resource_priority.clear()
        self.load_status.clear()
        self.load_events.clear()
        
        self.current_vram_usage_mb = 0
        print("资源管理器已关闭，所有资源已释放")
    
    def load_shader(self, vertex_path, fragment_path, shader_id=None, async_load=True):
        """
        加载着色器程序
        
        Args:
            vertex_path: 顶点着色器文件路径
            fragment_path: 片段着色器文件路径
            shader_id: 可选的着色器ID
            async_load: 是否异步加载
            
        Returns:
            str: 着色器ID
        """
        # 如果没有提供ID，使用路径组合作为ID
        if shader_id is None:
            shader_id = f"{vertex_path}_{fragment_path}"
        
        # 检查着色器是否已加载
        if shader_id in self.shaders:
            # 增加引用计数
            with self.resource_lock:
                self.reference_counts[shader_id] += 1
            return shader_id
        
        # 检查是否正在加载
        with self.load_lock:
            if shader_id in self.load_status:
                # 增加引用计数
                if shader_id not in self.reference_counts:
                    self.reference_counts[shader_id] = 1
                else:
                    self.reference_counts[shader_id] += 1
                return shader_id
        
        if self.async_loading_enabled and async_load:
            # 异步加载
            with self.load_lock:
                self.load_status[shader_id] = ResourceLoadStatus.PENDING
                self.reference_counts[shader_id] = 1
            
            # 创建加载任务
            task = {
                "type": "shader",
                "id": shader_id,
                "vertex_path": vertex_path,
                "fragment_path": fragment_path,
                "params": {}
            }
            
            # 将任务添加到加载队列
            self.load_queue.put(task)
            print(f"异步加载着色器: {shader_id}")
        else:
            # 同步加载
            self._load_shader_sync(vertex_path, fragment_path, shader_id)
        
        return shader_id
    
    def _load_shader_sync(self, vertex_path, fragment_path, shader_id):
        """
        同步加载着色器程序
        
        Args:
            vertex_path: 顶点着色器文件路径
            fragment_path: 片段着色器文件路径
            shader_id: 着色器ID
            
        Returns:
            bool: 是否加载成功
        """
        try:
            print(f"同步加载着色器: {shader_id}")
            
            # 加载着色器源代码
            vertex_code = self.load_shader_source(vertex_path)
            fragment_code = self.load_shader_source(fragment_path)
            
            # 创建着色器程序
            # 实际实现会使用平台特定的API创建着色器程序
            shader_program = {
                "vertex_code": vertex_code,
                "fragment_code": fragment_code,
                "id": shader_id
            }
            
            # 计算着色器内存使用量（估计）
            memory_usage = 0.1  # 估计值
            
            # 检查内存使用限制
            with self.resource_lock:
                if self._would_exceed_memory_limit(memory_usage):
                    # 释放一些低优先级资源
                    self._free_resources_to_make_space(memory_usage)

                # 更新VRAM使用量
                self.current_vram_usage_mb += memory_usage
                
                # 存储着色器
                self.shaders[shader_id] = shader_program
                if shader_id not in self.reference_counts:
                    self.reference_counts[shader_id] = 1
                
                # 更新资源优先级
                self._update_resource_priority(shader_id, "shader")
            
            return True
        except Exception as e:
            print(f"同步加载着色器失败: {shader_id}, 错误: {e}")
            return False
    
    def load_shader_source(self, shader_path):
        """
        加载着色器源代码
        
        Args:
            shader_path: 着色器文件路径
            
        Returns:
            str: 着色器源代码
        """
        try:
            # 确保路径是绝对路径
            if not os.path.isabs(shader_path):
                shader_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), shader_path)
            
            # 读取着色器文件，使用utf-8编码
            with open(shader_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"无法加载着色器源代码: {shader_path}, 错误: {e}")
            # 返回默认的PBR着色器代码
            if shader_path.endswith('.vert') or 'Vertex' in shader_path:
                return """
                #version 330 core
                layout(location = 0) in vec3 position;
                layout(location = 1) in vec3 normal;
                layout(location = 2) in vec2 texcoord;
                
                uniform mat4 model;
                uniform mat4 view;
                uniform mat4 projection;
                uniform mat3 normal_matrix;
                
                out vec3 frag_position;
                out vec3 frag_normal;
                out vec2 frag_texcoord;
                
                void main() {
                    gl_Position = projection * view * model * vec4(position, 1.0);
                    frag_position = vec3(model * vec4(position, 1.0));
                    frag_normal = normal_matrix * normal;
                    frag_texcoord = texcoord;
                }
                """
            else:
                return """
                #version 330 core
                
                in vec3 frag_position;
                in vec3 frag_normal;
                in vec2 frag_texcoord;
                
                uniform vec3 view_position;
                uniform vec3 light_position;
                uniform vec3 light_color;
                
                // PBR材质参数
                uniform vec3 u_baseColor;
                uniform float u_roughness;
                uniform float u_metallic;
                uniform float u_ao;
                uniform vec3 u_emissive;
                uniform float u_emissiveStrength;
                
                // 纹理采样器
                uniform sampler2D u_baseColorTexture;
                uniform sampler2D u_roughnessTexture;
                uniform sampler2D u_metallicTexture;
                uniform sampler2D u_normalTexture;
                uniform sampler2D u_aoTexture;
                uniform sampler2D u_emissiveTexture;
                
                out vec4 frag_color;
                
                // PBR相关函数
                vec3 fresnelSchlick(float cosTheta, vec3 F0) {
                    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
                }
                
                float DistributionGGX(vec3 N, vec3 H, float roughness) {
                    float a = roughness * roughness;
                    float a2 = a * a;
                    float NdotH = max(dot(N, H), 0.0);
                    float NdotH2 = NdotH * NdotH;
                    
                    float nom = a2;
                    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
                    denom = 3.14159265 * denom * denom;
                    
                    return nom / denom;
                }
                
                float GeometrySchlickGGX(float NdotV, float roughness) {
                    float r = (roughness + 1.0);
                    float k = (r * r) / 8.0;
                    
                    float nom = NdotV;
                    float denom = NdotV * (1.0 - k) + k;
                    
                    return nom / denom;
                }
                
                float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
                    float NdotV = max(dot(N, V), 0.0);
                    float NdotL = max(dot(N, L), 0.0);
                    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
                    float ggx1 = GeometrySchlickGGX(NdotL, roughness);
                    
                    return ggx1 * ggx2;
                }
                
                void main() {
                    // 标准化法线
                    vec3 N = normalize(frag_normal);
                    vec3 V = normalize(view_position - frag_position);
                    vec3 L = normalize(light_position - frag_position);
                    vec3 H = normalize(V + L);
                    
                    // 基础颜色
                    vec3 baseColor = u_baseColor;
                    float roughness = u_roughness;
                    float metallic = u_metallic;
                    float ao = u_ao;
                    
                    // 计算反射率
                    vec3 F0 = vec3(0.04);
                    F0 = mix(F0, baseColor, metallic);
                    
                    // 计算光照
                    vec3 radiance = light_color * max(dot(N, L), 0.0);
                    
                    // Cook-Torrance BRDF
                    float NDF = DistributionGGX(N, H, roughness);
                    float G = GeometrySmith(N, V, L, roughness);
                    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
                    
                    vec3 numerator = NDF * G * F;
                    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
                    vec3 specular = numerator / denominator;
                    
                    // 漫反射和镜面反射比例
                    vec3 kS = F;
                    vec3 kD = vec3(1.0) - kS;
                    kD *= 1.0 - metallic;
                    
                    // 最终光照
                    vec3 Lo = (kD * baseColor / 3.14159265 + specular) * radiance;
                    
                    // 环境光
                    vec3 ambient = vec3(0.03) * baseColor * ao;
                    
                    // 自发光
                    vec3 emissive = u_emissive * u_emissiveStrength;
                    
                    vec3 color = ambient + Lo + emissive;
                    
                    // HDR色调映射
                    color = color / (color + vec3(1.0));
                    
                    // 伽马校正
                    color = pow(color, vec3(1.0/2.2));
                    
                    frag_color = vec4(color, 1.0);
                }
                """
    
    def load_texture(self, texture_path, texture_id=None, compress_texture=True, mipmaps=True, async_load=True):
        """
        加载纹理，应用压缩和质量设置
        
        Args:
            texture_path: 纹理文件路径
            texture_id: 可选的纹理ID
            compress_texture: 是否压缩纹理
            mipmaps: 是否生成mipmap
            async_load: 是否异步加载
            
        Returns:
            str: 纹理ID
        """
        # 如果没有提供ID，使用路径作为ID
        if texture_id is None:
            texture_id = texture_path
        
        # 检查纹理是否已加载
        if texture_id in self.textures:
            # 增加引用计数
            with self.resource_lock:
                self.reference_counts[texture_id] += 1
            return texture_id
        
        # 检查是否正在加载
        with self.load_lock:
            if texture_id in self.load_status:
                # 增加引用计数
                if texture_id not in self.reference_counts:
                    self.reference_counts[texture_id] = 1
                else:
                    self.reference_counts[texture_id] += 1
                return texture_id
        
        if self.async_loading_enabled and async_load:
            # 异步加载
            with self.load_lock:
                self.load_status[texture_id] = ResourceLoadStatus.PENDING
                self.reference_counts[texture_id] = 1
            
            # 创建加载任务
            task = {
                "type": "texture",
                "id": texture_id,
                "path": texture_path,
                "params": {
                    "compress_texture": compress_texture,
                    "mipmaps": mipmaps
                }
            }
            
            # 将任务添加到加载队列
            self.load_queue.put(task)
            print(f"异步加载纹理: {texture_id}")
        else:
            # 同步加载
            self._load_texture_sync(texture_path, texture_id, compress_texture, mipmaps)
        
        return texture_id
    
    def _load_texture_sync(self, texture_path, texture_id, compress_texture=True, mipmaps=True):
        """
        同步加载纹理
        
        Args:
            texture_path: 纹理文件路径
            texture_id: 纹理ID
            compress_texture: 是否压缩纹理
            mipmaps: 是否生成mipmap
            
        Returns:
            bool: 是否加载成功
        """
        try:
            print(f"同步加载纹理: {texture_id}")
            
            # 应用纹理质量设置
            actual_path = self._get_texture_path_for_quality(texture_path)
            
            # 加载纹理数据
            # 实际实现会使用平台特定的纹理加载API
            texture_data = self._load_texture_data(actual_path)
            
            # 生成mipmap
            if mipmaps:
                texture_data = self._generate_mipmaps(texture_data)
            
            # 应用纹理压缩
            if compress_texture:
                compressed_texture = self._compress_texture(texture_data)
            else:
                compressed_texture = texture_data
            
            # 计算纹理内存使用量
            memory_usage = self._calculate_texture_memory_usage(compressed_texture)
            
            with self.resource_lock:
                # 检查内存使用限制
                if self._would_exceed_memory_limit(memory_usage):
                    # 释放一些低优先级资源
                    self._free_resources_to_make_space(memory_usage)
                
                # 更新VRAM使用量
                self.current_vram_usage_mb += memory_usage
                
                # 存储纹理
                self.textures[texture_id] = compressed_texture
                if texture_id not in self.reference_counts:
                    self.reference_counts[texture_id] = 1
                
                # 更新资源优先级
                self._update_resource_priority(texture_id, "texture")
            
            return True
        except Exception as e:
            print(f"同步加载纹理失败: {texture_id}, 错误: {e}")
            return False
    
    def unload_texture(self, texture_id):
        """
        卸载纹理（减少引用计数，当引用计数为0时释放）
        
        Args:
            texture_id: 纹理ID
        """
        if texture_id not in self.textures:
            return
        
        # 减少引用计数
        self.reference_counts[texture_id] -= 1
        
        # 如果引用计数为0，释放纹理
        if self.reference_counts[texture_id] <= 0:
            self._free_texture(texture_id)
    
    def load_mesh(self, mesh_path, mesh_id=None, async_load=True):
        """
        加载网格，生成LOD
        
        Args:
            mesh_path: 网格文件路径
            mesh_id: 可选的网格ID
            async_load: 是否异步加载
            
        Returns:
            str: 网格ID
        """
        # 如果没有提供ID，使用路径作为ID
        if mesh_id is None:
            mesh_id = mesh_path
        
        # 检查网格是否已加载
        if mesh_id in self.meshes:
            # 增加引用计数
            with self.resource_lock:
                self.reference_counts[mesh_id] += 1
            return mesh_id
        
        # 检查是否正在加载
        with self.load_lock:
            if mesh_id in self.load_status:
                # 增加引用计数
                if mesh_id not in self.reference_counts:
                    self.reference_counts[mesh_id] = 1
                else:
                    self.reference_counts[mesh_id] += 1
                return mesh_id
        
        if self.async_loading_enabled and async_load:
            # 异步加载
            with self.load_lock:
                self.load_status[mesh_id] = ResourceLoadStatus.PENDING
                self.reference_counts[mesh_id] = 1
            
            # 创建加载任务
            task = {
                "type": "mesh",
                "id": mesh_id,
                "path": mesh_path,
                "params": {}
            }
            
            # 将任务添加到加载队列
            self.load_queue.put(task)
            print(f"异步加载网格: {mesh_id}")
        else:
            # 同步加载
            self._load_mesh_sync(mesh_path, mesh_id)
        
        return mesh_id
    
    def _load_mesh_sync(self, mesh_path, mesh_id):
        """
        同步加载网格
        
        Args:
            mesh_path: 网格文件路径
            mesh_id: 网格ID
            
        Returns:
            bool: 是否加载成功
        """
        try:
            print(f"同步加载网格: {mesh_id}")
            
            # 加载网格数据
            mesh_data = self._load_mesh_data(mesh_path)
            
            # 生成LOD
            if self.enable_lod_system:
                mesh_data_with_lod = self._generate_lod(mesh_data)
            else:
                mesh_data_with_lod = {"lod_0": mesh_data}
            
            # 计算网格内存使用量
            memory_usage = self._calculate_mesh_memory_usage(mesh_data_with_lod)
            
            with self.resource_lock:
                # 检查内存使用限制
                if self._would_exceed_memory_limit(memory_usage):
                    # 释放一些低优先级资源
                    self._free_resources_to_make_space(memory_usage)
                
                # 更新VRAM使用量
                self.current_vram_usage_mb += memory_usage
                
                # 存储网格
                self.meshes[mesh_id] = mesh_data_with_lod
                if mesh_id not in self.reference_counts:
                    self.reference_counts[mesh_id] = 1
                
                # 更新资源优先级
                self._update_resource_priority(mesh_id, "mesh")
            
            return True
        except Exception as e:
            print(f"同步加载网格失败: {mesh_id}, 错误: {e}")
            return False
    
    def unload_mesh(self, mesh_id):
        """
        卸载网格（减少引用计数，当引用计数为0时释放）
        
        Args:
            mesh_id: 网格ID
        """
        if mesh_id not in self.meshes:
            return
        
        # 减少引用计数
        self.reference_counts[mesh_id] -= 1
        
        # 如果引用计数为0，释放网格
        if self.reference_counts[mesh_id] <= 0:
            self._free_mesh(mesh_id)
    
    def set_texture_quality(self, quality):
        """
        设置纹理质量
        
        Args:
            quality: 质量级别 (low/medium/high)
        """
        if quality in ["low", "medium", "high"]:
            self.texture_quality = quality
            print(f"纹理质量设置为: {quality}")
            
            # 根据新质量重新加载所有纹理
            if self.textures:  # 只有在已初始化后才重新加载
                self._reload_all_textures_for_quality(quality)
    
    def load_material(self, material_data, material_id=None, async_load=True):
        """
        加载材质资源
        
        Args:
            material_data: 材质数据
            material_id: 可选的材质ID
            async_load: 是否异步加载
            
        Returns:
            str: 材质ID
        """
        # 如果没有提供ID，使用材质数据的哈希作为ID
        if material_id is None:
            material_id = f"material_{hash(str(material_data))}"
        
        # 检查材质是否已加载
        if material_id in self.materials:
            # 增加引用计数
            with self.resource_lock:
                self.reference_counts[material_id] += 1
            return material_id
        
        # 检查是否正在加载
        with self.load_lock:
            if material_id in self.load_status:
                # 增加引用计数
                if material_id not in self.reference_counts:
                    self.reference_counts[material_id] = 1
                else:
                    self.reference_counts[material_id] += 1
                return material_id
        
        if self.async_loading_enabled and async_load:
            # 异步加载
            with self.load_lock:
                self.load_status[material_id] = ResourceLoadStatus.PENDING
                self.reference_counts[material_id] = 1
            
            # 创建加载任务
            task = {
                "type": "material",
                "id": material_id,
                "material_data": material_data,
                "params": {}
            }
            
            # 将任务添加到加载队列
            self.load_queue.put(task)
            print(f"异步加载材质: {material_id}")
        else:
            # 同步加载
            self._load_material_sync(material_data, material_id)
        
        return material_id
    
    def _load_material_sync(self, material_data, material_id):
        """
        同步加载材质
        
        Args:
            material_data: 材质数据
            material_id: 材质ID
            
        Returns:
            bool: 是否加载成功
        """
        try:
            print(f"同步加载材质: {material_id}")
            
            # 计算材质内存使用量（估计）
            memory_usage = 0.05  # 估计值
            
            with self.resource_lock:
                # 检查内存使用限制
                if self._would_exceed_memory_limit(memory_usage):
                    # 释放一些低优先级资源
                    self._free_resources_to_make_space(memory_usage)
                
                # 更新VRAM使用量
                self.current_vram_usage_mb += memory_usage
                
                # 存储材质
                self.materials[material_id] = material_data
                if material_id not in self.reference_counts:
                    self.reference_counts[material_id] = 1
                
                # 更新资源优先级
                self._update_resource_priority(material_id, "material")
            
            return True
        except Exception as e:
            print(f"同步加载材质失败: {material_id}, 错误: {e}")
            return False
    
    def unload_material(self, material_id):
        """
        卸载材质（减少引用计数，当引用计数为0时释放）
        
        Args:
            material_id: 材质ID
        """
        if material_id not in self.materials:
            return
        
        # 减少引用计数
        self.reference_counts[material_id] -= 1
        
        # 如果引用计数为0，释放材质
        if self.reference_counts[material_id] <= 0:
            self._free_material(material_id)
    
    def set_mesh_lod_bias(self, bias):
        """
        设置网格LOD偏差
        
        Args:
            bias: LOD偏差值，正值使LOD切换更早（使用更低细节的模型）
        """
        self.mesh_lod_bias = bias
        print(f"网格LOD偏差设置为: {bias}")
    
    def update_resource_priorities(self, camera_position, scene_objects):
        """
        根据摄像机位置和场景对象更新资源优先级
        用于决定哪些资源应该保留在内存中
        
        Args:
            camera_position: 摄像机位置
            scene_objects: 场景中的对象列表
        """
        # 简化实现
        # 实际实现会根据到摄像机的距离、可见性等因素更新资源优先级
        pass
    
    def _setup_texture_compression(self):
        """
        设置纹理压缩格式
        根据硬件能力选择最合适的压缩格式
        """
        # 这里可以根据平台检测结果选择最合适的压缩格式
        # 对于我们的目标硬件，BC7在NVIDIA上效果好，ETC2在移动平台上通用
        pass
    
    def _initialize_memory_monitoring(self):
        """
        初始化内存监控
        """
        # 设置内存使用限制
        # 这里可以根据实际检测到的VRAM大小设置更精确的限制
        pass
    
    def _get_texture_path_for_quality(self, base_path):
        """
        根据质量设置获取适当的纹理路径
        假设目录结构中包含不同质量级别的纹理文件夹
        
        Args:
            base_path: 基础纹理路径
            
        Returns:
            str: 调整后的纹理路径
        """
        # 简化实现
        # 实际实现可能会根据质量设置选择不同分辨率的纹理文件
        return base_path
    
    def _load_texture_data(self, texture_path):
        """
        加载纹理数据
        
        Args:
            texture_path: 纹理文件路径
            
        Returns:
            object: 纹理数据
        """
        # 简化实现
        # 实际实现会使用如Pillow或其他库加载图像数据
        return {"path": texture_path}
    
    def _generate_mipmaps(self, texture_data):
        """
        生成纹理mipmap
        
        Args:
            texture_data: 原始纹理数据
            
        Returns:
            object: 包含mipmap的纹理数据
        """
        print("生成纹理mipmap...")
        
        # 简化实现
        # 实际实现会生成多级mipmap
        texture_data["mipmaps"] = True
        texture_data["mipmap_levels"] = self.texture_lod_levels
        
        return texture_data
    
    def _compress_texture(self, texture_data):
        """
        压缩纹理数据
        
        Args:
            texture_data: 原始纹理数据
            
        Returns:
            object: 压缩后的纹理数据
        """
        # 简化实现
        # 实际实现会根据选择的压缩格式压缩纹理
        # 针对低端GPU，我们会优先使用BC7、ETC2等硬件加速的压缩格式
        print(f"压缩纹理使用格式: {self.texture_compression}, 质量: {self.texture_compression_quality}")
        
        # 添加压缩信息
        texture_data["compressed"] = True
        texture_data["compression_format"] = self.texture_compression
        texture_data["compression_quality"] = self.texture_compression_quality
        
        return texture_data
    
    def _calculate_texture_memory_usage(self, texture_data):
        """
        计算纹理内存使用量
        
        Args:
            texture_data: 纹理数据
            
        Returns:
            float: 内存使用量（MB）
        """
        # 简化实现，返回估计值
        # 实际实现会根据纹理尺寸和格式精确计算内存使用量
        if self.texture_quality == "high":
            return 16.0  # 估计4K纹理的内存使用量
        elif self.texture_quality == "medium":
            return 4.0  # 估计2K纹理的内存使用量
        else:
            return 1.0  # 估计1K纹理的内存使用量
    
    def _calculate_mesh_memory_usage(self, mesh_data_with_lod):
        """
        计算网格内存使用量
        
        Args:
            mesh_data_with_lod: 包含LOD的网格数据
            
        Returns:
            float: 内存使用量（MB）
        """
        # 简化实现，返回估计值
        # 实际实现会根据顶点数量、索引数量等精确计算内存使用量
        return 2.0  # 估计值
    
    def _would_exceed_memory_limit(self, additional_memory_mb):
        """
        检查是否会超过内存限制
        
        Args:
            additional_memory_mb: 额外的内存使用量（MB）
            
        Returns:
            bool: 是否会超过限制
        """
        return (self.current_vram_usage_mb + additional_memory_mb) > self.max_vram_usage_mb
    
    def _free_resources_to_make_space(self, required_space_mb):
        """
        释放低优先级资源以腾出空间
        
        Args:
            required_space_mb: 需要腾出的空间（MB）
        """
        print(f"需要释放内存空间: {required_space_mb}MB")
        
        # 计算需要释放的总空间
        target_space = self.current_vram_usage_mb + required_space_mb - self.max_vram_usage_mb
        freed_space = 0.0
        
        # 如果当前使用量没有超过限制，无需释放
        if target_space <= 0:
            return
        
        # 收集所有可卸载的资源，并按优先级排序（最久未使用的资源优先）
        current_time = time.time()
        resources_to_consider = []
        
        for resource_id, (last_used, resource_type) in self.resource_priority.items():
            # 跳过正在使用的资源
            if self.reference_counts.get(resource_id, 0) > 1:
                continue
            
            # 计算资源的优先级分数（最久未使用的分数最高）
            priority_score = current_time - last_used
            resources_to_consider.append((priority_score, resource_id, resource_type))
        
        # 按优先级分数降序排序（先释放最久未使用的资源）
        resources_to_consider.sort(key=lambda x: x[0], reverse=True)
        
        # 选择要卸载的资源
        resources_to_unload = []
        for priority_score, resource_id, resource_type in resources_to_consider:
            # 估计释放的空间
            if resource_type == "texture":
                freed_space += self._calculate_texture_memory_usage({})
            elif resource_type == "mesh":
                freed_space += self._calculate_mesh_memory_usage({})
            
            # 记录要卸载的资源
            resources_to_unload.append((resource_id, resource_type))
            
            # 如果已经释放了足够的空间，停止
            if freed_space >= target_space:
                break
        
        # 卸载选定的资源
        for resource_id, resource_type in resources_to_unload:
            print(f"自动卸载低优先级资源: {resource_id} ({resource_type})")
            if resource_type == "texture":
                self._free_texture(resource_id)
            elif resource_type == "mesh":
                self._free_mesh(resource_id)
    
    def _free_texture(self, texture_id):
        """
        释放纹理资源
        
        Args:
            texture_id: 纹理ID
        """
        if texture_id not in self.textures:
            return
        
        # 计算释放的内存
        memory_usage = self._calculate_texture_memory_usage(self.textures[texture_id])
        
        # 释放纹理资源
        del self.textures[texture_id]
        del self.reference_counts[texture_id]
        
        # 从所有优先级数据结构中移除
        if texture_id in self.resource_priority:
            del self.resource_priority[texture_id]
        if texture_id in self.resource_last_used:
            del self.resource_last_used[texture_id]
        if texture_id in self.resource_type_map:
            del self.resource_type_map[texture_id]
        
        # 更新VRAM使用量
        self.current_vram_usage_mb -= memory_usage
        
        print(f"释放纹理: {texture_id}, 释放内存: {memory_usage:.2f}MB")
    
    def _free_mesh(self, mesh_id):
        """
        释放网格资源
        
        Args:
            mesh_id: 网格ID
        """
        if mesh_id not in self.meshes:
            return
        
        # 计算释放的内存
        memory_usage = self._calculate_mesh_memory_usage(self.meshes[mesh_id])
        
        # 释放网格资源
        del self.meshes[mesh_id]
        del self.reference_counts[mesh_id]
        
        # 从所有优先级数据结构中移除
        if mesh_id in self.resource_priority:
            del self.resource_priority[mesh_id]
        if mesh_id in self.resource_last_used:
            del self.resource_last_used[mesh_id]
        if mesh_id in self.resource_type_map:
            del self.resource_type_map[mesh_id]
        
        # 更新VRAM使用量
        self.current_vram_usage_mb -= memory_usage
        
        print(f"释放网格: {mesh_id}, 释放内存: {memory_usage:.2f}MB")
    
    def create_static_batch(self, nodes):
        """
        创建静态批处理
        
        Args:
            nodes: 要合并的节点列表
            
        Returns:
            str: 批处理ID
        """
        # 检查节点列表是否为空
        if not nodes:
            return None
        
        # 合并节点的网格数据
        merged_mesh = self._merge_meshes([node.mesh for node in nodes], [node.world_matrix for node in nodes])
        
        # 生成批处理ID
        batch_id = f"batch_{id(merged_mesh)}"
        
        # 存储合并后的网格
        self.meshes[batch_id] = merged_mesh
        self.reference_counts[batch_id] = 1
        
        # 更新资源优先级
        self._update_resource_priority(batch_id, "mesh")
        
        print(f"创建静态批处理: {batch_id}, 包含 {len(nodes)} 个节点")
        return batch_id
    
    def create_instance_batch(self, nodes):
        """
        创建实例化批处理
        
        Args:
            nodes: 要进行实例化渲染的节点列表
            
        Returns:
            str: 实例化批处理ID
        """
        # 检查节点列表是否为空
        if not nodes:
            return None
        
        # 生成实例化批处理ID
        batch_id = f"instance_batch_{id(nodes[0].mesh)}_{id(nodes[0].material)}"
        
        # 存储实例化数据
        instance_data = {
            "mesh": nodes[0].mesh,
            "material": nodes[0].material,
            "instances": [node.world_matrix for node in nodes]
        }
        
        # 存储实例化批处理
        self.meshes[batch_id] = instance_data
        self.reference_counts[batch_id] = 1
        
        # 更新资源优先级
        self._update_resource_priority(batch_id, "mesh")
        
        print(f"创建实例化批处理: {batch_id}, 包含 {len(nodes)} 个实例")
        return batch_id
    
    def _merge_meshes(self, meshes, transforms):
        """
        合并多个网格到一个网格
        
        Args:
            meshes: 要合并的网格列表
            transforms: 每个网格的变换矩阵列表
            
        Returns:
            object: 合并后的网格
        """
        # 简化实现
        # 实际实现会合并顶点、法线、UV等数据，并调整索引
        merged_mesh = {
            "type": "merged_mesh",
            "original_meshes": meshes,
            "transforms": transforms,
            "triangle_count": sum(mesh.triangle_count for mesh in meshes),
            "vertex_count": sum(mesh.vertex_count for mesh in meshes),
            "merged": True,
            "is_batch": True
        }
        
        return merged_mesh
    
    def _free_shader(self, shader_id):
        """
        释放着色器资源
        
        Args:
            shader_id: 着色器ID
        """
        if shader_id not in self.shaders:
            return
        
        # 释放着色器资源
        del self.shaders[shader_id]
        del self.reference_counts[shader_id]
    
    def _free_material(self, material_id):
        """
        释放材质资源
        
        Args:
            material_id: 材质ID
        """
        if material_id not in self.materials:
            return
        
        # 释放材质资源
        del self.materials[material_id]
        del self.reference_counts[material_id]
    
    def _load_mesh_data(self, mesh_path):
        """
        加载网格数据
        
        Args:
            mesh_path: 网格文件路径
            
        Returns:
            object: 网格数据
        """
        # 简化实现
        # 实际实现会使用如PyAssimp或其他库加载3D模型
        return {"path": mesh_path}
    
    def _generate_lod(self, mesh_data):
        """
        为网格生成LOD（细节层次）
        
        Args:
            mesh_data: 原始网格数据
            
        Returns:
            dict: 包含多个LOD级别的网格数据
        """
        print("为网格生成LOD...")
        
        # 简化实现，返回模拟的LOD数据
        # 实际实现会使用网格简化算法，如边坍缩算法
        lod_data = {}
        
        # LOD 0: 原始网格
        lod_data["lod_0"] = mesh_data
        
        # LOD 1: 简化50%
        lod_data["lod_1"] = {"simplified": True, "reduction": 0.5}
        
        # LOD 2: 简化75%
        lod_data["lod_2"] = {"simplified": True, "reduction": 0.75}
        
        # LOD 3: 简化90%
        lod_data["lod_3"] = {"simplified": True, "reduction": 0.9}
        
        return lod_data
    
    def _update_resource_priority(self, resource_id, resource_type):
        """
        更新资源优先级
        更新资源的最后使用时间
        
        Args:
            resource_id: 资源ID
            resource_type: 资源类型
        """
        # 获取当前时间作为最后使用时间
        current_time = time.time()
        
        # 更新资源优先级信息
        self.resource_priority[resource_id] = (current_time, resource_type)
        self.resource_last_used[resource_id] = current_time
        self.resource_type_map[resource_id] = resource_type
    
    def _reload_all_textures_for_quality(self, quality):
        """
        根据新的质量设置重新加载所有纹理
        
        Args:
            quality: 新的质量级别
        """
        # 简化实现
        # 实际实现会重新加载所有已加载的纹理
        pass
    
    def get_memory_usage_stats(self):
        """
        获取内存使用统计信息
        
        Returns:
            dict: 内存使用统计
        """
        return {
            "current_vram_usage_mb": self.current_vram_usage_mb,
            "max_vram_usage_mb": self.max_vram_usage_mb,
            "texture_count": len(self.textures),
            "mesh_count": len(self.meshes),
            "memory_usage_percent": (self.current_vram_usage_mb / self.max_vram_usage_mb) * 100 if self.max_vram_usage_mb > 0 else 0
        }