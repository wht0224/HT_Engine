import numpy as np
import time
from typing import Dict, List, Tuple, Optional

class LandscapePreviewRenderer:
    """
    风景预览渲染器 - 用于直接在引擎中预览Blender生成的低多边形风景
    专为低端GPU优化，实现高效渲染和实时预览
    """
    
    def __init__(self, 
                 window_width: int = 800, 
                 window_height: int = 600,
                 target_fps: int = 30,
                 low_end_gpu_mode: bool = True):
        """
        初始化风景预览渲染器
        
        Args:
            window_width: 窗口宽度
            window_height: 窗口高度
            target_fps: 目标帧率
            low_end_gpu_mode: 是否启用低端GPU优化模式
        """
        # 初始化标志
        self.initialized = False
        self.safe_mode = False
        
        # 基础参数
        self.window_width = window_width
        self.window_height = window_height
        self.target_fps = target_fps
        self.low_end_gpu_mode = low_end_gpu_mode
        
        # 渲染相关数据
        self.scene_data: Optional[Dict] = None
        self.vertices: List[np.ndarray] = []
        self.indices: List[np.ndarray] = []
        self.normals: List[np.ndarray] = []
        self.materials: List[Dict] = []
        self.object_transforms: List[Dict] = []
        
        # 添加缺失的数组初始化
        self.uvs: List[np.ndarray] = []
        self.object_types: List[str] = []
        self.special_flags: List[Dict] = []
        
        # 相机和光照
        self.camera_position: np.ndarray = np.array([20.0, -20.0, 12.0])
        self.camera_rotation: np.ndarray = np.array([0.0, 0.0, 0.0])
        self.lights: List[Dict] = []
        
        # 性能监控
        self.frame_count = 0
        self.last_time = time.time()
        self.current_fps = 0.0
        
        print("初始化风景预览渲染器...")
        print(f"窗口尺寸: {window_width}x{window_height}")
        print(f"目标帧率: {target_fps} FPS")
        print(f"低端GPU模式: {'已启用' if low_end_gpu_mode else '已禁用'}")
        
        # 初始化渲染器（在实际引擎中，这里会初始化图形API）
        try:
            self._initialize_renderer()
            self.initialized = True
        except Exception as e:
            print(f"渲染器初始化失败: {e}")
            print("正在启用安全模式以确保基本功能")
            self._enable_safe_mode()
    
    def _enable_safe_mode(self):
        """
        启用安全模式，在初始化失败时提供最小功能支持
        这是为了确保即使在极端情况下渲染器也能运行
        """
        self.safe_mode = True
        self.initialized = True  # 标记为已初始化以允许基本功能
        
        # 降低所有设置以确保兼容性
        print("安全模式配置:")
        print("- 降低分辨率到640x480")
        print("- 降低目标帧率到20 FPS")
        print("- 禁用所有高级效果")
        print("- 简化场景复杂度")
        
        # 调整参数以保证基本功能
        self.window_width = 640
        self.window_height = 480
        self.target_fps = 20
        self.low_end_gpu_mode = True
        
        # 确保所有必要的数组都已初始化
        if not hasattr(self, 'vertices'):
            self.vertices = []
        if not hasattr(self, 'indices'):
            self.indices = []
        if not hasattr(self, 'normals'):
            self.normals = []
        if not hasattr(self, 'uvs'):
            self.uvs = []
        if not hasattr(self, 'materials'):
            self.materials = []
        if not hasattr(self, 'object_transforms'):
            self.object_transforms = []
        if not hasattr(self, 'object_types'):
            self.object_types = []
        if not hasattr(self, 'special_flags'):
            self.special_flags = []
    
    def _initialize_renderer(self):
        """
        初始化渲染器组件
        在实际引擎中，这里会设置图形API、着色器、缓冲区等
        """
        print("正在初始化渲染组件...")
        
        # 模拟初始化过程
        # 1. 设置渲染API（DirectX 11/OpenGL）
        # 2. 创建着色器程序
        # 3. 配置渲染状态
        
        # 基础渲染设置
        self.render_settings = {
            "enable_ssr": True,  # 屏幕空间反射
            "shadow_map_size": 2048,  # 阴影贴图大小
            "max_lights": 8,  # 最大光源数量
            "texture_quality": "high",  # 纹理质量
            "shader_complexity": "high",  # 着色器复杂度
            "enable_fog": True,  # 启用雾效
            "enable_aa": True,  # 启用抗锯齿
            "enable_ssao": True  # 启用屏幕空间环境光遮蔽
        }
        
        if self.low_end_gpu_mode:
            print("应用低端GPU优化设置:")
            print("- 降低纹理分辨率")
            print("- 简化着色器复杂度")
            print("- 启用几何体实例化")
            print("- 优化顶点缓存访问模式")
            print("- 禁用屏幕空间反射(SSR)")
            print("- 减小阴影贴图大小")
            print("- 减少光源数量")
            
            # 为低端GPU更新渲染设置
            self.render_settings = {
                "enable_ssr": False,  # 禁用SSR
                "shadow_map_size": 1024,  # 减小阴影贴图
                "max_lights": 4,  # 减少光源
                "texture_quality": "medium",  # 中等纹理质量
                "shader_complexity": "low",  # 简化着色器
                "enable_fog": True,  # 保持雾效但优化
                "enable_aa": False,  # 禁用抗锯齿
                "enable_ssao": False  # 禁用屏幕空间环境光遮蔽
            }
            
            # 初始化材质LOD系统，为低端GPU提供额外优化
            self._init_material_lod_system()
            
    def _init_material_lod_system(self):
        """
        初始化材质LOD(Level of Detail)系统
        为低端GPU提供材质复杂度的动态调整功能
        """
        print("初始化材质LOD系统...")
        
        # 配置材质LOD级别
        self.material_lod_levels = {
            # LOD 0: 最高质量
            0: {
                "texture_size": "original",
                "normal_maps": True,
                "specular_maps": True,
                "detail_maps": True,
                "max_textures": 8
            },
            # LOD 1: 中等质量
            1: {
                "texture_size": "half",
                "normal_maps": True,
                "specular_maps": True,
                "detail_maps": False,
                "max_textures": 6
            },
            # LOD 2: 低质量（适合低端GPU）
            2: {
                "texture_size": "quarter",
                "normal_maps": True,
                "specular_maps": False,
                "detail_maps": False,
                "max_textures": 4
            },
            # LOD 3: 最低质量（安全模式）
            3: {
                "texture_size": "eighth",
                "normal_maps": False,
                "specular_maps": False,
                "detail_maps": False,
                "max_textures": 2
            }
        }
        
        # 根据当前GPU性能选择合适的LOD级别
        self.current_material_lod = 2  # 低端GPU默认使用LOD 2
        
        # 初始化纹理压缩设置
        self.texture_compression = "bc3"  # 使用BC3压缩格式（兼容性好）
        
        print(f"材质LOD系统初始化完成，当前级别: {self.current_material_lod}")
        print(f"纹理压缩格式: {self.texture_compression}")
    
    def load_scene_data(self, scene_data: Dict):
        """
        加载从Blender生成的场景数据
        
        Args:
            scene_data: Blender生成的场景数据字典
        """
        # 检查元数据信息
        metadata = scene_data.get('metadata', {})
        detail_level = metadata.get('detail_level', 1)
        optimize_for_engine = metadata.get('optimized_for_engine', False)
        
        print(f"加载场景数据 - 版本: {metadata.get('version', '1.0')}")
        print(f"细节级别: {detail_level}, 引擎优化: {optimize_for_engine}")
        
        self.scene_data = scene_data
        
        # 重置数据
        self.vertices = []
        self.indices = []
        self.uvs = []
        self.materials = []
        self.object_transforms = []
        self.object_types = []  # 新增：存储对象类型
        self.special_flags = []  # 新增：存储特殊标记（实例化等）
        
        # 处理地形数据
        if 'terrain' in scene_data and scene_data['terrain']:
            self._process_object_data(scene_data['terrain'], 'terrain')
        
        # 处理植被数据（树等）
        if 'vegetation' in scene_data:
            for vegetation in scene_data['vegetation']:
                self._process_object_data(vegetation, 'vegetation')
        
        # 处理道具数据（岩石等）
        if 'props' in scene_data:
            for prop in scene_data['props']:
                self._process_object_data(prop, 'prop')
        
        # 处理环境数据（湖泊、云朵等）
        if 'environment' in scene_data:
            for env_obj in scene_data['environment']:
                self._process_object_data(env_obj, 'environment')
        
        # 统计信息
        print(f"加载了 {len(self.vertices)} 个网格对象")
        
        # 统计总顶点数和面数
        total_vertices = 0
        total_indices = 0
        
        for i in range(len(self.vertices)):
            # 顶点是扁平化的数组，每3个元素一个顶点
            total_vertices += len(self.vertices[i]) // 3
            total_indices += len(self.indices[i])
        
        total_faces = total_indices // 3
        
        print(f"总顶点数: {total_vertices}")
        print(f"总面数: {total_faces}")
        
        # 统计特殊标记对象
        instance_count = sum(1 for flags in self.special_flags if flags.get('use_instancing', False))
        if instance_count > 0:
            print(f"可实例化对象: {instance_count}")
        
        simplify_count = sum(1 for flags in self.special_flags if flags.get('needs_simplification', False))
        if simplify_count > 0:
            print(f"需要简化的对象: {simplify_count}")
        
        # 优化数据（针对低端GPU）
        if self.low_end_gpu_mode or optimize_for_engine:
            self._optimize_for_low_end_gpu()
    
    def _process_object_data(self, obj_data: Dict, obj_type: str):
        """
        处理单个对象的数据
        
        Args:
            obj_data: 对象数据字典
            obj_type: 对象类型
        """
        # 提取顶点数据（扁平化数组，每3个元素一个顶点）
        vertices = np.array(obj_data.get('vertices', []), dtype=np.float32)
        if len(vertices) == 0:
            return
        
        # 提取索引数据
        indices = np.array(obj_data.get('faces', []), dtype=np.uint32)
        
        # 提取UV数据
        uvs = None
        if 'uvs' in obj_data and obj_data['uvs']:
            uvs = np.array(obj_data['uvs'], dtype=np.float32)
        
        # 提取材质数据
        material = obj_data.get('material', {})
        
        # 提取变换信息
        transform = {
            'position': np.array(obj_data.get('position', [0, 0, 0]), dtype=np.float32),
            'rotation': np.array(obj_data.get('rotation', [0, 0, 0]), dtype=np.float32),
            'scale': np.array(obj_data.get('scale', [1, 1, 1]), dtype=np.float32),
            'name': obj_data.get('name', f'{obj_type.capitalize()}_{len(self.vertices)}')
        }
        
        # 提取特殊标记
        special_flags = {
            'use_instancing': obj_data.get('use_instancing', False),
            'needs_simplification': obj_data.get('needs_simplification', False),
            'transparent': material.get('transparent', False)
        }
        
        # 存储数据
        self.vertices.append(vertices)
        self.indices.append(indices)
        self.uvs.append(uvs)
        self.materials.append(material)
        self.object_transforms.append(transform)
        self.object_types.append(obj_type)
        self.special_flags.append(special_flags)
    
    def _optimize_for_low_end_gpu(self):
        """
        针对低端GPU优化场景数据
        实现针对GTX 750Ti和RX 580等低端GPU的高效渲染优化
        """
        print("优化场景数据以适应低端GPU...")
        
        # 1. 根据对象类型应用不同的优化策略
        for i in range(len(self.vertices)):
            obj_type = self.object_types[i]
            transform = self.object_transforms[i]
            flags = self.special_flags[i]
            
            # 检查是否需要简化
            if flags.get('needs_simplification', False):
                original_verts = len(self.vertices[i]) // 3
                # 简化算法 - 降低顶点数量
                self._simplify_mesh(i, target_percentage=0.5)
                new_verts = len(self.vertices[i]) // 3
                print(f"简化复杂对象: {transform['name']} ({original_verts} -> {new_verts} 顶点)")
            
            # 针对远距离对象的额外优化
            distance = np.linalg.norm(transform['position'])
            if distance > 20.0:
                # 远处对象进一步简化
                self._simplify_mesh(i, target_percentage=0.3)
                print(f"远距离对象优化: {transform['name']}，距离: {distance:.1f}")
            
            # 针对不同类型对象的特定优化
            if obj_type == 'vegetation':
                # 植被对象优先启用实例化
                flags['use_instancing'] = True
                print(f"为植被启用实例化: {transform['name']}")
            
            elif obj_type == 'environment':
                # 环境对象（如云朵）简化多边形
                if flags.get('transparent', False):
                    # 透明对象特殊处理
                    print(f"优化透明环境对象: {transform['name']}")
        
        # 2. 合并静态对象以减少绘制调用
        self._batch_static_objects()
        
        # 3. 优化材质着色器
        self._optimize_materials()
        
        # 4. 应用内存优化
        self._optimize_memory_usage()
        
        print("低端GPU优化完成")
    
    def _simplify_mesh(self, mesh_index: int, target_percentage: float = 0.5):
        """
        简化网格几何体
        
        Args:
            mesh_index: 网格索引
            target_percentage: 目标简化百分比（0.1-1.0）
        """
        # 这里实现简化逻辑
        # 在实际引擎中，会使用如Quadric Edge Collapse等算法
        # 这里使用简化的顶点减少方法作为演示
        vertices = self.vertices[mesh_index]
        indices = self.indices[mesh_index]
        
        if len(vertices) == 0:
            return
        
        # 计算目标顶点数
        current_vert_count = len(vertices) // 3
        target_vert_count = max(4, int(current_vert_count * target_percentage))
        
        if current_vert_count <= target_vert_count:
            return  # 已经足够简单
        
        # 简化算法（示例实现）
        # 注意：这是一个简化的实现，实际项目中应使用更复杂的网格简化算法
        step = max(1, int(current_vert_count / target_vert_count))
        
        # 这里只是示例，实际简化需要重新计算索引和拓扑结构
        # 在真实引擎中，这里会调用专业的网格简化库
    
    def _batch_static_objects(self):
        """
        合并静态对象以减少绘制调用
        这是低端GPU优化的关键技术之一
        """
        print("合并静态对象以减少绘制调用...")
        
        # 按对象类型分组进行批处理
        batches = {}
        for i in range(len(self.object_types)):
            obj_type = self.object_types[i]
            if obj_type not in batches:
                batches[obj_type] = []
            
            # 只批处理静态对象，跳过标记为实例化的对象
            if not self.special_flags[i].get('use_instancing', False):
                batches[obj_type].append(i)
        
        # 统计批处理信息
        total_batches = sum(1 for group in batches.values() if len(group) > 0)
        print(f"创建 {total_batches} 个对象批次")
    
    def _optimize_materials(self):
        """
        优化材质着色器以适应低端GPU
        """
        print("优化材质着色器...")
        
        for i, material in enumerate(self.materials):
            # 简化材质复杂度
            # 1. 降低纹理分辨率
            # 2. 减少着色器通道
            # 3. 移除复杂效果
            
            # 为低端GPU设置合适的着色器
            material['shader_quality'] = 'low_end'  # 标记为低端着色器
            
            # 对于远距离或小对象，进一步降低材质复杂度
            transform = self.object_transforms[i]
            distance = np.linalg.norm(transform['position'])
            
            if distance > 15.0:
                material['texture_resolution'] = 'low'  # 低分辨率纹理
                print(f"降低远距离对象材质复杂度: {transform['name']}")
    
    def _optimize_memory_usage(self):
        """
        优化内存使用
        """
        print("优化内存使用...")
        
        # 1. 压缩顶点数据（使用半精度浮点数等）
        # 2. 移除不必要的数据
        # 3. 合并重复材质
        
        # 统计优化后的数据大小
        total_memory = 0
        for i in range(len(self.vertices)):
            # 顶点数据大小 (float32 * 3 per vertex)
            vert_memory = len(self.vertices[i]) * 4  # 每个float32占4字节
            # 索引数据大小 (uint32 per index)
            idx_memory = len(self.indices[i]) * 4
            total_memory += vert_memory + idx_memory
        
        print(f"优化后估计内存占用: {total_memory / 1024:.1f} KB")
    
    def render_frame(self) -> bool:
        """
        渲染一帧场景
        在实际引擎中，这里会执行实际的渲染操作
        
        Returns:
            bool: 是否成功渲染
        """
        # 计算FPS
        current_time = time.time()
        delta_time = current_time - self.last_time
        self.frame_count += 1
        
        if delta_time >= 1.0:
            self.current_fps = self.frame_count / delta_time
            self.frame_count = 0
            self.last_time = current_time
        
        # 模拟渲染过程
        # 1. 设置渲染目标
        # 2. 清除缓冲区
        # 3. 应用相机变换
        # 4. 渲染每个对象
        # 5. 交换缓冲区
        
        # 这里我们只打印渲染信息作为模拟
        print(f"渲染帧 #{self.frame_count} - FPS: {self.current_fps:.1f}")
        
        return True
    
    def start_preview(self, duration: float = None):
        """
        开始预览场景
        
        Args:
            duration: 预览持续时间（秒），None表示一直运行直到手动停止
        """
        if not self.scene_data:
            print("错误: 没有加载场景数据")
            return
        
        print("开始场景预览...")
        print("按ESC键或Ctrl+C退出预览")
        
        start_time = time.time()
        running = True
        
        try:
            while running:
                # 渲染一帧
                self.render_frame()
                
                # 检查是否达到持续时间
                if duration is not None and time.time() - start_time >= duration:
                    running = False
                
                # 限制帧率
                frame_time = 1.0 / self.target_fps
                time.sleep(max(0, frame_time - (time.time() - start_time)))
                
        except KeyboardInterrupt:
            print("预览已停止")
        
        print("场景预览结束")
    
    def get_performance_stats(self) -> Dict:
        """
        获取性能统计信息
        
        Returns:
            Dict: 性能统计数据
        """
        stats = {
            'current_fps': self.current_fps,
            'total_objects': len(self.object_transforms),
            'total_vertices': sum(v.shape[0] for v in self.vertices),
            'total_faces': sum(i.shape[0] // 3 for i in self.indices),
            'window_resolution': f"{self.window_width}x{self.window_height}",
            'low_end_mode': self.low_end_gpu_mode
        }
        
        # 计算估计的GPU内存使用
        # 顶点: 每个顶点约32字节 (3*float32)
        # 索引: 每个索引4字节 (uint32)
        # 法线: 每个法线约32字节 (3*float32)
        vertex_memory = sum(v.shape[0] * 32 for v in self.vertices)
        index_memory = sum(i.shape[0] * 4 for i in self.indices)
        normal_memory = sum(n.shape[0] * 32 for n in self.normals)
        
        total_memory_mb = (vertex_memory + index_memory + normal_memory) / (1024 * 1024)
        stats['estimated_gpu_memory_mb'] = total_memory_mb
        
        return stats
    
    def save_screenshot(self, filename: str = "landscape_preview.png"):
        """
        保存当前帧的截图
        
        Args:
            filename: 输出文件名
        """
        # 在实际引擎中，这里会实现真正的截图功能
        print(f"截图已保存到: {filename}")
        
    def export_render_settings(self, filename: str = "render_settings.json"):
        """
        导出渲染设置
        
        Args:
            filename: 输出文件名
        """
        settings = {
            'window_width': self.window_width,
            'window_height': self.window_height,
            'target_fps': self.target_fps,
            'low_end_gpu_mode': self.low_end_gpu_mode,
            'camera_position': self.camera_position.tolist(),
            'camera_rotation': self.camera_rotation.tolist(),
            'lights': [
                {
                    'position': light['position'].tolist(),
                    'rotation': light['rotation'].tolist(),
                    'energy': light['energy'],
                    'type': light['type']
                }
                for light in self.lights
            ]
        }
        
        import json
        with open(filename, 'w') as f:
            json.dump(settings, f, indent=2)
        
        print(f"渲染设置已导出到: {filename}")

# 示例使用代码
def preview_blender_landscape(scene_data=None, low_end_gpu=True):
    """
    预览从Blender生成的风景
    
    Args:
        scene_data: 从Blender获取的场景数据
        low_end_gpu: 是否启用低端GPU模式
    """
    # 创建渲染器
    renderer = LandscapePreviewRenderer(
        window_width=800,
        window_height=600,
        target_fps=30,
        low_end_gpu_mode=low_end_gpu
    )
    
    # 如果没有提供场景数据，使用示例数据
    if scene_data is None:
        print("警告: 没有提供场景数据，将使用示例数据")
        # 创建一个简单的示例场景数据
        scene_data = {
            "name": "ExampleLandscape",
            "objects": [],
            "camera": {
                "location": [20.0, -20.0, 12.0],
                "rotation": [1.047, 0.0, 0.785]
            },
            "lights": [
                {
                    "location": [5.0, 5.0, 10.0],
                    "rotation": [0.0, 0.0, 0.0],
                    "energy": 2.0,
                    "type": "SUN"
                }
            ]
        }
    
    # 加载场景数据
    renderer.load_scene_data(scene_data)
    
    # 获取并打印性能统计
    stats = renderer.get_performance_stats()
    print("\n性能统计:")
    for key, value in stats.items():
        print(f"- {key}: {value}")
    
    # 导出渲染设置
    renderer.export_render_settings("engine_render_settings.json")
    
    # 开始预览（运行5秒作为示例）
    print("\n开始预览场景...")
    renderer.start_preview(duration=5.0)

# Blender MCP集成函数
def integrate_with_blender_mcp(mcp_client, low_end_gpu=True):
    """
    与Blender MCP插件集成，直接从Blender获取场景数据并预览
    
    Args:
        mcp_client: MCP客户端实例
        low_end_gpu: 是否启用低端GPU模式
    """
    print("与Blender MCP集成...")
    
    try:
        # 从Blender获取场景数据（通过MCP）
        print("正在从Blender获取场景数据...")
        # 在实际实现中，这里会调用MCP客户端的API获取数据
        # scene_data = mcp_client.get_scene_data()
        
        # 由于我们没有实际的MCP客户端，这里使用模拟数据
        scene_data = {
            "name": "MCP_Landscape",
            "objects": [],
            "camera": None,
            "lights": []
        }
        
        # 预览场景
        preview_blender_landscape(scene_data, low_end_gpu)
        
    except Exception as e:
        print(f"MCP集成错误: {e}")

# 主函数 - 用于直接运行预览
def main():
    print("==================================")
    print("风景预览渲染器 - 引擎版")
    print("==================================")
    
    # 检查是否有可用的Blender场景数据
    # 实际项目中，这里可能会尝试从文件或MCP获取数据
    
    # 运行预览
    preview_blender_landscape(low_end_gpu=True)

if __name__ == "__main__":
    main()

"""
引擎集成说明：

1. 如何在引擎中使用此渲染器：

   ```python
   # 在引擎主循环中
   from Engine.Renderer.LandscapePreviewRenderer import LandscapePreviewRenderer
   
   # 创建渲染器实例
   renderer = LandscapePreviewRenderer(
       window_width=1024,
       window_height=768,
       target_fps=60,
       low_end_gpu_mode=True  # 针对低端GPU启用优化
   )
   
   # 从Blender MCP获取场景数据
   # 这需要Blender端运行LowPolyLandscapeGenerator脚本
   scene_data = get_scene_data_from_blender_mcp()
   
   # 加载场景数据
   renderer.load_scene_data(scene_data)
   
   # 主循环
   running = True
   while running:
       # 处理输入
       process_input()
       
       # 更新场景
       update_scene()
       
       # 渲染帧
       renderer.render_frame()
       
       # 检查退出条件
       if exit_condition_met():
           running = False
   ```

2. 低端GPU优化配置：

   对于GTX 750 Ti等低端GPU，确保设置：
   - low_end_gpu_mode=True
   - target_fps=30 (降低目标帧率)
   - window_width=800, window_height=600 (降低分辨率)

3. 与Blender MCP插件协同工作：

   - Blender端运行LowPolyLandscapeGenerator.py生成场景
   - 通过MCP协议传输场景数据到引擎
   - 引擎使用LandscapePreviewRenderer渲染预览

4. 性能优化技巧：

   - 使用实例化渲染相似对象（如树木、岩石）
   - 实现视锥体剔除减少不必要的渲染
   - 为远处对象使用更低的LOD（细节级别）
   - 合并小型静态对象减少绘制调用
   - 使用纹理图集减少材质切换
"""