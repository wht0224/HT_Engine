import sys
import os
import json

# 添加引擎根目录到Python路径，确保能正确导入内部模块
# Add engine root directory to Python path for proper internal module imports
engine_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, engine_root)

# 导入日志系统
# Import logging system
from Logger import get_logger, LogLevel

class Engine:
    """
    高性能渲染引擎主类
    High-Performance Rendering Engine Main Class
    
    专为低端GPU（如NVIDIA GTX 750Ti和AMD RX 580）设计优化
    Designed and optimized for low-end GPUs like NVIDIA GTX 750Ti and AMD RX 580
    """
    
    def __init__(self):
        """
        引擎实例初始化
        Initialize engine instance
        """
        # 初始化日志器
        # Initialize logger
        self.logger = get_logger("Engine")
        
        # 引擎状态标志
        # Engine state flags
        self.is_initialized = False
        
        # 核心组件引用
        # Core component references
        self.renderer = None
        self.resource_manager = None
        self.scene_manager = None
        self.platform = None
        
        # MCP架构管理器（用于多组件并行处理）
        # MCP architecture manager (for multi-component parallel processing)
        self.mcp_manager = None
        
    def initialize(self, config=None, skip_renderer=False):
        """
        初始化引擎的所有核心组件
        Initialize all core components of the engine
        
        参数/Parameters:
            config: 引擎配置参数/Engine configuration parameters
            skip_renderer: 是否跳过渲染器初始化/Whether to skip renderer initialization
            
        返回值/Returns:
            bool: 初始化成功返回True，否则返回False/True if initialization succeeds, False otherwise
        """
        self.logger.info("正在初始化高性能渲染引擎...")
        
        # 初始化平台模块 - 第一步
        # Initialize platform module - Step 1
        try:
            self.logger.debug("1. 导入平台模块...")
            from Platform.Platform import Platform
            self.platform = Platform()
            self.platform.initialize()
            self.logger.info("平台模块初始化完成")
        except Exception as e:
            self.logger.error(f"平台模块初始化失败: {e}")
            return False
        
        # 初始化资源管理器 - 第二步
        # Initialize resource manager - Step 2
        try:
            self.logger.debug("2. 导入并初始化资源管理器...")
            from Renderer.Resources.ResourceManager import ResourceManager
            self.resource_manager = ResourceManager()
            self.resource_manager.initialize()
            self.logger.info("资源管理器初始化完成")
        except Exception as e:
            self.logger.error(f"资源管理器初始化失败: {e}")
            return False
        
        # 初始化渲染器 - 第三步（可跳过）
        # Initialize renderer - Step 3 (optional)
        if not skip_renderer:
            try:
                self.logger.debug("3. 导入并初始化渲染器...")
                from Renderer.Renderer import Renderer
                # 创建渲染器实例，传入平台和资源管理器
                # Create renderer instance with platform and resource manager
                self.renderer = Renderer(self.platform, self.resource_manager)
                # 初始化渲染器，传入配置参数
                # Initialize renderer with configuration parameters
                self.renderer.initialize(config)
                self.logger.info("渲染器初始化完成")
            except ImportError as e:
                # 处理渲染器导入失败的情况
                # Handle renderer import failure
                self.logger.error(f"渲染器导入失败: {e}")
                self.logger.error(f"Python路径: {sys.path}")
                self.logger.error("渲染器导入失败，跳过渲染器初始化")
                skip_renderer = True
                # 渲染器导入失败，继续初始化其他组件
            except Exception as e:
                # 处理渲染器初始化失败的情况
                # Handle renderer initialization failure
                self.logger.error(f"渲染器初始化失败: {e}")
                self.logger.error(f"错误类型: {type(e).__name__}")
                import traceback
                self.logger.error(f"错误堆栈: {traceback.format_exc()}")
                self.logger.error("渲染器初始化失败，跳过渲染器初始化")
                skip_renderer = True
                # 渲染器初始化失败，继续初始化其他组件
        
        # 初始化场景管理器 - 第四步
        # Initialize scene manager - Step 4
        try:
            self.logger.debug("4. 导入并初始化场景管理器...")
            from Scene.SceneManager import SceneManager
            # 创建场景管理器实例，传入引擎引用
            # Create scene manager instance with engine reference
            self.scene_manager = SceneManager(self)
            self.logger.info("场景管理器初始化完成")
        except ImportError as e:
            # 处理场景管理器导入失败的情况
            # Handle scene manager import failure
            self.logger.error(f"场景管理器导入失败: {e}")
            self.logger.error(f"Python路径: {sys.path}")
            return False
        except Exception as e:
            # 处理场景管理器初始化失败的情况
            # Handle scene manager initialization failure
            self.logger.error(f"场景管理器初始化失败: {e}")
            return False
        
        # 初始化物理系统 - 第五步（可选，失败不影响引擎运行）
        # Initialize physics system - Step 5 (optional, failure doesn't affect engine operation)
        try:
            self.logger.debug("5. 导入并初始化物理系统...")
            from Physics.PhysicsSystem import PhysicsSystem
            # 创建物理系统实例，传入引擎引用
            # Create physics system instance with engine reference
            self.physics_system = PhysicsSystem(self)
            self.logger.info("物理系统初始化完成")
        except Exception as e:
            # 物理系统不是核心组件，初始化失败不影响引擎继续运行
            # Physics system is not a core component, failure doesn't affect engine operation
            self.logger.error(f"物理系统初始化失败: {e}")
        
        try:
            # 创建默认测试场景
            self.logger.debug("6. 创建默认测试场景...")
            self._create_default_scene()
            self.logger.info("默认测试场景创建完成")
        except Exception as e:
            self.logger.error(f"创建默认测试场景失败: {e}")
            return False
        
        try:
            # 导入并初始化相机控制器
            self.logger.debug("6. 导入并初始化相机控制器...")
            from UI.CameraController import CameraController
            self.camera_controller = CameraController(self.scene_manager.active_camera)
            self.platform.camera_controller = self.camera_controller
            self.logger.info("相机控制器初始化完成")
        except Exception as e:
            self.logger.error(f"相机控制器初始化失败: {e}")
            return False
        
        try:
            # 导入并初始化变换操纵器
            self.logger.debug("7. 导入并初始化变换操纵器...")
            from UI.Controls.TransformManipulator import TransformManipulator
            self.transform_manipulator = TransformManipulator(self.scene_manager.active_camera)
            self.logger.info("变换操纵器初始化完成")
        except Exception as e:
            self.logger.error(f"变换操纵器初始化失败: {e}")
            return False
        
        try:
            # 导入并初始化UI管理器
            self.logger.debug("8. 导入并初始化UI管理器...")
            from UI.UIManager import UIManager
            self.ui_manager = UIManager(self.platform)
            self.logger.info("UI管理器初始化完成")
        except Exception as e:
            self.logger.error(f"UI管理器初始化失败: {e}")
            return False
        
        try:
            # 导入并初始化主窗口
            self.logger.debug("9. 导入并初始化主窗口...")
            from UI.MainWindow import MainWindow
            self.main_window = MainWindow(self.platform)
            self.ui_manager.add_control(self.main_window)
            self.logger.info("主窗口初始化完成")
        except Exception as e:
            self.logger.error(f"主窗口初始化失败: {e}")
            return False
        
        try:
            # 导入并初始化HTML UI服务器
            self.logger.debug("10. 导入并初始化HTML UI服务器...")
            from UI.HTMLUI.html_ui_server import HTMLUIServer
            self.html_ui_server = HTMLUIServer(self)
            self.html_ui_server.start()
            self.logger.info("HTML UI服务器已启动")
        except Exception as e:
            self.logger.error(f"HTML UI服务器启动失败: {e}")
            # HTML UI不是核心组件，失败不影响引擎运行
        
        try:
            # 导入并初始化Python UI
            self.logger.debug("11. 导入并初始化Python UI...")
            from UI.PythonUI.python_ui_tk import PythonUI
            self.python_ui = PythonUI(self)
            self.python_ui.start()
            self.logger.info("Python UI已启动")
        except Exception as e:
            self.logger.error(f"Python UI启动失败: {e}")
            # Python UI不是核心组件，失败不影响引擎运行
        
        try:
            # 初始化MCP架构管理器
            self.logger.debug("11. 初始化MCP架构管理器...")
            from Core.MCP.MCPManager import MCPManager
            self.mcp_manager = MCPManager(self)
            self.logger.info("MCP架构初始化完成")
        except Exception as e:
            self.logger.error(f"MCP架构初始化失败: {e}")
            return False
        
        try:
            # 导入并初始化场景编辑器
            self.logger.debug("12. 导入并初始化场景编辑器...")
            from Editor.SceneEditor import SceneEditor
            self.scene_editor = SceneEditor(self)
            self.logger.info("场景编辑器初始化完成")
        except Exception as e:
            self.logger.error(f"场景编辑器初始化失败: {e}")
            # 场景编辑器不是核心组件，失败不影响引擎运行
        
        self.is_initialized = True
        self.logger.info("引擎初始化完成")
        return True
    
    def _create_default_ui(self):
        """
        创建默认UI
        """
        self.logger.debug("创建默认UI...")
        
        from UI.Controls.Button import Button
        from UI.Controls.Label import Label
        from UI.Controls.Slider import Slider
        from UI.Controls.Menu import Menu
        from UI.Controls.PropertyPanel import PropertyPanel
        from UI.Controls.Toolbar import Toolbar
        
        # 创建工具栏
        self.toolbar = Toolbar(0, 0, self.platform.width, 40)
        self.ui_manager.add_control(self.toolbar)
        
        # 创建一个简单的菜单
        menu_items = [
            {"text": "创建立方体", "callback": self._create_cube},
            {"text": "创建球体", "callback": self._create_sphere},
            {"text": "创建圆柱体", "callback": self._create_cylinder},
            {"text": "导入模型", "callback": self._import_model},
            {"text": "退出", "callback": self.shutdown}
        ]
        menu = Menu(10, 50, 150, 30, menu_items)
        self.ui_manager.add_control(menu)
        
        # 创建一个标签
        label = Label(200, 50, 200, 30, "Low End GPU Rendering Engine")
        self.ui_manager.add_control(label)
        
        # 创建一个滑块
        slider = Slider(450, 50, 200, 30, 0.0, 1.0, 0.5)
        def on_slider_change(event):
            self.logger.debug(f"滑块值改变: {event.get_param('new_value')}")
        slider.add_event_handler("value_changed", on_slider_change)
        self.ui_manager.add_control(slider)
        
        # 创建一个按钮
        button = Button(700, 50, 100, 30, "测试按钮")
        def on_button_click(event):
            self.logger.debug("测试按钮被点击")
        button.add_event_handler("mouse_click", on_button_click)
        self.ui_manager.add_control(button)
        
        # 创建属性面板
        self.property_panel = PropertyPanel(self.platform.width - 300, 50, 290, self.platform.height - 60, "Properties")
        self.ui_manager.add_control(self.property_panel)
        
        self.logger.debug("默认UI创建完成")
    
    def _create_cube(self):
        """创建立方体"""
        self.logger.info("创建立方体")
        from Scene.SceneNode import SceneNode
        from Engine.Math import Vector3
        from Renderer.Resources.Mesh import Mesh
        from Renderer.Resources.Material import Material
        
        # 创建立方体网格
        cube_mesh = Mesh.create_cube(1.0)
        
        # 创建材质
        cube_material = Material()
        cube_material.set_color(Vector3(0.8, 0.2, 0.2))
        
        # 创建场景节点
        cube_node = SceneNode("Cube")
        cube_node.set_position(Vector3(0, 0, 0))
        cube_node.mesh = cube_mesh
        cube_node.material = cube_material
        
        # 添加到场景
        self.scene_manager.root_node.add_child(cube_node)
        self.logger.info(f"立方体 '{cube_node.name}' 已创建")
    
    def _create_sphere(self):
        """创建球体"""
        self.logger.info("创建球体")
        from Scene.SceneNode import SceneNode
        from Engine.Math import Vector3
        from Renderer.Resources.Mesh import Mesh
        from Renderer.Resources.Material import Material
        
        # 创建球体网格
        sphere_mesh = Mesh.create_sphere(0.5, 32, 32)
        
        # 创建材质
        sphere_material = Material()
        sphere_material.set_color(Vector3(0.2, 0.8, 0.2))
        
        # 创建场景节点
        sphere_node = SceneNode("Sphere")
        sphere_node.set_position(Vector3(2, 0, 0))
        sphere_node.mesh = sphere_mesh
        sphere_node.material = sphere_material
        
        # 添加到场景
        self.scene_manager.root_node.add_child(sphere_node)
        self.logger.info(f"球体 '{sphere_node.name}' 已创建")
    
    def _create_cylinder(self):
        """创建圆柱体"""
        self.logger.info("创建圆柱体")
        from Scene.SceneNode import SceneNode
        from Engine.Math import Vector3
        from Renderer.Resources.Mesh import Mesh
        from Renderer.Resources.Material import Material
        
        # 创建圆柱体网格
        cylinder_mesh = Mesh.create_cylinder(0.5, 1.0, 32)
        
        # 创建材质
        cylinder_material = Material()
        cylinder_material.set_color(Vector3(0.2, 0.2, 0.8))
        
        # 创建场景节点
        cylinder_node = SceneNode("Cylinder")
        cylinder_node.set_position(Vector3(-2, 0, 0))
        cylinder_node.mesh = cylinder_mesh
        cylinder_node.material = cylinder_material
        
        # 添加到场景
        self.scene_manager.root_node.add_child(cylinder_node)
        self.logger.info(f"圆柱体 '{cylinder_node.name}' 已创建")
    
    def _create_plane(self):
        """创建平面"""
        self.logger.info("创建平面")
        from Scene.SceneNode import SceneNode
        from Engine.Math import Vector3
        from Renderer.Resources.Mesh import Mesh
        from Renderer.Resources.Material import Material
        
        # 创建平面网格
        plane_mesh = Mesh.create_plane(2.0, 2.0, 20, 20)
        
        # 创建材质
        plane_material = Material()
        plane_material.set_color(Vector3(0.8, 0.8, 0.8))
        
        # 创建场景节点
        plane_node = SceneNode("Plane")
        plane_node.set_position(Vector3(0, 0, 2))
        plane_node.set_rotation(Vector3(-90, 0, 0))
        plane_node.mesh = plane_mesh
        plane_node.material = plane_material
        
        # 添加到场景
        self.scene_manager.root_node.add_child(plane_node)
        self.logger.info(f"平面 '{plane_node.name}' 已创建")
    
    def _create_cone(self):
        """创建圆锥体"""
        self.logger.info("创建圆锥体")
        from Scene.SceneNode import SceneNode
        from Engine.Math import Vector3
        from Renderer.Resources.Mesh import Mesh
        from Renderer.Resources.Material import Material
        
        # 创建圆锥体网格
        cone_mesh = Mesh.create_cone(0.5, 1.0, 32)
        
        # 创建材质
        cone_material = Material()
        cone_material.set_color(Vector3(0.8, 0.2, 0.8))
        
        # 创建场景节点
        cone_node = SceneNode("Cone")
        cone_node.set_position(Vector3(2, 0, 2))
        cone_node.mesh = cone_mesh
        cone_node.material = cone_material
        
        # 添加到场景
        self.scene_manager.root_node.add_child(cone_node)
        self.logger.info(f"圆锥体 '{cone_node.name}' 已创建")
    
    def _import_model(self):
        """导入模型"""
        self.logger.info("导入模型")
        from Assets.ModelImporter import ModelImporter
        
        # 创建模型导入器
        importer = ModelImporter(self.resource_manager)
        
        # 这里可以替换为实际的模型文件路径，或者添加文件选择对话框
        # 目前使用一个示例模型文件路径
        model_file_path = "example_model.obj"
        
        try:
            # 导入模型
            imported_nodes = importer.import_model(model_file_path)
            
            # 将导入的节点添加到场景
            for node in imported_nodes:
                self.scene_manager.root_node.add_child(node)
                self.logger.info(f"模型节点 '{node.name}' 已添加到场景")
            
            if imported_nodes:
                self.logger.info(f"成功导入 {len(imported_nodes)} 个模型节点")
            else:
                self.logger.warning("未导入任何模型节点")
        except Exception as e:
            self.logger.error(f"导入模型失败: {e}")
            
            # 如果导入失败，创建一个简单的占位符节点
            from Scene.SceneNode import SceneNode
            from Engine.Math import Vector3
            from Renderer.Resources.Mesh import Mesh
            from Renderer.Resources.Material import Material
            
            # 创建一个简单的模型节点作为占位符
            model_node = SceneNode("ImportedModel")
            model_node.set_position(Vector3(0, 0, -2))
            
            # 创建一个临时网格（实际导入时会替换为真实模型）
            temp_mesh = Mesh.create_cube(1.0)
            temp_material = Material()
            temp_material.set_color(Vector3(0.8, 0.8, 0.2))
            
            model_node.mesh = temp_mesh
            model_node.material = temp_material
            
            # 添加到场景
            self.scene_manager.root_node.add_child(model_node)
            self.logger.info(f"创建模型占位符 '{model_node.name}'")
    
    def _create_default_scene(self):
        """
        创建默认测试场景，包含相机、灯光和各种几何体
        Create default test scene with camera, lights, and various geometries
        """
        self.logger.debug("创建默认测试场景...")
        
        # 导入场景相关类
        # Import scene-related classes
        from Scene.Camera import Camera
        from Scene.Light import DirectionalLight, PointLight, SpotLight
        from Engine.Math import Vector3
        from Scene.SceneNode import SceneNode
        from Renderer.Resources.Mesh import Mesh
        from Renderer.Resources.Material import Material
        
        # 创建相机并设置为活动相机
        # Create camera and set as active camera
        camera = Camera()
        # 设置相机位置
        # Set camera position
        camera.set_position(Vector3(5, 3, 5))
        # 设置相机朝向
        # Set camera look direction
        camera.look_at(Vector3(0, 1, 0))
        # 设置活动相机
        # Set active camera
        self.scene_manager.active_camera = camera
        # 添加到相机列表
        # Add to camera list
        self.scene_manager.cameras.append(camera)
        
        # 创建方向光并添加到灯光管理器
        # Create directional light and add to light manager
        dir_light = DirectionalLight()
        # 设置方向光方向
        # Set directional light direction
        dir_light.set_direction(Vector3(1, -1, -1))
        # 设置方向光强度
        # Set directional light intensity
        dir_light.set_intensity(1.0)
        # 添加到灯光管理器
        # Add to light manager
        self.scene_manager.light_manager.add_light(dir_light)
        
        # 创建点光源
        # Create point light
        point_light = PointLight()
        # 设置点光源位置
        # Set point light position
        point_light.set_position(Vector3(2, 3, 2))
        # 设置点光源强度
        # Set point light intensity
        point_light.set_intensity(2.0)
        # 设置点光源颜色（暖白色）
        # Set point light color (warm white)
        point_light.set_color(Vector3(1.0, 0.8, 0.6))
        # 设置点光源影响半径
        # Set point light influence radius
        point_light.set_radius(10.0)
        # 添加到灯光管理器
        # Add to light manager
        self.scene_manager.light_manager.add_light(point_light)
        
        # 创建聚光灯
        # Create spot light
        spot_light = SpotLight()
        # 设置聚光灯位置
        # Set spot light position
        spot_light.set_position(Vector3(-3, 4, 3))
        # 设置聚光灯方向
        # Set spot light direction
        spot_light.set_direction(Vector3(1, -1, -1))
        # 设置聚光灯强度
        # Set spot light intensity
        spot_light.set_intensity(3.0)
        # 设置聚光灯颜色（冷蓝色）
        # Set spot light color (cool blue)
        spot_light.set_color(Vector3(0.6, 0.8, 1.0))
        # 设置聚光灯影响半径
        # Set spot light influence radius
        spot_light.set_radius(15.0)
        # 设置聚光灯内圆锥角度
        # Set spot light inner cone angle
        spot_light.set_spot_inner_angle(15.0)
        # 设置聚光灯外圆锥角度
        # Set spot light outer cone angle
        spot_light.set_spot_angle(30.0)
        # 添加到灯光管理器
        # Add to light manager
        self.scene_manager.light_manager.add_light(spot_light)
        
        # 创建地面
        ground_mesh = Mesh.create_plane(20.0, 20.0, 20, 20)
        ground_material = Material()
        ground_material.set_color(Vector3(0.3, 0.5, 0.3))
        ground_material.set_roughness(0.8)
        ground_material.set_metallic(0.2)
        
        ground_node = SceneNode("Ground")
        ground_node.set_position(Vector3(0, 0, 0))
        ground_node.set_rotation(Vector3(-90, 0, 0))
        ground_node.mesh = ground_mesh
        ground_node.material = ground_material
        self.scene_manager.root_node.add_child(ground_node)
        
        # 创建多个不同类型的物体，展示不同材质
        
        # 红色立方体
        cube_mesh = Mesh.create_cube(1.0)
        cube_material = Material()
        cube_material.set_color(Vector3(0.8, 0.2, 0.2))
        cube_material.set_roughness(0.5)
        cube_material.set_metallic(0.0)
        
        cube_node = SceneNode("RedCube")
        cube_node.set_position(Vector3(0, 0.5, 0))
        cube_node.mesh = cube_mesh
        cube_node.material = cube_material
        self.scene_manager.root_node.add_child(cube_node)
        
        # 绿色球体
        sphere_mesh = Mesh.create_sphere(0.5, 32, 32)
        sphere_material = Material()
        sphere_material.set_color(Vector3(0.2, 0.8, 0.2))
        sphere_material.set_roughness(0.3)
        sphere_material.set_metallic(0.0)
        
        sphere_node = SceneNode("GreenSphere")
        sphere_node.set_position(Vector3(2, 0.5, 0))
        sphere_node.mesh = sphere_mesh
        sphere_node.material = sphere_material
        self.scene_manager.root_node.add_child(sphere_node)
        
        # 蓝色圆柱体
        cylinder_mesh = Mesh.create_cylinder(0.5, 1.5, 32)
        cylinder_material = Material()
        cylinder_material.set_color(Vector3(0.2, 0.2, 0.8))
        cylinder_material.set_roughness(0.7)
        cylinder_material.set_metallic(0.0)
        
        cylinder_node = SceneNode("BlueCylinder")
        cylinder_node.set_position(Vector3(-2, 0.75, 0))
        cylinder_node.mesh = cylinder_mesh
        cylinder_node.material = cylinder_material
        self.scene_manager.root_node.add_child(cylinder_node)
        
        # 黄色金属球体
        metal_sphere_mesh = Mesh.create_sphere(0.5, 32, 32)
        metal_sphere_material = Material()
        metal_sphere_material.set_color(Vector3(0.9, 0.8, 0.2))
        metal_sphere_material.set_roughness(0.2)
        metal_sphere_material.set_metallic(1.0)
        
        metal_sphere_node = SceneNode("YellowMetalSphere")
        metal_sphere_node.set_position(Vector3(0, 0.5, 2))
        metal_sphere_node.mesh = metal_sphere_mesh
        metal_sphere_node.material = metal_sphere_material
        self.scene_manager.root_node.add_child(metal_sphere_node)
        
        # 紫色玻璃立方体
        glass_cube_mesh = Mesh.create_cube(1.0)
        glass_cube_material = Material()
        glass_cube_material.set_color(Vector3(0.8, 0.2, 0.8))
        glass_cube_material.set_roughness(0.0)
        glass_cube_material.set_metallic(0.0)
        glass_cube_material.set_transparency(0.3)
        glass_cube_material.set_blend_mode(Material.BLEND_MODE_TRANSPARENT)
        
        glass_cube_node = SceneNode("PurpleGlassCube")
        glass_cube_node.set_position(Vector3(0, 0.5, -2))
        glass_cube_node.mesh = glass_cube_mesh
        glass_cube_node.material = glass_cube_material
        self.scene_manager.root_node.add_child(glass_cube_node)
        
        # 创建一排小立方体作为装饰
        for i in range(5):
            small_cube_mesh = Mesh.create_cube(0.3)
            small_cube_material = Material()
            small_cube_material.set_color(Vector3(0.1 + i*0.15, 0.3, 0.8 - i*0.15))
            small_cube_material.set_roughness(0.6)
            small_cube_material.set_metallic(0.0)
            
            small_cube_node = SceneNode(f"SmallCube_{i}")
            small_cube_node.set_position(Vector3(-3 + i*1.5, 0.15, -3))
            small_cube_node.mesh = small_cube_mesh
            small_cube_node.material = small_cube_material
            self.scene_manager.root_node.add_child(small_cube_node)
        
        self.logger.debug("默认测试场景创建完成")
        
    def shutdown(self):
        """
        关闭引擎，释放资源
        """
        if not self.is_initialized:
            return
            
        self.logger.info("正在关闭引擎...")
        
        # 停止HTML UI服务器
        if hasattr(self, 'html_ui_server') and self.html_ui_server:
            self.html_ui_server.stop()
            self.html_ui_server = None
        
        # 停止Python UI
        if hasattr(self, 'python_ui') and self.python_ui:
            self.python_ui.shutdown()
            self.python_ui = None
        
        # 关闭物理系统
        if hasattr(self, 'physics_system') and self.physics_system:
            self.physics_system.shutdown()
            self.physics_system = None
        
        if self.scene_manager:
            self.scene_manager.shutdown()
            self.scene_manager = None
            
        if self.renderer:
            self.renderer.shutdown()
            self.renderer = None
            
        if self.resource_manager:
            self.resource_manager.shutdown()
            self.resource_manager = None
            
        if self.platform:
            self.platform.shutdown()
            self.platform = None
            
        self.is_initialized = False
        self.logger.info("引擎已关闭")
        
    def update(self, delta_time):
        """
        更新引擎状态
        
        Args:
            delta_time: 帧间隔时间
        """
        if not self.is_initialized:
            return
            
        # 使用MCP管理器更新
        if self.mcp_manager:
            self.mcp_manager.update(delta_time)
        else:
            # 兼容旧的更新方式
            # 更新物理系统
            if hasattr(self, 'physics_system') and self.physics_system:
                self.physics_system.update(delta_time)
            
            # 更新场景
            if self.scene_manager:
                self.scene_manager.update(delta_time)
            
            # 更新UI
            if hasattr(self, 'ui_manager') and self.ui_manager:
                self.ui_manager.update(delta_time)
        
        # 更新场景编辑器
        if hasattr(self, 'scene_editor') and self.scene_editor:
            self.scene_editor.update(delta_time)
    
    def render(self):
        """
        渲染当前帧
        """
        if not self.is_initialized:
            return
            
        # 使用MCP管理器渲染
        if self.mcp_manager:
            stats = self.mcp_manager.render()
        else:
            # 兼容旧的渲染方式
            # 渲染场景
            if self.renderer and self.scene_manager:
                # 简化的渲染调用，直接传递场景管理器
                self.renderer.render(self.scene_manager)
            
            # 渲染场景编辑器
            if hasattr(self, 'scene_editor') and self.scene_editor:
                self.scene_editor.render()
            
            # 渲染UI
            if hasattr(self, 'ui_manager') and self.ui_manager:
                # 在渲染UI之前重置渲染状态，确保UI能够正确渲染
                from OpenGL.GL import (
                    glEnable, glDisable, glBlendFunc, GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
                    glDepthFunc, GL_DEPTH_TEST, GL_ALWAYS, glMatrixMode, GL_PROJECTION, GL_MODELVIEW,
                    glLoadIdentity, glOrtho, glViewport
                )
                
                # 启用混合模式，以便正确显示半透明UI组件
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                
                # 禁用深度测试，或者将深度函数设置为GL_ALWAYS
                glDisable(GL_DEPTH_TEST)
                
                # 保存当前的矩阵模式
                glMatrixMode(GL_PROJECTION)
                glLoadIdentity()
                # 设置正交投影，用于渲染2D UI
                glOrtho(0, self.platform.width, 0, self.platform.height, -1, 1)
                
                # 设置模型视图矩阵
                glMatrixMode(GL_MODELVIEW)
                glLoadIdentity()
                
                # 设置正确的视口
                glViewport(0, 0, self.platform.width, self.platform.height)
                
                # 渲染UI
                self.ui_manager.render()
                
                # 恢复深度测试，以便下一帧渲染3D场景
                glEnable(GL_DEPTH_TEST)
        
        # 将渲染结果发送到HTML UI
        if hasattr(self, 'html_ui_server') and self.html_ui_server:
            try:
                # 获取渲染结果
                if self.renderer:
                    render_result = self.renderer.get_render_result()
                    if render_result:
                        # 通过WebSocket发送渲染结果
                        import asyncio
                        asyncio.run(self.html_ui_server.send_message(json.dumps({
                            'type': 'render_frame',
                            'data': render_result
                        })))
            except Exception as e:
                self.logger.error(f"发送渲染结果失败: {e}")



if __name__ == "__main__":
    # 导入日志系统
    from Logger import get_logger, LogLevel
    logger = get_logger("Main")
    
    # 引擎启动示例
    logger.info("开始启动引擎...")
    engine = Engine()
    
    try:
        # 初始化引擎
        logger.info("初始化引擎...")
        engine.initialize()
        
        # 检查引擎是否初始化成功
        if not engine.is_initialized:
            logger.error("引擎初始化失败！")
            exit(1)
        
        # 检查平台是否有图形支持
        if hasattr(engine.platform, 'has_graphics') and not engine.platform.has_graphics:
            logger.warning("警告：没有图形支持！")
        else:
            logger.info("图形支持已启用")
        
        # 检查窗口是否创建成功
        if hasattr(engine.platform, 'window_created') and not engine.platform.window_created:
            logger.warning("警告：窗口创建失败！")
        else:
            logger.info("窗口创建成功")
        
        # 检查HTML UI服务器是否启动成功
        if hasattr(engine, 'html_ui_server') and engine.html_ui_server:
            logger.info("HTML UI服务器已启动")
            logger.info(f"HTTP服务器运行在 http://{engine.html_ui_server.host}:{engine.html_ui_server.http_port}")
            logger.info(f"WebSocket服务器运行在 ws://{engine.html_ui_server.host}:{engine.html_ui_server.websocket_port}")
        else:
            logger.warning("警告：HTML UI服务器未启动！")
        
        # 真实主循环，持续运行直到窗口关闭
        logger.info("引擎启动成功！按 ESC 键或关闭窗口退出。")
        
        import time
        frame_count = 0
        last_time = time.time()
        target_fps = 60
        frame_time = 1.0 / target_fps
        
        while True:
            try:
                current_time = time.time()
                delta_time = current_time - last_time
                
                # 帧率控制，限制为target_fps
                if delta_time < frame_time:
                    time.sleep(frame_time - delta_time)
                    current_time = time.time()
                    delta_time = current_time - last_time
                
                last_time = current_time
                
                # 更新和渲染
                engine.update(delta_time)
                engine.render()
                
                # 交换缓冲区以显示渲染结果
                if hasattr(engine.platform, 'swap_buffers'):
                    engine.platform.swap_buffers()
                
                frame_count += 1
                
                # 每60帧打印一次性能统计
                if frame_count % 60 == 0:
                    # 获取渲染性能统计
                    if hasattr(engine, 'renderer') and engine.renderer:
                        stats = engine.renderer.get_performance_stats()
                        render_time = stats.get("render_time_ms", 0)
                        draw_calls = stats.get("draw_calls", 0)
                        triangles = stats.get("triangles", 0)
                        visible_objects = stats.get("visible_objects", 0)
                        culled_objects = stats.get("culled_objects", 0)
                        
                        logger.info(f"已渲染 {frame_count} 帧，渲染时间: {render_time:.2f}ms, 绘制调用: {draw_calls}, 三角形: {triangles}, 可见物体: {visible_objects}, 裁剪物体: {culled_objects}")
                    else:
                        logger.info(f"已渲染 {frame_count} 帧")
                
                # 检查窗口是否需要关闭
                if hasattr(engine.platform, 'is_window_open') and not engine.platform.is_window_open():
                    logger.info("窗口关闭请求，正在关闭引擎...")
                    break
                
                # 检查ESC键是否被按下
                if hasattr(engine.platform, 'is_key_pressed') and engine.platform.is_key_pressed(27):  # ESC键
                    logger.info("ESC键被按下，正在关闭引擎...")
                    break
            except KeyboardInterrupt:
                logger.info("\n接收到中断信号，正在关闭引擎...")
                break
            except Exception as e:
                logger.error(f"渲染循环错误: {e}", exc_info=True)
                # 短暂暂停，避免日志刷屏
                time.sleep(1)
    except Exception as e:
        logger.error(f"引擎运行错误：{e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("开始关闭引擎...")
        engine.shutdown()
        logger.info("引擎已关闭")