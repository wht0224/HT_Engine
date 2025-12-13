import sys
import os
import json

# 添加引擎根目录到Python路径，确保能正确导入内部模块
# Add engine root directory to Python path for proper internal module imports
engine_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, engine_root)

# 导入日志系统
# Import logging system
from .Logger import get_logger, LogLevel

class Engine:
    """
    游戏世界的魔法引擎！
    专为让老显卡也能施展光影魔法而设计
    就像给GTX 750Ti和RX 580装上了魔法加速器
"""
    
    def __init__(self):
        self.logger = get_logger("Engine")
        self.is_initialized = False
        
        # 引擎配置
        self.config = {
            "frontend": {
                "type": "html",
                "enable_html": True,
                "enable_tkinter": False
            },
            "logging": {
                "level": "WARNING",
                "enable_file_logging": False,
                "log_file": "engine.log"
            },
            "renderer": {
                "enable_render_result": True,
                "render_result_interval": 33
            }
        }
        
        # 引擎的魔法核心组件
        self.renderer = None  # 光影魔法师
        self.res_mgr = None   # 资源宝库守护者
        self.scene_mgr = None # 场景导演
        self.plt = None       # 平台魔法阵
        
        # 多核并行处理指挥官
        self.mcp_mgr = None

        # MCP服务器（Model Context Protocol）- AI建模API
        self.mcp_server = None
    
    def _load_config(self, config_override=None):
        """
        加载引擎配置文件
        
        参数:
            config_override: 配置覆盖，优先级高于配置文件
        """
        import json
        import os
        
        # 默认配置文件路径
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engine_config.json")
        
        # 尝试读取配置文件
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                # 更新配置
                self.config.update(file_config)
                self.logger.info(f"成功加载配置文件: {config_path}")
        except Exception as e:
            self.logger.warning(f"加载配置文件失败: {e}")
        
        # 应用配置覆盖
        if config_override:
            self.config.update(config_override)
            self.logger.info("应用配置覆盖")
        
        # 应用日志配置
        self._apply_logging_config()
    
    def _apply_logging_config(self):
        """
        应用日志配置
        """
        from Engine.Logger import set_global_log_level, LogLevel, get_logger
        
        log_level_str = self.config["logging"]["level"].upper()
        try:
            log_level = LogLevel[log_level_str]
            set_global_log_level(log_level)
            # 同时更新当前日志器的级别
            self.logger.set_log_level(log_level)
            # 获取全局日志器并更新其级别
            global_logger = get_logger()
            global_logger.set_log_level(log_level)
            self.logger.info(f"日志级别设置为: {log_level_str}")
        except KeyError:
            self.logger.warning(f"无效的日志级别: {log_level_str}，使用默认级别")
        
    def initialize(self, config=None, skip_renderer=False):
        """
        启动魔法引擎的仪式！
        让所有组件都活起来，准备创造游戏世界
        
        参数:
            config: 魔法配方（引擎配置）
            skip_renderer: 是否跳过光影魔法师的召唤
            
        返回值:
            bool: 仪式成功返回True，否则返回False
        """
        # 加载配置文件
        self._load_config(config)
        
        self.logger.info("正在初始化高性能渲染引擎...")
        
        # 初始化平台模块
        try:
            self.logger.debug("1. 导入平台模块...")
            from .Platform.Platform import Platform
            self.plt = Platform()
            self.plt.initialize()
            self.logger.info("平台模块初始化完成")
        except Exception as e:
            self.logger.error(f"平台模块初始化失败: {e}")
            return False
        
        # 初始化资源管理器
        try:
            self.logger.debug("2. 导入并初始化资源管理器...")
            from .Renderer.Resources.ResourceManager import ResourceManager
            self.res_mgr = ResourceManager()
            self.res_mgr.initialize()
            self.logger.info("资源管理器初始化完成")
        except Exception as e:
            self.logger.error(f"资源管理器初始化失败: {e}")
            return False
        
        # 延迟渲染器初始化（等待OpenGL上下文创建）
        # 注意：渲染器将在OpenGLViewport.initgl()回调中初始化
        self.renderer = None
        self.renderer_config = config  # 保存配置供后续使用
        self.logger.info("渲染器将在OpenGL上下文创建后初始化")
        
        # 初始化场景管理器
        try:
            self.logger.debug("4. 导入并初始化场景管理器...")
            from .Scene.SceneManager import SceneManager
            self.scene_mgr = SceneManager(self)
            self.logger.info("场景管理器初始化完成")
        except ImportError as e:
            self.logger.error(f"场景管理器导入失败: {e}")
            self.logger.error(f"Python路径: {sys.path}")
            return False
        except Exception as e:
            self.logger.error(f"场景管理器初始化失败: {e}")
            return False
        
        # 初始化物理系统（可选，失败不影响引擎运行）
        try:
            self.logger.debug("5. 导入并初始化物理系统...")
            from .Physics.PhysicsSystem import PhysicsSystem
            self.physics_system = PhysicsSystem(self)
            self.logger.info("物理系统初始化完成")
        except Exception as e:
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
            self.logger.debug("7. 导入并初始化相机控制器...")
            from .UI.CameraController import CameraController
            self.camera_controller = CameraController(self.scene_mgr.active_camera)
            self.plt.camera_controller = self.camera_controller
            self.logger.info("相机控制器初始化完成")
        except Exception as e:
            self.logger.error(f"相机控制器初始化失败: {e}")
            return False
        
        try:
            # 导入并初始化变换操纵器
            self.logger.debug("8. 导入并初始化变换操纵器...")
            from .UI.Controls.TransformManipulator import TransformManipulator
            self.transform_manipulator = TransformManipulator(self.scene_mgr.active_camera)
            self.logger.info("变换操纵器初始化完成")
        except Exception as e:
            self.logger.error(f"变换操纵器初始化失败: {e}")
            return False
        
        try:
            # 导入并初始化UI管理器
            self.logger.debug("9. 导入并初始化UI管理器...")
            from .UI.UIManager import UIManager
            self.ui_mgr = UIManager(self.plt)
            self.logger.info("UI管理器初始化完成")
        except Exception as e:
            self.logger.error(f"UI管理器初始化失败: {e}")
            return False
        
        try:
            # 导入并初始化主窗口
            self.logger.debug("10. 导入并初始化主窗口...")
            from .UI.MainWindow import MainWindow
            self.main_window = MainWindow(self.plt)
            self.ui_mgr.add_control(self.main_window)
            self.logger.info("主窗口初始化完成")
        except Exception as e:
            self.logger.error(f"主窗口初始化失败: {e}")
            return False
        
        try:
            # 初始化Tkinter UI（新的主UI，替代HTML UI）
            if self.config["frontend"]["enable_tkinter"] or self.config["frontend"]["type"] == "tkinter":
                self.logger.debug("11. 导入并初始化Tkinter UI...")
                from .UI.TkUI.tk_main_window import TkMainWindow
                self.tk_ui = TkMainWindow(self)
                self.logger.info("Tkinter UI已启动")
            else:
                self.logger.warning("Tkinter UI未启用，引擎将以无UI模式运行")
        except Exception as e:
            self.logger.error(f"Tkinter UI启动失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            # UI失败不应导致引擎初始化失败

        try:
            # 初始化MCP服务器（Model Context Protocol）- AI建模API
            self.logger.debug("12. 初始化MCP服务器（AI建模API）...")
            from .MCP import MCPServer
            self.mcp_server = MCPServer(self)
            self.logger.info("MCP服务器已启动 - AI可以通过代码建模了！")
        except Exception as e:
            self.logger.warning(f"MCP服务器初始化失败: {e}")
            # MCP服务器失败不影响引擎运行

        try:
            # 启动HTTP API服务器（让运行中的引擎接收MCP命令）
            self.logger.debug("12.1 启动HTTP API服务器...")
            from .MCP.api_server import EngineAPIServer
            self.api_server = EngineAPIServer(self)
            self.api_server.start()
            self.logger.info("HTTP API服务器已启动 - AI可以通过HTTP调用引擎了！")
        except Exception as e:
            self.logger.warning(f"HTTP API服务器启动失败: {e}")
            # API服务器失败不影响引擎运行

        
        try:
            # 初始化MCP架构管理器
            self.logger.debug("13. 初始化MCP架构管理器...")
            from .Core.MCP.MCPManager import MCPManager
            self.mcp_mgr = MCPManager(self)
            self.logger.info("MCP架构初始化完成")
        except Exception as e:
            self.logger.error(f"MCP架构初始化失败: {e}")
            return False
        
        try:
            # 导入并初始化场景编辑器（可选）
            self.logger.debug("14. 尝试导入并初始化场景编辑器...")
            # 检查Editor目录是否存在
            import os
            editor_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Editor")
            if not os.path.exists(editor_dir):
                self.logger.info("Editor目录不存在，跳过场景编辑器初始化")
            else:
                from Editor.SceneEditor import SceneEditor
                self.scene_editor = SceneEditor(self)
                self.logger.info("场景编辑器初始化完成")
        except ImportError as e:
            self.logger.info(f"场景编辑器模块不存在，跳过初始化: {e}")
        except Exception as e:
            self.logger.error(f"场景编辑器初始化失败: {e}")
        
        self.is_initialized = True
        self.logger.info("引擎初始化完成")
        return True
    
    def _create_default_ui(self):
        """
        搭建游戏世界的控制台
        给玩家提供各种魔法按钮和工具
        """
        self.logger.debug("创建默认UI...")
        
        from .UI.Controls.Button import Button
        from .UI.Controls.Label import Label
        from .UI.Controls.Slider import Slider
        from .UI.Controls.Menu import Menu
        from .UI.Controls.PropertyPanel import PropertyPanel
        from .UI.Controls.Toolbar import Toolbar
        
        # 创建工具栏
        self.toolbar = Toolbar(0, 0, self.plt.width, 40)
        self.ui_mgr.add_control(self.toolbar)
        
        # 创建一个简单的菜单
        menu_items = [
            {"text": "创建立方体", "callback": self._create_cube},
            {"text": "创建球体", "callback": self._create_sphere},
            {"text": "创建圆柱体", "callback": self._create_cylinder},
            {"text": "导入模型", "callback": self._import_model},
            {"text": "退出", "callback": self.shutdown}
        ]
        menu = Menu(10, 50, 150, 30, menu_items)
        self.ui_mgr.add_control(menu)
        
        # 创建一个标签
        label = Label(200, 50, 200, 30, "Low End GPU Rendering Engine")
        self.ui_mgr.add_control(label)
        
        # 创建一个滑块
        slider = Slider(450, 50, 200, 30, 0.0, 1.0, 0.5)
        def on_slider_change(event):
            self.logger.debug(f"滑块值改变: {event.get_param('new_value')}")
        slider.add_event_handler("value_changed", on_slider_change)
        self.ui_mgr.add_control(slider)
        
        # 创建一个按钮
        button = Button(700, 50, 100, 30, "测试按钮")
        def on_button_click(event):
            self.logger.debug("测试按钮被点击")
        button.add_event_handler("mouse_click", on_button_click)
        self.ui_mgr.add_control(button)
        
        # 创建属性面板
        self.property_panel = PropertyPanel(self.plt.width - 300, 50, 290, self.plt.height - 60, "Properties")
        self.ui_mgr.add_control(self.property_panel)
        
        self.logger.debug("默认UI创建完成")
    
    def _create_cube(self):
        """创建立方体"""
        self.logger.info("创建立方体")
        from .Scene.SceneNode import SceneNode
        from .Math import Vector3
        from .Renderer.Resources.Mesh import Mesh
        from .Renderer.Resources.Material import Material
        
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
        self.scene_mgr.root_node.add_child(cube_node)
        self.logger.info(f"立方体 '{cube_node.name}' 已创建")
    
    def _create_sphere(self):
        """创建球体"""
        self.logger.info("创建球体")
        from .Scene.SceneNode import SceneNode
        from .Math import Vector3
        from .Renderer.Resources.Mesh import Mesh
        from .Renderer.Resources.Material import Material
        
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
        self.scene_mgr.root_node.add_child(sphere_node)
        self.logger.info(f"球体 '{sphere_node.name}' 已创建")
    
    def _create_cylinder(self):
        """创建圆柱体"""
        self.logger.info("创建圆柱体")
        from .Scene.SceneNode import SceneNode
        from .Math import Vector3
        from .Renderer.Resources.Mesh import Mesh
        from .Renderer.Resources.Material import Material
        
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
        self.scene_mgr.root_node.add_child(cylinder_node)
        self.logger.info(f"圆柱体 '{cylinder_node.name}' 已创建")
    
    def _create_plane(self):
        """创建平面"""
        self.logger.info("创建平面")
        from .Scene.SceneNode import SceneNode
        from .Math import Vector3
        from .Renderer.Resources.Mesh import Mesh
        from .Renderer.Resources.Material import Material
        
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
        self.scene_mgr.root_node.add_child(plane_node)
        self.logger.info(f"平面 '{plane_node.name}' 已创建")
    
    def _create_cone(self):
        """创建圆锥体"""
        self.logger.info("创建圆锥体")
        from .Scene.SceneNode import SceneNode
        from .Math import Vector3
        from .Renderer.Resources.Mesh import Mesh
        from .Renderer.Resources.Material import Material
        
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
        self.scene_mgr.root_node.add_child(cone_node)
        self.logger.info(f"圆锥体 '{cone_node.name}' 已创建")
    
    def _import_model(self):
        """导入模型"""
        self.logger.info("导入模型")
        from .Assets.ModelImporter import ModelImporter
        
        # 创建模型导入器
        importer = ModelImporter(self.res_mgr)
        
        # 这里可以替换为实际的模型文件路径，或者添加文件选择对话框
        # 目前使用一个示例模型文件路径
        model_file_path = "example_model.obj"
        
        try:
            # 导入模型
            imported_nodes = importer.import_model(model_file_path)
            
            # 将导入的节点添加到场景
            for node in imported_nodes:
                self.scene_mgr.root_node.add_child(node)
                self.logger.info(f"模型节点 '{node.name}' 已添加到场景")
            
            if imported_nodes:
                self.logger.info(f"成功导入 {len(imported_nodes)} 个模型节点")
            else:
                self.logger.warning("未导入任何模型节点")
        except Exception as e:
            self.logger.error(f"导入模型失败: {e}")
            
            # 如果导入失败，创建一个简单的占位符节点
            from .Scene.SceneNode import SceneNode
            from .Math import Vector3
            from .Renderer.Resources.Mesh import Mesh
            from .Renderer.Resources.Material import Material
            
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
            self.scene_mgr.root_node.add_child(model_node)
            self.logger.info(f"创建模型占位符 '{model_node.name}'")
    
    def _create_default_scene(self):
        """
        创建默认测试场景，包含相机、灯光和各种几何体
        Create default test scene with camera, lights, and various geometries
        """
        self.logger.debug("创建默认测试场景...")
        
        # 导入场景相关类
        # Import scene-related classes
        from .Scene.Camera import Camera
        from .Scene.Light import DirectionalLight, PointLight, SpotLight
        from .Math import Vector3
        from .Scene.SceneNode import SceneNode
        from .Renderer.Resources.Mesh import Mesh
        from .Renderer.Resources.Material import Material
        
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
        self.scene_mgr.active_camera = camera
        # 添加到相机列表
        # Add to camera list
        self.scene_mgr.cameras.append(camera)
        
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
        self.scene_mgr.light_manager.add_light(dir_light)
        
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
        self.scene_mgr.light_manager.add_light(point_light)
        
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
        self.scene_mgr.light_manager.add_light(spot_light)
        
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
        self.scene_mgr.root_node.add_child(ground_node)
        
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
        self.scene_mgr.root_node.add_child(cube_node)
        
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
        self.scene_mgr.root_node.add_child(sphere_node)
        
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
        self.scene_mgr.root_node.add_child(cylinder_node)
        
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
        self.scene_mgr.root_node.add_child(metal_sphere_node)
        
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
        self.scene_mgr.root_node.add_child(glass_cube_node)
        
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
            self.scene_mgr.root_node.add_child(small_cube_node)
        
        self.logger.debug("默认测试场景创建完成")

    def initialize_renderer_deferred(self):
        """
        延迟初始化渲染器（在OpenGL上下文创建后调用）
        Deferred renderer initialization (called after OpenGL context is created)
        """
        if self.renderer is not None:
            self.logger.warning("渲染器已经初始化，跳过重复初始化")
            return True

        try:
            self.logger.info("开始延迟初始化渲染器...")
            from .Renderer.Renderer import Renderer
            self.renderer = Renderer(self.plt, self.res_mgr)

            # 使用保存的配置初始化渲染器
            config = self.renderer_config if hasattr(self, 'renderer_config') else None
            self.renderer.initialize(config)

            self.logger.info("渲染器延迟初始化成功")
            return True
        except Exception as e:
            self.logger.error(f"渲染器延迟初始化失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            self.renderer = None
            return False

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
        
        if self.scene_mgr:
            self.scene_mgr.shutdown()
            self.scene_mgr = None
            
        if self.renderer:
            self.renderer.shutdown()
            self.renderer = None
            
        if self.res_mgr:
            self.res_mgr.shutdown()
            self.res_mgr = None
            
        if self.plt:
            self.plt.shutdown()
            self.plt = None
            
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
        if self.mcp_mgr:
            self.mcp_mgr.update(delta_time)
        else:
            # 兼容旧的更新方式
            # 更新物理系统
            if hasattr(self, 'physics_system') and self.physics_system:
                self.physics_system.update(delta_time)
            
            # 更新场景
            if self.scene_mgr:
                self.scene_mgr.update(delta_time)
            
            # 更新UI
            if hasattr(self, 'ui_mgr') and self.ui_mgr:
                self.ui_mgr.update(delta_time)
        
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
        if hasattr(self, 'mcp_mgr') and self.mcp_mgr:
            stats = self.mcp_mgr.render()
        else:
            # 兼容旧的渲染方式
            # 渲染场景
            if self.renderer and self.scene_mgr:
                # 简化的渲染调用，直接传递场景管理器
                self.renderer.render(self.scene_mgr)
            
            # 渲染场景编辑器
            if hasattr(self, 'scene_editor') and self.scene_editor:
                self.scene_editor.render()
            
            # 渲染UI
            if hasattr(self, 'ui_mgr') and self.ui_mgr:
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
                glOrtho(0, self.plt.width, 0, self.plt.height, -1, 1)
                
                # 设置模型视图矩阵
                glMatrixMode(GL_MODELVIEW)
                glLoadIdentity()
                
                # 设置正确的视口
                glViewport(0, 0, self.plt.width, self.plt.height)
                
                # 渲染UI
                self.ui_mgr.render()
                
                # 恢复深度测试，以便下一帧渲染3D场景
                glEnable(GL_DEPTH_TEST)
        
        # 将渲染结果发送到HTML UI
        if hasattr(self, 'html_ui_server') and self.html_ui_server:
            try:
                # 获取渲染结果
                if self.renderer:
                    # 渲染结果现在通过HTML UI服务器的状态更新线程自动发送
                    pass
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
                
                # 移除帧计数输出，减少终端冗余信息
                if frame_count % 60 == 0:
                    pass
                
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