# -*- coding: utf-8 -*-
"""
MCP架构 - MCP管理器
负责管理MCP架构的各个组件
"""

from Engine.Core.MCP.Model import Model
from Engine.Core.MCP.Controller import Controller
from Engine.Core.MCP.Processor import Processor

class MCPManager:
    """MCP架构管理器
    负责管理MCP架构的Model、Controller和Processor层
    """
    
    def __init__(self, engine):
        """初始化MCP管理器
        
        Args:
            engine: 引擎实例
        """
        self.engine = engine
        
        # 初始化MCP架构的各个层
        self.model = Model(engine)
        self.controller = Controller(self.model)
        self.processor = Processor(self.model, self.controller)
    
    def get_model(self):
        """获取Model层实例
        
        Returns:
            Model: Model层实例
        """
        return self.model
    
    def get_controller(self):
        """获取Controller层实例
        
        Returns:
            Controller: Controller层实例
        """
        return self.controller
    
    def get_processor(self):
        """获取Processor层实例
        
        Returns:
            Processor: Processor层实例
        """
        return self.processor
    
    def update(self, delta_time):
        """更新MCP架构
        
        Args:
            delta_time: 帧时间
        """
        self.processor.update(delta_time)
    
    def render(self):
        """渲染场景
        
        Returns:
            dict: 渲染统计信息
        """
        return self.processor.render()
    
    def get_performance_stats(self):
        """获取性能统计信息
        
        Returns:
            dict: 性能统计信息
        """
        return self.processor.get_performance_stats()
    
    def handle_mouse_event(self, event_type, x, y, buttons, modifiers):
        """处理鼠标事件
        
        Args:
            event_type: 事件类型 (down, up, move, wheel)
            x: 鼠标X坐标
            y: 鼠标Y坐标
            buttons: 鼠标按钮状态
            modifiers: 键盘修饰键状态
            
        Returns:
            bool: 是否处理了事件
        """
        return self.controller.handle_mouse_event(event_type, x, y, buttons, modifiers)
    
    def handle_key_event(self, event_type, key, modifiers):
        """处理键盘事件
        
        Args:
            event_type: 事件类型 (down, up)
            key: 键码
            modifiers: 键盘修饰键状态
            
        Returns:
            bool: 是否处理了事件
        """
        return self.controller.handle_key_event(event_type, key, modifiers)
    
    def handle_ui_event(self, event_type, **kwargs):
        """处理UI事件
        
        Args:
            event_type: 事件类型
            **kwargs: 事件参数
            
        Returns:
            bool: 是否处理了事件
        """
        return self.controller.handle_ui_event(event_type, **kwargs)
    
    def execute_command(self, command):
        """执行命令
        
        Args:
            command: 命令对象
            
        Returns:
            bool: 命令是否执行成功
        """
        return self.controller.execute_command(command)
    
    def undo(self):
        """撤销上一条命令
        
        Returns:
            bool: 是否撤销成功
        """
        return self.controller.undo()
    
    def redo(self):
        """重做下一条命令
        
        Returns:
            bool: 是否重做成功
        """
        return self.controller.redo()
    
    def get_command_history(self):
        """获取命令历史记录
        
        Returns:
            list: 命令历史记录
        """
        return self.controller.get_command_history()
    
    def get_command_index(self):
        """获取当前命令索引
        
        Returns:
            int: 当前命令索引
        """
        return self.controller.get_command_index()
    
    def create_node(self, name, position=None, rotation=None, scale=None):
        """创建场景节点
        
        Args:
            name: 节点名称
            position: 位置
            rotation: 旋转（四元数）
            scale: 缩放
            
        Returns:
            SceneNode: 创建的节点
        """
        return self.controller.create_node(name, position, rotation, scale)
    
    def delete_node(self, node):
        """删除节点
        
        Args:
            node: 要删除的节点
            
        Returns:
            bool: 是否删除成功
        """
        return self.controller.delete_node(node)
    
    def select_node(self, node):
        """选择单个节点
        
        Args:
            node: 要选择的场景节点
        """
        self.controller.select_node(node)
    
    def clear_selection(self):
        """清空选择列表"""
        self.controller.clear_selection()
    
    def get_selected_node(self):
        """获取当前选中的单个节点
        
        Returns:
            SceneNode: 当前选中的节点
        """
        return self.model.get_selected_node()
    
    def get_selected_nodes(self):
        """获取所有选中的节点
        
        Returns:
            list: 选中的节点列表
        """
        return self.model.get_selected_nodes()
    
    def set_modeling_mode(self, mode):
        """设置建模模式
        
        Args:
            mode: 建模模式 (object, vertex, edge, face)
        """
        self.model.set_modeling_mode(mode)
    
    def set_transform_mode(self, mode):
        """设置变换模式
        
        Args:
            mode: 变换模式 (translate, rotate, scale)
        """
        self.model.set_transform_mode(mode)
    
    def toggle_snap(self):
        """切换捕捉功能"""
        self.model.toggle_snap()
    
    def set_snap_distance(self, distance):
        """设置捕捉距离
        
        Args:
            distance: 捕捉距离
        """
        self.model.set_snap_distance(distance)
    
    def get_scene_state(self):
        """获取场景状态
        
        Returns:
            dict: 场景状态信息
        """
        return self.model.get_scene_state()
    
    def set_scene_state(self, state):
        """设置场景状态
        
        Args:
            state: 场景状态信息
        """
        self.model.set_scene_state(state)
    
    def create_cube(self, size=1.0):
        """创建立方体
        
        Args:
            size: 立方体大小
            
        Returns:
            Mesh: 立方体网格
        """
        return self.processor.modeling_processor.create_cube(size)
    
    def create_sphere(self, radius=1.0, segments=32):
        """创建球体
        
        Args:
            radius: 球体半径
            segments: 球体分段数
            
        Returns:
            Mesh: 球体网格
        """
        return self.processor.modeling_processor.create_sphere(radius, segments)
    
    def create_cylinder(self, radius=1.0, height=2.0, segments=32):
        """创建圆柱体
        
        Args:
            radius: 圆柱体半径
            height: 圆柱体高度
            segments: 圆柱体分段数
            
        Returns:
            Mesh: 圆柱体网格
        """
        return self.processor.modeling_processor.create_cylinder(radius, height, segments)
    
    def create_plane(self, width=1.0, height=1.0, segments=1):
        """创建平面
        
        Args:
            width: 平面宽度
            height: 平面高度
            segments: 平面分段数
            
        Returns:
            Mesh: 平面网格
        """
        return self.processor.modeling_processor.create_plane(width, height, segments)
    
    def subdivide_mesh(self, mesh, levels=1):
        """细分网格
        
        Args:
            mesh: 要细分的网格
            levels: 细分级别
            
        Returns:
            Mesh: 细分后的网格
        """
        return self.processor.modeling_processor.subdivide_mesh(mesh, levels)
    
    def merge_meshes(self, meshes):
        """合并网格
        
        Args:
            meshes: 要合并的网格列表
            
        Returns:
            Mesh: 合并后的网格
        """
        return self.processor.modeling_processor.merge_meshes(meshes)
    
    def update_property_animated(self, node, property_name, value, duration=0.3):
        """动画更新节点属性
        
        Args:
            node: 要更新的节点
            property_name: 属性名称
            value: 新的属性值
            duration: 动画持续时间
        """
        return self.model.update_property_animated(node, property_name, value, duration)
    
    def update_visual_feedback(self, feedback_type, value):
        """更新视觉反馈设置
        
        Args:
            feedback_type: 视觉反馈类型
            value: 新的视觉反馈值
        """
        return self.model.update_visual_feedback(feedback_type, value)
    
    def toggle_visual_feedback(self, feedback_type):
        """切换视觉反馈设置
        
        Args:
            feedback_type: 视觉反馈类型
        """
        return self.model.toggle_visual_feedback(feedback_type)
    
    def get_visual_feedback(self, feedback_type=None):
        """获取视觉反馈设置
        
        Args:
            feedback_type: 视觉反馈类型，如果为None则返回所有设置
            
        Returns:
            dict or any: 视觉反馈设置
        """
        return self.model.get_visual_feedback(feedback_type)
    
    def get_modeling_stats(self):
        """获取建模统计信息
        
        Returns:
            dict: 建模统计信息
        """
        return self.model.get_modeling_stats()
    
    def update_modeling_stats(self):
        """更新建模统计信息"""
        return self.model.update_modeling_stats()
    
    def get_scene_state(self):
        """获取场景状态
        
        Returns:
            dict: 场景状态信息
        """
        state = self.model.get_scene_state()
        state["visual_feedback"] = self.model.get_visual_feedback()
        state["modeling_stats"] = self.model.get_modeling_stats()
        return state
    
    def set_scene_state(self, state):
        """设置场景状态
        
        Args:
            state: 场景状态信息
        """
        return self.model.set_scene_state(state)
    
    def update_node_transform(self, node, position=None, rotation=None, scale=None):
        """更新节点变换
        
        Args:
            node: 要更新的节点
            position: 新位置
            rotation: 新旋转
            scale: 新缩放
            
        Returns:
            bool: 是否更新成功
        """
        return self.controller.update_node_transform(node, position, rotation, scale)