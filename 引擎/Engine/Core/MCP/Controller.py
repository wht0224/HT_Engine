# -*- coding: utf-8 -*-
"""
MCP架构 - Controller层
负责处理用户输入和业务逻辑
"""

from Engine.Core.MCP.Model import Model
from Engine.Scene.SceneNode import SceneNode
from Engine.Math import Vector3, Quaternion, Matrix4x4
from Engine.Renderer.Resources.Mesh import Mesh
from Engine.Renderer.Resources.Material import Material

class Controller:
    """MCP架构的Controller层
    负责处理用户输入、命令系统、事件处理和状态管理
    """
    
    def __init__(self, model):
        """初始化Controller层
        
        Args:
            model: Model层实例
        """
        self.model = model
        self.engine = model.engine
        
        # 命令系统
        self.command_history = []
        self.command_index = -1
        self.max_history_size = 100
        
        # 事件系统
        self.event_handlers = {}
        
        # 选择状态
        self.is_selecting = False
        self.selection_start = None
        self.selection_end = None
        
        # 变换状态
        self.is_transforming = False
        self.transform_start = None
        self.transform_end = None
        
        # 建模状态
        self.is_modeling = False
        self.modeling_start = None
        self.modeling_end = None
    
    def get_model(self):
        """获取Model层实例
        
        Returns:
            Model: Model层实例
        """
        return self.model
    
    def execute_command(self, command):
        """执行命令
        
        Args:
            command: 命令对象，必须实现execute()和undo()方法
            
        Returns:
            bool: 命令是否执行成功
        """
        try:
            # 执行命令
            result = command.execute()
            if result:
                # 将命令添加到历史记录
                self.command_history = self.command_history[:self.command_index + 1]
                self.command_history.append(command)
                if len(self.command_history) > self.max_history_size:
                    self.command_history.pop(0)
                self.command_index = len(self.command_history) - 1
                
                # 触发命令执行事件
                self._trigger_event("command_executed", command=command)
                
                return True
            return False
        except Exception as e:
            self.engine.logger.error(f"执行命令失败: {e}")
            return False
    
    def undo(self):
        """撤销上一条命令
        
        Returns:
            bool: 是否撤销成功
        """
        if self.command_index >= 0:
            command = self.command_history[self.command_index]
            try:
                result = command.undo()
                if result:
                    self.command_index -= 1
                    self._trigger_event("command_undone", command=command)
                    return True
            except Exception as e:
                self.engine.logger.error(f"撤销命令失败: {e}")
        return False
    
    def redo(self):
        """重做下一条命令
        
        Returns:
            bool: 是否重做成功
        """
        if self.command_index < len(self.command_history) - 1:
            self.command_index += 1
            command = self.command_history[self.command_index]
            try:
                result = command.execute()
                if result:
                    self._trigger_event("command_redone", command=command)
                    return True
            except Exception as e:
                self.engine.logger.error(f"重做命令失败: {e}")
                self.command_index -= 1
        return False
    
    def add_event_handler(self, event_name, handler):
        """添加事件处理器
        
        Args:
            event_name: 事件名称
            handler: 事件处理函数
        """
        if event_name not in self.event_handlers:
            self.event_handlers[event_name] = []
        self.event_handlers[event_name].append(handler)
    
    def remove_event_handler(self, event_name, handler):
        """移除事件处理器
        
        Args:
            event_name: 事件名称
            handler: 事件处理函数
        """
        if event_name in self.event_handlers:
            if handler in self.event_handlers[event_name]:
                self.event_handlers[event_name].remove(handler)
    
    def _trigger_event(self, event_name, **kwargs):
        """触发事件
        
        Args:
            event_name: 事件名称
            **kwargs: 事件参数
        """
        if event_name in self.event_handlers:
            for handler in self.event_handlers[event_name]:
                try:
                    handler(**kwargs)
                except Exception as e:
                    self.engine.logger.error(f"执行事件处理器失败: {e}")
    
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
        if event_type == "down":
            return self._handle_mouse_down(x, y, buttons, modifiers)
        elif event_type == "up":
            return self._handle_mouse_up(x, y, buttons, modifiers)
        elif event_type == "move":
            return self._handle_mouse_move(x, y, buttons, modifiers)
        elif event_type == "wheel":
            return self._handle_mouse_wheel(x, y, buttons, modifiers)
        return False
    
    def _handle_mouse_down(self, x, y, buttons, modifiers):
        """处理鼠标按下事件
        
        Args:
            x: 鼠标X坐标
            y: 鼠标Y坐标
            buttons: 鼠标按钮状态
            modifiers: 键盘修饰键状态
            
        Returns:
            bool: 是否处理了事件
        """
        # 选择对象
        if buttons == 1:  # 左键
            node = self.model.scene_manager.pick_node(x, y)
            if node:
                if modifiers & 0x01:  # Shift键
                    if node in self.model.selected_nodes:
                        self.model.remove_from_selection(node)
                    else:
                        self.model.add_to_selection(node)
                else:
                    self.model.select_node(node)
                self.is_selecting = True
                self.selection_start = (x, y)
                self._trigger_event("node_selected", node=node)
                return True
        return False
    
    def _handle_mouse_up(self, x, y, buttons, modifiers):
        """处理鼠标释放事件
        
        Args:
            x: 鼠标X坐标
            y: 鼠标Y坐标
            buttons: 鼠标按钮状态
            modifiers: 键盘修饰键状态
            
        Returns:
            bool: 是否处理了事件
        """
        if self.is_selecting:
            self.is_selecting = False
            self.selection_end = (x, y)
            self._trigger_event("selection_complete", start=self.selection_start, end=self.selection_end)
            return True
        return False
    
    def _handle_mouse_move(self, x, y, buttons, modifiers):
        """处理鼠标移动事件
        
        Args:
            x: 鼠标X坐标
            y: 鼠标Y坐标
            buttons: 鼠标按钮状态
            modifiers: 键盘修饰键状态
            
        Returns:
            bool: 是否处理了事件
        """
        if self.is_selecting:
            self._trigger_event("selection_moved", x=x, y=y, start=self.selection_start)
            return True
        return False
    
    def _handle_mouse_wheel(self, x, y, buttons, modifiers):
        """处理鼠标滚轮事件
        
        Args:
            x: 鼠标X坐标
            y: 鼠标Y坐标
            buttons: 鼠标按钮状态
            modifiers: 键盘修饰键状态
            
        Returns:
            bool: 是否处理了事件
        """
        # 缩放视图
        if hasattr(self.engine, 'camera_controller'):
            zoom_factor = 1.1 if y > 0 else 0.9
            self.engine.camera_controller.zoom(zoom_factor)
            return True
        return False
    
    def handle_key_event(self, event_type, key, modifiers):
        """处理键盘事件
        
        Args:
            event_type: 事件类型 (down, up)
            key: 键码
            modifiers: 键盘修饰键状态
            
        Returns:
            bool: 是否处理了事件
        """
        if event_type == "down":
            return self._handle_key_down(key, modifiers)
        return False
    
    def _handle_key_down(self, key, modifiers):
        """处理键盘按下事件
        
        Args:
            key: 键码
            modifiers: 键盘修饰键状态
            
        Returns:
            bool: 是否处理了事件
        """
        # 快捷键处理
        if modifiers & 0x08:  # Ctrl键
            if key == 90:  # Z键
                return self.undo()
            elif key == 89:  # Y键
                return self.redo()
        
        # 变换模式切换
        if key == 87:  # W键 - 移动模式
            self.model.set_transform_mode("translate")
            return True
        elif key == 69:  # E键 - 旋转模式
            self.model.set_transform_mode("rotate")
            return True
        elif key == 82:  # R键 - 缩放模式
            self.model.set_transform_mode("scale")
            return True
        
        # 建模模式切换
        if key == 79:  # O键 - 对象模式
            self.model.set_modeling_mode("object")
            return True
        elif key == 86:  # V键 - 顶点模式
            self.model.set_modeling_mode("vertex")
            return True
        elif key == 69:  # E键 - 边模式
            self.model.set_modeling_mode("edge")
            return True
        elif key == 70:  # F键 - 面模式
            self.model.set_modeling_mode("face")
            return True
        
        # 删除选中对象
        if key == 46:  # Delete键
            for node in self.model.selected_nodes[:]:
                self.model.delete_node(node)
            return True
        
        return False
    
    def handle_ui_event(self, event_type, **kwargs):
        """处理UI事件
        
        Args:
            event_type: 事件类型
            **kwargs: 事件参数
            
        Returns:
            bool: 是否处理了事件
        """
        # 触发UI事件
        return self._trigger_event(event_type, **kwargs)
    
    def update_node_property(self, node, property_name, value):
        """更新节点属性
        
        Args:
            node: 要更新的节点
            property_name: 属性名称
            value: 新的属性值
            
        Returns:
            bool: 是否更新成功
        """
        # 创建并执行更新属性命令
        from Engine.Core.MCP.Commands.UpdatePropertyCommand import UpdatePropertyCommand
        command = UpdatePropertyCommand(node, property_name, value)
        return self.execute_command(command)
    
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
        # 创建并执行更新变换命令
        from Engine.Core.MCP.Commands.UpdateTransformCommand import UpdateTransformCommand
        command = UpdateTransformCommand(node, position, rotation, scale)
        return self.execute_command(command)
    
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
        # 创建并执行创建节点命令
        from Engine.Core.MCP.Commands.CreateNodeCommand import CreateNodeCommand
        command = CreateNodeCommand(self.model, name, position, rotation, scale)
        if self.execute_command(command):
            return command.node
        return None
    
    def delete_node(self, node):
        """删除节点
        
        Args:
            node: 要删除的节点
            
        Returns:
            bool: 是否删除成功
        """
        # 创建并执行删除节点命令
        from Engine.Core.MCP.Commands.DeleteNodeCommand import DeleteNodeCommand
        command = DeleteNodeCommand(self.model, node)
        return self.execute_command(command)
    
    def select_node(self, node):
        """选择单个节点
        
        Args:
            node: 要选择的场景节点
        """
        self.model.select_node(node)
        self._trigger_event("node_selected", node=node)
    
    def clear_selection(self):
        """清空选择列表"""
        self.model.clear_selection()
        self._trigger_event("selection_cleared")
    
    def get_command_history(self):
        """获取命令历史记录
        
        Returns:
            list: 命令历史记录
        """
        return self.command_history
    
    def get_command_index(self):
        """获取当前命令索引
        
        Returns:
            int: 当前命令索引
        """
        return self.command_index
    
    def _trigger_event(self, event_name, **kwargs):
        """触发事件
        
        Args:
            event_name: 事件名称
            **kwargs: 事件参数
        """
        if event_name in self.event_handlers:
            for handler in self.event_handlers[event_name]:
                try:
                    handler(**kwargs)
                except Exception as e:
                    self.engine.logger.error(f"执行事件处理器失败: {e}")
