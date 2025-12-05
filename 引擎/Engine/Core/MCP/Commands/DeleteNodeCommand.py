# -*- coding: utf-8 -*-
"""
MCP架构 - 删除节点命令
用于处理节点的删除操作
"""

from Engine.Core.MCP.Commands.Command import Command
from Engine.Scene.SceneNode import SceneNode

class DeleteNodeCommand(Command):
    """删除节点命令
    用于删除场景节点
    """
    
    def __init__(self, model, node):
        """初始化删除节点命令
        
        Args:
            model: Model层实例
            node: 要删除的节点
        """
        super().__init__()
        self.model = model
        self.node = node
        self.node_name = node.name
        self.parent = node.parent
        self.children = []
        self.position = node.position.copy()
        self.rotation = node.rotation.copy()
        self.scale = node.scale.copy()
        self.mesh = node.mesh
        self.material = node.material
        self.animation = node.animation
        self.components = {k: v.copy() for k, v in node.components.items()}
        
        # 保存子节点信息
        for child in node.children:
            self.children.append({
                "node": child,
                "position": child.position.copy(),
                "rotation": child.rotation.copy(),
                "scale": child.scale.copy()
            })
    
    def execute(self):
        """执行删除节点命令
        
        Returns:
            bool: 命令是否执行成功
        """
        try:
            # 删除节点
            result = self.model.delete_node(self.node)
            
            self.is_executed = result
            return result
        except Exception as e:
            self.model.engine.logger.error(f"删除节点命令执行失败: {e}")
            return False
    
    def undo(self):
        """撤销删除节点命令
        
        Returns:
            bool: 命令是否撤销成功
        """
        if not self.is_executed:
            return False
        
        try:
            # 重新创建节点
            new_node = self.model.create_node(
                self.node_name,
                self.position,
                self.rotation,
                self.scale
            )
            
            # 恢复节点属性
            new_node.mesh = self.mesh
            new_node.material = self.material
            new_node.animation = self.animation
            
            # 恢复组件
            for component_type, components in self.components.items():
                for component in components:
                    new_node.add_component(component)
            
            # 恢复子节点
            for child_info in self.children:
                child_node = child_info["node"]
                # 重新创建子节点
                new_child = self.model.create_node(
                    child_node.name,
                    child_info["position"],
                    child_info["rotation"],
                    child_info["scale"]
                )
                # 恢复子节点属性
                new_child.mesh = child_node.mesh
                new_child.material = child_node.material
                new_child.animation = child_node.animation
                # 添加到父节点
                new_node.add_child(new_child)
            
            # 更新引用
            self.node = new_node
            
            self.is_executed = False
            return True
        except Exception as e:
            self.model.engine.logger.error(f"撤销删除节点命令失败: {e}")
            return False