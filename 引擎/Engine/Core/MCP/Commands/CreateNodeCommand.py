# -*- coding: utf-8 -*-
"""
MCP架构 - 创建节点命令
用于处理节点的创建操作
"""

from Engine.Core.MCP.Commands.Command import Command
from Engine.Scene.SceneNode import SceneNode
from Engine.Math import Vector3, Quaternion, Matrix4x4

class CreateNodeCommand(Command):
    """创建节点命令
    用于创建新的场景节点
    """
    
    def __init__(self, model, name, position=None, rotation=None, scale=None):
        """初始化创建节点命令
        
        Args:
            model: Model层实例
            name: 节点名称
            position: 节点位置
            rotation: 节点旋转（四元数）
            scale: 节点缩放
        """
        super().__init__()
        self.model = model
        self.name = name
        self.position = position or Vector3(0, 0, 0)
        self.rotation = rotation or Quaternion.identity()
        self.scale = scale or Vector3(1, 1, 1)
        self.node = None
        self.parent = None
    
    def execute(self):
        """执行创建节点命令
        
        Returns:
            bool: 命令是否执行成功
        """
        try:
            # 创建节点
            self.node = self.model.create_node(self.name, self.position, self.rotation, self.scale)
            
            # 记录节点的父节点
            self.parent = self.node.parent
            
            self.is_executed = True
            return True
        except Exception as e:
            self.model.engine.logger.error(f"创建节点命令执行失败: {e}")
            return False
    
    def undo(self):
        """撤销创建节点命令
        
        Returns:
            bool: 命令是否撤销成功
        """
        if not self.is_executed or not self.node:
            return False
        
        try:
            # 删除创建的节点
            self.model.delete_node(self.node)
            
            self.is_executed = False
            return True
        except Exception as e:
            self.model.engine.logger.error(f"撤销创建节点命令失败: {e}")
            return False