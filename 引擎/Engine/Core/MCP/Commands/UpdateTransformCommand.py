# -*- coding: utf-8 -*-
"""
MCP架构 - 更新变换命令
用于处理节点变换的更新操作
"""

from Engine.Core.MCP.Commands.Command import Command
from Engine.Math import Vector3, Quaternion, Matrix4x4

class UpdateTransformCommand(Command):
    """更新变换命令
    用于更新节点的变换（位置、旋转、缩放）
    """
    
    def __init__(self, node, position=None, rotation=None, scale=None):
        """初始化更新变换命令
        
        Args:
            node: 要更新的节点
            position: 新位置
            rotation: 新旋转（四元数）
            scale: 新缩放
        """
        super().__init__()
        self.node = node
        self.new_position = position
        self.new_rotation = rotation
        self.new_scale = scale
        
        # 保存旧值
        self.old_position = node.position.copy()
        self.old_rotation = node.rotation.copy()
        self.old_scale = node.scale.copy()
    
    def execute(self):
        """执行更新变换命令
        
        Returns:
            bool: 命令是否执行成功
        """
        try:
            # 更新变换
            if self.new_position is not None:
                self.node.set_position(self.new_position)
            if self.new_rotation is not None:
                self.node.set_rotation(self.new_rotation)
            if self.new_scale is not None:
                self.node.set_scale(self.new_scale)
            
            self.is_executed = True
            return True
        except Exception as e:
            print(f"更新变换命令执行失败: {e}")
            return False
    
    def undo(self):
        """撤销更新变换命令
        
        Returns:
            bool: 命令是否撤销成功
        """
        if not self.is_executed:
            return False
        
        try:
            # 恢复旧变换
            self.node.set_position(self.old_position)
            self.node.set_rotation(self.old_rotation)
            self.node.set_scale(self.old_scale)
            
            self.is_executed = False
            return True
        except Exception as e:
            print(f"撤销更新变换命令失败: {e}")
            return False