# -*- coding: utf-8 -*-
"""
MCP架构 - 更新属性命令
用于处理节点属性的更新操作
"""

from Engine.Core.MCP.Commands.Command import Command

class UpdatePropertyCommand(Command):
    """更新属性命令
    用于更新节点的属性
    """
    
    def __init__(self, node, property_name, new_value):
        """初始化更新属性命令
        
        Args:
            node: 要更新的节点
            property_name: 属性名称
            new_value: 新的属性值
        """
        super().__init__()
        self.node = node
        self.property_name = property_name
        self.new_value = new_value
        self.old_value = None
        
        # 保存旧值
        if hasattr(node, property_name):
            self.old_value = getattr(node, property_name)
    
    def execute(self):
        """执行更新属性命令
        
        Returns:
            bool: 命令是否执行成功
        """
        try:
            # 更新属性
            setattr(self.node, self.property_name, self.new_value)
            
            # 标记节点变换为脏
            self.node.mark_transform_dirty()
            
            self.is_executed = True
            return True
        except Exception as e:
            print(f"更新属性命令执行失败: {e}")
            return False
    
    def undo(self):
        """撤销更新属性命令
        
        Returns:
            bool: 命令是否撤销成功
        """
        if not self.is_executed:
            return False
        
        try:
            # 恢复旧值
            setattr(self.node, self.property_name, self.old_value)
            
            # 标记节点变换为脏
            self.node.mark_transform_dirty()
            
            self.is_executed = False
            return True
        except Exception as e:
            print(f"撤销更新属性命令失败: {e}")
            return False