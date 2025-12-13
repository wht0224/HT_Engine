"""
MCP架构 - 命令基类
定义了命令的基本接口
"""

class Command:
    """命令基类
    所有命令都必须继承自此类，并实现execute()和undo()方法
    """
    
    def __init__(self):
        """初始化命令
        """
        self.name = self.__class__.__name__
        self.description = ""
        self.is_executed = False
    
    def execute(self):
        """执行命令
        
        Returns:
            bool: 命令是否执行成功
        """
        raise NotImplementedError("子类必须实现execute()方法")
    
    def undo(self):
        """撤销命令
        
        Returns:
            bool: 命令是否撤销成功
        """
        raise NotImplementedError("子类必须实现undo()方法")
    
    def redo(self):
        """重做命令
        
        Returns:
            bool: 命令是否重做成功
        """
        return self.execute()
    
    def get_name(self):
        """获取命令名称
        
        Returns:
            str: 命令名称
        """
        return self.name
    
    def get_description(self):
        """获取命令描述
        
        Returns:
            str: 命令描述
        """
        return self.description
    
    def is_executed(self):
        """检查命令是否已执行
        
        Returns:
            bool: 命令是否已执行
        """
        return self.is_executed