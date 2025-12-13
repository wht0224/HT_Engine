"""
标签控件类，用于显示文本
"""

from Engine.UI.Controls.Control import Control

class Label(Control):
    """标签控件类，用于显示文本"""
    
    def __init__(self, x, y, width, height, text="Label"):
        """初始化标签
        
        Args:
            x: 标签X坐标
            y: 标签Y坐标
            width: 标签宽度
            height: 标签高度
            text: 标签文本
        """
        super().__init__(x, y, width, height)
        self.text = text
        self.name = "Label"
        self.text_color = (1.0, 1.0, 1.0, 1.0)
        self.align = "center"  # 文本对齐方式：left, center, right
        self.valign = "center"  # 垂直对齐方式：top, center, bottom
    
    def _render_content(self):
        """渲染标签内容"""
        from OpenGL.GL import glColor4f, glRasterPos2f
        from OpenGL.GLUT import glutBitmapCharacter, GLUT_BITMAP_HELVETICA_12
        
        # 设置文本颜色
        glColor4f(*self.text_color)
        
        # 计算文本位置
        text_width = len(self.text) * self.font_size * 0.6
        text_height = self.font_size
        
        # 水平对齐
        if self.align == "left":
            text_x = self.x + self.padding[3]
        elif self.align == "right":
            text_x = self.x + self.width - text_width - self.padding[1]
        else:  # center
            text_x = self.x + (self.width - text_width) / 2
        
        # 垂直对齐
        if self.valign == "top":
            text_y = self.y + self.height - self.padding[0] - text_height
        elif self.valign == "bottom":
            text_y = self.y + self.padding[2]
        else:  # center
            text_y = self.y + (self.height + text_height) / 2
        
        # 设置文本位置
        glRasterPos2f(text_x, text_y)
        
        # 渲染文本
        for char in self.text:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(char))
    
    def set_text(self, text):
        """设置标签文本
        
        Args:
            text: 标签文本
        """
        self.text = text
    
    def get_text(self):
        """获取标签文本
        
        Returns:
            str: 标签文本
        """
        return self.text
    
    def set_text_color(self, color):
        """设置文本颜色
        
        Args:
            color: 文本颜色（RGBA）
        """
        self.text_color = color
    
    def set_align(self, align):
        """设置文本对齐方式
        
        Args:
            align: 对齐方式（left, center, right）
        """
        if align in ["left", "center", "right"]:
            self.align = align
    
    def set_valign(self, valign):
        """设置垂直对齐方式
        
        Args:
            valign: 垂直对齐方式（top, center, bottom）
        """
        if valign in ["top", "center", "bottom"]:
            self.valign = valign
