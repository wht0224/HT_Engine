# -*- coding: utf-8 -*-
"""
属性面板控件类，用于显示和编辑选中对象的属性
"""

from Engine.UI.Controls.Control import Control
from Engine.UI.Controls.Label import Label
from Engine.UI.Controls.TextInput import TextInput
from Engine.UI.Controls.Slider import Slider
from Engine.UI.Event import EventType

class PropertyPanel(Control):
    """属性面板控件类"""
    
    def __init__(self, x, y, width, height, title="Properties"):
        """初始化属性面板
        
        Args:
            x: 面板X坐标
            y: 面板Y坐标
            width: 面板宽度
            height: 面板高度
            title: 面板标题
        """
        super().__init__(x, y, width, height)
        self.title = title
        self.name = "PropertyPanel"
        self.selected_object = None
        self.property_controls = {}
        self.title_height = 25
        self.padding = 10
        self.spacing = 5
        self.title_color = (1.0, 1.0, 1.0, 1.0)
        # 设置属性面板背景色为深灰色，与主窗口背景色区分
        self.background_color = (0.2, 0.2, 0.2, 1.0)
        # 确保属性面板可见
        self.is_visible = True
        
    def _render_content(self):
        """渲染面板内容"""
        from OpenGL.GL import glColor4f, glRasterPos2f
        from OpenGL.GLUT import glutBitmapCharacter, GLUT_BITMAP_HELVETICA_12
        
        # 渲染标题
        glColor4f(*self.title_color)
        title_x = self.x + 10
        title_y = self.y + self.height - 15
        glRasterPos2f(title_x, title_y)
        
        for char in self.title:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(char))
    
    def set_selected_object(self, obj):
        """设置选中的对象
        
        Args:
            obj: 选中的对象
        """
        self.selected_object = obj
        self._update_property_controls()
    
    def _update_property_controls(self):
        """更新属性控件"""
        # 清除现有的属性控件
        for control in self.property_controls.values():
            self.remove_child(control)
        self.property_controls.clear()
        
        # 创建默认属性显示
        current_y = self.y + self.height - self.title_height - self.padding
        
        if not self.selected_object:
            # 显示默认提示信息
            no_selection_label = Label(self.x + self.padding, current_y - 15, self.width - 2 * self.padding, 20, "未选择对象")
            self.add_child(no_selection_label)
            self.property_controls["no_selection_label"] = no_selection_label
            return
        
        # 获取所有属性
        properties = self._get_object_properties()
        
        # 添加对象名称标签
        name_label = Label(self.x + self.padding, current_y - 15, self.width - 2 * self.padding, 20, f"对象: {self.selected_object.name}")
        name_label.foreground_color = (0.8, 0.8, 0.2, 1.0)  # 使用黄色突出显示
        self.add_child(name_label)
        self.property_controls["name_label"] = name_label
        current_y -= 25
        
        # 创建属性控件
        for prop_name, prop_value in properties.items():
            # 跳过特殊属性
            if prop_name.startswith('_') or prop_name in ['name', 'parent', 'children']:
                continue
            
            # 创建标签
            label = Label(self.x + self.padding, current_y - 15, self.width - 2 * self.padding, 20, prop_name)
            self.add_child(label)
            self.property_controls[f"{prop_name}_label"] = label
            
            current_y -= 20
            
            # 根据属性类型创建相应的控件
            if isinstance(prop_value, (int, float)):
                # 创建滑块和文本输入框
                slider = Slider(self.x + self.padding, current_y - 20, self.width - 2 * self.padding - 80, 20, 
                              min_value=0.0, max_value=100.0, value=float(prop_value))
                slider.add_event_handler(EventType.VALUE_CHANGED, lambda e, name=prop_name: self._on_property_changed(name, e.value))
                self.add_child(slider)
                self.property_controls[f"{prop_name}_slider"] = slider
                
                text_input = TextInput(self.x + self.width - self.padding - 70, current_y - 20, 60, 20, str(prop_value))
                text_input.add_event_handler(EventType.TEXT_CHANGED, lambda e, name=prop_name: self._on_text_changed(name, e.text))
                self.add_child(text_input)
                self.property_controls[f"{prop_name}_input"] = text_input
                
            elif isinstance(prop_value, str):
                # 创建文本输入框
                text_input = TextInput(self.x + self.padding, current_y - 20, self.width - 2 * self.padding, 20, prop_value)
                text_input.add_event_handler(EventType.TEXT_CHANGED, lambda e, name=prop_name: self._on_text_changed(name, e.text))
                self.add_child(text_input)
                self.property_controls[f"{prop_name}_input"] = text_input
            
            current_y -= 25
    

    
    def handle_mouse_down(self, button, x, y):
        """处理鼠标按下事件
        
        Args:
            button: 鼠标按钮
            x: 鼠标X坐标
            y: 鼠标Y坐标
            
        Returns:
            bool: 是否处理了事件
        """
        # 调用父类方法处理控件
        return super().handle_mouse_down(button, x, y)
    
    def _get_object_properties(self):
        """获取对象属性
        
        Returns:
            dict: 对象属性字典
        """
        properties = {}
        
        # 获取对象的所有公共属性
        for attr in dir(self.selected_object):
            if not attr.startswith('_') and not callable(getattr(self.selected_object, attr)):
                try:
                    value = getattr(self.selected_object, attr)
                    # 只添加基本类型的属性
                    if isinstance(value, (int, float, str, bool)):
                        properties[attr] = value
                except Exception:
                    pass
        
        return properties
    
    def _on_property_changed(self, prop_name, value):
        """属性值改变事件处理
        
        Args:
            prop_name: 属性名
            value: 新的属性值
        """
        if not self.selected_object:
            return
        
        # 更新对象属性
        try:
            current_value = getattr(self.selected_object, prop_name)
            if isinstance(current_value, int):
                value = int(value)
            setattr(self.selected_object, prop_name, value)
            
            # 更新对应的文本输入框
            if f"{prop_name}_input" in self.property_controls:
                input_control = self.property_controls[f"{prop_name}_input"]
                input_control.set_text(str(value))
        except Exception as e:
            print(f"Error updating property {prop_name}: {e}")
    
    def _on_text_changed(self, prop_name, text):
        """文本输入改变事件处理
        
        Args:
            prop_name: 属性名
            text: 新的文本值
        """
        if not self.selected_object:
            return
        
        # 更新对象属性
        try:
            current_value = getattr(self.selected_object, prop_name)
            if isinstance(current_value, int):
                value = int(text)
            elif isinstance(current_value, float):
                value = float(text)
            else:
                value = text
            
            setattr(self.selected_object, prop_name, value)
            
            # 更新对应的滑块
            if f"{prop_name}_slider" in self.property_controls:
                slider = self.property_controls[f"{prop_name}_slider"]
                slider.set_value(float(value))
        except Exception as e:
            print(f"Error updating property {prop_name}: {e}")
    
    def update(self, delta_time):
        """更新属性面板
        
        Args:
            delta_time: 帧时间（秒）
        """
        super().update(delta_time)
        
        # 实时更新属性控件的值（如果对象属性被外部修改）
        if self.selected_object:
            for prop_name in list(self.property_controls.keys()):
                if "_label" in prop_name:
                    continue
                    
                base_name = prop_name.split('_')[0]
                try:
                    current_value = getattr(self.selected_object, base_name)
                    
                    # 更新滑块
                    if f"{base_name}_slider" in self.property_controls:
                        slider = self.property_controls[f"{base_name}_slider"]
                        if abs(slider.value - float(current_value)) > 0.01:
                            slider.set_value(float(current_value))
                    
                    # 更新文本输入框
                    if f"{base_name}_input" in self.property_controls:
                        input_control = self.property_controls[f"{base_name}_input"]
                        if input_control.get_text() != str(current_value):
                            input_control.set_text(str(current_value))
                except Exception:
                    pass