# -*- coding: utf-8 -*-
"""
MCP架构 - Model层
负责数据存储和管理
"""

from Engine.Scene.SceneManager import SceneManager
from Engine.Scene.SceneNode import SceneNode
from Engine.Math import Vector3, Quaternion, Matrix4x4
from Engine.Renderer.Resources.Mesh import Mesh
from Engine.Renderer.Resources.Material import Material

class Model:
    """MCP架构的Model层
    负责管理场景数据、资源数据和UI状态
    """
    
    def __init__(self, engine):
        """初始化Model层
        
        Args:
            engine: 引擎实例
        """
        self.engine = engine
        self.scene_manager = SceneManager(engine)
        
        # 场景状态
        self.is_editing = False
        self.selected_node = None
        self.selected_nodes = []
        
        # 建模状态
        self.modeling_mode = "object"  # object, vertex, edge, face
        self.transform_mode = "translate"  # translate, rotate, scale
        self.snap_enabled = False
        self.snap_distance = 0.1
        
        # 实时更新标志
        self.needs_update = False
        self.needs_render = False
        self.needs_property_update = False
        
        # 视觉反馈状态
        self.visual_feedback = {
            "show_grid": True,
            "show_wireframe": False,
            "show_bounding_boxes": False,
            "show_normals": False,
            "show_vertices": False,
            "show_edges": False,
            "show_faces": False,
            "selection_color": Vector3(1.0, 0.0, 0.0),
            "hover_color": Vector3(0.0, 1.0, 0.0),
            "grid_color": Vector3(0.3, 0.3, 0.3),
            "grid_size": 1.0,
            "grid_divisions": 10
        }
        
        # 属性调整状态
        self.property_changes = {}
        self.is_property_animating = False
        self.property_animation_start = None
        self.property_animation_end = None
        self.property_animation_duration = 0.3
        
        # 建模统计信息
        self.modeling_stats = {
            "vertices_count": 0,
            "edges_count": 0,
            "faces_count": 0,
            "selected_vertices": 0,
            "selected_edges": 0,
            "selected_faces": 0,
            "objects_count": 0,
            "materials_count": 0,
            "textures_count": 0
        }
    
    def get_scene_manager(self):
        """获取场景管理器
        
        Returns:
            SceneManager: 场景管理器实例
        """
        return self.scene_manager
    
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
        node = self.scene_manager.create_node(name, position, rotation, scale)
        self.needs_update = True
        self.needs_render = True
        return node
    
    def select_node(self, node):
        """选择单个节点
        
        Args:
            node: 要选择的场景节点
        """
        self.selected_node = node
        self.selected_nodes = [node] if node else []
        self.scene_manager.select_node(node)
        self.needs_update = True
        self.needs_render = True
    
    def add_to_selection(self, node):
        """将节点添加到选择列表
        
        Args:
            node: 要添加的场景节点
        """
        if node not in self.selected_nodes:
            self.selected_nodes.append(node)
            self.selected_node = node
            self.scene_manager.add_to_selection(node)
            self.needs_update = True
            self.needs_render = True
    
    def remove_from_selection(self, node):
        """从选择列表中移除节点
        
        Args:
            node: 要移除的场景节点
        """
        if node in self.selected_nodes:
            self.selected_nodes.remove(node)
            if self.selected_node == node:
                self.selected_node = self.selected_nodes[-1] if self.selected_nodes else None
            self.scene_manager.remove_from_selection(node)
            self.needs_update = True
            self.needs_render = True
    
    def clear_selection(self):
        """清空选择列表"""
        self.selected_node = None
        self.selected_nodes = []
        self.scene_manager.clear_selection()
        self.needs_update = True
        self.needs_render = True
    
    def update_node_property(self, node, property_name, value):
        """更新节点属性
        
        Args:
            node: 要更新的节点
            property_name: 属性名称
            value: 新的属性值
        """
        if hasattr(node, property_name):
            setattr(node, property_name, value)
            node.mark_transform_dirty()
            self.needs_update = True
            self.needs_render = True
            return True
        return False
    
    def update_node_transform(self, node, position=None, rotation=None, scale=None):
        """更新节点变换
        
        Args:
            node: 要更新的节点
            position: 新位置
            rotation: 新旋转
            scale: 新缩放
        """
        if position is not None:
            node.set_position(position)
        if rotation is not None:
            node.set_rotation(rotation)
        if scale is not None:
            node.set_scale(scale)
        self.needs_update = True
        self.needs_render = True
    
    def delete_node(self, node):
        """删除节点
        
        Args:
            node: 要删除的节点
        """
        if node.parent:
            node.parent.remove_child(node)
            if node == self.selected_node:
                self.clear_selection()
            elif node in self.selected_nodes:
                self.remove_from_selection(node)
            self.needs_update = True
            self.needs_render = True
            return True
        return False
    
    def create_mesh_node(self, name, mesh, material, position=None, rotation=None, scale=None):
        """创建带网格和材质的节点
        
        Args:
            name: 节点名称
            mesh: 网格对象
            material: 材质对象
            position: 位置
            rotation: 旋转
            scale: 缩放
            
        Returns:
            SceneNode: 创建的节点
        """
        node = self.create_node(name, position, rotation, scale)
        node.mesh = mesh
        node.material = material
        self.needs_update = True
        self.needs_render = True
        return node
    
    def get_selected_node(self):
        """获取当前选中的单个节点
        
        Returns:
            SceneNode: 当前选中的节点
        """
        return self.selected_node
    
    def get_selected_nodes(self):
        """获取所有选中的节点
        
        Returns:
            list: 选中的节点列表
        """
        return self.selected_nodes
    
    def set_modeling_mode(self, mode):
        """设置建模模式
        
        Args:
            mode: 建模模式 (object, vertex, edge, face)
        """
        self.modeling_mode = mode
        self.needs_update = True
        self.needs_render = True
    
    def set_transform_mode(self, mode):
        """设置变换模式
        
        Args:
            mode: 变换模式 (translate, rotate, scale)
        """
        self.transform_mode = mode
        self.needs_update = True
        self.needs_render = True
    
    def toggle_snap(self):
        """切换捕捉功能"""
        self.snap_enabled = not self.snap_enabled
        self.needs_update = True
    
    def set_snap_distance(self, distance):
        """设置捕捉距离
        
        Args:
            distance: 捕捉距离
        """
        self.snap_distance = distance
        self.needs_update = True
    
    def update(self, delta_time):
        """更新模型
        
        Args:
            delta_time: 帧时间
        """
        if self.needs_update:
            self.scene_manager.update(delta_time)
            self.needs_update = False
    
    def render(self):
        """渲染模型
        
        Returns:
            dict: 渲染统计信息
        """
        if self.needs_render:
            stats = self.scene_manager.render()
            self.needs_render = False
            return stats
        return {}
    
    def get_scene_state(self):
        """获取场景状态
        
        Returns:
            dict: 场景状态信息
        """
        return {
            "selected_node": self.selected_node.name if self.selected_node else None,
            "selected_nodes_count": len(self.selected_nodes),
            "modeling_mode": self.modeling_mode,
            "transform_mode": self.transform_mode,
            "snap_enabled": self.snap_enabled,
            "snap_distance": self.snap_distance,
            "needs_update": self.needs_update,
            "needs_render": self.needs_render
        }
    
    def set_scene_state(self, state):
        """设置场景状态
        
        Args:
            state: 场景状态信息
        """
        if "modeling_mode" in state:
            self.set_modeling_mode(state["modeling_mode"])
        if "transform_mode" in state:
            self.set_transform_mode(state["transform_mode"])
        if "snap_enabled" in state:
            self.snap_enabled = state["snap_enabled"]
        if "snap_distance" in state:
            self.set_snap_distance(state["snap_distance"])
        if "visual_feedback" in state:
            self.visual_feedback.update(state["visual_feedback"])
        self.needs_update = True
        self.needs_render = True
    
    def update_property_animated(self, node, property_name, value, duration=0.3):
        """动画更新节点属性
        
        Args:
            node: 要更新的节点
            property_name: 属性名称
            value: 新的属性值
            duration: 动画持续时间
        """
        if hasattr(node, property_name):
            current_value = getattr(node, property_name)
            self.property_changes[(node, property_name)] = {
                "start_value": current_value,
                "end_value": value,
                "duration": duration,
                "elapsed": 0.0,
                "node": node,
                "property_name": property_name
            }
            self.is_property_animating = True
            self.needs_property_update = True
            self.needs_render = True
            return True
        return False
    
    def update_visual_feedback(self, feedback_type, value):
        """更新视觉反馈设置
        
        Args:
            feedback_type: 视觉反馈类型
            value: 新的视觉反馈值
        """
        if feedback_type in self.visual_feedback:
            self.visual_feedback[feedback_type] = value
            self.needs_render = True
            return True
        return False
    
    def get_visual_feedback(self, feedback_type=None):
        """获取视觉反馈设置
        
        Args:
            feedback_type: 视觉反馈类型，如果为None则返回所有设置
            
        Returns:
            dict or any: 视觉反馈设置
        """
        if feedback_type is None:
            return self.visual_feedback
        return self.visual_feedback.get(feedback_type)
    
    def toggle_visual_feedback(self, feedback_type):
        """切换视觉反馈设置
        
        Args:
            feedback_type: 视觉反馈类型
        """
        if feedback_type in self.visual_feedback and isinstance(self.visual_feedback[feedback_type], bool):
            self.visual_feedback[feedback_type] = not self.visual_feedback[feedback_type]
            self.needs_render = True
            return True
        return False
    
    def update_modeling_stats(self):
        """更新建模统计信息"""
        # 计算场景中的对象数量
        objects_count = 0
        vertices_count = 0
        edges_count = 0
        faces_count = 0
        
        # 遍历所有节点，计算统计信息
        def count_node_stats(node):
            nonlocal objects_count, vertices_count, edges_count, faces_count
            objects_count += 1
            
            if hasattr(node, 'mesh') and node.mesh:
                # 计算网格的顶点、边和面数量
                vertices_count += len(node.mesh.vertices) if hasattr(node.mesh, 'vertices') else 0
                # 简化计算：边数约为顶点数的2倍，面数约为顶点数的1/2
                edges_count += vertices_count * 2
                faces_count += vertices_count // 2
            
            # 递归计算子节点
            for child in node.children:
                count_node_stats(child)
        
        count_node_stats(self.scene_manager.root_node)
        
        # 更新统计信息
        self.modeling_stats["objects_count"] = objects_count
        self.modeling_stats["vertices_count"] = vertices_count
        self.modeling_stats["edges_count"] = edges_count
        self.modeling_stats["faces_count"] = faces_count
        
        # 更新选中对象的统计信息
        self.modeling_stats["selected_vertices"] = 0
        self.modeling_stats["selected_edges"] = 0
        self.modeling_stats["selected_faces"] = 0
        
        # 计算材质和纹理数量
        materials = set()
        textures = set()
        
        def count_resource_stats(node):
            nonlocal materials, textures
            if hasattr(node, 'material') and node.material:
                materials.add(id(node.material))
                if hasattr(node.material, 'textures'):
                    for texture in node.material.textures.values():
                        textures.add(id(texture))
            
            # 递归计算子节点
            for child in node.children:
                count_resource_stats(child)
        
        count_resource_stats(self.scene_manager.root_node)
        
        self.modeling_stats["materials_count"] = len(materials)
        self.modeling_stats["textures_count"] = len(textures)
    
    def get_modeling_stats(self):
        """获取建模统计信息
        
        Returns:
            dict: 建模统计信息
        """
        return self.modeling_stats
    
    def update(self, delta_time):
        """更新模型
        
        Args:
            delta_time: 帧时间
        """
        # 更新场景
        if self.needs_update:
            self.scene_manager.update(delta_time)
            self.needs_update = False
        
        # 更新属性动画
        if self.is_property_animating:
            self._update_property_animations(delta_time)
        
        # 更新建模统计
        self.update_modeling_stats()
    
    def _update_property_animations(self, delta_time):
        """更新属性动画
        
        Args:
            delta_time: 帧时间
        """
        completed_changes = []
        
        for key, change in self.property_changes.items():
            change["elapsed"] += delta_time
            
            if change["elapsed"] >= change["duration"]:
                # 动画完成，设置最终值
                setattr(change["node"], change["property_name"], change["end_value"])
                completed_changes.append(key)
            else:
                # 计算当前值（使用线性插值）
                t = change["elapsed"] / change["duration"]
                if isinstance(change["start_value"], Vector3):
                    # 手动实现Vector3线性插值
                    start = change["start_value"]
                    end = change["end_value"]
                    current_value = Vector3(
                        start.x + (end.x - start.x) * t,
                        start.y + (end.y - start.y) * t,
                        start.z + (end.z - start.z) * t
                    )
                elif isinstance(change["start_value"], Quaternion):
                    # 手动实现Quaternion球面线性插值
                    start = change["start_value"]
                    end = change["end_value"]
                    
                    # 简化实现，使用线性插值
                    current_value = Quaternion(
                        start.x + (end.x - start.x) * t,
                        start.y + (end.y - start.y) * t,
                        start.z + (end.z - start.z) * t,
                        start.w + (end.w - start.w) * t
                    )
                    # 归一化四元数
                    current_value.normalize()
                elif isinstance(change["start_value"], (int, float)):
                    current_value = change["start_value"] + (change["end_value"] - change["start_value"]) * t
                else:
                    # 不支持的类型，直接设置最终值
                    current_value = change["end_value"]
                    completed_changes.append(key)
                
                setattr(change["node"], change["property_name"], current_value)
                change["node"].mark_transform_dirty()
        
        # 移除已完成的属性变化
        for key in completed_changes:
            del self.property_changes[key]
        
        # 检查是否还有属性动画在运行
        if not self.property_changes:
            self.is_property_animating = False
        
        self.needs_render = True
    
    def render(self):
        """渲染模型
        
        Returns:
            dict: 渲染统计信息
        """
        stats = {}
        
        if self.needs_render:
            # 渲染场景
            scene_stats = self.scene_manager.render()
            stats.update(scene_stats)
            
            # 渲染视觉反馈
            self._render_visual_feedback()
            
            self.needs_render = False
        
        return stats
    
    def _render_visual_feedback(self):
        """渲染视觉反馈"""
        # 渲染网格
        if self.visual_feedback["show_grid"]:
            self._render_grid()
        
        # 渲染选中对象的视觉反馈
        if self.selected_node:
            self._render_selection_feedback(self.selected_node)
        
        # 渲染所有选中节点的视觉反馈
        for node in self.selected_nodes:
            self._render_selection_feedback(node)
    
    def _render_grid(self):
        """渲染网格"""
        # 简化实现，实际项目中需要使用OpenGL渲染网格
        pass
    
    def _render_selection_feedback(self, node):
        """渲染选中对象的视觉反馈
        
        Args:
            node: 选中的节点
        """
        # 简化实现，实际项目中需要使用OpenGL渲染选中对象的视觉反馈
        pass
