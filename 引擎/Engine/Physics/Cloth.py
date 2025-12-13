# -*- coding: utf-8 -*-
"""
布料物理系统实现
基于Bullet物理引擎的布料模拟
"""

from Engine.Math import Vector3, Quaternion, Matrix4x4
from Engine.Scene.SceneNode import SceneNode

class Cloth:
    """布料类
    表示物理世界中的一个布料对象"""
    
    def __init__(self, scene_node, width=1.0, height=1.0, segments_x=10, segments_y=10, mass=0.1):
        """
        初始化布料
        
        Args:
            scene_node: 关联的场景节点
            width: 布料宽度
            height: 布料高度
            segments_x: X方向的段数
            segments_y: Y方向的段数
            mass: 布料总质量
        """
        self.scene_node = scene_node
        self.width = width
        self.height = height
        self.segments_x = segments_x
        self.segments_y = segments_y
        self.mass = mass
        
        # 物理引擎ID
        self.physics_id = None
        
        # 布料属性
        self.enabled = True
        self.use_gravity = True
        self.damping = 0.01
        self.stiffness = 0.9
        self.friction = 0.5
        self.restitution = 0.0
        
        # 碰撞属性
        self.collision_enabled = True
        self.collision_group = 1
        self.collision_mask = -1
        
        # 顶点数据
        self.vertices = []
        self.normals = []
        self.indices = []
        
        # 布料约束
        self.pin_constraints = []
        
        # 初始化布料网格
        self._initialize_cloth_mesh()
        
        # 场景节点关联
        if scene_node:
            scene_node.cloth = self
    
    def _initialize_cloth_mesh(self):
        """初始化布料网格"""
        # 生成布料顶点
        for y in range(self.segments_y + 1):
            for x in range(self.segments_x + 1):
                # 计算顶点位置
                px = (x / self.segments_x - 0.5) * self.width
                py = 0.0
                pz = (y / self.segments_y - 0.5) * self.height
                self.vertices.append(Vector3(px, py, pz))
                self.normals.append(Vector3(0, 1, 0))
        
        # 生成布料三角形索引
        for y in range(self.segments_y):
            for x in range(self.segments_x):
                # 计算当前顶点索引
                current = y * (self.segments_x + 1) + x
                
                # 添加第一个三角形
                self.indices.append(current)
                self.indices.append(current + self.segments_x + 1)
                self.indices.append(current + 1)
                
                # 添加第二个三角形
                self.indices.append(current + 1)
                self.indices.append(current + self.segments_x + 1)
                self.indices.append(current + self.segments_x + 2)
    
    def add_pin_constraint(self, vertex_index, position=None):
        """
        添加布料固定约束
        
        Args:
            vertex_index: 要固定的顶点索引
            position: 固定位置，如果为None则使用当前顶点位置
        """
        if position is None:
            position = self.vertices[vertex_index]
        
        self.pin_constraints.append({
            "vertex_index": vertex_index,
            "position": position
        })
    
    def set_stiffness(self, stiffness):
        """
        设置布料刚度
        
        Args:
            stiffness: 刚度值，范围[0, 1]
        """
        self.stiffness = max(0.0, min(1.0, stiffness))
    
    def set_damping(self, damping):
        """
        设置布料阻尼
        
        Args:
            damping: 阻尼值，范围[0, 1]
        """
        self.damping = max(0.0, min(1.0, damping))
    
    def set_friction(self, friction):
        """
        设置布料摩擦系数
        
        Args:
            friction: 摩擦系数，范围[0, 1]
        """
        self.friction = max(0.0, min(1.0, friction))
    
    def enable_collision(self, enable):
        """
        启用或禁用碰撞
        
        Args:
            enable: 是否启用碰撞
        """
        self.collision_enabled = enable
    
    def update_mesh(self):
        """
        更新场景节点的网格数据
        """
        # 简化实现，实际需要更新场景节点的顶点和法线数据
        pass