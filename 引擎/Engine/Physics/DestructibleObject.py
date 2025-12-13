# -*- coding: utf-8 -*-
"""
破坏系统实现
支持物体破碎成多个物理块
"""

from Engine.Math import Vector3, Quaternion, Matrix4x4
from Engine.Scene.SceneNode import SceneNode

class DestructibleObject:
    """可破坏物体类
    表示可以被破坏成多个物理块的物体"""
    
    class DestructionLevel:
        """破坏级别枚举"""
        INTACT = 0      # 完整
        CRACKED = 1     # 开裂
        BROKEN = 2      # 破碎
        SHATTERED = 3   # 粉碎
    
    def __init__(self, scene_node, destruction_level=DestructionLevel.INTACT):
        """
        初始化可破坏物体
        
        Args:
            scene_node: 关联的场景节点
            destruction_level: 初始破坏级别
        """
        self.scene_node = scene_node
        self.destruction_level = destruction_level
        self.max_destruction_level = self.DestructionLevel.SHATTERED
        
        # 物理引擎ID
        self.physics_id = None
        
        # 破坏属性
        self.enabled = True
        self.durability = 100.0  # 耐久性，0表示完全破坏
        self.break_force_threshold = 1000.0  # 破坏所需的力阈值
        self.fracture_pattern = "radial"  # 断裂模式: "radial", "grid", "random"
        
        # 碎片属性
        self.fragment_count = 8  # 破碎后的碎片数量
        self.fragment_mass = 0.5  # 每个碎片的质量
        self.fragment_restitution = 0.3  # 碎片弹性
        self.fragment_friction = 0.8  # 碎片摩擦
        
        # 碎片列表
        self.fragments = []
        self.fragment_bodies = []
        
        # 破坏状态
        self.is_destructed = False
        self.is_breaking = False
        
        # 场景节点关联
        if scene_node:
            scene_node.destructible = self
    
    def apply_damage(self, damage, position=None, force_direction=None):
        """
        应用伤害到物体
        
        Args:
            damage: 伤害值
            position: 伤害位置
            force_direction: 伤害方向
        """
        if not self.enabled or self.is_destructed:
            return
        
        # 减少耐久性
        self.durability = max(0.0, self.durability - damage)
        
        # 检查是否需要破坏
        if self.durability <= 0.0:
            self.destruct()
            return
        
        # 根据耐久性更新破坏级别
        self._update_destruction_level()
    
    def _update_destruction_level(self):
        """根据耐久性更新破坏级别"""
        durability_ratio = self.durability / 100.0
        
        if durability_ratio > 0.75:
            self.destruction_level = self.DestructionLevel.INTACT
        elif durability_ratio > 0.5:
            self.destruction_level = self.DestructionLevel.CRACKED
        elif durability_ratio > 0.25:
            self.destruction_level = self.DestructionLevel.BROKEN
        else:
            self.destruction_level = self.DestructionLevel.SHATTERED
    
    def destruct(self):
        """破坏物体，将其分解为碎片"""
        if self.is_destructed or self.is_breaking:
            return
        
        self.is_breaking = True
        
        # 根据破坏级别确定碎片数量
        if self.destruction_level == self.DestructionLevel.CRACKED:
            self.fragment_count = 2
        elif self.destruction_level == self.DestructionLevel.BROKEN:
            self.fragment_count = 4
        else:  # SHATTERED
            self.fragment_count = 8
        
        # 生成碎片
        self._generate_fragments()
        
        self.is_destructed = True
        self.is_breaking = False
    
    def _generate_fragments(self):
        """生成破坏后的碎片"""
        # 清空现有碎片
        self.fragments.clear()
        self.fragment_bodies.clear()
        
        # 根据断裂模式生成碎片
        if self.fracture_pattern == "radial":
            self._generate_radial_fragments()
        elif self.fracture_pattern == "grid":
            self._generate_grid_fragments()
        else:  # random
            self._generate_random_fragments()
    
    def _generate_radial_fragments(self):
        """生成放射状碎片"""
        # 简化实现，实际需要根据物体形状生成放射状碎片
        for i in range(self.fragment_count):
            # 创建碎片节点
            fragment_node = SceneNode(f"Fragment_{i}")
            fragment_node.position = self.scene_node.position
            fragment_node.rotation = self.scene_node.rotation
            fragment_node.scale = self.scene_node.scale * 0.5
            
            self.fragments.append(fragment_node)
    
    def _generate_grid_fragments(self):
        """生成网格状碎片"""
        # 简化实现，实际需要根据物体形状生成网格状碎片
        for i in range(self.fragment_count):
            # 创建碎片节点
            fragment_node = SceneNode(f"Fragment_{i}")
            fragment_node.position = self.scene_node.position
            fragment_node.rotation = self.scene_node.rotation
            fragment_node.scale = self.scene_node.scale * 0.5
            
            self.fragments.append(fragment_node)
    
    def _generate_random_fragments(self):
        """生成随机形状碎片"""
        # 简化实现，实际需要根据物体形状生成随机碎片
        for i in range(self.fragment_count):
            # 创建碎片节点
            fragment_node = SceneNode(f"Fragment_{i}")
            fragment_node.position = self.scene_node.position
            fragment_node.rotation = self.scene_node.rotation
            fragment_node.scale = self.scene_node.scale * 0.5
            
            self.fragments.append(fragment_node)
    
    def add_fragment_body(self, rigid_body):
        """
        添加碎片刚体
        
        Args:
            rigid_body: 碎片的刚体对象
        """
        self.fragment_bodies.append(rigid_body)
    
    def update(self, delta_time):
        """
        更新可破坏物体
        
        Args:
            delta_time: 帧时间
        """
        if not self.enabled:
            return
        
        # 更新碎片状态
        for fragment, body in zip(self.fragments, self.fragment_bodies):
            if body and body.enabled:
                # 简化实现，实际需要同步碎片和刚体的变换
                pass
    
    def is_intact(self):
        """检查物体是否完整"""
        return self.destruction_level == self.DestructionLevel.INTACT
    
    def is_completely_destructed(self):
        """检查物体是否完全破坏"""
        return self.is_destructed
    
    def set_durability(self, durability):
        """
        设置物体耐久性
        
        Args:
            durability: 新的耐久性值
        """
        self.durability = max(0.0, min(100.0, durability))
        self._update_destruction_level()
    
    def set_fracture_pattern(self, pattern):
        """
        设置断裂模式
        
        Args:
            pattern: 断裂模式 ("radial", "grid", "random")
        """
        if pattern in ["radial", "grid", "random"]:
            self.fracture_pattern = pattern
    
    def set_fragment_count(self, count):
        """
        设置碎片数量
        
        Args:
            count: 碎片数量
        """
        self.fragment_count = max(2, min(32, count))