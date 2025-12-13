# -*- coding: utf-8 -*-
"""
流体物理系统实现
基于粒子的简化流体模拟，适合低端GPU
"""

from Engine.Math import Vector3, Quaternion, Matrix4x4
from Engine.Scene.SceneNode import SceneNode
import random

class Fluid:
    """流体类
    表示物理世界中的一个流体对象"""
    
    def __init__(self, scene_node, particle_count=100, fluid_type="water"):
        """
        初始化流体
        
        Args:
            scene_node: 关联的场景节点
            particle_count: 粒子数量
            fluid_type: 流体类型 ("water", "oil", "lava", "smoke")
        """
        self.scene_node = scene_node
        self.particle_count = particle_count
        self.fluid_type = fluid_type
        
        # 物理引擎ID
        self.physics_id = None
        
        # 流体属性
        self.enabled = True
        self.use_gravity = True
        self.density = 1000.0  # 水的密度
        self.viscosity = 0.001  # 流体粘度
        self.surface_tension = 0.0728  # 表面张力
        self.damping = 0.99
        
        # 粒子属性
        self.particle_radius = 0.1
        self.particle_mass = 0.1
        
        # 碰撞属性
        self.collision_enabled = True
        self.collision_group = 1
        self.collision_mask = -1
        
        # 粒子数据
        self.particles = []
        self.velocities = []
        self.accelerations = []
        self.forces = []
        
        # 邻居列表
        self.neighbors = []
        
        # 初始化粒子
        self._initialize_particles()
        
        # 场景节点关联
        if scene_node:
            scene_node.fluid = self
    
    def _initialize_particles(self):
        """初始化流体粒子"""
        # 根据流体类型设置属性
        if self.fluid_type == "water":
            self.density = 1000.0
            self.viscosity = 0.001
            self.surface_tension = 0.0728
        elif self.fluid_type == "oil":
            self.density = 920.0
            self.viscosity = 0.05
            self.surface_tension = 0.028
        elif self.fluid_type == "lava":
            self.density = 3000.0
            self.viscosity = 100.0
            self.surface_tension = 0.08
        elif self.fluid_type == "smoke":
            self.density = 1.0
            self.viscosity = 0.0001
            self.surface_tension = 0.0
            self.use_gravity = False
        
        # 生成粒子
        for i in range(self.particle_count):
            # 随机初始位置
            x = random.uniform(-1.0, 1.0)
            y = random.uniform(2.0, 4.0)  # 在高处生成
            z = random.uniform(-1.0, 1.0)
            
            self.particles.append(Vector3(x, y, z))
            self.velocities.append(Vector3(0, 0, 0))
            self.accelerations.append(Vector3(0, 0, 0))
            self.forces.append(Vector3(0, 0, 0))
            self.neighbors.append([])
    
    def _calculate_neighbors(self):
        """计算粒子邻居"""
        # 重置邻居列表
        for i in range(self.particle_count):
            self.neighbors[i].clear()
        
        # 计算邻居（简化实现，O(n^2)复杂度，实际应该使用空间划分算法）
        for i in range(self.particle_count):
            for j in range(i + 1, self.particle_count):
                # 计算粒子间距离
                distance = (self.particles[i] - self.particles[j]).length()
                
                # 如果距离小于粒子半径的2倍，视为邻居
                if distance < self.particle_radius * 2:
                    self.neighbors[i].append(j)
                    self.neighbors[j].append(i)
    
    def _calculate_forces(self, gravity):
        """计算粒子受力"""
        # 重置所有力
        for i in range(self.particle_count):
            self.forces[i] = Vector3(0, 0, 0)
        
        # 应用重力
        if self.use_gravity:
            for i in range(self.particle_count):
                self.forces[i] += gravity * self.particle_mass
        
        # 计算压力和粘性力（简化实现）
        for i in range(self.particle_count):
            for j in self.neighbors[i]:
                # 计算粒子间距离
                distance = (self.particles[i] - self.particles[j]).length()
                direction = (self.particles[j] - self.particles[i]).normalized()
                
                # 计算压力（简化的弹簧力）
                pressure_magnitude = float(self.particle_radius * 2 - distance) * 100.0
                pressure_force = direction * pressure_magnitude
                self.forces[i] += pressure_force
                
                # 计算粘性力
                velocity_diff = self.velocities[j] - self.velocities[i]
                viscosity_force = velocity_diff * float(self.viscosity)
                self.forces[i] += viscosity_force
    
    def _integrate(self, delta_time):
        """积分更新粒子位置和速度"""
        for i in range(self.particle_count):
            # 计算加速度
            self.accelerations[i] = self.forces[i] * (1.0 / self.particle_mass)
            
            # 更新速度
            self.velocities[i] += self.accelerations[i] * delta_time
            self.velocities[i] *= self.damping
            
            # 更新位置
            self.particles[i] += self.velocities[i] * delta_time
            
            # 地面碰撞检测
            if self.particles[i].y < self.particle_radius:
                self.particles[i].y = self.particle_radius
                self.velocities[i].y *= -0.5  # 反弹
    
    def update(self, delta_time, gravity):
        """更新流体模拟
        
        Args:
            delta_time: 帧时间
            gravity: 重力向量
        """
        if not self.enabled:
            return
        
        # 计算邻居
        self._calculate_neighbors()
        
        # 计算力
        self._calculate_forces(gravity)
        
        # 积分更新
        self._integrate(delta_time)
    
    def set_particle_count(self, count):
        """设置粒子数量
        
        Args:
            count: 新的粒子数量
        """
        self.particle_count = count
        self._initialize_particles()
    
    def set_fluid_type(self, fluid_type):
        """设置流体类型
        
        Args:
            fluid_type: 流体类型
        """
        self.fluid_type = fluid_type
        # 重新初始化粒子以应用新属性
        self._initialize_particles()
    
    def enable_collision(self, enable):
        """启用或禁用碰撞
        
        Args:
            enable: 是否启用碰撞
        """
        self.collision_enabled = enable