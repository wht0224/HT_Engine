# -*- coding: utf-8 -*-
"""
车辆物理系统实现
支持基础的车辆动力学
"""

import math
from Engine.Math import Vector3, Quaternion, Matrix4x4
from Engine.Scene.SceneNode import SceneNode

class Vehicle:
    """车辆类
    表示物理世界中的一个车辆对象"""
    
    def __init__(self, scene_node):
        """
        初始化车辆
        
        Args:
            scene_node: 关联的场景节点
        """
        self.scene_node = scene_node
        
        # 物理引擎ID
        self.physics_id = None
        
        # 车辆属性
        self.enabled = True
        self.mass = 1500.0  # 车辆质量(kg)
        self.max_speed = 30.0  # 最大速度(m/s)
        self.max_acceleration = 5.0  # 最大加速度(m/s²)
        self.max_deceleration = 8.0  # 最大减速度(m/s²)
        self.steering_angle = 0.0  # 当前转向角度
        self.max_steering_angle = 30.0  # 最大转向角度(度)
        self.steering_rate = 100.0  # 转向速率(度/秒)
        
        # 车辆状态
        self.velocity = Vector3(0, 0, 0)
        self.angular_velocity = Vector3(0, 0, 0)
        self.acceleration = Vector3(0, 0, 0)
        self.brake_force = 0.0
        self.throttle = 0.0
        
        # 车轮属性
        self.wheel_count = 4
        self.wheel_radius = 0.3
        self.wheel_width = 0.2
        self.suspension_length = 0.2
        self.suspension_stiffness = 20000.0
        self.suspension_damping = 2000.0
        self.suspension_compression = 4000.0
        
        # 车轮位置 (相对于车辆中心)
        self.wheel_positions = [
            Vector3(-1.5, 0, 1.0),   # 左前轮
            Vector3(1.5, 0, 1.0),    # 右前轮
            Vector3(-1.5, 0, -1.0),  # 左后轮
            Vector3(1.5, 0, -1.0)    # 右后轮
        ]
        
        # 车轮旋转角度
        self.wheel_rotations = [0.0, 0.0, 0.0, 0.0]
        
        # 场景节点关联
        if scene_node:
            scene_node.vehicle = self
    
    def set_throttle(self, throttle):
        """
        设置油门
        
        Args:
            throttle: 油门值 (0.0 - 1.0)
        """
        self.throttle = max(0.0, min(1.0, throttle))
    
    def set_brake(self, brake_force):
        """
        设置刹车
        
        Args:
            brake_force: 刹车力 (0.0 - 1.0)
        """
        self.brake_force = max(0.0, min(1.0, brake_force))
    
    def set_steering(self, steering_input):
        """
        设置转向
        
        Args:
            steering_input: 转向输入 (-1.0 到 1.0)
        """
        target_angle = steering_input * self.max_steering_angle
        
        # 平滑转向
        angle_diff = target_angle - self.steering_angle
        max_angle_change = self.steering_rate * 0.016  # 假设60fps
        self.steering_angle += max(-max_angle_change, min(max_angle_change, angle_diff))
        self.steering_angle = max(-self.max_steering_angle, min(self.max_steering_angle, self.steering_angle))
    
    def update(self, delta_time):
        """
        更新车辆物理
        
        Args:
            delta_time: 帧时间
        """
        if not self.enabled:
            return
        
        # 计算加速度
        if self.throttle > 0:
            # 应用油门
            forward_force = self.max_acceleration * self.throttle
            self.acceleration = self._get_forward_direction() * forward_force
        elif self.brake_force > 0:
            # 应用刹车
            braking_force = -self.max_deceleration * self.brake_force
            self.acceleration = self._get_forward_direction() * braking_force
        else:
            # 自然减速 (空气阻力和滚动阻力)
            drag_force = Vector3(0, 0, 0) - self.velocity * 0.1
            self.acceleration = drag_force
        
        # 更新速度
        self.velocity += self.acceleration * delta_time
        
        # 限制最大速度
        speed = self.velocity.length()
        if speed > self.max_speed:
            self.velocity = self.velocity.normalized() * self.max_speed
        
        # 更新位置
        self.scene_node.position += self.velocity * delta_time
        
        # 更新转向
        if abs(self.steering_angle) > 0 and speed > 0.1:
            # 计算转向半径
            wheel_base = 3.0  # 轴距
            turn_radius = wheel_base / abs(math.sin(math.radians(self.steering_angle)))
            
            # 计算旋转角速度
            angular_speed = speed / turn_radius
            
            # 计算旋转方向
            rotation_direction = 1.0 if self.steering_angle > 0 else -1.0
            
            # 更新车辆旋转
            yaw_rotation = angular_speed * delta_time * rotation_direction
            self.scene_node.rotation.y += yaw_rotation
        
        # 更新车轮旋转
        for i in range(self.wheel_count):
            # 计算车轮旋转角度
            wheel_speed = speed / self.wheel_radius
            self.wheel_rotations[i] += wheel_speed * delta_time
    
    def _get_forward_direction(self):
        """
        获取车辆前进方向
        
        Returns:
            Vector3: 前进方向向量
        """
        # 根据车辆旋转计算前进方向
        yaw = math.radians(self.scene_node.rotation.y)
        forward = Vector3(math.sin(yaw), 0, math.cos(yaw))
        return forward
    
    def get_speed(self):
        """
        获取车辆当前速度
        
        Returns:
            float: 当前速度 (m/s)
        """
        return self.velocity.length()
    
    def get_speed_kmh(self):
        """
        获取车辆当前速度 (km/h)
        
        Returns:
            float: 当前速度 (km/h)
        """
        return self.get_speed() * 3.6
    
    def reset(self):
        """
        重置车辆状态
        """
        self.velocity = Vector3(0, 0, 0)
        self.angular_velocity = Vector3(0, 0, 0)
        self.acceleration = Vector3(0, 0, 0)
        self.steering_angle = 0.0
        self.throttle = 0.0
        self.brake_force = 0.0
        self.wheel_rotations = [0.0, 0.0, 0.0, 0.0]