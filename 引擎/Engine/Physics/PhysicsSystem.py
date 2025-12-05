# -*- coding: utf-8 -*-
"""
物理系统模块
集成Bullet物理引擎，实现基础刚体物理
"""

import time
from Engine.Math.Math import Vector3, Quaternion, Matrix4x4
from Engine.Scene.SceneNode import SceneNode

class PhysicsSystem:
    """物理系统类
    集成Bullet物理引擎，管理物理世界、刚体、约束、布料、流体、可破坏物体和车辆
    """
    
    def __init__(self, engine):
        """初始化物理系统
        
        Args:
            engine: 引擎实例
        """
        self.engine = engine
        self.physics_world = None
        self.rigid_bodies = []
        self.constraints = []
        self.cloths = []
        self.fluids = []
        self.destructible_objects = []
        self.vehicles = []
        
        # 物理设置
        self.gravity = Vector3(0, -9.8, 0)
        self.fixed_time_step = 1.0 / 60.0
        self.max_sub_steps = 4
        
        # 物理引擎类型
        self.physics_engine = "bullet"  # "bullet" or "builtin"
        
        # 性能统计
        self.physics_stats = {
            "collisions": 0,
            "rigid_bodies": 0,
            "constraints": 0,
            "cloths": 0,
            "fluids": 0,
            "destructible_objects": 0,
            "vehicles": 0,
            "simulation_time": 0.0
        }
        
        # 初始化物理引擎
        self._initialize_physics_engine()
    
    def _initialize_physics_engine(self):
        """初始化物理引擎"""
        try:
            # 尝试导入pybullet
            import pybullet as p
            import pybullet_data
            
            # 创建物理世界
            self.physics_world = p.connect(p.DIRECT)  # 直接模式，不显示GUI
            
            # 设置重力
            p.setGravity(self.gravity.x, self.gravity.y, self.gravity.z)
            
            # 设置物理引擎参数
            p.setPhysicsEngineParameter(fixedTimeStep=self.fixed_time_step)
            p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
            
            # 设置碰撞检测算法
            p.setPhysicsEngineParameter(enableConeFriction=1)
            p.setPhysicsEngineParameter(enableFileCaching=0)
            
            self.engine.logger.info("Bullet物理引擎初始化成功")
        except ImportError as e:
            self.engine.logger.warning(f"无法导入pybullet，使用内置物理引擎: {e}")
            self.physics_engine = "builtin"
            self._initialize_builtin_physics()
    
    def _initialize_builtin_physics(self):
        """初始化内置物理引擎（简化版）"""
        self.engine.logger.info("使用内置物理引擎")
        # 内置物理引擎的简化实现
        pass
    
    def update(self, delta_time):
        """更新物理系统
        
        Args:
            delta_time: 帧时间
        """
        start_time = time.time()
        
        if self.physics_engine == "bullet" and self.physics_world is not None:
            import pybullet as p
            
            # 步进物理模拟
            p.stepSimulation()
            
            # 更新刚体位置和旋转
            for rigid_body in self.rigid_bodies:
                if rigid_body.enabled:
                    self._update_rigid_body_transform(rigid_body)
            
            # 更新布料（简化实现）
            for cloth in self.cloths:
                if cloth.enabled:
                    self._update_cloth_transform(cloth)
        elif self.physics_engine == "builtin":
            # 内置物理引擎的简化更新
            self._update_builtin_physics(delta_time)
            
            # 更新布料（内置实现）
            for cloth in self.cloths:
                if cloth.enabled:
                    self._update_builtin_cloth(cloth, delta_time)
            
            # 更新流体（内置实现）
            for fluid in self.fluids:
                if fluid.enabled:
                    fluid.update(delta_time, self.gravity)
        
        # 更新可破坏物体
        for destructible in self.destructible_objects:
            if destructible.enabled:
                destructible.update(delta_time)
        
        # 更新车辆
        for vehicle in self.vehicles:
            if vehicle.enabled:
                vehicle.update(delta_time)
        
        # 更新统计信息
        self.physics_stats["rigid_bodies"] = len(self.rigid_bodies)
        self.physics_stats["constraints"] = len(self.constraints)
        self.physics_stats["cloths"] = len(self.cloths)
        self.physics_stats["fluids"] = len(self.fluids)
        self.physics_stats["destructible_objects"] = len(self.destructible_objects)
        self.physics_stats["vehicles"] = len(self.vehicles)
        self.physics_stats["simulation_time"] = time.time() - start_time
    
    def add_vehicle(self, scene_node):
        """添加车辆到物理世界
        
        Args:
            scene_node: 场景节点
            
        Returns:
            Vehicle: 车辆对象
        """
        from .Vehicle import Vehicle
        vehicle = Vehicle(scene_node)
        
        # 添加到车辆列表
        self.vehicles.append(vehicle)
        
        return vehicle
    
    def remove_vehicle(self, vehicle):
        """从物理世界中移除车辆
        
        Args:
            vehicle: 车辆对象
        """
        if vehicle in self.vehicles:
            # 从列表中移除
            self.vehicles.remove(vehicle)
    
    def set_vehicle_throttle(self, vehicle, throttle):
        """设置车辆油门
        
        Args:
            vehicle: 车辆对象
            throttle: 油门值 (0.0 - 1.0)
        """
        if vehicle in self.vehicles:
            vehicle.set_throttle(throttle)
    
    def set_vehicle_brake(self, vehicle, brake_force):
        """设置车辆刹车
        
        Args:
            vehicle: 车辆对象
            brake_force: 刹车力 (0.0 - 1.0)
        """
        if vehicle in self.vehicles:
            vehicle.set_brake(brake_force)
    
    def set_vehicle_steering(self, vehicle, steering_input):
        """设置车辆转向
        
        Args:
            vehicle: 车辆对象
            steering_input: 转向输入 (-1.0 到 1.0)
        """
        if vehicle in self.vehicles:
            vehicle.set_steering(steering_input)
    
    def add_destructible_object(self, scene_node, destruction_level=0):
        """添加可破坏物体到物理世界
        
        Args:
            scene_node: 场景节点
            destruction_level: 初始破坏级别
            
        Returns:
            DestructibleObject: 可破坏物体对象
        """
        from .DestructibleObject import DestructibleObject
        destructible = DestructibleObject(scene_node, destruction_level)
        
        # 添加到可破坏物体列表
        self.destructible_objects.append(destructible)
        
        return destructible
    
    def remove_destructible_object(self, destructible):
        """从物理世界中移除可破坏物体
        
        Args:
            destructible: 可破坏物体对象
        """
        if destructible in self.destructible_objects:
            # 移除所有碎片刚体
            for fragment_body in destructible.fragment_bodies:
                if fragment_body in self.rigid_bodies:
                    self.remove_rigid_body(fragment_body)
            
            # 从列表中移除
            self.destructible_objects.remove(destructible)
    
    def apply_damage_to_object(self, scene_node, damage, position=None, force_direction=None):
        """对场景节点应用伤害
        
        Args:
            scene_node: 场景节点
            damage: 伤害值
            position: 伤害位置
            force_direction: 伤害方向
        """
        # 检查场景节点是否有可破坏组件
        if hasattr(scene_node, 'destructible') and scene_node.destructible:
            scene_node.destructible.apply_damage(damage, position, force_direction)
            
            # 如果物体被破坏，创建碎片刚体
            if scene_node.destructible.is_completely_destructed():
                self._create_fragment_bodies(scene_node.destructible)
    
    def _create_fragment_bodies(self, destructible):
        """为可破坏物体创建碎片刚体
        
        Args:
            destructible: 可破坏物体对象
        """
        for fragment in destructible.fragments:
            # 创建碎片刚体
            fragment_body = self.add_rigid_body(
                fragment,
                mass=destructible.fragment_mass,
                shape_type="box",
                dimensions=fragment.scale
            )
            
            if fragment_body:
                # 设置碎片属性
                fragment_body.restitution = destructible.fragment_restitution
                fragment_body.friction = destructible.fragment_friction
                
                # 将刚体添加到碎片列表
                destructible.add_fragment_body(fragment_body)
    
    def add_cloth(self, scene_node, width=1.0, height=1.0, segments_x=10, segments_y=10, mass=0.1):
        """添加布料到物理世界
        
        Args:
            scene_node: 场景节点
            width: 布料宽度
            height: 布料高度
            segments_x: X方向的段数
            segments_y: Y方向的段数
            mass: 布料总质量
            
        Returns:
            Cloth: 布料对象
        """
        from .Cloth import Cloth
        cloth = Cloth(scene_node, width, height, segments_x, segments_y, mass)
        
        if self.physics_engine == "bullet" and self.physics_world is not None:
            self._add_bullet_cloth(cloth)
        
        self.cloths.append(cloth)
        return cloth
    
    def _add_bullet_cloth(self, cloth):
        """添加布料到Bullet物理世界
        
        Args:
            cloth: 布料对象
        """
        import pybullet as p
        
        # 简化实现，实际需要创建布料网格并添加到Bullet
        pass
    
    def remove_cloth(self, cloth):
        """从物理世界中移除布料
        
        Args:
            cloth: 布料对象
        """
        if cloth in self.cloths:
            if self.physics_engine == "bullet" and cloth.physics_id is not None:
                import pybullet as p
                p.removeBody(cloth.physics_id)
            
            self.cloths.remove(cloth)
    
    def _update_cloth_transform(self, cloth):
        """更新布料的变换
        
        Args:
            cloth: 布料对象
        """
        # 简化实现，实际需要从Bullet获取布料顶点位置并更新
        pass
    
    def _update_builtin_cloth(self, cloth, delta_time):
        """更新内置物理引擎的布料
        
        Args:
            cloth: 布料对象
            delta_time: 帧时间
        """
        # 简化的内置布料模拟
        pass
    
    def add_fluid(self, scene_node, particle_count=100, fluid_type="water"):
        """添加流体到物理世界
        
        Args:
            scene_node: 场景节点
            particle_count: 粒子数量
            fluid_type: 流体类型
            
        Returns:
            Fluid: 流体对象
        """
        from .Fluid import Fluid
        fluid = Fluid(scene_node, particle_count, fluid_type)
        
        if self.physics_engine == "bullet" and self.physics_world is not None:
            self._add_bullet_fluid(fluid)
        
        self.fluids.append(fluid)
        return fluid
    
    def _add_bullet_fluid(self, fluid):
        """添加流体到Bullet物理世界
        
        Args:
            fluid: 流体对象
        """
        import pybullet as p
        
        # 简化实现，实际需要创建流体粒子并添加到Bullet
        pass
    
    def remove_fluid(self, fluid):
        """从物理世界中移除流体
        
        Args:
            fluid: 流体对象
        """
        if fluid in self.fluids:
            if self.physics_engine == "bullet" and fluid.physics_id is not None:
                import pybullet as p
                p.removeBody(fluid.physics_id)
            
            self.fluids.remove(fluid)
    
    def _update_rigid_body_transform(self, rigid_body):
        """更新刚体的变换
        
        Args:
            rigid_body: 刚体对象
        """
        import pybullet as p
        
        if rigid_body.physics_id is not None:
            # 获取刚体的位置和旋转
            position, orientation = p.getBasePositionAndOrientation(rigid_body.physics_id)
            
            # 转换为引擎的坐标系统
            # Bullet使用Y轴向上，引擎可能使用Z轴向上，需要转换
            # 注意：这里假设引擎使用Y轴向上，与Bullet一致
            # 如果引擎使用Z轴向上，需要进行坐标转换
            
            # 更新场景节点的变换
            scene_node = rigid_body.scene_node
            if scene_node:
                scene_node.set_position(Vector3(*position))
                scene_node.set_rotation(Quaternion(*orientation))
    
    def _update_builtin_physics(self, delta_time):
        """更新内置物理引擎
        
        Args:
            delta_time: 帧时间
        """
        # 简化的内置物理实现
        # 只处理重力和简单的碰撞检测
        for rigid_body in self.rigid_bodies:
            if rigid_body.enabled and rigid_body.use_gravity:
                # 应用重力
                rigid_body.velocity += self.gravity * delta_time
                
                # 更新位置
                rigid_body.scene_node.set_position(
                    rigid_body.scene_node.get_position() + rigid_body.velocity * delta_time
                )
    
    def add_rigid_body(self, scene_node, mass=1.0, shape_type="box", dimensions=None):
        """添加刚体到物理世界
        
        Args:
            scene_node: 场景节点
            mass: 质量
            shape_type: 碰撞形状类型 ("box", "sphere", "cylinder")
            dimensions: 碰撞形状尺寸
            
        Returns:
            RigidBody: 刚体对象
        """
        rigid_body = RigidBody(scene_node, mass, shape_type, dimensions)
        
        if self.physics_engine == "bullet" and self.physics_world is not None:
            self._add_bullet_rigid_body(rigid_body)
        
        self.rigid_bodies.append(rigid_body)
        return rigid_body
    
    def _add_bullet_rigid_body(self, rigid_body):
        """添加刚体到Bullet物理世界
        
        Args:
            rigid_body: 刚体对象
        """
        import pybullet as p
        
        # 获取场景节点的位置和旋转
        position = rigid_body.scene_node.get_position()
        rotation = rigid_body.scene_node.get_rotation()
        
        # 创建碰撞形状
        if rigid_body.shape_type == "box":
            shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=rigid_body.dimensions * 0.5)
        elif rigid_body.shape_type == "sphere":
            shape_id = p.createCollisionShape(p.GEOM_SPHERE, radius=rigid_body.dimensions.x)
        elif rigid_body.shape_type == "cylinder":
            shape_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=rigid_body.dimensions.x, height=rigid_body.dimensions.y)
        else:
            shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=Vector3(0.5, 0.5, 0.5))
        
        # 创建刚体
        rigid_body.physics_id = p.createMultiBody(
            baseMass=rigid_body.mass,
            baseCollisionShapeIndex=shape_id,
            basePosition=[position.x, position.y, position.z],
            baseOrientation=[rotation.x, rotation.y, rotation.z, rotation.w]
        )
        
        # 设置刚体属性
        if rigid_body.mass > 0:
            # 动态刚体
            p.changeDynamics(rigid_body.physics_id, -1, linearDamping=0.05, angularDamping=0.05)
        else:
            # 静态刚体
            p.changeDynamics(rigid_body.physics_id, -1, mass=0)
    
    def remove_rigid_body(self, rigid_body):
        """从物理世界中移除刚体
        
        Args:
            rigid_body: 刚体对象
        """
        if rigid_body in self.rigid_bodies:
            if self.physics_engine == "bullet" and rigid_body.physics_id is not None:
                import pybullet as p
                p.removeBody(rigid_body.physics_id)
            
            self.rigid_bodies.remove(rigid_body)
    
    def add_constraint(self, constraint):
        """添加约束到物理世界
        
        Args:
            constraint: 约束对象
        """
        self.constraints.append(constraint)
        
        if self.physics_engine == "bullet" and self.physics_world is not None:
            self._add_bullet_constraint(constraint)
    
    def _add_bullet_constraint(self, constraint):
        """添加约束到Bullet物理世界
        
        Args:
            constraint: 约束对象
        """
        import pybullet as p
        
        # 简化实现，实际需要根据约束类型创建不同的约束
        pass
    
    def remove_constraint(self, constraint):
        """从物理世界中移除约束
        
        Args:
            constraint: 约束对象
        """
        if constraint in self.constraints:
            if self.physics_engine == "bullet" and constraint.physics_id is not None:
                import pybullet as p
                p.removeConstraint(constraint.physics_id)
            
            self.constraints.remove(constraint)
    
    def set_gravity(self, gravity):
        """设置重力
        
        Args:
            gravity: 重力向量
        """
        self.gravity = gravity
        
        if self.physics_engine == "bullet" and self.physics_world is not None:
            import pybullet as p
            p.setGravity(gravity.x, gravity.y, gravity.z)
    
    def get_physics_stats(self):
        """获取物理统计信息
        
        Returns:
            dict: 物理统计信息
        """
        return self.physics_stats
    
    def shutdown(self):
        """关闭物理系统"""
        if self.physics_engine == "bullet" and self.physics_world is not None:
            import pybullet as p
            p.disconnect(self.physics_world)
        
        # 清理物理对象
        self.rigid_bodies.clear()
        self.constraints.clear()
        self.cloths.clear()
        self.fluids.clear()
        self.destructible_objects.clear()
        self.vehicles.clear()


class RigidBody:
    """刚体类
    表示物理世界中的一个刚体
    """
    
    def __init__(self, scene_node, mass=1.0, shape_type="box", dimensions=None):
        """初始化刚体
        
        Args:
            scene_node: 关联的场景节点
            mass: 质量
            shape_type: 碰撞形状类型
            dimensions: 碰撞形状尺寸
        """
        self.scene_node = scene_node
        self.mass = mass
        self.shape_type = shape_type
        
        # 设置默认尺寸
        if dimensions is None:
            # 从场景节点的网格获取尺寸
            if hasattr(scene_node, "mesh") and scene_node.mesh:
                # 简化实现，假设网格有bounds属性
                self.dimensions = Vector3(1.0, 1.0, 1.0)  # 临时值
            else:
                self.dimensions = Vector3(1.0, 1.0, 1.0)
        else:
            self.dimensions = dimensions
        
        # 物理引擎ID
        self.physics_id = None
        
        # 刚体属性
        self.enabled = True
        self.use_gravity = True
        self.velocity = Vector3(0, 0, 0)
        self.angular_velocity = Vector3(0, 0, 0)
        self.friction = 0.5
        self.restitution = 0.3
        
        # 碰撞属性
        self.collision_enabled = True
        self.collision_group = 1
        self.collision_mask = -1
        
        # 场景节点关联
        if scene_node:
            scene_node.rigid_body = self
    
    def apply_force(self, force, position=None):
        """施加力到刚体
        
        Args:
            force: 力向量
            position: 施加力的位置（相对于刚体原点）
        """
        if self.physics_engine == "bullet" and self.physics_id is not None:
            import pybullet as p
            
            if position is None:
                p.applyCentralForce(self.physics_id, force.x, force.y, force.z)
            else:
                p.applyForce(self.physics_id, force.x, force.y, force.z, position.x, position.y, position.z)
        else:
            # 内置物理引擎的简化实现
            self.velocity += force * (1.0 / self.mass)
    
    def apply_impulse(self, impulse, position=None):
        """施加冲量到刚体
        
        Args:
            impulse: 冲量向量
            position: 施加冲量的位置（相对于刚体原点）
        """
        if self.physics_engine == "bullet" and self.physics_id is not None:
            import pybullet as p
            
            if position is None:
                p.applyCentralImpulse(self.physics_id, impulse.x, impulse.y, impulse.z)
            else:
                p.applyImpulse(self.physics_id, impulse.x, impulse.y, impulse.z, position.x, position.y, position.z)
        else:
            # 内置物理引擎的简化实现
            self.velocity += impulse * (1.0 / self.mass)
    
    def set_velocity(self, velocity):
        """设置刚体的线速度
        
        Args:
            velocity: 线速度向量
        """
        self.velocity = velocity
        
        if self.physics_engine == "bullet" and self.physics_id is not None:
            import pybullet as p
            p.resetBaseVelocity(self.physics_id, [velocity.x, velocity.y, velocity.z], [0, 0, 0])
    
    def set_angular_velocity(self, angular_velocity):
        """设置刚体的角速度
        
        Args:
            angular_velocity: 角速度向量
        """
        self.angular_velocity = angular_velocity
        
        if self.physics_engine == "bullet" and self.physics_id is not None:
            import pybullet as p
            p.resetBaseVelocity(self.physics_id, [0, 0, 0], [angular_velocity.x, angular_velocity.y, angular_velocity.z])
    
    def enable_collision(self, enable):
        """启用或禁用碰撞
        
        Args:
            enable: 是否启用碰撞
        """
        self.collision_enabled = enable
        
        if self.physics_engine == "bullet" and self.physics_id is not None:
            import pybullet as p
            # Bullet中启用或禁用碰撞的方法
            # 简化实现，实际需要更复杂的处理
            pass


class Constraint:
    """约束类
    表示物理世界中的约束
    """
    
    def __init__(self, body_a, body_b, constraint_type="point"):
        """初始化约束
        
        Args:
            body_a: 第一个刚体
            body_b: 第二个刚体
            constraint_type: 约束类型 ("point", "hinge", "slider", "fixed")
        """
        self.body_a = body_a
        self.body_b = body_b
        self.constraint_type = constraint_type
        
        # 约束参数
        self.pivot_in_a = Vector3(0, 0, 0)
        self.pivot_in_b = Vector3(0, 0, 0)
        self.frame_in_a = Quaternion.identity()
        self.frame_in_b = Quaternion.identity()
        
        # 约束限制
        self.lower_limit = 0.0
        self.upper_limit = 0.0
        
        # 物理引擎ID
        self.physics_id = None
        
        # 约束属性
        self.enabled = True
        self.breaking_impulse_threshold = 0.0  # 0表示不可断裂
    
    def set_limits(self, lower_limit, upper_limit):
        """设置约束限制
        
        Args:
            lower_limit: 下限
            upper_limit: 上限
        """
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        
        if self.physics_engine == "bullet" and self.physics_id is not None:
            import pybullet as p
            # 根据约束类型设置限制
            if self.constraint_type == "hinge":
                p.changeConstraint(self.physics_id, lowerLimit=lower_limit, upperLimit=upper_limit)
