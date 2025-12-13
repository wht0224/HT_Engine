# -*- coding: utf-8 -*-
"""
Quaternion类修复版本
"""

import numpy as np
from Engine.Math import Vector3

class Quaternion:
    """
    四元数类，用于表示旋转
    优化版本，针对低端GPU
    """
    
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        self.data = np.array([x, y, z, w], dtype=np.float32)
    
    def __mul__(self, other):
        if isinstance(other, Vector3):
            # 四元数旋转向量
            # 直接实现四元数旋转向量的公式，避免递归调用
            qx, qy, qz, qw = self.x, self.y, self.z, self.w
            vx, vy, vz = other.x, other.y, other.z
            
            # 向量旋转大冒险，四元数的魔法公式
            # 公式：v' = v + 2q × (q × v + qw v)
            # 叉积就像魔法棒，挥舞起来！
            
            # 第一步：q × v，四元数和向量的第一次亲密接触
            cross_qv_x = qy * vz - qz * vy
            cross_qv_y = qz * vx - qx * vz
            cross_qv_z = qx * vy - qy * vx
            
            # 第二步：q × v + qw v，混合两种魔法
            temp_x = cross_qv_x + qw * vx
            temp_y = cross_qv_y + qw * vy
            temp_z = cross_qv_z + qw * vz
            
            # 第三步：q × (q × v + qw v)，再来一次叉积
            cross_qtemp_x = qy * temp_z - qz * temp_y
            cross_qtemp_y = qz * temp_x - qx * temp_z
            cross_qtemp_z = qx * temp_y - qy * temp_x
            
            # 最终结果：v + 2 * q × (q × v + qw v)，旋转完成！
            result_x = vx + 2.0 * cross_qtemp_x
            result_y = vy + 2.0 * cross_qtemp_y
            result_z = vz + 2.0 * cross_qtemp_z
            
            return Vector3(result_x, result_y, result_z)
        elif isinstance(other, Quaternion):
            # 四元数乘法
            q1 = self
            q2 = other
            
            x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y
            y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x
            z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w
            w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.w
            
            return Quaternion(x, y, z, w)
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'Quaternion' and '{type(other).__name__}'")
    
    @property
    def x(self):
        return self.data[0]
    
    @x.setter
    def x(self, value):
        self.data[0] = value
    
    @property
    def y(self):
        return self.data[1]
    
    @y.setter
    def y(self, value):
        self.data[1] = value
    
    @property
    def z(self):
        return self.data[2]
    
    @z.setter
    def z(self, value):
        self.data[2] = value
    
    @property
    def w(self):
        return self.data[3]
    
    @w.setter
    def w(self, value):
        self.data[3] = value
    
    def normalize(self):
        """
        归一化四元数
        """
        length = np.linalg.norm(self.data)
        if length > 1e-6:
            self.data /= length
        return self
    
    def copy(self):
        """
        创建四元数的副本
        
        Returns:
            Quaternion: 四元数的副本
        """
        return Quaternion(self.x, self.y, self.z, self.w)
    
    @classmethod
    def identity(cls):
        """
        创建单位四元数
        """
        return cls(0.0, 0.0, 0.0, 1.0)
    
    @classmethod
    def from_euler(cls, pitch, yaw, roll):
        """
        欧拉角变四元数，就像把三个角度变成魔法药水
        """
        # 第一步：角度转弧度，数学的魔法转换
        pitch *= 0.5
        yaw *= 0.5
        roll *= 0.5
        
        # 第二步：召唤正弦余弦，三角函数的小精灵
        sin_pitch = np.sin(pitch)
        cos_pitch = np.cos(pitch)
        sin_yaw = np.sin(yaw)
        cos_yaw = np.cos(yaw)
        sin_roll = np.sin(roll)
        cos_roll = np.cos(roll)
        
        # 构造四元数
        q = cls()
        q.w = cos_pitch * cos_yaw * cos_roll + sin_pitch * sin_yaw * sin_roll
        q.x = sin_pitch * cos_yaw * cos_roll - cos_pitch * sin_yaw * sin_roll
        q.y = cos_pitch * sin_yaw * cos_roll + sin_pitch * cos_yaw * sin_roll
        q.z = cos_pitch * cos_yaw * sin_roll - sin_pitch * sin_yaw * cos_roll
        
        return q
    
    @classmethod
    def from_matrix(cls, matrix):
        """
        从旋转矩阵创建四元数
        """
        m = matrix.data
        q = cls()
        
        # 计算四元数分量
        trace = m[0, 0] + m[1, 1] + m[2, 2]
        
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            q.w = 0.25 * s
            q.x = (m[2, 1] - m[1, 2]) / s
            q.y = (m[0, 2] - m[2, 0]) / s
            q.z = (m[1, 0] - m[0, 1]) / s
        elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            q.w = (m[2, 1] - m[1, 2]) / s
            q.x = 0.25 * s
            q.y = (m[0, 1] + m[1, 0]) / s
            q.z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            q.w = (m[0, 2] - m[2, 0]) / s
            q.x = (m[0, 1] + m[1, 0]) / s
            q.y = 0.25 * s
            q.z = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            q.w = (m[1, 0] - m[0, 1]) / s
            q.x = (m[0, 2] + m[2, 0]) / s
            q.y = (m[1, 2] + m[2, 1]) / s
            q.z = 0.25 * s
        
        return q
    
    def to_matrix(self):
        """
        将四元数转换为旋转矩阵
        """
        from Engine.Math import Matrix4x4
        matrix = Matrix4x4()
        m = matrix.data
        
        x, y, z, w = self.x, self.y, self.z, self.w
        
        # 计算旋转矩阵元素
        m[0, 0] = 1 - 2*y*y - 2*z*z
        m[0, 1] = 2*x*y - 2*z*w
        m[0, 2] = 2*x*z + 2*y*w
        
        m[1, 0] = 2*x*y + 2*z*w
        m[1, 1] = 1 - 2*x*x - 2*z*z
        m[1, 2] = 2*y*z - 2*x*w
        
        m[2, 0] = 2*x*z - 2*y*w
        m[2, 1] = 2*y*z + 2*x*w
        m[2, 2] = 1 - 2*x*x - 2*y*y
        
        return matrix
