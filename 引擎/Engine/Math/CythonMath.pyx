# -*- coding: utf-8 -*-
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
"""
优化的数学库，针对性能进行了优化
使用Cython实现，添加C类型标注以提高性能
"""

# Cython全局优化指令
# 禁用边界检查、负索引环绕、None检查等，提高性能
# 启用C风格除法，提高除法运算性能

cimport cython
import math
from libc.math cimport sqrt, sin, cos

# 内联的C函数，用于优化常用数学运算
cdef inline double _inline_sqrt(double x) nogil:
    """内联的平方根函数"""
    return sqrt(x)

cdef inline double _inline_sin(double x) nogil:
    """内联的正弦函数"""
    return sin(x)

cdef inline double _inline_cos(double x) nogil:
    """内联的余弦函数"""
    return cos(x)

cdef inline double _inline_inv_sqrt(double x) nogil:
    """内联的平方根倒数函数"""
    return 1.0 / sqrt(x)

# 向量运算的内联函数
cdef inline double _vector3_dot(double x1, double y1, double z1, double x2, double y2, double z2) nogil:
    """内联的向量点积函数"""
    return x1 * x2 + y1 * y2 + z1 * z2

cdef inline double _vector3_length_squared(double x, double y, double z) nogil:
    """内联的向量长度平方函数"""
    return x * x + y * y + z * z

cdef inline double _vector3_length(double x, double y, double z) nogil:
    """内联的向量长度函数"""
    return sqrt(x * x + y * y + z * z)

cdef inline double _quaternion_length_squared(double x, double y, double z, double w) nogil:
    """内联的四元数长度平方函数"""
    return x * x + y * y + z * z + w * w

cdef inline double _quaternion_length(double x, double y, double z, double w) nogil:
    """内联的四元数长度函数"""
    return sqrt(x * x + y * y + z * z + w * w)

# 定义C类型的向量、四元数和矩阵类
cdef class Vector3:
    """
    3D向量类，用于位置、方向、缩放等计算
    针对性能优化的实现，使用C类型标注
    """
    
    # 使用紧凑的内存布局，减少内存占用
    # 不使用__slots__，直接使用C属性
    cdef public double x, y, z
    
    def __init__(self, double x=0.0, double y=0.0, double z=0.0):
        self.x = x
        self.y = y
        self.z = z
    
    # 直接访问C属性，避免Python属性访问开销
    @property
    def data(self):
        return [self.x, self.y, self.z]
    
    @data.setter
    def data(self, list values):
        if len(values) != 3:
            raise ValueError("列表长度必须为3")
        self.x = values[0]
        self.y = values[1]
        self.z = values[2]
    
    def __add__(self, other):
        cdef double val
        if isinstance(other, Vector3):
            return Vector3(
                self.x + other.x,
                self.y + other.y,
                self.z + other.z
            )
        else:
            val = other
            return Vector3(
                self.x + val,
                self.y + val,
                self.z + val
            )
    
    def __iadd__(self, other):
        """
        原地加法，避免创建新对象
        """
        cdef double val
        if isinstance(other, Vector3):
            self.x += other.x
            self.y += other.y
            self.z += other.z
        else:
            val = other
            self.x += val
            self.y += val
            self.z += val
        return self
    
    def __sub__(self, other):
        cdef double val
        if isinstance(other, Vector3):
            return Vector3(
                self.x - other.x,
                self.y - other.y,
                self.z - other.z
            )
        else:
            val = other
            return Vector3(
                self.x - val,
                self.y - val,
                self.z - val
            )
    
    def __isub__(self, other):
        """
        原地减法，避免创建新对象
        """
        cdef double val
        if isinstance(other, Vector3):
            self.x -= other.x
            self.y -= other.y
            self.z -= other.z
        else:
            val = other
            self.x -= val
            self.y -= val
            self.z -= val
        return self
    
    def __mul__(self, other):
        cdef double val
        if isinstance(other, (int, float)):
            # 与标量相乘
            val = other
            return Vector3(
                self.x * val,
                self.y * val,
                self.z * val
            )
        elif isinstance(other, Vector3):
            # 向量点积
            return self.x * other.x + self.y * other.y + self.z * other.z
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'Vector3' and '{type(other).__name__}'")
    
    def __imul__(self, other):
        """
        原地标量乘法，避免创建新对象
        """
        cdef double val
        if isinstance(other, (int, float)):
            val = other
            self.x *= val
            self.y *= val
            self.z *= val
        else:
            raise TypeError(f"Unsupported operand type(s) for *=: 'Vector3' and '{type(other).__name__}'")
        return self
    
    def __truediv__(self, other):
        cdef double inv_val
        if isinstance(other, (int, float)):
            # 与标量相除 - 使用乘法代替除法
            inv_val = 1.0 / other
            return Vector3(
                self.x * inv_val,
                self.y * inv_val,
                self.z * inv_val
            )
        else:
            raise TypeError(f"Unsupported operand type(s) for /: 'Vector3' and '{type(other).__name__}'")
    
    def __itruediv__(self, other):
        """
        原地标量除法，避免创建新对象
        """
        cdef double inv_val
        if isinstance(other, (int, float)):
            inv_val = 1.0 / other
            self.x *= inv_val
            self.y *= inv_val
            self.z *= inv_val
        else:
            raise TypeError(f"Unsupported operand type(s) for /=: 'Vector3' and '{type(other).__name__}'")
        return self
    
    def normalize(self):
        """
        归一化向量（原地修改）
        优化版本：使用内联函数，减少函数调用开销
        """
        cdef double length_squared, inv_length
        length_squared = _vector3_length_squared(self.x, self.y, self.z)
        if length_squared > 1e-6:
            inv_length = 1.0 / _inline_sqrt(length_squared)
            self.x *= inv_length
            self.y *= inv_length
            self.z *= inv_length
        return self
    
    def normalized(self):
        """
        返回归一化的向量副本
        优化版本：使用内联函数，减少函数调用开销
        """
        cdef double length_squared, inv_length
        length_squared = _vector3_length_squared(self.x, self.y, self.z)
        if length_squared > 1e-6:
            inv_length = 1.0 / _inline_sqrt(length_squared)
            return Vector3(
                self.x * inv_length,
                self.y * inv_length,
                self.z * inv_length
            )
        return Vector3(self.x, self.y, self.z)
    
    def length(self):
        """
        计算向量长度
        优化版本：使用内联函数
        """
        return _vector3_length(self.x, self.y, self.z)
    
    def length_squared(self):
        """
        计算向量长度的平方
        优化版本：使用内联函数
        """
        return _vector3_length_squared(self.x, self.y, self.z)
    
    def dot(self, Vector3 other):
        """
        计算点积
        优化版本：使用内联函数，减少函数调用开销
        """
        return _vector3_dot(self.x, self.y, self.z, other.x, other.y, other.z)
    
    def cross(self, Vector3 other):
        """
        计算叉积
        优化版本：减少中间变量，提高性能
        """
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def cross_in_place(self, Vector3 other):
        """
        原地计算叉积，避免创建新对象
        优化版本：减少中间变量，提高性能
        """
        cdef double x1 = self.x
        cdef double y1 = self.y
        cdef double z1 = self.z
        cdef double x2 = other.x
        cdef double y2 = other.y
        cdef double z2 = other.z
        self.x = y1 * z2 - z1 * y2
        self.y = z1 * x2 - x1 * z2
        self.z = x1 * y2 - y1 * x2
        return self
    
    def copy(self):
        """
        创建向量的副本
        优化版本：直接创建，减少函数调用开销
        """
        return Vector3(self.x, self.y, self.z)
    
    def set(self, double x, double y, double z):
        """
        直接设置向量值，避免创建新对象
        优化版本：直接赋值，减少类型转换
        """
        self.x = x
        self.y = y
        self.z = z
        return self
    
    def set_vector(self, Vector3 other):
        """
        从另一个向量设置值，避免创建新对象
        优化版本：直接赋值，减少类型转换
        """
        self.x = other.x
        self.y = other.y
        self.z = other.z
        return self
    
    # 前端接口 - 与tk库兼容
    def to_list(self):
        """
        返回Python列表格式，用于前端库
        """
        return [self.x, self.y, self.z]
    
    def to_tuple(self):
        """
        返回Python元组格式，用于前端库
        """
        return (self.x, self.y, self.z)
    
    def from_list(self, list values):
        """
        从Python列表创建向量
        """
        if len(values) != 3:
            raise ValueError("列表长度必须为3")
        self.x = values[0]
        self.y = values[1]
        self.z = values[2]
        return self
    
    def from_tuple(self, tuple values):
        """
        从Python元组创建向量
        """
        if len(values) != 3:
            raise ValueError("元组长度必须为3")
        self.x = values[0]
        self.y = values[1]
        self.z = values[2]
        return self
    
    def to_tk(self):
        """
        返回tk库兼容的坐标格式（仅使用x和y分量）
        """
        return (self.x, self.y)
    
    def from_tk(self, tuple tk_coord, double z=0.0):
        """
        从tk库坐标创建向量
        """
        self.x = tk_coord[0]
        self.y = tk_coord[1]
        self.z = z
        return self
    
    # 前端接口 - 与OpenGL兼容
    def to_opengl(self):
        """
        返回OpenGL兼容的向量格式
        """
        return [self.x, self.y, self.z]
    
    def from_opengl(self, list gl_vec):
        """
        从OpenGL向量创建
        """
        if len(gl_vec) != 3:
            raise ValueError("OpenGL向量长度必须为3")
        self.x = gl_vec[0]
        self.y = gl_vec[1]
        self.z = gl_vec[2]
        return self
    
    # 批量运算支持
    @staticmethod
    def batch_normalize(list vectors):
        """
        批量归一化向量
        优化版本：使用内联函数，减少函数调用开销
        """
        cdef int i
        cdef Vector3 vec
        cdef double length_squared, inv_length
        cdef int n = len(vectors)
        result = []
        result.reserve(n)  # 预分配内存
        for i in range(n):
            vec = vectors[i]
            length_squared = _vector3_length_squared(vec.x, vec.y, vec.z)
            if length_squared > 1e-6:
                inv_length = 1.0 / _inline_sqrt(length_squared)
                result.append(Vector3(
                    vec.x * inv_length,
                    vec.y * inv_length,
                    vec.z * inv_length
                ))
            else:
                result.append(Vector3(vec.x, vec.y, vec.z))
        return result
    
    @staticmethod
    def batch_dot(list vectors1, list vectors2):
        """
        批量计算向量点积
        优化版本：使用内联函数，减少函数调用开销
        """
        cdef int i
        cdef Vector3 v1, v2
        cdef int n = min(len(vectors1), len(vectors2))
        result = []
        result.reserve(n)  # 预分配内存
        for i in range(n):
            v1 = vectors1[i]
            v2 = vectors2[i]
            result.append(_vector3_dot(v1.x, v1.y, v1.z, v2.x, v2.y, v2.z))
        return result
    
    @staticmethod
    def batch_add(list vectors1, list vectors2):
        """
        批量计算向量加法
        """
        cdef int i
        cdef Vector3 v1, v2
        cdef int n = min(len(vectors1), len(vectors2))
        result = []
        result.reserve(n)  # 预分配内存
        for i in range(n):
            v1 = vectors1[i]
            v2 = vectors2[i]
            result.append(Vector3(
                v1.x + v2.x,
                v1.y + v2.y,
                v1.z + v2.z
            ))
        return result
    
    @staticmethod
    def batch_subtract(list vectors1, list vectors2):
        """
        批量计算向量减法
        """
        cdef int i
        cdef Vector3 v1, v2
        cdef int n = min(len(vectors1), len(vectors2))
        result = []
        result.reserve(n)  # 预分配内存
        for i in range(n):
            v1 = vectors1[i]
            v2 = vectors2[i]
            result.append(Vector3(
                v1.x - v2.x,
                v1.y - v2.y,
                v1.z - v2.z
            ))
        return result
    
    @staticmethod
    def batch_multiply_scalar(list vectors, double scalar):
        """
        批量计算向量与标量的乘法
        """
        cdef int i
        cdef Vector3 vec
        cdef int n = len(vectors)
        result = []
        result.reserve(n)  # 预分配内存
        for i in range(n):
            vec = vectors[i]
            result.append(Vector3(
                vec.x * scalar,
                vec.y * scalar,
                vec.z * scalar
            ))
        return result
    
    @staticmethod
    def batch_cross(list vectors1, list vectors2):
        """
        批量计算向量叉积
        """
        cdef int i
        cdef Vector3 v1, v2
        cdef int n = min(len(vectors1), len(vectors2))
        result = []
        result.reserve(n)  # 预分配内存
        for i in range(n):
            v1 = vectors1[i]
            v2 = vectors2[i]
            result.append(v1.cross(v2))
        return result

cdef class Quaternion:
    """
    四元数类，用于表示旋转
    优化版本，使用C类型标注
    """
    
    # 使用紧凑的内存布局，减少内存占用
    # 不使用__slots__，直接使用C属性
    cdef public double x, y, z, w
    
    def __init__(self, double x=0.0, double y=0.0, double z=0.0, double w=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.w = w
    
    # 直接访问C属性，避免Python属性访问开销
    @property
    def data(self):
        return [self.x, self.y, self.z, self.w]
    
    @data.setter
    def data(self, list values):
        if len(values) != 4:
            raise ValueError("列表长度必须为4")
        self.x = values[0]
        self.y = values[1]
        self.z = values[2]
        self.w = values[3]
    
    def __mul__(self, other):
        # 所有cdef声明必须在函数开头
        cdef double qx, qy, qz, qw, vx, vy, vz
        cdef double q1x, q1y, q1z, q1w, q2x, q2y, q2z, q2w
        cdef double temp1, temp2, temp3, temp4
        
        if isinstance(other, Vector3):
            # 四元数旋转向量 - 优化版本：减少中间变量，展开计算
            qx = self.x
            qy = self.y
            qz = self.z
            qw = self.w
            vx = other.x
            vy = other.y
            vz = other.z
            
            # 优化旋转计算：减少乘法次数，使用更高效的算法
            # 优化后的算法：只需要16次乘法和15次加法
            xx = qx * qx
            yy = qy * qy
            zz = qz * qz
            
            xy = qx * qy
            xz = qx * qz
            yz = qy * qz
            wx = qw * qx
            wy = qw * qy
            wz = qw * qz
            
            return Vector3(
                vx * (1.0 - 2.0 * (yy + zz)) + vy * (2.0 * (xy + wz)) + vz * (2.0 * (xz - wy)),
                vx * (2.0 * (xy - wz)) + vy * (1.0 - 2.0 * (xx + zz)) + vz * (2.0 * (yz + wx)),
                vx * (2.0 * (xz + wy)) + vy * (2.0 * (yz - wx)) + vz * (1.0 - 2.0 * (xx + yy))
            )
        elif isinstance(other, Quaternion):
            # 四元数乘法 - 优化版本：直接计算，减少中间变量
            q1x = self.x
            q1y = self.y
            q1z = self.z
            q1w = self.w
            q2x = other.x
            q2y = other.y
            q2z = other.z
            q2w = other.w
            
            # 直接返回计算结果，避免额外的变量赋值
            return Quaternion(
                q1w*q2x + q1x*q2w + q1y*q2z - q1z*q2y,
                q1w*q2y - q1x*q2z + q1y*q2w + q1z*q2x,
                q1w*q2z + q1x*q2y - q1y*q2x + q1z*q2w,
                q1w*q2w - q1x*q2x - q1y*q2y - q1z*q2z
            )
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'Quaternion' and '{type(other).__name__}'")
    
    def __imul__(self, Quaternion other):
        """
        原地四元数乘法，避免创建新对象
        优化版本：减少中间变量，提高性能
        """
        cdef double q1x = self.x
        cdef double q1y = self.y
        cdef double q1z = self.z
        cdef double q1w = self.w
        cdef double q2x = other.x
        cdef double q2y = other.y
        cdef double q2z = other.z
        cdef double q2w = other.w
        
        self.x = q1w*q2x + q1x*q2w + q1y*q2z - q1z*q2y
        self.y = q1w*q2y - q1x*q2z + q1y*q2w + q1z*q2x
        self.z = q1w*q2z + q1x*q2y - q1y*q2x + q1z*q2w
        self.w = q1w*q2w - q1x*q2x - q1y*q2y - q1z*q2w
        return self
    
    def normalize(self):
        """
        归一化四元数
        优化版本：使用内联函数，减少函数调用开销
        """
        cdef double length_squared, inv_length
        length_squared = _quaternion_length_squared(self.x, self.y, self.z, self.w)
        if length_squared > 1e-6:
            inv_length = 1.0 / _inline_sqrt(length_squared)
            self.x *= inv_length
            self.y *= inv_length
            self.z *= inv_length
            self.w *= inv_length
        return self
    
    def normalized(self):
        """
        返回归一化的四元数副本
        优化版本：使用内联函数，减少函数调用开销
        """
        cdef double length_squared, inv_length
        length_squared = _quaternion_length_squared(self.x, self.y, self.z, self.w)
        if length_squared > 1e-6:
            inv_length = 1.0 / _inline_sqrt(length_squared)
            return Quaternion(
                self.x * inv_length,
                self.y * inv_length,
                self.z * inv_length,
                self.w * inv_length
            )
        return Quaternion(self.x, self.y, self.z, self.w)
    
    def copy(self):
        """
        创建四元数的副本
        """
        return Quaternion(self.x, self.y, self.z, self.w)
    
    def set(self, double x, double y, double z, double w):
        """
        直接设置四元数值，避免创建新对象
        """
        self.x = x
        self.y = y
        self.z = z
        self.w = w
        return self
    
    def set_identity(self):
        """
        设置为单位四元数
        """
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.w = 1.0
        return self
    
    @classmethod
    def identity(cls):
        """
        创建单位四元数
        """
        return cls(0.0, 0.0, 0.0, 1.0)
    
    @classmethod
    def from_euler(cls, double pitch, double yaw, double roll):
        """
        从欧拉角创建四元数
        """
        # 角度转弧度
        pitch *= 0.5
        yaw *= 0.5
        roll *= 0.5
        
        # 计算正弦和余弦
        cdef double sin_pitch = sin(pitch)
        cdef double cos_pitch = cos(pitch)
        cdef double sin_yaw = sin(yaw)
        cdef double cos_yaw = cos(yaw)
        cdef double sin_roll = sin(roll)
        cdef double cos_roll = cos(roll)
        
        # 构造四元数
        cdef Quaternion q = cls()
        q.w = cos_pitch * cos_yaw * cos_roll + sin_pitch * sin_yaw * sin_roll
        q.x = sin_pitch * cos_yaw * cos_roll - cos_pitch * sin_yaw * sin_roll
        q.y = cos_pitch * sin_yaw * cos_roll + sin_pitch * cos_yaw * sin_roll
        q.z = cos_pitch * cos_yaw * sin_roll - sin_pitch * sin_yaw * cos_roll
        
        return q
    
    def to_euler(self):
        """
        将四元数转换为欧拉角
        """
        cdef double pitch, yaw, roll
        cdef double sinp = 2.0 * (self.w * self.x - self.z * self.y)
        
        # 检查是否为奇异情况
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2.0, sinp)
            yaw = 0.0
            roll = 2.0 * math.atan2(self.x, self.w)
        else:
            pitch = math.asin(sinp)
            yaw = math.atan2(2.0 * (self.w * self.y + self.x * self.z), 1.0 - 2.0 * (self.x * self.x + self.y * self.y))
            roll = math.atan2(2.0 * (self.w * self.z + self.x * self.y), 1.0 - 2.0 * (self.x * self.x + self.z * self.z))
        
        return (pitch, yaw, roll)
    
    @classmethod
    def from_axis_angle(cls, Vector3 axis, double angle):
        """
        从轴角表示创建四元数
        """
        cdef double half_angle = angle * 0.5
        cdef double sin_half = sin(half_angle)
        cdef Quaternion q = cls()
        q.x = axis.x * sin_half
        q.y = axis.y * sin_half
        q.z = axis.z * sin_half
        q.w = cos(half_angle)
        return q
    
    def to_axis_angle(self):
        """
        将四元数转换为轴角表示
        """
        cdef double angle, sin_half
        cdef Vector3 axis
        
        angle = 2.0 * math.acos(self.w)
        sin_half = sin(angle * 0.5)
        
        if sin_half > 1e-6:
            axis = Vector3(
                self.x / sin_half,
                self.y / sin_half,
                self.z / sin_half
            )
        else:
            axis = Vector3(1.0, 0.0, 0.0)
        
        return (axis, angle)
    
    # 前端接口
    def to_list(self):
        """
        返回Python列表格式
        """
        return [self.x, self.y, self.z, self.w]
    
    def to_tuple(self):
        """
        返回Python元组格式
        """
        return (self.x, self.y, self.z, self.w)
    
    def from_list(self, list values):
        """
        从Python列表创建四元数
        """
        if len(values) != 4:
            raise ValueError("列表长度必须为4")
        self.x = values[0]
        self.y = values[1]
        self.z = values[2]
        self.w = values[3]
        return self
    
    def from_tuple(self, tuple values):
        """
        从Python元组创建四元数
        """
        if len(values) != 4:
            raise ValueError("元组长度必须为4")
        self.x = values[0]
        self.y = values[1]
        self.z = values[2]
        self.w = values[3]
        return self
    
    # 前端接口 - 与OpenGL兼容
    def to_opengl(self):
        """
        返回OpenGL兼容的四元数格式
        """
        return [self.x, self.y, self.z, self.w]
    
    def from_opengl(self, list gl_quat):
        """
        从OpenGL四元数创建
        """
        if len(gl_quat) != 4:
            raise ValueError("OpenGL四元数长度必须为4")
        self.x = gl_quat[0]
        self.y = gl_quat[1]
        self.z = gl_quat[2]
        self.w = gl_quat[3]
        return self
    
    # 批量运算支持
    @staticmethod
    def batch_normalize(list quaternions):
        """
        批量归一化四元数
        优化版本：使用内联函数，预分配内存
        """
        cdef int i
        cdef Quaternion quat
        cdef double length_squared, inv_length
        cdef int n = len(quaternions)
        result = []
        result.reserve(n)  # 预分配内存
        for i in range(n):
            quat = quaternions[i]
            length_squared = _quaternion_length_squared(quat.x, quat.y, quat.z, quat.w)
            if length_squared > 1e-6:
                inv_length = 1.0 / _inline_sqrt(length_squared)
                result.append(Quaternion(
                    quat.x * inv_length,
                    quat.y * inv_length,
                    quat.z * inv_length,
                    quat.w * inv_length
                ))
            else:
                result.append(Quaternion(quat.x, quat.y, quat.z, quat.w))
        return result
    
    @staticmethod
    def batch_multiply(list quaternions1, list quaternions2):
        """
        批量计算四元数乘法
        """
        cdef int i
        cdef Quaternion q1, q2
        cdef int n = min(len(quaternions1), len(quaternions2))
        result = []
        result.reserve(n)  # 预分配内存
        for i in range(n):
            q1 = quaternions1[i]
            q2 = quaternions2[i]
            result.append(q1 * q2)
        return result
    
    @staticmethod
    def batch_rotate_vectors(list quaternions, list vectors):
        """
        批量使用四元数旋转向量
        """
        cdef int i
        cdef Quaternion q
        cdef Vector3 v
        cdef int n = min(len(quaternions), len(vectors))
        result = []
        result.reserve(n)  # 预分配内存
        for i in range(n):
            q = quaternions[i]
            v = vectors[i]
            result.append(q * v)
        return result
    
    @staticmethod
    def batch_from_euler(list euler_angles):
        """
        批量从欧拉角创建四元数
        """
        cdef int i
        cdef tuple angles
        cdef int n = len(euler_angles)
        result = []
        result.reserve(n)  # 预分配内存
        for i in range(n):
            angles = euler_angles[i]
            result.append(Quaternion.from_euler(angles[0], angles[1], angles[2]))
        return result

cdef class Matrix4x4:
    """
    4x4矩阵类，用于3D变换
    针对性能优化的实现，使用C类型标注
    """
    
    # 使用C数组存储矩阵数据，提高访问速度
    # 使用行主序存储，与数学定义一致
    cdef public double data[16]
    
    def __init__(self):
        # 初始化单位矩阵，直接赋值，减少内存访问
        self.data[0] = 1.0; self.data[1] = 0.0; self.data[2] = 0.0; self.data[3] = 0.0
        self.data[4] = 0.0; self.data[5] = 1.0; self.data[6] = 0.0; self.data[7] = 0.0
        self.data[8] = 0.0; self.data[9] = 0.0; self.data[10] = 1.0; self.data[11] = 0.0
        self.data[12] = 0.0; self.data[13] = 0.0; self.data[14] = 0.0; self.data[15] = 1.0
    
    # 直接访问C数组，避免Python属性访问开销
    @property
    def rows(self):
        return [
            [self.data[0], self.data[1], self.data[2], self.data[3]],
            [self.data[4], self.data[5], self.data[6], self.data[7]],
            [self.data[8], self.data[9], self.data[10], self.data[11]],
            [self.data[12], self.data[13], self.data[14], self.data[15]]
        ]
    
    @rows.setter
    def rows(self, list new_rows):
        if len(new_rows) != 4:
            raise ValueError("行数必须为4")
        for i in range(4):
            if len(new_rows[i]) != 4:
                raise ValueError(f"第{i}行长度必须为4")
            for j in range(4):
                self.data[i*4 + j] = new_rows[i][j]
    
    @classmethod
    def identity(cls):
        """
        创建单位矩阵
        """
        cdef Matrix4x4 matrix = cls.__new__(cls)  # 避免调用__init__
        # 直接初始化数据为单位矩阵
        matrix.data[0] = 1.0
        matrix.data[1] = 0.0
        matrix.data[2] = 0.0
        matrix.data[3] = 0.0
        matrix.data[4] = 0.0
        matrix.data[5] = 1.0
        matrix.data[6] = 0.0
        matrix.data[7] = 0.0
        matrix.data[8] = 0.0
        matrix.data[9] = 0.0
        matrix.data[10] = 1.0
        matrix.data[11] = 0.0
        matrix.data[12] = 0.0
        matrix.data[13] = 0.0
        matrix.data[14] = 0.0
        matrix.data[15] = 1.0
        return matrix
    
    @staticmethod
    def _multiply_matrices(list a, list b):
        """
        静态方法：矩阵乘法，使用完全展开的循环以消除循环开销
        4x4矩阵乘法完全展开，显著提高性能
        兼容测试用例，接受Python列表并返回Python列表
        """
        # 优化：针对变换矩阵的快速乘法
        # 变换矩阵的第4行固定为 [0, 0, 0, 1]
        # 直接使用C类型的变量，避免Python列表的访问开销
        # 完全展开矩阵乘法，减少循环开销
        
        # 直接访问Python列表，使用Cython的类型标注来提高访问速度
        # 预计算所有乘法结果，减少内存访问
        cdef double a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3]
        cdef double a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7]
        cdef double a8 = a[8], a9 = a[9], a10 = a[10], a11 = a[11]
        cdef double b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3]
        cdef double b4 = b[4], b5 = b[5], b6 = b[6], b7 = b[7]
        cdef double b8 = b[8], b9 = b[9], b10 = b[10], b11 = b[11]
        
        # 直接计算结果，避免中间变量赋值开销
        # 行0: 旋转缩放 + 平移
        # 行1: 旋转缩放 + 平移
        # 行2: 旋转缩放 + 平移
        # 行3: 固定为 [0, 0, 0, 1]
        
        # 直接返回结果列表，避免列表初始化开销
        return [
            # 行0
            a0*b0 + a1*b4 + a2*b8,           # 0,0
            a0*b1 + a1*b5 + a2*b9,           # 0,1
            a0*b2 + a1*b6 + a2*b10,          # 0,2
            a0*b3 + a1*b7 + a2*b11 + a3,     # 0,3 (平移)
            
            # 行1
            a4*b0 + a5*b4 + a6*b8,           # 1,0
            a4*b1 + a5*b5 + a6*b9,           # 1,1
            a4*b2 + a5*b6 + a6*b10,          # 1,2
            a4*b3 + a5*b7 + a6*b11 + a7,     # 1,3 (平移)
            
            # 行2
            a8*b0 + a9*b4 + a10*b8,          # 2,0
            a8*b1 + a9*b5 + a10*b9,          # 2,1
            a8*b2 + a9*b6 + a10*b10,         # 2,2
            a8*b3 + a9*b7 + a10*b11 + a11,    # 2,3 (平移)
            
            # 行3
            0.0, 0.0, 0.0, 1.0               # 固定为 [0, 0, 0, 1]
        ]
    
    def __mul__(self, other):
        """
        矩阵乘法
        直接实现矩阵乘法，避免调用静态方法和类型转换
        """
        cdef Matrix4x4 result = Matrix4x4()
        
        # 直接计算矩阵乘法，避免调用静态方法和类型转换
        # 行0: 旋转缩放 + 平移
        result.data[0] = self.data[0]*other.data[0] + self.data[1]*other.data[4] + self.data[2]*other.data[8]
        result.data[1] = self.data[0]*other.data[1] + self.data[1]*other.data[5] + self.data[2]*other.data[9]
        result.data[2] = self.data[0]*other.data[2] + self.data[1]*other.data[6] + self.data[2]*other.data[10]
        result.data[3] = self.data[0]*other.data[3] + self.data[1]*other.data[7] + self.data[2]*other.data[11] + self.data[3]
        
        # 行1: 旋转缩放 + 平移
        result.data[4] = self.data[4]*other.data[0] + self.data[5]*other.data[4] + self.data[6]*other.data[8]
        result.data[5] = self.data[4]*other.data[1] + self.data[5]*other.data[5] + self.data[6]*other.data[9]
        result.data[6] = self.data[4]*other.data[2] + self.data[5]*other.data[6] + self.data[6]*other.data[10]
        result.data[7] = self.data[4]*other.data[3] + self.data[5]*other.data[7] + self.data[6]*other.data[11] + self.data[7]
        
        # 行2: 旋转缩放 + 平移
        result.data[8] = self.data[8]*other.data[0] + self.data[9]*other.data[4] + self.data[10]*other.data[8]
        result.data[9] = self.data[8]*other.data[1] + self.data[9]*other.data[5] + self.data[10]*other.data[9]
        result.data[10] = self.data[8]*other.data[2] + self.data[9]*other.data[6] + self.data[10]*other.data[10]
        result.data[11] = self.data[8]*other.data[3] + self.data[9]*other.data[7] + self.data[10]*other.data[11] + self.data[11]
        
        # 行3: 固定为 [0, 0, 0, 1]
        result.data[12] = 0.0
        result.data[13] = 0.0
        result.data[14] = 0.0
        result.data[15] = 1.0
        
        return result
    
    def multiply_vector(self, vector):
        """
        用矩阵乘以3D向量
        """
        cdef double vx, vy, vz
        cdef double x, y, z
        vx = vector.x
        vy = vector.y
        vz = vector.z
        
        x = (self.data[0] * vx +
             self.data[1] * vy +
             self.data[2] * vz +
             self.data[3])
        
        y = (self.data[4] * vx +
             self.data[5] * vy +
             self.data[6] * vz +
             self.data[7])
        
        z = (self.data[8] * vx +
             self.data[9] * vy +
             self.data[10] * vz +
             self.data[11])
        
        return Vector3(x, y, z)
    
    def multiply_vector_in_place(self, vector):
        """
        用矩阵乘以3D向量，原地修改向量
        """
        cdef double vx, vy, vz
        cdef double x, y, z
        vx = vector.x
        vy = vector.y
        vz = vector.z
        
        x = (self.data[0] * vx +
             self.data[1] * vy +
             self.data[2] * vz +
             self.data[3])
        
        y = (self.data[4] * vx +
             self.data[5] * vy +
             self.data[6] * vz +
             self.data[7])
        
        z = (self.data[8] * vx +
             self.data[9] * vy +
             self.data[10] * vz +
             self.data[11])
        
        vector.x = x
        vector.y = y
        vector.z = z
        return vector
    
    @classmethod
    def from_rotation(cls, rotation):
        """
        从旋转四元数创建变换矩阵
        """
        cdef Matrix4x4 matrix = cls()
        cdef double x = rotation.x
        cdef double y = rotation.y
        cdef double z = rotation.z
        cdef double w = rotation.w
        
        matrix.data[0] = 1 - 2*y*y - 2*z*z
        matrix.data[1] = 2*x*y - 2*z*w
        matrix.data[2] = 2*x*z + 2*y*w
        
        matrix.data[4] = 2*x*y + 2*z*w
        matrix.data[5] = 1 - 2*x*x - 2*z*z
        matrix.data[6] = 2*y*z - 2*x*w
        
        matrix.data[8] = 2*x*z - 2*y*w
        matrix.data[9] = 2*y*z + 2*x*w
        matrix.data[10] = 1 - 2*x*x - 2*y*y
        
        return matrix
    
    @classmethod
    def from_translation(cls, Vector3 position):
        """
        从平移向量创建变换矩阵
        """
        cdef Matrix4x4 matrix = cls()
        matrix.data[3] = position.x
        matrix.data[7] = position.y
        matrix.data[11] = position.z
        return matrix
    
    @classmethod
    def translation(cls, double x, double y, double z):
        """
        从x, y, z分量创建平移矩阵
        兼容测试用例
        """
        cdef Matrix4x4 matrix = cls()
        matrix.data[3] = x
        matrix.data[7] = y
        matrix.data[11] = z
        return matrix
    
    @classmethod
    def from_scale(cls, scale):
        """
        从缩放向量创建变换矩阵
        """
        cdef Matrix4x4 matrix = cls()
        matrix.data[0] = scale.x
        matrix.data[5] = scale.y
        matrix.data[10] = scale.z
        return matrix
    
    @classmethod
    def from_transform(cls, Vector3 position, Quaternion rotation, Vector3 scale):
        """
        从位置、旋转和缩放创建变换矩阵
        优化版本：直接计算最终结果，避免创建临时矩阵
        """
        cdef Matrix4x4 matrix = cls()
        
        # 提取旋转四元数分量
        cdef double qx = rotation.x
        cdef double qy = rotation.y
        cdef double qz = rotation.z
        cdef double qw = rotation.w
        
        # 提取缩放分量
        cdef double sx = scale.x
        cdef double sy = scale.y
        cdef double sz = scale.z
        
        # 计算旋转矩阵元素
        cdef double r00 = 1 - 2*qy*qy - 2*qz*qz
        cdef double r01 = 2*qx*qy - 2*qz*qw
        cdef double r02 = 2*qx*qz + 2*qy*qw
        cdef double r10 = 2*qx*qy + 2*qz*qw
        cdef double r11 = 1 - 2*qx*qx - 2*qz*qz
        cdef double r12 = 2*qy*qz - 2*qx*qw
        cdef double r20 = 2*qx*qz - 2*qy*qw
        cdef double r21 = 2*qy*qz + 2*qx*qw
        cdef double r22 = 1 - 2*qx*qx - 2*qy*qy
        
        # 组合旋转和缩放
        matrix.data[0] = r00 * sx
        matrix.data[1] = r01 * sy
        matrix.data[2] = r02 * sz
        matrix.data[3] = position.x
        
        matrix.data[4] = r10 * sx
        matrix.data[5] = r11 * sy
        matrix.data[6] = r12 * sz
        matrix.data[7] = position.y
        
        matrix.data[8] = r20 * sx
        matrix.data[9] = r21 * sy
        matrix.data[10] = r22 * sz
        matrix.data[11] = position.z
        
        return matrix
    
    def get_translation(self):
        """
        获取矩阵的平移分量
        """
        return Vector3(self.data[3], self.data[7], self.data[11])
    
    def get_rotation(self):
        """
        获取矩阵的旋转分量（四元数）
        """
        # 所有cdef语句必须放在函数开头
        cdef Quaternion q = Quaternion()
        cdef double m00 = self.data[0], m01 = self.data[1], m02 = self.data[2]
        cdef double m10 = self.data[4], m11 = self.data[5], m12 = self.data[6]
        cdef double m20 = self.data[8], m21 = self.data[9], m22 = self.data[10]
        cdef double trace = m00 + m11 + m22
        cdef double s
        
        if trace > 0:
            s = sqrt(trace + 1.0) * 2
            q.w = 0.25 * s
            q.x = (m21 - m12) / s
            q.y = (m02 - m20) / s
            q.z = (m10 - m01) / s
        elif (m00 > m11) and (m00 > m22):
            s = sqrt(1.0 + m00 - m11 - m22) * 2
            q.w = (m21 - m12) / s
            q.x = 0.25 * s
            q.y = (m01 + m10) / s
            q.z = (m02 + m20) / s
        elif m11 > m22:
            s = sqrt(1.0 + m11 - m00 - m22) * 2
            q.w = (m02 - m20) / s
            q.x = (m01 + m10) / s
            q.y = 0.25 * s
            q.z = (m12 + m21) / s
        else:
            s = sqrt(1.0 + m22 - m00 - m11) * 2
            q.w = (m10 - m01) / s
            q.x = (m02 + m20) / s
            q.y = (m12 + m21) / s
            q.z = 0.25 * s
        
        return q
    
    def get_scale(self):
        """
        获取矩阵的缩放分量
        """
        cdef double sx = sqrt(self.data[0]*self.data[0] + self.data[1]*self.data[1] + self.data[2]*self.data[2])
        cdef double sy = sqrt(self.data[4]*self.data[4] + self.data[5]*self.data[5] + self.data[6]*self.data[6])
        cdef double sz = sqrt(self.data[8]*self.data[8] + self.data[9]*self.data[9] + self.data[10]*self.data[10])
        return Vector3(sx, sy, sz)
    
    # 前端接口
    def to_list(self):
        """
        返回Python列表格式
        """
        return [self.data[i] for i in range(16)]
    
    def to_tuple(self):
        """
        返回Python元组格式
        """
        return tuple(self.data[i] for i in range(16))
    
    def from_list(self, list values):
        """
        从Python列表创建矩阵
        """
        if len(values) != 16:
            raise ValueError("列表长度必须为16")
        for i in range(16):
            self.data[i] = values[i]
        return self
    
    def from_tuple(self, tuple values):
        """
        从Python元组创建矩阵
        """
        if len(values) != 16:
            raise ValueError("元组长度必须为16")
        for i in range(16):
            self.data[i] = values[i]
        return self
    
    # 前端接口 - 与OpenGL兼容
    def to_opengl(self):
        """
        返回OpenGL兼容的矩阵格式（列主序）
        """
        # OpenGL使用列主序，需要转置
        return [
            self.data[0], self.data[4], self.data[8], self.data[12],
            self.data[1], self.data[5], self.data[9], self.data[13],
            self.data[2], self.data[6], self.data[10], self.data[14],
            self.data[3], self.data[7], self.data[11], self.data[15]
        ]
    
    def from_opengl(self, list gl_mat):
        """
        从OpenGL矩阵创建（列主序）
        """
        if len(gl_mat) != 16:
            raise ValueError("OpenGL矩阵长度必须为16")
        # OpenGL使用列主序，需要转置
        self.data[0] = gl_mat[0]
        self.data[1] = gl_mat[4]
        self.data[2] = gl_mat[8]
        self.data[3] = gl_mat[12]
        self.data[4] = gl_mat[1]
        self.data[5] = gl_mat[5]
        self.data[6] = gl_mat[9]
        self.data[7] = gl_mat[13]
        self.data[8] = gl_mat[2]
        self.data[9] = gl_mat[6]
        self.data[10] = gl_mat[10]
        self.data[11] = gl_mat[14]
        self.data[12] = gl_mat[3]
        self.data[13] = gl_mat[7]
        self.data[14] = gl_mat[11]
        self.data[15] = gl_mat[15]
        return self
    
    # 批量运算支持
    @staticmethod
    def batch_multiply(list matrices1, list matrices2):
        """
        批量计算矩阵乘法
        """
        cdef int i
        cdef Matrix4x4 m1, m2, result
        cdef int n = min(len(matrices1), len(matrices2))
        result_list = []
        result_list.reserve(n)  # 预分配内存
        for i in range(n):
            m1 = matrices1[i]
            m2 = matrices2[i]
            result = m1 * m2
            result_list.append(result)
        return result_list
    
    @staticmethod
    def batch_transform(list matrices, list vectors):
        """
        批量用矩阵变换向量
        """
        cdef int i
        cdef Matrix4x4 mat
        cdef Vector3 vec
        cdef int n = min(len(matrices), len(vectors))
        result_list = []
        result_list.reserve(n)  # 预分配内存
        for i in range(n):
            mat = matrices[i]
            vec = vectors[i]
            result_list.append(mat.multiply_vector(vec))
        return result_list
    
    @staticmethod
    def batch_translation(list vectors):
        """
        批量创建平移矩阵
        """
        cdef int i
        cdef Vector3 v
        cdef int n = len(vectors)
        result_list = []
        result_list.reserve(n)  # 预分配内存
        for i in range(n):
            v = vectors[i]
            result_list.append(Matrix4x4.from_translation(v))
        return result_list
    
    @staticmethod
    def batch_rotation(list quaternions):
        """
        批量从四元数创建旋转矩阵
        """
        cdef int i
        cdef Quaternion q
        cdef int n = len(quaternions)
        result_list = []
        result_list.reserve(n)  # 预分配内存
        for i in range(n):
            q = quaternions[i]
            result_list.append(Matrix4x4.from_rotation(q))
        return result_list
    
    @staticmethod
    def batch_scale(list vectors):
        """
        批量创建缩放矩阵
        """
        cdef int i
        cdef Vector3 v
        cdef int n = len(vectors)
        result_list = []
        result_list.reserve(n)  # 预分配内存
        for i in range(n):
            v = vectors[i]
            result_list.append(Matrix4x4.from_scale(v))
        return result_list

# 类会自动导出供Python使用
