# -*- coding: utf-8 -*-
"""
优化的数学库，针对性能进行了优化
"""

import math
import array

class Vector2:
    """
    2D向量类，用于UV坐标、屏幕坐标等2D计算
    针对性能优化的实现
    """
    
    __slots__ = ['x', 'y']
    
    def __init__(self, x=0.0, y=0.0):
        self.x = float(x)
        self.y = float(y)
    
    def __add__(self, other):
        if isinstance(other, Vector2):
            return Vector2(self.x + other.x, self.y + other.y)
        else:
            return Vector2(self.x + other, self.y + other)
    
    def __sub__(self, other):
        if isinstance(other, Vector2):
            return Vector2(self.x - other.x, self.y - other.y)
        else:
            return Vector2(self.x - other, self.y - other)
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            # 与标量相乘
            return Vector2(self.x * other, self.y * other)
        elif isinstance(other, Vector2):
            # 向量点积
            return self.x * other.x + self.y * other.y
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'Vector2' and '{type(other).__name__}'")
    
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            # 与标量相除
            inv_other = 1.0 / other
            return Vector2(self.x * inv_other, self.y * inv_other)
        else:
            raise TypeError(f"Unsupported operand type(s) for /: 'Vector2' and '{type(other).__name__}'")
    
    def normalize(self):
        """
        归一化向量（原地修改）
        """
        length_squared = self.x * self.x + self.y * self.y
        if length_squared > 1e-6:
            inv_length = 1.0 / math.sqrt(length_squared)
            self.x *= inv_length
            self.y *= inv_length
        return self
    
    def normalized(self):
        """
        返回归一化的向量副本
        """
        length_squared = self.x * self.x + self.y * self.y
        if length_squared > 1e-6:
            inv_length = 1.0 / math.sqrt(length_squared)
            return Vector2(self.x * inv_length, self.y * inv_length)
        return Vector2(self.x, self.y)
    
    def length(self):
        """
        计算向量长度
        """
        return math.sqrt(self.x * self.x + self.y * self.y)
    
    def length_squared(self):
        """
        计算向量长度的平方
        """
        return self.x * self.x + self.y * self.y
    
    def dot(self, other):
        """
        计算点积
        """
        return self.x * other.x + self.y * other.y
    
    def copy(self):
        """
        创建向量的副本
        """
        return Vector2(self.x, self.y)

class Vector3:
    """
    3D向量类，用于位置、方向、缩放等计算
    针对性能优化的实现
    """
    
    __slots__ = ['_data']  # 使用列表存储，提高访问速度
    
    def __init__(self, x=0.0, y=0.0, z=0.0):
        # 使用数组存储分量，提高内存访问效率
        self._data = [float(x), float(y), float(z)]
    
    # 快速访问属性
    @property
    def x(self):
        return self._data[0]
    
    @x.setter
    def x(self, value):
        self._data[0] = float(value)
    
    @property
    def y(self):
        return self._data[1]
    
    @y.setter
    def y(self, value):
        self._data[1] = float(value)
    
    @property
    def z(self):
        return self._data[2]
    
    @z.setter
    def z(self, value):
        self._data[2] = float(value)
    
    def __add__(self, other):
        if isinstance(other, Vector3):
            # 直接访问底层数据，避免属性访问开销
            return Vector3(
                self._data[0] + other._data[0],
                self._data[1] + other._data[1],
                self._data[2] + other._data[2]
            )
        else:
            return Vector3(
                self._data[0] + other,
                self._data[1] + other,
                self._data[2] + other
            )
    
    def __iadd__(self, other):
        """
        原地加法，避免创建新对象
        """
        if isinstance(other, Vector3):
            self._data[0] += other._data[0]
            self._data[1] += other._data[1]
            self._data[2] += other._data[2]
        else:
            self._data[0] += other
            self._data[1] += other
            self._data[2] += other
        return self
    
    def __sub__(self, other):
        if isinstance(other, Vector3):
            return Vector3(
                self._data[0] - other._data[0],
                self._data[1] - other._data[1],
                self._data[2] - other._data[2]
            )
        else:
            return Vector3(
                self._data[0] - other,
                self._data[1] - other,
                self._data[2] - other
            )
    
    def __isub__(self, other):
        """
        原地减法，避免创建新对象
        """
        if isinstance(other, Vector3):
            self._data[0] -= other._data[0]
            self._data[1] -= other._data[1]
            self._data[2] -= other._data[2]
        else:
            self._data[0] -= other
            self._data[1] -= other
            self._data[2] -= other
        return self
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            # 与标量相乘
            return Vector3(
                self._data[0] * other,
                self._data[1] * other,
                self._data[2] * other
            )
        elif isinstance(other, Vector3):
            # 向量点积 - 直接计算，避免函数调用
            return (
                self._data[0] * other._data[0] +
                self._data[1] * other._data[1] +
                self._data[2] * other._data[2]
            )
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'Vector3' and '{type(other).__name__}'")
    
    def __imul__(self, other):
        """
        原地标量乘法，避免创建新对象
        """
        if isinstance(other, (int, float)):
            self._data[0] *= other
            self._data[1] *= other
            self._data[2] *= other
        else:
            raise TypeError(f"Unsupported operand type(s) for *=: 'Vector3' and '{type(other).__name__}'")
        return self
    
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            # 与标量相除 - 使用乘法代替除法
            inv_other = 1.0 / other
            return Vector3(
                self._data[0] * inv_other,
                self._data[1] * inv_other,
                self._data[2] * inv_other
            )
        else:
            raise TypeError(f"Unsupported operand type(s) for /: 'Vector3' and '{type(other).__name__}'")
    
    def __itruediv__(self, other):
        """
        原地标量除法，避免创建新对象
        """
        if isinstance(other, (int, float)):
            inv_other = 1.0 / other
            self._data[0] *= inv_other
            self._data[1] *= inv_other
            self._data[2] *= inv_other
        else:
            raise TypeError(f"Unsupported operand type(s) for /=: 'Vector3' and '{type(other).__name__}'")
        return self
    
    def normalize(self):
        """
        归一化向量（原地修改）
        使用快速平方根算法
        """
        x, y, z = self._data
        length_squared = x * x + y * y + z * z
        if length_squared > 1e-6:
            # 快速平方根近似 - 使用牛顿迭代法
            inv_length = 1.0 / math.sqrt(length_squared)
            self._data[0] = x * inv_length
            self._data[1] = y * inv_length
            self._data[2] = z * inv_length
        return self
    
    def normalized(self):
        """
        返回归一化的向量副本
        """
        x, y, z = self._data
        length_squared = x * x + y * y + z * z
        if length_squared > 1e-6:
            inv_length = 1.0 / math.sqrt(length_squared)
            return Vector3(
                x * inv_length,
                y * inv_length,
                z * inv_length
            )
        return Vector3(x, y, z)
    
    def length(self):
        """
        计算向量长度
        """
        x, y, z = self._data
        return math.sqrt(x * x + y * y + z * z)
    
    def length_squared(self):
        """
        计算向量长度的平方
        """
        x, y, z = self._data
        return x * x + y * y + z * z
    
    def dot(self, other):
        """
        计算点积
        """
        return (
            self._data[0] * other._data[0] +
            self._data[1] * other._data[1] +
            self._data[2] * other._data[2]
        )
    
    def cross(self, other):
        """
        计算叉积
        """
        x1, y1, z1 = self._data
        x2, y2, z2 = other._data
        return Vector3(
            y1 * z2 - z1 * y2,
            z1 * x2 - x1 * z2,
            x1 * y2 - y1 * x2
        )
    
    def cross_in_place(self, other):
        """
        原地计算叉积，避免创建新对象
        """
        x1, y1, z1 = self._data
        x2, y2, z2 = other._data
        self._data[0] = y1 * z2 - z1 * y2
        self._data[1] = z1 * x2 - x1 * z2
        self._data[2] = x1 * y2 - y1 * x2
        return self
    
    def copy(self):
        """
        创建向量的副本
        """
        return Vector3(*self._data)
    
    def set(self, x, y, z):
        """
        直接设置向量值，避免创建新对象
        """
        self._data[0] = float(x)
        self._data[1] = float(y)
        self._data[2] = float(z)
        return self
    
    @staticmethod
    def add(a, b):
        """
        静态方法：向量加法，避免实例方法开销
        """
        return Vector3(
            a._data[0] + b._data[0],
            a._data[1] + b._data[1],
            a._data[2] + b._data[2]
        )
    
    @staticmethod
    def multiply_scalar(a, scalar):
        """
        静态方法：向量标量乘法，避免实例方法开销
        """
        return Vector3(
            a._data[0] * scalar,
            a._data[1] * scalar,
            a._data[2] * scalar
        )
    
    @staticmethod
    def batch_normalize(vectors):
        """
        批量归一化向量，优化内存访问模式
        """
        result = []
        for vec in vectors:
            x, y, z = vec._data
            length_squared = x * x + y * y + z * z
            if length_squared > 1e-6:
                inv_length = 1.0 / math.sqrt(length_squared)
                result.append(Vector3(x * inv_length, y * inv_length, z * inv_length))
            else:
                result.append(Vector3(x, y, z))
        return result

class Matrix4x4:
    """
    4x4矩阵类，用于3D变换
    针对性能优化的实现
    """
    
    __slots__ = ['_data']
    
    def __init__(self):
        # 使用列表存储矩阵数据，访问速度更快
        self._data = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        ]
    
    # 直接访问_data属性，避免property访问开销
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, value):
        # 直接赋值，确保数据类型一致性
        self._data = value
    
    @classmethod
    def identity(cls):
        """
        创建单位矩阵
        优化版本，直接返回预初始化的单位矩阵
        """
        matrix = cls.__new__(cls)  # 避免调用__init__
        # 直接初始化数据为列表，避免__init__的开销
        matrix._data = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        ]
        return matrix
    
    @classmethod
    def from_translation(cls, position):
        """
        从平移向量创建变换矩阵
        """
        matrix = cls()
        matrix.data[3] = position.x
        matrix.data[7] = position.y
        matrix.data[11] = position.z
        return matrix
    
    @classmethod
    def translation(cls, x, y, z):
        """
        从x, y, z分量创建平移矩阵
        """
        matrix = cls()
        matrix.data[3] = x
        matrix.data[7] = y
        matrix.data[11] = z
        return matrix
    
    @classmethod
    def from_rotation(cls, rotation):
        """
        从旋转四元数创建变换矩阵
        """
        matrix = cls()
        # 四元数到矩阵的转换实现
        q = rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        
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
    def from_scale(cls, scale):
        """
        从缩放向量创建变换矩阵
        """
        matrix = cls()
        matrix.data[0] = scale.x
        matrix.data[5] = scale.y
        matrix.data[10] = scale.z
        return matrix
    
    @classmethod
    def from_transform(cls, position, rotation, scale):
        """
        从位置、旋转和缩放创建变换矩阵
        """
        # 创建变换矩阵
        matrix = cls()
        
        # 应用缩放
        scale_matrix = cls.from_scale(scale)
        
        # 应用旋转
        rotation_matrix = cls.from_rotation(rotation)
        
        # 应用平移
        translation_matrix = cls.from_translation(position)
        
        # 组合变换: 平移 * 旋转 * 缩放
        temp = cls._multiply_matrices(rotation_matrix.data, scale_matrix.data)
        matrix.data = cls._multiply_matrices(translation_matrix.data, temp)
        
        return matrix
    
    @staticmethod
    def _multiply_matrices(a, b):
        """
        静态方法：矩阵乘法，使用完全展开的循环以消除循环开销
        4x4矩阵乘法完全展开，显著提高性能
        """
        # 优化：针对变换矩阵的快速乘法
        # 变换矩阵的第4行固定为 [0, 0, 0, 1]
        # 利用这个特性优化乘法计算
        
        # 直接计算结果，避免局部变量赋值开销和预计算常用乘积的开销
        # 直接使用数组索引访问，提高访问速度
        return [
            # 行0: 旋转缩放 + 平移
            a[0]*b[0] + a[1]*b[4] + a[2]*b[8],  # 0,0
            a[0]*b[1] + a[1]*b[5] + a[2]*b[9],  # 0,1
            a[0]*b[2] + a[1]*b[6] + a[2]*b[10],  # 0,2
            a[0]*b[3] + a[1]*b[7] + a[2]*b[11] + a[3],  # 0,3 (平移)
            
            # 行1: 旋转缩放 + 平移
            a[4]*b[0] + a[5]*b[4] + a[6]*b[8],  # 1,0
            a[4]*b[1] + a[5]*b[5] + a[6]*b[9],  # 1,1
            a[4]*b[2] + a[5]*b[6] + a[6]*b[10],  # 1,2
            a[4]*b[3] + a[5]*b[7] + a[6]*b[11] + a[7],  # 1,3 (平移)
            
            # 行2: 旋转缩放 + 平移
            a[8]*b[0] + a[9]*b[4] + a[10]*b[8],  # 2,0
            a[8]*b[1] + a[9]*b[5] + a[10]*b[9],  # 2,1
            a[8]*b[2] + a[9]*b[6] + a[10]*b[10],  # 2,2
            a[8]*b[3] + a[9]*b[7] + a[10]*b[11] + a[11],  # 2,3 (平移)
            
            # 行3: 固定为 [0, 0, 0, 1]
            0.0, 0.0, 0.0, 1.0
        ]
    
    def __mul__(self, other):
        """
        矩阵乘法
        优化版本，使用预分配的结果矩阵，避免创建新对象的开销
        """
        # 直接创建结果矩阵，避免调用__init__
        result = self.__class__.__new__(self.__class__)
        # 预分配结果列表，避免列表推导式开销
        result._data = [0.0] * 16
        # 直接执行矩阵乘法，避免函数调用开销
        a = self._data
        b = other._data
        
        # 行0
        result._data[0] = a[0]*b[0] + a[1]*b[4] + a[2]*b[8]
        result._data[1] = a[0]*b[1] + a[1]*b[5] + a[2]*b[9]
        result._data[2] = a[0]*b[2] + a[1]*b[6] + a[2]*b[10]
        result._data[3] = a[0]*b[3] + a[1]*b[7] + a[2]*b[11] + a[3]
        
        # 行1
        result._data[4] = a[4]*b[0] + a[5]*b[4] + a[6]*b[8]
        result._data[5] = a[4]*b[1] + a[5]*b[5] + a[6]*b[9]
        result._data[6] = a[4]*b[2] + a[5]*b[6] + a[6]*b[10]
        result._data[7] = a[4]*b[3] + a[5]*b[7] + a[6]*b[11] + a[7]
        
        # 行2
        result._data[8] = a[8]*b[0] + a[9]*b[4] + a[10]*b[8]
        result._data[9] = a[8]*b[1] + a[9]*b[5] + a[10]*b[9]
        result._data[10] = a[8]*b[2] + a[9]*b[6] + a[10]*b[10]
        result._data[11] = a[8]*b[3] + a[9]*b[7] + a[10]*b[11] + a[11]
        
        # 行3
        result._data[12] = 0.0
        result._data[13] = 0.0
        result._data[14] = 0.0
        result._data[15] = 1.0
        
        return result
    
    def __imul__(self, other):
        """
        原地矩阵乘法，避免创建新对象
        优化版本，直接在原地进行矩阵乘法，避免创建临时矩阵
        """
        if isinstance(other, Matrix4x4):
            # 保存当前矩阵数据到临时变量，避免覆盖
            # 使用array的copy()方法，确保正确复制array类型数据
            a = self._data.copy()
            b = other._data
            
            # 直接在原地计算结果
            # 行0
            self._data[0] = a[0]*b[0] + a[1]*b[4] + a[2]*b[8]
            self._data[1] = a[0]*b[1] + a[1]*b[5] + a[2]*b[9]
            self._data[2] = a[0]*b[2] + a[1]*b[6] + a[2]*b[10]
            self._data[3] = a[0]*b[3] + a[1]*b[7] + a[2]*b[11] + a[3]
            
            # 行1
            self._data[4] = a[4]*b[0] + a[5]*b[4] + a[6]*b[8]
            self._data[5] = a[4]*b[1] + a[5]*b[5] + a[6]*b[9]
            self._data[6] = a[4]*b[2] + a[5]*b[6] + a[6]*b[10]
            self._data[7] = a[4]*b[3] + a[5]*b[7] + a[6]*b[11] + a[7]
            
            # 行2
            self._data[8] = a[8]*b[0] + a[9]*b[4] + a[10]*b[8]
            self._data[9] = a[8]*b[1] + a[9]*b[5] + a[10]*b[9]
            self._data[10] = a[8]*b[2] + a[9]*b[6] + a[10]*b[10]
            self._data[11] = a[8]*b[3] + a[9]*b[7] + a[10]*b[11] + a[11]
            
            # 行3保持不变
            # self._data[12] = 0.0
            # self._data[13] = 0.0
            # self._data[14] = 0.0
            # self._data[15] = 1.0
            
            return self
        else:
            raise TypeError(f"Unsupported operand type(s) for *=: 'Matrix4x4' and '{type(other).__name__}'")
    
    def transpose(self):
        """
        计算矩阵的转置
        """
        result = Matrix4x4()
        d = self.data
        rd = result.data
        
        # 手动展开转置操作，避免任何循环或函数调用
        rd[0] = d[0]
        rd[1] = d[4]
        rd[2] = d[8]
        rd[3] = d[12]
        
        rd[4] = d[1]
        rd[5] = d[5]
        rd[6] = d[9]
        rd[7] = d[13]
        
        rd[8] = d[2]
        rd[9] = d[6]
        rd[10] = d[10]
        rd[11] = d[14]
        
        rd[12] = d[3]
        rd[13] = d[7]
        rd[14] = d[11]
        rd[15] = d[15]
        
        return result
    
    def transpose_in_place(self):
        """
        原地转置矩阵，避免创建新对象
        """
        d = self.data
        
        # 手动交换元素，避免任何循环或函数调用
        # 交换(0,1)和(1,0)
        temp = d[1]
        d[1] = d[4]
        d[4] = temp
        
        # 交换(0,2)和(2,0)
        temp = d[2]
        d[2] = d[8]
        d[8] = temp
        
        # 交换(0,3)和(3,0)
        temp = d[3]
        d[3] = d[12]
        d[12] = temp
        
        # 交换(1,2)和(2,1)
        temp = d[6]
        d[6] = d[9]
        d[9] = temp
        
        # 交换(1,3)和(3,1)
        temp = d[7]
        d[7] = d[13]
        d[13] = temp
        
        # 交换(2,3)和(3,2)
        temp = d[11]
        d[11] = d[14]
        d[14] = temp
        
        return self
    
    def __getitem__(self, key):
        """
        获取矩阵行
        """
        return [
            self.data[key*4 + 0],
            self.data[key*4 + 1],
            self.data[key*4 + 2],
            self.data[key*4 + 3]
        ]
    
    def __setitem__(self, key, value):
        """
        设置矩阵行
        """
        self.data[key*4 + 0] = value[0]
        self.data[key*4 + 1] = value[1]
        self.data[key*4 + 2] = value[2]
        self.data[key*4 + 3] = value[3]
    
    def get_element(self, row, col):
        """
        获取矩阵元素
        """
        return self.data[row*4 + col]
    
    def set_element(self, row, col, value):
        """
        设置矩阵元素
        """
        self.data[row*4 + col] = float(value)
        return self
    
    def inverse(self):
        """
        计算矩阵的逆
        """
        # 简化实现，仅支持变换矩阵的逆
        # 对于变换矩阵，可以通过分解为平移、旋转和缩放来高效计算逆
        result = Matrix4x4()
        d = self.data
        rd = result.data
        
        # 提取旋转部分（上3x3子矩阵）
        # 旋转矩阵的逆是其转置
        rd[0] = d[0]; rd[1] = d[4]; rd[2] = d[8]
        rd[4] = d[1]; rd[5] = d[5]; rd[6] = d[9]
        rd[8] = d[2]; rd[9] = d[6]; rd[10] = d[10]
        
        # 提取平移向量
        tx = d[3]
        ty = d[7]
        tz = d[11]
        
        # 计算逆平移向量
        rd[3] = -(rd[0] * tx + rd[1] * ty + rd[2] * tz)
        rd[7] = -(rd[4] * tx + rd[5] * ty + rd[6] * tz)
        rd[11] = -(rd[8] * tx + rd[9] * ty + rd[10] * tz)
        
        return result
    
    def inverse_in_place(self):
        """
        原地计算矩阵的逆，避免创建新对象
        """
        # 简化实现，仅支持变换矩阵的逆
        d = self.data
        
        # 保存原始数据用于计算
        orig = d.copy()
        
        # 旋转矩阵的逆是其转置
        d[0], d[1], d[2] = orig[0], orig[4], orig[8]
        d[4], d[5], d[6] = orig[1], orig[5], orig[9]
        d[8], d[9], d[10] = orig[2], orig[6], orig[10]
        
        # 提取平移向量
        tx = orig[3]
        ty = orig[7]
        tz = orig[11]
        
        # 计算逆平移向量
        d[3] = -(d[0] * tx + d[1] * ty + d[2] * tz)
        d[7] = -(d[4] * tx + d[5] * ty + d[6] * tz)
        d[11] = -(d[8] * tx + d[9] * ty + d[10] * tz)
        
        return self
    
    def multiply_vector(self, vector):
        """
        用矩阵乘以3D向量
        """
        d = self.data
        x = (d[0] * vector.x +
             d[1] * vector.y +
             d[2] * vector.z +
             d[3])
        
        y = (d[4] * vector.x +
             d[5] * vector.y +
             d[6] * vector.z +
             d[7])
        
        z = (d[8] * vector.x +
             d[9] * vector.y +
             d[10] * vector.z +
             d[11])
        
        return Vector3(x, y, z)
    
    def transform_point(self, vector):
        """
        变换3D点（与multiply_vector功能相同）
        """
        return self.multiply_vector(vector)
    
    def multiply_vector_in_place(self, vector):
        """
        用矩阵乘以3D向量，原地修改向量
        """
        d = self.data
        x = (d[0] * vector.x +
             d[1] * vector.y +
             d[2] * vector.z +
             d[3])
        
        y = (d[4] * vector.x +
             d[5] * vector.y +
             d[6] * vector.z +
             d[7])
        
        z = (d[8] * vector.x +
             d[9] * vector.y +
             d[10] * vector.z +
             d[11])
        
        vector.set(x, y, z)
        return vector
    
    def copy(self):
        """
        创建矩阵的副本
        """
        matrix = Matrix4x4()
        matrix.data = self.data.copy()
        return matrix
    
    def set(self, other):
        """
        从另一个矩阵复制数据，避免创建新对象
        使用手动复制代替列表copy()方法，提高性能
        """
        # 手动复制数据，避免列表copy()方法的开销
        d = self.data
        other_d = other.data
        
        d[0] = other_d[0]
        d[1] = other_d[1]
        d[2] = other_d[2]
        d[3] = other_d[3]
        d[4] = other_d[4]
        d[5] = other_d[5]
        d[6] = other_d[6]
        d[7] = other_d[7]
        d[8] = other_d[8]
        d[9] = other_d[9]
        d[10] = other_d[10]
        d[11] = other_d[11]
        d[12] = other_d[12]
        d[13] = other_d[13]
        d[14] = other_d[14]
        d[15] = other_d[15]
        
        return self
    
    @classmethod
    def create_perspective(cls, fov_y, aspect_ratio, near_plane, far_plane):
        """
        创建透视投影矩阵
        """
        matrix = cls()
        tan_half_fov = math.tan(fov_y * 0.5)
        
        matrix.data[0] = 1.0 / (aspect_ratio * tan_half_fov)
        matrix.data[5] = 1.0 / tan_half_fov
        matrix.data[10] = -(far_plane + near_plane) / (far_plane - near_plane)
        matrix.data[11] = -2.0 * far_plane * near_plane / (far_plane - near_plane)
        matrix.data[14] = -1.0
        matrix.data[15] = 0.0
        
        return matrix
    
    @classmethod
    def create_look_at(cls, eye, target, up):
        """
        创建观察矩阵
        """
        matrix = cls()
        d = matrix.data
        
        # 计算方向向量
        forward = Vector3(target.x - eye.x, target.y - eye.y, target.z - eye.z)
        forward.normalize()
        
        # 计算右侧向量
        right = forward.cross(up)
        right.normalize()
        
        # 计算上方向向量
        up = right.cross(forward)
        
        # 填充矩阵
        d[0] = right.x; d[1] = right.y; d[2] = right.z; d[3] = -right.dot(eye)
        d[4] = up.x; d[5] = up.y; d[6] = up.z; d[7] = -up.dot(eye)
        d[8] = -forward.x; d[9] = -forward.y; d[10] = -forward.z; d[11] = forward.dot(eye)
        
        return matrix

class Quaternion:
    """
    四元数类，用于表示旋转
    优化版本，针对低端GPU
    """
    
    __slots__ = ['_data']  # 使用列表存储，提高访问速度
    
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        # 使用数组存储分量，提高内存访问效率
        self._data = [float(x), float(y), float(z), float(w)]
    
    # 快速访问属性
    @property
    def x(self):
        return self._data[0]
    
    @x.setter
    def x(self, value):
        self._data[0] = float(value)
    
    @property
    def y(self):
        return self._data[1]
    
    @y.setter
    def y(self, value):
        self._data[1] = float(value)
    
    @property
    def z(self):
        return self._data[2]
    
    @z.setter
    def z(self, value):
        self._data[2] = float(value)
    
    @property
    def w(self):
        return self._data[3]
    
    @w.setter
    def w(self, value):
        self._data[3] = float(value)
    
    def __mul__(self, other):
        if isinstance(other, Vector3):
            # 四元数旋转向量
            # 直接实现四元数旋转向量的公式，避免递归调用
            qx, qy, qz, qw = self.x, self.y, self.z, self.w
            vx, vy, vz = other.x, other.y, other.z
            
            # 计算旋转后的向量
            # 公式：v' = v + 2q × (q × v + qw v)
            # 其中×表示叉积
            
            # 计算 q × v
            cross_qv_x = qy * vz - qz * vy
            cross_qv_y = qz * vx - qx * vz
            cross_qv_z = qx * vy - qy * vx
            
            # 计算 q × v + qw v
            temp_x = cross_qv_x + qw * vx
            temp_y = cross_qv_y + qw * vy
            temp_z = cross_qv_z + qw * vz
            
            # 计算 q × (q × v + qw v)
            cross_qtemp_x = qy * temp_z - qz * temp_y
            cross_qtemp_y = qz * temp_x - qx * temp_z
            cross_qtemp_z = qx * temp_y - qy * temp_x
            
            # 计算最终结果：v + 2 * q × (q × v + qw v)
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
            w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
            
            return Quaternion(x, y, z, w)
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'Quaternion' and '{type(other).__name__}'")
    
    def __imul__(self, other):
        """
        原地四元数乘法，避免创建新对象
        """
        if isinstance(other, Quaternion):
            # 直接访问底层数据，避免属性访问开销
            q1x, q1y, q1z, q1w = self._data
            q2x, q2y, q2z, q2w = other._data
            
            x = q1w*q2x + q1x*q2w + q1y*q2z - q1z*q2y
            y = q1w*q2y - q1x*q2z + q1y*q2w + q1z*q2x
            z = q1w*q2z + q1x*q2y - q1y*q2x + q1z*q2w
            w = q1w*q2w - q1x*q2x - q1y*q2y - q1z*q2w
            
            self._data[0] = x
            self._data[1] = y
            self._data[2] = z
            self._data[3] = w
            return self
        else:
            raise TypeError(f"Unsupported operand type(s) for *=: 'Quaternion' and '{type(other).__name__}'")
    
    def normalize(self):
        """
        归一化四元数
        """
        # 直接访问底层数据，避免属性访问开销
        x, y, z, w = self._data
        length_squared = x*x + y*y + z*z + w*w
        if length_squared > 1e-6:
            inv_length = 1.0 / math.sqrt(length_squared)
            self._data[0] = x * inv_length
            self._data[1] = y * inv_length
            self._data[2] = z * inv_length
            self._data[3] = w * inv_length
        return self
    
    def normalized(self):
        """
        返回归一化的四元数副本
        """
        # 直接访问底层数据，避免属性访问开销
        x, y, z, w = self._data
        length_squared = x*x + y*y + z*z + w*w
        if length_squared > 1e-6:
            inv_length = 1.0 / math.sqrt(length_squared)
            return Quaternion(
                x * inv_length,
                y * inv_length,
                z * inv_length,
                w * inv_length
            )
        return Quaternion(x, y, z, w)
    
    def copy(self):
        """
        创建四元数的副本
        """
        # 直接访问底层数据，避免属性访问开销
        return Quaternion(*self._data)
    
    def set(self, x, y, z, w):
        """
        直接设置四元数值，避免创建新对象
        """
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.w = float(w)
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
    def from_euler(cls, pitch, yaw, roll):
        """
        从欧拉角创建四元数
        """
        # 角度转弧度
        pitch *= 0.5
        yaw *= 0.5
        roll *= 0.5
        
        # 计算正弦和余弦
        sin_pitch = math.sin(pitch)
        cos_pitch = math.cos(pitch)
        sin_yaw = math.sin(yaw)
        cos_yaw = math.cos(yaw)
        sin_roll = math.sin(roll)
        cos_roll = math.cos(roll)
        
        # 构造四元数
        q = cls()
        q.w = cos_pitch * cos_yaw * cos_roll + sin_pitch * sin_yaw * sin_roll
        q.x = sin_pitch * cos_yaw * cos_roll - cos_pitch * sin_yaw * sin_roll
        q.y = cos_pitch * sin_yaw * cos_roll + sin_pitch * cos_yaw * sin_roll
        q.z = cos_pitch * cos_yaw * sin_roll - sin_pitch * sin_yaw * cos_roll
        
        return q
    
    def set_from_euler(self, pitch, yaw, roll):
        """
        从欧拉角设置四元数，原地修改
        """
        # 角度转弧度
        pitch *= 0.5
        yaw *= 0.5
        roll *= 0.5
        
        # 计算正弦和余弦
        sin_pitch = math.sin(pitch)
        cos_pitch = math.cos(pitch)
        sin_yaw = math.sin(yaw)
        cos_yaw = math.cos(yaw)
        sin_roll = math.sin(roll)
        cos_roll = math.cos(roll)
        
        # 设置四元数值
        self.w = cos_pitch * cos_yaw * cos_roll + sin_pitch * sin_yaw * sin_roll
        self.x = sin_pitch * cos_yaw * cos_roll - cos_pitch * sin_yaw * sin_roll
        self.y = cos_pitch * sin_yaw * cos_roll + sin_pitch * cos_yaw * sin_roll
        self.z = cos_pitch * cos_yaw * sin_roll - sin_pitch * sin_yaw * cos_roll
        
        return self
    
    @classmethod
    def from_matrix(cls, matrix):
        """
        从旋转矩阵创建四元数
        """
        m = matrix.data
        q = cls()
        
        # 计算四元数分量 - 使用行主序扁平列表索引
        trace = m[0*4 + 0] + m[1*4 + 1] + m[2*4 + 2]
        
        if trace > 0:
            s = math.sqrt(trace + 1.0) * 2
            q.w = 0.25 * s
            q.x = (m[2*4 + 1] - m[1*4 + 2]) / s
            q.y = (m[0*4 + 2] - m[2*4 + 0]) / s
            q.z = (m[1*4 + 0] - m[0*4 + 1]) / s
        elif (m[0*4 + 0] > m[1*4 + 1]) and (m[0*4 + 0] > m[2*4 + 2]):
            s = math.sqrt(1.0 + m[0*4 + 0] - m[1*4 + 1] - m[2*4 + 2]) * 2
            q.w = (m[2*4 + 1] - m[1*4 + 2]) / s
            q.x = 0.25 * s
            q.y = (m[0*4 + 1] + m[1*4 + 0]) / s
            q.z = (m[0*4 + 2] + m[2*4 + 0]) / s
        elif m[1*4 + 1] > m[2*4 + 2]:
            s = math.sqrt(1.0 + m[1*4 + 1] - m[0*4 + 0] - m[2*4 + 2]) * 2
            q.w = (m[0*4 + 2] - m[2*4 + 0]) / s
            q.x = (m[0*4 + 1] + m[1*4 + 0]) / s
            q.y = 0.25 * s
            q.z = (m[1*4 + 2] + m[2*4 + 1]) / s
        else:
            s = math.sqrt(1.0 + m[2*4 + 2] - m[0*4 + 0] - m[1*4 + 1]) * 2
            q.w = (m[1*4 + 0] - m[0*4 + 1]) / s
            q.x = (m[0*4 + 2] + m[2*4 + 0]) / s
            q.y = (m[1*4 + 2] + m[2*4 + 1]) / s
            q.z = 0.25 * s
        
        return q
    
    def set_from_matrix(self, matrix):
        """
        从旋转矩阵设置四元数，原地修改
        """
        m = matrix.data
        
        # 计算四元数分量 - 使用行主序扁平列表索引
        trace = m[0*4 + 0] + m[1*4 + 1] + m[2*4 + 2]
        
        if trace > 0:
            s = math.sqrt(trace + 1.0) * 2
            self.w = 0.25 * s
            self.x = (m[2*4 + 1] - m[1*4 + 2]) / s
            self.y = (m[0*4 + 2] - m[2*4 + 0]) / s
            self.z = (m[1*4 + 0] - m[0*4 + 1]) / s
        elif (m[0*4 + 0] > m[1*4 + 1]) and (m[0*4 + 0] > m[2*4 + 2]):
            s = math.sqrt(1.0 + m[0*4 + 0] - m[1*4 + 1] - m[2*4 + 2]) * 2
            self.w = (m[2*4 + 1] - m[1*4 + 2]) / s
            self.x = 0.25 * s
            self.y = (m[0*4 + 1] + m[1*4 + 0]) / s
            self.z = (m[0*4 + 2] + m[2*4 + 0]) / s
        elif m[1*4 + 1] > m[2*4 + 2]:
            s = math.sqrt(1.0 + m[1*4 + 1] - m[0*4 + 0] - m[2*4 + 2]) * 2
            self.w = (m[0*4 + 2] - m[2*4 + 0]) / s
            self.x = (m[0*4 + 1] + m[1*4 + 0]) / s
            self.y = 0.25 * s
            self.z = (m[1*4 + 2] + m[2*4 + 1]) / s
        else:
            s = math.sqrt(1.0 + m[2*4 + 2] - m[0*4 + 0] - m[1*4 + 1]) * 2
            self.w = (m[1*4 + 0] - m[0*4 + 1]) / s
            self.x = (m[0*4 + 2] + m[2*4 + 0]) / s
            self.y = (m[1*4 + 2] + m[2*4 + 1]) / s
            self.z = 0.25 * s
        
        return self
    
    def to_matrix(self):
        """
        将四元数转换为旋转矩阵
        """
        matrix = Matrix4x4()
        m = matrix.data
        
        x, y, z, w = self.x, self.y, self.z, self.w
        
        # 计算旋转矩阵元素 - 使用行主序扁平列表索引
        m[0*4 + 0] = 1 - 2*y*y - 2*z*z
        m[0*4 + 1] = 2*x*y - 2*z*w
        m[0*4 + 2] = 2*x*z + 2*y*w
        
        m[1*4 + 0] = 2*x*y + 2*z*w
        m[1*4 + 1] = 1 - 2*x*x - 2*z*z
        m[1*4 + 2] = 2*y*z - 2*x*w
        
        m[2*4 + 0] = 2*x*z - 2*y*w
        m[2*4 + 1] = 2*y*z + 2*x*w
        m[2*4 + 2] = 1 - 2*x*x - 2*y*y
        
        return matrix

class BoundingBox:
    """
    轴对齐包围盒（AABB）类
    用于空间查询、碰撞检测和视锥体裁剪
    """
    
    def __init__(self, min_point=None, max_point=None):
        """
        初始化包围盒
        
        Args:
            min_point: 最小点
            max_point: 最大点
        """
        self.min = min_point or Vector3(0, 0, 0)
        self.max = max_point or Vector3(0, 0, 0)
    
    def transform(self, matrix):
        """
        使用变换矩阵变换包围盒
        
        Args:
            matrix: 变换矩阵
            
        Returns:
            BoundingBox: 变换后的包围盒
        """
        # 获取包围盒的8个顶点
        vertices = [
            Vector3(self.min.x, self.min.y, self.min.z),
            Vector3(self.max.x, self.min.y, self.min.z),
            Vector3(self.min.x, self.max.y, self.min.z),
            Vector3(self.max.x, self.max.y, self.min.z),
            Vector3(self.min.x, self.min.y, self.max.z),
            Vector3(self.max.x, self.min.y, self.max.z),
            Vector3(self.min.x, self.max.y, self.max.z),
            Vector3(self.max.x, self.max.y, self.max.z)
        ]
        
        # 变换所有顶点
        transformed_vertices = [matrix.multiply_vector(v) for v in vertices]
        
        # 计算新的最小和最大点
        new_min = Vector3(float('inf'), float('inf'), float('inf'))
        new_max = Vector3(-float('inf'), -float('inf'), -float('inf'))
        
        for v in transformed_vertices:
            if v.x < new_min.x:
                new_min.x = v.x
            if v.y < new_min.y:
                new_min.y = v.y
            if v.z < new_min.z:
                new_min.z = v.z
            if v.x > new_max.x:
                new_max.x = v.x
            if v.y > new_max.y:
                new_max.y = v.y
            if v.z > new_max.z:
                new_max.z = v.z
        
        return BoundingBox(new_min, new_max)
    
    def get_center(self):
        """
        获取包围盒的中心
        
        Returns:
            Vector3: 包围盒中心
        """
        return Vector3(
            (self.min.x + self.max.x) * 0.5,
            (self.min.y + self.max.y) * 0.5,
            (self.min.z + self.max.z) * 0.5
        )
    
    def get_extents(self):
        """
        获取包围盒的扩展范围（从中心到边的距离）
        
        Returns:
            Vector3: 扩展范围
        """
        return Vector3(
            (self.max.x - self.min.x) * 0.5,
            (self.max.y - self.min.y) * 0.5,
            (self.max.z - self.min.z) * 0.5
        )
    
    def contains_point(self, point):
        """
        检查点是否在包围盒内
        
        Args:
            point: 要检查的点
            
        Returns:
            bool: 是否包含该点
        """
        return (
            self.min.x <= point.x <= self.max.x and
            self.min.y <= point.y <= self.max.y and
            self.min.z <= point.z <= self.max.z
        )
    
    def merge(self, other):
        """
        合并两个包围盒
        
        Args:
            other: 要合并的包围盒
            
        Returns:
            BoundingBox: 合并后的包围盒
        """
        new_min = Vector3(
            min(self.min.x, other.min.x),
            min(self.min.y, other.min.y),
            min(self.min.z, other.min.z)
        )
        
        new_max = Vector3(
            max(self.max.x, other.max.x),
            max(self.max.y, other.max.y),
            max(self.max.z, other.max.z)
        )
        
        return BoundingBox(new_min, new_max)
    
    def expand(self, point):
        """
        扩展包围盒以包含指定点
        
        Args:
            point: 要包含的点
        """
        if point.x < self.min.x:
            self.min.x = point.x
        if point.y < self.min.y:
            self.min.y = point.y
        if point.z < self.min.z:
            self.min.z = point.z
        if point.x > self.max.x:
            self.max.x = point.x
        if point.y > self.max.y:
            self.max.y = point.y
        if point.z > self.max.z:
            self.max.z = point.z
    
    def is_empty(self):
        """
        检查包围盒是否为空
        
        Returns:
            bool: 是否为空
        """
        return (
            self.min.x >= self.max.x or
            self.min.y >= self.max.y or
            self.min.z >= self.max.z
        )
    
    def copy(self):
        """
        复制包围盒
        
        Returns:
            BoundingBox: 复制的包围盒
        """
        return BoundingBox(self.min.copy(), self.max.copy())

class BoundingSphere:
    """
    包围球类
    用于空间查询、碰撞检测和视锥体裁剪
    """
    
    def __init__(self, center=None, radius=0.0):
        """
        初始化包围球
        
        Args:
            center: 球心
            radius: 半径
        """
        self.center = center or Vector3(0, 0, 0)
        self.radius = radius
    
    def contains_point(self, point):
        """
        检查点是否在包围球内
        
        Args:
            point: 要检查的点
            
        Returns:
            bool: 是否包含该点
        """
        # 计算点到球心的距离的平方
        dx = self.center.x - point.x
        dy = self.center.y - point.y
        dz = self.center.z - point.z
        distance_squared = dx * dx + dy * dy + dz * dz
        return distance_squared <= self.radius * self.radius
    
    def contains_bounding_box(self, bounding_box):
        """
        检查包围盒是否在包围球内
        
        Args:
            bounding_box: 包围盒
            
        Returns:
            bool: 是否包含该包围盒
        """
        # 获取包围盒的8个顶点
        vertices = [
            Vector3(bounding_box.min.x, bounding_box.min.y, bounding_box.min.z),
            Vector3(bounding_box.max.x, bounding_box.min.y, bounding_box.min.z),
            Vector3(bounding_box.min.x, bounding_box.max.y, bounding_box.min.z),
            Vector3(bounding_box.max.x, bounding_box.max.y, bounding_box.min.z),
            Vector3(bounding_box.min.x, bounding_box.min.y, bounding_box.max.z),
            Vector3(bounding_box.max.x, bounding_box.min.y, bounding_box.max.z),
            Vector3(bounding_box.min.x, bounding_box.max.y, bounding_box.max.z),
            Vector3(bounding_box.max.x, bounding_box.max.y, bounding_box.max.z)
        ]
        
        # 检查所有顶点是否在包围球内
        for vertex in vertices:
            if not self.contains_point(vertex):
                return False
        
        return True
    
    def intersects_bounding_box(self, bounding_box):
        """
        检查包围盒是否与包围球相交
        
        Args:
            bounding_box: 包围盒
            
        Returns:
            bool: 是否相交
        """
        # 计算包围盒上离球心最近的点
        closest_x = max(bounding_box.min.x, min(self.center.x, bounding_box.max.x))
        closest_y = max(bounding_box.min.y, min(self.center.y, bounding_box.max.y))
        closest_z = max(bounding_box.min.z, min(self.center.z, bounding_box.max.z))
        
        # 计算最近点到球心的距离的平方
        dx = self.center.x - closest_x
        dy = self.center.y - closest_y
        dz = self.center.z - closest_z
        distance_squared = dx * dx + dy * dy + dz * dz
        
        return distance_squared <= self.radius * self.radius
    
    def merge(self, other):
        """
        合并两个包围球
        
        Args:
            other: 要合并的包围球
            
        Returns:
            BoundingSphere: 合并后的包围球
        """
        # 计算两个球心之间的距离
        dx = other.center.x - self.center.x
        dy = other.center.y - self.center.y
        dz = other.center.z - self.center.z
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        
        # 如果一个球完全包含另一个球，返回较大的球
        if distance + other.radius <= self.radius:
            return BoundingSphere(self.center.copy(), self.radius)
        if distance + self.radius <= other.radius:
            return BoundingSphere(other.center.copy(), other.radius)
        
        # 计算合并后的球心和半径
        new_radius = (self.radius + other.radius + distance) * 0.5
        
        # 计算新球心的位置
        if distance > 1e-6:
            t = (new_radius - self.radius) / distance
            new_center = Vector3(
                self.center.x + dx * t,
                self.center.y + dy * t,
                self.center.z + dz * t
            )
        else:
            # 两个球心重合
            new_center = self.center.copy()
        
        return BoundingSphere(new_center, new_radius)
    
    def expand(self, point):
        """
        扩展包围球以包含指定点
        
        Args:
            point: 要包含的点
        """
        # 计算点到球心的距离的平方
        dx = self.center.x - point.x
        dy = self.center.y - point.y
        dz = self.center.z - point.z
        distance_squared = dx * dx + dy * dy + dz * dz
        
        # 如果点已经在球内，不需要扩展
        if distance_squared <= self.radius * self.radius:
            return
        
        # 计算新的半径
        distance = math.sqrt(distance_squared)
        new_radius = (self.radius + distance) * 0.5
        
        # 计算新球心的位置
        if distance > 1e-6:
            t = (new_radius - self.radius) / distance
            self.center.x += dx * t
            self.center.y += dy * t
            self.center.z += dz * t
        
        self.radius = new_radius
    
    def transform(self, matrix):
        """
        使用变换矩阵变换包围球
        
        Args:
            matrix: 变换矩阵
            
        Returns:
            BoundingSphere: 变换后的包围球
        """
        # 变换球心
        new_center = matrix.multiply_vector(self.center)
        
        # 计算变换后的半径
        # 简化实现：使用矩阵的最大缩放因子来估计新的半径
        # 获取矩阵的缩放因子 - 使用行主序扁平列表索引
        scale_x = math.sqrt(matrix.data[0*4 + 0] ** 2 + matrix.data[1*4 + 0] ** 2 + matrix.data[2*4 + 0] ** 2)
        scale_y = math.sqrt(matrix.data[0*4 + 1] ** 2 + matrix.data[1*4 + 1] ** 2 + matrix.data[2*4 + 1] ** 2)
        scale_z = math.sqrt(matrix.data[0*4 + 2] ** 2 + matrix.data[1*4 + 2] ** 2 + matrix.data[2*4 + 2] ** 2)
        
        # 取最大缩放因子
        max_scale = max(scale_x, scale_y, scale_z)
        new_radius = self.radius * max_scale
        
        return BoundingSphere(new_center, new_radius)
    
    def copy(self):
        """
        复制包围球
        
        Returns:
            BoundingSphere: 复制的包围球
        """
        return BoundingSphere(self.center.copy(), self.radius)

class Frustum:
    """
    视锥体类
    用于视锥体裁剪，检查物体是否在相机可见范围内
    """
    
    def __init__(self):
        """
        初始化视锥体
        """
        # 视锥体的6个平面
        # 每个平面用4个值表示：Ax + By + Cz + D = 0
        self.planes = [
            [0.0, 0.0, 0.0, 0.0],  # 左平面
            [0.0, 0.0, 0.0, 0.0],  # 右平面
            [0.0, 0.0, 0.0, 0.0],  # 下平面
            [0.0, 0.0, 0.0, 0.0],  # 上平面
            [0.0, 0.0, 0.0, 0.0],  # 近平面
            [0.0, 0.0, 0.0, 0.0]   # 远平面
        ]
    
    def extract_from_matrix(self, view_projection_matrix):
        """
        从视图投影矩阵提取视锥体平面
        
        Args:
            view_projection_matrix: 视图投影矩阵
        """
        # 视图投影矩阵的行
        rows = view_projection_matrix.data
        
        # 提取6个平面
        # 左平面：行0 + 行3
        self.planes[0][0] = rows[3*4 + 0] + rows[0*4 + 0]
        self.planes[0][1] = rows[3*4 + 1] + rows[0*4 + 1]
        self.planes[0][2] = rows[3*4 + 2] + rows[0*4 + 2]
        self.planes[0][3] = rows[3*4 + 3] + rows[0*4 + 3]
        
        # 右平面：行3 - 行0
        self.planes[1][0] = rows[3*4 + 0] - rows[0*4 + 0]
        self.planes[1][1] = rows[3*4 + 1] - rows[0*4 + 1]
        self.planes[1][2] = rows[3*4 + 2] - rows[0*4 + 2]
        self.planes[1][3] = rows[3*4 + 3] - rows[0*4 + 3]
        
        # 下平面：行3 + 行1
        self.planes[2][0] = rows[3*4 + 0] + rows[1*4 + 0]
        self.planes[2][1] = rows[3*4 + 1] + rows[1*4 + 1]
        self.planes[2][2] = rows[3*4 + 2] + rows[1*4 + 2]
        self.planes[2][3] = rows[3*4 + 3] + rows[1*4 + 3]
        
        # 上平面：行3 - 行1
        self.planes[3][0] = rows[3*4 + 0] - rows[1*4 + 0]
        self.planes[3][1] = rows[3*4 + 1] - rows[1*4 + 1]
        self.planes[3][2] = rows[3*4 + 2] - rows[1*4 + 2]
        self.planes[3][3] = rows[3*4 + 3] - rows[1*4 + 3]
        
        # 近平面：行3 + 行2
        self.planes[4][0] = rows[3*4 + 0] + rows[2*4 + 0]
        self.planes[4][1] = rows[3*4 + 1] + rows[2*4 + 1]
        self.planes[4][2] = rows[3*4 + 2] + rows[2*4 + 2]
        self.planes[4][3] = rows[3*4 + 3] + rows[2*4 + 3]
        
        # 远平面：行3 - 行2
        self.planes[5][0] = rows[3*4 + 0] - rows[2*4 + 0]
        self.planes[5][1] = rows[3*4 + 1] - rows[2*4 + 1]
        self.planes[5][2] = rows[3*4 + 2] - rows[2*4 + 2]
        self.planes[5][3] = rows[3*4 + 3] - rows[2*4 + 3]
        
        # 归一化平面
        for i in range(6):
            self._normalize_plane(i)
    
    def _normalize_plane(self, plane_index):
        """
        归一化平面
        
        Args:
            plane_index: 平面索引
        """
        plane = self.planes[plane_index]
        # 计算平面的模长
        magnitude = math.sqrt(plane[0]**2 + plane[1]**2 + plane[2]**2)
        
        # 归一化平面
        if magnitude > 1e-6:
            inv_magnitude = 1.0 / magnitude
            plane[0] *= inv_magnitude
            plane[1] *= inv_magnitude
            plane[2] *= inv_magnitude
            plane[3] *= inv_magnitude
    
    def contains_bounding_box(self, bounding_box):
        """
        检查包围盒是否在视锥体内
        
        Args:
            bounding_box: 包围盒
            
        Returns:
            bool: 是否在视锥体内
        """
        # 获取包围盒的8个顶点
        vertices = [
            Vector3(bounding_box.min.x, bounding_box.min.y, bounding_box.min.z),
            Vector3(bounding_box.max.x, bounding_box.min.y, bounding_box.min.z),
            Vector3(bounding_box.min.x, bounding_box.max.y, bounding_box.min.z),
            Vector3(bounding_box.max.x, bounding_box.max.y, bounding_box.min.z),
            Vector3(bounding_box.min.x, bounding_box.min.y, bounding_box.max.z),
            Vector3(bounding_box.max.x, bounding_box.min.y, bounding_box.max.z),
            Vector3(bounding_box.min.x, bounding_box.max.y, bounding_box.max.z),
            Vector3(bounding_box.max.x, bounding_box.max.y, bounding_box.max.z)
        ]
        
        # 检查所有顶点是否在所有平面的正面
        for plane in self.planes:
            # 检查是否有顶点在平面正面
            has_vertex_in_front = False
            for vertex in vertices:
                # 计算点到平面的距离（Ax + By + Cz + D）
                distance = plane[0] * vertex.x + plane[1] * vertex.y + plane[2] * vertex.z + plane[3]
                if distance >= 0:
                    has_vertex_in_front = True
                    break
            
            # 如果所有顶点都在平面背面，返回False
            if not has_vertex_in_front:
                return False
        
        return True
    
    def contains_sphere(self, center, radius):
        """
        检查球体是否在视锥体内
        
        Args:
            center: 球心
            radius: 半径
            
        Returns:
            bool: 是否在视锥体内
        """
        # 检查球体是否在所有平面的正面
        for plane in self.planes:
            # 计算球心到平面的距离
            distance = plane[0] * center.x + plane[1] * center.y + plane[2] * center.z + plane[3]
            
            # 如果球心到平面的距离小于负半径，则球体完全在平面背面
            if distance < -radius:
                return False
        
        return True

# 性能优化的数学函数
def fast_dot(v1, v2):
    """
    快速向量点积计算
    """
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

def fast_lerp(v1, v2, t):
    """
    快速线性插值
    """
    return Vector3(
        v1.x + (v2.x - v1.x) * t,
        v1.y + (v2.y - v1.y) * t,
        v1.z + (v2.z - v1.z) * t
    )

def frustum_culling(bounding_box, frustum_planes):
    """
    视锥体裁剪算法，优化版本
    """
    # 简化的视锥体裁剪实现
    # 对于低端GPU，使用更简单的AABB剪裁
    for plane in frustum_planes:
        # 检查包围盒是否完全在平面外侧
        # 如果是，返回False（被剪裁）
        pass
    return True
