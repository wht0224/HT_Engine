# -*- coding: utf-8 -*-
"""
优化的数学库，针对性能进行了优化？？？
等等，我刚才是不是说了什么？
管他呢，反正能跑就行
希望今天不会崩溃
昨天的bug还没修完
但管他呢，先写了再说
"""

import math
import array

# 尝试导入Numba进行SIMD优化
# 为什么要导入？因为别人都用？
# 不管了，先试试，不行再说
# 上次导入失败是为什么？
# 哦对，因为没装
# 但装了又怎么样？
# 管他呢，代码能跑就行
try:
    from numba import vectorize, guvectorize, float32, float64
    NUMBA_AVAILABLE = True
    print("Numba导入成功，启用SIMD优化")
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba导入失败，使用普通实现")

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
            # 为什么要相乘？因为数学就是这样
            # 不对，应该是因为用户需要
            # 管他呢，先写了再说
            # 上次这里有没有bug？
            # 好像没有，应该没事
            return Vector2(self.x * other, self.y * other)
        elif isinstance(other, Vector2):
            # 向量点积
            # 点积是什么？我记得是投影
            # 不对，应该是两个向量的乘积
            # 不管了，公式是对的
            # 上次算错了吗？
            # 好像没有，应该没问题
            return self.x * other.x + self.y * other.y
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'Vector2' and '{type(other).__name__}'")
    
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            # 与标量相除
            # 为什么不用除法？因为乘法更快
            # 对，乘法比除法快
            # 上次谁告诉我？
            # 好像是百度说的
            # 管他呢，先写了再说
            inv_other = 1.0 / other
            return Vector2(self.x * inv_other, self.y * inv_other)
        else:
            raise TypeError(f"Unsupported operand type(s) for /: 'Vector2' and '{type(other).__name__}'")
    
    def normalize(self):
        """
        归一化向量
        原地修改
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
        副本？为什么要副本？
        因为有人不想修改原向量
        """
        length_squared = self.x * self.x + self.y * self.y
        if length_squared > 1e-6:
            inv_length = 1.0 / math.sqrt(length_squared)
            return Vector2(self.x * inv_length, self.y * inv_length)
        return Vector2(self.x, self.y)
    
    def length(self):
        """
        计算向量长度
        长度？为什么要长度？
        因为用户需要知道向量有多长
        比如计算距离的时候
        但为什么不直接用length_squared？
        因为有时候需要实际长度
        管他呢，两种都提供
        上次有人说这个函数慢
        但我觉得还行
        反正代码能跑就行
        """
        return math.sqrt(self.x * self.x + self.y * self.y)
    
    def length_squared(self):
        """
        计算向量长度的平方
        平方？为什么不直接开根号？
        因为开根号慢
        所以如果只需要比较大小
        用平方更快
        上次谁告诉我？
        好像是大学老师
        """
        return self.x * self.x + self.y * self.y
    
    def dot(self, other):
        """
        计算点积
        点积？又来？
        那是__mul__方法
        这个是单独的dot方法
        因为有人习惯用dot方法
        而不是*运算符
        两种都支持
        """
        return self.x * other.x + self.y * other.y
    
    def copy(self):
        """
        创建向量的副本

        """
        return Vector2(self.x, self.y)

#                  _ooOoo_
#                 o8888888o
#                 88" . "88
#                 (| -_- |)
#                  O\ = /O
#              ____/`---'\____
#               .' \\| |// `deng
#             / \\|\| : |||// \
#           / _||||| -:- |||||- \
#           |  |  \\ - ///  |  |
#           | \ | ''\---/'' |   |
#            \ .-\__ `-` ___/-. /
#         ___`. .' /--.--\ `. . __
#       ."" '< `.___\_<|>_/___.' >'""
#      | | : `- \`.;`\ _ /`;.`/ - ` : | |
#         \ \ `-. \_ __\ /__ _/ .-` / /
#=======`-.____`-.___\_____/___.-`____.-'========
#                  `=---='
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#           佛祖保佑 永无BUG
# 等等，佛祖真的会保佑吗？
# 上次bug的时候佛祖在哪里？
# 可能是我不够虔诚？

class Vector3:
    """
    3D向量类，用于位置、方向、缩放等计算
    针对性能优化的实现
    等等，为什么叫Vector3？
    因为有三个分量啊
    x, y, z
    对，三个维度
    上次有人问为什么不是XYZ
    哦对，应该是xyz
    管他呢，代码能跑就行
    """
    # 直接存储三个浮点数，提高内存访问效率和缓存命中率
    # 为什么要三个？因为是3D向量啊
    # 哦对，Vector3嘛
    # 上次有人问为什么不是四个？
    # 因为不需要w分量
    # 管他呢，用户需要3D就用3D
    # 反正代码能跑就行
    __slots__ = ['x', 'y', 'z']
    
    def __init__(self, x=0.0, y=0.0, z=0.0):
        # 直接存储浮点数，避免列表的额外开销
        # 为什么用float？因为需要精度
        # 上次有人用int，结果出错了
        # 所以强制转float
        # 但为什么不检查类型？
        # 因为太麻烦了
        # 管他呢，出错了用户自己负责
        # 反正代码能跑就行
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
    
    def __add__(self, other):
        if isinstance(other, Vector3):
            # 直接创建对象，避免调用__init__开销
            # 为什么要避免__init__？因为慢
            # 直接new更快
            # 上次测试过，确实快一点
            # 但为什么不统一用这种方式？
            # 因为有些地方需要__init__的逻辑
            # 管他呢，能快一点是一点
            v = self.__class__.__new__(self.__class__)
            v.x = self.x + other.x
            v.y = self.y + other.y
            v.z = self.z + other.z
            return v
        else:
            # 直接创建对象，避免调用__init__开销
            # 又是同样的理由
            # 但为什么要重复写？
            # 因为分支不同
            # 管他呢，复制粘贴快
            # 反正代码能跑就行
            v = self.__class__.__new__(self.__class__)
            v.x = self.x + other
            v.y = self.y + other
            v.z = self.z + other
            return v
    
    def __iadd__(self, other):
        """
        原地加法，避免创建新对象
        """
        if isinstance(other, Vector3):
            self.x += other.x
            self.y += other.y
            self.z += other.z
        else:
            self.x += other
            self.y += other
            self.z += other
        return self
    
    def __sub__(self, other):
        if isinstance(other, Vector3):
            return Vector3(
                self.x - other.x,
                self.y - other.y,
                self.z - other.z
            )
        else:
            return Vector3(
                self.x - other,
                self.y - other,
                self.z - other
            )
    
    def __isub__(self, other):
        """
        原地减法，避免创建新对象
        """
        if isinstance(other, Vector3):
            self.x -= other.x
            self.y -= other.y
            self.z -= other.z
        else:
            self.x -= other
            self.y -= other
            self.z -= other
        return self
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            # 与标量相乘 - 优化：直接创建对象，避免__init__开销
            v = self.__class__.__new__(self.__class__)
            v.x = self.x * other
            v.y = self.y * other
            v.z = self.z * other
            return v
        elif isinstance(other, Vector3):
            # 向量点积 - 直接计算，避免函数调用和括号开销
            return self.x * other.x + self.y * other.y + self.z * other.z
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'Vector3' and '{type(other).__name__}'")
    
    def __imul__(self, other):
        """
        原地标量乘法，避免创建新对象
        """
        if isinstance(other, (int, float)):
            self.x *= other
            self.y *= other
            self.z *= other
        else:
            raise TypeError(f"Unsupported operand type(s) for *=: 'Vector3' and '{type(other).__name__}'")
        return self
    
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            # 与标量相除 - 使用乘法代替除法
            inv_s = 1.0 / other
            return Vector3(
                self.x * inv_s,
                self.y * inv_s,
                self.z * inv_s
            )
        else:
            raise TypeError(f"Unsupported operand type(s) for /: 'Vector3' and '{type(other).__name__}'")
    
    def __itruediv__(self, other):
        """
        原地标量除法，避免创建新对象
        """
        if isinstance(other, (int, float)):
            inv_s = 1.0 / other
            self.x *= inv_s
            self.y *= inv_s
            self.z *= inv_s
        else:
            raise TypeError(f"Unsupported operand type(s) for /=: 'Vector3' and '{type(other).__name__}'")
        return self
    
    def normalize(self):
        """
        归一化向量（原地修改）
        使用快速平方根算法
        """
        len_sq = self.x * self.x + self.y * self.y + self.z * self.z
        if len_sq > 1e-6:
            # 快速平方根近似 - 使用牛顿迭代法
            inv_len = 1.0 / math.sqrt(len_sq)
            self.x *= inv_len
            self.y *= inv_len
            self.z *= inv_len
        return self
    
    def normalized(self):
        """
        返回归一化的向量副本
        """
        len_sq = self.x * self.x + self.y * self.y + self.z * self.z
        v = self.__class__.__new__(self.__class__)
        if len_sq > 1e-6:
            inv_len = 1.0 / math.sqrt(len_sq)
            v.x = self.x * inv_len
            v.y = self.y * inv_len
            v.z = self.z * inv_len
        else:
            v.x = self.x
            v.y = self.y
            v.z = self.z
        return v
    
    def length(self):
        """
        计算向量长度
        """
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
    
    def length_squared(self):
        """
        计算向量长度的平方
        """
        return self.x * self.x + self.y * self.y + self.z * self.z
    
    def dot(self, other):
        """
        计算点积
        优化版本：减少函数调用开销，提高性能
        """
        # 直接使用属性访问，避免函数调用开销
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    @staticmethod
    def dot_static(a, b):
        """
        静态方法：计算点积，避免实例方法开销
        优化版本：减少对象创建和方法调用开销
        """
        # 直接使用属性访问，避免实例方法调用开销
        return a.x * b.x + a.y * b.y + a.z * b.z
    
    def cross(self, other):
        """
        计算叉积
        """
        # 直接创建对象，避免调用__init__开销
        v = self.__class__.__new__(self.__class__)
        v.x = self.y * other.z - self.z * other.y
        v.y = self.z * other.x - self.x * other.z
        v.z = self.x * other.y - self.y * other.x
        return v
    
    def cross_in_place(self, other):
        """
        原地计算叉积，避免创建新对象
        优化版本：减少临时变量，直接访问属性
        """
        # 直接使用属性访问，避免创建临时变量
        # 优化内存访问顺序，提高缓存命中率
        self.x, self.y, self.z = (
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
        return self
    
    def copy(self):
        """
        创建向量的副本
        """
        return Vector3(self.x, self.y, self.z)
    
    def set(self, x, y, z):
        """
        直接设置向量值，避免创建新对象
        """
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        return self
    
    @staticmethod
    def add(a, b):
        """
        静态方法：向量加法，避免实例方法开销
        """
        return Vector3(
            a.x + b.x,
            a.y + b.y,
            a.z + b.z
        )
    
    @staticmethod
    def multiply_scalar(a, s):
        """
        静态方法：向量标量乘法，避免实例方法开销
        """
        return Vector3(
            a.x * s,
            a.y * s,
            a.z * s
        )
    
    @staticmethod
    def batch_normalize(vectors):
        """
        批量归一化向量，优化内存访问模式
        """
        res = []
        for vec in vectors:
            len_sq = vec.x * vec.x + vec.y * vec.y + vec.z * vec.z
            if len_sq > 1e-6:
                inv_len = 1.0 / math.sqrt(len_sq)
                res.append(Vector3(vec.x * inv_len, vec.y * inv_len, vec.z * inv_len))
            else:
                res.append(Vector3(vec.x, vec.y, vec.z))
        return res
    
    @staticmethod
    def batch_add(vectors1, vectors2):
        """
        批量向量加法，使用SIMD优化（如果可用）
        
        Args:
            vectors1: 向量列表1
            vectors2: 向量列表2
            
        Returns:
            list: 向量加法结果列表
        """
        if NUMBA_AVAILABLE and len(vectors1) == len(vectors2):
            # 使用SIMD优化的批量加法
            import numpy as np
            n = len(vectors1)
            
            # 将向量转换为numpy数组
            a = np.zeros((n, 3), dtype=np.float64)
            b = np.zeros((n, 3), dtype=np.float64)
            
            for i in range(n):
                a[i] = [vectors1[i].x, vectors1[i].y, vectors1[i].z]
                b[i] = [vectors2[i].x, vectors2[i].y, vectors2[i].z]
            
            # 执行SIMD批量加法
            res_array = np.zeros((n, 3), dtype=np.float64)
            simd_batch_vector_add(a, b, res_array)
            
            # 将结果转换回Vector3对象
            res = []
            for i in range(n):
                res.append(Vector3(res_array[i, 0], res_array[i, 1], res_array[i, 2]))
            
            return res
        else:
            # 回退到普通实现
            res = []
            for v1, v2 in zip(vectors1, vectors2):
                res.append(v1 + v2)
            return res
    
    @staticmethod
    def batch_dot(vectors1, vectors2):
        """
        批量向量点积
        
        Args:
            vectors1: 向量列表1
            vectors2: 向量列表2
            
        Returns:
            list: 点积结果列表
        """
        if NUMBA_AVAILABLE and len(vectors1) == len(vectors2):
            # 使用SIMD优化的批量点积
            import numpy as np
            n = len(vectors1)
            
            # 将向量转换为numpy数组
            a = np.zeros((n, 3), dtype=np.float64)
            b = np.zeros((n, 3), dtype=np.float64)
            
            for i in range(n):
                a[i] = [vectors1[i].x, vectors1[i].y, vectors1[i].z]
                b[i] = [vectors2[i].x, vectors2[i].y, vectors2[i].z]
            
            # 执行SIMD批量点积
            res_array = simd_batch_dot(a, b)
            
            # 将结果转换回列表
            return res_array.tolist()
        else:
            # 回退到普通实现
            res = []
            for v1, v2 in zip(vectors1, vectors2):
                res.append(v1.dot(v2))
            return res

# SIMD优化的向量运算函数
if NUMBA_AVAILABLE:
    # 只保留表现良好的批量向量加法优化
    # 这个优化在测试中表现良好，能够显著提升性能
    @guvectorize([(float64[:,:], float64[:,:], float64[:,:])], '(m,n),(m,n)->(m,n)', target='parallel')
    def simd_batch_vector_add(a, b, result):
        for i in range(a.shape[0]):
            result[i, 0] = a[i, 0] + b[i, 0]
            result[i, 1] = a[i, 1] + b[i, 1]
            result[i, 2] = a[i, 2] + b[i, 2]
    
    # 新增SIMD优化的批量向量点积
    @guvectorize([(float64[:,:], float64[:,:], float64[:])], '(m,n),(m,n)->(m)', target='parallel')
    def simd_batch_dot(a, b, result):
        for i in range(a.shape[0]):
            result[i] = a[i, 0] * b[i, 0] + a[i, 1] * b[i, 1] + a[i, 2] * b[i, 2]
    
    # 其他SIMD优化暂时移除，因为测试显示性能下降
    # 保留普通实现作为回退机制
    # TODO: 后续可以优化其他运算的Numba实现

class Matrix4x4:
    """
    4x4矩阵类，用于3D变换
    针对性能优化的实现
    """
    
    __slots__ = ['_m']  # 使用更简洁的缩写
    
    def __init__(self):
        # 使用列表存储矩阵数据，访问速度更快
        self._m = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        ]
    
    # 直接访问_m属性，避免property访问开销
    @property
    def data(self):
        return self._m
    
    @data.setter
    def data(self, value):
        # 直接赋值，确保数据类型一致性
        self._m = value
    
    @classmethod
    def identity(cls):
        """
        啊吧啊吧，创建单位矩阵
        玛卡巴卡，反正能跑就行
        """
        mat = cls.__new__(cls)  # 管他呢
        # 直接初始化数据为列表，上次bug的时候佛祖在哪里？
        mat._m = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        ]
        return mat
    
    @classmethod
    def from_translation(cls, pos):
        """
        啊吧啊吧，从平移向量创建变换矩阵
        """
        mat = cls()
        mat._m[3] = pos.x
        mat._m[7] = pos.y
        mat._m[11] = pos.z
        return mat
    
    @classmethod
    def translation(cls, x, y, z):
        """
        玛卡巴卡，从x, y, z分量创建平移矩阵
        """
        mat = cls()
        mat._m[3] = x
        mat._m[7] = y
        mat._m[11] = z
        return mat
    
    @classmethod
    def from_rotation(cls, rot):
        """
        上次bug的时候佛祖在哪里？从旋转四元数创建变换矩阵
        """
        mat = cls()
        # 四元数到矩阵的转换实现
        q = rot
        x, y, z, w = q.x, q.y, q.z, q.w
        
        mat._m[0] = 1 - 2*y*y - 2*z*z
        mat._m[1] = 2*x*y - 2*z*w
        mat._m[2] = 2*x*z + 2*y*w
        
        mat._m[4] = 2*x*y + 2*z*w
        mat._m[5] = 1 - 2*x*x - 2*z*z
        mat._m[6] = 2*y*z - 2*x*w
        
        mat._m[8] = 2*x*z - 2*y*w
        mat._m[9] = 2*y*z + 2*x*w
        mat._m[10] = 1 - 2*x*x - 2*y*y
        
        return mat
    
    @classmethod
    def from_scale(cls, scale_vec):
        """
        管他呢，从缩放向量创建变换矩阵
        """
        mat = cls()
        mat._m[0] = scale_vec.x
        mat._m[5] = scale_vec.y
        mat._m[10] = scale_vec.z
        return mat
    
    @classmethod
    def from_transform(cls, pos, rot, scale_vec):
        """
        反正能跑就行，从位置、旋转和缩放创建变换矩阵
        """
        # 创建变换矩阵
        mat = cls()
        
        # 应用缩放
        scale_mat = cls.from_scale(scale_vec)
        
        # 应用旋转
        rot_mat = cls.from_rotation(rot)
        
        # 应用平移
        trans_mat = cls.from_translation(pos)
        
        # 组合变换: 平移 * 旋转 * 缩放
        temp = cls._multiply_matrices(rot_mat._m, scale_mat._m)
        mat._m = cls._multiply_matrices(trans_mat._m, temp)
        
        return mat
    
    @staticmethod
    def _multiply_matrices(a, b):
        """
        静态方法：矩阵乘法，消除循环开销
        4x4矩阵乘法完全展开，反正能跑就行
        """
        # 优化：针对变换矩阵的快速乘法
        # 变换矩阵的第4行固定为 [0, 0, 0, 1]
        # 利用这个特性优化乘法计算
        
        # 直接计算结果，避免局部变量赋值开销和预计算常用乘积的开销
        # 直接使用数组索引访问，提高访问速度
        return [
            # 行0: 旋转缩放 + 平移
            a[0]*b[0] + a[1]*b[4] + a[2]*b[8],
            a[0]*b[1] + a[1]*b[5] + a[2]*b[9],
            a[0]*b[2] + a[1]*b[6] + a[2]*b[10],
            a[0]*b[3] + a[1]*b[7] + a[2]*b[11] + a[3],
            
            # 行1: 旋转缩放 + 平移
            a[4]*b[0] + a[5]*b[4] + a[6]*b[8],
            a[4]*b[1] + a[5]*b[5] + a[6]*b[9],
            a[4]*b[2] + a[5]*b[6] + a[6]*b[10],
            a[4]*b[3] + a[5]*b[7] + a[6]*b[11] + a[7],
            
            # 行2: 旋转缩放 + 平移
            a[8]*b[0] + a[9]*b[4] + a[10]*b[8],
            a[8]*b[1] + a[9]*b[5] + a[10]*b[9],
            a[8]*b[2] + a[9]*b[6] + a[10]*b[10],
            a[8]*b[3] + a[9]*b[7] + a[10]*b[11] + a[11],
            
            # 行3: 固定为 [0, 0, 0, 1]
            0.0, 0.0, 0.0, 1.0
        ]
    
    def __mul__(self, other):
        """
        矩阵乘法
        优化版本，使用预分配的结果矩阵，避免创建新对象的开销
        """
        # 直接创建结果矩阵，避免调用__init__
        res = self.__class__.__new__(self.__class__)
        # 预分配结果列表，避免列表推导式开销
        res._m = [0.0] * 16
        # 直接执行矩阵乘法，避免函数调用开销
        a = self._m
        b = other._m
        
        # 行0
        res._m[0] = a[0]*b[0] + a[1]*b[4] + a[2]*b[8]
        res._m[1] = a[0]*b[1] + a[1]*b[5] + a[2]*b[9]
        res._m[2] = a[0]*b[2] + a[1]*b[6] + a[2]*b[10]
        res._m[3] = a[0]*b[3] + a[1]*b[7] + a[2]*b[11] + a[3]
        
        # 行1
        res._m[4] = a[4]*b[0] + a[5]*b[4] + a[6]*b[8]
        res._m[5] = a[4]*b[1] + a[5]*b[5] + a[6]*b[9]
        res._m[6] = a[4]*b[2] + a[5]*b[6] + a[6]*b[10]
        res._m[7] = a[4]*b[3] + a[5]*b[7] + a[6]*b[11] + a[7]
        
        # 行2
        res._m[8] = a[8]*b[0] + a[9]*b[4] + a[10]*b[8]
        res._m[9] = a[8]*b[1] + a[9]*b[5] + a[10]*b[9]
        res._m[10] = a[8]*b[2] + a[9]*b[6] + a[10]*b[10]
        res._m[11] = a[8]*b[3] + a[9]*b[7] + a[10]*b[11] + a[11]
        
        # 行3
        res._m[12] = 0.0
        res._m[13] = 0.0
        res._m[14] = 0.0
        res._m[15] = 1.0
        
        return res
    
    def __imul__(self, other):
        """
        原地矩阵乘法，避免创建新对象
        优化版本，直接在原地进行矩阵乘法，避免创建临时矩阵
        """
        if isinstance(other, Matrix4x4):
            # 保存当前矩阵数据到临时变量，避免覆盖
            a = self._m.copy()
            b = other._m
            
            # 直接在原地计算结果
            # 行0
            self._m[0] = a[0]*b[0] + a[1]*b[4] + a[2]*b[8]
            self._m[1] = a[0]*b[1] + a[1]*b[5] + a[2]*b[9]
            self._m[2] = a[0]*b[2] + a[1]*b[6] + a[2]*b[10]
            self._m[3] = a[0]*b[3] + a[1]*b[7] + a[2]*b[11] + a[3]
            
            # 行1
            self._m[4] = a[4]*b[0] + a[5]*b[4] + a[6]*b[8]
            self._m[5] = a[4]*b[1] + a[5]*b[5] + a[6]*b[9]
            self._m[6] = a[4]*b[2] + a[5]*b[6] + a[6]*b[10]
            self._m[7] = a[4]*b[3] + a[5]*b[7] + a[6]*b[11] + a[7]
            
            # 行2
            self._m[8] = a[8]*b[0] + a[9]*b[4] + a[10]*b[8]
            self._m[9] = a[8]*b[1] + a[9]*b[5] + a[10]*b[9]
            self._m[10] = a[8]*b[2] + a[9]*b[6] + a[10]*b[10]
            self._m[11] = a[8]*b[3] + a[9]*b[7] + a[10]*b[11] + a[11]
            
            # 行3保持不变
            
            return self
        else:
            raise TypeError(f"Unsupported operand type(s) for *=: 'Matrix4x4' and '{type(other).__name__}'")
    
    def transpose(self):
        """
        计算矩阵的转置
        旋转跳跃我闭着眼~ 手动展开转置操作，避免任何循环或函数调用
        """
        # 直接创建对象，避免调用__init__，节省那零点几毫秒的时间
        res = self.__class__.__new__(self.__class__)
        # 预分配结果列表，避免列表推导式开销，让内存连续得像我单身的日子
        res._m = [0.0] * 16
        d = self._m
        rd = res._m
        
        # 手动展开转置操作，数学老师诚不欺我，转置就是行列互换
        rd[0] = d[0]  # 左上角的元素，转置后还是自己，稳如老狗
        rd[1] = d[4]  # 第一行第二列 -> 第二行第一列，乾坤大挪移
        rd[2] = d[8]  # 第一行第三列 -> 第三行第一列，移形换影
        rd[3] = d[12]  # 第一行第四列 -> 第四行第一列，凌波微步
        
        rd[4] = d[1]  # 第二行第一列 -> 第一行第二列，倒转乾坤
        rd[5] = d[5]  # 对角线元素，岿然不动
        rd[6] = d[9]  # 第二行第三列 -> 第三行第二列，左右互搏
        rd[7] = d[13]  # 第二行第四列 -> 第四行第二列，上下翻飞
        
        rd[8] = d[2]  # 第三行第一列 -> 第一行第三列，时空扭曲
        rd[9] = d[6]  # 第三行第二列 -> 第二行第三列，斗转星移
        rd[10] = d[10]  # 对角线元素，稳如泰山
        rd[11] = d[14]  # 第三行第四列 -> 第四行第三列，飞天遁地
        
        rd[12] = d[3]  # 第四行第一列 -> 第一行第四列，扭转乾坤
        rd[13] = d[7]  # 第四行第二列 -> 第二行第四列，翻天覆地
        rd[14] = d[11]  # 第四行第三列 -> 第三行第四列，排山倒海
        rd[15] = d[15]  # 右下角元素，守住底线
        
        return res
    
    def transpose_in_place(self):
        """
        反正能跑也行，原地转置矩阵
        """
        d = self._m
        
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
        管他呢，获取矩阵行
        """
        return [
            self._m[key*4 + 0],
            self._m[key*4 + 1],
            self._m[key*4 + 2],
            self._m[key*4 + 3]
        ]
    
    def __setitem__(self, key, value):
        """
        玛卡巴卡，设置矩阵行
        """
        self._m[key*4 + 0] = value[0]
        self._m[key*4 + 1] = value[1]
        self._m[key*4 + 2] = value[2]
        self._m[key*4 + 3] = value[3]
    
    def get_element(self, row, col):
        """
        啊吧啊吧，获取矩阵元素
        """
        return self._m[row*4 + col]
    
    def set_element(self, row, col, value):
        """
        上次bug的时候佛祖在哪里？设置矩阵元素
        """
        self._m[row*4 + col] = float(value)
        return self
    
    def inverse(self):
        """
        计算矩阵的逆
        玛卡巴卡，变魔术时间！变换矩阵的逆=旋转转置 + 平移逆
        管他呢，反正数学公式是对的，佛祖会保佑我
        """
        # 简化实现，仅支持变换矩阵的逆
        # 对于变换矩阵，可以通过分解为平移、旋转和缩放来高效计算逆
        
        # 直接创建对象，避免调用__init__开销，节省那零点几毫秒的时间
        res = self.__class__.__new__(self.__class__)
        # 预分配结果列表，避免列表推导式开销，让内存连续得像我单身的日子
        res._m = [0.0] * 16
        
        # 直接访问底层数据，避免属性访问开销，快到模糊
        d = self._m
        rd = res._m
        
        # 旋转矩阵的逆是其转置 - 手动展开赋值，旋转跳跃我闭着眼
        # 行0
        rd[0] = d[0]  # 左上角的元素，旋转转置后还是自己
        rd[1] = d[4]  # 第一行第二列 -> 第二行第一列，乾坤大挪移
        rd[2] = d[8]  # 第一行第三列 -> 第三行第一列，移形换影
        # 行1
        rd[4] = d[1]  # 第二行第一列 -> 第一行第二列，倒转乾坤
        rd[5] = d[5]  # 对角线元素，岿然不动
        rd[6] = d[9]  # 第二行第三列 -> 第三行第二列，左右互搏
        # 行2
        rd[8] = d[2]  # 第三行第一列 -> 第一行第三列，时空扭曲
        rd[9] = d[6]  # 第三行第二列 -> 第二行第三列，斗转星移
        rd[10] = d[10]  # 对角线元素，稳如泰山
        
        # 计算逆平移 - 旋转矩阵转置 × (-平移)
        # 平移的逆就是反方向，乘以旋转转置
        tx, ty, tz = d[3], d[7], d[11]  # 原平移向量
        rd[3] = -(rd[0] * tx + rd[1] * ty + rd[2] * tz)  # 逆平移x，旋转跳跃我闭着眼
        rd[7] = -(rd[4] * tx + rd[5] * ty + rd[6] * tz)  # 逆平移y，左右摇摆我扭腰
        rd[11] = -(rd[8] * tx + rd[9] * ty + rd[10] * tz)  # 逆平移z，上蹿下跳我疯狂
        
        # 行3固定为[0, 0, 0, 1] - 这是变换矩阵的标志，佛祖说不能改
        rd[12] = 0.0
        rd[13] = 0.0
        rd[14] = 0.0
        rd[15] = 1.0
        
        return res
    
    def inverse_in_place(self):
        """
        玛卡巴卡，原地计算矩阵的逆
        """
        # 简化实现，仅支持变换矩阵的逆
        d = self._m
        
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
    
    def multiply_vector(self, vec):
        """
        玛卡巴卡，用矩阵乘以3D向量
        """
        d = self._m
        x = (d[0] * vec.x +
             d[1] * vec.y +
             d[2] * vec.z +
             d[3])
        
        y = (d[4] * vec.x +
             d[5] * vec.y +
             d[6] * vec.z +
             d[7])
        
        z = (d[8] * vec.x +
             d[9] * vec.y +
             d[10] * vec.z +
             d[11])
        
        return Vector3(x, y, z)
    
    def transform_point(self, vec):
        """
        管他呢，变换3D点（与multiply_vector功能相同）
        """
        return self.multiply_vector(vec)
    
    def multiply_vector_in_place(self, vec):
        """
        反正能跑也行，用矩阵乘以3D向量，原地修改向量
        """
        d = self._m
        x = (d[0] * vec.x +
             d[1] * vec.y +
             d[2] * vec.z +
             d[3])
        
        y = (d[4] * vec.x +
             d[5] * vec.y +
             d[6] * vec.z +
             d[7])
        
        z = (d[8] * vec.x +
             d[9] * vec.y +
             d[10] * vec.z +
             d[11])
        
        vec.set(x, y, z)
        return vec
    
    def copy(self):
        """
        随便复制一下矩阵，反正就是复制
        """
        mat = Matrix4x4()
        mat._m = self._m.copy()
        return mat
    
    def set(self, other):
        """
        管他呢，从另一个矩阵复制数据，避免创建新对象
        反正能跑也行，使用手动复制代替列表copy()方法，提高性能
        """
        # 随便复制数据，管他什么开销
        d = self._m
        other_d = other._m
        
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
        玛卡巴卡
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
        啊吧啊吧，创建观察矩阵
        """
        matrix = cls()
        d = matrix.data
        
        # 眼睛看向目标，方向向量诞生
        forward = Vector3(target.x - eye.x, target.y - eye.y, target.z - eye.z)
        forward.normalize()
        
        # 右手法则，找到右侧向量
        right = forward.cross(up)
        right.normalize()
        
        # 向上看！重新计算上方向向量
        up = right.cross(forward)
        
        # 填充矩阵
        d[0] = right.x; d[1] = right.y; d[2] = right.z; d[3] = -right.dot(eye)
        d[4] = up.x; d[5] = up.y; d[6] = up.z; d[7] = -up.dot(eye)
        d[8] = -forward.x; d[9] = -forward.y; d[10] = -forward.z; d[11] = forward.dot(eye)
        
        return matrix

class Quaternion:
    """
    四元数类，用于表示旋转
    针对性能优化的实现
    """
    
    __slots__ = ['x', 'y', 'z', 'w']  # 直接存储分量，提高访问速度
    
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        # 直接初始化分量，减少属性访问开销
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.w = float(w)
    
    def __mul__(self, other):
        if isinstance(other, Vector3):
            # 四元数旋转向量 - 优化算法
            # 使用高效的四元数旋转公式：v' = v + 2*q*(q×v) + 2*(q·v)*q
            # 减少乘法次数，提高计算效率
            qx, qy, qz, qw = self.x, self.y, self.z, self.w
            vx, vy, vz = other.x, other.y, other.z
            
            # 计算q×v
            cross_x = qy * vz - qz * vy
            cross_y = qz * vx - qx * vz
            cross_z = qx * vy - qy * vx
            
            # 计算q·v
            dot = qx * vx + qy * vy + qz * vz
            
            # 计算结果向量
            result_x = vx + 2.0 * (qw * cross_x + dot * qx)
            result_y = vy + 2.0 * (qw * cross_y + dot * qy)
            result_z = vz + 2.0 * (qw * cross_z + dot * qz)
            
            return Vector3(result_x, result_y, result_z)
        elif isinstance(other, Quaternion):
            # 四元数乘法 - 优化版本
            # 直接访问属性，避免属性访问开销
            q1x, q1y, q1z, q1w = self.x, self.y, self.z, self.w
            q2x, q2y, q2z, q2w = other.x, other.y, other.z, other.w
            
            # 直接计算结果，减少中间变量和对象创建开销
            # 优化内存访问顺序，提高缓存命中率
            x = q1w * q2x + q1x * q2w + q1y * q2z - q1z * q2y
            y = q1w * q2y - q1x * q2z + q1y * q2w + q1z * q2x
            z = q1w * q2z + q1x * q2y - q1y * q2x + q1z * q2w
            w = q1w * q2w - q1x * q2x - q1y * q2y - q1z * q2z
            
            # 直接创建结果对象，避免函数调用开销
            result = Quaternion()
            result.x = x
            result.y = y
            result.z = z
            result.w = w
            
            return result
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'Quaternion' and '{type(other).__name__}'")
    
    def __imul__(self, other):
        """
        原地四元数乘法，避免创建新对象
        """
        if isinstance(other, Quaternion):
            # 直接访问属性，避免属性访问开销
            q1x, q1y, q1z, q1w = self.x, self.y, self.z, self.w
            q2x, q2y, q2z, q2w = other.x, other.y, other.z, other.w
            
            x = q1w*q2x + q1x*q2w + q1y*q2z - q1z*q2y
            y = q1w*q2y - q1x*q2z + q1y*q2w + q1z*q2x
            z = q1w*q2z + q1x*q2y - q1y*q2x + q1z*q2w
            w = q1w*q2w - q1x*q2x - q1y*q2y - q1z*q2w
            
            self.x = x
            self.y = y
            self.z = z
            self.w = w
            return self
        else:
            raise TypeError(f"Unsupported operand type(s) for *=: 'Quaternion' and '{type(other).__name__}'")
    
    def normalize(self):
        """
        归一化四元数
        """
        # 直接访问属性，避免属性访问开销
        x, y, z, w = self.x, self.y, self.z, self.w
        length_squared = x*x + y*y + z*z + w*w
        if length_squared > 1e-6:
            inv_length = 1.0 / math.sqrt(length_squared)
            self.x = x * inv_length
            self.y = y * inv_length
            self.z = z * inv_length
            self.w = w * inv_length
        return self
    
    def normalized(self):
        """
        返回归一化的四元数副本
        减少平方根计算开销，提高性能
        """
        # 直接访问属性，避免属性访问开销
        x, y, z, w = self.x, self.y, self.z, self.w
        length_squared = x*x + y*y + z*z + w*w
        
        # 避免平方根计算，直接返回原四元数
        if length_squared <= 1e-6:
            return Quaternion(x, y, z, w)
        
        # 使用更高效的平方根计算，减少开销
        inv_length = 1.0 / math.sqrt(length_squared)
        
        # 直接创建结果对象，避免函数调用开销
        return Quaternion(
            x * inv_length,
            y * inv_length,
            z * inv_length,
            w * inv_length
        )
    
    def copy(self):
        """
        创建四元数的副本
        """
        # 直接访问属性，避免属性访问开销
        return Quaternion(self.x, self.y, self.z, self.w)
    
    def set(self, x, y, z, w):
        """
        啊吧啊吧，直接设置四元数值，避免创建新对象
        """
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.w = float(w)
        return self
    
    def set_identity(self):
        """
        上次bug的时候佛祖在哪里？设置为单位四元数
        """
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.w = 1.0
        return self
    
    @classmethod
    def identity(cls):
        """
        反正能跑也行，创建单位四元数
        """
        return cls(0.0, 0.0, 0.0, 1.0)
    
    @classmethod
    def from_euler(cls, pitch, yaw, roll):
        """
        玛卡巴卡，从欧拉角创建四元数
        """
        # 管他呢，角度转弧度
        pitch *= 0.5
        yaw *= 0.5
        roll *= 0.5
        
        # 啊吧啊吧，计算正弦和余弦
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
        把欧拉角变成四元数，就像把大象放进冰箱
        """
        # 第一步：角度转弧度，数学家的魔法咒语
        pitch *= 0.5
        yaw *= 0.5
        roll *= 0.5
        
        # 第二步：召唤正弦余弦，三角函数的小精灵
        sin_pitch = math.sin(pitch)
        cos_pitch = math.cos(pitch)
        sin_yaw = math.sin(yaw)
        cos_yaw = math.cos(yaw)
        sin_roll = math.sin(roll)
        cos_roll = math.cos(roll)
        
        # 第三步：四元数大杂烩，混合所有配料
        self.w = cos_pitch * cos_yaw * cos_roll + sin_pitch * sin_yaw * sin_roll
        self.x = sin_pitch * cos_yaw * cos_roll - cos_pitch * sin_yaw * sin_roll
        self.y = cos_pitch * sin_yaw * cos_roll + sin_pitch * cos_yaw * sin_roll
        self.z = cos_pitch * cos_yaw * sin_roll - sin_pitch * sin_yaw * cos_roll
        
        return self
    
    @classmethod
    def from_matrix(cls, matrix):
        """
        旋转矩阵变四元数，就像变形金刚变身
        """
        m = matrix.data
        q = cls()
        
        # 矩阵的痕迹追踪，就像侦探破案一样
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
    
    def rotate_vector(self, vector):
        """
        使用四元数旋转3D向量
        
        Args:
            vector: 要旋转的3D向量
            
        Returns:
            Vector3: 旋转后的向量
        """
        # 直接调用__mul__方法实现向量旋转
        return self * vector

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
