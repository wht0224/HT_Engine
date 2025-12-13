/*
 * asm_math.h
 * 汇编优化的数学函数头文件
 * 针对英特尔Core i3-1215U到i9-14900K处理器优化
 */

#ifndef ASM_MATH_H
#define ASM_MATH_H

#ifdef __cplusplus
extern "C" {
#endif

// 向量类型定义
struct Vector3 {
    double x, y, z;
};

struct Quaternion {
    double x, y, z, w;
};

struct Matrix4x4 {
    double data[16];
};

// 向量运算函数

/**
 * 计算两个向量的点积
 * @param v1 第一个向量的double数组
 * @param v2 第二个向量的double数组
 * @return 点积结果
 */
double asm_vec3_dot(const double* v1, const double* v2);

/**
 * Skylake架构优化的向量点积
 * @param v1 第一个向量的double数组
 * @param v2 第二个向量的double数组
 * @return 点积结果
 */
double asm_vec3_dot_skylake(const double* v1, const double* v2);

/**
 * Ice Lake架构优化的向量点积
 * @param v1 第一个向量的double数组
 * @param v2 第二个向量的double数组
 * @return 点积结果
 */
double asm_vec3_dot_ice_lake(const double* v1, const double* v2);

/**
 * Alder Lake架构优化的向量点积
 * @param v1 第一个向量的double数组
 * @param v2 第二个向量的double数组
 * @return 点积结果
 */
double asm_vec3_dot_alder_lake(const double* v1, const double* v2);

/**
 * 计算两个向量的叉积
 * @param v1 第一个向量的double数组
 * @param v2 第二个向量的double数组
 * @param result 结果向量的double数组
 */
void asm_vec3_cross(const double* v1, const double* v2, double* result);

/**
 * Skylake架构优化的向量叉积
 * @param v1 第一个向量的double数组
 * @param v2 第二个向量的double数组
 * @param result 结果向量的double数组
 */
void asm_vec3_cross_skylake(const double* v1, const double* v2, double* result);

/**
 * Ice Lake架构优化的向量叉积
 * @param v1 第一个向量的double数组
 * @param v2 第二个向量的double数组
 * @param result 结果向量的double数组
 */
void asm_vec3_cross_ice_lake(const double* v1, const double* v2, double* result);

/**
 * Alder Lake架构优化的向量叉积
 * @param v1 第一个向量的double数组
 * @param v2 第二个向量的double数组
 * @param result 结果向量的double数组
 */
void asm_vec3_cross_alder_lake(const double* v1, const double* v2, double* result);

/**
 * AMD Zen架构优化的向量叉积
 * @param v1 第一个向量的double数组
 * @param v2 第二个向量的double数组
 * @param result 结果向量的double数组
 */
void asm_vec3_cross_zen(const double* v1, const double* v2, double* result);

/**
 * 计算向量的长度
 * @param v 向量的double数组
 * @return 向量长度
 */
double asm_vec3_length(const double* v);

/**
 * 归一化向量
 * @param v 输入向量的double数组
 * @param result 归一化后的向量的double数组
 */
void asm_vec3_normalize(const double* v, double* result);

/**
 * Skylake架构优化的向量归一化
 * @param v 输入向量的double数组
 * @param result 归一化后的向量的double数组
 */
void asm_vec3_normalize_skylake(const double* v, double* result);

/**
 * Ice Lake架构优化的向量归一化
 * @param v 输入向量的double数组
 * @param result 归一化后的向量的double数组
 */
void asm_vec3_normalize_ice_lake(const double* v, double* result);

/**
 * Alder Lake架构优化的向量归一化
 * @param v 输入向量的double数组
 * @param result 归一化后的向量的double数组
 */
void asm_vec3_normalize_alder_lake(const double* v, double* result);

/**
 * AMD Zen架构优化的向量归一化
 * @param v 输入向量的double数组
 * @param result 归一化后的向量的double数组
 */
void asm_vec3_normalize_zen(const double* v, double* result);

/**
 * 计算向量长度的平方
 * @param v 向量的double数组
 * @return 长度的平方
 */
double asm_vec3_length_squared(const double* v);

// 矩阵运算函数

/**
 * 4x4矩阵乘法
 * @param m1 第一个矩阵的double数组
 * @param m2 第二个矩阵的double数组
 * @param result 结果矩阵的double数组
 */
void asm_mat4_multiply(const double* m1, const double* m2, double* result);

/**
 * Skylake架构优化的4x4矩阵乘法
 * @param m1 第一个矩阵的double数组
 * @param m2 第二个矩阵的double数组
 * @param result 结果矩阵的double数组
 */
void asm_mat4_multiply_skylake(const double* m1, const double* m2, double* result);

/**
 * Ice Lake架构优化的4x4矩阵乘法
 * @param m1 第一个矩阵的double数组
 * @param m2 第二个矩阵的double数组
 * @param result 结果矩阵的double数组
 */
void asm_mat4_multiply_ice_lake(const double* m1, const double* m2, double* result);

/**
 * Alder Lake架构优化的4x4矩阵乘法
 * @param m1 第一个矩阵的double数组
 * @param m2 第二个矩阵的double数组
 * @param result 结果矩阵的double数组
 */
void asm_mat4_multiply_alder_lake(const double* m1, const double* m2, double* result);

/**
 * AMD Zen架构优化的4x4矩阵乘法
 * @param m1 第一个矩阵的double数组
 * @param m2 第二个矩阵的double数组
 * @param result 结果矩阵的double数组
 */
void asm_mat4_multiply_zen(const double* m1, const double* m2, double* result);

/**
 * 矩阵乘以向量
 * @param m 矩阵的double数组
 * @param v 向量的double数组
 * @param result 结果向量的double数组
 */
void asm_mat4_multiply_vector(const double* m, const double* v, double* result);

/**
 * 创建平移矩阵
 * @param x 平移X
 * @param y 平移Y
 * @param z 平移Z
 * @param result 结果矩阵的double数组
 */
void asm_mat4_translate(double x, double y, double z, double* result);

/**
 * 创建旋转矩阵
 * @param angle 旋转角度（弧度）
 * @param axis 旋转轴的double数组
 * @param result 结果矩阵的double数组
 */
void asm_mat4_rotate(double angle, const double* axis, double* result);

/**
 * 创建缩放矩阵
 * @param x 缩放X
 * @param y 缩放Y
 * @param z 缩放Z
 * @param result 结果矩阵的double数组
 */
void asm_mat4_scale(double x, double y, double z, double* result);

// 四元数运算函数

/**
 * 计算两个四元数的点积
 * @param q1 第一个四元数的double数组
 * @param q2 第二个四元数的double数组
 * @return 点积结果
 */
double asm_quat_dot(const double* q1, const double* q2);

/**
 * 归一化四元数
 * @param q 输入四元数的double数组
 * @param result 归一化后的四元数的double数组
 */
void asm_quat_normalize(const double* q, double* result);

/**
 * Skylake架构优化的四元数归一化
 * @param q 输入四元数的double数组
 * @param result 归一化后的四元数的double数组
 */
void asm_quat_normalize_skylake(const double* q, double* result);

/**
 * Ice Lake架构优化的四元数归一化
 * @param q 输入四元数的double数组
 * @param result 归一化后的四元数的double数组
 */
void asm_quat_normalize_ice_lake(const double* q, double* result);

/**
 * Alder Lake架构优化的四元数归一化
 * @param q 输入四元数的double数组
 * @param result 归一化后的四元数的double数组
 */
void asm_quat_normalize_alder_lake(const double* q, double* result);

/**
 * AMD Zen架构优化的四元数归一化
 * @param q 输入四元数的double数组
 * @param result 归一化后的四元数的double数组
 */
void asm_quat_normalize_zen(const double* q, double* result);

/**
 * 计算两个四元数的乘法
 * @param q1 第一个四元数的double数组
 * @param q2 第二个四元数的double数组
 * @param result 结果四元数的double数组
 */
void asm_quat_multiply(const double* q1, const double* q2, double* result);

/**
 * Skylake架构优化的四元数乘法
 * @param q1 第一个四元数的double数组
 * @param q2 第二个四元数的double数组
 * @param result 结果四元数的double数组
 */
void asm_quat_multiply_skylake(const double* q1, const double* q2, double* result);

/**
 * Ice Lake架构优化的四元数乘法
 * @param q1 第一个四元数的double数组
 * @param q2 第二个四元数的double数组
 * @param result 结果四元数的double数组
 */
void asm_quat_multiply_ice_lake(const double* q1, const double* q2, double* result);

/**
 * Alder Lake架构优化的四元数乘法
 * @param q1 第一个四元数的double数组
 * @param q2 第二个四元数的double数组
 * @param result 结果四元数的double数组
 */
void asm_quat_multiply_alder_lake(const double* q1, const double* q2, double* result);

/**
 * AMD Zen架构优化的四元数乘法
 * @param q1 第一个四元数的double数组
 * @param q2 第二个四元数的double数组
 * @param result 结果四元数的double数组
 */
void asm_quat_multiply_zen(const double* q1, const double* q2, double* result);

/**
 * 使用四元数旋转向量
 * @param q 四元数的double数组
 * @param v 输入向量的double数组
 * @param result 旋转后的向量的double数组
 */
void asm_quat_rotate_vector(const double* q, const double* v, double* result);

/**
 * Skylake架构优化的四元数旋转向量
 * @param q 四元数的double数组
 * @param v 输入向量的double数组
 * @param result 旋转后的向量的double数组
 */
void asm_quat_rotate_vector_skylake(const double* q, const double* v, double* result);

/**
 * Ice Lake架构优化的四元数旋转向量
 * @param q 四元数的double数组
 * @param v 输入向量的double数组
 * @param result 旋转后的向量的double数组
 */
void asm_quat_rotate_vector_ice_lake(const double* q, const double* v, double* result);

/**
 * Alder Lake架构优化的四元数旋转向量
 * @param q 四元数的double数组
 * @param v 输入向量的double数组
 * @param result 旋转后的向量的double数组
 */
void asm_quat_rotate_vector_alder_lake(const double* q, const double* v, double* result);

/**
 * AMD Zen架构优化的四元数旋转向量
 * @param q 四元数的double数组
 * @param v 输入向量的double数组
 * @param result 旋转后的向量的double数组
 */
void asm_quat_rotate_vector_zen(const double* q, const double* v, double* result);

// 数学函数

/**
 * 快速平方根倒数
 * @param x 输入值
 * @return 平方根倒数
 */
double asm_fast_inv_sqrt(double x);

/**
 * 快速平方根
 * @param x 输入值
 * @return 平方根
 */
double asm_fast_sqrt(double x);

// CPU特性检测函数

/**
 * 检测CPU是否支持AVX2
 * @return 1 if AVX2 is supported, 0 otherwise
 */
int asm_has_avx2();

/**
 * 检测CPU是否支持AVX-512
 * @return 1 if AVX-512 is supported, 0 otherwise
 */
int asm_has_avx512();

/**
 * 获取CPU架构ID
 * @return CPU架构ID：
 *         0 = 未知
 *         1 = Skylake/Kaby Lake/Coffee Lake (6-10代)
 *         2 = Ice Lake/Tiger Lake/Rocket Lake (11-12代)
 *         3 = Alder Lake/Raptor Lake (12-13代)
 *         4 = Sapphire Rapids (服务器)
 */
int asm_get_cpu_architecture();

// 动态函数调度

/**
 * 初始化动态函数调度
 * 根据CPU架构选择最优的函数实现
 */
void asm_init_dynamic_dispatch();

/**
 * 动态调度的向量点积
 * @param v1 第一个向量的double数组
 * @param v2 第二个向量的double数组
 * @return 点积结果
 */
double asm_dynamic_vec3_dot(const double* v1, const double* v2);

/**
 * 动态调度的向量叉积
 * @param v1 第一个向量的double数组
 * @param v2 第二个向量的double数组
 * @param result 结果向量的double数组
 */
void asm_dynamic_vec3_cross(const double* v1, const double* v2, double* result);

/**
 * 动态调度的向量归一化
 * @param v 输入向量的double数组
 * @param result 归一化后的向量的double数组
 */
void asm_dynamic_vec3_normalize(const double* v, double* result);

/**
 * 动态调度的四元数乘法
 * @param q1 第一个四元数的double数组
 * @param q2 第二个四元数的double数组
 * @param result 结果四元数的double数组
 */
void asm_dynamic_quat_multiply(const double* q1, const double* q2, double* result);

/**
 * 动态调度的四元数旋转向量
 * @param q 四元数的double数组
 * @param v 输入向量的double数组
 * @param result 旋转后的向量的double数组
 */
void asm_dynamic_quat_rotate_vector(const double* q, const double* v, double* result);

/**
 * 动态调度的四元数归一化
 * @param q 输入四元数的double数组
 * @param result 归一化后的四元数的double数组
 */
void asm_dynamic_quat_normalize(const double* q, double* result);

#ifdef __cplusplus
}
#endif

#endif /* ASM_MATH_H */
