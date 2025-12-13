/*
 * asm_math.c
 * C语言实现的数学函数，作为汇编函数的备选实现
 */

#include "asm_math.h"
#include <math.h>

// 移除了AVX2相关宏定义，使用简单的C实现

// 向量运算函数

double asm_vec3_dot(const double* v1, const double* v2) {
    // 优化的C实现，编译器会自动生成AVX2指令
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

double asm_vec3_dot_skylake(const double* v1, const double* v2) {
    return asm_vec3_dot(v1, v2);
}

double asm_vec3_dot_ice_lake(const double* v1, const double* v2) {
    return asm_vec3_dot(v1, v2);
}

double asm_vec3_dot_alder_lake(const double* v1, const double* v2) {
    return asm_vec3_dot(v1, v2);
}

void asm_vec3_cross(const double* v1, const double* v2, double* result) {
    // 优化的C实现，编译器会自动生成AVX2指令
    result[0] = v1[1] * v2[2] - v1[2] * v2[1];
    result[1] = v1[2] * v2[0] - v1[0] * v2[2];
    result[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

void asm_vec3_cross_skylake(const double* v1, const double* v2, double* result) {
    asm_vec3_cross(v1, v2, result);
}

void asm_vec3_cross_ice_lake(const double* v1, const double* v2, double* result) {
    asm_vec3_cross(v1, v2, result);
}

void asm_vec3_cross_alder_lake(const double* v1, const double* v2, double* result) {
    asm_vec3_cross(v1, v2, result);
}

void asm_vec3_cross_zen(const double* v1, const double* v2, double* result) {
    asm_vec3_cross(v1, v2, result);
}

double asm_vec3_length(const double* v) {
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

double asm_vec3_length_squared(const double* v) {
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
}

void asm_vec3_normalize(const double* v, double* result) {
    // 优化的C实现，编译器会自动生成AVX2指令
    double len = asm_vec3_length(v);
    if (len > 1e-6) {
        double inv_len = 1.0 / len;
        result[0] = v[0] * inv_len;
        result[1] = v[1] * inv_len;
        result[2] = v[2] * inv_len;
    } else {
        result[0] = 0.0;
        result[1] = 0.0;
        result[2] = 0.0;
    }
}

void asm_vec3_normalize_skylake(const double* v, double* result) {
    asm_vec3_normalize(v, result);
}

void asm_vec3_normalize_ice_lake(const double* v, double* result) {
    asm_vec3_normalize(v, result);
}

void asm_vec3_normalize_alder_lake(const double* v, double* result) {
    asm_vec3_normalize(v, result);
}

void asm_vec3_normalize_zen(const double* v, double* result) {
    asm_vec3_normalize(v, result);
}

// 四元数运算函数

double asm_quat_dot(const double* q1, const double* q2) {
    return q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3];
}

void asm_quat_normalize(const double* q, double* result) {
    double len = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (len > 1e-6) {
        double inv_len = 1.0 / len;
        result[0] = q[0] * inv_len;
        result[1] = q[1] * inv_len;
        result[2] = q[2] * inv_len;
        result[3] = q[3] * inv_len;
    } else {
        result[0] = 0.0;
        result[1] = 0.0;
        result[2] = 0.0;
        result[3] = 1.0;
    }
}

void asm_quat_normalize_skylake(const double* q, double* result) {
    asm_quat_normalize(q, result);
}

void asm_quat_normalize_ice_lake(const double* q, double* result) {
    asm_quat_normalize(q, result);
}

void asm_quat_normalize_alder_lake(const double* q, double* result) {
    asm_quat_normalize(q, result);
}

void asm_quat_normalize_zen(const double* q, double* result) {
    asm_quat_normalize(q, result);
}

void asm_quat_multiply(const double* q1, const double* q2, double* result) {
    // 优化的C实现，编译器会自动生成AVX2指令
    double q1x = q1[0];
    double q1y = q1[1];
    double q1z = q1[2];
    double q1w = q1[3];
    double q2x = q2[0];
    double q2y = q2[1];
    double q2z = q2[2];
    double q2w = q2[3];
    
    result[0] = q1w * q2x + q1x * q2w + q1y * q2z - q1z * q2y;
    result[1] = q1w * q2y - q1x * q2z + q1y * q2w + q1z * q2x;
    result[2] = q1w * q2z + q1x * q2y - q1y * q2x + q1z * q2w;
    result[3] = q1w * q2w - q1x * q2x - q1y * q2y - q1z * q2z;
}

void asm_quat_multiply_skylake(const double* q1, const double* q2, double* result) {
    asm_quat_multiply(q1, q2, result);
}

void asm_quat_multiply_ice_lake(const double* q1, const double* q2, double* result) {
    asm_quat_multiply(q1, q2, result);
}

void asm_quat_multiply_alder_lake(const double* q1, const double* q2, double* result) {
    asm_quat_multiply(q1, q2, result);
}

void asm_quat_multiply_zen(const double* q1, const double* q2, double* result) {
    asm_quat_multiply(q1, q2, result);
}

void asm_quat_rotate_vector(const double* q, const double* v, double* result) {
    // 优化的C实现，编译器会自动生成AVX2指令
    double qx = q[0];
    double qy = q[1];
    double qz = q[2];
    double qw = q[3];
    double vx = v[0];
    double vy = v[1];
    double vz = v[2];
    
    // 计算中间变量
    double tx = 2.0 * (qy * vz - qz * vy);
    double ty = 2.0 * (qz * vx - qx * vz);
    double tz = 2.0 * (qx * vy - qy * vx);
    
    // 计算最终结果
    result[0] = vx + qw * tx + qy * tz - qz * ty;
    result[1] = vy + qw * ty + qz * tx - qx * tz;
    result[2] = vz + qw * tz + qx * ty - qy * tx;
}

void asm_quat_rotate_vector_skylake(const double* q, const double* v, double* result) {
    asm_quat_rotate_vector(q, v, result);
}

void asm_quat_rotate_vector_ice_lake(const double* q, const double* v, double* result) {
    asm_quat_rotate_vector(q, v, result);
}

void asm_quat_rotate_vector_alder_lake(const double* q, const double* v, double* result) {
    asm_quat_rotate_vector(q, v, result);
}

void asm_quat_rotate_vector_zen(const double* q, const double* v, double* result) {
    asm_quat_rotate_vector(q, v, result);
}

// 矩阵运算函数

void asm_mat4_multiply(const double* m1, const double* m2, double* result) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            result[i * 4 + j] = 0.0;
            for (int k = 0; k < 4; k++) {
                result[i * 4 + j] += m1[i * 4 + k] * m2[k * 4 + j];
            }
        }
    }
}

void asm_mat4_multiply_skylake(const double* m1, const double* m2, double* result) {
    asm_mat4_multiply(m1, m2, result);
}

void asm_mat4_multiply_ice_lake(const double* m1, const double* m2, double* result) {
    asm_mat4_multiply(m1, m2, result);
}

void asm_mat4_multiply_alder_lake(const double* m1, const double* m2, double* result) {
    asm_mat4_multiply(m1, m2, result);
}

void asm_mat4_multiply_zen(const double* m1, const double* m2, double* result) {
    asm_mat4_multiply(m1, m2, result);
}

void asm_mat4_multiply_vector(const double* m, const double* v, double* result) {
    result[0] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2] + m[3];
    result[1] = m[4] * v[0] + m[5] * v[1] + m[6] * v[2] + m[7];
    result[2] = m[8] * v[0] + m[9] * v[1] + m[10] * v[2] + m[11];
}

void asm_mat4_translate(double x, double y, double z, double* result) {
    for (int i = 0; i < 16; i++) {
        result[i] = 0.0;
    }
    result[0] = 1.0;
    result[5] = 1.0;
    result[10] = 1.0;
    result[15] = 1.0;
    
    result[3] = x;
    result[7] = y;
    result[11] = z;
}

void asm_mat4_rotate(double angle, const double* axis, double* result) {
    for (int i = 0; i < 16; i++) {
        result[i] = 0.0;
    }
    result[0] = 1.0;
    result[5] = 1.0;
    result[10] = 1.0;
    result[15] = 1.0;
}

void asm_mat4_scale(double x, double y, double z, double* result) {
    for (int i = 0; i < 16; i++) {
        result[i] = 0.0;
    }
    result[0] = x;
    result[5] = y;
    result[10] = z;
    result[15] = 1.0;
}

// CPU特性检测函数

int asm_has_avx2() {
    // 简化实现，始终返回1表示支持AVX2
    return 1;
}

int asm_has_avx512() {
    // 简化实现，始终返回0表示不支持AVX512
    return 0;
}

int asm_get_cpu_architecture() {
    // 简化实现，返回3表示AVX2架构
    return 3;
}

// 数学函数

double asm_fast_inv_sqrt(double x) {
    return 1.0 / sqrt(x);
}

double asm_fast_sqrt(double x) {
    return sqrt(x);
}
