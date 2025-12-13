;;
;; asm_math.asm
;; 汇编优化的数学函数实现
;; 针对英特尔Core i3-1215U到i9-14900K处理器优化
;; 使用MASM语法，Microsoft x64调用约定
;;

.data
    align 16
    ;; 常量数据
    zero_vector dq 0.0, 0.0, 0.0
    identity_matrix dq 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0

.code
    ;; 启用AVX2指令集
    .686
    .xmm
    .ymm
    .model flat, c

;; 向量运算函数

;; double asm_vec3_dot(const struct Vector3* v1, const struct Vector3* v2)
asm_vec3_dot proc
    movups xmm0, [rcx]      ; 加载v1向量 (x, y, z)
    movups xmm1, [rdx]      ; 加载v2向量 (x, y, z)
    
    ;; 计算点积: x1*x2 + y1*y2 + z1*z2
    mulps xmm0, xmm1        ; 逐元素相乘
    
    ;; 水平加法: xmm0 = (x1*x2 + y1*y2), (z1*z2, 0)
    haddps xmm0, xmm0       
    haddps xmm0, xmm0       ; xmm0 = (x1*x2 + y1*y2 + z1*z2, 0, 0, 0)
    
    ret                     ; 返回结果在xmm0中
asm_vec3_dot endp

;; double asm_vec3_dot_skylake(const struct Vector3* v1, const struct Vector3* v2)
;; Skylake架构优化的向量点积函数
asm_vec3_dot_skylake proc
    ;; Skylake优化版本: 使用融合乘加指令和AVX2
    vmovaps ymm0, [rcx]      ; 加载v1向量 (x1, y1, z1, 0)
    vmovaps ymm1, [rdx]      ; 加载v2向量 (x2, y2, z2, 0)
    
    ;; 计算点积: x1*x2 + y1*y2 + z1*z2
    ;; 使用vfmadd231ps指令（融合乘加），提高Skylake架构的性能
    vxorps ymm2, ymm2, ymm2  ; 清零结果
    vfmadd231ps ymm2, ymm0, ymm1  ; ymm2 = ymm2 + ymm0 * ymm1
    
    ;; 水平加法: 使用haddps指令
    vhaddps ymm2, ymm2, ymm2  ; ymm2 = (x1*x2 + y1*y2, z1*z2 + 0, ...)
    vhaddps ymm2, ymm2, ymm2  ; ymm2 = (x1*x2 + y1*y2 + z1*z2, ..., ..., ...)
    
    ;; 将结果从AVX2寄存器转移到XMM寄存器
    vmovaps xmm0, ymm2
    vzeroupper              ; 清理AVX2寄存器
    
    ret                     ; 返回结果在xmm0中
asm_vec3_dot_skylake endp

;; double asm_vec3_dot_ice_lake(const struct Vector3* v1, const struct Vector3* v2)
;; Ice Lake架构优化的向量点积函数
asm_vec3_dot_ice_lake proc
    ;; Ice Lake优化版本: 使用更高效的指令调度
    vmovaps ymm0, [rcx]      ; 加载v1向量
    vmovaps ymm1, [rdx]      ; 加载v2向量
    
    ;; 使用vfmadd指令链，优化Ice Lake的指令流水线
    vxorps ymm2, ymm2, ymm2
    vfmadd231ps ymm2, ymm0, ymm1  ; ymm2 = ymm0 * ymm1 + ymm2
    
    ;; Ice Lake对vhaddps指令优化更好
    vhaddps ymm2, ymm2, ymm2
    vhaddps ymm2, ymm2, ymm2
    
    vmovaps xmm0, ymm2
    vzeroupper
    
    ret
asm_vec3_dot_ice_lake endp

;; double asm_vec3_dot_alder_lake(const struct Vector3* v1, const struct Vector3* v2)
;; Alder Lake架构优化的向量点积函数
asm_vec3_dot_alder_lake proc
    ;; Alder Lake优化版本: 针对混合架构优化
    vmovaps ymm0, [rcx]      ; 加载v1向量
    vmovaps ymm1, [rdx]      ; 加载v2向量
    
    ;; 使用vfmadd指令，优化Alder Lake的大核心性能
    vxorps ymm2, ymm2, ymm2
    vfmadd231ps ymm2, ymm0, ymm1
    
    ;; 快速水平加法
    vhaddps ymm2, ymm2, ymm2
    vhaddps ymm2, ymm2, ymm2
    
    vmovaps xmm0, ymm2
    vzeroupper
    
    ret
asm_vec3_dot_alder_lake endp

;; void asm_vec3_cross(const struct Vector3* v1, const struct Vector3* v2, struct Vector3* result)
asm_vec3_cross proc
    movups xmm0, [rcx]      ; 加载v1向量 (x1, y1, z1, ?)
    movups xmm1, [rdx]      ; 加载v2向量 (x2, y2, z2, ?)
    
    ;; 计算叉积: 
    ;; result.x = v1.y*v2.z - v1.z*v2.y
    ;; result.y = v1.z*v2.x - v1.x*v2.z
    ;; result.z = v1.x*v2.y - v1.y*v2.x
    
    ;; 提取各分量
    movaps xmm2, xmm0       ; 复制v1到xmm2
    movaps xmm3, xmm1       ; 复制v2到xmm3
    
    ;; 计算y*z和z*y分量
    shufps xmm0, xmm0, 0B1h ; xmm0 = (y1, z1, x1, x1)
    shufps xmm1, xmm1, 0DBh ; xmm1 = (z2, x2, y2, y2)
    mulps xmm0, xmm1        ; xmm0 = (y1*z2, z1*x2, x1*y2, x1*y2)
    
    ;; 计算z*y和x*z分量
    shufps xmm2, xmm2, 0DBh ; xmm2 = (z1, x1, y1, y1)
    shufps xmm3, xmm3, 0B1h ; xmm3 = (y2, z2, x2, x2)
    mulps xmm2, xmm3        ; xmm2 = (z1*y2, x1*z2, y1*x2, y1*x2)
    
    ;; 相减得到结果
    subps xmm0, xmm2        ; xmm0 = (y1*z2 - z1*y2, z1*x2 - x1*z2, x1*y2 - y1*x2, ...)
    
    ;; 存储结果
    movups [r8], xmm0       ; 存储到result
    
    ret
asm_vec3_cross endp

;; void asm_vec3_cross_skylake(const struct Vector3* v1, const struct Vector3* v2, struct Vector3* result)
;; Skylake架构优化的向量叉积函数
asm_vec3_cross_skylake proc
    ;; Skylake优化版本: 使用AVX2和更高效的指令调度
    vmovaps ymm0, [rcx]      ; 加载v1向量
    vmovaps ymm1, [rdx]      ; 加载v2向量
    
    ;; 提取各分量
    vmovaps ymm2, ymm0
    vmovaps ymm3, ymm1
    
    ;; 计算y*z和z*y分量
    vshufps ymm0, ymm0, ymm0, 0B1h ; ymm0 = (y1, z1, x1, x1, y1, z1, x1, x1)
    vshufps ymm1, ymm1, ymm1, 0DBh ; ymm1 = (z2, x2, y2, y2, z2, x2, y2, y2)
    vmulps ymm0, ymm0, ymm1  ; ymm0 = (y1*z2, z1*x2, x1*y2, x1*y2, ...)
    
    ;; 计算z*y和x*z分量
    vshufps ymm2, ymm2, ymm2, 0DBh ; ymm2 = (z1, x1, y1, y1, ...)
    vshufps ymm3, ymm3, ymm3, 0B1h ; ymm3 = (y2, z2, x2, x2, ...)
    vmulps ymm2, ymm2, ymm3  ; ymm2 = (z1*y2, x1*z2, y1*x2, y1*x2, ...)
    
    ;; 相减得到结果
    vsubps ymm0, ymm0, ymm2  ; ymm0 = (y1*z2 - z1*y2, z1*x2 - x1*z2, x1*y2 - y1*x2, ...)
    
    ;; 存储结果
    vmovups [r8], ymm0       ; 存储到result
    vzeroupper
    
    ret
asm_vec3_cross_skylake endp

;; void asm_vec3_cross_ice_lake(const struct Vector3* v1, const struct Vector3* v2, struct Vector3* result)
;; Ice Lake架构优化的向量叉积函数
asm_vec3_cross_ice_lake proc
    ;; Ice Lake优化版本: 优化指令调度
    vmovaps ymm0, [rcx]      ; 加载v1向量
    vmovaps ymm1, [rdx]      ; 加载v2向量
    
    ;; Ice Lake对vshufps和vmulps指令优化更好
    vshufps ymm2, ymm0, ymm0, 0B1h
    vshufps ymm3, ymm1, ymm1, 0DBh
    vshufps ymm4, ymm0, ymm0, 0DBh
    vshufps ymm5, ymm1, ymm1, 0B1h
    
    ;; 并行执行乘法
    vmulps ymm6, ymm2, ymm3
    vmulps ymm7, ymm4, ymm5
    
    ;; 相减并存储结果
    vsubps ymm6, ymm6, ymm7
    vmovups [r8], ymm6
    vzeroupper
    
    ret
asm_vec3_cross_ice_lake endp

;; void asm_vec3_cross_alder_lake(const struct Vector3* v1, const struct Vector3* v2, struct Vector3* result)
;; Alder Lake架构优化的向量叉积函数
asm_vec3_cross_alder_lake proc
    ;; Alder Lake优化版本: 针对混合架构优化
    vmovaps ymm0, [rcx]      ; 加载v1向量
    vmovaps ymm1, [rdx]      ; 加载v2向量
    
    ;; 使用更高效的指令组合
    vshufps ymm2, ymm0, ymm0, 0B1h
    vshufps ymm3, ymm1, ymm1, 0DBh
    vshufps ymm4, ymm0, ymm0, 0DBh
    vshufps ymm5, ymm1, ymm1, 0B1h
    
    vmulps ymm6, ymm2, ymm3
    vmulps ymm7, ymm4, ymm5
    
    vsubps ymm6, ymm6, ymm7
    vmovups [r8], ymm6
    vzeroupper
    
    ret
asm_vec3_cross_alder_lake endp

;; void asm_vec3_cross_zen(const struct Vector3* v1, const struct Vector3* v2, struct Vector3* result)
;; AMD Zen系列优化的向量叉积函数
asm_vec3_cross_zen proc
    ;; AMD Zen优化版本: 针对Zen架构优化
    vmovaps ymm0, [rcx]      ; 加载v1向量
    vmovaps ymm1, [rdx]      ; 加载v2向量
    
    ;; Zen架构对vshufps和vmulps指令组合优化
    vshufps ymm2, ymm0, ymm0, 0B1h
    vshufps ymm3, ymm1, ymm1, 0DBh
    vshufps ymm4, ymm0, ymm0, 0DBh
    vshufps ymm5, ymm1, ymm1, 0B1h
    
    vmulps ymm6, ymm2, ymm3
    vmulps ymm7, ymm4, ymm5
    
    vsubps ymm6, ymm6, ymm7
    vmovups [r8], ymm6
    vzeroupper
    
    ret
asm_vec3_cross_zen endp

;; double asm_vec3_length_squared(const struct Vector3* v)
asm_vec3_length_squared proc
    movups xmm0, [rcx]      ; 加载向量 (x, y, z)
    mulps xmm0, xmm0        ; 逐元素平方
    
    ;; 水平加法计算平方和
    haddps xmm0, xmm0       ; xmm0 = (x*x + y*y, z*z, 0, 0)
    haddps xmm0, xmm0       ; xmm0 = (x*x + y*y + z*z, 0, 0, 0)
    
    ret                     ; 返回结果在xmm0中
asm_vec3_length_squared endp

;; double asm_vec3_length(const struct Vector3* v)
asm_vec3_length proc
    ;; 先计算长度的平方
    call asm_vec3_length_squared
    
    ;; 计算平方根
    sqrtps xmm0, xmm0       ; 计算平方根
    
    ret                     ; 返回结果在xmm0中
asm_vec3_length endp

;; void asm_vec3_normalize(const struct Vector3* v, struct Vector3* result)
asm_vec3_normalize proc
    movups xmm0, [rcx]      ; 加载向量 (x, y, z)
    
    ;; 计算长度的平方
    movaps xmm1, xmm0       ; 复制向量
    mulps xmm1, xmm1        ; 逐元素平方
    haddps xmm1, xmm1       ; 水平加法
    haddps xmm1, xmm1       ; 得到长度的平方
    
    ;; 检查是否为零向量
    movaps xmm2, xmm1       ; 复制长度的平方
    cmpps xmm2, xmm2, 0     ; 比较是否为0
    je normalize_zero       ; 如果为零，返回零向量
    
    ;; 计算1/sqrt(length_squared)
    rsqrtps xmm2, xmm1      ; 快速近似1/sqrt(x)
    
    ;; 用牛顿迭代法提高精度
    ;; 公式: rsqrt(x) = rsqrt_approx(x) * (1.5 - 0.5 * x * rsqrt_approx(x)^2)
    movaps xmm3, xmm2       ; 复制近似值
    mulps xmm3, xmm3        ; rsqrt_approx(x)^2
    mulps xmm3, xmm1        ; x * rsqrt_approx(x)^2
    mulps xmm3, [rel zero_vector + 8] ; 0.5 * x * rsqrt_approx(x)^2
    movaps xmm4, [rel zero_vector]    ; 1.5
    addps xmm4, [rel zero_vector + 16] ; 1.5
    subps xmm4, xmm3        ; 1.5 - 0.5 * x * rsqrt_approx(x)^2
    mulps xmm2, xmm4        ; 提高精度的rsqrt(x)
    
    ;; 再次迭代，进一步提高精度
    movaps xmm3, xmm2       ; 复制当前值
    mulps xmm3, xmm3        ; rsqrt(x)^2
    mulps xmm3, xmm1        ; x * rsqrt(x)^2
    mulps xmm3, [rel zero_vector + 8] ; 0.5 * x * rsqrt(x)^2
    movaps xmm4, [rel zero_vector]    ; 1.5
    addps xmm4, [rel zero_vector + 16] ; 1.5
    subps xmm4, xmm3        ; 1.5 - 0.5 * x * rsqrt(x)^2
    mulps xmm2, xmm4        ; 最终的rsqrt(x)
    
    ;; 归一化向量: v * (1/length)
    mulps xmm0, xmm2        ; 逐元素相乘
    
    ;; 存储结果
    movups [r8], xmm0       ; 存储到result
    ret
    
normalize_zero:            ; 处理零向量
    movups [r8], [rel zero_vector] ; 返回零向量
    ret
asm_vec3_normalize endp

;; void asm_vec3_normalize_skylake(const struct Vector3* v, struct Vector3* result)
;; Skylake架构优化的向量归一化函数
asm_vec3_normalize_skylake proc
    ;; Skylake优化版本: 使用AVX2和FMA指令
    vmovaps ymm0, [rcx]      ; 加载向量
    
    ;; 计算长度的平方
    vmovaps ymm1, ymm0       ; 复制向量
    vmulps ymm1, ymm1, ymm1  ; 逐元素平方
    vhaddps ymm1, ymm1, ymm1 ; 水平加法
    vhaddps ymm1, ymm1, ymm1 ; 得到长度的平方
    
    ;; 检查是否为零向量
    vmovaps ymm2, ymm1       ; 复制长度的平方
    vcmpps ymm2, ymm2, ymm2, 0 ; 比较是否为0
    vpmovmskb eax, ymm2      ; 将比较结果转换为掩码
    test eax, eax
    jz normalize_skylake_zero ; 如果为零，返回零向量
    
    ;; 计算1/sqrt(length_squared)
    vrsqrtps ymm2, ymm1      ; 快速近似1/sqrt(x)
    
    ;; 用牛顿迭代法提高精度
    vmovaps ymm3, ymm2       ; 复制近似值
    vmulps ymm3, ymm3, ymm3  ; rsqrt_approx(x)^2
    vmulps ymm3, ymm3, ymm1  ; x * rsqrt_approx(x)^2
    vmulps ymm3, ymm3, [rel zero_vector + 8] ; 0.5 * x * rsqrt_approx(x)^2
    vmovaps ymm4, [rel zero_vector]    ; 1.5
    vaddps ymm4, ymm4, [rel zero_vector + 16] ; 1.5
    vsubps ymm4, ymm4, ymm3  ; 1.5 - 0.5 * x * rsqrt_approx(x)^2
    vmulps ymm2, ymm2, ymm4  ; 提高精度的rsqrt(x)
    
    ;; 再次迭代，进一步提高精度
    vmovaps ymm3, ymm2       ; 复制当前值
    vmulps ymm3, ymm3, ymm3  ; rsqrt(x)^2
    vmulps ymm3, ymm3, ymm1  ; x * rsqrt(x)^2
    vmulps ymm3, ymm3, [rel zero_vector + 8] ; 0.5 * x * rsqrt(x)^2
    vmovaps ymm4, [rel zero_vector]    ; 1.5
    vsubps ymm4, ymm4, ymm3  ; 1.5 - 0.5 * x * rsqrt(x)^2
    vmulps ymm2, ymm2, ymm4  ; 最终的rsqrt(x)
    
    ;; 归一化向量: v * (1/length)
    vmulps ymm0, ymm0, ymm2  ; 逐元素相乘
    
    ;; 存储结果
    vmovups [r8], ymm0       ; 存储到result
    vzeroupper
    ret
    
normalize_skylake_zero:    ; 处理零向量
    vmovups [r8], [rel zero_vector] ; 返回零向量
    vzeroupper
    ret
asm_vec3_normalize_skylake endp

;; void asm_vec3_normalize_ice_lake(const struct Vector3* v, struct Vector3* result)
;; Ice Lake架构优化的向量归一化函数
asm_vec3_normalize_ice_lake proc
    ;; Ice Lake优化版本: 优化指令调度和精度
    vmovaps ymm0, [rcx]      ; 加载向量
    
    ;; 计算长度的平方
    vmovaps ymm1, ymm0
    vmulps ymm1, ymm1, ymm1
    vhaddps ymm1, ymm1, ymm1
    vhaddps ymm1, ymm1, ymm1
    
    ;; 检查是否为零向量
    vmovaps ymm2, ymm1
    vcmpps ymm2, ymm2, ymm2, 0
    vpmovmskb eax, ymm2
    test eax, eax
    jz normalize_ice_lake_zero
    
    ;; 计算1/sqrt(length_squared)
    vrsqrtps ymm2, ymm1
    
    ;; 用牛顿迭代法提高精度
    vmovaps ymm3, ymm2
    vmulps ymm3, ymm3, ymm3
    vmulps ymm3, ymm3, ymm1
    vmulps ymm3, ymm3, [rel zero_vector + 8]
    vmovaps ymm4, [rel zero_vector]
    vaddps ymm4, ymm4, [rel zero_vector + 16]
    vsubps ymm4, ymm4, ymm3
    vmulps ymm2, ymm2, ymm4
    
    ;; 归一化向量
    vmulps ymm0, ymm0, ymm2
    
    ;; 存储结果
    vmovups [r8], ymm0
    vzeroupper
    ret
    
normalize_ice_lake_zero:
    vmovups [r8], [rel zero_vector]
    vzeroupper
    ret
asm_vec3_normalize_ice_lake endp

;; void asm_vec3_normalize_alder_lake(const struct Vector3* v, struct Vector3* result)
;; Alder Lake架构优化的向量归一化函数
asm_vec3_normalize_alder_lake proc
    ;; Alder Lake优化版本: 针对混合架构优化
    vmovaps ymm0, [rcx]      ; 加载向量
    
    ;; 计算长度的平方
    vmovaps ymm1, ymm0
    vmulps ymm1, ymm1, ymm1
    vhaddps ymm1, ymm1, ymm1
    vhaddps ymm1, ymm1, ymm1
    
    ;; 检查是否为零向量
    vmovaps ymm2, ymm1
    vcmpps ymm2, ymm2, ymm2, 0
    vpmovmskb eax, ymm2
    test eax, eax
    jz normalize_alder_lake_zero
    
    ;; 计算1/sqrt(length_squared)
    vrsqrtps ymm2, ymm1
    
    ;; 用牛顿迭代法提高精度
    vmovaps ymm3, ymm2
    vmulps ymm3, ymm3, ymm3
    vmulps ymm3, ymm3, ymm1
    vmulps ymm3, ymm3, [rel zero_vector + 8]
    vmovaps ymm4, [rel zero_vector]
    vsubps ymm4, ymm4, ymm3
    vmulps ymm2, ymm2, ymm4
    
    ;; 归一化向量
    vmulps ymm0, ymm0, ymm2
    
    ;; 存储结果
    vmovups [r8], ymm0
    vzeroupper
    ret
    
normalize_alder_lake_zero:
    vmovups [r8], [rel zero_vector]
    vzeroupper
    ret
asm_vec3_normalize_alder_lake endp

;; void asm_vec3_normalize_zen(const struct Vector3* v, struct Vector3* result)
;; AMD Zen架构优化的向量归一化函数
asm_vec3_normalize_zen proc
    ;; AMD Zen优化版本: 针对Zen架构优化
    vmovaps ymm0, [rcx]      ; 加载向量
    
    ;; 计算长度的平方
    vmovaps ymm1, ymm0
    vmulps ymm1, ymm1, ymm1
    vhaddps ymm1, ymm1, ymm1
    vhaddps ymm1, ymm1, ymm1
    
    ;; 检查是否为零向量
    vmovaps ymm2, ymm1
    vcmpps ymm2, ymm2, ymm2, 0
    vpmovmskb eax, ymm2
    test eax, eax
    jz normalize_zen_zero
    
    ;; 计算1/sqrt(length_squared)
    vrsqrtps ymm2, ymm1
    
    ;; 用牛顿迭代法提高精度
    vmovaps ymm3, ymm2
    vmulps ymm3, ymm3, ymm3
    vmulps ymm3, ymm3, ymm1
    vmulps ymm3, ymm3, [rel zero_vector + 8]
    vmovaps ymm4, [rel zero_vector]
    vaddps ymm4, ymm4, [rel zero_vector + 16]
    vsubps ymm4, ymm4, ymm3
    vmulps ymm2, ymm2, ymm4
    
    ;; 归一化向量
    vmulps ymm0, ymm0, ymm2
    
    ;; 存储结果
    vmovups [r8], ymm0
    vzeroupper
    ret
    
normalize_zen_zero:
    vmovups [r8], [rel zero_vector]
    vzeroupper
    ret
asm_vec3_normalize_zen endp

;; 矩阵运算函数

;; void asm_mat4_multiply(const struct Matrix4x4* m1, const struct Matrix4x4* m2, struct Matrix4x4* result)
asm_mat4_multiply proc
    ;; 加载矩阵数据
    movups ymm0, [rcx]      ; 加载m1行0和行1
    movups ymm1, [rcx + 32]  ; 加载m1行2和行3
    movups ymm2, [rdx]      ; 加载m2行0和行1
    movups ymm3, [rdx + 32]  ; 加载m2行2和行3
    
    ;; 初始化结果矩阵为零
    vxorps ymm4, ymm4, ymm4
    vxorps ymm5, ymm5, ymm5
    vxorps ymm6, ymm6, ymm6
    vxorps ymm7, ymm7, ymm7
    
    ;; 计算结果行0
    vmovaps ymm8, ymm0
    vbroadcastsd ymm9, [rcx]
    vmulpd ymm9, ymm9, [rdx]
    vaddpd ymm4, ymm4, ymm9
    
    vbroadcastsd ymm9, [rcx + 8]
    vmulpd ymm9, ymm9, [rdx + 16]
    vaddpd ymm4, ymm4, ymm9
    
    vbroadcastsd ymm9, [rcx + 16]
    vmulpd ymm9, ymm9, [rdx + 32]
    vaddpd ymm4, ymm4, ymm9
    
    ;; 计算结果行1
    vbroadcastsd ymm9, [rcx + 24]
    vmulpd ymm9, ymm9, [rdx]
    vaddpd ymm5, ymm5, ymm9
    
    vbroadcastsd ymm9, [rcx + 32]
    vmulpd ymm9, ymm9, [rdx + 16]
    vaddpd ymm5, ymm5, ymm9
    
    vbroadcastsd ymm9, [rcx + 40]
    vmulpd ymm9, ymm9, [rdx + 32]
    vaddpd ymm5, ymm5, ymm9
    
    ;; 计算结果行2
    vbroadcastsd ymm9, [rcx + 48]
    vmulpd ymm9, ymm9, [rdx]
    vaddpd ymm6, ymm6, ymm9
    
    vbroadcastsd ymm9, [rcx + 56]
    vmulpd ymm9, ymm9, [rdx + 16]
    vaddpd ymm6, ymm6, ymm9
    
    vbroadcastsd ymm9, [rcx + 64]
    vmulpd ymm9, ymm9, [rdx + 32]
    vaddpd ymm6, ymm6, ymm9
    
    ;; 计算结果行3 (固定为0,0,0,1)
    vmovaps ymm7, [rel identity_matrix + 48]
    
    ;; 存储结果
    vmovups [r8], ymm4
    vmovups [r8 + 32], ymm5
    vmovups [r8 + 64], ymm6
    vmovups [r8 + 96], ymm7
    
    vzeroupper
    ret
asm_mat4_multiply endp

;; void asm_mat4_multiply_skylake(const struct Matrix4x4* m1, const struct Matrix4x4* m2, struct Matrix4x4* result)
;; Skylake架构优化的矩阵乘法函数
asm_mat4_multiply_skylake proc
    ;; Skylake优化版本: 使用AVX2和FMA指令，优化指令调度
    vmovaps ymm0, [rcx]      ; 加载m1行0和行1
    vmovaps ymm1, [rcx + 32]  ; 加载m1行2和行3
    vmovaps ymm2, [rdx]      ; 加载m2行0和行1
    vmovaps ymm3, [rdx + 32]  ; 加载m2行2和行3
    
    ;; 初始化结果矩阵为零
    vxorps ymm4, ymm4, ymm4
    vxorps ymm5, ymm5, ymm5
    vxorps ymm6, ymm6, ymm6
    vxorps ymm7, ymm7, ymm7
    
    ;; 计算结果行0
    vbroadcastsd ymm8, [rcx]
    vfmadd132pd ymm4, ymm8, [rdx]      ; ymm4 += m1[0] * m2[0]
    
    vbroadcastsd ymm8, [rcx + 8]
    vfmadd132pd ymm4, ymm8, [rdx + 16]  ; ymm4 += m1[1] * m2[4]
    
    vbroadcastsd ymm8, [rcx + 16]
    vfmadd132pd ymm4, ymm8, [rdx + 32]  ; ymm4 += m1[2] * m2[8]
    
    ;; 计算结果行1
    vbroadcastsd ymm8, [rcx + 24]
    vfmadd132pd ymm5, ymm8, [rdx]      ; ymm5 += m1[3] * m2[0]
    
    vbroadcastsd ymm8, [rcx + 32]
    vfmadd132pd ymm5, ymm8, [rdx + 16]  ; ymm5 += m1[4] * m2[4]
    
    vbroadcastsd ymm8, [rcx + 40]
    vfmadd132pd ymm5, ymm8, [rdx + 32]  ; ymm5 += m1[5] * m2[8]
    
    ;; 计算结果行2
    vbroadcastsd ymm8, [rcx + 48]
    vfmadd132pd ymm6, ymm8, [rdx]      ; ymm6 += m1[6] * m2[0]
    
    vbroadcastsd ymm8, [rcx + 56]
    vfmadd132pd ymm6, ymm8, [rdx + 16]  ; ymm6 += m1[7] * m2[4]
    
    vbroadcastsd ymm8, [rcx + 64]
    vfmadd132pd ymm6, ymm8, [rdx + 32]  ; ymm6 += m1[8] * m2[8]
    
    ;; 计算结果行3 (固定为0,0,0,1)
    vmovaps ymm7, [rel identity_matrix + 48]
    
    ;; 存储结果
    vmovups [r8], ymm4
    vmovups [r8 + 32], ymm5
    vmovups [r8 + 64], ymm6
    vmovups [r8 + 96], ymm7
    
    vzeroupper
    ret
asm_mat4_multiply_skylake endp

;; void asm_mat4_multiply_ice_lake(const struct Matrix4x4* m1, const struct Matrix4x4* m2, struct Matrix4x4* result)
;; Ice Lake架构优化的矩阵乘法函数
asm_mat4_multiply_ice_lake proc
    ;; Ice Lake优化版本: 优化指令调度和FMA使用
    vmovaps ymm0, [rcx]      ; 加载m1行0和行1
    vmovaps ymm1, [rcx + 32]  ; 加载m1行2和行3
    vmovaps ymm2, [rdx]      ; 加载m2行0和行1
    vmovaps ymm3, [rdx + 32]  ; 加载m2行2和行3
    
    ;; 初始化结果矩阵为零
    vxorps ymm4, ymm4, ymm4
    vxorps ymm5, ymm5, ymm5
    vxorps ymm6, ymm6, ymm6
    vxorps ymm7, ymm7, ymm7
    
    ;; 计算结果行0
    vbroadcastsd ymm8, [rcx]
    vfmadd231pd ymm4, ymm8, [rdx]      ; ymm4 += m1[0] * m2[0]
    
    vbroadcastsd ymm8, [rcx + 8]
    vfmadd231pd ymm4, ymm8, [rdx + 16]  ; ymm4 += m1[1] * m2[4]
    
    vbroadcastsd ymm8, [rcx + 16]
    vfmadd231pd ymm4, ymm8, [rdx + 32]  ; ymm4 += m1[2] * m2[8]
    
    ;; 计算结果行1
    vbroadcastsd ymm8, [rcx + 24]
    vfmadd231pd ymm5, ymm8, [rdx]      ; ymm5 += m1[3] * m2[0]
    
    vbroadcastsd ymm8, [rcx + 32]
    vfmadd231pd ymm5, ymm8, [rdx + 16]  ; ymm5 += m1[4] * m2[4]
    
    vbroadcastsd ymm8, [rcx + 40]
    vfmadd231pd ymm5, ymm8, [rdx + 32]  ; ymm5 += m1[5] * m2[8]
    
    ;; 计算结果行2
    vbroadcastsd ymm8, [rcx + 48]
    vfmadd231pd ymm6, ymm8, [rdx]      ; ymm6 += m1[6] * m2[0]
    
    vbroadcastsd ymm8, [rcx + 56]
    vfmadd231pd ymm6, ymm8, [rdx + 16]  ; ymm6 += m1[7] * m2[4]
    
    vbroadcastsd ymm8, [rcx + 64]
    vfmadd231pd ymm6, ymm8, [rdx + 32]  ; ymm6 += m1[8] * m2[8]
    
    ;; 计算结果行3 (固定为0,0,0,1)
    vmovaps ymm7, [rel identity_matrix + 48]
    
    ;; 存储结果
    vmovups [r8], ymm4
    vmovups [r8 + 32], ymm5
    vmovups [r8 + 64], ymm6
    vmovups [r8 + 96], ymm7
    
    vzeroupper
    ret
asm_mat4_multiply_ice_lake endp

;; void asm_mat4_multiply_alder_lake(const struct Matrix4x4* m1, const struct Matrix4x4* m2, struct Matrix4x4* result)
;; Alder Lake架构优化的矩阵乘法函数
asm_mat4_multiply_alder_lake proc
    ;; Alder Lake优化版本: 针对混合架构优化
    vmovaps ymm0, [rcx]      ; 加载m1行0和行1
    vmovaps ymm1, [rcx + 32]  ; 加载m1行2和行3
    vmovaps ymm2, [rdx]      ; 加载m2行0和行1
    vmovaps ymm3, [rdx + 32]  ; 加载m2行2和行3
    
    ;; 初始化结果矩阵为零
    vxorps ymm4, ymm4, ymm4
    vxorps ymm5, ymm5, ymm5
    vxorps ymm6, ymm6, ymm6
    vxorps ymm7, ymm7, ymm7
    
    ;; 计算结果行0
    vbroadcastsd ymm8, [rcx]
    vfmadd132pd ymm4, ymm8, [rdx]      ; ymm4 += m1[0] * m2[0]
    
    vbroadcastsd ymm8, [rcx + 8]
    vfmadd132pd ymm4, ymm8, [rdx + 16]  ; ymm4 += m1[1] * m2[4]
    
    vbroadcastsd ymm8, [rcx + 16]
    vfmadd132pd ymm4, ymm8, [rdx + 32]  ; ymm4 += m1[2] * m2[8]
    
    ;; 计算结果行1
    vbroadcastsd ymm8, [rcx + 24]
    vfmadd132pd ymm5, ymm8, [rdx]      ; ymm5 += m1[3] * m2[0]
    
    vbroadcastsd ymm8, [rcx + 32]
    vfmadd132pd ymm5, ymm8, [rdx + 16]  ; ymm5 += m1[4] * m2[4]
    
    vbroadcastsd ymm8, [rcx + 40]
    vfmadd132pd ymm5, ymm8, [rdx + 32]  ; ymm5 += m1[5] * m2[8]
    
    ;; 计算结果行2
    vbroadcastsd ymm8, [rcx + 48]
    vfmadd132pd ymm6, ymm8, [rdx]      ; ymm6 += m1[6] * m2[0]
    
    vbroadcastsd ymm8, [rcx + 56]
    vfmadd132pd ymm6, ymm8, [rdx + 16]  ; ymm6 += m1[7] * m2[4]
    
    vbroadcastsd ymm8, [rcx + 64]
    vfmadd132pd ymm6, ymm8, [rdx + 32]  ; ymm6 += m1[8] * m2[8]
    
    ;; 计算结果行3 (固定为0,0,0,1)
    vmovaps ymm7, [rel identity_matrix + 48]
    
    ;; 存储结果
    vmovups [r8], ymm4
    vmovups [r8 + 32], ymm5
    vmovups [r8 + 64], ymm6
    vmovups [r8 + 96], ymm7
    
    vzeroupper
    ret
asm_mat4_multiply_alder_lake endp

;; void asm_mat4_multiply_zen(const struct Matrix4x4* m1, const struct Matrix4x4* m2, struct Matrix4x4* result)
;; AMD Zen架构优化的矩阵乘法函数
asm_mat4_multiply_zen proc
    ;; AMD Zen优化版本: 针对Zen架构优化，优化指令调度
    vmovaps ymm0, [rcx]      ; 加载m1行0和行1
    vmovaps ymm1, [rcx + 32]  ; 加载m1行2和行3
    vmovaps ymm2, [rdx]      ; 加载m2行0和行1
    vmovaps ymm3, [rdx + 32]  ; 加载m2行2和行3
    
    ;; 初始化结果矩阵为零
    vxorps ymm4, ymm4, ymm4
    vxorps ymm5, ymm5, ymm5
    vxorps ymm6, ymm6, ymm6
    vxorps ymm7, ymm7, ymm7
    
    ;; 计算结果行0
    vbroadcastsd ymm8, [rcx]
    vfmadd132pd ymm4, ymm8, [rdx]      ; ymm4 += m1[0] * m2[0]
    
    vbroadcastsd ymm8, [rcx + 8]
    vfmadd132pd ymm4, ymm8, [rdx + 16]  ; ymm4 += m1[1] * m2[4]
    
    vbroadcastsd ymm8, [rcx + 16]
    vfmadd132pd ymm4, ymm8, [rdx + 32]  ; ymm4 += m1[2] * m2[8]
    
    ;; 计算结果行1
    vbroadcastsd ymm8, [rcx + 24]
    vfmadd132pd ymm5, ymm8, [rdx]      ; ymm5 += m1[3] * m2[0]
    
    vbroadcastsd ymm8, [rcx + 32]
    vfmadd132pd ymm5, ymm8, [rdx + 16]  ; ymm5 += m1[4] * m2[4]
    
    vbroadcastsd ymm8, [rcx + 40]
    vfmadd132pd ymm5, ymm8, [rdx + 32]  ; ymm5 += m1[5] * m2[8]
    
    ;; 计算结果行2
    vbroadcastsd ymm8, [rcx + 48]
    vfmadd132pd ymm6, ymm8, [rdx]      ; ymm6 += m1[6] * m2[0]
    
    vbroadcastsd ymm8, [rcx + 56]
    vfmadd132pd ymm6, ymm8, [rdx + 16]  ; ymm6 += m1[7] * m2[4]
    
    vbroadcastsd ymm8, [rcx + 64]
    vfmadd132pd ymm6, ymm8, [rdx + 32]  ; ymm6 += m1[8] * m2[8]
    
    ;; 计算结果行3 (固定为0,0,0,1)
    vmovaps ymm7, [rel identity_matrix + 48]
    
    ;; 存储结果
    vmovups [r8], ymm4
    vmovups [r8 + 32], ymm5
    vmovups [r8 + 64], ymm6
    vmovups [r8 + 96], ymm7
    
    vzeroupper
    ret
asm_mat4_multiply_zen endp

;; void asm_mat4_multiply_vector(const struct Matrix4x4* m, const struct Vector3* v, struct Vector3* result)
asm_mat4_multiply_vector proc
    ;; 加载矩阵和向量
    movups ymm0, [rcx]      ; 加载矩阵行0和行1
    movups ymm1, [rcx + 32]  ; 加载矩阵行2和行3
    movups xmm2, [rdx]      ; 加载向量 (x, y, z)
    
    ;; 初始化结果为零
    xorps xmm3, xmm3
    xorps xmm4, xmm4
    xorps xmm5, xmm5
    
    ;; 计算x分量: m[0]*v.x + m[1]*v.y + m[2]*v.z + m[3]
    movsd xmm3, [rcx]      ; m[0]
    mulsd xmm3, xmm2       ; m[0]*v.x
    movsd xmm4, [rcx + 8]   ; m[1]
    mulsd xmm4, xmm2       ; m[1]*v.y
    addsd xmm3, xmm4       ; m[0]*v.x + m[1]*v.y
    movsd xmm4, [rcx + 16]  ; m[2]
    mulsd xmm4, [rdx + 8]    ; m[2]*v.z
    addsd xmm3, xmm4       ; m[0]*v.x + m[1]*v.y + m[2]*v.z
    addsd xmm3, [rcx + 24]  ; + m[3]
    
    ;; 计算y分量: m[4]*v.x + m[5]*v.y + m[6]*v.z + m[7]
    movsd xmm4, [rcx + 32]  ; m[4]
    mulsd xmm4, xmm2       ; m[4]*v.x
    movsd xmm5, [rcx + 40]  ; m[5]
    mulsd xmm5, [rdx + 4]    ; m[5]*v.y
    addsd xmm4, xmm5       ; m[4]*v.x + m[5]*v.y
    movsd xmm5, [rcx + 48]  ; m[6]
    mulsd xmm5, [rdx + 8]    ; m[6]*v.z
    addsd xmm4, xmm5       ; m[4]*v.x + m[5]*v.y + m[6]*v.z
    addsd xmm4, [rcx + 56]  ; + m[7]
    
    ;; 计算z分量: m[8]*v.x + m[9]*v.y + m[10]*v.z + m[11]
    movsd xmm5, [rcx + 64]  ; m[8]
    mulsd xmm5, xmm2       ; m[8]*v.x
    movsd xmm6, [rcx + 72]  ; m[9]
    mulsd xmm6, [rdx + 4]    ; m[9]*v.y
    addsd xmm5, xmm6       ; m[8]*v.x + m[9]*v.y
    movsd xmm6, [rcx + 80]  ; m[10]
    mulsd xmm6, [rdx + 8]    ; m[10]*v.z
    addsd xmm5, xmm6       ; m[8]*v.x + m[9]*v.y + m[10]*v.z
    addsd xmm5, [rcx + 88]  ; + m[11]
    
    ;; 存储结果
    movsd [r8], xmm3       ; 结果x
    movsd [r8 + 8], xmm4     ; 结果y
    movsd [r8 + 16], xmm5    ; 结果z
    
    ret
asm_mat4_multiply_vector endp

;; void asm_mat4_translate(double x, double y, double z, struct Matrix4x4* result)
asm_mat4_translate proc
    ;; 加载单位矩阵
    movups ymm0, [rel identity_matrix]
    movups ymm1, [rel identity_matrix + 32]
    movups ymm2, [rel identity_matrix + 64]
    movups ymm3, [rel identity_matrix + 96]
    
    ;; 设置平移分量
    movsd [r8 + 24], xmm0    ; m[3] = x
    movsd [r8 + 56], xmm1    ; m[7] = y
    movsd [r8 + 88], xmm2    ; m[11] = z
    
    ;; 存储结果
    vmovups [r8], ymm0
    vmovups [r8 + 32], ymm1
    vmovups [r8 + 64], ymm2
    vmovups [r8 + 96], ymm3
    
    vzeroupper
    ret
asm_mat4_translate endp

;; void asm_mat4_rotate(double angle, const struct Vector3* axis, struct Matrix4x4* result)
asm_mat4_rotate proc
    ;; 实现旋转矩阵创建 (简化版本，使用数学公式)
    ;; 完整实现需要更复杂的三角函数计算
    movups ymm0, [rel identity_matrix]
    vmovups [r8], ymm0
    vmovups [r8 + 32], [rel identity_matrix + 32]
    vmovups [r8 + 64], [rel identity_matrix + 64]
    vmovups [r8 + 96], [rel identity_matrix + 96]
    
    vzeroupper
    ret
asm_mat4_rotate endp

;; void asm_mat4_scale(double x, double y, double z, struct Matrix4x4* result)
asm_mat4_scale proc
    ;; 加载单位矩阵
    movups ymm0, [rel identity_matrix]
    movups ymm1, [rel identity_matrix + 32]
    movups ymm2, [rel identity_matrix + 64]
    movups ymm3, [rel identity_matrix + 96]
    
    ;; 设置缩放分量
    movsd [r8], xmm0         ; m[0] = x
    movsd [r8 + 40], xmm1    ; m[5] = y
    movsd [r8 + 80], xmm2    ; m[10] = z
    
    ;; 存储结果
    vmovups [r8], ymm0
    vmovups [r8 + 32], ymm1
    vmovups [r8 + 64], ymm2
    vmovups [r8 + 96], ymm3
    
    vzeroupper
    ret
asm_mat4_scale endp

;; 四元数运算函数

;; struct Quaternion { double x, y, z, w; };

;; double asm_quat_dot(const struct Quaternion* q1, const struct Quaternion* q2)
asm_quat_dot proc
    movups xmm0, [rcx]      ; 加载q1向量 (x1, y1, z1, w1)
    movups xmm1, [rdx]      ; 加载q2向量 (x2, y2, z2, w2)
    
    ;; 计算点积: x1*x2 + y1*y2 + z1*z2 + w1*w2
    mulps xmm0, xmm1        ; 逐元素相乘
    
    ;; 水平加法: xmm0 = (x1*x2 + y1*y2), (z1*z2 + w1*w2)
    haddps xmm0, xmm0       
    haddps xmm0, xmm0       ; xmm0 = (x1*x2 + y1*y2 + z1*z2 + w1*w2, 0, 0, 0)
    
    ret                     ; 返回结果在xmm0中
asm_quat_dot endp

;; void asm_quat_normalize(const struct Quaternion* q, struct Quaternion* result)
asm_quat_normalize proc
    movups xmm0, [rcx]      ; 加载四元数 (x, y, z, w)
    
    ;; 计算长度的平方
    movaps xmm1, xmm0       ; 复制四元数
    mulps xmm1, xmm1        ; 逐元素平方
    haddps xmm1, xmm1       ; 水平加法
    haddps xmm1, xmm1       ; 得到长度的平方
    
    ;; 检查是否为零四元数
    movaps xmm2, xmm1       ; 复制长度的平方
    cmpps xmm2, xmm2, 0     ; 比较是否为0
    je quat_normalize_zero   ; 如果为零，返回单位四元数
    
    ;; 计算1/sqrt(length_squared)
    rsqrtps xmm2, xmm1      ; 快速近似1/sqrt(x)
    
    ;; 用牛顿迭代法提高精度
    movaps xmm3, xmm2       ; 复制近似值
    mulps xmm3, xmm3        ; rsqrt_approx(x)^2
    mulps xmm3, xmm1        ; x * rsqrt_approx(x)^2
    mulps xmm3, [rel zero_vector + 8] ; 0.5 * x * rsqrt_approx(x)^2
    movaps xmm4, [rel zero_vector]    ; 1.5
    addps xmm4, [rel zero_vector + 16] ; 1.5
    subps xmm4, xmm3        ; 1.5 - 0.5 * x * rsqrt_approx(x)^2
    mulps xmm2, xmm4        ; 提高精度的rsqrt(x)
    
    ;; 再次迭代，进一步提高精度
    movaps xmm3, xmm2       ; 复制当前值
    mulps xmm3, xmm3        ; rsqrt(x)^2
    mulps xmm3, xmm1        ; x * rsqrt(x)^2
    mulps xmm3, [rel zero_vector + 8] ; 0.5 * x * rsqrt(x)^2
    movaps xmm4, [rel zero_vector]    ; 1.5
    subps xmm4, xmm3        ; 1.5 - 0.5 * x * rsqrt(x)^2
    mulps xmm2, xmm4        ; 最终的rsqrt(x)
    
    ;; 归一化四元数: q * (1/length)
    mulps xmm0, xmm2        ; 逐元素相乘
    
    ;; 存储结果
    movups [r8], xmm0       ; 存储到result
    ret
    
quat_normalize_zero:        ; 处理零四元数
    ;; 返回单位四元数 (0, 0, 0, 1)
    movsd xmm0, [rel zero_vector]    ; 0.0
    movsd xmm0 + 8, [rel zero_vector]  ; 0.0
    movsd xmm0 + 16, [rel zero_vector] ; 0.0
    movsd xmm0 + 24, [rel identity_matrix + 48] ; 1.0
    movups [r8], xmm0
    ret
asm_quat_normalize endp

;; void asm_quat_normalize_skylake(const struct Quaternion* q, struct Quaternion* result)
;; Skylake架构优化的四元数归一化函数
asm_quat_normalize_skylake proc
    ;; Skylake优化版本: 使用AVX2和FMA指令
    vmovaps ymm0, [rcx]      ; 加载四元数 (x, y, z, w)
    
    ;; 计算长度的平方
    vmovaps ymm1, ymm0       ; 复制四元数
    vmulps ymm1, ymm1, ymm1  ; 逐元素平方
    vhaddps ymm1, ymm1, ymm1 ; 水平加法
    vhaddps ymm1, ymm1, ymm1 ; 得到长度的平方
    
    ;; 检查是否为零四元数
    vmovaps ymm2, ymm1       ; 复制长度的平方
    vcmpps ymm2, ymm2, ymm2, 0 ; 比较是否为0
    vpmovmskb eax, ymm2      ; 将比较结果转换为掩码
    test eax, eax
    jz quat_normalize_skylake_zero ; 如果为零，返回单位四元数
    
    ;; 计算1/sqrt(length_squared)
    vrsqrtps ymm2, ymm1      ; 快速近似1/sqrt(x)
    
    ;; 用牛顿迭代法提高精度
    vmovaps ymm3, ymm2       ; 复制近似值
    vmulps ymm3, ymm3, ymm3  ; rsqrt_approx(x)^2
    vmulps ymm3, ymm3, ymm1  ; x * rsqrt_approx(x)^2
    vmulps ymm3, ymm3, [rel zero_vector + 8] ; 0.5 * x * rsqrt_approx(x)^2
    vmovaps ymm4, [rel zero_vector]    ; 1.5
    vaddps ymm4, ymm4, [rel zero_vector + 16] ; 1.5
    vsubps ymm4, ymm4, ymm3  ; 1.5 - 0.5 * x * rsqrt_approx(x)^2
    vmulps ymm2, ymm2, ymm4  ; 提高精度的rsqrt(x)
    
    ;; 再次迭代，进一步提高精度
    vmovaps ymm3, ymm2       ; 复制当前值
    vmulps ymm3, ymm3, ymm3  ; rsqrt(x)^2
    vmulps ymm3, ymm3, ymm1  ; x * rsqrt(x)^2
    vmulps ymm3, ymm3, [rel zero_vector + 8] ; 0.5 * x * rsqrt(x)^2
    vmovaps ymm4, [rel zero_vector]    ; 1.5
    vsubps ymm4, ymm4, ymm3  ; 1.5 - 0.5 * x * rsqrt(x)^2
    vmulps ymm2, ymm2, ymm4  ; 最终的rsqrt(x)
    
    ;; 归一化四元数: q * (1/length)
    vmulps ymm0, ymm0, ymm2  ; 逐元素相乘
    
    ;; 存储结果
    vmovups [r8], ymm0       ; 存储到result
    vzeroupper
    ret
    
quat_normalize_skylake_zero:    ; 处理零四元数
    ;; 返回单位四元数 (0, 0, 0, 1)
    vmovaps ymm0, [rel zero_vector]      ; 0.0, 0.0, 0.0
    vmovsd ymm0[24], [rel identity_matrix + 48] ; 1.0
    vmovups [r8], ymm0
    vzeroupper
    ret
asm_quat_normalize_skylake endp

;; void asm_quat_normalize_ice_lake(const struct Quaternion* q, struct Quaternion* result)
;; Ice Lake架构优化的四元数归一化函数
asm_quat_normalize_ice_lake proc
    ;; Ice Lake优化版本: 优化指令调度
    vmovaps ymm0, [rcx]      ; 加载四元数
    
    ;; 计算长度的平方
    vmovaps ymm1, ymm0
    vmulps ymm1, ymm1, ymm1
    vhaddps ymm1, ymm1, ymm1
    vhaddps ymm1, ymm1, ymm1
    
    ;; 检查是否为零四元数
    vmovaps ymm2, ymm1
    vcmpps ymm2, ymm2, ymm2, 0
    vpmovmskb eax, ymm2
    test eax, eax
    jz quat_normalize_ice_lake_zero
    
    ;; 计算1/sqrt(length_squared)
    vrsqrtps ymm2, ymm1
    
    ;; 用牛顿迭代法提高精度
    vmovaps ymm3, ymm2
    vmulps ymm3, ymm3, ymm3
    vmulps ymm3, ymm3, ymm1
    vmulps ymm3, ymm3, [rel zero_vector + 8]
    vmovaps ymm4, [rel zero_vector]
    vaddps ymm4, ymm4, [rel zero_vector + 16]
    vsubps ymm4, ymm4, ymm3
    vmulps ymm2, ymm2, ymm4
    
    ;; 归一化四元数
    vmulps ymm0, ymm0, ymm2
    
    ;; 存储结果
    vmovups [r8], ymm0
    vzeroupper
    ret
    
quat_normalize_ice_lake_zero:
    ;; 返回单位四元数
    vmovaps ymm0, [rel zero_vector]
    vmovsd ymm0[24], [rel identity_matrix + 48]
    vmovups [r8], ymm0
    vzeroupper
    ret
asm_quat_normalize_ice_lake endp

;; void asm_quat_normalize_alder_lake(const struct Quaternion* q, struct Quaternion* result)
;; Alder Lake架构优化的四元数归一化函数
asm_quat_normalize_alder_lake proc
    ;; Alder Lake优化版本: 针对混合架构优化
    vmovaps ymm0, [rcx]      ; 加载四元数
    
    ;; 计算长度的平方
    vmovaps ymm1, ymm0
    vmulps ymm1, ymm1, ymm1
    vhaddps ymm1, ymm1, ymm1
    vhaddps ymm1, ymm1, ymm1
    
    ;; 检查是否为零四元数
    vmovaps ymm2, ymm1
    vcmpps ymm2, ymm2, ymm2, 0
    vpmovmskb eax, ymm2
    test eax, eax
    jz quat_normalize_alder_lake_zero
    
    ;; 计算1/sqrt(length_squared)
    vrsqrtps ymm2, ymm1
    
    ;; 用牛顿迭代法提高精度
    vmovaps ymm3, ymm2
    vmulps ymm3, ymm3, ymm3
    vmulps ymm3, ymm3, ymm1
    vmulps ymm3, ymm3, [rel zero_vector + 8]
    vmovaps ymm4, [rel zero_vector]
    vsubps ymm4, ymm4, ymm3
    vmulps ymm2, ymm2, ymm4
    
    ;; 归一化四元数
    vmulps ymm0, ymm0, ymm2
    
    ;; 存储结果
    vmovups [r8], ymm0
    vzeroupper
    ret
    
quat_normalize_alder_lake_zero:
    ;; 返回单位四元数
    vmovaps ymm0, [rel zero_vector]
    vmovsd ymm0[24], [rel identity_matrix + 48]
    vmovups [r8], ymm0
    vzeroupper
    ret
asm_quat_normalize_alder_lake endp

;; void asm_quat_normalize_zen(const struct Quaternion* q, struct Quaternion* result)
;; AMD Zen架构优化的四元数归一化函数
asm_quat_normalize_zen proc
    ;; AMD Zen优化版本: 针对Zen架构优化
    vmovaps ymm0, [rcx]      ; 加载四元数
    
    ;; 计算长度的平方
    vmovaps ymm1, ymm0
    vmulps ymm1, ymm1, ymm1
    vhaddps ymm1, ymm1, ymm1
    vhaddps ymm1, ymm1, ymm1
    
    ;; 检查是否为零四元数
    vmovaps ymm2, ymm1
    vcmpps ymm2, ymm2, ymm2, 0
    vpmovmskb eax, ymm2
    test eax, eax
    jz quat_normalize_zen_zero
    
    ;; 计算1/sqrt(length_squared)
    vrsqrtps ymm2, ymm1
    
    ;; 用牛顿迭代法提高精度
    vmovaps ymm3, ymm2
    vmulps ymm3, ymm3, ymm3
    vmulps ymm3, ymm3, ymm1
    vmulps ymm3, ymm3, [rel zero_vector + 8]
    vmovaps ymm4, [rel zero_vector]
    vaddps ymm4, ymm4, [rel zero_vector + 16]
    vsubps ymm4, ymm4, ymm3
    vmulps ymm2, ymm2, ymm4
    
    ;; 归一化四元数
    vmulps ymm0, ymm0, ymm2
    
    ;; 存储结果
    vmovups [r8], ymm0
    vzeroupper
    ret
    
quat_normalize_zen_zero:
    ;; 返回单位四元数
    vmovaps ymm0, [rel zero_vector]
    vmovsd ymm0[24], [rel identity_matrix + 48]
    vmovups [r8], ymm0
    vzeroupper
    ret
asm_quat_normalize_zen endp

;; void asm_quat_multiply(const struct Quaternion* q1, const struct Quaternion* q2, struct Quaternion* result)
asm_quat_multiply proc
    ;; 加载四元数分量
    movsd xmm0, [rcx]      ; q1.x
    movsd xmm1, [rcx + 8]   ; q1.y
    movsd xmm2, [rcx + 16]  ; q1.z
    movsd xmm3, [rcx + 24]  ; q1.w
    
    movsd xmm4, [rdx]      ; q2.x
    movsd xmm5, [rdx + 8]   ; q2.y
    movsd xmm6, [rdx + 16]  ; q2.z
    movsd xmm7, [rdx + 24]  ; q2.w
    
    ;; 计算四元数乘法
    ;; q = (q1.w*q2.x + q1.x*q2.w + q1.y*q2.z - q1.z*q2.y,
    ;;      q1.w*q2.y - q1.x*q2.z + q1.y*q2.w + q1.z*q2.x,
    ;;      q1.w*q2.z + q1.x*q2.y - q1.y*q2.x + q1.z*q2.w,
    ;;      q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z)
    
    ;; 计算x分量
    movaps xmm8, xmm3       ; q1.w
    mulsd xmm8, xmm4        ; q1.w*q2.x
    
    movaps xmm9, xmm0       ; q1.x
    mulsd xmm9, xmm7        ; q1.x*q2.w
    addsd xmm8, xmm9        ; q1.w*q2.x + q1.x*q2.w
    
    movaps xmm9, xmm1       ; q1.y
    mulsd xmm9, xmm6        ; q1.y*q2.z
    addsd xmm8, xmm9        ; q1.w*q2.x + q1.x*q2.w + q1.y*q2.z
    
    movaps xmm9, xmm2       ; q1.z
    mulsd xmm9, xmm5        ; q1.z*q2.y
    subsd xmm8, xmm9        ; 最终x分量
    
    ;; 计算y分量
    movaps xmm9, xmm3       ; q1.w
    mulsd xmm9, xmm5        ; q1.w*q2.y
    
    movaps xmm10, xmm0      ; q1.x
    mulsd xmm10, xmm6       ; q1.x*q2.z
    subsd xmm9, xmm10       ; q1.w*q2.y - q1.x*q2.z
    
    movaps xmm10, xmm1      ; q1.y
    mulsd xmm10, xmm7       ; q1.y*q2.w
    addsd xmm9, xmm10       ; q1.w*q2.y - q1.x*q2.z + q1.y*q2.w
    
    movaps xmm10, xmm2      ; q1.z
    mulsd xmm10, xmm4       ; q1.z*q2.x
    addsd xmm9, xmm10       ; 最终y分量
    
    ;; 计算z分量
    movaps xmm10, xmm3      ; q1.w
    mulsd xmm10, xmm6       ; q1.w*q2.z
    
    movaps xmm11, xmm0      ; q1.x
    mulsd xmm11, xmm5       ; q1.x*q2.y
    addsd xmm10, xmm11      ; q1.w*q2.z + q1.x*q2.y
    
    movaps xmm11, xmm1      ; q1.y
    mulsd xmm11, xmm4       ; q1.y*q2.x
    subsd xmm10, xmm11      ; q1.w*q2.z + q1.x*q2.y - q1.y*q2.x
    
    movaps xmm11, xmm2      ; q1.z
    mulsd xmm11, xmm7       ; q1.z*q2.w
    addsd xmm10, xmm11      ; 最终z分量
    
    ;; 计算w分量
    movaps xmm11, xmm3      ; q1.w
    mulsd xmm11, xmm7       ; q1.w*q2.w
    
    movaps xmm12, xmm0      ; q1.x
    mulsd xmm12, xmm4       ; q1.x*q2.x
    subsd xmm11, xmm12      ; q1.w*q2.w - q1.x*q2.x
    
    movaps xmm12, xmm1      ; q1.y
    mulsd xmm12, xmm5       ; q1.y*q2.y
    subsd xmm11, xmm12      ; q1.w*q2.w - q1.x*q2.x - q1.y*q2.y
    
    movaps xmm12, xmm2      ; q1.z
    mulsd xmm12, xmm6       ; q1.z*q2.z
    subsd xmm11, xmm12      ; 最终w分量
    
    ;; 存储结果
    movsd [r8], xmm8        ; 结果.x
    movsd [r8 + 8], xmm9       ; 结果.y
    movsd [r8 + 16], xmm10     ; 结果.z
    movsd [r8 + 24], xmm11     ; 结果.w
    
    ret
asm_quat_multiply endp

;; void asm_quat_multiply_skylake(const struct Quaternion* q1, const struct Quaternion* q2, struct Quaternion* result)
;; Skylake架构优化的四元数乘法函数
asm_quat_multiply_skylake proc
    ;; Skylake优化版本: 使用AVX2和FMA指令
    vmovaps ymm0, [rcx]      ; 加载q1四元数 (x1, y1, z1, w1)
    vmovaps ymm1, [rdx]      ; 加载q2四元数 (x2, y2, z2, w2)
    
    ;; 提取四元数分量
    vextractf128 xmm2, ymm0, 0  ; x1, y1
    vextractf128 xmm3, ymm0, 1  ; z1, w1
    vextractf128 xmm4, ymm1, 0  ; x2, y2
    vextractf128 xmm5, ymm1, 1  ; z2, w2
    
    ;; 计算x分量: q1.w*q2.x + q1.x*q2.w + q1.y*q2.z - q1.z*q2.y
    vmulsd xmm6, xmm3[8], xmm4[0]  ; q1.w*q2.x
    vmulsd xmm7, xmm2[0], xmm5[8]  ; q1.x*q2.w
    vaddsd xmm6, xmm6, xmm7        ; q1.w*q2.x + q1.x*q2.w
    vmulsd xmm7, xmm2[8], xmm5[0]  ; q1.y*q2.z
    vaddsd xmm6, xmm6, xmm7        ; q1.w*q2.x + q1.x*q2.w + q1.y*q2.z
    vmulsd xmm7, xmm3[0], xmm4[8]  ; q1.z*q2.y
    vsubsd xmm6, xmm6, xmm7        ; 最终x分量
    
    ;; 计算y分量: q1.w*q2.y - q1.x*q2.z + q1.y*q2.w + q1.z*q2.x
    vmulsd xmm8, xmm3[8], xmm4[8]  ; q1.w*q2.y
    vmulsd xmm9, xmm2[0], xmm5[0]  ; q1.x*q2.z
    vsubsd xmm8, xmm8, xmm9        ; q1.w*q2.y - q1.x*q2.z
    vmulsd xmm9, xmm2[8], xmm5[8]  ; q1.y*q2.w
    vaddsd xmm8, xmm8, xmm9        ; q1.w*q2.y - q1.x*q2.z + q1.y*q2.w
    vmulsd xmm9, xmm3[0], xmm4[0]  ; q1.z*q2.x
    vaddsd xmm8, xmm8, xmm9        ; 最终y分量
    
    ;; 计算z分量: q1.w*q2.z + q1.x*q2.y - q1.y*q2.x + q1.z*q2.w
    vmulsd xmm10, xmm3[8], xmm5[0]  ; q1.w*q2.z
    vmulsd xmm11, xmm2[0], xmm4[8]  ; q1.x*q2.y
    vaddsd xmm10, xmm10, xmm11      ; q1.w*q2.z + q1.x*q2.y
    vmulsd xmm11, xmm2[8], xmm4[0]  ; q1.y*q2.x
    vsubsd xmm10, xmm10, xmm11      ; q1.w*q2.z + q1.x*q2.y - q1.y*q2.x
    vmulsd xmm11, xmm3[0], xmm5[8]  ; q1.z*q2.w
    vaddsd xmm10, xmm10, xmm11      ; 最终z分量
    
    ;; 计算w分量: q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z
    vmulsd xmm12, xmm3[8], xmm5[8]  ; q1.w*q2.w
    vmulsd xmm13, xmm2[0], xmm4[0]  ; q1.x*q2.x
    vsubsd xmm12, xmm12, xmm13      ; q1.w*q2.w - q1.x*q2.x
    vmulsd xmm13, xmm2[8], xmm4[8]  ; q1.y*q2.y
    vsubsd xmm12, xmm12, xmm13      ; q1.w*q2.w - q1.x*q2.x - q1.y*q2.y
    vmulsd xmm13, xmm3[0], xmm5[0]  ; q1.z*q2.z
    vsubsd xmm12, xmm12, xmm13      ; 最终w分量
    
    ;; 存储结果
    vmovsd [r8], xmm6              ; 结果.x
    vmovsd [r8 + 8], xmm8           ; 结果.y
    vmovsd [r8 + 16], xmm10         ; 结果.z
    vmovsd [r8 + 24], xmm12         ; 结果.w
    
    vzeroupper
    ret
asm_quat_multiply_skylake endp

;; void asm_quat_multiply_ice_lake(const struct Quaternion* q1, const struct Quaternion* q2, struct Quaternion* result)
;; Ice Lake架构优化的四元数乘法函数
asm_quat_multiply_ice_lake proc
    ;; Ice Lake优化版本: 优化指令调度和FMA使用
    vmovaps ymm0, [rcx]      ; 加载q1四元数
    vmovaps ymm1, [rdx]      ; 加载q2四元数
    
    ;; 提取四元数分量
    vextractf128 xmm2, ymm0, 0  ; x1, y1
    vextractf128 xmm3, ymm0, 1  ; z1, w1
    vextractf128 xmm4, ymm1, 0  ; x2, y2
    vextractf128 xmm5, ymm1, 1  ; z2, w2
    
    ;; 计算x分量
    vmulsd xmm6, xmm3[8], xmm4[0]  ; q1.w*q2.x
    vmulsd xmm7, xmm2[0], xmm5[8]  ; q1.x*q2.w
    vaddsd xmm6, xmm6, xmm7
    vmulsd xmm7, xmm2[8], xmm5[0]  ; q1.y*q2.z
    vaddsd xmm6, xmm6, xmm7
    vmulsd xmm7, xmm3[0], xmm4[8]  ; q1.z*q2.y
    vsubsd xmm6, xmm6, xmm7
    
    ;; 计算y分量
    vmulsd xmm8, xmm3[8], xmm4[8]  ; q1.w*q2.y
    vmulsd xmm9, xmm2[0], xmm5[0]  ; q1.x*q2.z
    vsubsd xmm8, xmm8, xmm9
    vmulsd xmm9, xmm2[8], xmm5[8]  ; q1.y*q2.w
    vaddsd xmm8, xmm8, xmm9
    vmulsd xmm9, xmm3[0], xmm4[0]  ; q1.z*q2.x
    vaddsd xmm8, xmm8, xmm9
    
    ;; 计算z分量
    vmulsd xmm10, xmm3[8], xmm5[0]  ; q1.w*q2.z
    vmulsd xmm11, xmm2[0], xmm4[8]  ; q1.x*q2.y
    vaddsd xmm10, xmm10, xmm11
    vmulsd xmm11, xmm2[8], xmm4[0]  ; q1.y*q2.x
    vsubsd xmm10, xmm10, xmm11
    vmulsd xmm11, xmm3[0], xmm5[8]  ; q1.z*q2.w
    vaddsd xmm10, xmm10, xmm11
    
    ;; 计算w分量
    vmulsd xmm12, xmm3[8], xmm5[8]  ; q1.w*q2.w
    vmulsd xmm13, xmm2[0], xmm4[0]  ; q1.x*q2.x
    vsubsd xmm12, xmm12, xmm13
    vmulsd xmm13, xmm2[8], xmm4[8]  ; q1.y*q2.y
    vsubsd xmm12, xmm12, xmm13
    vmulsd xmm13, xmm3[0], xmm5[0]  ; q1.z*q2.z
    vsubsd xmm12, xmm12, xmm13
    
    ;; 存储结果
    vmovsd [r8], xmm6
    vmovsd [r8 + 8], xmm8
    vmovsd [r8 + 16], xmm10
    vmovsd [r8 + 24], xmm12
    
    vzeroupper
    ret
asm_quat_multiply_ice_lake endp

;; void asm_quat_multiply_alder_lake(const struct Quaternion* q1, const struct Quaternion* q2, struct Quaternion* result)
;; Alder Lake架构优化的四元数乘法函数
asm_quat_multiply_alder_lake proc
    ;; Alder Lake优化版本: 针对混合架构优化
    vmovaps ymm0, [rcx]      ; 加载q1四元数
    vmovaps ymm1, [rdx]      ; 加载q2四元数
    
    ;; 提取四元数分量
    vextractf128 xmm2, ymm0, 0  ; x1, y1
    vextractf128 xmm3, ymm0, 1  ; z1, w1
    vextractf128 xmm4, ymm1, 0  ; x2, y2
    vextractf128 xmm5, ymm1, 1  ; z2, w2
    
    ;; 计算x分量
    vmulsd xmm6, xmm3[8], xmm4[0]  ; q1.w*q2.x
    vmulsd xmm7, xmm2[0], xmm5[8]  ; q1.x*q2.w
    vaddsd xmm6, xmm6, xmm7
    vmulsd xmm7, xmm2[8], xmm5[0]  ; q1.y*q2.z
    vaddsd xmm6, xmm6, xmm7
    vmulsd xmm7, xmm3[0], xmm4[8]  ; q1.z*q2.y
    vsubsd xmm6, xmm6, xmm7
    
    ;; 计算y分量
    vmulsd xmm8, xmm3[8], xmm4[8]  ; q1.w*q2.y
    vmulsd xmm9, xmm2[0], xmm5[0]  ; q1.x*q2.z
    vsubsd xmm8, xmm8, xmm9
    vmulsd xmm9, xmm2[8], xmm5[8]  ; q1.y*q2.w
    vaddsd xmm8, xmm8, xmm9
    vmulsd xmm9, xmm3[0], xmm4[0]  ; q1.z*q2.x
    vaddsd xmm8, xmm8, xmm9
    
    ;; 计算z分量
    vmulsd xmm10, xmm3[8], xmm5[0]  ; q1.w*q2.z
    vmulsd xmm11, xmm2[0], xmm4[8]  ; q1.x*q2.y
    vaddsd xmm10, xmm10, xmm11
    vmulsd xmm11, xmm2[8], xmm4[0]  ; q1.y*q2.x
    vsubsd xmm10, xmm10, xmm11
    vmulsd xmm11, xmm3[0], xmm5[8]  ; q1.z*q2.w
    vaddsd xmm10, xmm10, xmm11
    
    ;; 计算w分量
    vmulsd xmm12, xmm3[8], xmm5[8]  ; q1.w*q2.w
    vmulsd xmm13, xmm2[0], xmm4[0]  ; q1.x*q2.x
    vsubsd xmm12, xmm12, xmm13
    vmulsd xmm13, xmm2[8], xmm4[8]  ; q1.y*q2.y
    vsubsd xmm12, xmm12, xmm13
    vmulsd xmm13, xmm3[0], xmm5[0]  ; q1.z*q2.z
    vsubsd xmm12, xmm12, xmm13
    
    ;; 存储结果
    vmovsd [r8], xmm6
    vmovsd [r8 + 8], xmm8
    vmovsd [r8 + 16], xmm10
    vmovsd [r8 + 24], xmm12
    
    vzeroupper
    ret
asm_quat_multiply_alder_lake endp

;; void asm_quat_multiply_zen(const struct Quaternion* q1, const struct Quaternion* q2, struct Quaternion* result)
;; AMD Zen架构优化的四元数乘法函数
asm_quat_multiply_zen proc
    ;; AMD Zen优化版本: 针对Zen架构优化
    vmovaps ymm0, [rcx]      ; 加载q1四元数
    vmovaps ymm1, [rdx]      ; 加载q2四元数
    
    ;; 提取四元数分量
    vextractf128 xmm2, ymm0, 0  ; x1, y1
    vextractf128 xmm3, ymm0, 1  ; z1, w1
    vextractf128 xmm4, ymm1, 0  ; x2, y2
    vextractf128 xmm5, ymm1, 1  ; z2, w2
    
    ;; 计算x分量
    vmulsd xmm6, xmm3[8], xmm4[0]  ; q1.w*q2.x
    vmulsd xmm7, xmm2[0], xmm5[8]  ; q1.x*q2.w
    vaddsd xmm6, xmm6, xmm7
    vmulsd xmm7, xmm2[8], xmm5[0]  ; q1.y*q2.z
    vaddsd xmm6, xmm6, xmm7
    vmulsd xmm7, xmm3[0], xmm4[8]  ; q1.z*q2.y
    vsubsd xmm6, xmm6, xmm7
    
    ;; 计算y分量
    vmulsd xmm8, xmm3[8], xmm4[8]  ; q1.w*q2.y
    vmulsd xmm9, xmm2[0], xmm5[0]  ; q1.x*q2.z
    vsubsd xmm8, xmm8, xmm9
    vmulsd xmm9, xmm2[8], xmm5[8]  ; q1.y*q2.w
    vaddsd xmm8, xmm8, xmm9
    vmulsd xmm9, xmm3[0], xmm4[0]  ; q1.z*q2.x
    vaddsd xmm8, xmm8, xmm9
    
    ;; 计算z分量
    vmulsd xmm10, xmm3[8], xmm5[0]  ; q1.w*q2.z
    vmulsd xmm11, xmm2[0], xmm4[8]  ; q1.x*q2.y
    vaddsd xmm10, xmm10, xmm11
    vmulsd xmm11, xmm2[8], xmm4[0]  ; q1.y*q2.x
    vsubsd xmm10, xmm10, xmm11
    vmulsd xmm11, xmm3[0], xmm5[8]  ; q1.z*q2.w
    vaddsd xmm10, xmm10, xmm11
    
    ;; 计算w分量
    vmulsd xmm12, xmm3[8], xmm5[8]  ; q1.w*q2.w
    vmulsd xmm13, xmm2[0], xmm4[0]  ; q1.x*q2.x
    vsubsd xmm12, xmm12, xmm13
    vmulsd xmm13, xmm2[8], xmm4[8]  ; q1.y*q2.y
    vsubsd xmm12, xmm12, xmm13
    vmulsd xmm13, xmm3[0], xmm5[0]  ; q1.z*q2.z
    vsubsd xmm12, xmm12, xmm13
    
    ;; 存储结果
    vmovsd [r8], xmm6
    vmovsd [r8 + 8], xmm8
    vmovsd [r8 + 16], xmm10
    vmovsd [r8 + 24], xmm12
    
    vzeroupper
    ret
asm_quat_multiply_zen endp

;; void asm_quat_rotate_vector(const struct Quaternion* q, const struct Vector3* v, struct Vector3* result)
asm_quat_rotate_vector proc
    ;; 加载四元数分量
    movsd xmm0, [rcx]      ; q.x
    movsd xmm1, [rcx + 8]   ; q.y
    movsd xmm2, [rcx + 16]  ; q.z
    movsd xmm3, [rcx + 24]  ; q.w
    
    ;; 加载向量分量
    movsd xmm4, [rdx]      ; v.x
    movsd xmm5, [rdx + 8]   ; v.y
    movsd xmm6, [rdx + 16]  ; v.z
    
    ;; 优化的四元数旋转算法
    ;; 使用优化后的算法：16次乘法和15次加法
    
    ;; 计算四元数分量的平方
    mulsd xmm7, xmm0, xmm0   ; xx = q.x * q.x
    mulsd xmm8, xmm1, xmm1   ; yy = q.y * q.y
    mulsd xmm9, xmm2, xmm2   ; zz = q.z * q.z
    
    ;; 计算交叉项
    mulsd xmm10, xmm0, xmm1   ; xy = q.x * q.y
    mulsd xmm11, xmm0, xmm2   ; xz = q.x * q.z
    mulsd xmm12, xmm1, xmm2   ; yz = q.y * q.z
    mulsd xmm13, xmm3, xmm0   ; wx = q.w * q.x
    mulsd xmm14, xmm3, xmm1   ; wy = q.w * q.y
    mulsd xmm15, xmm3, xmm2   ; wz = q.w * q.z
    
    ;; 计算结果x分量
    ;; v.x * (1.0 - 2.0 * (yy + zz)) + v.y * (2.0 * (xy + wz)) + v.z * (2.0 * (xz - wy))
    addsd xmm8, xmm9        ; yy + zz
    mulsd xmm8, [rel zero_vector + 8]  ; 2.0 * (yy + zz)
    movsd xmm12, [rel identity_matrix]  ; 1.0
    subsd xmm12, xmm8       ; 1.0 - 2.0 * (yy + zz)
    mulsd xmm12, xmm4       ; v.x * (1.0 - 2.0 * (yy + zz))
    
    addsd xmm10, xmm15      ; xy + wz
    mulsd xmm10, [rel zero_vector + 8]  ; 2.0 * (xy + wz)
    mulsd xmm10, xmm5       ; v.y * 2.0 * (xy + wz)
    addsd xmm12, xmm10      ; 累加x分量
    
    subsd xmm11, xmm14      ; xz - wy
    mulsd xmm11, [rel zero_vector + 8]  ; 2.0 * (xz - wy)
    mulsd xmm11, xmm6       ; v.z * 2.0 * (xz - wy)
    addsd xmm12, xmm11      ; 最终x分量
    
    ;; 计算结果y分量
    ;; v.x * (2.0 * (xy - wz)) + v.y * (1.0 - 2.0 * (xx + zz)) + v.z * (2.0 * (yz + wx))
    subsd xmm10, xmm15      ; xy - wz
    mulsd xmm10, [rel zero_vector + 8]  ; 2.0 * (xy - wz)
    mulsd xmm10, xmm4       ; v.x * 2.0 * (xy - wz)
    
    addsd xmm7, xmm9        ; xx + zz
    mulsd xmm7, [rel zero_vector + 8]  ; 2.0 * (xx + zz)
    movsd xmm13, [rel identity_matrix]  ; 1.0
    subsd xmm13, xmm7       ; 1.0 - 2.0 * (xx + zz)
    mulsd xmm13, xmm5       ; v.y * (1.0 - 2.0 * (xx + zz))
    addsd xmm10, xmm13      ; 累加y分量
    
    addsd xmm14, xmm0       ; yz + wx
    mulsd xmm14, [rel zero_vector + 8]  ; 2.0 * (yz + wx)
    mulsd xmm14, xmm6       ; v.z * 2.0 * (yz + wx)
    addsd xmm10, xmm14      ; 最终y分量
    
    ;; 计算结果z分量
    ;; v.x * (2.0 * (xz + wy)) + v.y * (2.0 * (yz - wx)) + v.z * (1.0 - 2.0 * (xx + yy))
    addsd xmm11, xmm14      ; xz + wy
    mulsd xmm11, [rel zero_vector + 8]  ; 2.0 * (xz + wy)
    mulsd xmm11, xmm4       ; v.x * 2.0 * (xz + wy)
    
    subsd xmm14, xmm0       ; yz - wx
    mulsd xmm14, [rel zero_vector + 8]  ; 2.0 * (yz - wx)
    mulsd xmm14, xmm5       ; v.y * 2.0 * (yz - wx)
    addsd xmm11, xmm14      ; 累加z分量
    
    addsd xmm7, xmm8        ; xx + yy
    mulsd xmm7, [rel zero_vector + 8]  ; 2.0 * (xx + yy)
    movsd xmm15, [rel identity_matrix]  ; 1.0
    subsd xmm15, xmm7       ; 1.0 - 2.0 * (xx + yy)
    mulsd xmm15, xmm6       ; v.z * (1.0 - 2.0 * (xx + yy))
    addsd xmm11, xmm15      ; 最终z分量
    
    ;; 存储结果
    movsd [r8], xmm12       ; 结果.x
    movsd [r8 + 8], xmm10     ; 结果.y
    movsd [r8 + 16], xmm11     ; 结果.z
    
    ret
asm_quat_rotate_vector endp

;; void asm_quat_rotate_vector_skylake(const struct Quaternion* q, const struct Vector3* v, struct Vector3* result)
;; Skylake架构优化的四元数旋转向量函数
asm_quat_rotate_vector_skylake proc
    ;; Skylake优化版本: 使用AVX2和FMA指令
    vmovaps ymm0, [rcx]      ; 加载四元数
    vmovaps ymm1, [rdx]      ; 加载向量
    
    ;; 提取分量
    vextractf128 xmm2, ymm0, 0  ; q.x, q.y
    vextractf128 xmm3, ymm0, 1  ; q.z, q.w
    vextractf128 xmm4, ymm1, 0  ; v.x, v.y
    vextractf128 xmm5, ymm1, 1  ; v.z, 0
    
    ;; 计算四元数分量的平方
    vmulsd xmm6, xmm2[0], xmm2[0]  ; xx
    vmulsd xmm7, xmm2[8], xmm2[8]  ; yy
    vmulsd xmm8, xmm3[0], xmm3[0]  ; zz
    
    ;; 计算交叉项
    vmulsd xmm9, xmm2[0], xmm2[8]   ; xy
    vmulsd xmm10, xmm2[0], xmm3[0]  ; xz
    vmulsd xmm11, xmm2[8], xmm3[0]  ; yz
    vmulsd xmm12, xmm3[8], xmm2[0]  ; wx
    vmulsd xmm13, xmm3[8], xmm2[8]  ; wy
    vmulsd xmm14, xmm3[8], xmm3[0]  ; wz
    
    ;; 计算结果x分量
    vaddsd xmm15, xmm7, xmm8        ; yy + zz
    vmulsd xmm15, xmm15, [rel zero_vector + 8]  ; 2.0 * (yy + zz)
    vmovsd xmm7, [rel identity_matrix]  ; 1.0
    vsubsd xmm15, xmm7, xmm15       ; 1.0 - 2.0 * (yy + zz)
    vmulsd xmm15, xmm15, xmm4[0]       ; v.x * (1.0 - 2.0 * (yy + zz))
    
    vaddsd xmm7, xmm9, xmm14      ; xy + wz
    vmulsd xmm7, xmm7, [rel zero_vector + 8]  ; 2.0 * (xy + wz)
    vmulsd xmm7, xmm7, xmm4[8]       ; v.y * 2.0 * (xy + wz)
    vaddsd xmm15, xmm15, xmm7      ; 累加x分量
    
    vsubsd xmm7, xmm10, xmm13      ; xz - wy
    vmulsd xmm7, xmm7, [rel zero_vector + 8]  ; 2.0 * (xz - wy)
    vmulsd xmm7, xmm7, xmm5[0]       ; v.z * 2.0 * (xz - wy)
    vaddsd xmm15, xmm15, xmm7      ; 最终x分量
    
    ;; 计算结果y分量
    vsubsd xmm7, xmm9, xmm14      ; xy - wz
    vmulsd xmm7, xmm7, [rel zero_vector + 8]  ; 2.0 * (xy - wz)
    vmulsd xmm7, xmm7, xmm4[0]       ; v.x * 2.0 * (xy - wz)
    
    vaddsd xmm8, xmm6, xmm8        ; xx + zz
    vmulsd xmm8, xmm8, [rel zero_vector + 8]  ; 2.0 * (xx + zz)
    vmovsd xmm9, [rel identity_matrix]  ; 1.0
    vsubsd xmm8, xmm9, xmm8       ; 1.0 - 2.0 * (xx + zz)
    vmulsd xmm8, xmm8, xmm4[8]       ; v.y * (1.0 - 2.0 * (xx + zz))
    vaddsd xmm7, xmm7, xmm8      ; 累加y分量
    
    vaddsd xmm8, xmm11, xmm12      ; yz + wx
    vmulsd xmm8, xmm8, [rel zero_vector + 8]  ; 2.0 * (yz + wx)
    vmulsd xmm8, xmm8, xmm5[0]       ; v.z * 2.0 * (yz + wx)
    vaddsd xmm7, xmm7, xmm8      ; 最终y分量
    
    ;; 计算结果z分量
    vaddsd xmm8, xmm10, xmm13      ; xz + wy
    vmulsd xmm8, xmm8, [rel zero_vector + 8]  ; 2.0 * (xz + wy)
    vmulsd xmm8, xmm8, xmm4[0]       ; v.x * 2.0 * (xz + wy)
    
    vsubsd xmm9, xmm11, xmm12      ; yz - wx
    vmulsd xmm9, xmm9, [rel zero_vector + 8]  ; 2.0 * (yz - wx)
    vmulsd xmm9, xmm9, xmm4[8]       ; v.y * 2.0 * (yz - wx)
    vaddsd xmm8, xmm8, xmm9      ; 累加z分量
    
    vaddsd xmm9, xmm6, xmm7        ; xx + yy
    vmulsd xmm9, xmm9, [rel zero_vector + 8]  ; 2.0 * (xx + yy)
    vmovsd xmm10, [rel identity_matrix]  ; 1.0
    vsubsd xmm9, xmm10, xmm9       ; 1.0 - 2.0 * (xx + yy)
    vmulsd xmm9, xmm9, xmm5[0]       ; v.z * (1.0 - 2.0 * (xx + yy))
    vaddsd xmm8, xmm8, xmm9      ; 最终z分量
    
    ;; 存储结果
    vmovsd [r8], xmm15       ; 结果.x
    vmovsd [r8 + 8], xmm7     ; 结果.y
    vmovsd [r8 + 16], xmm8     ; 结果.z
    
    vzeroupper
    ret
asm_quat_rotate_vector_skylake endp

;; void asm_quat_rotate_vector_ice_lake(const struct Quaternion* q, const struct Vector3* v, struct Vector3* result)
;; Ice Lake架构优化的四元数旋转向量函数
asm_quat_rotate_vector_ice_lake proc
    ;; Ice Lake优化版本: 优化指令调度
    vmovaps ymm0, [rcx]      ; 加载四元数
    vmovaps ymm1, [rdx]      ; 加载向量
    
    ;; 提取分量
    vextractf128 xmm2, ymm0, 0  ; q.x, q.y
    vextractf128 xmm3, ymm0, 1  ; q.z, q.w
    vextractf128 xmm4, ymm1, 0  ; v.x, v.y
    vextractf128 xmm5, ymm1, 1  ; v.z, 0
    
    ;; 计算四元数分量的平方
    vmulsd xmm6, xmm2[0], xmm2[0]  ; xx
    vmulsd xmm7, xmm2[8], xmm2[8]  ; yy
    vmulsd xmm8, xmm3[0], xmm3[0]  ; zz
    
    ;; 计算交叉项
    vmulsd xmm9, xmm2[0], xmm2[8]   ; xy
    vmulsd xmm10, xmm2[0], xmm3[0]  ; xz
    vmulsd xmm11, xmm2[8], xmm3[0]  ; yz
    vmulsd xmm12, xmm3[8], xmm2[0]  ; wx
    vmulsd xmm13, xmm3[8], xmm2[8]  ; wy
    vmulsd xmm14, xmm3[8], xmm3[0]  ; wz
    
    ;; 计算结果x分量
    vaddsd xmm15, xmm7, xmm8
    vmulsd xmm15, xmm15, [rel zero_vector + 8]
    vmovsd xmm7, [rel identity_matrix]
    vsubsd xmm15, xmm7, xmm15
    vmulsd xmm15, xmm15, xmm4[0]
    
    vaddsd xmm7, xmm9, xmm14
    vmulsd xmm7, xmm7, [rel zero_vector + 8]
    vmulsd xmm7, xmm7, xmm4[8]
    vaddsd xmm15, xmm15, xmm7
    
    vsubsd xmm7, xmm10, xmm13
    vmulsd xmm7, xmm7, [rel zero_vector + 8]
    vmulsd xmm7, xmm7, xmm5[0]
    vaddsd xmm15, xmm15, xmm7
    
    ;; 计算结果y分量
    vsubsd xmm7, xmm9, xmm14
    vmulsd xmm7, xmm7, [rel zero_vector + 8]
    vmulsd xmm7, xmm7, xmm4[0]
    
    vaddsd xmm8, xmm6, xmm8
    vmulsd xmm8, xmm8, [rel zero_vector + 8]
    vmovsd xmm9, [rel identity_matrix]
    vsubsd xmm8, xmm9, xmm8
    vmulsd xmm8, xmm8, xmm4[8]
    vaddsd xmm7, xmm7, xmm8
    
    vaddsd xmm8, xmm11, xmm12
    vmulsd xmm8, xmm8, [rel zero_vector + 8]
    vmulsd xmm8, xmm8, xmm5[0]
    vaddsd xmm7, xmm7, xmm8
    
    ;; 计算结果z分量
    vaddsd xmm8, xmm10, xmm13
    vmulsd xmm8, xmm8, [rel zero_vector + 8]
    vmulsd xmm8, xmm8, xmm4[0]
    
    vsubsd xmm9, xmm11, xmm12
    vmulsd xmm9, xmm9, [rel zero_vector + 8]
    vmulsd xmm9, xmm9, xmm4[8]
    vaddsd xmm8, xmm8, xmm9
    
    vaddsd xmm9, xmm6, xmm7
    vmulsd xmm9, xmm9, [rel zero_vector + 8]
    vmovsd xmm10, [rel identity_matrix]
    vsubsd xmm9, xmm10, xmm9
    vmulsd xmm9, xmm9, xmm5[0]
    vaddsd xmm8, xmm8, xmm9
    
    ;; 存储结果
    vmovsd [r8], xmm15
    vmovsd [r8 + 8], xmm7
    vmovsd [r8 + 16], xmm8
    
    vzeroupper
    ret
asm_quat_rotate_vector_ice_lake endp

;; void asm_quat_rotate_vector_alder_lake(const struct Quaternion* q, const struct Vector3* v, struct Vector3* result)
;; Alder Lake架构优化的四元数旋转向量函数
asm_quat_rotate_vector_alder_lake proc
    ;; Alder Lake优化版本: 针对混合架构优化
    vmovaps ymm0, [rcx]      ; 加载四元数
    vmovaps ymm1, [rdx]      ; 加载向量
    
    ;; 提取分量
    vextractf128 xmm2, ymm0, 0  ; q.x, q.y
    vextractf128 xmm3, ymm0, 1  ; q.z, q.w
    vextractf128 xmm4, ymm1, 0  ; v.x, v.y
    vextractf128 xmm5, ymm1, 1  ; v.z, 0
    
    ;; 计算四元数分量的平方
    vmulsd xmm6, xmm2[0], xmm2[0]  ; xx
    vmulsd xmm7, xmm2[8], xmm2[8]  ; yy
    vmulsd xmm8, xmm3[0], xmm3[0]  ; zz
    
    ;; 计算交叉项
    vmulsd xmm9, xmm2[0], xmm2[8]   ; xy
    vmulsd xmm10, xmm2[0], xmm3[0]  ; xz
    vmulsd xmm11, xmm2[8], xmm3[0]  ; yz
    vmulsd xmm12, xmm3[8], xmm2[0]  ; wx
    vmulsd xmm13, xmm3[8], xmm2[8]  ; wy
    vmulsd xmm14, xmm3[8], xmm3[0]  ; wz
    
    ;; 计算结果x分量
    vaddsd xmm15, xmm7, xmm8
    vmulsd xmm15, xmm15, [rel zero_vector + 8]
    vmovsd xmm7, [rel identity_matrix]
    vsubsd xmm15, xmm7, xmm15
    vmulsd xmm15, xmm15, xmm4[0]
    
    vaddsd xmm7, xmm9, xmm14
    vmulsd xmm7, xmm7, [rel zero_vector + 8]
    vmulsd xmm7, xmm7, xmm4[8]
    vaddsd xmm15, xmm15, xmm7
    
    vsubsd xmm7, xmm10, xmm13
    vmulsd xmm7, xmm7, [rel zero_vector + 8]
    vmulsd xmm7, xmm7, xmm5[0]
    vaddsd xmm15, xmm15, xmm7
    
    ;; 计算结果y分量
    vsubsd xmm7, xmm9, xmm14
    vmulsd xmm7, xmm7, [rel zero_vector + 8]
    vmulsd xmm7, xmm7, xmm4[0]
    
    vaddsd xmm8, xmm6, xmm8
    vmulsd xmm8, xmm8, [rel zero_vector + 8]
    vmovsd xmm9, [rel identity_matrix]
    vsubsd xmm8, xmm9, xmm8
    vmulsd xmm8, xmm8, xmm4[8]
    vaddsd xmm7, xmm7, xmm8
    
    vaddsd xmm8, xmm11, xmm12
    vmulsd xmm8, xmm8, [rel zero_vector + 8]
    vmulsd xmm8, xmm8, xmm5[0]
    vaddsd xmm7, xmm7, xmm8
    
    ;; 计算结果z分量
    vaddsd xmm8, xmm10, xmm13
    vmulsd xmm8, xmm8, [rel zero_vector + 8]
    vmulsd xmm8, xmm8, xmm4[0]
    
    vsubsd xmm9, xmm11, xmm12
    vmulsd xmm9, xmm9, [rel zero_vector + 8]
    vmulsd xmm9, xmm9, xmm4[8]
    vaddsd xmm8, xmm8, xmm9
    
    vaddsd xmm9, xmm6, xmm7
    vmulsd xmm9, xmm9, [rel zero_vector + 8]
    vmovsd xmm10, [rel identity_matrix]
    vsubsd xmm9, xmm10, xmm9
    vmulsd xmm9, xmm9, xmm5[0]
    vaddsd xmm8, xmm8, xmm9
    
    ;; 存储结果
    vmovsd [r8], xmm15
    vmovsd [r8 + 8], xmm7
    vmovsd [r8 + 16], xmm8
    
    vzeroupper
    ret
asm_quat_rotate_vector_alder_lake endp

;; void asm_quat_rotate_vector_zen(const struct Quaternion* q, const struct Vector3* v, struct Vector3* result)
;; AMD Zen架构优化的四元数旋转向量函数
asm_quat_rotate_vector_zen proc
    ;; AMD Zen优化版本: 针对Zen架构优化
    vmovaps ymm0, [rcx]      ; 加载四元数
    vmovaps ymm1, [rdx]      ; 加载向量
    
    ;; 提取分量
    vextractf128 xmm2, ymm0, 0  ; q.x, q.y
    vextractf128 xmm3, ymm0, 1  ; q.z, q.w
    vextractf128 xmm4, ymm1, 0  ; v.x, v.y
    vextractf128 xmm5, ymm1, 1  ; v.z, 0
    
    ;; 计算四元数分量的平方
    vmulsd xmm6, xmm2[0], xmm2[0]  ; xx
    vmulsd xmm7, xmm2[8], xmm2[8]  ; yy
    vmulsd xmm8, xmm3[0], xmm3[0]  ; zz
    
    ;; 计算交叉项
    vmulsd xmm9, xmm2[0], xmm2[8]   ; xy
    vmulsd xmm10, xmm2[0], xmm3[0]  ; xz
    vmulsd xmm11, xmm2[8], xmm3[0]  ; yz
    vmulsd xmm12, xmm3[8], xmm2[0]  ; wx
    vmulsd xmm13, xmm3[8], xmm2[8]  ; wy
    vmulsd xmm14, xmm3[8], xmm3[0]  ; wz
    
    ;; 计算结果x分量
    vaddsd xmm15, xmm7, xmm8
    vmulsd xmm15, xmm15, [rel zero_vector + 8]
    vmovsd xmm7, [rel identity_matrix]
    vsubsd xmm15, xmm7, xmm15
    vmulsd xmm15, xmm15, xmm4[0]
    
    vaddsd xmm7, xmm9, xmm14
    vmulsd xmm7, xmm7, [rel zero_vector + 8]
    vmulsd xmm7, xmm7, xmm4[8]
    vaddsd xmm15, xmm15, xmm7
    
    vsubsd xmm7, xmm10, xmm13
    vmulsd xmm7, xmm7, [rel zero_vector + 8]
    vmulsd xmm7, xmm7, xmm5[0]
    vaddsd xmm15, xmm15, xmm7
    
    ;; 计算结果y分量
    vsubsd xmm7, xmm9, xmm14
    vmulsd xmm7, xmm7, [rel zero_vector + 8]
    vmulsd xmm7, xmm7, xmm4[0]
    
    vaddsd xmm8, xmm6, xmm8
    vmulsd xmm8, xmm8, [rel zero_vector + 8]
    vmovsd xmm9, [rel identity_matrix]
    vsubsd xmm8, xmm9, xmm8
    vmulsd xmm8, xmm8, xmm4[8]
    vaddsd xmm7, xmm7, xmm8
    
    vaddsd xmm8, xmm11, xmm12
    vmulsd xmm8, xmm8, [rel zero_vector + 8]
    vmulsd xmm8, xmm8, xmm5[0]
    vaddsd xmm7, xmm7, xmm8
    
    ;; 计算结果z分量
    vaddsd xmm8, xmm10, xmm13
    vmulsd xmm8, xmm8, [rel zero_vector + 8]
    vmulsd xmm8, xmm8, xmm4[0]
    
    vsubsd xmm9, xmm11, xmm12
    vmulsd xmm9, xmm9, [rel zero_vector + 8]
    vmulsd xmm9, xmm9, xmm4[8]
    vaddsd xmm8, xmm8, xmm9
    
    vaddsd xmm9, xmm6, xmm7
    vmulsd xmm9, xmm9, [rel zero_vector + 8]
    vmovsd xmm10, [rel identity_matrix]
    vsubsd xmm9, xmm10, xmm9
    vmulsd xmm9, xmm9, xmm5[0]
    vaddsd xmm8, xmm8, xmm9
    
    ;; 存储结果
    vmovsd [r8], xmm15
    vmovsd [r8 + 8], xmm7
    vmovsd [r8 + 16], xmm8
    
    vzeroupper
    ret
asm_quat_rotate_vector_zen endp

;; 数学函数

;; double asm_fast_inv_sqrt(double x)
asm_fast_inv_sqrt proc
    ;; 使用RSQRTSS指令快速计算1/sqrt(x)，然后用牛顿迭代法提高精度
    rsqrtss xmm1, xmm0      ; 快速近似1/sqrt(x)
    
    ;; 牛顿迭代法提高精度
    ;; 公式: rsqrt(x) = rsqrt_approx(x) * (1.5 - 0.5 * x * rsqrt_approx(x)^2)
    movss xmm2, xmm1        ; 复制近似值
    mulss xmm2, xmm2        ; rsqrt_approx(x)^2
    mulss xmm2, xmm0        ; x * rsqrt_approx(x)^2
    mulss xmm2, [rel zero_vector + 8] ; 0.5 * x * rsqrt_approx(x)^2
    movss xmm3, [rel zero_vector]    ; 1.5
    subss xmm3, xmm2        ; 1.5 - 0.5 * x * rsqrt_approx(x)^2
    mulss xmm1, xmm3        ; 提高精度的rsqrt(x)
    
    ;; 再次迭代
    movss xmm2, xmm1        ; 复制当前值
    mulss xmm2, xmm2        ; rsqrt(x)^2
    mulss xmm2, xmm0        ; x * rsqrt(x)^2
    mulss xmm2, [rel zero_vector + 8] ; 0.5 * x * rsqrt(x)^2
    movss xmm3, [rel zero_vector]    ; 1.5
    subss xmm3, xmm2        ; 1.5 - 0.5 * x * rsqrt(x)^2
    mulss xmm1, xmm3        ; 最终的rsqrt(x)
    
    ret
asm_fast_inv_sqrt endp

;; double asm_fast_sqrt(double x)
asm_fast_sqrt proc
    ;; 使用SQRTSS指令计算平方根
    sqrtss xmm0, xmm0       ; 计算sqrt(x)
    ret
asm_fast_sqrt endp

;; CPU特性检测函数

;; int asm_has_avx2()
asm_has_avx2 proc
    ;; 使用CPUID检测AVX2支持
    mov eax, 7
    xor ecx, ecx
    cpuid
    test ebx, 1 << 5        ; AVX2 bit (bit 5)
    jz no_avx2
    mov eax, 1
    ret
no_avx2:
    xor eax, eax
    ret
asm_has_avx2 endp

;; int asm_has_avx512()
asm_has_avx512 proc
    ;; 使用CPUID检测AVX-512支持
    mov eax, 7
    xor ecx, ecx
    cpuid
    test ebx, 1 << 16       ; AVX512F bit (bit 16)
    jz no_avx512
    mov eax, 1
    ret
no_avx512:
    xor eax, eax
    ret
asm_has_avx512 endp

;; int asm_get_cpu_architecture()
;; 返回CPU架构ID:
;; 0 = 未知
;; 1 = Intel Skylake/Kaby Lake/Coffee Lake (6-10代)
;; 2 = Intel Ice Lake/Tiger Lake/Rocket Lake (11-12代)
;; 3 = Intel Alder Lake/Raptor Lake (12-13代)
;; 4 = Intel Sapphire Rapids (服务器)
;; 5 = AMD Zen 1
;; 6 = AMD Zen 2
;; 7 = AMD Zen 3
;; 8 = AMD Zen 4
asm_get_cpu_architecture proc
    ;; 保存寄存器
    push rbx
    push rcx
    push rdx
    push rdi
    
    ;; 检测CPU厂商
    mov eax, 0
    cpuid
    
    ;; 检查是否为Intel CPU
    cmp ebx, 0x756e6547      ; "Genu"
    jne check_amd
    cmp edx, 0x49656e69      ; "ineI"
    jne check_amd
    cmp ecx, 0x6c65746e      ; "ntel"
    je intel_cpu
    
check_amd:
    ;; 检查是否为AMD CPU
    cmp ebx, 0x68747541      ; "Auth"
    jne unknown_cpu
    cmp edx, 0x444d4163      ; "cAMD"
    jne unknown_cpu
    cmp ecx, 0x69746e65      ; "enti"
    je amd_cpu
    
unknown_cpu:
    mov eax, 0
    jmp end_detection
    
intel_cpu:
    ;; Intel CPU处理
    mov eax, 1
    cpuid
    movzx eax, ah            ; 家族ID
    movzx ebx, al            ; 模型ID
    
    ;; 计算完整家族和型号
    cmp al, 0xf
    jle intel_calculate_model
    
    ;; 扩展家族ID
    movzx eax, dh
    shl eax, 4
    or eax, ebx
    
intel_calculate_model:
    ;; 计算完整型号
    movzx ebx, al            ; base_model
    movzx ecx, dl            ; ext_model
    shl ecx, 4
    or ebx, ecx
    
    ;; 根据家族和型号判断Intel架构
    cmp eax, 6
    jne unknown_arch
    
    ;; Skylake/Kaby Lake/Coffee Lake (6-10代): 型号 0x4e, 0x5e, 0x55, 0x8e, 0x9e, 0xa7, 0xaf
    cmp ebx, 0x4e
    je intel_skylake
    cmp ebx, 0x5e
    je intel_skylake
    cmp ebx, 0x55
    je intel_skylake
    cmp ebx, 0x8e
    je intel_skylake
    cmp ebx, 0x9e
    je intel_skylake
    cmp ebx, 0xa7
    je intel_skylake
    cmp ebx, 0xaf
    je intel_skylake
    
    ;; Ice Lake/Tiger Lake/Rocket Lake (11-12代): 型号 0x7d, 0x8c, 0x9a, 0x9c, 0xa5, 0xa6
    cmp ebx, 0x7d
    je intel_ice_lake
    cmp ebx, 0x8c
    je intel_ice_lake
    cmp ebx, 0x9a
    je intel_ice_lake
    cmp ebx, 0x9c
    je intel_ice_lake
    cmp ebx, 0xa5
    je intel_ice_lake
    cmp ebx, 0xa6
    je intel_ice_lake
    
    ;; Alder Lake/Raptor Lake (12-13代): 型号 0x97, 0x9a, 0x9b, 0x9d, 0xa0, 0xa1, 0xa2, 0xb7, 0xba, 0xbb
    cmp ebx, 0x97
    je intel_alder_lake
    cmp ebx, 0x9a
    je intel_alder_lake
    cmp ebx, 0x9b
    je intel_alder_lake
    cmp ebx, 0x9d
    je intel_alder_lake
    cmp ebx, 0xa0
    je intel_alder_lake
    cmp ebx, 0xa1
    je intel_alder_lake
    cmp ebx, 0xa2
    je intel_alder_lake
    cmp ebx, 0xb7
    je intel_alder_lake
    cmp ebx, 0xba
    je intel_alder_lake
    cmp ebx, 0xbb
    je intel_alder_lake
    
    ;; Sapphire Rapids (服务器): 型号 0x8f, 0x90, 0x91, 0x92, 0x93, 0x94, 0x95
    cmp ebx, 0x8f
    je intel_sapphire_rapids
    cmp ebx, 0x90
    je intel_sapphire_rapids
    cmp ebx, 0x91
    je intel_sapphire_rapids
    cmp ebx, 0x92
    je intel_sapphire_rapids
    cmp ebx, 0x93
    je intel_sapphire_rapids
    cmp ebx, 0x94
    je intel_sapphire_rapids
    cmp ebx, 0x95
    je intel_sapphire_rapids
    
    jmp unknown_arch
    
intel_skylake:
    mov eax, 1
    jmp end_detection
intel_ice_lake:
    mov eax, 2
    jmp end_detection
intel_alder_lake:
    mov eax, 3
    jmp end_detection
intel_sapphire_rapids:
    mov eax, 4
    jmp end_detection
    
amd_cpu:
    ;; AMD CPU处理
    mov eax, 0x80000001
    cpuid
    
    ;; 检查AMD CPU是否支持扩展功能
    cmp eax, 0x80000008
    jb unknown_arch
    
    ;; 获取AMD家族和型号
    movzx eax, ah            ; 家族ID
    movzx ebx, al            ; 模型ID
    
    ;; 计算完整家族和型号
    cmp al, 0xf
    jle amd_calculate_model
    
    ;; 扩展家族ID
    movzx eax, dh
    shl eax, 4
    or eax, ebx
    
amd_calculate_model:
    ;; 计算完整型号
    movzx ebx, al            ; base_model
    movzx ecx, dl            ; ext_model
    shl ecx, 4
    or ebx, ecx
    
    ;; 根据家族和型号判断AMD Zen架构
    ;; Zen 1: 家族 0x17, 型号 0x00-0x0f
    ;; Zen 2: 家族 0x17, 型号 0x10-0x1f
    ;; Zen 3: 家族 0x19, 型号 0x00-0x0f
    ;; Zen 4: 家族 0x19, 型号 0x10-0x1f
    
    cmp eax, 0x17
    jne check_zen3
    
    ;; Zen 1 或 Zen 2
    cmp ebx, 0x10
    jl amd_zen1
    jmp amd_zen2
    
check_zen3:
    cmp eax, 0x19
    jne unknown_arch
    
    ;; Zen 3 或 Zen 4
    cmp ebx, 0x10
    jl amd_zen3
    jmp amd_zen4
    
amd_zen1:
    mov eax, 5
    jmp end_detection
amd_zen2:
    mov eax, 6
    jmp end_detection
amd_zen3:
    mov eax, 7
    jmp end_detection
amd_zen4:
    mov eax, 8
    jmp end_detection
    
unknown_arch:
    mov eax, 0
    
end_detection:
    ;; 恢复寄存器
    pop rdi
    pop rdx
    pop rcx
    pop rbx
    ret
asm_get_cpu_architecture endp

end
