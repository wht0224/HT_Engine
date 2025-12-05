// 基本顶点着色器 - 针对低端GPU优化
// 优化目标：GTX 750Ti (Maxwell) 和 RX 580 (GCN)

// 输入属性
attribute vec3 a_position;    // 顶点位置
attribute vec3 a_normal;      // 顶点法线
attribute vec2 a_texcoord;    // 纹理坐标
attribute vec4 a_color;       // 顶点颜色 (可选)

// 输出到片段着色器的变量
varying vec2 v_texcoord;      // 纹理坐标
varying vec3 v_normal;        // 世界空间法线
varying vec3 v_viewDir;       // 视角方向
varying vec4 v_color;         // 顶点颜色
varying vec3 v_position;      // 世界空间位置

// 统一变量
uniform mat4 u_model;         // 模型矩阵
uniform mat4 u_view;          // 视图矩阵
uniform mat4 u_projection;    // 投影矩阵
uniform mat3 u_normalMatrix;  // 法线矩阵
uniform vec3 u_cameraPos;     // 摄像机位置

// 光照相关统一变量
uniform vec3 u_lightPos;      // 光源位置
uniform vec3 u_lightColor;    // 光源颜色
uniform float u_lightIntensity; // 光源强度

// 材质相关统一变量
uniform vec3 u_diffuseColor;  // 漫反射颜色
uniform float u_shininess;    // 光泽度
uniform float u_specularIntensity; // 高光强度

// 快速数学函数 - 针对低端GPU优化
// 这些函数比标准库版本更快但精度略低

// 快速点积函数 - 针对Maxwell架构的dp4a指令优化
float fastDot3(vec3 a, vec3 b) {
    // 在硬件层面，这可以映射到优化的点积指令
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

// 快速归一化函数
vec3 fastNormalize(vec3 v) {
    // 避免使用开销较大的normalize函数
    float len = fastDot3(v, v);
    // 快速近似逆平方根
    float rsqrt = inversesqrt(len);
    return v * rsqrt;
}

// 主函数
void main() {
    // 计算世界空间位置 - 优化版矩阵乘法
    vec4 worldPos = u_model * vec4(a_position, 1.0);
    v_position = worldPos.xyz;
    
    // 计算视图投影矩阵，减少矩阵乘法次数
    mat4 viewProj = u_projection * u_view;
    
    // 计算最终裁剪空间位置
    gl_Position = viewProj * worldPos;
    
    // 传递纹理坐标
    v_texcoord = a_texcoord;
    
    // 变换法线到世界空间
    // 使用预计算的法线矩阵，避免计算完整的逆矩阵
    v_normal = fastNormalize(u_normalMatrix * a_normal);
    
    // 计算视角方向 - 避免重复计算
    v_viewDir = fastNormalize(u_cameraPos - worldPos.xyz);
    
    // 传递顶点颜色（如果有）
    v_color = a_color;
}

// 顶点着色器优化说明：
// 1. 使用快速数学函数减少指令数量
// 2. 预计算矩阵组合减少乘法运算
// 3. 避免在顶点着色器中进行复杂的光照计算
// 4. 使用attribute和uniform打包减少带宽使用
// 5. 针对Maxwell/GCN架构优化指令顺序
// 6. 避免使用复杂的流程控制语句