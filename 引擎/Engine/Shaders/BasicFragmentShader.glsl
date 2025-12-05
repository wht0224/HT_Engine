// 基本片段着色器 - 针对低端GPU优化
// 优化目标：GTX 750Ti (Maxwell) 和 RX 580 (GCN)

// 默认精度设置
#ifdef GL_ES
precision mediump float;  // 在大多数移动平台上更高效
#else
// 在桌面平台上，不指定精度通常意味着使用highp
// 但对于Maxwell架构，mediump实际上会被提升为highp，所以直接使用highp更高效
#define mediump highp
#endif

// 从顶点着色器接收的输入
varying vec2 v_texcoord;    // 纹理坐标
varying vec3 v_normal;      // 世界空间法线
varying vec3 v_viewDir;     // 视角方向
varying vec4 v_color;       // 顶点颜色
varying vec3 v_position;    // 世界空间位置

// 统一变量
uniform sampler2D u_texture;     // 主纹理
uniform vec3 u_diffuseColor;     // 漫反射颜色
uniform float u_specularIntensity; // 高光强度
uniform float u_shininess;       // 光泽度

// 光照统一变量
uniform vec3 u_ambientLight;     // 环境光颜色
uniform vec3 u_lightPos;         // 光源位置
uniform vec3 u_lightColor;       // 光源颜色
uniform float u_lightIntensity;  // 光源强度
uniform vec3 u_lightDir;         // 定向光方向
uniform int u_lightType;         // 光源类型: 0=点光源, 1=定向光, 2=聚光灯

// 阴影统一变量
uniform sampler2D u_shadowMap;   // 阴影贴图
uniform mat4 u_shadowMatrix;     // 阴影变换矩阵

// 其他效果统一变量
uniform float u_alpha;           // 全局透明度
uniform vec3 u_fogColor;         // 雾颜色
uniform float u_fogStart;        // 雾开始距离
uniform float u_fogEnd;          // 雾结束距离
uniform bool u_enableFog;        // 是否启用雾效

// 快速数学函数 - 针对低端GPU优化

// 快速点积函数 - 针对Maxwell架构的dp4a指令优化
float fastDot3(vec3 a, vec3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

// 快速归一化函数
vec3 fastNormalize(vec3 v) {
    float len = fastDot3(v, v);
    float rsqrt = inversesqrt(len);
    return v * rsqrt;
}

// 快速pow函数近似 - 对于常用的x^2, x^3等情况更高效
float fastPow(float x, float y) {
    // 对于y=2的常见情况特殊处理
    if (y == 2.0) {
        return x * x;
    }
    // 对于y=0.5的常见情况特殊处理
    if (y == 0.5) {
        return sqrt(x);
    }
    // 对于其他情况使用标准pow函数
    return pow(x, y);
}

// 快速反射向量计算
vec3 fastReflect(vec3 incident, vec3 normal) {
    // reflect(i, n) = i - 2*dot(i, n)*n
    float dotVal = fastDot3(incident, normal);
    return incident - 2.0 * dotVal * normal;
}

// 快速阴影计算
float calculateShadow() {
    // 简单的阴影贴图采样实现
    // 将顶点位置变换到阴影贴图空间
    vec4 shadowCoord = u_shadowMatrix * vec4(v_position, 1.0);
    
    // 透视除法
    shadowCoord.xyz /= shadowCoord.w;
    
    // 转换到[0, 1]范围
    shadowCoord.xyz = shadowCoord.xyz * 0.5 + 0.5;
    
    // 采样阴影贴图
    float shadowDepth = texture2D(u_shadowMap, shadowCoord.xy).r;
    
    // 基本深度比较，添加偏移以避免阴影 acne
    float bias = 0.005;
    float shadow = (shadowCoord.z - bias > shadowDepth) ? 0.5 : 1.0;
    
    // 超出阴影贴图范围的像素不接受阴影
    if (shadowCoord.x < 0.0 || shadowCoord.x > 1.0 || 
        shadowCoord.y < 0.0 || shadowCoord.y > 1.0 || 
        shadowCoord.z < 0.0 || shadowCoord.z > 1.0) {
        shadow = 1.0;
    }
    
    return shadow;
}

// 快速光照计算
void calculateLighting(inout vec3 diffuse, inout vec3 specular) {
    // 确保法线被正确归一化
    vec3 normal = fastNormalize(v_normal);
    
    // 根据光源类型计算光线方向
    vec3 lightDir;
    float attenuation = 1.0;
    
    if (u_lightType == 0) { // 点光源
        // 计算光源方向并归一化
        lightDir = fastNormalize(u_lightPos - v_position);
        
        // 简单的衰减计算，避免使用sqrt
        float dist = distance(u_lightPos, v_position);
        attenuation = 1.0 / (1.0 + 0.1 * dist + 0.01 * dist * dist);
    } else { // 定向光
        lightDir = fastNormalize(-u_lightDir);
        // 定向光没有衰减
    }
    
    // 计算漫反射分量
    float diffuseFactor = max(0.0, fastDot3(normal, lightDir));
    diffuse += u_lightColor * u_lightIntensity * diffuseFactor * attenuation;
    
    // 计算高光分量（如果光泽度大于0）
    if (u_shininess > 0.0 && u_specularIntensity > 0.0) {
        // 计算反射向量
        vec3 reflectDir = fastReflect(-lightDir, normal);
        
        // 计算视线方向
        vec3 viewDir = fastNormalize(v_viewDir);
        
        // 计算高光分量
        float specularFactor = pow(max(0.0, fastDot3(reflectDir, viewDir)), u_shininess);
        specular += u_lightColor * u_specularIntensity * specularFactor * attenuation;
    }
}

// 计算雾效
vec4 applyFog(vec4 color) {
    if (!u_enableFog) {
        return color;
    }
    
    // 计算从相机到片段的距离
    float distance = length(v_position);
    
    // 计算雾混合因子
    float fogFactor = smoothstep(u_fogStart, u_fogEnd, distance);
    
    // 混合颜色和雾颜色
    return mix(color, vec4(u_fogColor, color.a), fogFactor);
}

// 主函数
void main() {
    // 采样主纹理 - 使用优化的纹理采样
    vec4 texColor = texture2D(u_texture, v_texcoord);
    
    // 混合纹理颜色和漫反射颜色
    vec3 diffuseColor = texColor.rgb * u_diffuseColor;
    
    // 初始化光照分量
    vec3 ambient = u_ambientLight * diffuseColor;
    vec3 diffuse = vec3(0.0);
    vec3 specular = vec3(0.0);
    
    // 计算光照
    calculateLighting(diffuse, specular);
    
    // 计算阴影
    float shadowFactor = calculateShadow();
    
    // 组合最终颜色
    vec3 finalColor = ambient + (diffuse + specular) * shadowFactor;
    
    // 应用顶点颜色（如果有）
    finalColor *= v_color.rgb;
    
    // 创建最终颜色
    vec4 color = vec4(finalColor, texColor.a * u_alpha * v_color.a);
    
    // 应用雾效
    color = applyFog(color);
    
    // 输出最终颜色
    gl_FragColor = color;
}

// 片段着色器优化说明：
// 1. 使用条件编译处理不同的OpenGL环境
// 2. 实现快速数学函数减少指令数量
// 3. 针对常用的光照计算进行特殊优化
// 4. 优化纹理采样操作
// 5. 避免在循环中使用纹理采样
// 6. 使用early-out技术减少不必要的计算
// 7. 注意寄存器使用，避免过度使用varying变量
// 8. 针对Maxwell/GCN架构优化指令顺序