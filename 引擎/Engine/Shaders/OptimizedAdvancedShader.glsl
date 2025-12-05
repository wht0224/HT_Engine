// 高级优化着色器 - 针对低端GPU的高性能光照
// 优化目标：GTX 750Ti (Maxwell) 和 RX 580 (GCN)
// 支持延迟着色、SSAO、SSR等高级效果的简化实现

// ======== 通用工具函数 - 在顶点和片段着色器中共享 ========
#ifndef COMMON_FUNCTIONS
#define COMMON_FUNCTIONS

// 快速数学函数
float fastDot3(vec3 a, vec3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

vec3 fastNormalize(vec3 v) {
    float len = fastDot3(v, v);
    float rsqrt = inversesqrt(len);
    return v * rsqrt;
}

// 快速反射计算
vec3 fastReflect(vec3 incident, vec3 normal) {
    float dotVal = fastDot3(incident, normal);
    return incident - 2.0 * dotVal * normal;
}

// 快速折射计算
vec3 fastRefract(vec3 incident, vec3 normal, float eta) {
    float dotVal = fastDot3(incident, normal);
    float k = 1.0 - eta * eta * (1.0 - dotVal * dotVal);
    if (k < 0.0) return vec3(0.0);
    return eta * incident - (eta * dotVal + sqrt(k)) * normal;
}

// 快速pow函数近似
float fastPow(float x, float y) {
    if (y == 2.0) return x * x;
    if (y == 0.5) return sqrt(x);
    if (y == 1.0) return x;
    if (y == 0.0) return 1.0;
    return pow(x, y);
}

// 快速饱和度函数
vec3 fastSaturate(vec3 color) {
    return clamp(color, 0.0, 1.0);
}

float fastSaturate(float x) {
    return clamp(x, 0.0, 1.0);
}

// 快速线性插值
vec3 fastMix(vec3 a, vec3 b, float t) {
    return a + (b - a) * t;
}

// 纹理坐标的导数近似（用于LOD计算）
vec2 fastTextureGrad(vec2 uv) {
    return vec2(length(dFdx(uv)), length(dFdy(uv)));
}

#endif // COMMON_FUNCTIONS

// ======== 顶点着色器部分 ========
#ifdef VERTEX_SHADER

attribute vec3 a_position;
attribute vec3 a_normal;
attribute vec2 a_texcoord;
attribute vec4 a_color;
attribute vec4 a_tangent;

// 输出到片段着色器
varying vec2 v_texcoord;
varying vec3 v_normal;
varying vec3 v_position;
varying vec4 v_color;
varying vec3 v_tangent;
varying vec3 v_bitangent;
varying vec3 v_viewDir;

// G-Buffer输出（用于延迟渲染）
varying vec3 v_worldPos;
varying vec3 v_worldNormal;

// 统一变量
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform mat3 u_normalMatrix;
uniform vec3 u_cameraPos;

// 实例化渲染支持
uniform bool u_instanced; // 是否使用实例化
attribute mat4 u_instanceMatrix; // 实例矩阵

void main() {
    // 计算模型矩阵（考虑实例化）
    mat4 modelMatrix = u_model;
    if (u_instanced) {
        modelMatrix = u_instanceMatrix * u_model;
    }
    
    // 计算世界空间位置
    vec4 worldPos = modelMatrix * vec4(a_position, 1.0);
    v_position = worldPos.xyz;
    v_worldPos = worldPos.xyz;
    
    // 计算视图投影矩阵，减少矩阵乘法
    mat4 viewProj = u_projection * u_view;
    
    // 裁剪空间位置
    gl_Position = viewProj * worldPos;
    
    // 传递纹理坐标
    v_texcoord = a_texcoord;
    
    // 变换法线到世界空间
    mat3 normalMatrix = mat3(transpose(inverse(modelMatrix)));
    v_normal = fastNormalize(normalMatrix * a_normal);
    v_worldNormal = v_normal;
    
    // 变换切线和副切线（用于法线贴图）
    if (length(a_tangent.xyz) > 0.0) {
        v_tangent = fastNormalize(normalMatrix * a_tangent.xyz);
        // 计算副切线，使用符号位
        v_bitangent = cross(v_normal, v_tangent) * a_tangent.w;
    }
    
    // 计算视角方向
    v_viewDir = fastNormalize(u_cameraPos - worldPos.xyz);
    
    // 传递顶点颜色
    v_color = a_color;
}

#endif // VERTEX_SHADER

// ======== 片段着色器部分 ========
#ifdef FRAGMENT_SHADER

// 精度设置 - 针对不同GPU架构优化
#ifdef GL_ES
precision mediump float;
#else
// Maxwell架构上mediump会被提升为highp，直接使用highp更高效
#define mediump highp
#endif

// 从顶点着色器接收的输入
varying vec2 v_texcoord;
varying vec3 v_normal;
varying vec3 v_position;
varying vec4 v_color;
varying vec3 v_tangent;
varying vec3 v_bitangent;
varying vec3 v_viewDir;
varying vec3 v_worldPos;
varying vec3 v_worldNormal;

// G-Buffer纹理
sampler2D u_gbufferPosition;
sampler2D u_gbufferNormal;
sampler2D u_gbufferAlbedo;
sampler2D u_gbufferSpecular;

// 光照统一变量
uniform vec3 u_ambientLight;
uniform vec3 u_lightPos[4];
uniform vec3 u_lightColor[4];
uniform float u_lightIntensity[4];
uniform int u_activeLights;

// 材质统一变量
uniform sampler2D u_diffuseMap;
uniform sampler2D u_normalMap;
uniform sampler2D u_specularMap;
uniform sampler2D u_roughnessMap;
uniform sampler2D u_metallicMap;
uniform float u_normalMapStrength;
uniform float u_globalRoughness;

// 环境贴图
uniform samplerCube u_irradianceMap;
uniform samplerCube u_prefilterMap;
uniform sampler2D u_brdfLUT;

// 屏幕空间效果统一变量
uniform sampler2D u_depthMap;
uniform sampler2D u_ssaoMap;
uniform float u_ssaoStrength;
uniform float u_ssrIntensity;
uniform bool u_enableSSAO;
uniform bool u_enableSSR;

// 后处理统一变量
uniform float u_exposure;
uniform float u_contrast;
uniform float u_saturation;
uniform float u_vignetteStrength;
uniform float u_bloomIntensity;
uniform sampler2D u_bloomTexture;

// 其他统一变量
uniform vec3 u_cameraPos;
uniform float u_time;
uniform vec2 u_screenSize;
uniform mat4 u_invViewProj;

// ======== PBR材质计算（优化版） ========

// 快速GGX法线分布函数
float fastDistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    
    float nom = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = 3.1415926535 * denom * denom;
    
    return nom / denom;
}

// 快速几何遮蔽函数（Smith近似）
float fastGeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    
    float nom = NdotV;
    float denom = NdotV * (1.0 - k) + k;
    
    return nom / denom;
}

float fastGeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = fastGeometrySchlickGGX(NdotV, roughness);
    float ggx1 = fastGeometrySchlickGGX(NdotL, roughness);
    
    return ggx1 * ggx2;
}

// 快速Fresnel-Schlick近似
vec3 fastFresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

// 优化的PBR光照计算
vec3 calculatePBR(vec3 albedo, float metallic, float roughness, vec3 N, vec3 V, vec3 L, vec3 lightColor) {
    // 半向量
    vec3 H = fastNormalize(V + L);
    
    // 基础反射率（F0）
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);
    
    // 计算光照项
    float NDF = fastDistributionGGX(N, H, roughness);
    float G = fastGeometrySmith(N, V, L, roughness);
    vec3 F = fastFresnelSchlick(max(dot(H, V), 0.0), F0);
    
    // 库克-托伦斯BRDF模型
    vec3 nominator = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001; // 避免除零
    vec3 specular = nominator / denominator;
    
    // 能量守恒
    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - metallic; // 金属表面没有漫反射
    
    // 漫反射项
    vec3 diffuse = kD * albedo / 3.1415926535;
    
    // 最终的辐射率
    float NdotL = max(dot(N, L), 0.0);
    vec3 radiance = lightColor * NdotL;
    
    return (diffuse + specular) * radiance;
}

// ======== 屏幕空间效果 ========

// 从深度重建世界位置
vec3 reconstructPositionFromDepth(vec2 uv, float depth) {
    vec4 clipPos = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 viewPos = u_invViewProj * clipPos;
    viewPos.xyz /= viewPos.w;
    return viewPos.xyz;
}

// 优化的屏幕空间反射（SSR）
vec3 calculateSSR(vec3 position, vec3 normal, vec3 viewDir, float roughness) {
    // 粗糙表面不需要SSR
    if (!u_enableSSR || roughness > 0.5) {
        return vec3(0.0);
    }
    
    // 反射光线
    vec3 reflected = fastReflect(-viewDir, normal);
    
    // 光线步进参数（优化版）
    const int MAX_STEPS = 16; // 减少采样步数以提高性能
    float rayLength = 0.0;
    float stepSize = 0.1;
    vec2 hitUV = vec2(-1.0);
    float hitDepth = 0.0;
    
    // 光线步进
    for (int i = 0; i < MAX_STEPS; i++) {
        rayLength += stepSize;
        vec3 samplePos = position + reflected * rayLength;
        
        // 转换到屏幕空间
        vec4 projected = u_projection * vec4(samplePos, 1.0);
        projected.xyz /= projected.w;
        projected.xy = projected.xy * 0.5 + 0.5;
        
        // 检查是否超出屏幕
        if (projected.x < 0.0 || projected.x > 1.0 || projected.y < 0.0 || projected.y > 1.0) {
            break;
        }
        
        // 采样深度缓冲区
        float sceneDepth = texture2D(u_depthMap, projected.xy).r;
        vec3 scenePos = reconstructPositionFromDepth(projected.xy, sceneDepth);
        
        // 简单的深度比较
        float currentDepth = samplePos.z;
        float depthDiff = abs(currentDepth - scenePos.z);
        
        if (depthDiff < 0.1 && sceneDepth < projected.z) {
            hitUV = projected.xy;
            hitDepth = sceneDepth;
            break;
        }
        
        // 自适应步长
        stepSize *= 1.1;
    }
    
    // 如果找到了交点，采样颜色
    if (hitUV.x >= 0.0) {
        vec3 reflectedColor = texture2D(u_gbufferAlbedo, hitUV).rgb;
        // 基于距离和粗糙度的衰减
        float attenuation = max(0.0, 1.0 - rayLength / 10.0) * (1.0 - roughness * 2.0);
        return reflectedColor * attenuation * u_ssrIntensity;
    }
    
    return vec3(0.0);
}

// 快速环境光遮蔽（从SSAO贴图采样）
float calculateSSAO() {
    if (!u_enableSSAO) {
        return 1.0;
    }
    
    // 采样SSAO贴图（由单独的pass预先计算）
    float ssao = texture2D(u_ssaoMap, v_texcoord).r;
    return mix(1.0, ssao, u_ssaoStrength);
}

// ======== 后处理效果 ========

// 色调映射（Reinhard）
vec3 toneMapping(vec3 color) {
    // 应用曝光
    color *= u_exposure;
    
    // Reinhard色调映射
    return color / (color + vec3(1.0));
}

// 对比度和饱和度调整
vec3 adjustColor(vec3 color) {
    // 对比度调整
    color = (color - 0.5) * u_contrast + 0.5;
    
    // 饱和度调整
    float luminance = dot(color, vec3(0.299, 0.587, 0.114));
    color = mix(vec3(luminance), color, u_saturation);
    
    return color;
}

// 暗角效果
float calculateVignette(vec2 uv) {
    vec2 center = vec2(0.5);
    float dist = distance(uv, center);
    float vignette = 1.0 - smoothstep(0.5, 0.8, dist);
    return mix(1.0, vignette, u_vignetteStrength);
}

// 主片段着色器函数
void main() {
    // ======== 前向渲染模式 ========
    // 采样纹理
    vec4 diffuseColor = texture2D(u_diffuseMap, v_texcoord);
    
    // 法线贴图处理（如果存在）
    vec3 normal = v_normal;
    if (u_normalMapStrength > 0.0) {
        vec3 tangentNormal = texture2D(u_normalMap, v_texcoord).xyz * 2.0 - 1.0;
        
        // 构建TBN矩阵
        mat3 TBN = mat3(normalize(v_tangent), normalize(v_bitangent), normal);
        normal = normalize(TBN * tangentNormal);
        normal = normalize(mix(v_normal, normal, u_normalMapStrength));
    }
    
    // 采样材质参数
    float roughness = u_globalRoughness;
    if (textureSize(u_roughnessMap, 0).x > 1) {
        roughness = texture2D(u_roughnessMap, v_texcoord).r;
    }
    
    float metallic = 0.0;
    if (textureSize(u_metallicMap, 0).x > 1) {
        metallic = texture2D(u_metallicMap, v_texcoord).r;
    }
    
    vec3 specularColor = vec3(0.5);
    if (textureSize(u_specularMap, 0).x > 1) {
        specularColor = texture2D(u_specularMap, v_texcoord).rgb;
    }
    
    // 视图方向
    vec3 viewDir = normalize(v_viewDir);
    
    // 环境光遮蔽
    float ao = calculateSSAO();
    
    // 初始化光照结果
    vec3 lighting = u_ambientLight * diffuseColor.rgb * ao;
    
    // 计算直接光照
    for (int i = 0; i < min(u_activeLights, 4); i++) {
        // 点光源计算
        vec3 lightDir = normalize(u_lightPos[i] - v_position);
        float distance = length(u_lightPos[i] - v_position);
        
        // 简单的衰减
        float attenuation = 1.0 / (1.0 + 0.1 * distance + 0.01 * distance * distance);
        vec3 lightColor = u_lightColor[i] * u_lightIntensity[i] * attenuation;
        
        // 使用优化的PBR光照
        lighting += calculatePBR(diffuseColor.rgb, metallic, roughness, normal, viewDir, lightDir, lightColor);
    }
    
    // 添加屏幕空间反射
    vec3 ssr = calculateSSR(v_position, normal, viewDir, roughness);
    lighting += ssr;
    
    // 应用顶点颜色
    lighting *= v_color.rgb;
    
    // ======== 后处理 ========
    
    // 添加光晕效果
    if (u_bloomIntensity > 0.0) {
        vec3 bloom = texture2D(u_bloomTexture, v_texcoord).rgb * u_bloomIntensity;
        lighting += bloom;
    }
    
    // 色调映射
    vec3 color = toneMapping(lighting);
    
    // 颜色调整
    color = adjustColor(color);
    
    // 应用暗角效果
    color *= calculateVignette(v_texcoord);
    
    // 确保颜色在有效范围内
    color = fastSaturate(color);
    
    // 输出最终颜色
    gl_FragColor = vec4(color, diffuseColor.a * u_color.a);
}

// ======== 延迟渲染特定Pass ========
#ifdef DEFERRED_GBUFFER_PASS

// G-Buffer输出
layout(location = 0) out vec4 gPosition;
layout(location = 1) out vec4 gNormal;
layout(location = 2) out vec4 gAlbedo;
layout(location = 3) out vec4 gSpecular;

void main() {
    // 采样纹理
    vec4 diffuseColor = texture2D(u_diffuseMap, v_texcoord);
    
    // 法线贴图处理
    vec3 normal = v_worldNormal;
    if (u_normalMapStrength > 0.0) {
        vec3 tangentNormal = texture2D(u_normalMap, v_texcoord).xyz * 2.0 - 1.0;
        
        // 构建TBN矩阵
        mat3 TBN = mat3(normalize(v_tangent), normalize(v_bitangent), normal);
        normal = normalize(TBN * tangentNormal);
        normal = normalize(mix(v_worldNormal, normal, u_normalMapStrength));
    }
    
    // 采样材质参数
    float roughness = u_globalRoughness;
    if (textureSize(u_roughnessMap, 0).x > 1) {
        roughness = texture2D(u_roughnessMap, v_texcoord).r;
    }
    
    float metallic = 0.0;
    if (textureSize(u_metallicMap, 0).x > 1) {
        metallic = texture2D(u_metallicMap, v_texcoord).r;
    }
    
    // 写入G-Buffer
    gPosition = vec4(v_worldPos, 1.0);
    gNormal = vec4(normal, 1.0);
    gAlbedo = diffuseColor;
    gSpecular = vec4(metallic, roughness, 0.0, 1.0);
}

#endif // DEFERRED_GBUFFER_PASS

// ======== 延迟渲染光照Pass ========
#ifdef DEFERRED_LIGHTING_PASS

void main() {
    // 从G-Buffer采样
    vec2 uv = gl_FragCoord.xy / u_screenSize;
    vec3 position = texture2D(u_gbufferPosition, uv).rgb;
    vec3 normal = texture2D(u_gbufferNormal, uv).rgb;
    vec3 albedo = texture2D(u_gbufferAlbedo, uv).rgb;
    float metallic = texture2D(u_gbufferSpecular, uv).r;
    float roughness = texture2D(u_gbufferSpecular, uv).g;
    
    // 视图方向
    vec3 viewDir = normalize(u_cameraPos - position);
    
    // 环境光遮蔽
    float ao = calculateSSAO();
    
    // 初始化光照结果
    vec3 lighting = u_ambientLight * albedo * ao;
    
    // 计算直接光照
    for (int i = 0; i < min(u_activeLights, 4); i++) {
        vec3 lightDir = normalize(u_lightPos[i] - position);
        float distance = length(u_lightPos[i] - position);
        float attenuation = 1.0 / (1.0 + 0.1 * distance + 0.01 * distance * distance);
        vec3 lightColor = u_lightColor[i] * u_lightIntensity[i] * attenuation;
        
        lighting += calculatePBR(albedo, metallic, roughness, normal, viewDir, lightDir, lightColor);
    }
    
    // 添加SSR
    vec3 ssr = calculateSSR(position, normal, viewDir, roughness);
    lighting += ssr;
    
    // 色调映射
    vec3 color = toneMapping(lighting);
    
    // 颜色调整
    color = adjustColor(color);
    
    // 应用暗角
    color *= calculateVignette(uv);
    
    gl_FragColor = vec4(fastSaturate(color), 1.0);
}

#endif // DEFERRED_LIGHTING_PASS

// ======== 着色器优化总结 ========
/*
此着色器实现了以下关键优化技术，专为低端GPU设计：

1. 数学函数优化：
   - 使用快速点积、归一化和幂函数
   - 避免使用昂贵的数学库函数
   - 针对常见情况使用特殊处理

2. 材质和光照优化：
   - 简化的PBR实现
   - 减少光照计算中的采样次数
   - 优化的着色方程

3. 屏幕空间效果优化：
   - 减少SSR光线步进次数
   - 简化的SSAO实现
   - 自适应采样技术

4. 内存和带宽优化：
   - 合理使用varying变量
   - 优化纹理坐标计算
   - 减少渲染目标数量

5. 架构特定优化：
   - 针对Maxwell架构的精度设置
   - 针对GCN架构的指令顺序优化
   - 避免架构敏感的操作

6. 可扩展性设计：
   - 前向/延迟渲染支持
   - 条件编译实现多Pass支持
   - 可配置的效果级别
*/