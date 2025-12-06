# 低端GPU优化渲染引擎

![GitHub release](https://img.shields.io/github/release/yourusername/lowend-rendering-engine.svg)
![Python version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 项目简介

这个渲染引擎是我为NVIDIA GTX 750Ti、AMD RX 580这类低端GPU专门开发的，通过一系列优化技术，让这些老旧显卡也能跑出接近高端GPU的视觉效果。

我在开发过程中特别关注了资源限制下的性能表现，针对2GB显存的情况做了很多优化，确保引擎在低配置硬件上也能流畅运行。

## 核心特性

### 硬件优化
- 自适应硬件检测：能自动识别GTX 750Ti、RX 580等不同GPU架构，并应用对应的优化方案
- VRAM智能管理：针对2GB显存做了极限优化，支持纹理压缩和动态加载
- 架构特定优化：针对Maxwell和GCN架构的指令集做了专门优化
- 性能预算系统：严格控制每个渲染阶段的性能开销，确保整体流畅

### 视觉效果系统
- 快速近似全局光照：GPU加速的GI解决方案，耗时不到1毫秒
- 优化的屏幕空间反射：低分辨率、高效采样的SSR实现，兼顾效果和性能
- 轻量级环境光遮蔽：优化的AO效果，在保持视觉质量的同时降低性能开销
- 效果管理器：智能组合和排序各种视觉效果，在性能预算内实现最佳视觉表现

### 性能管理
- 多级LOD系统：自动生成不同细节级别的模型，远距离物体使用低多边形版本，保持视觉一致性
- 实例化渲染：支持大量相同几何体的高效渲染
- 自动批处理：减少渲染状态切换和绘制调用次数
- 动态质量调整：根据实际帧率自动调整视觉质量，保证流畅度

### 材质系统
- 基于物理的材质：在有限光照条件下也能保持真实感
- 纹理压缩：支持BC7、ETC2等高性能压缩格式
- 纹理图集：减少纹理切换和内存碎片，提高渲染效率

## 支持的硬件

### 主要目标平台
- **NVIDIA GTX 750Ti**：512 CUDA核心，2GB GDDR5，86GB/s带宽
- **AMD RX 580**：2304流处理器，4GB GDDR5，256GB/s带宽

### 最低配置要求
- GPU：支持DX11或OpenGL 4.5的显卡，至少1GB VRAM
- CPU：双核处理器
- 内存：4GB系统内存
- 操作系统：Windows 7/10/11，Linux (Ubuntu 18.04+)

## 项目结构

```
引擎/
├── Engine/               # 引擎核心代码
│   ├── Renderer/         # 渲染系统
│   │   ├── Effects/      # 优化的视觉效果
│   │   ├── Shaders/      # 针对低端GPU优化的着色器
│   │   ├── Renderer.py   # 主渲染器类
│   │   └── ...
│   ├── Scene/            # 场景管理系统
│   ├── Resources/        # 资源管理系统
│   ├── Platform/         # 平台抽象层
│   └── Utils/            # 工具类
├── Examples/             # 示例代码
├── Assets/               # 示例资产
├── Documentation/        # 技术文档
├── Tests/                # 性能测试和单元测试
├── build.py              # 构建脚本
└── requirements.txt      # 依赖列表
```

## 安装说明

### 系统要求

- Python 3.8或更高版本
- 确保显卡驱动已更新到最新版本

### 依赖安装

```bash
# 克隆仓库
# git clone <仓库地址>
cd 引擎

# 安装依赖
pip install -r requirements.txt
```

### 安装可选依赖

为了获得最佳性能，可以安装以下可选依赖：

```bash
# 用于JIT优化
pip install numba cython

# 用于纹理压缩
pip install moderngl

# 用于Blender集成（可选）
pip install bpy
```

## 快速开始

### 运行演示程序

```bash
cd Examples
python LowEndOptimizationDemo.py
```

### 创建基本应用

```python
from Engine.Renderer.Renderer import Renderer, RenderQuality
from Engine.Platform.WindowPlatform import WindowPlatform
from Engine.Resources.ResourceManager import ResourceManager
from Engine.Scene.Scene import Scene
from Engine.Scene.Camera import Camera

# 初始化平台
platform = WindowPlatform("低端GPU优化演示", 1280, 720)
platform.initialize()

# 初始化资源管理器和渲染器
resource_manager = ResourceManager(platform)
resource_manager.initialize()

renderer = Renderer(platform, resource_manager)
renderer.initialize()

# 自动优化当前硬件
renderer.optimize_for_hardware()

# 创建场景和相机
scene = Scene()
camera = Camera()
camera.set_position(0, 2, -5)
camera.look_at(0, 0, 0)
scene.set_active_camera(camera)

# 主循环
while running:
    # 渲染场景
    renderer.render(scene)
    platform.swap_buffers()
```

## 渲染优化技术

### 1. 内存优化策略

- **纹理压缩**：使用BC7/ETC2格式减少60-80%的纹理内存占用
- **纹理流式加载**：根据摄像机距离动态加载/卸载纹理
- **几何简化**：自动生成多级LOD，远距离物体使用低多边形模型
- **实例化渲染**：渲染100个相同物体只需1个绘制调用

### 2. 着色器优化

- **半精度计算**：对非关键计算使用FP16减少带宽占用
- **指令合并**：优化着色器指令以减少指令数
- **预计算照明**：烘焙复杂光照以减少实时计算
- **选择性效果**：根据性能预算动态启用/禁用视觉效果

### 3. 低端GPU特定优化

#### NVIDIA Maxwell架构 (GTX 750Ti)
- 优化的纹理采样器配置
- 降低着色器复杂度，优先使用寄存器而不是纹理
- 针对128位内存总线的带宽优化
- 最大使用4个光源进行实时渲染

#### AMD GCN架构 (RX 580)
- 优化计算着色器以充分利用2304流处理器
- 利用异步计算功能并行处理效果
- 纹理过滤优化以匹配256位内存总线
- 支持更多光源和更高分辨率的阴影

## 效果系统指南

### 环境光遮蔽 (AO)

```python
# 启用/禁用AO
renderer.enable_feature('ambient_occlusion', True)

# 在不同质量级别上的性能影响：
# ULTRA_LOW: ~0.3ms, 64x64降采样
# LOW: ~0.5ms, 128x128降采样
# MEDIUM: ~0.8ms, 256x256降采样
# HIGH: ~1.2ms, 512x512降采样
```

### 快速近似全局光照 (FAGI)

```python
# 启用/禁用FAGI
renderer.enable_feature('fagi', True)

# 在不同质量级别上的性能影响：
# ULTRA_LOW: ~0.4ms, 128x128分辨率
# LOW: ~0.8ms, 256x256分辨率
# MEDIUM: ~1.5ms, 512x512分辨率
# HIGH: ~2.5ms, 1024x1024分辨率
```

### 屏幕空间反射 (SSR)

```python
# 启用/禁用SSR
renderer.enable_feature('ssr', True)

# 在低端GPU上的性能优化：
# - 使用射线步进和二分查找减少采样次数
# - 降低反射分辨率
# - 限制反射距离
# - 在GTX 750Ti上默认禁用，可手动启用
```

## 性能调优指南

### 监控性能指标

渲染器会收集以下性能指标：

- **FPS**：每秒帧数
- **VRAM使用**：显存占用（MB）
- **绘制调用数**：CPU到GPU的绘制请求次数
- **三角形数量**：渲染的三角形数量

### 针对GTX 750Ti的建议设置

```python
# 优化GTX 750Ti设置
renderer.set_render_quality(RenderQuality.LOW)
renderer.set_max_draw_calls(200)
renderer.set_max_visible_lights(4)
renderer.set_shadow_map_resolution(1024)

# 选择性启用效果
renderer.enable_feature('ambient_occlusion', True)
renderer.enable_feature('fagi', True)
renderer.enable_feature('ssr', False)  # 对GTX 750Ti影响较大
```

### 针对RX 580的建议设置

```python
# 优化RX 580设置
renderer.set_render_quality(RenderQuality.MEDIUM)
renderer.set_max_draw_calls(500)
renderer.set_max_visible_lights(8)
renderer.set_shadow_map_resolution(2048)

# 可以启用所有效果
renderer.enable_feature('ambient_occlusion', True)
renderer.enable_feature('fagi', True)
renderer.enable_feature('ssr', True)
```

## Blender集成

引擎提供了与Blender的集成工具，可用于：

- 导出优化的模型和材质
- 批量处理纹理压缩
- 生成LOD
- 烘焙光照信息

详细使用方法请参考 `Documentation/Blender集成指南.md`

## 常见问题解答

### Q: 在GTX 750Ti上运行时FPS较低怎么办？

A: 尝试降低渲染质量，禁用SSR效果，减少场景中的光源数量，或启用动态分辨率缩放。

### Q: 如何判断当前硬件的最佳设置？

A: 使用自动优化功能 `renderer.optimize_for_hardware()`，引擎会自动根据检测到的硬件应用最佳配置。

### Q: 纹理加载占用过多内存怎么办？

A: 确保启用了纹理压缩，并调整纹理流设置以降低远处物体的纹理分辨率。

### Q: 如何优化大量相似物体的渲染？

A: 使用实例化渲染，将相似物体添加到同一个实例批次中：

```python
# 创建实例批次
resource_manager.create_instance_batch(similar_objects)
```

## 开发指南

### 贡献代码

欢迎提交Pull Request来改进引擎！在提交之前，请确保：

1. 代码遵循项目的编码规范
2. 测试了在GTX 750Ti和RX 580上的性能
3. 提供了适当的文档

### 性能测试

```bash
cd Tests
python performance_benchmark.py
```

这将运行一系列性能测试，并生成报告显示各种效果和设置的性能影响。

## 许可证

本项目采用MIT许可证 - 详情请查看LICENSE文件

## 联系方式

- 邮箱：[13910593150@139.com]

---

Grant reality, unto all beings!

真正的优化大师能让低端GPU产生高端视觉效果，关键在于理解硬件限制并创造性地绕过它们。
