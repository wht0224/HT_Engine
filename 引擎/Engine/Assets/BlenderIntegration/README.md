# 低端GPU优化的Blender建模工具

本目录包含为低端GPU（如GTX 750Ti和RX 580）优化的Blender集成工具，旨在提供高性能渲染解决方案，同时保持良好的视觉质量。

## 工具列表

### 1. LowEndModeling.py

这是一个完整的Blender脚本，提供了以下功能：

- 创建优化的低多边形模型
- 应用简化高效的材质系统
- 设置针对低端GPU优化的光照
- 配置Eevee渲染引擎以获得最佳性能
- 添加基础合成效果提升视觉质量

## 使用方法

### 在Blender中使用

1. 打开Blender
2. 进入脚本编辑器（Scripting工作区）
3. 打开 `LowEndModeling.py` 文件
4. 点击"运行脚本"按钮
5. 脚本将自动创建一个优化的演示场景

### 脚本功能说明

脚本创建了一个 `LowEndModeling` 类，提供了多种优化功能：

- **优化级别设置**：
  - 级别1：最高优化，适合GTX 750Ti等最低端GPU
  - 级别2：中等优化，平衡性能和质量
  - 级别3：轻度优化，适合RX 580等相对较好的低端GPU

- **关键优化技术**：
  - 减少多边形数量（Decimation）
  - 简化材质节点树
  - 优化光照计算
  - 降低Eevee渲染采样和高级功能
  - 调整阴影和环境光遮蔽质量

## 渲染性能优化指南

### 针对GTX 750Ti的特别优化

- 设置 `optimization_level = 1`
- 降低渲染分辨率至720p
- 关闭SSR（屏幕空间反射）
- 使用16x采样
- 减少光源数量至2-3个

### 针对RX 580的优化设置

- 设置 `optimization_level = 2` 或 `3`
- 可使用1080p分辨率
- 可启用SSR（屏幕空间反射）
- 使用32x采样

## 自定义场景

要创建自定义场景而不是使用默认演示场景，可以修改脚本末尾：

```python
# 创建低端GPU优化建模工具实例
low_end_modeler = LowEndModeling()

# 设置优化级别
low_end_modeler.optimization_level = 2

# 清理现有场景
low_end_modeler.clear_scene()

# 设置相机和光照
low_end_modeler.setup_camera()
low_end_modeler.setup_optimized_lighting()

# 创建自定义模型
cube = low_end_modeler.create_low_poly_model("cube")
low_end_modeler.create_optimized_material(cube, base_color=(0.8, 0.2, 0.2, 1.0))

# 设置渲染
low_end_modeler.setup_eevee_render()
low_end_modeler.setup_compositing()
```

## 性能基准

- **GTX 750Ti**：优化后可达到30-45 FPS的视口性能
- **RX 580**：优化后可达到50-60 FPS的视口性能

## 已知限制

- 复杂的动态效果可能会降低性能
- 大量的几何体即使优化后也可能导致性能下降
- 某些高级着色器效果已被禁用以提高性能

## 未来计划

- 添加更多低多边形预设模型
- 实现自动LOD（细节层次）系统
- 添加纹理优化和压缩功能
- 增加实时性能监控工具

---

## 联系方式

- 邮件：13910593150@139.com、wswht333@qq.com

**注意**：本工具专为低端GPU设计，在高端GPU上可能无法充分发挥硬件性能。如果您使用的是中高端GPU，可以尝试使用更高的优化级别或修改脚本中的参数以获得更好的视觉质量。