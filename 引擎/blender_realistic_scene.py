#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超级写实美景生成器
适用于 Blender 4.x
直接在Blender中执行: blender --python blender_realistic_scene.py

或在Blender脚本编辑器中运行
"""

import bpy
import bmesh
import math
import random
from mathutils import Vector

# ============================================================
# 配置参数
# ============================================================

CONFIG = {
    # 地形参数
    'terrain_size': 100,
    'terrain_resolution': 200,  # 降低以提高性能

    # 水体参数
    'water_size': 30,
    'water_height': 0.5,

    # 树木参数
    'num_trees': 20,  # 树木数量

    # 相机参数
    'camera_location': (20, -20, 12),
    'camera_rotation': (math.radians(65), 0, math.radians(45)),

    # 渲染参数
    'render_samples': 256,  # 渲染采样数
    'resolution_x': 1920,
    'resolution_y': 1080,
}

# ============================================================
# 工具函数
# ============================================================

def clear_scene():
    """清除场景中的所有对象"""
    print("清空场景...")
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # 清除孤立数据
    for material in bpy.data.materials:
        if not material.users:
            bpy.data.materials.remove(material)

    for mesh in bpy.data.meshes:
        if not mesh.users:
            bpy.data.meshes.remove(mesh)

    print("✓ 场景已清空")

def simple_noise(x, y, scale=1.0, octaves=4):
    """简单的噪声函数（不依赖mathutils.noise）"""
    result = 0.0
    amplitude = 1.0
    frequency = scale

    for _ in range(octaves):
        # 使用简单的伪随机函数
        nx = x * frequency
        ny = y * frequency

        # 简化的Perlin噪声近似
        n = math.sin(nx * 12.9898 + ny * 78.233) * 43758.5453
        n = n - math.floor(n)

        result += n * amplitude
        amplitude *= 0.5
        frequency *= 2.0

    return result / 2.0  # 归一化到[-1, 1]

# ============================================================
# 场景元素创建
# ============================================================

def create_camera():
    """创建相机"""
    print("创建相机...")
    bpy.ops.object.camera_add(
        location=CONFIG['camera_location'],
        rotation=CONFIG['camera_rotation']
    )
    camera = bpy.context.object
    camera.name = "MainCamera"

    # 设置相机参数
    camera.data.lens = 35
    camera.data.sensor_width = 36
    camera.data.clip_start = 0.1
    camera.data.clip_end = 1000

    # 景深
    camera.data.dof.use_dof = True
    camera.data.dof.focus_distance = 20
    camera.data.dof.aperture_fstop = 2.8

    # 设置为活动相机
    bpy.context.scene.camera = camera

    print("✓ 相机已创建")
    return camera

def create_sun_light():
    """创建太阳光"""
    print("创建太阳光...")
    bpy.ops.object.light_add(
        type='SUN',
        location=(0, 0, 50)
    )
    sun = bpy.context.object
    sun.name = "Sun"
    sun.data.energy = 3.5
    sun.rotation_euler = (math.radians(45), math.radians(30), math.radians(90))
    sun.data.color = (1.0, 0.95, 0.8)

    print("✓ 太阳光已创建")
    return sun

def create_terrain():
    """创建地形"""
    print("创建地形...")
    size = CONFIG['terrain_size']
    res = CONFIG['terrain_resolution']

    # 创建网格
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=res,
        y_subdivisions=res,
        size=size,
        location=(0, 0, 0)
    )

    terrain = bpy.context.object
    terrain.name = "Terrain"

    # 应用噪声高度
    mesh = terrain.data
    for v in mesh.vertices:
        x, y = v.co.x, v.co.y

        # 多层噪声
        h1 = simple_noise(x, y, 0.02, 3) * 6
        h2 = simple_noise(x + 100, y + 100, 0.05, 2) * 2
        h3 = simple_noise(x + 200, y + 200, 0.15, 2) * 0.3

        v.co.z = h1 + h2 + h3

    # 更新网格
    mesh.update()

    # 平滑着色
    bpy.ops.object.shade_smooth()

    # 添加细分修改器
    subdiv = terrain.modifiers.new(name="Subdivision", type='SUBSURF')
    subdiv.levels = 1
    subdiv.render_levels = 2

    print("✓ 地形已创建")
    return terrain

def create_terrain_material(terrain):
    """创建地形材质"""
    print("创建地形材质...")
    mat = bpy.data.materials.new(name="TerrainMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # 清除默认节点
    nodes.clear()

    # 输出
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (600, 0)

    # Principled BSDF
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    principled.location = (300, 0)

    # 纹理坐标
    tex_coord = nodes.new('ShaderNodeTexCoord')
    tex_coord.location = (-800, 0)

    # 噪声纹理（草地）
    noise1 = nodes.new('ShaderNodeTexNoise')
    noise1.location = (-500, 200)
    noise1.inputs['Scale'].default_value = 5.0

    # 颜色渐变
    ramp = nodes.new('ShaderNodeValToRGB')
    ramp.location = (-200, 200)
    ramp.color_ramp.elements[0].color = (0.1, 0.2, 0.05, 1)  # 深绿
    ramp.color_ramp.elements[1].color = (0.3, 0.5, 0.15, 1)  # 浅绿

    # 岩石颜色
    noise2 = nodes.new('ShaderNodeTexNoise')
    noise2.location = (-500, -100)
    noise2.inputs['Scale'].default_value = 3.0

    ramp2 = nodes.new('ShaderNodeValToRGB')
    ramp2.location = (-200, -100)
    ramp2.color_ramp.elements[0].color = (0.3, 0.25, 0.2, 1)
    ramp2.color_ramp.elements[1].color = (0.5, 0.45, 0.4, 1)

    # 混合
    mix = nodes.new('ShaderNodeMixRGB')
    mix.location = (0, 0)

    # 根据Z高度混合
    separate = nodes.new('ShaderNodeSeparateXYZ')
    separate.location = (-500, -300)

    map_range = nodes.new('ShaderNodeMapRange')
    map_range.location = (-300, -300)
    map_range.inputs['From Min'].default_value = 0
    map_range.inputs['From Max'].default_value = 6

    # 连接
    links.new(tex_coord.outputs['Object'], noise1.inputs['Vector'])
    links.new(tex_coord.outputs['Object'], noise2.inputs['Vector'])
    links.new(tex_coord.outputs['Object'], separate.inputs['Vector'])

    links.new(noise1.outputs['Fac'], ramp.inputs['Fac'])
    links.new(noise2.outputs['Fac'], ramp2.inputs['Fac'])

    links.new(separate.outputs['Z'], map_range.inputs['Value'])
    links.new(map_range.outputs['Result'], mix.inputs['Fac'])

    links.new(ramp.outputs['Color'], mix.inputs['Color1'])
    links.new(ramp2.outputs['Color'], mix.inputs['Color2'])

    links.new(mix.outputs['Color'], principled.inputs['Base Color'])
    principled.inputs['Roughness'].default_value = 0.9

    links.new(principled.outputs['BSDF'], output.inputs['Surface'])

    # 应用材质
    if terrain.data.materials:
        terrain.data.materials[0] = mat
    else:
        terrain.data.materials.append(mat)

    print("✓ 地形材质已创建")
    return mat

def create_water():
    """创建水体"""
    print("创建水体...")
    size = CONFIG['water_size']

    bpy.ops.mesh.primitive_plane_add(
        size=size,
        location=(0, 0, CONFIG['water_height'])
    )

    water = bpy.context.object
    water.name = "Water"

    # 细分
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.subdivide(number_cuts=40)
    bpy.ops.object.mode_set(mode='OBJECT')

    # 平滑
    bpy.ops.object.shade_smooth()

    # 细分修改器
    subdiv = water.modifiers.new(name="Subdivision", type='SUBSURF')
    subdiv.levels = 2
    subdiv.render_levels = 3

    print("✓ 水体已创建")
    return water

def create_water_material(water):
    """创建水体材质"""
    print("创建水体材质...")
    mat = bpy.data.materials.new(name="WaterMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    nodes.clear()

    # 输出
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (400, 0)

    # Principled BSDF
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    principled.location = (0, 0)

    # 水的属性
    principled.inputs['Base Color'].default_value = (0.05, 0.2, 0.3, 1)
    principled.inputs['Metallic'].default_value = 0.0
    principled.inputs['Roughness'].default_value = 0.05
    principled.inputs['IOR'].default_value = 1.333
    principled.inputs['Transmission Weight'].default_value = 0.9
    principled.inputs['Alpha'].default_value = 0.9

    # 法线贴图
    noise = nodes.new('ShaderNodeTexNoise')
    noise.location = (-600, -200)
    noise.inputs['Scale'].default_value = 8.0
    noise.inputs['Detail'].default_value = 8.0

    bump = nodes.new('ShaderNodeBump')
    bump.location = (-300, -200)
    bump.inputs['Strength'].default_value = 0.2

    links.new(noise.outputs['Fac'], bump.inputs['Height'])
    links.new(bump.outputs['Normal'], principled.inputs['Normal'])

    links.new(principled.outputs['BSDF'], output.inputs['Surface'])

    # 应用材质
    if water.data.materials:
        water.data.materials[0] = mat
    else:
        water.data.materials.append(mat)

    # 混合模式
    mat.blend_method = 'BLEND'
    mat.shadow_method = 'HASHED'

    print("✓ 水体材质已创建")
    return mat

def create_tree(location, height=5.0, radius=0.3):
    """创建单棵树"""
    x, y, z = location

    # 树干
    bpy.ops.mesh.primitive_cylinder_add(
        radius=radius,
        depth=height,
        location=(x, y, z + height / 2)
    )
    trunk = bpy.context.object
    trunk.name = "TreeTrunk"

    # 树干材质
    trunk_mat = bpy.data.materials.new(name="TrunkMat")
    trunk_mat.use_nodes = True
    trunk_bsdf = trunk_mat.node_tree.nodes.get("Principled BSDF")
    trunk_bsdf.inputs['Base Color'].default_value = (0.3, 0.2, 0.1, 1)
    trunk_bsdf.inputs['Roughness'].default_value = 0.9
    trunk.data.materials.append(trunk_mat)

    # 树冠
    crown_z = z + height * 0.8
    bpy.ops.mesh.primitive_ico_sphere_add(
        subdivisions=2,
        radius=height * 0.5,
        location=(x, y, crown_z)
    )
    crown = bpy.context.object
    crown.name = "TreeCrown"

    # 随机缩放
    crown.scale = (
        random.uniform(0.8, 1.2),
        random.uniform(0.8, 1.2),
        random.uniform(0.9, 1.1)
    )

    # 树叶材质
    crown_mat = bpy.data.materials.new(name="CrownMat")
    crown_mat.use_nodes = True
    crown_bsdf = crown_mat.node_tree.nodes.get("Principled BSDF")
    crown_bsdf.inputs['Base Color'].default_value = (0.1, 0.35, 0.05, 1)
    crown_bsdf.inputs['Roughness'].default_value = 0.8
    crown_bsdf.inputs['Subsurface Weight'].default_value = 0.1
    crown.data.materials.append(crown_mat)

    # 合并
    bpy.ops.object.select_all(action='DESELECT')
    trunk.select_set(True)
    crown.select_set(True)
    bpy.context.view_layer.objects.active = trunk
    bpy.ops.object.join()

    tree = bpy.context.object
    tree.name = f"Tree_{random.randint(1000, 9999)}"

    return tree

def create_forest():
    """创建森林"""
    print(f"创建森林（{CONFIG['num_trees']}棵树）...")
    trees = []
    terrain_size = CONFIG['terrain_size']

    for i in range(CONFIG['num_trees']):
        # 随机位置
        x = random.uniform(-terrain_size / 3, terrain_size / 3)
        y = random.uniform(-terrain_size / 3, terrain_size / 3)

        # 估算高度
        h1 = simple_noise(x, y, 0.02, 3) * 6
        h2 = simple_noise(x + 100, y + 100, 0.05, 2) * 2
        z = h1 + h2

        # 只在较低处种树
        if z < 5:
            height = random.uniform(4, 7)
            radius = random.uniform(0.2, 0.4)
            tree = create_tree((x, y, z), height, radius)
            trees.append(tree)

    print(f"✓ 已创建 {len(trees)} 棵树")
    return trees

def setup_world():
    """设置世界环境"""
    print("设置世界环境...")
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    nodes.clear()

    # 输出
    output = nodes.new('ShaderNodeOutputWorld')
    output.location = (400, 0)

    # 天空纹理
    sky = nodes.new('ShaderNodeTexSky')
    sky.location = (-200, 0)
    sky.sky_type = 'NISHITA'
    sky.sun_elevation = math.radians(40)
    sky.sun_rotation = math.radians(90)
    sky.altitude = 10

    # 背景
    background = nodes.new('ShaderNodeBackground')
    background.location = (0, 0)
    background.inputs['Strength'].default_value = 1.0

    links.new(sky.outputs['Color'], background.inputs['Color'])
    links.new(background.outputs['Background'], output.inputs['Surface'])

    print("✓ 世界环境已设置")

def setup_render():
    """配置渲染设置"""
    print("配置渲染设置...")
    scene = bpy.context.scene

    # 渲染引擎
    scene.render.engine = 'CYCLES'

    # GPU设置
    try:
        cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
        cycles_prefs.compute_device_type = 'CUDA'  # 或 'OPTIX'
        for device in cycles_prefs.devices:
            device.use = True
        scene.cycles.device = 'GPU'
    except:
        scene.cycles.device = 'CPU'

    # 采样
    scene.cycles.samples = CONFIG['render_samples']
    scene.cycles.preview_samples = 128

    # 光追
    scene.cycles.max_bounces = 12
    scene.cycles.diffuse_bounces = 4
    scene.cycles.glossy_bounces = 4
    scene.cycles.transmission_bounces = 12

    # 降噪
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = 'OPENIMAGEDENOISE'

    # 分辨率
    scene.render.resolution_x = CONFIG['resolution_x']
    scene.render.resolution_y = CONFIG['resolution_y']
    scene.render.resolution_percentage = 100

    # 输出
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '16'

    # 色彩管理
    scene.view_settings.view_transform = 'Filmic'
    scene.view_settings.look = 'Medium High Contrast'

    print("✓ 渲染设置已配置")

# ============================================================
# 主函数
# ============================================================

def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("超级写实美景生成器")
    print("=" * 70 + "\n")

    # 1. 清空场景
    clear_scene()

    # 2. 创建相机
    camera = create_camera()

    # 3. 创建光源
    sun = create_sun_light()

    # 4. 创建地形
    terrain = create_terrain()
    create_terrain_material(terrain)

    # 5. 创建水体
    water = create_water()
    create_water_material(water)

    # 6. 创建森林
    trees = create_forest()

    # 7. 设置环境
    setup_world()

    # 8. 配置渲染
    setup_render()

    print("\n" + "=" * 70)
    print("✓ 场景创建完成！")
    print("=" * 70)
    print("\n使用提示：")
    print("  1. 按 F12 开始渲染")
    print("  2. 按 Z 键切换视口着色模式")
    print("  3. 在 Shading 工作空间中可调整材质")
    print("\n性能提示：")
    print("  - 首次渲染建议降低采样数（128）")
    print("  - 可以在脚本顶部 CONFIG 中调整参数")
    print("  - 树木数量影响性能，可适当减少")
    print("\n输出路径：")
    print("  默认保存到 Blender 临时目录")
    print("  可在 Output Properties 中修改\n")

# ============================================================
# 执行
# ============================================================

if __name__ == "__main__":
    main()
