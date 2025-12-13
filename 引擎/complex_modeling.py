"""
使用渲染引擎MCP API构建复杂3D场景的示例

演示了如何创建包含多种几何体的3D场景，包括：
- 基础几何体（立方体、球体、圆柱体等）
- 自定义复杂几何体（十二面体）
- 场景组织和装饰元素
"""

import sys
import os
import math

# 添加引擎路径到系统路径
enigne_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, enigne_path)

try:
    # 导入引擎相关模块
    from Engine.MCP import ModelingAPI
    from Engine.Engine import Engine
    from Engine.Math.Math import Vector3, Vector2, Quaternion
    print("引擎模块导入成功")
except Exception as e:
    print(f"导入引擎模块失败: {e}")
    sys.exit(1)

def create_complex_scene():
    """创建复杂的3D场景"""
    print("开始构建3D场景...")
    
    # 配置引擎
    config = {
        "frontend": {
            "type": "tkinter",
            "enable_tkinter": True
        }
    }
    
    try:
        engine = Engine()
        engine.initialize(config)
        print("引擎启动成功")
    except Exception as e:
        print(f"引擎启动失败: {e}")
        return
    
    # 创建建模API
    try:
        modeling = ModelingAPI(engine)
        print("建模API初始化成功")
    except Exception as e:
        print(f"建模API初始化失败: {e}")
        return
    
    print("开始创建场景...")
    
    # 创建地面
    print("创建地面...")
    
    # 创建平面作为地面
    ground = modeling.create_plane(width=20, height=20, name="Ground")
    modeling.add_to_scene(ground, position=(0, -3, 0), name="GroundNode")
    print("地面创建完成")
    
    # 创建建筑结构
    print("创建建筑...")
    
    # 定义建筑位置
    building_positions = [
        (-5, 0, -5), (0, 0, -5), (5, 0, -5),
        (-5, 0, 0),  (5, 0, 0),
        (-5, 0, 5),  (0, 0, 5),  (5, 0, 5)
    ]
    
    for i, pos in enumerate(building_positions):
        # 设置建筑大小
        size = 1.0 + (i % 3) * 0.5
        cube = modeling.create_cube(size=size, name=f"Building{i}")
        modeling.add_to_scene(cube, position=pos, name=f"BuildingNode{i}")
    
    print("建筑创建完成")
    
    # 创建装饰性几何体
    print("创建装饰性几何体...")
    
    # 创建球体阵列
    sphere_positions = [
        (-2, 2, -2), (2, 2, -2),
        (-2, 2, 2),  (2, 2, 2)
    ]
    
    for i, pos in enumerate(sphere_positions):
        sphere = modeling.create_sphere(radius=0.5 + i * 0.2, segments=24, rings=12, name=f"Orb{i}")
        modeling.add_to_scene(sphere, position=pos, name=f"OrbNode{i}")
    
    # 创建圆柱体阵列
    cylinder_positions = [
        (-3, 1, 0), (3, 1, 0),
        (0, 1, -3), (0, 1, 3)
    ]
    
    for i, pos in enumerate(cylinder_positions):
        cylinder = modeling.create_cylinder(radius=0.3, height=1.5, segments=16, name=f"Pillar{i}")
        modeling.add_to_scene(cylinder, position=pos, name=f"PillarNode{i}")
    
    # 创建圆锥体阵列
    cone_positions = [
        (-4, 1.5, -4), (4, 1.5, -4),
        (-4, 1.5, 4),  (4, 1.5, 4)
    ]
    
    for i, pos in enumerate(cone_positions):
        cone = modeling.create_cone(radius=0.4, height=1.2, segments=16, name=f"Cone{i}")
        modeling.add_to_scene(cone, position=pos, name=f"ConeNode{i}")
    
    print("装饰性几何体创建完成")
    
    # 创建自定义复杂模型
    print("创建自定义模型...")
    
    # 创建十二面体
    def create_dodecahedron_vertices():
        """生成十二面体顶点"""
        phi = (1 + math.sqrt(5)) / 2  # 黄金比例
        
        vertices = [
            [0, 1, phi], [0, -1, phi], [1, phi, 0], [-1, phi, 0], [phi, 0, 1],
            [-phi, 0, 1], [1, -phi, 0], [-1, -phi, 0], [phi, 0, -1], [-phi, 0, -1],
            [0, 1, -phi], [0, -1, -phi]
        ]
        
        scale = 1.5
        return [[v[0] * scale, v[1] * scale, v[2] * scale] for v in vertices]
    
    def create_dodecahedron_indices():
        """生成十二面体索引"""
        return [
            0, 2, 4, 0, 4, 3, 0, 3, 5, 0, 5, 2,
            2, 6, 4, 4, 6, 8, 4, 8, 10, 5, 3, 7, 5, 7, 9, 5, 9, 11,
            1, 7, 3, 1, 9, 7, 1, 11, 9, 1, 6, 2, 1, 10, 8, 1, 8, 6
        ]
    
    dodecahedron_vertices = create_dodecahedron_vertices()
    dodecahedron_indices = create_dodecahedron_indices()
    
    dodecahedron = modeling.create_custom_mesh(
        vertices=dodecahedron_vertices,
        indices=dodecahedron_indices,
        name="Dodecahedron"
    )
    modeling.add_to_scene(dodecahedron, position=(0, 3, 0), name="DodecahedronNode")
    
    print("自定义模型创建完成")
    
    # 创建道路标记
    print("创建道路标记...")
    
    # 创建道路标记
    for i in range(-8, 9, 2):
        marker1 = modeling.create_cylinder(radius=0.15, height=0.3, segments=8, name=f"RoadMarkerH{i}")
        modeling.add_to_scene(marker1, position=(i, -2.8, 0), name=f"RoadMarkerHNode{i}")
        
        marker2 = modeling.create_cylinder(radius=0.15, height=0.3, segments=8, name=f"RoadMarkerV{i}")
        modeling.add_to_scene(marker2, position=(0, -2.8, i), name=f"RoadMarkerVNode{i}")
    
    print("道路标记创建完成")
    
    print("复杂场景创建完成")
    print("操作提示:")
    print("- 鼠标右键旋转视角")
    print("- 鼠标中键平移视角")
    print("- 鼠标滚轮缩放")
    print("- 点击工具栏切换渲染模式")
    print("- 探索场景中的各种几何体")
    
    # 启动引擎主循环
    if hasattr(engine, 'tk_ui') and engine.tk_ui:
        def engine_update():
            """引擎更新循环"""
            engine.update(1/60)
            if hasattr(engine.tk_ui, 'update'):
                engine.tk_ui.update()
            engine.tk_ui.root.after(16, engine_update)
        
        print("\n🚀 启动引擎主循环...")
        engine.tk_ui.root.after(16, engine_update)
        engine.tk_ui.root.mainloop()
    else:
        print("⚠️  无法启动主循环，UI组件不可用")
    
    return engine

if __name__ == "__main__":
    print("=" * 60)
    print("📱 渲染引擎MCP复杂场景建模示例")
    print("=" * 60)
    create_complex_scene()
