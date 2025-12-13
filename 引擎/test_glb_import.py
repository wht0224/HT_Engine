"""
测试GLB模型导入功能
"""

import sys
import os

# 添加引擎路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Engine.Renderer.Resources.ModelLoader import ModelLoader

def test_glb_import(glb_path):
    """测试GLB文件导入"""
    print(f"正在测试导入GLB文件: {glb_path}")
    print("-" * 60)

    # 检查文件是否存在
    if not os.path.exists(glb_path):
        print(f"[错误] 文件不存在: {glb_path}")
        return False

    print(f"[OK] 文件存在")
    print(f"文件大小: {os.path.getsize(glb_path) / 1024:.2f} KB")

    # 尝试加载模型
    try:
        mesh = ModelLoader.load_model(glb_path)

        if mesh:
            print(f"\n[成功] GLB模型加载成功！")
            print(f"顶点数: {len(mesh.vertices)}")
            print(f"法线数: {len(mesh.normals)}")
            print(f"UV数: {len(mesh.uvs)}")
            print(f"索引数: {len(mesh.indices)}")
            print(f"三角形数: {len(mesh.indices) // 3}")

            if mesh.bounding_box:
                print(f"\n包围盒:")
                print(f"  最小点: {mesh.bounding_box.min}")
                print(f"  最大点: {mesh.bounding_box.max}")

            return True
        else:
            print(f"\n[失败] 模型加载返回None")
            return False

    except Exception as e:
        print(f"\n[异常] 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 测试用户的GLB文件
    # 请将下面的路径替换为实际的GLB文件路径

    # 示例1: 如果GLB文件在桌面
    # glb_path = r"C:\Users\Administrator\Desktop\未标题.glb"

    # 示例2: 如果用户提供了路径
    if len(sys.argv) > 1:
        glb_path = sys.argv[1]
    else:
        print("请指定GLB文件路径:")
        print("用法: python test_glb_import.py <GLB文件路径>")
        print("\n或者直接编辑此脚本，设置 glb_path 变量")
        sys.exit(1)

    test_glb_import(glb_path)
