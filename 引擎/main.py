#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
引擎启动脚本
这是引擎的主要启动入口，负责初始化引擎并进入主循环。
"""

import sys
import os
import traceback

# 添加引擎模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Engine import get_engine

def main():
    """主函数，负责启动和运行引擎"""
    print("开始启动引擎...")
    
    # 获取引擎实例
    engine = get_engine()
    
    try:
        # 初始化引擎
        print("初始化引擎...")
        engine.initialize()
        
        if not engine.is_initialized:
            print("引擎初始化失败！")
            return 1
        
        # 检查平台支持
        if not engine.platform.has_graphics:
            print("警告：没有图形支持！")
        else:
            print("图形支持已启用")
        
        if not engine.platform.window_created:
            print("警告：窗口创建失败！")
        else:
            print("窗口创建成功")
        
        print("引擎启动成功！按 ESC 键或关闭窗口退出。")
        
        frame_count = 0
        # 主循环
        while engine.platform.is_window_open():
            # 使用固定时间步长更新
            engine.update(1/60)
            # 渲染当前帧
            engine.render()
            # 交换缓冲区
            if hasattr(engine.platform, 'swap_buffers'):
                engine.platform.swap_buffers()
            
            frame_count += 1
            # 每60帧打印一次
            if frame_count % 60 == 0:
                print(f"已渲染 {frame_count} 帧")
    except Exception as e:
        print(f"引擎运行错误：{e}")
        traceback.print_exc()
        return 1
    except KeyboardInterrupt:
        print("\n接收到中断信号，正在关闭引擎...")
    finally:
        # 清理资源
        print("开始关闭引擎...")
        engine.shutdown()
        print("引擎已关闭")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())