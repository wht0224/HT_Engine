#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
引擎启动脚本
Engine Startup Script
这是引擎的主要启动入口，负责初始化引擎并进入主循环。
This is the main startup entry for the engine, responsible for initializing the engine and entering the main loop.
"""

import sys, os, traceback

# 添加引擎模块路径
engine_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(engine_path)

from Engine import get_engine

def main():
    """
    主函数，负责启动和运行引擎
    Main function, responsible for starting and running the engine
    """
    print("开始启动引擎...")
    print("Starting engine...")

    # 初始化引擎实例
    engine = get_engine()

    try:
        print("初始化引擎...")
        print("Initializing engine...")
        engine.initialize()

        if not engine.is_initialized:
            print("引擎初始化失败！")
            print("Engine initialization failed!")
            return 1

        print("引擎启动成功！")
        print("Engine started successfully!")

        # 如果有Tkinter UI，使用Tkinter主循环
        if hasattr(engine, 'tk_ui') and engine.tk_ui:
            print("进入Tkinter UI主循环...")
            print("Entering Tkinter UI main loop...")

            # 设置引擎更新回调
            def engine_update():
                """引擎更新循环（60 FPS）"""
                try:
                    # 更新引擎（固定时间步长）
                    engine.update(1/60)

                    # 渲染场景（由OpenGL视口的redraw自动调用）
                    # engine.render() 不需要手动调用，pyopengltk会自动触发redraw

                    # 更新UI状态
                    if hasattr(engine.tk_ui, 'update'):
                        engine.tk_ui.update()

                    # 16ms后再次调用（约60 FPS）
                    engine.tk_ui.root.after(16, engine_update)
                except Exception as e:
                    print(f"引擎更新错误: {e}")
                    traceback.print_exc()

            # 启动引擎更新循环
            engine.tk_ui.root.after(16, engine_update)

            # 进入Tkinter主循环（阻塞直到窗口关闭）
            engine.tk_ui.running = True
            engine.tk_ui.root.mainloop()

        else:
            # 无UI模式（降级方案，使用传统循环）
            print("警告：未检测到Tkinter UI，使用无UI模式")
            print("Warning: Tkinter UI not detected, running in headless mode")

            frame = 0
            while engine.plt.is_window_open():
                engine.update(1/60)
                engine.render()

                # 交换缓冲区（Platform.py中已经是空操作）
                if hasattr(engine.plt, 'swap_buffers'):
                    engine.plt.swap_buffers()

                frame += 1
                if frame % 60 == 0:
                    pass  # 可以在这里打印调试信息

    except Exception as e:
        print(f"引擎运行错误：{e}")
        print(f"Engine run error: {e}")
        traceback.print_exc()
        return 1
    except KeyboardInterrupt:
        print("\n接收到中断信号，正在关闭引擎...")
        print("\nReceived interrupt signal, shutting down engine...")
    finally:
        # 清理资源
        print("开始关闭引擎...")
        print("Starting to shutdown engine...")
        engine.shutdown()
        print("引擎已关闭")
        print("Engine shut down")

    return 0

# 程序入口
if __name__ == "__main__":
    sys.exit(main())