import sys
import os
import traceback
import time

# 启用详细的错误输出
sys.tracebacklimit = 1000

# 添加引擎目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("正在启动引擎...")

# 导入引擎并启动
try:
    from Engine.Engine import Engine
    print("成功导入Engine类")
    
    # 创建引擎实例
    engine = Engine()
    print("成功创建Engine实例")
    
    # 初始化引擎
    print("开始初始化引擎...")
    result = engine.initialize()
    print(f"引擎初始化结果: {result}")
    print(f"引擎初始化状态: {engine.is_initialized}")
    
    if engine.is_initialized:
        print("引擎初始化成功！")
        print("按 Ctrl+C 退出")
        
        try:
            # 运行10帧来测试引擎功能
            for i in range(10):
                print(f"运行第 {i+1} 帧")
                engine.update(1/60)
                engine.render()
                # 短暂休眠以模拟实际帧率
                time.sleep(1/60)
            print("主循环运行完成")
        except KeyboardInterrupt:
            print("\n正在关闭引擎...")
        finally:
            # 清理资源
            engine.shutdown()
            print("引擎已关闭")
    else:
        print("引擎初始化失败！")
        error_msg = getattr(engine, '_last_error', '未知错误')
        print(f"引擎错误信息: {error_msg}")
except Exception as e:
    print(f"启动引擎时发生错误: {e}")
    print("详细错误信息:")
    traceback.print_exc()
