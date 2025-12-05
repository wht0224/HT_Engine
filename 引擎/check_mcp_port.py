import socket
import sys

def check_port(port, host='localhost'):
    """
    检查指定主机上的端口是否被占用并正在监听
    Check if a port is occupied and listening on the specified host
    
    参数/Parameters:
        port (int): 要检查的端口号/Port number to check
        host (str): 主机名或IP地址/Hostname or IP address
    
    返回/Returns:
        bool: 如果端口正在监听则返回True，否则返回False/True if port is listening, False otherwise
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 设置超时时间为2秒，避免长时间阻塞
    # Set timeout to 2 seconds to avoid long blocking
    sock.settimeout(2)
    
    try:
        # 尝试连接到指定端口
        # Try to connect to the specified port
        result = sock.connect_ex((host, port))
        sock.close()
        # 如果连接成功，返回码为0
        # Return code 0 means connection successful
        return result == 0
    except Exception as e:
        print(f"检查端口时出错: {str(e)}")
        return False

if __name__ == "__main__":
    # MCP服务默认使用的端口号
    # Default port used by MCP service
    default_port = 9875
    
    print(f"正在检查端口 {default_port} 是否被监听...")
    is_listening = check_port(default_port)
    
    if is_listening:
        print(f"端口 {default_port} 正在被监听 - MCP服务可能正在运行")
    else:
        print(f"端口 {default_port} 未被监听 - MCP服务未运行")
        print("\n请启动MCP服务后再运行测试")
    
    # 根据检查结果返回不同的退出码
    # Return different exit codes based on check result
    sys.exit(0 if is_listening else 1)