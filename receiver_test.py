
import socket
import json
import time
from threading import Thread


def udp_receiver():
    """模拟脑电设备接收标记"""
    host = '127.0.0.1'
        # '192.168.32.136'  # 与主程序相同的IP
    receiver_port = 9987  # 接收器使用不同端口
    sender_port = 9986  # 主程序端口

    # 创建UDP套接字
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # 绑定到不同端口避免冲突
    sock.bind((host, receiver_port))  # 随机端口

    print(f"🧪 模拟脑电设备启动 | 监听: {host}:{receiver_port}")

    # 向主程序发送连接请求
    sock.sendto(b"CONNECT", (host, sender_port))

    # 接收消息线程
    def recv_thread():
        while True:
            data, addr = sock.recvfrom(1024)
            if not data:
                continue

            try:
                # 解析JSON消息
                message = data.decode().strip()
                if message.endswith("\r\n"):
                    message = message[:-2]

                event = json.loads(message)
                print(f"📥 收到标记: {event}")

                # 记录到文件
                with open("eeg_markers.log", "a") as f:
                    f.write(f"{time.time()},{event['timestamp']},{event['marker']},{event['stimulus']}\n")

            except json.JSONDecodeError:
                print(f"⚠️ 无效消息: {data.decode()}")

    # 启动接收线程
    Thread(target=recv_thread, daemon=True).start()

    # 保持运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("模拟设备关闭")


if __name__ == '__main__':
    print("=== 脑电数据采集模拟器 ===")
    print("1. 此程序模拟脑电设备接收标记")
    print("2. 将记录所有收到的标记到 eeg_markers.log")
    print("3. 格式: 接收时间,标记值,事件时间戳,刺激类型\n")

    udp_receiver()