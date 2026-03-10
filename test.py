# import eel
import time
import socket
import json
from threading import Thread

# 全局变量修改
board = None
board_id = None
brainflow_file_name = ''
server_socket = None  # 服务器套接字改为全局
last_client_address = None  # 新增用于存储最后连接的客户端地址
count = 0

def trigger(marker):
    global server_socket, last_client_address
    if not last_client_address:  # 没有客户端连接时直接返回
        return
    response = json.dumps({"action": "trigger", "marker": marker})
    # UDP使用sendto发送到指定地址
    server_socket.sendto(f"{response} \r\n".encode('utf-8'), last_client_address)
    return 1

def start():
    global server_socket, last_client_address
    if not last_client_address:
        return
    response = json.dumps({"action": "trigger", "marker": 110})
    server_socket.sendto(f"{response} \r\n".encode('utf-8'), last_client_address)

def stop():
    global server_socket, last_client_address
    if not last_client_address:
        return
    response = json.dumps({"action": "trigger", "marker": 114})
    server_socket.sendto(f"{response} \r\n".encode('utf-8'), last_client_address)

def socket_thread():
    global server_socket, last_client_address
    server_ip = '192.168.32.136'
    server_port = 9986
    buffer_size = 1024

    # 创建UDP套接字
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind((server_ip, server_port))
    print(f"UDP服务器已启动，绑定在 {server_ip}:{server_port}...")

    while True:
        # 接收数据和客户端地址
        message, client_addr = server_socket.recvfrom(buffer_size)
        last_client_address = client_addr  # 更新最后连接的客户端地址
        print('message', message)
        if message:
            trigger(1)
            print(f"来自 {client_addr} 的消息: {message.decode()}")

def main():
    thread1 = Thread(target=socket_thread)
    thread1.start()  # 需要实际启动线程
    count = 0
    while True:
        # 保持主线程运行
        count += 1
        time.sleep(1)
        trigger(count)
if __name__ == '__main__':
    main()