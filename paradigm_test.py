##emotion paradigm
import time
import socket
import json
import random
import pygame
from threading import Thread
import cv2  # 用于视频播放和获取时长
from pygame.locals import *
import vlc  # 使用vlc库播放视频和音频
import logging
# 设置日志
logging.basicConfig(
    filename='video.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
# 全局变量
server_socket = None
last_client_address = None
experiment_running = False
current_stimulus = None
screen = None
resting_data_collected = False
vlc_instance = None  # VLC实例
player = None  # VLC播放器

# 视频素材库
negative_videos = [
    {"path": "SEED-IV-Negative1.mp4", "emotion": "negative", "marker": 1},
    {"path": "SEED-IV-Negative2.mp4", "emotion": "negative", "marker": 1},
    {"path": "SEED-IV-Negative3.mp4", "emotion": "negative", "marker": 1},
    {"path": "SEED-IV-Negative4.mp4", "emotion": "negative", "marker": 1},
    {"path": "SEED-IV-Negative5.mp4", "emotion": "negative", "marker": 1},
]

neutral_videos = [
    {"path": "SEED-IV-Neutral1.mp4", "emotion": "neutral", "marker": 2},
    {"path": "SEED-IV-Neutral2.mp4", "emotion": "neutral", "marker": 2},
    {"path": "SEED-IV-Neutral3.mp4", "emotion": "neutral", "marker": 2},
    {"path": "SEED-IV-Neutral4.mp4", "emotion": "neutral", "marker": 2},
    {"path": "SEED-IV-Neutral5.mp4", "emotion": "neutral", "marker": 2},
    {"path": "Neutral_1.mp4", "emotion": "neutral", "marker": 2},
    {"path": "Neutral_2.mp4", "emotion": "neutral", "marker": 2},
    {"path": "Neutral_3.mp4", "emotion": "neutral", "marker": 2},
    {"path": "Neutral_4.mp4", "emotion": "neutral", "marker": 2},
    {"path": "Neutral_5.mp4", "emotion": "neutral", "marker": 2},
]

positive_videos = [
    {"path": "sims_Positive_1.mp4", "emotion": "positive", "marker": 3},
    {"path": "sims_Positive_2.mp4", "emotion": "positive", "marker": 3},
    {"path": "sims_Positive_3.mp4", "emotion": "positive", "marker": 3},
    {"path": "sims_Positive_4.mp4", "emotion": "positive", "marker": 3},
    {"path": "sims_Positive_5.mp4", "emotion": "positive", "marker": 3},
]

# 休息视频
rest_video_negative = {"path": "rest_negative_cn.mp4", "emotion": "rest", "marker": None}
rest_video_neutral = {"path": "rest_neutral_cn.mp4", "emotion": "rest", "marker": None}
rest_video_positive = {"path": "rest_positive_cn.mp4", "emotion": "rest", "marker": None}
rest_video = {"path": "rest.mp4", "emotion": "rest", "marker": None}

def trigger(marker):
    """发送触发器标记到脑电设备"""
    global server_socket, last_client_address

    if not last_client_address:
        print("⚠️ 没有连接的脑电设备!")
        return

    # 获取当前时间戳（精确到毫秒）
    timestamp = time.time()

    response = json.dumps({
        "action": "trigger",
        "marker": marker,
        "timestamp": timestamp,
        "stimulus": current_stimulus["emotion"] if current_stimulus else "resting"
    })

    server_socket.sendto(f"{response}\r\n".encode('utf-8'), last_client_address)
    print(
        f"🚀 发送标记: {marker} | 刺激类型: {current_stimulus['emotion'] if current_stimulus else 'resting'} | 时间戳: {timestamp:.6f}")


def start_experiment():
    """开始实验"""
    global experiment_running
    experiment_running = True
    trigger(100)  # 实验开始标记


def stop_experiment():
    """结束实验"""
    global experiment_running
    experiment_running = False
    trigger(200)  # 实验结束标记


def get_video_duration(path):
    """获取视频的实际时长（秒）"""
    try:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        cap.release()
        return duration
    except:
        print(f"⚠️ 无法获取视频时长: {path}, 使用默认30秒")
        return 30.0


def initialize_vlc_player():
    """初始化VLC播放器并设置全屏模式"""
    global vlc_instance, player

    # 创建VLC实例和播放器
    vlc_instance = vlc.Instance()
    player = vlc_instance.media_player_new()

    # 设置全屏模式
    player.set_fullscreen(True)

def play_video(video):
    """使用vlc库播放视频并精确打标"""
    global current_stimulus, player

    current_stimulus = video

    # 获取视频实际时长
    video_path = video["path"]
    duration = get_video_duration(video_path)
    print(f"▶️ 开始播放: {video_path} | 时长: {duration:.2f}秒 | 情绪: {video['emotion']}")

    # # 加载视频
    media = vlc_instance.media_new(video_path)
    player.set_media(media)

    # 发送开始标记（如果有）
    if video["marker"] is not None:
        trigger(video["marker"])
        start_time = time.time()

    # 开始播放
    player.play()

    # 等待视频播放完成
    while player.get_state() != vlc.State.Ended and experiment_running:
        time.sleep(0.1)

    # 发送结束标记（如果有）
    if video["marker"] is not None:
        actual_duration = time.time() - start_time
        print(f"⏹️ 播放完成 | 实际时长: {actual_duration:.2f}秒 | 理论时长: {duration:.2f}秒")
        trigger(video["marker"]+100)
        # 记录日志
        log_message = f"视频名称: {video_path}, 播放时长: {actual_duration:.2f}秒, 情绪类别: {video['emotion']}, 标签值: {video['marker']}"
        logging.info(log_message)

    # 释放资源
    # player.set_media(None)
    current_stimulus = None

def initialize_pygame_fullscreen():
    """初始化pygame并设置为全屏模式"""
    pygame.init()
    # pygame.display.set_caption("Emotion recognition test")
    pygame.display.set_caption("情绪识别试验")
    # 设置全屏模式
    screen = pygame.display.set_mode((1920, 1080))
    return screen

def collect_resting_data():
    """收集5分钟静息态数据"""
    global resting_data_collected, screen

    if resting_data_collected:
        return

    print("🧠 开始收集5分钟静息态数据...")
    try:
        # 创建静息态屏幕
        screen = pygame.display.set_mode((1920, 1080))
        screen.fill((0, 0, 0))  # 黑色背景

        # 添加说明文字
        # font = pygame.font.SysFont(None, 48)
        font = get_chinese_font(48)
        text = font.render("请闭上双眼，保持身体静止，清空思绪，自然呼吸(5分钟)...", True, (255, 255, 255))
        # text = font.render("Resting state data collection (5 minutes)...", True, (255, 255, 255))
        text_rect = text.get_rect(center=(960, 540))
        screen.blit(text, text_rect)

        # 添加倒计时
        countdown_font = pygame.font.SysFont(None, 72)

        start_time = time.time()
        remaining = 300  # 5分钟 = 300秒
        trigger(5)
        while remaining > 0:
            # 更新倒计时
            screen.fill((0, 0, 0), (760, 640, 400, 160))
            countdown_text = countdown_font.render(f"{remaining}s", True, (255, 255, 255))
            countdown_rect = countdown_text.get_rect(center=(960, 720))
            screen.blit(text, text_rect)
            screen.blit(countdown_text, countdown_rect)
            pygame.display.flip()

            # 检查退出事件
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    return False

            # 计算剩余时间
            elapsed = time.time() - start_time
            remaining = max(0, 300 - int(elapsed))
            time.sleep(0.1)

        print("✅ 静息态数据收集完成")
        trigger(105)
        resting_data_collected = True

    finally:
        pygame.display.quit()  # 只关闭显示，不退出pygame系统
        # 注意：这里不调用pygame.quit()，因为后面还需要用pygame

    return True


import pygame
import os


def get_chinese_font(size=48):
    """获取中文字体"""
    # 常见的中文字体名称，根据你的系统选择
    chinese_fonts = [
        'SimHei',  # 黑体
        'Microsoft YaHei',  # 微软雅黑
        'SimSun',  # 宋体
        'KaiTi',  # 楷体
        'FangSong',  # 仿宋
        'Arial Unicode MS',
        'Noto Sans CJK SC',
        'PingFang SC',  # macOS
        'Heiti SC',  # macOS
    ]

    for font_name in chinese_fonts:
        try:
            font = pygame.font.SysFont(font_name, size)
            # 测试字体是否能渲染中文
            test_surface = font.render('测试', True, (255, 255, 255))
            if test_surface.get_width() > 0:
                return font
        except:
            continue

    # 如果系统字体都不行，使用默认字体
    return pygame.font.Font(None, size)


def run_experiment():
    """运行实验范式"""
    global experiment_running

    # # 收集静息态数据
    # if not collect_resting_data():
    #     return

    # 初始化VLC播放器
    initialize_vlc_player()

    # 开始实验
    start_experiment()

    # # 创建随机顺序的视频列表
    # all_videos = negative_videos + neutral_videos + positive_videos
    # random.shuffle(all_videos)


    # 创建positive-neutral-negative-neutral-positive...顺序的随机视频列表
    all_videos = []
    pos = positive_videos.copy()
    neg = negative_videos.copy()
    neu = neutral_videos.copy()

    # 打乱各自子列表，避免固定顺序
    random.shuffle(pos)
    random.shuffle(neg)
    random.shuffle(neu)

    # 按规则交叉插入
    for i in range(5):
        all_videos.append(pos[i])
        all_videos.append(neu[i])
        all_videos.append(neg[i])
        all_videos.append(neu[5 + i])


    # 运行实验范式
    for i, video in enumerate(all_videos):
        if not experiment_running:
            break

        print(f"\n=== 试验 {i + 1}/20 ===")
        print(f"情绪类型: {video['emotion'].capitalize()} | 标记: {video['marker']}")

        ##################在这里添加逻辑，搞三个带提示语字幕+音频的休息视频，用video['emotion']分割###############
        # 播放休息视频（15秒）
        print("🛌 播放休息视频 (15秒)")
        if video['emotion'] == 'negative':
            play_video(rest_video_negative)
        elif video['emotion'] == 'positive':
            play_video(rest_video_positive)
        else :
            play_video(rest_video_neutral)
        # play_video(rest_video)
        # time.sleep(8)  # 确保15秒间隔

        # 播放情绪视频
        play_video(video)

    play_video(rest_video)

    #释放vlc播放器
    if player:
        player.stop()
        player.release()
        vlc_instance.release()

    # 收集静息态数据
    if not collect_resting_data():
        return

    # 结束实验
    stop_experiment()
    print("\n🎉 实验完成!")


def socket_thread():
    """UDP服务器线程"""
    global server_socket, last_client_address
    server_ip = '127.0.0.1'
    # server_ip = '192.168.32.53'
    # server_ip = '192.168.32.85'
    # server_ip = '192.168.10.5'
    server_port = 9986

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind((server_ip, server_port))
    print(f"📡 UDP服务器监听 {server_ip}:{server_port}...")

    while True:
        message, client_addr = server_socket.recvfrom(1024)
        last_client_address = client_addr
        print(f"📶 收到来自 {client_addr} 的连接")
        trigger(1)  # 发送连接确认标记


def main():
    """主函数"""
    global experiment_running, screen

    # 初始化pygame
    pygame.init()
    # pygame.display.set_caption("Emotion recognition test")
    pygame.display.set_caption("情绪识别试验")
    screen = pygame.display.set_mode((1920, 1080))
    # 启动UDP线程
    thread1 = Thread(target=socket_thread, daemon=True)
    thread1.start()

    # 等待设备连接
    print("⏳ 等待脑电设备连接...")
    while not last_client_address:
        screen.fill((0, 0, 0))
        # font = pygame.font.SysFont(None, 48)
        # text = font.render("Waiting for the EEG device to connect...", True, (255, 255, 255))
        font = get_chinese_font(48)
        text = font.render("等待多模态脑机接口设备连接...", True, (255, 255, 255))
        text_rect = text.get_rect(center=(960, 540))
        screen.blit(text, text_rect)
        pygame.display.flip()

        # 检查退出事件
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                return

        time.sleep(0.5)

    # 显示连接成功
    screen.fill((0, 100, 0))  # 绿色背景
    # font = pygame.font.SysFont(None, 48)
    # text = font.render("Press the spacebar to start the experiment", True, (255, 255, 255))
    font = get_chinese_font(48)
    text = font.render("按下空格开始试验", True, (255, 255, 255))
    text_rect = text.get_rect(center=(960, 540))
    screen.blit(text, text_rect)
    pygame.display.flip()

    # 等待开始指令
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                waiting = False
            elif event.type == KEYDOWN and event.key == K_SPACE:
                waiting = False
                experiment_running = True
                pygame.display.quit()

    # 运行实验
    if experiment_running:
        run_experiment()

    # 清理
    pygame.quit()
    print("程序退出")


if __name__ == '__main__':
    main()