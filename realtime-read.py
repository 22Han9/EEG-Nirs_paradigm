# app data
from brainflow import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes,  AggOperations, NoiseTypes
import time
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pyqtgraph as pg
import threading
from PyQt5.QtWidgets import QApplication, QMainWindow, QScrollArea, QWidget, QVBoxLayout
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
from processing_fNIRS_data import get_fNIRS_data, process_origin_to_fNIRS, get_processing_from_origin_data_48_ch
import scipy.signal as signal
import datetime
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
import requests 
fs = 1000  # 采样频率 (Hz)
f0 = 50    # 要滤除的频率 (Hz)
Q = 30     # 品质因数，控制滤波器的带
b, a = signal.iirnotch(f0, Q, fs)
board = None
boardId = 548
params = BrainFlowInputParams()
params.ip_port = 9527
params.ip_address = "192.168.32.132"
board = BoardShim(int(boardId), params)
data_780 = np.array([])
data_850 = np.array([])
data_eeg = np.array([])
flag = False
showfNIRS = False
data_18_36 = 0
fNIRS_channels= ['FPZ_FP1', 'FPZ_FP2', 'FPZ_AFZ', 'AF7_FP1', 'AF3_FP1', 'AF3_AFZ', 'AF4_FP2', 'AF4_AFZ', 'AF8_FP2', 'FC3_FC5', 'FC3_FC1', 
 'FC3_C3', 'FC4_FC2', 'FC4_FC6', 'FC4_C4', 'C5_FC5', 'C5_C3', 'C5_CP5', 'C1_FC1', 'C1_C3', 'C1_CP1', 'C2_FC2', 'C2_C4', 
 'C2_CP1', 'C6_FC6', 'C6_C4', 'C6_CP6', 'CP3_C3', 'CP3_CP5', 'CP3_CP1', 'CP4_C4', 'CP4_CP2', 'CP4_CP6', 'OZ_POZ', 'OZ_O1', 'OZ_O2']
#最小-最大归一化
def get_battary():
    global params, board
    # 满电视8400 ，即8.4V，电量低于7500就要充电（7.5V），实验前请充满电
    res = requests.get(f"http://{params.ip_address}/BatteryVoltage")
    BatteryVoltage = res.json()['BatteryVoltage']
    print(BatteryVoltage)
get_battary()
def main() :
    global hb, hbr, data_eeg, board, params, data_780, data_850, fNIRS_channels
    board.prepare_session()
    # board.start_stream()
    board.start_stream(num_samples=45000,streamer_params = 'file://./data/test.csv:w')

    # 假设的数据生成函数，用于模拟 fNIRS 信号
    hbr, hb = np.array([]), np.array([])
    time.sleep(3)
    while 1:
        if flag == False :
            continue
        time.sleep(0.4)
        data = board.get_current_board_data(30000)
        eeg = data[1:33, -4000: ]
        # for channel in range(1, 33):
        #     DataFilter.perform_bandpass(data[channel], 1000, 5,  45, 4,
        #                                             FilterTypes.BUTTERWORTH.value, 0)
        #     # DataFilter.perform_bandpass(eeg[channel], 1000, 0.,  30, 4,
        #     #                                         FilterTypes.BUTTERWORTH.value, 0)
        # for channel in range(17, 48):
        #         DataFilter.perform_bandstop(data[channel], 1000, 48, 52, 4,
        #                                             FilterTypes.BUTTERWORTH.value, 0)
                # DataFilter.perform_bandpass(eeg[channel], 1000, 5,  45, 4,
                #                                     FilterTypes.BUTTERWORTH.value, 0)
        fNIRS_channels, data_780, data_850, triggers = get_processing_from_origin_data_48_ch(data, 56)
        if len(data_780) > 0 and len(data_780[0]) > 0:
            data_780 = np.array(data_780)
            data_850 = np.array(data_850)
            minLength = min(data_780.shape[1], data_850.shape[1])
            oxy, deoxy = process_origin_to_fNIRS(data_850[:, :minLength].T, data_780[:, :minLength].T, [850, 780])
            data_eeg = eeg[:, 1000:]
            board.insert_marker(1)
      
def update():
    if flag == False :
        return
    global data_780, data_850, show_wave, hb, hbr, data_eeg, showfNIRS, data_18_36
    if len(data_780)> 0 and len(data_780[1]) > 10 :
        # for i in range(0, 32):
        #     curves_1[i].setData(data_eeg[i])
        for i in range(18, 36):
            if showfNIRS:
                curves_1[i-18].setData(hb[i])
                curves_2[i-18].setData(hbr[i])
            else:
                curves_1[i-18].setData(data_780[i-data_18_36])
                curves_2[i-18].setData(data_850[i-data_18_36])
         


# # # 定义按钮的点击事件
def on_start_clicked():
    print("Start button clicked")
    global flag, data_thread
    flag = True
    data_thread = threading.Thread(target=main, args=())
    data_thread.start()

def on_stop_clicked():
    print("Stop button clicked")    
    global flag , data_thread, board
    flag = False
    data_thread.join()
    data = board.get_board_data()
    board.stop_stream()
    board.release_all_sessions()
    time_now = datetime.datetime.now()
    time_string = time_now.strftime("%Y-%m-%d_%H-%M-%S")
    brainflow_file_name ="./data/BrainFlow-RAW_"+time_string + '_0' + '.csv'
    DataFilter.write_file(data, brainflow_file_name, 'w')

def on_pause_clicked():
    print(" pause button clicked")
    global flag , data_thread
    flag = False
    data_thread.join()

def on_trigger_task_clicked():
    global data_18_36
    # print(" trigger task button clicked")
    # global board
    # board.insert_marker(110)

    # showfNIRS = not showfNIRS
    data_18_36 = abs(data_18_36 - 18)
def on_trigger_rest_clicked():
    global showfNIRS
    print(" trigger rest button clicked")
    # global board
    # board.insert_marker(120)
    showfNIRS = not showfNIRS

def on_trigger_stop_clicked():
    print(" trigger stop button clicked")
    global board
    board.insert_marker(100)

if __name__ == "__main__":
    fNIRS_data_channel = [9, 10, 11, 12, 13, 14, 15, 16]
    # 初始化绘图
    fig, axes = plt.subplots(8, 2, figsize=(10, 8))  # 4x2的子图布局
   
    # 创建应用程序
    app = QtWidgets.QApplication([])

    # 创建主窗口
    main_widget = QtWidgets.QWidget()
    main_layout = QtWidgets.QVBoxLayout()
    main_widget.setLayout(main_layout)
    win = pg.GraphicsLayoutWidget( title="Real-time Waveform Plot")
    win.resize(800, 600)
    win.setWindowTitle('Waveform')
    win.setBackground('#000000')  # 设置背景为白色
    main_layout.addWidget(win)

        # 创建按钮布局
    button_layout = QtWidgets.QHBoxLayout()

    # 创建开始按钮
    start_button = QtWidgets.QPushButton("开始")
    button_layout.addWidget(start_button)

    # 创建暂停按钮
    pause_button = QtWidgets.QPushButton("暂停")
    button_layout.addWidget(pause_button)
    # 创建结束按钮
    stop_button = QtWidgets.QPushButton("结束")
    button_layout.addWidget(stop_button)
    # 创建trigger按钮
    trigger_button_task = QtWidgets.QPushButton("显示下面18个数据")
    button_layout.addWidget(trigger_button_task)

    trigger_button_rest = QtWidgets.QPushButton("显示血氧数据")
    button_layout.addWidget(trigger_button_rest)

      # 创建trigger按钮
    trigger_button_stop = QtWidgets.QPushButton("打标结束开始")
    button_layout.addWidget(trigger_button_stop)

    # 将按钮布局添加到主窗口的底部
    main_layout.addLayout(button_layout)
    # 连接按钮的点击事件
    start_button.clicked.connect(on_start_clicked)
    stop_button.clicked.connect(on_stop_clicked)
    pause_button.clicked.connect(on_pause_clicked)
    trigger_button_task.clicked.connect(on_trigger_task_clicked)
    trigger_button_rest.clicked.connect(on_trigger_rest_clicked)
    trigger_button_stop.clicked.connect(on_trigger_stop_clicked)

    plots = []
    curves_1 = []
    curves_2 = []
    # for i in range(48):
    #     if i >= 32:
    #         p = win.addPlot(row=i//2, col=i%2)
    #         p.showGrid(x=True, y=True)
    #         p.setTitle(f"fNIRS 11")
    #         curve1 = p.plot(pen=pg.mkPen('r', width=2), name="780")  # 使用红色曲线
    #         curve2 = p.plot(pen=pg.mkPen('g', width=2), name="850")  # 使用蓝色曲线
    #         plots.append(p)
    #         curves_1.append(curve1)
    #         curves_2.append(curve2)
    #     else:
    #         p = win.addPlot(row=i//2, col=i%2)
    #         p.showGrid(x=True, y=True)
    #         p.setTitle(f"EEG {i+1}")
    #         curve1 = p.plot(pen=pg.mkPen('r', width=2), name="eeg")  # 使用红色曲线
    #         plots.append(p)
    #         curves_1.append(curve1)

    for i in range(18, 36):
        p = win.addPlot(row=i//2, col=i%2)
        p.showGrid(x=True, y=True)
        p.setTitle(f"{fNIRS_channels[i-data_18_36]}")
        # p.setFixedHeight(250)  # 固定每个绘图的高度
        curve1 = p.plot(pen=pg.mkPen('r', width=2), name="780")  # 使用红色曲线
        curve2 = p.plot(pen=pg.mkPen('g', width=2), name="850")  # 使用蓝色曲线
        plots.append(p)
        curves_1.append(curve1)
        curves_2.append(curve2)
   
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start()
    # 显示主窗口
    main_widget.show()

    # 运行应用程序
    app.exec_()
