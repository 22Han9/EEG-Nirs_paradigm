#new app processing
import math

import mne
import numpy as np

from preprocessor.procutil_get_extinctions import procutil_get_extinctions
import json


try:
    with open('boardInfo.json', 'r', encoding='utf-8') as f:
        board_info = json.load(f)
except FileNotFoundError:
    print("错误: 请确保 boardInfo.json 文件与当前 Python 文件在同一目录")
    board_info = {}
except Exception as e:
    print(f"加载 boardInfo.json 失败: {e}")
    board_info = {}
boardInfo = board_info
# with open('boardInfo.json', 'r') as f:
#     boardInfo = json.load(f)

# lightChannelName= boardInfo['lightName']
# sensorChannelName= boardInfo['senserName']

def get_fNIRS_data(data, labels_number, marker):
    # Step 1: 基于 labels 数组计算分段的索引范围
    segments_indices = []
    start = 0
    labels = data[labels_number]
    for i in range(0, len(labels)-1):
        if labels[i] != labels[start]:  # 标签发生变化，记录一个段的结束
            segments_indices.append((start, i))
            start = i  # 更新起始索引
    # 最后一个段
    segments_indices.append((start, len(labels)))
    # Step 2: 根据分段索引来分割 data
    segments = []

    for start_idx, end_idx in segments_indices:
        # 对每个分段的起始和结束索引，根据这些索引从 data 中提取对应段的数据
        segment = data[:, start_idx:end_idx]  # 提取通道 * 该段样本的部分
        segments.append(segment)
    averages = []
    for segment in segments:
        segment_samples = segment.shape[1]  # 获取该段的样本数量
        middle_start = segment_samples // 3  # 中间 1/3 的起始位置
        middle_end = 2 * segment_samples // 3  # 中间 1/3 的结束位置
        # 提取中间 1/3 的数据
        middle_segment = segment[:, middle_start:middle_end]
        
        # 计算每个通道的平均值
        average = np.mean(middle_segment, axis=1)  # 对样本维度求平均，保留通道维度
        
        averages.append(average)

    origin_fNIRS_data = np.array(averages).T
    data_780 = origin_fNIRS_data[: ,origin_fNIRS_data[marker] == 1]
    data_850 = origin_fNIRS_data[: ,origin_fNIRS_data[marker] == 2]
    min_length = min(data_780.shape[1], data_850.shape[1])
    data_780 = data_780[:, :min_length]
    data_850 = data_850[:, :min_length]
    return data_780, data_850, []

def get_fNIRS_data_mean(data, labels):
    # Step 1: 基于 labels 数组计算分段的索引范围
    segments_indices = []
    start = 0
    for i in range(0, len(labels)-1):
        if labels[i] != labels[start]:  # 标签发生变化，记录一个段的结束
            segments_indices.append((start, i))
            start = i  # 更新起始索引
    # 最后一个段
    segments_indices.append((start, len(labels)))
    # Step 2: 根据分段索引来分割 data
    segments = []

    for start_idx, end_idx in segments_indices:
        # 对每个分段的起始和结束索引，根据这些索引从 data 中提取对应段的数据
        segment = data[:, start_idx:end_idx]  # 提取通道 * 该段样本的部分
        segments.append(segment)
    averages = []
    for segment in segments:
        segment_samples = segment.shape[1]  # 获取该段的样本数量
        middle_start = segment_samples // 3  # 中间 1/3 的起始位置
        middle_end = 2 * segment_samples // 3  # 中间 1/3 的结束位置
        # 提取中间 1/3 的数据
        middle_segment = segment[:, middle_start:middle_end]
        
        # 计算每个通道的平均值
        average = np.mean(middle_segment, axis=1)  # 对样本维度求平均，保留通道维度
        
        averages.append(average)

    origin_fNIRS_data = np.array(averages).T
    data_780 = origin_fNIRS_data[: ,origin_fNIRS_data[21] == 1]
    data_850 = origin_fNIRS_data[: ,origin_fNIRS_data[21] == 2]
    min_length = min(data_780.shape[1], data_850.shape[1])
    data_780 = data_780[:, :min_length]
    data_850 = data_850[:, :min_length]
    return np.array(data_780, axis=1).tolist(), np.array(data_850, axis=1).tolist()

def process_origin_to_fNIRS(wave1, wave2, waveLength):
    wave2 , wave1 = get_min_length_origin_data(wave1, wave2)
    wave1 = np.array(wave1)
    wave2 = np.array(wave2)
    wave = np.concatenate((wave1, wave2), axis=1)
    data = dict({
        "x":wave,
        'wavelengths': waveLength,
        "clab": []
    })
    fNIRS_signal = proc_BeerLambert(data)
    channel_length = wave1.shape[1]
    return fNIRS_signal["x"][:, :channel_length].T, fNIRS_signal["x"][:,channel_length:].T
   
def proc_BeerLambert(dat, **kwargs):
    """
    PROC_BEERLAMBERT - 使用Beer-Lambert定律分析NIRS数据。
    根据总光路长度计算相对浓度。

    参数：
        dat: 包含NIRS数据的字典。需要有'x'字段，存储分割的NIRS数据。
        kwargs: 可选参数，以键值对形式传递：
            'Citation' - 消光系数的文献编号（默认值1）。
            'Epsilon' - 自定义消光系数（覆盖文献编号）。
            'Opdist' - 探头（光源-探测器）间距，单位为cm（默认值3）。
            'Ival' - 用于LB变换的基线区间（'all'或[start, end]，默认值'all'）。
            'DPF' - 差分路径长度因子（默认值[5.98, 7.15]）。
            'Verbose' - 输出详细信息的级别（默认值0）。

    返回值：
        dat: 更新后的字典，包含氧合血红蛋白和脱氧血红蛋白浓度（单位：mmol/L）。
    """
    # 默认参数设置
    props = {
        'Citation': 1,  # 消光系数的默认文献编号
        'Opdist': 3,    # 探头距离的默认值（cm）
        'Ival': 'all',  # 基线区间的默认值
        'DPF': [5.98, 7.15],  # 差分路径长度因子的默认值
        'Epsilon': None,      # 消光系数
        'Verbose': 0          # 输出详细信息的级别
    }

    # 更新默认参数
    props.update(kwargs)

    # 检查输入数据是否包含'x'字段
    if 'x' not in dat:
        raise ValueError("dat必须包含字段'x'。")

    # 设置基线区间
    if props['Ival'] == 'all':
        props['Ival'] = [0, dat['x'].shape[0]]  # 如果为'all'，则使用整个数据范围
    s1 = dat['x'].shape[0]  # 时间点数量
    s2 = dat['x'].shape[1] // 2  # 每个波长的通道数量

    # 将数据分为低波长和高波长部分
    wl1 = dat['x'][:, :s2]
    wl2 = dat['x'][:, s2:]
    # 获取或使用提供的消光系数
    if props['Epsilon'] is None:
        # 如果没有提供波长信息，则报错
        if 'wavelengths' not in dat:
            raise ValueError("dat字典中必须提供'wavelengths'字段。")
        # 根据文献编号获取消光系数
        ext, citation = procutil_get_extinctions(dat['wavelengths'], props['Citation'])
        if props['Verbose']:
            print(f"使用的文献编号: {citation}")
        epsilon = ext[:, :2] / 1000  # 将单位转换为mmol/L
    else:
        # 使用用户提供的消光系数
        epsilon = np.array(props['Epsilon'])

    # 确保较高波长的消光系数排在顶部
    max_wavelength_idx = np.argmax(dat['wavelengths'])
    if max_wavelength_idx == 1:  # 如果较高波长排在底部
        epsilon = np.flipud(epsilon)  # 调整顺序
        if props['Verbose']:
            print("已调整Epsilon矩阵，使较高波长的消光系数排在顶部。")
    eps = np.finfo(float).eps  # 浮点数的最小正值
    # 计算基线均值，用于归一化
    mean_wl2 = np.mean(wl2[props['Ival'][0]:props['Ival'][1], :], axis=0)
    mean_wl1 = np.mean(wl1[props['Ival'][0]:props['Ival'][1], :], axis=0)
        # 避免分母为零
    mean_wl2 = np.clip(mean_wl2, eps, None)
    mean_wl1 = np.clip(mean_wl1, eps, None)

    # 计算衰减值，避免对数中出现非正值
    Att_highWL = np.real(-np.log10(np.clip(wl2 / mean_wl2, eps, None)))
    Att_lowWL = np.real(-np.log10(np.clip(wl1 / mean_wl1, eps, None)))

    # 准备吸收矩阵，用于线性方程求解
    A = np.zeros((s1 * s2, 2))
    A[:, 0] = Att_highWL.ravel()
    A[:, 1] = Att_lowWL.ravel()

    # 根据DPF和探头距离对消光系数进行缩放
    e2 = epsilon * np.array(props['DPF'])[:, None] * props['Opdist']

    # 使用矩阵求解计算浓度
    c = np.linalg.inv(e2) @ A.T

    # 更新数据字段'x'，包含氧合和脱氧血红蛋白的浓度
    dat['x'] = np.hstack([
        c[0, :].reshape(s1, s2),
        c[1, :].reshape(s1, s2)
    ])
    lowWL= [label[0].replace('lowWL', 'oxy') for label in dat['clab'] if 'lowWL' in label[0]]
    highWL = [label[0].replace('highWL', 'deoxy') for label in dat['clab'] if 'highWL' in label[0]]
    dat['clab'] = np.hstack([
        lowWL, 
        highWL
    ])
    # # 更新数据的元信息
    # dat['signal'] = 'NIRS (oxy, deoxy)'  # 信号类型
    # dat['yUnit'] = 'mmol/L'  # 浓度单位

    return dat

def decimal_to_16bit_array(n):
    if not 0 <= n <= 0xFFFF:
        raise ValueError("数值超出16位二进制范围（0-65535）")
    # 转换为16位二进制字符串，去掉前缀，左侧补零
    binary_str = bin(n)[2:].zfill(16)
    # 转为整数数组
    return [int(bit) for bit in binary_str]

def max_every_10_points(arr):
    result = []  # 存储结果的列表
    n = len(arr)
    i = 0
    while i < n:
        # 取当前区块：从i开始到min(i+10, n)
        block = arr[i:i+10]
        sample = [i for i in block if i != 0]

        # 求区块最大值（可能是0或者trigger）并添加到结果
        if len(sample) == 0:
            # 0表示非trigger数据
            result.append(0)
        elif len(sample) == 1:
            # 大部分情况一个区块中最多有一个 trigger
            result.extend(sample)
        # 由于fnirs采样率比较低，可能在一个区块中设备有多个trigger
        elif len(sample) == 2:
            if result[-1] != 0:
                raise RuntimeError("处理fnirs数据时出现错误：在几个采样点中出现多个(>2)trigger！")

            # 替换前一个为 0 的 trigger
            result[-1] = sample[0]
            result.append(sample[1])
        else:
            raise RuntimeError("处理fnirs数据时出现错误：在几个采样点中出现多个(>2)trigger！")

        i += 10  # 移动到下一个区块起始位置
    return np.array(result)

def get_channel_data_by_marker(marker):
    global boardInfo
    if boardInfo == None:
        with open('boardInfo.json', 'r') as f:
            boardInfo = json.load(f)
    fNIRS_indexs =  []
    light= '780'
    if marker > 29999:
        light= '850'
        light_arr = boardInfo['light_flash_groups'][str(marker-30000 + 1)]
    else:
        light_arr = boardInfo['light_flash_groups'][str(marker + 1)]
    light_arr = [item[1] for item in light_arr]
    for k in range(len(light_arr)):
        i = light_arr[k] 
        fNIRS_indexs.append([boardInfo['lightIndex'][str((i+1 ))],
                             [boardInfo['lightName'][i] +'-' + boardInfo['senserName'][j-1] for j in boardInfo['lightIndex'][str((i+1))]] 
            ])
    return fNIRS_indexs, light

def find_contiguous_segments(data, index):
    """
    data每一行对应一个sample。eeg采样率为1000hz，以eeg为基准，data每1000行对应1秒。
    fnirs通道共分成10组，780波长和850波长各5组。
    各组轮流采样，每次20个sample。完整采完一次fnirs所有通道的数据，需要10组，也就是200个sample。
    data每1000个sample对应一秒，所以fnirs采样率为5hz
    :param data: 原始eeg、fnirs数据，每一行包括
    :param index: fnirs 通道组编号所在列的索引
    :return: fnirs数据（5hz），对齐后的trigger
    """
    global boardInfo
    all_data = dict({})
    all_data['780'] = dict({})
    all_data['850'] = dict({})
    if boardInfo == None:
        with open('boardInfo.json', 'r') as f:
            boardInfo = json.load(f)
    for name in boardInfo['fNIRSChannels']:
        all_data['780'][name] = []
        all_data['850'][name] = []
    chunnels_start = 1
    if data.shape[0] == 64:
        chunnels_start = 33

    col = data[index].tolist()
    if not col:
        return all_data
    segments_indices = []
    start = 0
    for i in range(1, len(col)):
        if col[i] != col[i-1]:
            segments_indices.append([start, i, col[i]])
            start = i
    # 添加最后一个段
    segments_indices.append([start, len(col)-1, col[ len(col)-1]])
    segments = []
    markers = []
    triggers = []

    for start_idx, end_idx, marker in segments_indices:
        # 对每个分段的起始和结束索引，根据这些索引从 data 中提取对应段的数据
        segment = data[:, start_idx:end_idx]  # 提取通道 * 该段样本的部分
        # if segment.shape[1] < 5 and segment.shape[1] > 1:
        #     segment_samples = segment.shape[1]
        #     triggers.append(np.max(segment[-1]))
        #     average = np.median(segment, axis=1, keepdims=True)  # 对样本维度求平均，保留通道维度
        #     markers.append(average[index])
        #     segments.append(average)
        # else:
        if segment.shape[1] > 5:

            segment_samples = segment.shape[1]
            middle_start = segment_samples // 3  # 中间 1/3 的起始位置
            middle_end = 2 * segment_samples // 3  # 中间 1/3 的结束位置
            triggers.append(np.max(segment[-1]))
            middle_segment = segment[:, middle_start:middle_end]
            average = np.average(middle_segment, axis=1, keepdims=True)  # 对样本维度求平均，保留通道维度
            markers.append(average[index])
            segments.append(average)
            
    triggers_array = max_every_10_points(triggers)
    for index in range(len(segments)):
        segment = segments[index]
        marker = int(markers[index][0])
        segment_samples = segment.shape[1]  # 获取该段的样本数量
       
        fNIRSIndexs, light = get_channel_data_by_marker(marker)
        c_current = all_data[light]
        for _id in range(len(fNIRSIndexs)):
            fNIRSIndex= fNIRSIndexs[_id]
            for id in range(len(fNIRSIndex[0])):
                # -1 是从 0 - 16个通道  33是 前32个通道是eeg，第33个通道是fNIRS
                sensor = fNIRSIndex[0][id] - 1 + chunnels_start
                c_current[fNIRSIndex[1][id]].append(segment[sensor][0])
        all_data[light] = c_current
    return all_data, triggers_array

def get_min_length_origin_data(data_780, data_850):
    try:
        min_length = min(len(sublist) for sublist in data_780)  # 输出: 2
        min_length_850 = min(len(sublist) for sublist in data_850)  # 输出: 2
        min_length = min(min_length, min_length_850)
        min_length_850 = min_length
        # 2. 截取所有子列表至最小长度
        trimmed_data = [sublist[:min_length] for sublist in data_780]
        # 3. 转换为NumPy数组
        np_array_780 = np.array(trimmed_data)
        # 2. 截取所有子列表至最小长度
        trimmed_data_850 = [sublist[:min_length_850] for sublist in data_850]
        # 3. 转换为NumPy数组
        np_array_850 = np.array(trimmed_data_850)
        return np_array_850.tolist(), np_array_780.tolist()
    except Exception as e:
        return [], []

def get_processing_from_origin_data_48_ch(data, data_marker):
    segments, triggers = find_contiguous_segments(data, data_marker)
    channels = list(segments["780"].keys())
    data_780 = []
    data_850 = []
    for channel in channels:
        data_780.append(segments["780"][channel])
        data_850.append(segments["850"][channel])
    data_780, data_850 = get_min_length_origin_data(data_780, data_850)
    return segments["780"].keys(), data_780, data_850, triggers


# def get_preprocessed_fnirs_data(
#         raw_arr,
#         freq=5,
#         target_freq=5,
#         low_freq=0.01,
#         high_freq=0.1
# ):
#     """
#     fNIRS 预处理流程：
#     1. 比尔朗伯
#     2. 带通滤波（low_freq - high_freq）
#     3. 重采样（target_freq）
#     4. 分段
#     """
#     marker = boardInfo['fNIRSMarker']
#     waves = boardInfo['lightWave']
#     fNIRS_channels, data_780, data_850, triggers = get_processing_from_origin_data_48_ch(raw_arr, marker)
#     hbo, hbr = process_origin_to_fNIRS(np.array(data_850).T, np.array(data_780).T, waves)
#
#     ch_names = board_info['fNIRSChannels']
#     ch_num = len(ch_names)
#
#     fnirs_data = np.vstack([hbo, hbr])
#     info = mne.create_info(
#         sfreq=freq,
#         ch_names=[f'{name}_HbO' for name in ch_names] + [f'{name}_HbR' for name in ch_names],
#         ch_types=['hbo'] * ch_num + ['hbr'] * ch_num
#     )
#     raw = mne.io.RawArray(fnirs_data, info)
#
#     raw.filter(
#         l_freq=low_freq,
#         h_freq=high_freq,
#         method='iir',
#         iir_params=dict(order=4, ftype='butter')
#     )
#
#     idx = np.where((triggers > 0) & (triggers <= 5))[0]
#
#     # 构造 events 数组
#     events = np.zeros((len(idx), 3), dtype=int)
#     events[:, 0] = idx  # 事件发生的采样点
#     events[:, 1] = 0  # previous_event_id，这里设为 0
#     events[:, 2] = triggers[idx]  # 当前事件 ID
#
#     _, resampled_events = raw.resample(target_freq, events=events, npad='auto')
#     resampled_events[-1][0] -= 1*target_freq  # 静息态开始时间前移一秒，防止分段时时长不够
#
#     emotion_epoch = get_epoch(raw, resampled_events, mode='emotion')
#     resting_epoch = get_epoch(raw, resampled_events, mode='resting')
#
#     return emotion_epoch, resting_epoch
