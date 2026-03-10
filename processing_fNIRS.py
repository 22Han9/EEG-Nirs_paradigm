import math
import numpy as np
from scipy.interpolate import interp1d
from eeg_positions import get_elec_coords, plot_coords
from procutil_get_extinctions import procutil_get_extinctions
from pyedflib import highlevel
import json 
with open('boardInfo.json', 'r') as f:
    boardInfo = json.load(f)

lightChannelName= boardInfo['lightName']
sensorChannelName= boardInfo['senserName']
led_sensor = {
    "1": {"channel":[1, 2, 3], "name": [lightChannelName[0] +"_"+sensorChannelName[0], lightChannelName[0] +"_"+sensorChannelName[1], lightChannelName[0] +"_"+sensorChannelName[2]]},    # 2
    "2": {"channel":[1], "name": [lightChannelName[1] +"_"+sensorChannelName[0]]},       # 1
    "3": {"channel":[1,3] , "name": [lightChannelName[2] +"_"+sensorChannelName[0], lightChannelName[2] +"_"+sensorChannelName[2]]},    # 2
    "4": {"channel":[2, 3], "name": [lightChannelName[3] +"_"+sensorChannelName[1], lightChannelName[3] +"_"+sensorChannelName[2]]},    # 2
    "5": {"channel":[2], "name": [lightChannelName[4] +"_"+sensorChannelName[1]]},       # 2 
    "6": {"channel":[4, 5, 8], "name": [lightChannelName[5] +"_"+sensorChannelName[3], lightChannelName[5] +"_"+sensorChannelName[4], lightChannelName[5] +"_"+sensorChannelName[7]]}, # 3
    "7": {"channel":[6, 7, 9], "name": [lightChannelName[6] +"_"+sensorChannelName[5], lightChannelName[6] +"_"+sensorChannelName[6], lightChannelName[6] +"_"+sensorChannelName[8]]}, # 3
    "8": {"channel":[4, 8, 10], "name": [lightChannelName[7] +"_"+sensorChannelName[3], lightChannelName[7] +"_"+sensorChannelName[7], lightChannelName[7] +"_"+sensorChannelName[9]]},# 3
    "9": {"channel":[5, 8, 11], "name": [lightChannelName[8] +"_"+sensorChannelName[4], lightChannelName[8] +"_"+sensorChannelName[7], lightChannelName[8] +"_"+sensorChannelName[10]]},# 3
    "10": {"channel":[6, 9, 12], "name": [lightChannelName[9] +"_"+sensorChannelName[5], lightChannelName[9] +"_"+sensorChannelName[8], lightChannelName[9] +"_"+sensorChannelName[11]]},# 3
    "11": {"channel":[7, 9, 13], "name": [lightChannelName[10] +"_"+sensorChannelName[6], lightChannelName[10] +"_"+sensorChannelName[8], lightChannelName[10] +"_"+sensorChannelName[12]]}, # 3
    "12": {"channel":[8, 10, 11], "name": [lightChannelName[11] +"_"+sensorChannelName[7], lightChannelName[11] +"_"+sensorChannelName[9], lightChannelName[11] +"_"+sensorChannelName[10]]}, # 3
    "13": {"channel":[9, 12, 13], "name": [lightChannelName[12] +"_"+sensorChannelName[8], lightChannelName[12] +"_"+sensorChannelName[11], lightChannelName[12] +"_"+sensorChannelName[12]]}, # 3
    "14": {"channel":[14, 15, 16], "name": [lightChannelName[13] +"_"+sensorChannelName[13], lightChannelName[13] +"_"+sensorChannelName[14], lightChannelName[13] +"_"+sensorChannelName[15]]} # 3
}

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

def max_every_8_points(arr):
    result = []  # 存储结果的列表
    n = len(arr)
    i = 0
    while i < n:
        # 取当前区块：从i开始到min(i+8, n)
        block = arr[i:i+8]
        # 求区块最大值并添加到结果
        result.append(max(block))
        i += 8  # 移动到下一个区块起始位置
    return result

def get_channel_data_by_marker(marker):
    fNIRS_indexs =  []
    light= '780'
    light_arr = decimal_to_16bit_array(marker)
    if light_arr[0] == 0:
        light= '850'
    for i in range(len(light_arr)):
        if i > 13 :
            continue
        item = light_arr[15-i]
        if item == 0:
            continue
        fNIRS_indexs.append([led_sensor[str((i+1 ))]["channel"],led_sensor[str((i+1))]["name"]])
    return fNIRS_indexs, light

def find_contiguous_segments(data, index):
    all_data = dict({
        "780":dict({
            lightChannelName[0]+'_'+sensorChannelName[0]: [],
            lightChannelName[0]+'_'+sensorChannelName[1]: [],
            lightChannelName[0]+'_'+sensorChannelName[2]: [],
            lightChannelName[1]+'_'+sensorChannelName[0]: [],
            lightChannelName[2]+'_'+sensorChannelName[0]: [],
            lightChannelName[2]+'_'+sensorChannelName[2]: [],
            lightChannelName[3]+'_'+sensorChannelName[1]: [],
            lightChannelName[3]+'_'+sensorChannelName[2]: [],
            lightChannelName[4]+'_'+sensorChannelName[1]: [],
            lightChannelName[5]+'_'+sensorChannelName[3]: [],
            lightChannelName[5]+'_'+sensorChannelName[4]: [],
            lightChannelName[5]+'_'+sensorChannelName[7]: [],
            lightChannelName[6]+'_'+sensorChannelName[5]: [],
            lightChannelName[6]+'_'+sensorChannelName[6]: [],
            lightChannelName[6]+'_'+sensorChannelName[8]: [],
            lightChannelName[7]+'_'+sensorChannelName[3]: [],
            lightChannelName[7]+'_'+sensorChannelName[7]: [],
            lightChannelName[7]+'_'+sensorChannelName[9]: [],
            lightChannelName[8]+'_'+sensorChannelName[4]: [],
            lightChannelName[8]+'_'+sensorChannelName[7]: [],
            lightChannelName[8]+'_'+sensorChannelName[10]: [],
            lightChannelName[9]+'_'+sensorChannelName[5]: [],
            lightChannelName[9]+'_'+sensorChannelName[8]: [],
            lightChannelName[9]+'_'+sensorChannelName[11]: [],
            lightChannelName[10]+'_'+sensorChannelName[6]: [],
            lightChannelName[10]+'_'+sensorChannelName[8]: [],
            lightChannelName[10]+'_'+sensorChannelName[12]: [],
            lightChannelName[11]+'_'+sensorChannelName[7]: [],
            lightChannelName[11]+'_'+sensorChannelName[9]: [],
            lightChannelName[11]+'_'+sensorChannelName[10]: [],
            lightChannelName[12]+'_'+sensorChannelName[8]: [],
            lightChannelName[12]+'_'+sensorChannelName[11]: [],
            lightChannelName[12]+'_'+sensorChannelName[12]: [],
            lightChannelName[13]+'_'+sensorChannelName[13]: [],
            lightChannelName[13]+'_'+sensorChannelName[14]: [],
            lightChannelName[13]+'_'+sensorChannelName[15]: [],
        }),
        "850":dict({
            lightChannelName[0]+'_'+sensorChannelName[0]: [],
            lightChannelName[0]+'_'+sensorChannelName[1]: [],
            lightChannelName[0]+'_'+sensorChannelName[2]: [],
            lightChannelName[1]+'_'+sensorChannelName[0]: [],
            lightChannelName[2]+'_'+sensorChannelName[0]: [],
            lightChannelName[2]+'_'+sensorChannelName[2]: [],
            lightChannelName[3]+'_'+sensorChannelName[1]: [],
            lightChannelName[3]+'_'+sensorChannelName[2]: [],
            lightChannelName[4]+'_'+sensorChannelName[1]: [],
            lightChannelName[5]+'_'+sensorChannelName[3]: [],
            lightChannelName[5]+'_'+sensorChannelName[4]: [],
            lightChannelName[5]+'_'+sensorChannelName[7]: [],
            lightChannelName[6]+'_'+sensorChannelName[5]: [],
            lightChannelName[6]+'_'+sensorChannelName[6]: [],
            lightChannelName[6]+'_'+sensorChannelName[8]: [],
            lightChannelName[7]+'_'+sensorChannelName[3]: [],
            lightChannelName[7]+'_'+sensorChannelName[7]: [],
            lightChannelName[7]+'_'+sensorChannelName[9]: [],
            lightChannelName[8]+'_'+sensorChannelName[4]: [],
            lightChannelName[8]+'_'+sensorChannelName[7]: [],
            lightChannelName[8]+'_'+sensorChannelName[10]: [],
            lightChannelName[9]+'_'+sensorChannelName[5]: [],
            lightChannelName[9]+'_'+sensorChannelName[8]: [],
            lightChannelName[9]+'_'+sensorChannelName[11]: [],
            lightChannelName[10]+'_'+sensorChannelName[6]: [],
            lightChannelName[10]+'_'+sensorChannelName[8]: [],
            lightChannelName[10]+'_'+sensorChannelName[12]: [],
            lightChannelName[11]+'_'+sensorChannelName[7]: [],
            lightChannelName[11]+'_'+sensorChannelName[9]: [],
            lightChannelName[11]+'_'+sensorChannelName[10]: [],
            lightChannelName[12]+'_'+sensorChannelName[8]: [],
            lightChannelName[12]+'_'+sensorChannelName[11]: [],
            lightChannelName[12]+'_'+sensorChannelName[12]: [],
            lightChannelName[13]+'_'+sensorChannelName[13]: [],
            lightChannelName[13]+'_'+sensorChannelName[14]: [],
            lightChannelName[13]+'_'+sensorChannelName[15]: [],
        })
    })

    chunnels_start = 1
    if data.shape[0] == 64:
        chunnels_start = 33
    
    # 提取第21列的数据，假设索引为20
    col21 = data[index].tolist()
    if not col21:
        return all_data
    segments_indices = []
    start = 0
    for i in range(1, len(col21)):
        if col21[i] != col21[i-1]:
            segments_indices.append([start, i, col21[i]])
            start = i
    # 添加最后一个段
    segments_indices.append([start, len(col21)-1, col21[ len(col21)-1]])
    segments = []
    markers = []
    triggers = []

    for start_idx, end_idx, marker in segments_indices:
        # 对每个分段的起始和结束索引，根据这些索引从 data 中提取对应段的数据
        segment = data[:, start_idx:end_idx]  # 提取通道 * 该段样本的部分
        if segment.shape[1] < 5:
            continue
        segment_samples = segment.shape[1]
        middle_start = segment_samples // 3  # 中间 1/3 的起始位置
        middle_end = 2 * segment_samples // 3  # 中间 1/3 的结束位置
        triggers.append(np.max(segment[-1]))
        middle_segment = segment[:, middle_start:middle_end]
        average = np.average(middle_segment, axis=1, keepdims=True)  # 对样本维度求平均，保留通道维度
        markers.append(average[index])
        segments.append(average)
        
    triggers_array = max_every_8_points(triggers)
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
    except Exception as e :
        return [], []
def get_position_by_channel(channels):
    coords = get_elec_coords(
        system="1005",
        dim="3d",
    )
    channelNames = coords['label'].to_list()
    index2 = [channelNames.index(name) for name in channels]
    coords = coords[coords['label'].isin(channels)]
    return {
        "x": coords['x'][index2].to_list(),
        "y": coords['y'][index2].to_list(),
        "z": coords['z'][index2].to_list()
    }
def get_position_by_light_sensor_position(lightChannelName, sensorChannelName, channels):
    lightChannelPosition = get_position_by_channel(lightChannelName)
    sensorChannelPosition = get_position_by_channel(sensorChannelName)
    positions = []
    for channel in channels:
        lightName, sensorName = channel.split('_')
        lightChannelIndex = lightChannelName.index(lightName)
        sensorChannelIndex = sensorChannelName.index(sensorName)
        positions.append({
            "x": (lightChannelPosition['x'][lightChannelIndex] + sensorChannelPosition['x'][sensorChannelIndex])/2 ,
            "y": (lightChannelPosition['y'][lightChannelIndex] + sensorChannelPosition['y'][sensorChannelIndex]) /2,
            "z": (lightChannelPosition['z'][lightChannelIndex] + sensorChannelPosition['z'][sensorChannelIndex]) /2,
        })
    return positions
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

def get_processiing_from_origin_data_48_ch_mean(data, data_marker):
    segments, triggers = find_contiguous_segments(data, data_marker)
    channels = list(segments["780"].keys())
    data_780 = []
    data_850 = []
    for channel in channels:
        data_780.append(segments["780"][channel])
        data_850.append(segments["850"][channel])
    data_780, data_850 = get_min_length_origin_data(data_780, data_850) 
    return segments["780"].keys(), np.mean(data_780, axis=1).tolist(), np.mean(data_850, axis=1).tolist()

def get_position_by_light_sensor_position(lightChannelName, sensorChannelName, channels):
    lightChannelPosition = get_position_by_channel(lightChannelName)
    sensorChannelPosition = get_position_by_channel(sensorChannelName)
    positions = []
    for channel in channels:
        lightName, sensorName = channel.split('_')
        lightChannelIndex = lightChannelName.index(lightName)
        sensorChannelIndex = sensorChannelName.index(sensorName)
        positions.append({
            "x": (lightChannelPosition['x'][lightChannelIndex] + sensorChannelPosition['x'][sensorChannelIndex])/2 ,
            "y": (lightChannelPosition['y'][lightChannelIndex] + sensorChannelPosition['y'][sensorChannelIndex]) /2,
            "z": (lightChannelPosition['z'][lightChannelIndex] + sensorChannelPosition['z'][sensorChannelIndex]) /2,
            's_x': sensorChannelPosition['x'][sensorChannelIndex],
            's_y': sensorChannelPosition['y'][sensorChannelIndex],
            's_z': sensorChannelPosition['z'][sensorChannelIndex],
            'l_x': lightChannelPosition['x'][lightChannelIndex],
            'l_y': lightChannelPosition['y'][lightChannelIndex],
            'l_z': lightChannelPosition['z'][lightChannelIndex] 
        })
    return positions


# # EEG 数据存储
# class EEGSAVEDATA(object):
#     def __init__(self):
#         super(EEGSAVEDATA, self).__init__()
#         print('inint')
#         self.name = 'name'
    
#     def saveFile(self,fileName, data, channels, sampleRate, otherInfo):
#         # try:
#             """
#             A convenience function to create an EDF header (a dictionary) that
#             can be used by pyedflib to update the main header of the EDF

#             Parameters
#             ----------
#             technician : str, optional
#                 name of the technician. The default is ''.
#             recording_additional : str, optional
#                 comments etc. The default is ''.
#             patientname : str, optional
#                 the name of the patient. The default is ''.
#             patient_additional : TYPE, optional
#                 more info about the patient. The default is ''.
#             patientcode : str, optional
#                 alphanumeric code. The default is ''.
#             equipment : str, optional
#                 which system was used. The default is ''.
#             admincode : str, optional
#                 code of the admin. The default is ''.
#             gender : str, optional
#                 gender of patient. The default is ''.
#             startdate : datetime.datetime, optional
#                 startdate of recording. The default is None.
#             birthdate : str/datetime.datetime, optional
#                 date of birth of the patient. The default is ''.
#             """
#             data = data.T
#             data = np.ascontiguousarray(np.array(data))
#             signals = []
#             for channel in range(len(data)):
#                 DataFilter.detrend(data[channel], DetrendOperations.NO_DETREND.value)
#                 signals.append(data[channel]/1)
#             signals = np.ascontiguousarray(np.array(signals))
#             signalHeaders = highlevel.make_signal_headers(
#                 list_of_labels=channels,
#                 sample_frequency=sampleRate, 
#                 sample_rate=sampleRate,
#                 physical_max=187500,
#                 physical_min=-187500,
#                 digital_max= 187500,
#                 digital_min= -187500
#             )
#             technician = ''
#             recording_additional = ''
#             patientname = ''
#             patient_additional = ''
#             patientcode = ''
#             equipment = ''
#             admincode = ''
#             gender = ''
#             # birthdate= datetime.datetime(1900, 1, 1).strftime('%d %b %Y')
#             keys = list(otherInfo.keys())
#             if 'technician' in keys:
#                 technician = otherInfo["technician"]
#             if 'recording_additional' in keys:
#                 recording_additional = otherInfo['recording_additional']
#             if 'patientname' in keys:
#                 patientname = otherInfo['patientname']
#             if 'patient_additional' in keys:
#                 patient_additional = otherInfo['patient_additional']
#             if 'patientcode' in keys:
#                 patientcode = otherInfo['patientcode']
#             if 'equipment' in keys:
#                 equipment = otherInfo['equipment']
#             if 'admincode' in keys:
#                 admincode=otherInfo['admincode']
#             # if 'birthdate' in keys:
#             #     birthdate = otherInfo['birthdate']
#             header = highlevel.make_header(technician=technician, 
#                                         recording_additional=recording_additional,
#                                         patientname=patientname,
#                                         patient_additional=patient_additional, 
#                                         patientcode=patientcode, 
#                                         equipment=equipment, 
#                                         admincode=admincode,
#                                         gender=gender)
#             print(signals.shape, len(signalHeaders))
#             res = highlevel.write_edf(fileName, signals=signals, signal_headers=signalHeaders, digital=False,file_type=3, header=header)
#         # except Exception as e :
#         #     print(e)

## 先把数据分割，然后用data读数据

##EEG 原始数据
# eeg = data[1:33, -4000: ]
#
# fNIRS_channels, data_780, data_850, triggers = get_processing_from_origin_data_48_ch(data, 56)
#
# data_780 = np.array(data_780)
# data_850 = np.array(data_850)
# minLength = min(data_780.shape[1], data_850.shape[1])
# ##含氧和脱氧的原始数据，后续预处理还要基线矫正（以前面休息状态3s作为基线）、滤波等
# oxy, deoxy = process_origin_to_fNIRS(data_850[:, :minLength].T, data_780[:, :minLength].T, [850, 780])
