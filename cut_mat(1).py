import numpy as np
from processing_fNIRS_new import (
    get_processing_from_origin_data_48_ch,
    process_origin_to_fNIRS
)
# from processing_fNIRS import (
#     get_processing_from_origin_data_48_ch,
#     process_origin_to_fNIRS
# )
from scipy.io import savemat
from matplotlib import pyplot as plt
csv_path  = r"E:\graduate\EEG+fNIRS\app2 _1000Hz\data\2026_01_07_14_52_57_fnris 秦裕东1.csv"
# csv_path  = r"E:\graduate\EEG+fNIRS\app\data\2026_01_06_16_05_43_fnris 黎振锋1.csv"

out_mat   = 'datasets\sub_011.mat'
# out_mat   = 'doc\sub_001.mat'
# out_mat   = 'visualization\sub_李鑫1.mat'
fs        = 1000
trial_len = 33 * fs          # 33 s数据
pre_base  = 3 * fs           # 基线3 s

raw = np.loadtxt(csv_path).T
raw = raw[:, raw[55, :].argsort()]
# print(raw.shape)
count = 0
for i in range(len(raw[55]) -1):
    position = raw[55, i]
    if raw[55, i +1] - raw[55, i] > 1:
        count += raw[55, i +1] - raw[55, i]


print(count, count / raw.shape[1] * 100)
marker = raw[-1].astype(int)
# 截取实验100~200之间的数据，把实验前的标签去掉
idx100 = np.where(marker == 100)[0][0]
idx200 = np.where(marker == 200)[0][0]
raw = raw[:, idx100:idx200]
marker = marker[idx100:idx200]

# 找20个情绪trial的标签
trial_on  = np.where(np.isin(marker, [1, 2, 3]))[0]
trial_off = np.where(np.isin(marker, [101, 102, 103]))[0]
mat_dict = {}
for idx, (on, off) in enumerate(zip(trial_on, trial_off), 1):
    start = on - pre_base #标签开始
    end   = start + trial_len #30s实验结束
    seg   = raw[:, start:end] #分段，一个trial
    # 进行解码
    fNIRS_channels, data_780, data_850, triggers = get_processing_from_origin_data_48_ch(seg, 56)
    data_780 = np.array(data_780)
    data_850 = np.array(data_850)
    minLength = min(data_780.shape[1], data_850.shape[1])
    ##含氧和脱氧的原始数据，后续预处理还要基线矫正（以前面休息状态3s作为基线）、滤波等
    oxy, deoxy = process_origin_to_fNIRS(data_850[:, :minLength].T, data_780[:, :minLength].T, [850, 780])
    #分字段保存
    mat_dict[f'EEG_raw_{idx}']      = raw[1:33, start:end]
    mat_dict[f'fNIRS780_raw_{idx}'] = data_780
    mat_dict[f'fNIRS850_raw_{idx}'] = data_850
    mat_dict[f'HbO_{idx}']          = oxy.T
    mat_dict[f'HbR_{idx}']          = deoxy.T
    print(oxy.shape, deoxy.shape)
    plt.plot(oxy[0])
    plt.plot(deoxy[0])
    plt.show()
savemat(out_mat, mat_dict)
print('已生成', out_mat, '（20 个 trial的数据）')