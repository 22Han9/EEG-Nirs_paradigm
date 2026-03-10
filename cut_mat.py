# old app data extract and split
import numpy as np
from processing_fNIRS import (
    get_processing_from_origin_data_48_ch,
    process_origin_to_fNIRS
)
from scipy.io import savemat

csv_path  = r"D:\graduate\app2 _1000Hz\data\2025_07_16_16_45_45_fnris 承霖2.csv"
out_mat   = 'sub_001.mat'
fs        = 1000
trial_len = 33 * fs          # 33 s数据
pre_base  = 3 * fs           # 基线3 s

raw = np.loadtxt(csv_path).T
marker = raw[-1].astype(int)
# 截取实验100~200之间的数据，把实验前的标签去掉
idx100 = np.where(marker == 100)[0][0]
idx200 = np.where(marker == 200)[0][0]
raw = raw[:, idx100:idx200]
marker = marker[idx100:idx200]

# 找20个情绪trial的标签
trial_on  = np.where(np.isin(marker, [1, 2, 3]))[0]
trial_off = np.where(np.isin(marker, [101, 102, 103]))[0]
# assert len(trial_on) == len(trial_off) == 20

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

savemat(out_mat, mat_dict)
print('已生成', out_mat, '（20 个 trial的数据）')