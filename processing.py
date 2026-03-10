import numpy as np
import os
import mne
import scipy.io as sio
from sklearn.decomposition import PCA
from scipy import signal
import matplotlib.pyplot as plt

# 设置参数
EEG_SFREQ_ORIGINAL = 1000  # 原始EEG采样率
EEG_SFREQ_TARGET = 250  # 目标EEG采样率
FNIRS_SFREQ = 5  # fNIRS采样率
BASELINE_DURATION = 3  # 基线时长(秒)
TRIAL_DURATION = 33  # 总trial时长(秒)
N_TRIALS = 20  # 每个subject的trial数量
N_EEG_CHANNELS = 32  # EEG通道数
N_FNIRS_CHANNELS = 36  # fNIRS通道数
WINDOW_LENGTH = 10         # 窗口长度(秒)
WINDOW_STEP = 4            # 窗口步长(秒)
TRIAL_LABELS = [3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2]
# eeg_chn_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8',
#                  'O1', 'Oz', 'O2']
eeg_chn_names = ["AFp1", "AFp2", "AFF1h", "AFF2h", "F1", "F3", "F5", "F2", "F4", "F6", "FCC3h", "FCC5h", "CCP5h", "CCP3h", "FCz", "FCC4h",
                "FCC6h", "CCP4h", "CCP6h", "T7", "T8", "P3", "P1", "Pz", "P2", "P4", "CPz", "Cz", "PPO1h", "PPO2h", "POO1", "POO2"]

# fNIRS通道名称 - 需要根据您的实际设置修改
fnirs_chn_names = [ "FPZ_FP1", "FPZ_FP2", "FPZ_AFZ", "AF7_FP1", "AF3_FP1", "AF3_AFZ", "AF4_FP2", "AF4_AFZ", "AF8_FP2", "FC3_FC5", "FC3_FC1",
                    "FC3_C3", "FC4_FC2", "FC4_FC6", "FC4_C4", "C5_FC5", "C5_C3", "C5_CP5", "C1_FC1", "C1_C3", "C1_CP1", "C2_FC2", "C2_C4", "C2_CP1",
                    "C6_FC6", "C6_C4", "C6_CP6", "CP3_C3", "CP3_CP5", "CP3_CP1", "CP4_C4", "CP4_CP2", "CP4_CP6", "OZ_POZ", "OZ_O1", "OZ_O2"]
# 如果您的通道名称不同，请替换为实际的36个通道名称

# 创建EEG信息结构
eeg_info = mne.create_info(ch_names=eeg_chn_names, sfreq=EEG_SFREQ_ORIGINAL, ch_types='eeg')
eeg_info.set_montage('standard_1005')

# 创建fNIRS信息结构
fnirs_info = mne.create_info(ch_names=fnirs_chn_names, sfreq=FNIRS_SFREQ, ch_types='fnirs_cw_amplitude')

# 获取subject列表
subject_list = []
folder_path = 'doc'  # 修改为您的数据路径
for filename in os.listdir(folder_path):
    if filename.startswith('sub_') and filename.endswith('.mat'):
        subject_no = filename.split('_')[1].split('.')[0]
        subject_list.append(subject_no)

# PCA运动伪影校正
def pca_artifact_removal(data, n_components=2):
    """使用PCA去除运动伪影"""
    # 转置数据以便PCA处理（时间点×通道）
    data_t = data.T

    # 检查数据是否包含NaN或无穷大值
    if np.any(np.isnan(data_t)) or np.any(np.isinf(data_t)):
        print("警告: 输入数据包含NaN或无穷大值，返回原始数据")
        return data

    # 标准化数据
    mean = np.mean(data_t, axis=0)
    std = np.std(data_t, axis=0)

    # 检查标准差是否为零
    zero_std_mask = std == 0
    if np.any(zero_std_mask):
        print(f"警告: {np.sum(zero_std_mask)} 个通道的标准差为零，跳过这些通道的PCA")
        # 对于标准差为零的通道，返回原始数据
        data_corrected = data.copy()
        return data_corrected

    std[std == 0] = 1  # 避免除以零
    data_normalized = (data_t - mean) / std

    # 检查标准化后的数据是否包含NaN或无穷大值
    if np.any(np.isnan(data_normalized)) or np.any(np.isinf(data_normalized)):
        print("警告: 标准化后的数据包含NaN或无穷大值，返回原始数据")
        return data

    # 应用PCA
    try:
        pca = PCA(n_components=min(n_components, data_normalized.shape[0], data_normalized.shape[1]))
        components = pca.fit_transform(data_normalized)

        # 创建一个零矩阵，用于重建信号（排除第一个主成分）
        components_clean = np.zeros_like(components)
        components_clean[:, 1:] = components[:, 1:]  # 保留除第一个外的所有成分

        # 使用所有成分重建信号，但第一个成分被置零
        reconstructed = pca.inverse_transform(components_clean)

        # 反标准化
        data_corrected = reconstructed * std + mean

        # 检查最终结果是否包含NaN或无穷大值
        if np.any(np.isnan(data_corrected)) or np.any(np.isinf(data_corrected)):
            print("警告: PCA校正后的数据包含NaN或无穷大值，返回原始数据")
            return data_t.T  # 返回原始形状的数据

        return data_corrected.T  # 转回原始形状（通道×时间点）

    except Exception as e:
        print(f"PCA处理失败: {e}")
        return data  # 返回原始数据



def sliding_window(data, window_length, window_step, sfreq):
    """对数据进行滑动窗口分割"""
    window_samples = int(window_length * sfreq)
    step_samples = int(window_step * sfreq)

    n_samples = data.shape[1]
    windows = []

    # 计算可以提取的窗口数量
    n_windows = (n_samples - window_samples) // step_samples + 1

    # 提取窗口
    for i in range(n_windows):
        start = i * step_samples
        end = start + window_samples
        window = data[:, start:end]
        windows.append(window)

    return np.array(windows)


def check_raw_data_quality(mat_data, trial_idx):
    """
    检查原始数据的质量
    """
    print(f"\n检查 trial {trial_idx} 的原始数据质量:")

    # 提取当前trial的数据
    eeg_key = f'EEG_raw_{trial_idx}'
    hbo_key = f'HbO_{trial_idx}'
    hbr_key = f'HbR_{trial_idx}'

    eeg_data = mat_data[eeg_key] / 1e6 # 单位从 μV 转为 V
    hbo_data = mat_data[hbo_key]
    hbr_data = mat_data[hbr_key]

    # 检查数据范围
    print(f"原始EEG数据范围: [{np.min(eeg_data):.4f}, {np.max(eeg_data):.4f}]")
    print(f"原始HbO数据范围: [{np.min(hbo_data):.6f}, {np.max(hbo_data):.6f}]")
    print(f"原始HbR数据范围: [{np.min(hbr_data):.6f}, {np.max(hbr_data):.6f}]")

    # 检查NaN值
    if np.any(np.isnan(eeg_data)):
        print("⚠ 原始EEG数据中包含NaN值")
        print(f"  NaN值数量: {np.sum(np.isnan(eeg_data))}")

    if np.any(np.isnan(hbo_data)):
        print("⚠ 原始HbO数据中包含NaN值")
        print(f"  NaN值数量: {np.sum(np.isnan(hbo_data))}")

    if np.any(np.isnan(hbr_data)):
        print("⚠ 原始HbR数据中包含NaN值")
        print(f"  NaN值数量: {np.sum(np.isnan(hbr_data))}")

    # 检查无穷大值
    if np.any(np.isinf(eeg_data)):
        print("⚠ 原始EEG数据中包含无穷大值")

    if np.any(np.isinf(hbo_data)):
        print("⚠ 原始HbO数据中包含无穷大值")

    if np.any(np.isinf(hbr_data)):
        print("⚠ 原始HbR数据中包含无穷大值")

    # 检查常数通道
    eeg_std = np.std(eeg_data, axis=1)
    hbo_std = np.std(hbo_data, axis=1)
    hbr_std = np.std(hbr_data, axis=1)

    zero_std_eeg = np.sum(eeg_std == 0)
    zero_std_hbo = np.sum(hbo_std == 0)
    zero_std_hbr = np.sum(hbr_std == 0)

    if zero_std_eeg > 0:
        print(f"⚠ EEG数据中有 {zero_std_eeg} 个常数通道")

    if zero_std_hbo > 0:
        print(f"⚠ HbO数据中有 {zero_std_hbo} 个常数通道")

    if zero_std_hbr > 0:
        print(f"⚠ HbR数据中有 {zero_std_hbr} 个常数通道")

    return zero_std_hbr > 0  # 返回HbR是否有常数通道

# 对每个subject进行处理
for subject_no in subject_list:
    print(f"Processing subject {subject_no}")

    # 加载.mat文件
    mat_data = sio.loadmat(os.path.join(folder_path, f'sub_{subject_no}.mat'))

    # 初始化存储所有trial的列表
    all_eeg_windows = []
    all_fnirs_windows = []
    all_labels = []

    # 对每个trial进行处理
    for trial_idx in range(1, N_TRIALS + 1):
        print(f"  Processing trial {trial_idx}")

        # 提取当前trial的数据
        eeg_key = f'EEG_raw_{trial_idx}'
        hbo_key = f'HbO_{trial_idx}'
        hbr_key = f'HbR_{trial_idx}'

        eeg_data = mat_data[eeg_key] / 1e6  # 形状应为 (32, 33000)
        hbo_data = mat_data[hbo_key].T  # 形状应为 (36, 137)
        hbr_data = mat_data[hbr_key].T  # 形状应为 (36, 137)

        # 检查原始数据质量
        has_constant_hbr = check_raw_data_quality(mat_data, trial_idx)

        # ========== EEG预处理 ==========
        # 创建Raw对象
        raw_eeg = mne.io.RawArray(eeg_data, eeg_info)

        # 降采样到250Hz
        raw_eeg.resample(EEG_SFREQ_TARGET, npad='auto')

        # 陷波滤波（去除50Hz工频干扰）
        raw_eeg_notch = raw_eeg.notch_filter(np.arange(50, 101, 50))

        # 带通滤波（0.5-50Hz）
        raw_eeg_filtered = raw_eeg_notch.filter(0.5, 50., method='iir', iir_params=dict(order=6, ftype='butter'))

        # 平均参考
        raw_eeg_avg_ref = raw_eeg_filtered.set_eeg_reference(ref_channels="average")

        # 加载数据到内存
        raw_eeg_avg_ref.load_data()

        # filtering just for ICA
        raw_for_ica = raw_eeg_avg_ref.copy().filter(l_freq=1., h_freq=None)

        # ICA去伪迹
        ica = mne.preprocessing.ICA(n_components=15, random_state=97, method='infomax')
        ica.fit(raw_for_ica)
        # ica.plot_components(inst=raw_eeg_avg_ref)
        raw_eeg_ica = ica.apply(raw_eeg_avg_ref)

        # 基线校正（使用前3秒作为基线）
        # 计算基线期间的均值
        baseline_samples = int(BASELINE_DURATION * EEG_SFREQ_TARGET)
        eeg_data_processed = raw_eeg_ica.get_data()
        baseline_mean = np.mean(eeg_data_processed[:, :baseline_samples], axis=1, keepdims=True)
        eeg_data_processed = eeg_data_processed - baseline_mean
        # 去除基线期数据（只保留任务期）
        eeg_data_processed = eeg_data_processed[:, baseline_samples:]


        # ========== fNIRS预处理 ==========
        # 创建Raw对象
        raw_hbo = mne.io.RawArray(hbo_data, fnirs_info)
        raw_hbr = mne.io.RawArray(hbr_data, fnirs_info)

        # 带通滤波（0.01-0.1Hz）
        raw_hbo_filtered = raw_hbo.filter(0.01, 0.1, method='iir', iir_params=dict(order=6, ftype='butter'))
        raw_hbr_filtered = raw_hbr.filter(0.01, 0.1, method='iir', iir_params=dict(order=6, ftype='butter'))

        # 获取滤波后的数据
        hbo_data_filtered = raw_hbo_filtered.get_data()
        hbr_data_filtered = raw_hbr_filtered.get_data()

        # 检查滤波后的数据
        print(f"滤波后HbO数据范围: [{np.min(hbo_data_filtered):.6f}, {np.max(hbo_data_filtered):.6f}]")
        print(f"滤波后HbR数据范围: [{np.min(hbr_data_filtered):.6f}, {np.max(hbr_data_filtered):.6f}]")

        # 对HbO和HbR应用PCA校正
        hbo_data_corrected = pca_artifact_removal(hbo_data_filtered)
        hbr_data_corrected = pca_artifact_removal(hbr_data_filtered)

        # 检查PCA校正后的数据
        print(f"PCA校正后HbO数据范围: [{np.min(hbo_data_corrected):.6f}, {np.max(hbo_data_corrected):.6f}]")
        print(f"PCA校正后HbR数据范围: [{np.min(hbr_data_corrected):.6f}, {np.max(hbr_data_corrected):.6f}]")

        # 基线校正（使用前3秒作为基线）
        baseline_samples_fnirs = int(BASELINE_DURATION * FNIRS_SFREQ)
        baseline_mean_hbo = np.mean(hbo_data_corrected[:, :baseline_samples_fnirs], axis=1, keepdims=True)
        baseline_mean_hbr = np.mean(hbr_data_corrected[:, :baseline_samples_fnirs], axis=1, keepdims=True)
        hbo_data_corrected = hbo_data_corrected - baseline_mean_hbo
        hbr_data_corrected = hbr_data_corrected - baseline_mean_hbr

        # 去除基线期数据（只保留任务期）
        hbo_data_corrected = hbo_data_corrected[:, baseline_samples_fnirs:]
        hbr_data_corrected = hbr_data_corrected[:, baseline_samples_fnirs:]

        # 检查最终数据
        print(f"最终EEG数据范围: [{np.min(eeg_data_processed):.6f}, {np.max(eeg_data_processed):.6f}]")
        print(f"最终HbO数据范围: [{np.min(hbo_data_corrected):.6f}, {np.max(hbo_data_corrected):.6f}]")
        print(f"最终HbR数据范围: [{np.min(hbr_data_corrected):.6f}, {np.max(hbr_data_corrected):.6f}]")

        # ========== 滑动窗口分割 ==========
        # 对EEG数据进行滑动窗口分割
        eeg_windows = sliding_window(eeg_data_processed, WINDOW_LENGTH, WINDOW_STEP, EEG_SFREQ_TARGET)

        # 对fNIRS数据进行滑动窗口分割
        hbo_windows = sliding_window(hbo_data_corrected, WINDOW_LENGTH, WINDOW_STEP, FNIRS_SFREQ)
        hbr_windows = sliding_window(hbr_data_corrected, WINDOW_LENGTH, WINDOW_STEP, FNIRS_SFREQ)

        # 合并HbO和HbR数据
        fnirs_windows = np.concatenate([hbo_windows, hbr_windows], axis=1)  # 沿通道维度合并

        # 为每个窗口分配标签
        trial_label = TRIAL_LABELS[trial_idx - 1]  # 获取当前trial的标签
        window_labels = np.full(eeg_windows.shape[0], trial_label)  # 为每个窗口分配相同的标签

        # 添加到总列表
        all_eeg_windows.append(eeg_windows)
        all_fnirs_windows.append(fnirs_windows)
        all_labels.append(window_labels)

    # 将所有trial的数据转换为numpy数组
    eeg_all_windows = np.concatenate(all_eeg_windows, axis=0)
    fnirs_all_windows = np.concatenate(all_fnirs_windows, axis=0)
    labels_all = np.concatenate(all_labels, axis=0)

    # 检查合并后的HbR数据是否有NaN
    # 因为fnirs_all_windows中前36个通道是HbO，后36个是HbR
    hbr_data = fnirs_all_windows[:, 36:, :]
    if np.any(np.isnan(hbr_data)):
        print(f"警告: 合并后的HbR数据包含NaN值，数量: {np.sum(np.isnan(hbr_data))}")
    else:
        print("合并后的HbR数据没有NaN值")

    # 保存处理后的数据
    save_dict = {
        'eeg': eeg_all_windows,          # 形状: (samples, 32, time_points)(120, 32, 2500)
        'fnirs': fnirs_all_windows,      # 形状: (samples, 72, time_points) [36 HbO + 36 HbR](120, 36, 84)
        'labels': labels_all             # 形状: (samples,) (120,)
    }

    save_dir = 'doc\doc'
    os.makedirs(save_dir, exist_ok=True)
    save_name = f'sub_{subject_no}.npy'
    np.save(os.path.join(save_dir, save_name), save_dict)
    print(f"Saved processed data for subject {subject_no}")

    #
    # 立即加载并检查
    loaded_data = np.load(os.path.join(save_dir, save_name), allow_pickle=True).item()

    hbr_check = loaded_data['fnirs'][:, 36:, :]
    if np.any(np.isnan(hbr_check)):
        print(f"警告: 保存后的文件中的HbR数据包含NaN值，数量: {np.sum(np.isnan(hbr_check))}")
    else:
        print("保存后的文件中的HbR数据没有NaN值")

print("All subjects processed successfully!")