import numpy as np
import os
import mne
import scipy.io as sio
from sklearn.decomposition import PCA
from scipy import signal
import matplotlib.pyplot as plt
#sub1 - 8：文1、鑫1、宏1、尹2、国2、金2、宇2、源2
# 设置参数（未改动）
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
eeg_chn_names = ["AFp1", "AFp2", "AFF1h", "AFF2h", "F1", "F3", "F5", "F2", "F4", "F6", "FCC3h", "FCC5h", "CCP5h", "CCP3h", "FCz", "FCC4h",
                "FCC6h", "CCP4h", "CCP6h", "T7", "T8", "P3", "P1", "Pz", "P2", "P4", "CPz", "Cz", "PPO1h", "PPO2h", "POO1", "POO2"]
fnirs_chn_names = [ "FPZ_FP1", "FPZ_FP2", "FPZ_AFZ", "AF7_FP1", "AF3_FP1", "AF3_AFZ", "AF4_FP2", "AF4_AFZ", "AF8_FP2", "FC3_FC5", "FC3_FC1",
                    "FC3_C3", "FC4_FC2", "FC4_FC6", "FC4_C4", "C5_FC5", "C5_C3", "C5_CP5", "C1_FC1", "C1_C3", "C1_CP1", "C2_FC2", "C2_C4", "C2_CP1",
                    "C6_FC6", "C6_C4", "C6_CP6", "CP3_C3", "CP3_CP5", "CP3_CP1", "CP4_C4", "CP4_CP2", "CP4_CP6", "OZ_POZ", "OZ_O1", "OZ_O2"]

# 创建信息结构（未改动）
eeg_info = mne.create_info(ch_names=eeg_chn_names, sfreq=EEG_SFREQ_ORIGINAL, ch_types='eeg')
eeg_info.set_montage('standard_1005')
fnirs_info = mne.create_info(ch_names=fnirs_chn_names, sfreq=FNIRS_SFREQ, ch_types='fnirs_cw_amplitude')

# 获取subject列表（未改动）
subject_list = []
folder_path = 'temp'  # 修改为您的数据路径
for filename in os.listdir(folder_path):
    if filename.startswith('sub_') and filename.endswith('.mat'):
        subject_no = filename.split('_')[1].split('.')[0]
        subject_list.append(subject_no)

# PCA运动伪影校正（未改动）
def pca_artifact_removal(data, n_components=2):
    data_t = data.T
    if np.any(np.isnan(data_t)) or np.any(np.isinf(data_t)):
        print("警告: 输入数据包含NaN或无穷大值，返回原始数据")
        return data
    mean = np.mean(data_t, axis=0)
    std = np.std(data_t, axis=0)
    zero_std_mask = std == 0
    if np.any(zero_std_mask):
        print(f"警告: {np.sum(zero_std_mask)} 个通道的标准差为零，跳过这些通道的PCA")
        return data
    std[std == 0] = 1
    data_normalized = (data_t - mean) / std
    if np.any(np.isnan(data_normalized)) or np.any(np.isinf(data_normalized)):
        print("警告: 标准化后的数据包含NaN或无穷大值，返回原始数据")
        return data
    try:
        pca = PCA(n_components=min(n_components, data_normalized.shape[0], data_normalized.shape[1]))
        components = pca.fit_transform(data_normalized)
        components_clean = np.zeros_like(components)
        components_clean[:, 1:] = components[:, 1:]
        reconstructed = pca.inverse_transform(components_clean)
        data_corrected = reconstructed * std + mean
        if np.any(np.isnan(data_corrected)) or np.any(np.isinf(data_corrected)):
            print("警告: PCA校正后的数据包含NaN或无穷大值，返回原始数据")
            return data_t.T
        return data_corrected.T
    except Exception as e:
        print(f"PCA处理失败: {e}")
        return data

# 滑动窗口分割（未改动）
def sliding_window(data, window_length, window_step, sfreq):
    window_samples = int(window_length * sfreq)
    step_samples = int(window_step * sfreq)
    n_samples = data.shape[1]
    windows = []
    n_windows = (n_samples - window_samples) // step_samples + 1
    for i in range(n_windows):
        start = i * step_samples
        end = start + window_samples
        window = data[:, start:end]
        windows.append(window)
    return np.array(windows)

# 数据质量检查（未改动）
def check_raw_data_quality(mat_data, trial_idx):
    print(f"\n检查 trial {trial_idx} 的原始数据质量:")
    eeg_key = f'EEG_raw_{trial_idx}'
    hbo_key = f'HbO_{trial_idx}'
    hbr_key = f'HbR_{trial_idx}'
    eeg_data = mat_data[eeg_key] / 1e6
    hbo_data = mat_data[hbo_key]
    hbr_data = mat_data[hbr_key]
    print(f"原始EEG数据范围: [{np.min(eeg_data):.4f}, {np.max(eeg_data):.4f}]")
    print(f"原始HbO数据范围: [{np.min(hbo_data):.6f}, {np.max(hbo_data):.6f}]")
    print(f"原始HbR数据范围: [{np.min(hbr_data):.6f}, {np.max(hbr_data):.6f}]")
    if np.any(np.isnan(eeg_data)):
        print("⚠ 原始EEG数据中包含NaN值")
        print(f"  NaN值数量: {np.sum(np.isnan(eeg_data))}")
    if np.any(np.isnan(hbo_data)):
        print("⚠ 原始HbO数据中包含NaN值")
        print(f"  NaN值数量: {np.sum(np.isnan(hbo_data))}")
    if np.any(np.isnan(hbr_data)):
        print("⚠ 原始HbR数据中包含NaN值")
        print(f"  NaN值数量: {np.sum(np.isnan(hbr_data))}")
    if np.any(np.isinf(eeg_data)):
        print("⚠ 原始EEG数据中包含无穷大值")
    if np.any(np.isinf(hbo_data)):
        print("⚠ 原始HbO数据中包含无穷大值")
    if np.any(np.isinf(hbr_data)):
        print("⚠ 原始HbR数据中包含无穷大值")
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
    return zero_std_hbr > 0

# 对每个subject进行处理（核心修改在trial分窗后）
for subject_no in subject_list:
    print(f"Processing subject {subject_no}")
    mat_data = sio.loadmat(os.path.join(folder_path, f'sub_{subject_no}.mat'))
    all_eeg_windows = []
    all_fnirs_windows = []
    all_labels = []

    for trial_idx in range(1, N_TRIALS + 1):
        print(f"  Processing trial {trial_idx}")
        # 提取原始数据（未改动）
        eeg_key = f'EEG_raw_{trial_idx}'
        hbo_key = f'HbO_{trial_idx}'
        hbr_key = f'HbR_{trial_idx}'
        eeg_data = mat_data[eeg_key] / 1e6
        hbo_data = mat_data[hbo_key].T
        hbr_data = mat_data[hbr_key].T
        # 数据质量检查（未改动）
        has_constant_hbr = check_raw_data_quality(mat_data, trial_idx)

        # ========== EEG预处理（未改动） ==========
        raw_eeg = mne.io.RawArray(eeg_data, eeg_info)
        raw_eeg.resample(EEG_SFREQ_TARGET, npad='auto')
        raw_eeg_notch = raw_eeg.notch_filter(np.arange(50, 101, 50))
        raw_eeg_filtered = raw_eeg_notch.filter(0.5, 50., method='iir', iir_params=dict(order=6, ftype='butter'))
        raw_eeg_avg_ref = raw_eeg_filtered.set_eeg_reference(ref_channels="average")
        raw_eeg_avg_ref.load_data()
        raw_for_ica = raw_eeg_avg_ref.copy().filter(l_freq=1., h_freq=None)
        ica = mne.preprocessing.ICA(n_components=15, random_state=97, method='infomax')
        ica.fit(raw_for_ica)
        raw_eeg_ica = ica.apply(raw_eeg_avg_ref)
        # 基线校正+去除基线期
        baseline_samples = int(BASELINE_DURATION * EEG_SFREQ_TARGET)
        eeg_data_processed = raw_eeg_ica.get_data()
        baseline_mean = np.mean(eeg_data_processed[:, :baseline_samples], axis=1, keepdims=True)
        eeg_data_processed = eeg_data_processed - baseline_mean
        eeg_data_processed = eeg_data_processed[:, baseline_samples:]

        # ========== fNIRS预处理（未改动） ==========
        raw_hbo = mne.io.RawArray(hbo_data, fnirs_info)
        raw_hbr = mne.io.RawArray(hbr_data, fnirs_info)
        raw_hbo_filtered = raw_hbo.filter(0.01, 0.1, method='iir', iir_params=dict(order=6, ftype='butter'))
        raw_hbr_filtered = raw_hbr.filter(0.01, 0.1, method='iir', iir_params=dict(order=6, ftype='butter'))
        hbo_data_filtered = raw_hbo_filtered.get_data()
        hbr_data_filtered = raw_hbr_filtered.get_data()
        print(f"滤波后HbO数据范围: [{np.min(hbo_data_filtered):.6f}, {np.max(hbo_data_filtered):.6f}]")
        print(f"滤波后HbR数据范围: [{np.min(hbr_data_filtered):.6f}, {np.max(hbr_data_filtered):.6f}]")
        # PCA校正
        hbo_data_corrected = pca_artifact_removal(hbo_data_filtered)
        hbr_data_corrected = pca_artifact_removal(hbr_data_filtered)
        print(f"PCA校正后HbO数据范围: [{np.min(hbo_data_corrected):.6f}, {np.max(hbo_data_corrected):.6f}]")
        print(f"PCA校正后HbR数据范围: [{np.min(hbr_data_corrected):.6f}, {np.max(hbr_data_corrected):.6f}]")
        # 基线校正+去除基线期
        baseline_samples_fnirs = int(BASELINE_DURATION * FNIRS_SFREQ)
        baseline_mean_hbo = np.mean(hbo_data_corrected[:, :baseline_samples_fnirs], axis=1, keepdims=True)
        baseline_mean_hbr = np.mean(hbr_data_corrected[:, :baseline_samples_fnirs], axis=1, keepdims=True)
        hbo_data_corrected = hbo_data_corrected - baseline_mean_hbo
        hbr_data_corrected = hbr_data_corrected - baseline_mean_hbr
        hbo_data_corrected = hbo_data_corrected[:, baseline_samples_fnirs:]
        hbr_data_corrected = hbr_data_corrected[:, baseline_samples_fnirs:]

        # ========== 滑动窗口分割（核心修改处） ==========
        # 1. 对EEG和fNIRS进行分窗
        eeg_windows = sliding_window(eeg_data_processed, WINDOW_LENGTH, WINDOW_STEP, EEG_SFREQ_TARGET)
        hbo_windows = sliding_window(hbo_data_corrected, WINDOW_LENGTH, WINDOW_STEP, FNIRS_SFREQ)
        hbr_windows = sliding_window(hbr_data_corrected, WINDOW_LENGTH, WINDOW_STEP, FNIRS_SFREQ)
        fnirs_windows = np.concatenate([hbo_windows, hbr_windows], axis=1)  # 合并HbO和HbR

        # 2. 获取当前trial的标签，仅标签为2时保留后面一半窗口
        trial_label = TRIAL_LABELS[trial_idx - 1]
        if trial_label == 2:
            n_total_windows = eeg_windows.shape[0]  # 单个trial的总窗口数
            n_keep_windows = n_total_windows // 2   # 保留后面一半窗口（整数除法，确保取整）
            # 切片保留后面一半窗口（如6个窗口→保留后3个，索引[-3:]）
            eeg_windows = eeg_windows[-n_keep_windows:]
            fnirs_windows = fnirs_windows[-n_keep_windows:]
            print(f"    Trial {trial_idx} (label=2): 保留后 {n_keep_windows}/{n_total_windows} 个窗口")

        # 3. 为切片后的窗口分配标签（数量与窗口数一致）
        window_labels = np.full(eeg_windows.shape[0], trial_label)

        # 添加到总列表（未改动）
        all_eeg_windows.append(eeg_windows)
        all_fnirs_windows.append(fnirs_windows)
        all_labels.append(window_labels)

    # 合并数据并保存（未改动）
    eeg_all_windows = np.concatenate(all_eeg_windows, axis=0)
    fnirs_all_windows = np.concatenate(all_fnirs_windows, axis=0)
    labels_all = np.concatenate(all_labels, axis=0)

    # 检查HbR数据（未改动）
    hbr_data = fnirs_all_windows[:, 36:, :]
    if np.any(np.isnan(hbr_data)):
        print(f"警告: 合并后的HbR数据包含NaN值，数量: {np.sum(np.isnan(hbr_data))}")
    else:
        print("合并后的HbR数据没有NaN值")

    # 保存数据（未改动）
    save_dict = {
        'eeg': eeg_all_windows,          # 形状: (平衡后总样本数, 32, 2500)
        'fnirs': fnirs_all_windows,      # 形状: (平衡后总样本数, 72, 84)
        'labels': labels_all             # 形状: (平衡后总样本数,)
    }
    # save_dir = 'preprocessed_LOSO'
    save_dir = 'temp'
    os.makedirs(save_dir, exist_ok=True)
    save_name = f'sub_{subject_no}.npy'
    np.save(os.path.join(save_dir, save_name), save_dict)
    print(f"Saved processed data for subject {subject_no}")

    # 加载检查（未改动）
    loaded_data = np.load(os.path.join(save_dir, save_name), allow_pickle=True).item()
    hbr_check = loaded_data['fnirs'][:, 36:, :]
    if np.any(np.isnan(hbr_check)):
        print(f"警告: 保存后的文件中的HbR数据包含NaN值，数量: {np.sum(np.isnan(hbr_check))}")
    else:
        print("保存后的文件中的HbR数据没有NaN值")

print("All subjects processed successfully!")
