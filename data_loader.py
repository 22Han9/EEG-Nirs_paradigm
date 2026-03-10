# load and check data
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter


def load_npy_data(data_dir, subject_ids=None):
    """
    加载预处理后的 npy 数据

    参数:
        data_dir: 数据目录路径
        subject_ids: 要加载的受试者ID列表，如果为None则加载所有受试者

    返回:
        包含所有受试者数据的字典
    """
    # 获取所有 npy 文件
    npy_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]

    if subject_ids is not None:
        # 过滤出指定的受试者
        npy_files = [f for f in npy_files if any(f'sub_{sid}.npy' == f for sid in subject_ids)]

    if not npy_files:
        raise ValueError(f"在目录 {data_dir} 中未找到 npy 文件")

    # 加载数据
    data_dict = {}
    for file in npy_files:
        file_path = os.path.join(data_dir, file)
        subject_data = np.load(file_path, allow_pickle=True).item()
        subject_id = file.split('.')[0]  # 提取 subject ID
        data_dict[subject_id] = subject_data
        #检查加载数据是否有NAN
        # comprehensive_data_diagnosis(data_dict, subject_id)

        # 打印数据信息
        print(f"加载 {subject_id}:")
        print(f"  EEG 形状: {subject_data['eeg'].shape}")
        print(f"  fNIRS 形状: {subject_data['fnirs'].shape}")
        print(f"  标签形状: {subject_data['labels'].shape}")
        print(f"  标签分布: {dict(Counter(subject_data['labels']))}")
        print()

    return data_dict


def analyze_data_statistics(data_dict):
    """
    分析数据的统计信息

    参数:
        data_dict: 包含所有受试者数据的字典
    """
    print("=" * 50)
    print("数据统计分析")
    print("=" * 50)

    total_samples = 0
    label_counts = {}
    eeg_shapes = []
    fnirs_shapes = []

    for subject_id, data in data_dict.items():
        total_samples += data['eeg'].shape[0]
        eeg_shapes.append(data['eeg'].shape)
        fnirs_shapes.append(data['fnirs'].shape)

        # 统计标签
        for label in data['labels']:
            label_counts[label] = label_counts.get(label, 0) + 1

    # 检查所有受试者的数据维度是否一致
    eeg_consistent = all(shape == eeg_shapes[0] for shape in eeg_shapes)
    fnirs_consistent = all(shape == fnirs_shapes[0] for shape in fnirs_shapes)

    print(f"总样本数: {total_samples}")
    print(f"受试者数量: {len(data_dict)}")
    print(f"标签分布: {label_counts}")
    print(f"EEG 维度一致性: {eeg_consistent}")
    print(f"fNIRS 维度一致性: {fnirs_consistent}")

    if eeg_consistent:
        print(f"EEG 数据维度: {eeg_shapes[0]}")
    else:
        print("警告: EEG 数据维度不一致")
        for i, shape in enumerate(eeg_shapes):
            print(f"  受试者 {i + 1}: {shape}")

    if fnirs_consistent:
        print(f"fNIRS 数据维度: {fnirs_shapes[0]}")
    else:
        print("警告: fNIRS 数据维度不一致")
        for i, shape in enumerate(fnirs_shapes):
            print(f"  受试者 {i + 1}: {shape}")


def visualize_sample(data_dict, subject_id, sample_idx=0):
    """
    可视化单个样本的数据

    参数:
        data_dict: 包含所有受试者数据的字典
        subject_id: 受试者ID
        sample_idx: 样本索引
    """
    if subject_id not in data_dict:
        print(f"错误: 未找到受试者 {subject_id}")
        return

    data = data_dict[subject_id]

    if sample_idx >= data['eeg'].shape[0]:
        print(f"错误: 样本索引 {sample_idx} 超出范围 (0-{data['eeg'].shape[0] - 1})")
        return

    # 创建图形
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # 绘制EEG数据
    eeg_data = data['eeg'][sample_idx]
    time_points_eeg = np.arange(eeg_data.shape[1]) / 250  # 假设采样率为250Hz
    for i in range(min(5, eeg_data.shape[0])):  # 只显示前5个通道
        axes[0].plot(time_points_eeg, eeg_data[i] + i * 50, label=f'通道 {i + 1}')
    axes[0].set_title(f'{subject_id} - 样本 {sample_idx} - EEG数据 (前5个通道)')
    axes[0].set_xlabel('时间 (秒)')
    axes[0].set_ylabel('幅度 (μV)')
    axes[0].legend()
    axes[0].grid(True)

    # 绘制fNIRS HbO数据
    fnirs_data = data['fnirs'][sample_idx]
    time_points_fnirs = np.arange(fnirs_data.shape[1]) / 4.2  # 假设采样率为4.2Hz

    # HbO通道 (前36个通道)
    hbo_data = fnirs_data[:36]  # 前36个通道是HbO
    for i in range(min(5, hbo_data.shape[0])):
        axes[1].plot(time_points_fnirs, hbo_data[i],
                     label=f'HbO 通道 {i + 1}', alpha=0.7)
    axes[1].set_title(f'{subject_id} - 样本 {sample_idx} - HbO数据 (前5个通道)')
    axes[1].set_xlabel('时间 (秒)')
    axes[1].set_ylabel('HbO浓度变化 (μM)')
    axes[1].legend()
    axes[1].grid(True)

    # 绘制fNIRS HbR数据
    hbr_data = fnirs_data[36:]  # 后36个通道是HbR
    for i in range(min(5, hbr_data.shape[0])):
        axes[2].plot(time_points_fnirs, hbr_data[i],
                     label=f'HbR 通道 {i + 1}', alpha=0.7)
    axes[2].set_title(f'{subject_id} - 样本 {sample_idx} - HbR数据 (前5个通道)')
    axes[2].set_xlabel('时间 (秒)')
    axes[2].set_ylabel('HbR浓度变化 (μM)')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

    # 打印数据统计信息
    print(f"样本标签: {data['labels'][sample_idx]}")
    print(f"EEG数据范围: [{np.min(eeg_data):.4f}, {np.max(eeg_data):.4f}]")
    print(f"HbO数据范围: [{np.min(hbo_data):.6f}, {np.max(hbo_data):.6f}]")
    print(f"HbR数据范围: [{np.min(hbr_data):.6f}, {np.max(hbr_data):.6f}]")

    # 检查fNIRS数据是否有变化
    hbo_std = np.std(hbo_data, axis=1)
    hbr_std = np.std(hbr_data, axis=1)
    print(f"HbO标准差范围: [{np.min(hbo_std):.6f}, {np.max(hbo_std):.6f}]")
    print(f"HbR标准差范围: [{np.min(hbr_std):.6f}, {np.max(hbr_std):.6f}]")

    # 如果标准差非常小，可能是预处理有问题
    if np.max(hbo_std) < 1e-6 or np.max(hbr_std) < 1e-6:
        print("警告: fNIRS数据的标准差非常小，可能是预处理过程中出现了问题")


def prepare_data_for_training(data_dict, subjects=None):
    """
    准备用于训练的数据

    参数:
        data_dict: 包含所有受试者数据的字典
        subjects: 要包含的受试者列表，如果为None则包含所有受试者

    返回:
        X_eeg: EEG特征矩阵
        X_fnirs: fNIRS特征矩阵
        y: 标签向量
    """
    if subjects is None:
        subjects = list(data_dict.keys())

    X_eeg_list = []
    X_fnirs_list = []
    y_list = []

    for subject in subjects:
        if subject not in data_dict:
            print(f"警告: 跳过不存在的受试者 {subject}")
            continue

        data = data_dict[subject]
        X_eeg_list.append(data['eeg'])
        X_fnirs_list.append(data['fnirs'])
        y_list.append(data['labels'])

    # 合并所有受试者的数据
    X_eeg = np.concatenate(X_eeg_list, axis=0)
    X_fnirs = np.concatenate(X_fnirs_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    print(f"准备训练数据:")
    print(f"  EEG 特征形状: {X_eeg.shape}")
    print(f"  fNIRS 特征形状: {X_fnirs.shape}")
    print(f"  标签形状: {y.shape}")
    print(f"  标签分布: {dict(Counter(y))}")

    return X_eeg, X_fnirs, y


def comprehensive_data_diagnosis(data_dict, subject_no):
    """综合数据诊断"""
    print(f"\n=== 综合数据诊断 - {subject_no} ===")

    data = data_dict[subject_no]

    # 检查EEG数据
    eeg_data = data['eeg']
    print(f"EEG数据形状: {eeg_data.shape}")
    print(f"EEG数据范围: [{np.min(eeg_data):.4f}, {np.max(eeg_data):.4f}]")
    print(f"EEG数据是否有NaN: {np.any(np.isnan(eeg_data))}")
    print(f"EEG数据是否有Inf: {np.any(np.isinf(eeg_data))}")

    # 检查fNIRS数据
    fnirs_data = data['fnirs']
    print(f"fNIRS数据形状: {fnirs_data.shape}")

    # 检查HbO数据 (前36个通道)
    hbo_data = fnirs_data[:, :36, :]
    print(f"HbO数据范围: [{np.min(hbo_data):.6f}, {np.max(hbo_data):.6f}]")
    print(f"HbO数据是否有NaN: {np.any(np.isnan(hbo_data))}")
    print(f"HbO数据是否有Inf: {np.any(np.isinf(hbo_data))}")

    # 检查HbR数据 (后36个通道)
    hbr_data = fnirs_data[:, 36:, :]
    print(f"HbR数据范围: [{np.min(hbr_data):.6f}, {np.max(hbr_data):.6f}]")
    print(f"HbR数据是否有NaN: {np.any(np.isnan(hbr_data))}")
    print(f"HbR数据是否有Inf: {np.any(np.isinf(hbr_data))}")

    # 检查标签数据
    labels = data['labels']
    print(f"标签数据形状: {labels.shape}")
    print(f"标签数据是否有NaN: {np.any(np.isnan(labels))}")
    print(f"标签数据是否有Inf: {np.any(np.isinf(labels))}")

    # 计算标准差
    try:
        eeg_std = np.std(eeg_data, axis=(1, 2))
        hbo_std = np.std(hbo_data, axis=(1, 2))
        hbr_std = np.std(hbr_data, axis=(1, 2))

        print(f"EEG标准差范围: [{np.min(eeg_std):.4f}, {np.max(eeg_std):.4f}]")
        print(f"HbO标准差范围: [{np.min(hbo_std):.6f}, {np.max(hbo_std):.6f}]")
        print(f"HbR标准差范围: [{np.min(hbr_std):.6f}, {np.max(hbr_std):.6f}]")
    except Exception as e:
        print(f"计算标准差时出错: {e}")

    # 检查是否有常数样本
    constant_eeg = np.sum(np.std(eeg_data, axis=(1, 2)) == 0)
    constant_hbo = np.sum(np.std(hbo_data, axis=(1, 2)) == 0)
    constant_hbr = np.sum(np.std(hbr_data, axis=(1, 2)) == 0)

    print(f"常数EEG样本数: {constant_eeg}")
    print(f"常数HbO样本数: {constant_hbo}")
    print(f"常数HbR样本数: {constant_hbr}")

def main():
    # 设置数据目录
    data_dir = 'preprocessed'

    try:
        # 加载所有受试者数据
        data_dict = load_npy_data(data_dir)

        # 分析数据统计信息
        analyze_data_statistics(data_dict)

        # 可视化第一个受试者的第一个样本
        first_subject = list(data_dict.keys())[0]
        visualize_sample(data_dict, first_subject, sample_idx=0)

        # 准备训练数据
        X_eeg, X_fnirs, y = prepare_data_for_training(data_dict)

        # 这里可以添加进一步的数据处理或模型训练代码
        # 例如:
        # from sklearn.model_selection import train_test_split
        # X_train_eeg, X_test_eeg, X_train_fnirs, X_test_fnirs, y_train, y_test = train_test_split(
        #     X_eeg, X_fnirs, y, test_size=0.2, random_state=42, stratify=y
        # )

    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()