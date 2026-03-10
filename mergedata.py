import numpy as np
import os
from pathlib import Path


def merge_subject_data(subject_pairs, folder1, folder2, output_dir):
    """
    合并同一个subject在两个文件夹中的数据

    参数:
    subject_pairs: 包含subject对应关系的列表，如[('001', '011'), ...]
    folder1: 第一个文件夹路径
    folder2: 第二个文件夹路径
    output_dir: 输出目录
    """

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    for sub1, sub2 in subject_pairs:
        print(f"正在合并 subject {sub1} 和 {sub2}...")

        # 构建文件路径
        file1 = os.path.join(folder1, f"sub_{sub1}.npy")
        file2 = os.path.join(folder2, f"sub_{sub2}.npy")

        # 检查文件是否存在
        if not os.path.exists(file1):
            print(f"警告: 文件 {file1} 不存在，跳过")
            continue
        if not os.path.exists(file2):
            print(f"警告: 文件 {file2} 不存在，跳过")
            continue

        try:
            # 加载数据
            data1 = np.load(file1, allow_pickle=True).item()
            data2 = np.load(file2, allow_pickle=True).item()

            # 合并数据
            merged_data = {}

            for key in ['eeg', 'fnirs', 'labels']:
                if key in data1 and key in data2:
                    # 沿第一个维度(samples)连接
                    merged_data[key] = np.concatenate([data1[key], data2[key]], axis=0)
                    print(f"  {key}: {data1[key].shape} + {data2[key].shape} -> {merged_data[key].shape}")
                else:
                    print(f"警告: 键 '{key}' 在其中一个文件中不存在")

            # 保存合并后的数据
            output_file = os.path.join(output_dir, f"sub_{sub1}.npy")
            np.save(output_file, merged_data)
            print(f"已保存合并数据到: {output_file}\n")

        except Exception as e:
            print(f"处理 subject {sub1} 和 {sub2} 时出错: {e}")
            continue


def main():
    # 定义文件夹路径，两个文件夹合并
    folder1 = "preprocessed_LOSO"
    folder2 = "preprocessed_LOSO_2"
    # output_dir = "preprocessed_merged"
    output_dir = "temp"
    # 定义subject对应关系
    # subject_pairs = [
    #     ('001', '011'),
    #     ('002', '012'),
    #     ('003', '013'),
    #     ('004', '014'),
    #     ('005', '015'),
    #     ('006', '016'),
    #     ('007', '017'),
    #     ('008', '018')
    # ]
    subject_pairs = [
        ('009', '029'),
        ('010', '030'),
        ('011', '031')
    ]

    # 合并数据
    merge_subject_data(subject_pairs, folder1, folder2, output_dir)
    print("所有subject数据合并完成！")


if __name__ == "__main__":
    main()