# print trigger
import numpy as np

# data = np.loadtxt("D:/graduate/EEG+fNIRS/app2 _1000Hz/data/2025_10_28_11_33_30_fnris.csv").T
data = np.loadtxt("E:/graduate/EEG+fNIRS/app/data/2026_01_07_14_52_57_fnris 秦裕东1.csv").T
print(data.shape)
print(data.shape[0])
label = data[-1]

print([int(i) for i in label[label!= 0]])

