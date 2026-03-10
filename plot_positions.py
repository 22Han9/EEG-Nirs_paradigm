"""
========================
Plot electrode positions
========================

.. currentmodule:: eeg_positions
"""  # noqa: D400 D205

# %%
# We need to import some functions.
import matplotlib
from PyQt5 import QtCore, QtWidgets
matplotlib.use("Qt5Agg")  # 声明使用QT5
import matplotlib.pyplot as plt
from eeg_positions import get_elec_coords, plot_coords
# %%
# Let's start with the basic 10-20 system in two dimensions:

import json
# labels = ['Nz', 'T10', 'Iz', 'T9', 'Cz', 'Fpz', 'AFz', 'Fz', 'FCz', 'CPz', 'Pz', 'POz', 'Oz', 'F10', 'FT10', 'P10', 'PO10', 'I2', 'F9', 'FT9', 'P9', 'PO9', 'I1', 'T7',
#        'C5', 'C3', 'C1', 'C2', 'C4', 'C6', 'T8', 'Fp2', 'AF8', 'F8', 'FT8', 'TP8', 'P8', 'PO8', 'O2', 'Fp1', 'AF7', 'F7', 'FT7', 'TP7', 'P7', 'PO7', 'O1', 'F5', 'F3', 'F1', 
#        'F2', 'F4', 'F6', 'FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6', 'CP5', 'CP3', 'CP1', 'CP2', 'CP4', 'CP6', 'P5', 'P3', 'P1', 'P2', 'P4', 'P6', 'AF3', 'AF4', 'PO3', "PO4"]
labels = ['AFp1', 'AFp2', 'AFF1h', 'AFF2h', 'F1', 'F3', 'F5', 'F2', 'F4', 'F6', 'FCC3h', 'FCC5h', 'CCP5h', 'CCP3h', 'FCz', 'FCC4h', 'FCC6h', 'CCP4h', 'CCP6h', 'T7', 'T8', 'P3', 'P1', 'Pz', 'P2', 'P4', 'CPz', 'Cz', 'PPO1h', 'PPO2h', 'POO1', 'POO2']
# labels = ["FP1", "FP2","F7", "F3", "FZ", "F4", "F8", "FT7",
#                "FC3", "FCZ", "FC4", "FT8", "T7", "C3", "CZ", "C4",
#                "T8", "TP7", "CP3", "block", "CP4", "TP8", "P7", "P3",
#                "PZ", "P4", "P8", "O1", "OZ", "O2",  "EOG1", "EOG2"
#                ]
# 假设你已经获得了 coords 的数据
coords = get_elec_coords(system='1005', dim="3d", as_mne_montage=False)
data = [coords['label'].tolist(), coords['x'].tolist(), coords['y'].tolist(), coords['z'].tolist()]
# 要存储的结果列表
results = []
count = 0
# 遍历数据并筛选符合条件的标签
for index in range(len(data[0])):  # 0 表示 'label' 列表
    label = data[0][index]
    # if label.upper() in labels:
    if label.upper() in [l.upper() for l in labels]:
        result = {
            'label': label,
            'x': data[1][index],  # 1 表示 'x' 列表
            'y': data[2][index],   # 2 表示 'y' 列表
            'z': data[3][index],   # 2 表示 'y' 列表


        }
        results.append(result)
        
    # else:
        # print(label)
# 将结果写入 JSON 文件
print(len(results))
with open('position.json', 'w') as json_file:
    json.dump(results, json_file, indent=4)

print("Data successfully written to selected_coords.json")




# labels = ['Nz', 'T10', 'Iz', 'T9', 'Cz', 'Fpz', 'AFz', 'Fz', 'FCz', 'CPz', 'Pz', 'POz', 'Oz', 'F10', 'FT10', 'P10', 'PO10', 'I2', 'F9', 'FT9', 'P9', 'PO9', 'I1', 'T7',
#        'C5', 'C3', 'C1', 'C2', 'C4', 'C6', 'T8', 'Fp2', 'AF8', 'F8', 'FT8', 'TP8', 'P8', 'PO8', 'O2', 'Fp1', 'AF7', 'F7', 'FT7', 'TP7', 'P7', 'PO7', 'O1', 'F5', 'F3', 'F1', 
#        'F2', 'F4', 'F6', 'FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6', 'CP5', 'CP3', 'CP1', 'CP2', 'CP4', 'CP6', 'P5', 'P3', 'P1', 'P2', 'P4', 'P6', 'AF3', 'AF4', 'PO3', "PO4"]

# coords=get_elec_coords(system = '1005', as_mne_montage = False)
# data=[coords['label'].tolist(), coords['x'].tolist(),
#                 coords['y'].tolist()]

# # print(data)
# for item in range(len(data)):
#     # print(item)
#     # print(len(item))
#     if item > 0:
#         continue
#     for index in range(len(data[item])):
#         label = data[item][index]
#         if label in labels:
#             print({'labes': label, "x": data[1][index], "y": data[2][index] })
# coords = get_elec_coords(
#     system="1020",
#     dim="2d",
# )

# # %%
# # This function returns a ``pandas.DataFrame`` object:

# coords.head()

# # %%
# # Now let's plot these coordinates.
# # We can supply some style arguments to :func:`eeg_positions.plot_coords` to control
# # the color of the electrodes and the text annotations.

# fig, ax = plot_coords(
#     coords, scatter_kwargs={"color": "g"}, text_kwargs={"fontsize": 10}
# )

# fig

# # %%
# # Notice that the "landmarks" ``NAS``, ``LPA``, and ``RPA`` are included. We can drop
# # these by passing ``drop_landmarks=True`` to :func:`get_elec_coords`:

# coords = get_elec_coords(
#     system="1005",
#     drop_landmarks=True,
#     dim="2d",
# )
# fig, ax = plot_coords(
#     coords, scatter_kwargs={"color": "g"}, text_kwargs={"fontsize": 10}
# )

# fig

# # %%
# # Often, we might have a list of electrode names that we would like to plot. For
# # example, let's assume we have the following 64 channel labels (based on the 10-05
# # system):

# chans = """Fp1 AF7 AF3 F1 F3 F5 F7 Fp2 AF8 AF4 F2 F4 F6 F8 FT7 FC5 FC3
# FC1 C1 C3 C5 T7 TP7 CP5 CP3 CP1 FT8 FC6 FC4 FC2 C2 C4 C6 T8 TP8 CP6 CP4
# CP2 P1 P3 P5 P7 P9 PO7 PO3 O1 P2 P4 P6 P8 P10 PO8 PO4 O2 Iz Oz POz Pz
# Fz AFz Fpz CPz Cz FCz""".split()

# # %%
# # Many experiments aggregate electrodes into regions of interest (ROIs), which we could
# # visualize with different colors. Let's get their coordinates first:

# coords = get_elec_coords(elec_names=chans)

# # %%
# # Now we specifiy individual colors using the ``scatter_kwargs``` argument. We create a
# # list of 64 colors corresponding to our 64 coordinates (in the original order as
# # provided by ``chans``):
# colors = (
#     ["salmon"] * 14
#     + ["skyblue"] * 24
#     + ["violet"] * 16
#     + ["lightgreen"] * 7
#     + ["khaki"] * 3
# )

# # sphinx_gallery_thumbnail_number = 3
# fig, ax = plot_coords(
#     coords,
#     scatter_kwargs={
#         "s": 150,  # electrode size
#         "color": colors,
#         "edgecolors": "black",  # black electrode outline
#         "linewidths": 0.5,  # thin outline
#     },
#     text_kwargs={
#         "ha": "center",  # center electrode label horizontally
#         "va": "center",  # center electrode label vertically
#         "fontsize": 5,  # smaller font size
#     },
# )

# # %%
# # We can also plot in 3D. Let's pick a system with more electrodes now:

# coords = get_elec_coords(
#     system="1010",
#     drop_landmarks=True,
#     dim="3d",
# )

# fig, ax = plot_coords(coords, text_kwargs=dict(fontsize=7))

# fig

# # %%
# # When using these commands from an interactive Python session, try to set
# # the IPython magic ``%matplotlib`` or ``%matplotlib qt``, which will allow you to
# # freely view the 3D plot and rotate the camera.

# plt.show()