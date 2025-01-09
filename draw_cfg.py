import matplotlib.pyplot as plt
import numpy as np

# 数据
x = np.arange(0, 4.25, 0.25)  # 从 0 到 4，步长为 0.25

# nfe32 数据
wer_32 = [8.193, 4.066, 3.307, 2.846, 2.493, 2.467, 2.407, 2.292, 2.294, 2.294, 2.292, 2.209, 2.277, 2.18, 2.188, 2.176, 2.195]
sim_o_32 = [0.507, 0.577, 0.611, 0.628, 0.64, 0.648, 0.653, 0.658, 0.659, 0.659, 0.659, 0.657, 0.654, 0.65, 0.646, 0.642, 0.638]

# nfe16 数据
wer_16 = [11.66, 5.389, 3.852, 3.162, 2.875, 2.606, 2.515, 2.443, 2.408, 2.302, 2.373, 2.317, 2.346, 2.403, 2.363, 2.517, 2.591]
sim_o_16 = [0.484, 0.568, 0.609, 0.63, 0.642, 0.648, 0.653, 0.656, 0.656, 0.654, 0.648, 0.639, 0.625, 0.607, 0.585, 0.557, 0.53]

# 创建图形
fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制 nfe32 的 WER (蓝色曲线)
ax1.plot(x, wer_32, 'o--', color='blue', label='WER (nfe32)', linewidth=1.5, markersize=8)

# 绘制 nfe16 的 WER (橙色曲线)
ax1.plot(x, wer_16, 'o-', color='orange', label='WER (nfe16)', linewidth=1.5, markersize=8)

# 设置横纵坐标标签
ax1.set_xlabel('CFG scale', fontsize=12)
ax1.set_ylabel('WER', fontsize=12)
ax1.tick_params(axis='y')

# 添加第二个 y 轴用于 SIM-O
ax2 = ax1.twinx()

# 绘制 nfe32 的 SIM-O (绿色曲线)
ax2.plot(x, sim_o_32, 's--', color='green', label='SIM-O (nfe32)', linewidth=1.5, markersize=8)

# 绘制 nfe16 的 SIM-O (红色曲线)
ax2.plot(x, sim_o_16, 's-', color='red', label='SIM-O (nfe16)', linewidth=1.5, markersize=8)

# 设置 SIM-O 的 y 轴标签
ax2.set_ylabel('SIM-O', fontsize=12)
ax2.tick_params(axis='y')

# 添加图例
fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9), bbox_transform=ax1.transAxes)

# 设置标题
plt.title('CFG scale search for nfe32 and nfe16', fontsize=14)

# 设置横坐标显示 0.25 的刻度
ax1.set_xticks(np.arange(0, 4.25, 0.25))
ax1.grid(axis='x', linestyle='--', alpha=0.5)

# 添加垂直线
plt.axvline(x=2, color='black', linestyle='-.', linewidth=1.2)

# 调整布局
fig.tight_layout()  # 确保标签不重叠

# 保存图片
plt.savefig("nfe_cfg_scale_search.png", dpi=300, bbox_inches='tight')

# 显示图像
plt.show()
