# -*- coding: utf-8 -*-

"""
经典的TDA分析脚本 (基线版)

功能:
1.  使用固定的 `max_edge_length` 对所有epoch的soft prompt进行TDA分析。
2.  为所有实验提供一个统一的拓扑分析基线。
3.  计算并可视化核心的TDA指标（H₀/H₁数量、平均寿命、持久性熵）。
"""

import gudhi as gd
import torch
import numpy as np
import os
import matplotlib.pyplot as plt

# --- 1. 配置区域 ---

# <<< 关键修改：指向“经典训练”脚本的输出目录
MODEL_SAVE_DIR = "../models/"
# <<< 关键修改：结果将保存在一个新的子目录中
RESULTS_SAVE_DIR = "../tda_analysis/"
os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)

# <<< 关键修改：定义一个固定的、全局的观察尺度
# 这个值需要根据经验设定，如果条带图/持久性图中的特征过少或过多，可以调整此值
MAX_EDGE_LENGTH = 3

# --- 2. 加载 Soft Prompt 嵌入向量 ---
print(f"Loading classic soft prompt embeddings from '{MODEL_SAVE_DIR}'...")
# 文件名已根据经典训练脚本的输出格式更新
soft_prompt_files = sorted(
    [f for f in os.listdir(MODEL_SAVE_DIR) if f.startswith("soft_prompt_epoch_") and f.endswith(".pt")])

if not soft_prompt_files:
    print(f"错误: 在 '{MODEL_SAVE_DIR}' 目录中未找到 'soft_prompt_epoch_*.pt' 文件。")
    print("请确认您已经运行了“经典softprompt训练”脚本，并且路径正确。")
    exit()

epoch_to_embeds = {}
for filename in soft_prompt_files:
    try:
        epoch_str = filename.split('_')[-1].replace('.pt', '')
        epoch = int(epoch_str)
        filepath = os.path.join(MODEL_SAVE_DIR, filename)
        loaded_optim_embeds = torch.load(filepath, map_location=torch.device('cpu'))
        # 确保数据是numpy array格式
        epoch_to_embeds[epoch] = loaded_optim_embeds.squeeze(0).numpy()
    except Exception as e:
        print(f"加载或处理文件 {filename} 时出错: {e}")

print(f"成功加载 {len(epoch_to_embeds)} 个epoch的数据。")

# --- 3. 初始化指标存储 ---
tda_metrics = {
    'epoch': [],
    'H0_count': [], 'H1_count': [],
    'H0_avg_lifetime': [], 'H1_avg_lifetime': [],
    'H0_max_lifetime': [], 'H1_max_lifetime': [],
    'persistent_entropy': []
}


# --- 4. 遍历所有Epoch并执行TDA分析 ---
for epoch in sorted(epoch_to_embeds.keys()):
    print(f"\n--- 正在对 Epoch {epoch} 进行TDA分析 ---")
    points = epoch_to_embeds[epoch]

    if points.shape[0] < 2:
        print(f"跳过 epoch {epoch}: 数据点不足 ({points.shape[0]} 个点).")
        continue

    # 使用固定的 MAX_EDGE_LENGTH 构建Rips复形
    rips_complex = gd.RipsComplex(points=points, max_edge_length=MAX_EDGE_LENGTH)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)

    # 计算持久同调
    diag = simplex_tree.persistence(homology_coeff_field=2)

    # --- 分析和量化拓扑特征 ---
    h0_intervals = [interval[1] for interval in diag if interval[0] == 0]
    h1_intervals = [interval[1] for interval in diag if interval[0] == 1]

    # 计算生命周期（排除无限生命周期的H0特征，以进行有意义的统计）
    h0_lifetimes = [d - b for b, d in h0_intervals if d != float('inf')]
    h1_lifetimes = [d - b for b, d in h1_intervals if d != float('inf')]

    # 存储统计数据
    tda_metrics['epoch'].append(epoch)
    tda_metrics['H0_count'].append(len(h0_intervals))
    tda_metrics['H1_count'].append(len(h1_intervals))
    tda_metrics['H0_avg_lifetime'].append(np.mean(h0_lifetimes) if h0_lifetimes else 0)
    tda_metrics['H1_avg_lifetime'].append(np.mean(h1_lifetimes) if h1_lifetimes else 0)
    tda_metrics['H0_max_lifetime'].append(np.max(h0_lifetimes) if h0_lifetimes else 0)
    tda_metrics['H1_max_lifetime'].append(np.max(h1_lifetimes) if h1_lifetimes else 0)

    # 计算持久性熵
    all_lifetimes = h0_lifetimes + h1_lifetimes
    total_lifetime = np.sum(all_lifetimes)
    if total_lifetime > 1e-9: # 避免除以零
        normalized_lifetimes = [l / total_lifetime for l in all_lifetimes]
        entropy = -np.sum([p * np.log(p) for p in normalized_lifetimes if p > 0])
        tda_metrics['persistent_entropy'].append(entropy)
    else:
        tda_metrics['persistent_entropy'].append(0)

    # --- 可视化部分epoch的结果 ---
    if epoch % 10 == 0 or epoch == sorted(epoch_to_embeds.keys())[-1]:
        print(f"正在为 epoch {epoch} 生成图像...")
        # Barcode
        plt.figure(figsize=(10, 6))
        gd.plot_persistence_barcode(diag)
        plt.title(f"Persistence Barcode - Epoch {epoch}")
        plt.savefig(os.path.join(RESULTS_SAVE_DIR, f"barcode_epoch_{epoch:03d}.png"))
        plt.close()

        # Persistence Diagram
        plt.figure(figsize=(8, 8))
        gd.plot_persistence_diagram(diag)
        plt.title(f"Persistence Diagram - Epoch {epoch}")
        plt.savefig(os.path.join(RESULTS_SAVE_DIR, f"diagram_epoch_{epoch:03d}.png"))
        plt.close()

# --- 5. 绘制跨所有epoch的汇总图表 ---
print("\n正在生成所有epoch的汇总图表...")

# 特征数量变化图
plt.figure(figsize=(12, 8))
plt.plot(tda_metrics['epoch'], tda_metrics['H0_count'], label='H0 Count (Connected Components)')
plt.plot(tda_metrics['epoch'], tda_metrics['H1_count'], label='H1 Count (Loops)')
plt.xlabel('Epoch')
plt.ylabel('Count')
plt.title('Topological Feature Count Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_SAVE_DIR, "feature_counts_over_epochs.png"))
plt.close()

# 平均生命周期变化图
plt.figure(figsize=(12, 8))
plt.plot(tda_metrics['epoch'], tda_metrics['H0_avg_lifetime'], label='H0 Average Lifetime')
plt.plot(tda_metrics['epoch'], tda_metrics['H1_avg_lifetime'], label='H1 Average Lifetime')
plt.xlabel('Epoch')
plt.ylabel('Average Lifetime')
plt.title('Average Lifetime of Topological Features Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_SAVE_DIR, "avg_lifetimes_over_epochs.png"))
plt.close()

# 持久性熵变化图
plt.figure(figsize=(12, 8))
plt.plot(tda_metrics['epoch'], tda_metrics['persistent_entropy'], label='Persistent Entropy')
plt.xlabel('Epoch')
plt.ylabel('Entropy')
plt.title('Persistent Entropy Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_SAVE_DIR, "persistent_entropy_over_epochs.png"))
plt.close()

print(f"\n经典TDA分析完成。所有图像已保存至 '{RESULTS_SAVE_DIR}' 目录。")