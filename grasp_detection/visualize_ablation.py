"""
论文实验数据收集与可视化脚本
用于生成 paper-ready 的对比图和消融实验表格

使用方法:
    python visualize_ablation.py --results_dir ./ablation_results --output_dir ./paper_figures
"""
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

parser = argparse.ArgumentParser(description='消融实验结果可视化')
parser.add_argument('--results_dir', default='./ablation_results', help='实验结果目录')
parser.add_argument('--output_dir', default='./paper_figures', help='论文图片输出目录')
parser.add_argument('--data_dir', default='./example_data', help='原始数据目录')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ===== 颜色方案（论文风格）=====
COLORS = {
    'A': '#E74C3C',  # 红色 - 基线最弱
    'B': '#F39C12',  # 橙色 - 有噪声
    'C': '#3498DB',  # 蓝色 - Qd改善
    'D': '#27AE60',  # 绿色 - 完整方法最优
}

GROUP_LABELS = {
    'A': 'RGB-only',
    'B': 'RGB-D (raw)',
    'C': 'RGB-D + Qd',
    'D': 'RGB-D + Qd + SAM2\n(Ours)',
}


def load_all_results(results_dir: str) -> dict:
    """加载所有组的实验结果"""
    results = {}
    for g in ['A', 'B', 'C', 'D']:
        json_path = os.path.join(results_dir, f'result_group_{g}.json')
        if os.path.exists(json_path):
            with open(json_path) as f:
                results[g] = json.load(f)
    return results


def plot_bar_comparison(results: dict, output_dir: str):
    """
    绘制四组对比柱状图（Top-1 抓取质量分数）
    适合论文图 Figure X: Ablation Study Results
    """
    groups = sorted(results.keys())
    scores = [results[g]['top1_score'] for g in groups]
    counts = [results[g]['num_grasps'] for g in groups]
    labels = [GROUP_LABELS.get(g, g) for g in groups]
    colors = [COLORS.get(g, '#7F8C8D') for g in groups]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- 子图1：Top-1 抓取得分 ---
    bars = ax1.bar(labels, scores, color=colors, edgecolor='white',
                   linewidth=1.5, width=0.6, zorder=3)
    ax1.set_ylabel('Top-1 Grasp Score', fontsize=12, fontweight='bold')
    ax1.set_title('Grasp Quality (Top-1 Score)', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, max(scores) * 1.3 if scores else 1.0)
    ax1.grid(axis='y', alpha=0.3, zorder=1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 在柱子上方标注数值
    for bar, score in zip(bars, scores):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{score:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 标注 "Ours" 箭头
    if 'D' in groups and 'A' in groups:
        improvement = scores[groups.index('D')] - scores[groups.index('A')]
        ax1.annotate(f'+{improvement:.3f}\n(vs RGB-only)',
                     xy=(groups.index('D'), scores[groups.index('D')]),
                     xytext=(groups.index('D') - 0.5, scores[groups.index('D')] * 1.15),
                     fontsize=9, color='#27AE60',
                     arrowprops=dict(arrowstyle='->', color='#27AE60', lw=1.5))

    # --- 子图2：检测到的抓取数量 ---
    bars2 = ax2.bar(labels, counts, color=colors, edgecolor='white',
                    linewidth=1.5, width=0.6, zorder=3)
    ax2.set_ylabel('Number of Valid Grasps', fontsize=12, fontweight='bold')
    ax2.set_title('Number of Detected Grasps', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, max(counts) * 1.3 if counts else 10)
    ax2.grid(axis='y', alpha=0.3, zorder=1)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    for bar, count in zip(bars2, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'ablation_bar_comparison.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    save_path_png = save_path.replace('.pdf', '.png')
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 柱状对比图保存到: {save_path_png}")
    return save_path_png


def plot_qd_visualization(data_dir: str, results_dir: str, output_dir: str):
    """
    绘制 Qd 图可视化（Figure: Depth Quality Map Construction）
    展示原始深度图 vs Qd 图 vs Qd 过滤后的点云投影
    """
    color_path = os.path.join(data_dir, 'color.png')
    depth_path = os.path.join(data_dir, 'depth.png')
    qd_path = os.path.join(results_dir, 'qd_map.png')

    if not all(os.path.exists(p) for p in [color_path, depth_path]):
        print("  ⚠ 数据文件不存在，跳过 Qd 可视化")
        return None

    color_img = np.array(Image.open(color_path))
    depth_img = np.array(Image.open(depth_path))

    fig, axes = plt.subplots(1, 3 if os.path.exists(qd_path) else 2,
                             figsize=(15, 5))

    # 子图1：RGB 图像
    axes[0].imshow(color_img)
    axes[0].set_title('RGB Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # 子图2：深度图（可视化）
    depth_vis = depth_img.astype(float)
    depth_vis[depth_vis == 0] = np.nan
    im2 = axes[1].imshow(depth_vis, cmap='viridis')
    axes[1].set_title('Raw Depth Map\n(noisy on transparent bag)',
                      fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label='Depth (mm)')

    # 子图3：Qd 图（如果存在）
    if os.path.exists(qd_path):
        qd_img = np.array(Image.open(qd_path))
        im3 = axes[2].imshow(qd_img, cmap='RdYlGn')
        axes[2].set_title('Depth Quality Map $Q_d$\n(red=unreliable, green=reliable)',
                          fontsize=12, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04, label='Quality Score')

    plt.suptitle('Depth Quality Map ($Q_d$) Construction', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'qd_visualization.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Qd 可视化图保存到: {save_path}")
    return save_path


def generate_latex_table(results: dict, output_dir: str):
    """
    生成 LaTeX 格式的消融实验表格，可直接粘贴到论文 TeX 文件中
    """
    groups = sorted(results.keys())
    
    table_lines = [
        r"% 消融实验表格 - 自动生成",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Ablation Study on Transparent Bag Grasping Detection.}",
        r"  \label{tab:ablation}",
        r"  \begin{tabular}{l|cc|cc}",
        r"    \hline",
        r"    \textbf{Method} & \textbf{RGB} & \textbf{Depth ($Q_d$+SAM2)} & \textbf{Top-1 Score} & \textbf{\#Grasps} \\",
        r"    \hline",
    ]

    method_configs = {
        'A': ('\\checkmark', '\\texttimes', '\\texttimes'),
        'B': ('\\checkmark', '\\checkmark', '\\texttimes'),
        'C': ('\\checkmark', '\\checkmark (filtered)', '\\texttimes'),
        'D': ('\\checkmark', '\\checkmark ($Q_d$)', '\\checkmark (SAM2)'),
    }

    for g in groups:
        r = results[g]
        score = r['top1_score']
        count = r['num_grasps']
        label = GROUP_LABELS.get(g, g).replace('\n', ' ')
        
        # 完整方法加粗
        fmt = r'\textbf{%s}' if g == 'D' else '%s'
        
        line = f"    {fmt % label} & \\checkmark & {'$Q_d$+SAM2' if g == 'D' else ('filtered' if g == 'C' else ('raw' if g == 'B' else 'const'))} & {fmt % f'{score:.3f}'} & {fmt % str(count)} \\\\"
        table_lines.append(line)

    table_lines.extend([
        r"    \hline",
        r"  \end{tabular}",
        r"\end{table}",
    ])

    latex_path = os.path.join(output_dir, 'ablation_table.tex')
    with open(latex_path, 'w') as f:
        f.write('\n'.join(table_lines))
    print(f"✓ LaTeX 表格保存到: {latex_path}")

    # 同时打印 Markdown 版本方便快速查看
    print("\n" + "=" * 60)
    print("消融实验 Markdown 表格：")
    print("=" * 60)
    print(f"| {'方法':<30} | {'Top-1分数':^12} | {'抓取数':^8} | {'耗时(s)':^8} |")
    print(f"|{'-'*32}|{'-'*14}|{'-'*10}|{'-'*10}|")
    for g in groups:
        r = results[g]
        label = GROUP_LABELS.get(g, g).replace('\n', ' ')
        marker = " ← **Ours**" if g == 'D' else ""
        print(f"| {label:<30} | {r['top1_score']:^12.4f} | {r['num_grasps']:^8} | {r['inference_time_s']:^8.2f} |{marker}")
    print("=" * 60)

    return latex_path


def main():
    results = load_all_results(args.results_dir)

    if not results:
        print(f"⚠ 在 {args.results_dir} 中未找到实验结果 JSON 文件")
        print("请先运行: python ablation_experiments.py --checkpoint_path <path>")
        return

    print(f"✓ 加载了 {len(results)} 组实验结果: {list(results.keys())}")

    # 生成各类图表
    bar_path = plot_bar_comparison(results, args.output_dir)
    qd_path = plot_qd_visualization(args.data_dir, args.results_dir, args.output_dir)
    latex_path = generate_latex_table(results, args.output_dir)

    print(f"\n✓ 所有论文图表已生成到: {args.output_dir}/")
    print("  - ablation_bar_comparison.png  ← 柱状对比图（可直接插入论文）")
    print("  - qd_visualization.png          ← Qd 图可视化")
    print("  - ablation_table.tex            ← LaTeX 表格代码")


if __name__ == '__main__':
    main()
