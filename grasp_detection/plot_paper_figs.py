import os
import csv
import numpy as np

# 设置无 GUI 后端，防止 headless 环境报错
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    base_dir = "/home/gb/My_respositories/anygrasp_sdk/grasp_detection"
    csv_path = os.path.join(base_dir, "paper_exp_per_sample_results.csv")
    
    # 1. 读取数据
    sample_ids = []
    anygrasp_b = []
    anygrasp_d = []
    cgn_nat_palm = []
    cgn_msk_palm = []
    cgn_nat_contact = []
    cgn_msk_contact = []
    cgn_nat_grasps = []
    cgn_msk_grasps = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_ids.append(int(row['sample_id']))
            anygrasp_b.append(float(row['AnyGrasp_Group_B_precision']))
            anygrasp_d.append(float(row['AnyGrasp_Group_D_precision']))
            cgn_nat_palm.append(float(row['CGN_Native_palm_precision']))
            cgn_msk_palm.append(float(row['CGN_Mask_palm_precision']))
            cgn_nat_contact.append(float(row['CGN_Native_contact_precision']))
            cgn_msk_contact.append(float(row['CGN_Mask_contact_precision']))
            cgn_nat_grasps.append(float(row['CGN_Native_num_grasps']))
            cgn_msk_grasps.append(float(row['CGN_Mask_num_grasps']))
            
    sample_ids = np.array(sample_ids)
    anygrasp_b = np.array(anygrasp_b)
    anygrasp_d = np.array(anygrasp_d)
    cgn_nat_palm = np.array(cgn_nat_palm)
    cgn_msk_palm = np.array(cgn_msk_palm)
    cgn_nat_contact = np.array(cgn_nat_contact)
    cgn_msk_contact = np.array(cgn_msk_contact)
    cgn_nat_grasps = np.array(cgn_nat_grasps)
    cgn_msk_grasps = np.array(cgn_msk_grasps)
    
    # ------------------ Fig 1: fig_precision_bar_with_error.png ------------------
    # 各组均值和标准差
    means = [
        np.mean(anygrasp_b), np.mean(anygrasp_d),
        np.mean(cgn_nat_palm), np.mean(cgn_msk_palm),
        np.mean(cgn_nat_contact), np.mean(cgn_msk_contact)
    ]
    stds = [
        np.std(anygrasp_b, ddof=1), np.std(anygrasp_d, ddof=1),
        np.std(cgn_nat_palm, ddof=1), np.std(cgn_msk_palm, ddof=1),
        np.std(cgn_nat_contact, ddof=1), np.std(cgn_msk_contact, ddof=1)
    ]
    labels = [
        "AnyGrasp\n(Orig. Depth)", "AnyGrasp\n+Qd+SAM2 (Strict)",
        "CGN Native\n(Palm Center)", "CGN Mask\n(Palm Center)",
        "CGN Native\n(Contact Point)", "CGN Mask\n(Contact Point)"
    ]
    
    plt.figure(figsize=(10, 6), facecolor='white')
    colors = ['#aec7e8', '#1f77b4', '#ffbb78', '#ff7f0e', '#c5b0d5', '#9467bd']
    bars = plt.bar(labels, means, yerr=stds, capsize=8, color=colors, edgecolor='black', alpha=0.85)
    plt.ylabel("Target Mask Precision", fontsize=12)
    plt.title("Target Mask Precision Comparison Across Different Configuration Groups", fontsize=14, pad=15)
    plt.ylim(0, 1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    # 标注数值
    for bar, val in zip(bars, means):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{val:.3f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    plt.tight_layout()
    fig1_path = os.path.join(base_dir, "fig_precision_bar_with_error.png")
    plt.savefig(fig1_path, dpi=300, facecolor='white')
    plt.close()
    print(f"Fig 1 saved to {fig1_path}")
    
    # ------------------ Fig 2: fig_cgn_native_vs_mask_paired.png ------------------
    # 21 个样本的 paired line plot
    plt.figure(figsize=(12, 6), facecolor='white')
    plt.plot(sample_ids, cgn_nat_contact, 'o-', color='#d62728', label='CGN-Native (Contact Point)', alpha=0.7, markersize=8)
    plt.plot(sample_ids, cgn_msk_contact, 's-', color='#2ca02c', label='CGN-Mask (Contact Point)', alpha=0.7, markersize=8)
    
    # 用轻微的虚线连接同一个样本的两个点
    for i in range(len(sample_ids)):
        plt.vlines(sample_ids[i], cgn_nat_contact[i], cgn_msk_contact[i], colors='gray', linestyles='dashed', alpha=0.5)
        
    plt.xlabel("Sample ID", fontsize=12)
    plt.ylabel("Target Mask Precision", fontsize=12)
    plt.xticks(sample_ids)
    plt.title("Paired Performance Comparison of Contact-GraspNet on 21 Samples", fontsize=14, pad=15)
    plt.ylim(-0.05, 1.05)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(axis='both', linestyle='--', alpha=0.3)
    plt.tight_layout()
    fig2_path = os.path.join(base_dir, "fig_cgn_native_vs_mask_paired.png")
    plt.savefig(fig2_path, dpi=300, facecolor='white')
    plt.close()
    print(f"Fig 2 saved to {fig2_path}")
    
    # ------------------ Fig 3: fig_candidate_count_comparison.png ------------------
    # 候选数量对比
    # AnyGrasp (B): 50.0, AnyGrasp (D): 48.71
    # CGN-Native: 50.0, CGN-Mask: 15.52
    c_means = [50.0, 48.71, np.mean(cgn_nat_grasps), np.mean(cgn_msk_grasps)]
    c_labels = [
        "AnyGrasp\n(Orig. Depth)", "AnyGrasp\n+Qd+SAM2",
        "CGN Native\n(Full Scene)", "CGN Mask\n(Local Region)"
    ]
    plt.figure(figsize=(8, 6), facecolor='white')
    c_colors = ['#1f77b4', '#1f77b4', '#aec7e8', '#aec7e8']
    c_bars = plt.bar(c_labels, c_means, color=c_colors, edgecolor='black', alpha=0.8, width=0.5)
    plt.ylabel("Average Number of Candidate Grasps", fontsize=12)
    plt.title("Comparison of Average Generated Candidate Grasps", fontsize=14, pad=15)
    plt.ylim(0, 60)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    for bar, val in zip(c_bars, c_means):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{val:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    plt.tight_layout()
    fig3_path = os.path.join(base_dir, "fig_candidate_count_comparison.png")
    plt.savefig(fig3_path, dpi=300, facecolor='white')
    plt.close()
    print(f"Fig 3 saved to {fig3_path}")
    
    # ------------------ Fig 4: fig_palm_vs_contact_explanation.png ------------------
    # CGN-Mask 配置下，Palm Center 与 Contact Point Precision 对比
    x_indices = np.arange(len(sample_ids))
    width = 0.35
    
    plt.figure(figsize=(12, 6), facecolor='white')
    plt.bar(x_indices - width/2, cgn_msk_palm, width, label='CGN Mask (Palm Center)', color='#ffbb78', edgecolor='black', alpha=0.8)
    plt.bar(x_indices + width/2, cgn_msk_contact, width, label='CGN Mask (Contact Point)', color='#9467bd', edgecolor='black', alpha=0.8)
    
    plt.xlabel("Sample ID", fontsize=12)
    plt.ylabel("Target Mask Precision", fontsize=12)
    plt.title("Precision Discrepancy: Palm Center vs. Contact Point Projection (CGN-Mask)", fontsize=14, pad=15)
    plt.xticks(x_indices, [f"{i:02d}" for i in sample_ids])
    plt.ylim(-0.05, 1.05)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig4_path = os.path.join(base_dir, "fig_palm_vs_contact_explanation.png")
    plt.savefig(fig4_path, dpi=300, facecolor='white')
    plt.close()
    print(f"Fig 4 saved to {fig4_path}")

if __name__ == "__main__":
    main()
