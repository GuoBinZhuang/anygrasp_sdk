import os
import json
import numpy as np

# 设置无 GUI 后端，防止 headless 环境报错
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

def project_points(pts, fx, fy, cx, cy):
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    z = np.clip(z, 1e-5, None)
    u = (x * fx / z) + cx
    v = (y * fy / z) + cy
    return np.round(u).astype(int), np.round(v).astype(int)

def read_intrinsics(sample_path):
    txt_path = os.path.join(sample_path, 'intrinsics.txt')
    with open(txt_path, 'r') as f:
        line = f.readline().strip()
        parts = [float(x) for x in line.split()]
        return parts[0], parts[1], parts[2], parts[3], parts[4]

def draw_panel(rgb, depth, mask, pts_nat, pts_msk, K, out_path, panel_title, nat_label, msk_label):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    H, W = mask.shape
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), facecolor='white')
    
    # 1. RGB
    axes[0].imshow(rgb)
    axes[0].set_title("RGB Image", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # 2. Depth
    axes[1].imshow(depth, cmap='viridis')
    axes[1].set_title("Depth Map", fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # 3. SAM2 Mask
    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title("SAM2 Mask", fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # 4. Native Projection
    axes[3].imshow(rgb)
    axes[3].set_title(nat_label, fontsize=12, fontweight='bold')
    axes[3].axis('off')
    if len(pts_nat) > 0:
        u, v = project_points(pts_nat, fx, fy, cx, cy)
        u_clip = np.clip(u, 0, W - 1)
        v_clip = np.clip(v, 0, H - 1)
        in_mask = mask[v_clip, u_clip]
        precision_nat = np.mean(in_mask)
        
        # 绿色表示落在 mask 内，红色表示在 mask 外面
        axes[3].scatter(u[in_mask], v[in_mask], c='#2ca02c', s=25, edgecolors='black', linewidths=0.5, alpha=0.9, label='In Target')
        axes[3].scatter(u[~in_mask], v[~in_mask], c='#d62728', s=25, edgecolors='black', linewidths=0.5, alpha=0.9, label='Outside')
        
        # 标上精度
        axes[3].text(15, 45, f"Precision = {precision_nat:.3f}", color='white', fontsize=11, fontweight='bold',
                     bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3'))
    else:
        axes[3].text(W/2, H/2, "No grasp candidates", color='#d62728', fontsize=12, fontweight='bold',
                     ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='#d62728'))
        axes[3].text(15, 45, "Precision = 0.000", color='white', fontsize=11, fontweight='bold',
                     bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3'))
        
    # 5. Mask-filtered Projection
    axes[4].imshow(rgb)
    axes[4].set_title(msk_label, fontsize=12, fontweight='bold')
    axes[4].axis('off')
    if len(pts_msk) > 0:
        u, v = project_points(pts_msk, fx, fy, cx, cy)
        u_clip = np.clip(u, 0, W - 1)
        v_clip = np.clip(v, 0, H - 1)
        in_mask = mask[v_clip, u_clip]
        precision_msk = np.mean(in_mask)
        
        axes[4].scatter(u[in_mask], v[in_mask], c='#2ca02c', s=25, edgecolors='black', linewidths=0.5, alpha=0.9)
        axes[4].scatter(u[~in_mask], v[~in_mask], c='#d62728', s=25, edgecolors='black', linewidths=0.5, alpha=0.9)
        
        axes[4].text(15, 45, f"Precision = {precision_msk:.3f}", color='white', fontsize=11, fontweight='bold',
                     bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3'))
    else:
        axes[4].text(W/2, H/2, "No grasp candidates", color='#d62728', fontsize=12, fontweight='bold',
                     ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='#d62728'))
        axes[4].text(15, 45, "Precision = 0.000", color='white', fontsize=11, fontweight='bold',
                     bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3'))
        
    plt.suptitle(panel_title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor='white')
    plt.close()
    print(f"Panel saved successfully to {out_path}")

def main():
    base_dir = "/home/gb/My_respositories/anygrasp_sdk/grasp_detection"
    tfb_dir = os.path.join(base_dir, "tfb_extracted_data")
    
    # ------------------ Panel 1: sample_07 (CGN Improvement) ------------------
    s_idx = 7
    s_path = os.path.join(tfb_dir, f"sample_{s_idx:02d}")
    fx, fy, cx, cy, scale = read_intrinsics(s_path)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    
    rgb = np.array(Image.open(os.path.join(s_path, 'color.png')))
    depth = np.array(Image.open(os.path.join(s_path, 'depth.png'))).astype(np.float32) / scale
    mask = np.load(os.path.join(s_path, "sam2_mask.npy"))
    
    with open(os.path.join(s_path, "result_group_CGN_Native.json"), 'r') as f:
        nat_data = json.load(f)
    pts_nat = np.array(nat_data.get('contact_pts', []))
    
    with open(os.path.join(s_path, "result_group_CGN_Mask.json"), 'r') as f:
        msk_data = json.load(f)
    pts_msk = np.array(msk_data.get('contact_pts', []))
    
    draw_panel(
        rgb, depth, mask, pts_nat, pts_msk, K, 
        out_path=os.path.join(base_dir, "qual_case_sample_07_cgn_improvement.png"),
        panel_title="Case 1: Significant Performance Gain by $Q_d$+SAM2 Constraints on Contact-GraspNet",
        nat_label="CGN-Native (Contact Pt)",
        msk_label="CGN-Mask (Contact Pt)"
    )

    # ------------------ Panel 2: sample_12 (Failure case) ------------------
    s_idx = 12
    s_path = os.path.join(tfb_dir, f"sample_{s_idx:02d}")
    fx, fy, cx, cy, scale = read_intrinsics(s_path)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    
    rgb = np.array(Image.open(os.path.join(s_path, 'color.png')))
    depth = np.array(Image.open(os.path.join(s_path, 'depth.png'))).astype(np.float32) / scale
    mask = np.load(os.path.join(s_path, "sam2_mask.npy"))
    
    with open(os.path.join(s_path, "result_group_CGN_Native.json"), 'r') as f:
        nat_data = json.load(f)
    pts_nat = np.array(nat_data.get('contact_pts', []))
    
    with open(os.path.join(s_path, "result_group_CGN_Mask.json"), 'r') as f:
        msk_data = json.load(f)
    pts_msk = np.array(msk_data.get('contact_pts', []))
    
    draw_panel(
        rgb, depth, mask, pts_nat, pts_msk, K, 
        out_path=os.path.join(base_dir, "qual_case_sample_12_failure.png"),
        panel_title="Case 2: Detection Failure on CGN-Mask due to Extreme Depth Voids Inside Target Mask",
        nat_label="CGN-Native (Contact Pt)",
        msk_label="CGN-Mask (Contact Pt)"
    )

    # ------------------ Panel 3: sample_16 (AnyGrasp Success case) ------------------
    s_idx = 16
    s_path = os.path.join(tfb_dir, f"sample_{s_idx:02d}")
    fx, fy, cx, cy, scale = read_intrinsics(s_path)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    
    rgb = np.array(Image.open(os.path.join(s_path, 'color.png')))
    depth = np.array(Image.open(os.path.join(s_path, 'depth.png'))).astype(np.float32) / scale
    mask = np.load(os.path.join(s_path, "sam2_mask.npy"))
    
    # AnyGrasp Group B
    b_json = os.path.join(base_dir, f"ablation_results_sample_{s_idx:02d}", "result_group_B.json")
    with open(b_json, 'r') as f:
        b_data = json.load(f)
    pts_b = np.array(b_data.get('translations', []))
    
    # AnyGrasp Group D
    d_json = os.path.join(base_dir, f"ablation_results_sample_{s_idx:02d}", "result_group_D.json")
    with open(d_json, 'r') as f:
        d_data = json.load(f)
    pts_d = np.array(d_data.get('translations', []))
    
    draw_panel(
        rgb, depth, mask, pts_b, pts_d, K, 
        out_path=os.path.join(base_dir, "qual_case_sample_16_anygrasp_success.png"),
        panel_title="Case 3: Complete Drift Correction by $Q_d$+SAM2 Constraints on AnyGrasp Model",
        nat_label="AnyGrasp (Group B - Raw)",
        msk_label="AnyGrasp (Group D - Mask)"
    )

    # ------------------ Panel 4: sample_13 (Native Drift case) ------------------
    s_idx = 13
    s_path = os.path.join(tfb_dir, f"sample_{s_idx:02d}")
    fx, fy, cx, cy, scale = read_intrinsics(s_path)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    
    rgb = np.array(Image.open(os.path.join(s_path, 'color.png')))
    depth = np.array(Image.open(os.path.join(s_path, 'depth.png'))).astype(np.float32) / scale
    mask = np.load(os.path.join(s_path, "sam2_mask.npy"))
    
    with open(os.path.join(s_path, "result_group_CGN_Native.json"), 'r') as f:
        nat_data = json.load(f)
    pts_nat = np.array(nat_data.get('contact_pts', []))
    
    with open(os.path.join(s_path, "result_group_CGN_Mask.json"), 'r') as f:
        msk_data = json.load(f)
    pts_msk = np.array(msk_data.get('contact_pts', []))
    
    draw_panel(
        rgb, depth, mask, pts_nat, pts_msk, K, 
        out_path=os.path.join(base_dir, "qual_case_sample_13_native_drift.png"),
        panel_title="Case 4: Complete Background Drift Failure on CGN-Native backbone",
        nat_label="CGN-Native (Contact Pt)",
        msk_label="CGN-Mask (Contact Pt)"
    )

if __name__ == "__main__":
    main()
