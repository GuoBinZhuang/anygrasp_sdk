import os
import json
import csv
import numpy as np
import scipy.stats

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
        # fx, fy, cx, cy, scale
        return parts[0], parts[1], parts[2], parts[3], parts[4]

def get_bootstrap_ci(data_before, data_after, confidence=0.95, n_resamples=5000, random_seed=42):
    """
    计算配对差值 (data_after - data_before) 均值的 Bootstrap 95% 置信区间
    """
    rng = np.random.default_rng(random_seed)
    diffs = np.array(data_after) - np.array(data_before)
    n = len(diffs)
    bootstrap_means = []
    for _ in range(n_resamples):
        resampled_diffs = rng.choice(diffs, size=n, replace=True)
        bootstrap_means.append(np.mean(resampled_diffs))
    bootstrap_means = np.sort(bootstrap_means)
    lower_pct = (1.0 - confidence) / 2.0
    upper_pct = 1.0 - lower_pct
    lower_idx = int(lower_pct * n_resamples)
    upper_idx = int(upper_pct * n_resamples)
    mean_diff = np.mean(diffs)
    return mean_diff, bootstrap_means[lower_idx], bootstrap_means[upper_idx]

def run_paired_test(data_before, data_after, label_before, label_after):
    diffs = np.array(data_after) - np.array(data_before)
    mean_before = np.mean(data_before)
    mean_after = np.mean(data_after)
    std_before = np.std(data_before, ddof=1) if len(data_before) > 1 else 0.0
    std_after = np.std(data_after, ddof=1) if len(data_after) > 1 else 0.0
    
    absolute_gain = mean_after - mean_before
    relative_gain = absolute_gain / (mean_before + 1e-8)
    
    # paired t-test
    if np.all(diffs == 0):
        t_p = 1.0
        w_p = 1.0
    else:
        try:
            _, t_p = scipy.stats.ttest_rel(data_after, data_before)
        except Exception:
            t_p = float('nan')
        
        # wilcoxon signed rank test
        try:
            _, w_p = scipy.stats.wilcoxon(data_after, data_before)
        except Exception:
            w_p = float('nan')
            
    # bootstrap 95% CI
    mean_diff, ci_lower, ci_upper = get_bootstrap_ci(data_before, data_after)
    
    return {
        "test_name": f"{label_before} vs {label_after}",
        "mean_before": float(mean_before),
        "mean_after": float(mean_after),
        "absolute_gain": float(absolute_gain),
        "relative_gain": float(relative_gain),
        "std_before": float(std_before),
        "std_after": float(std_after),
        "paired_t_test_p_value": float(t_p),
        "wilcoxon_signed_rank_p_value": float(w_p),
        "bootstrap_95CI_for_mean_difference": [float(ci_lower), float(ci_upper)]
    }

def main():
    base_dir = "/home/gb/My_respositories/anygrasp_sdk/grasp_detection"
    tfb_dir = os.path.join(base_dir, "tfb_extracted_data")
    
    samples = sorted([d for d in os.listdir(tfb_dir) if d.startswith("sample_")])
    print(f"Total samples detected: {len(samples)}")
    
    # 结果容器
    rows = []
    
    # 用于统计学检验的数据数组
    anygrasp_b_precs = []
    anygrasp_d_precs = []
    cgn_native_palm_precs = []
    cgn_mask_palm_precs = []
    cgn_native_contact_precs = []
    cgn_mask_contact_precs = []
    
    for s_name in samples:
        s_idx = int(s_name.split("_")[1])
        s_path = os.path.join(tfb_dir, s_name)
        
        # 1. 加载内参和 sam2_mask
        fx, fy, cx, cy, scale = read_intrinsics(s_path)
        sam2_mask = np.load(os.path.join(s_path, "sam2_mask.npy"))
        H, W = sam2_mask.shape
        
        # 2. AnyGrasp Group B
        b_json = os.path.join(base_dir, f"ablation_results_sample_{s_idx:02d}", "result_group_B.json")
        anygrasp_b_prec = 0.0
        if os.path.exists(b_json):
            with open(b_json, 'r') as f:
                data = json.load(f)
            pts = np.array(data.get('translations', []))
            if len(pts) > 0:
                u, v = project_points(pts, fx, fy, cx, cy)
                u = np.clip(u, 0, W - 1)
                v = np.clip(v, 0, H - 1)
                anygrasp_b_prec = np.mean(sam2_mask[v, u])
        anygrasp_b_precs.append(anygrasp_b_prec)
        
        # 3. AnyGrasp Group D
        d_json = os.path.join(base_dir, f"ablation_results_sample_{s_idx:02d}", "result_group_D.json")
        anygrasp_d_prec = 0.0
        if os.path.exists(d_json):
            with open(d_json, 'r') as f:
                data = json.load(f)
            pts = np.array(data.get('translations', []))
            if len(pts) > 0:
                u, v = project_points(pts, fx, fy, cx, cy)
                u = np.clip(u, 0, W - 1)
                v = np.clip(v, 0, H - 1)
                anygrasp_d_prec = np.mean(sam2_mask[v, u])
        anygrasp_d_precs.append(anygrasp_d_prec)
        
        # 4. CGN Native
        cgn_nat_json = os.path.join(s_path, "result_group_CGN_Native.json")
        cgn_nat_palm_prec = 0.0
        cgn_nat_contact_prec = 0.0
        cgn_nat_grasps = 0
        if os.path.exists(cgn_nat_json):
            with open(cgn_nat_json, 'r') as f:
                data = json.load(f)
            palm_pts = np.array(data.get('translations', []))
            contact_pts = np.array(data.get('contact_pts', []))
            cgn_nat_grasps = len(palm_pts)
            if len(palm_pts) > 0:
                u, v = project_points(palm_pts, fx, fy, cx, cy)
                u = np.clip(u, 0, W - 1)
                v = np.clip(v, 0, H - 1)
                cgn_nat_palm_prec = np.mean(sam2_mask[v, u])
            if len(contact_pts) > 0:
                u, v = project_points(contact_pts, fx, fy, cx, cy)
                u = np.clip(u, 0, W - 1)
                v = np.clip(v, 0, H - 1)
                cgn_nat_contact_prec = np.mean(sam2_mask[v, u])
        cgn_native_palm_precs.append(cgn_nat_palm_prec)
        cgn_native_contact_precs.append(cgn_nat_contact_prec)
        
        # 5. CGN Mask
        cgn_msk_json = os.path.join(s_path, "result_group_CGN_Mask.json")
        cgn_msk_palm_prec = 0.0
        cgn_msk_contact_prec = 0.0
        cgn_msk_grasps = 0
        if os.path.exists(cgn_msk_json):
            with open(cgn_msk_json, 'r') as f:
                data = json.load(f)
            palm_pts = np.array(data.get('translations', []))
            contact_pts = np.array(data.get('contact_pts', []))
            cgn_msk_grasps = len(palm_pts)
            if len(palm_pts) > 0:
                u, v = project_points(palm_pts, fx, fy, cx, cy)
                u = np.clip(u, 0, W - 1)
                v = np.clip(v, 0, H - 1)
                cgn_msk_palm_prec = np.mean(sam2_mask[v, u])
            if len(contact_pts) > 0:
                u, v = project_points(contact_pts, fx, fy, cx, cy)
                u = np.clip(u, 0, W - 1)
                v = np.clip(v, 0, H - 1)
                cgn_msk_contact_prec = np.mean(sam2_mask[v, u])
        cgn_mask_palm_precs.append(cgn_msk_palm_prec)
        cgn_mask_contact_precs.append(cgn_msk_contact_prec)
        
        # 6. Gain
        abs_gain = cgn_msk_contact_prec - cgn_nat_contact_prec
        rel_gain = abs_gain / (cgn_nat_contact_prec + 1e-8)
        
        rows.append({
            "sample_id": s_idx,
            "AnyGrasp_Group_B_precision": anygrasp_b_prec,
            "AnyGrasp_Group_D_precision": anygrasp_d_prec,
            "CGN_Native_palm_precision": cgn_nat_palm_prec,
            "CGN_Mask_palm_precision": cgn_msk_palm_prec,
            "CGN_Native_contact_precision": cgn_nat_contact_prec,
            "CGN_Mask_contact_precision": cgn_msk_contact_prec,
            "CGN_Native_num_grasps": cgn_nat_grasps,
            "CGN_Mask_num_grasps": cgn_msk_grasps,
            "CGN_contact_absolute_gain": abs_gain,
            "CGN_contact_relative_gain": rel_gain
        })
        print(f"Sample {s_idx:02d} processed successfully.")

    # ------------------ Task 1: 写入 CSV ------------------
    csv_fields = ["sample_id", "AnyGrasp_Group_B_precision", "AnyGrasp_Group_D_precision",
                  "CGN_Native_palm_precision", "CGN_Mask_palm_precision",
                  "CGN_Native_contact_precision", "CGN_Mask_contact_precision",
                  "CGN_Native_num_grasps", "CGN_Mask_num_grasps",
                  "CGN_contact_absolute_gain", "CGN_contact_relative_gain"]
    
    csv_path = os.path.join(base_dir, "paper_exp_per_sample_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"CSV file exported to {csv_path}")

    # ------------------ Task 1: 写入 MD 表格 ------------------
    md_path = os.path.join(base_dir, "paper_exp_per_sample_results.md")
    with open(md_path, 'w') as f:
        f.write("# Per-Sample Experimental Results\n\n")
        f.write("| Sample ID | AnyGrasp B Prec | AnyGrasp D Prec | CGN Nat Palm | CGN Mask Palm | CGN Nat Contact | CGN Mask Contact | CGN Nat Grasps | CGN Mask Grasps | Absolute Gain (Contact) | Relative Gain (Contact) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in rows:
            f.write(f"| {r['sample_id']:02d} | {r['AnyGrasp_Group_B_precision']:.4f} | {r['AnyGrasp_Group_D_precision']:.4f} | {r['CGN_Native_palm_precision']:.4f} | {r['CGN_Mask_palm_precision']:.4f} | {r['CGN_Native_contact_precision']:.4f} | {r['CGN_Mask_contact_precision']:.4f} | {r['CGN_Native_num_grasps']} | {r['CGN_Mask_num_grasps']} | {r['CGN_contact_absolute_gain']:.4f} | {r['CGN_contact_relative_gain']:.4f} |\n")
    print(f"Markdown table exported to {md_path}")

    # ------------------ Task 2: 统计学检验 ------------------
    tests_summary = []
    
    # 检验 1: AnyGrasp Group B vs Group D
    tests_summary.append(run_paired_test(anygrasp_b_precs, anygrasp_d_precs, "AnyGrasp Group B", "AnyGrasp Group D"))
    # 检验 2: CGN Native Contact vs CGN Mask Contact
    tests_summary.append(run_paired_test(cgn_native_contact_precs, cgn_mask_contact_precs, "CGN Native Contact Point", "CGN Mask Contact Point"))
    # 检验 3: CGN Native Palm vs CGN Mask Palm
    tests_summary.append(run_paired_test(cgn_native_palm_precs, cgn_mask_palm_precs, "CGN Native Palm Center", "CGN Mask Palm Center"))
    
    # 写入 JSON
    json_path = os.path.join(base_dir, "paper_exp_statistical_tests.json")
    with open(json_path, 'w') as f:
        json.dump(tests_summary, f, indent=2)
    print(f"Statistical tests JSON exported to {json_path}")
    
    # 写入 MD
    md_test_path = os.path.join(base_dir, "paper_exp_statistical_tests.md")
    with open(md_test_path, 'w') as f:
        f.write("# Statistical Significance Tests (Paired Analysis)\n\n")
        f.write("> [!NOTE]\n")
        f.write("> If any group possesses zero variance (e.g. AnyGrasp Group D having constant 1.0 precision across samples), it is handled robustly in the Wilcoxon and paired t-test procedures.\n\n")
        f.write("| Test Pair | Mean (Before) | Mean (After) | Abs Gain | Rel Gain | Std (Before) | Std (After) | Paired T-Test p-value | Wilcoxon p-value | Bootstrap 95% CI (Mean Diff) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for t in tests_summary:
            ci_str = f"[{t['bootstrap_95CI_for_mean_difference'][0]:.4f}, {t['bootstrap_95CI_for_mean_difference'][1]:.4f}]"
            f.write(f"| {t['test_name']} | {t['mean_before']:.4f} | {t['mean_after']:.4f} | {t['absolute_gain']:.4f} | {t['relative_gain']:.4f} | {t['std_before']:.4f} | {t['std_after']:.4f} | {t['paired_t_test_p_value']:.4e} | {t['wilcoxon_signed_rank_p_value']:.4e} | {ci_str} |\n")
    print(f"Statistical tests Markdown table exported to {md_test_path}")

if __name__ == "__main__":
    main()
