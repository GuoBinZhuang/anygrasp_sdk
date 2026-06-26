import os
import json
import numpy as np

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
        return parts[0], parts[1], parts[2], parts[3] # fx, fy, cx, cy

def main():
    data_dir = "/home/gb/My_respositories/anygrasp_sdk/grasp_detection/tfb_extracted_data"
    samples = sorted([d for d in os.listdir(data_dir) if d.startswith("sample_")])
    
    # 统计项
    results = {
        "CGN_Native": {"palm_precs": [], "contact_precs": [], "counts": []},
        "CGN_Mask": {"palm_precs": [], "contact_precs": [], "counts": []}
    }
    
    for sample in samples:
        sample_path = os.path.join(data_dir, sample)
        sam2_mask_path = os.path.join(sample_path, "sam2_mask.npy")
        if not os.path.exists(sam2_mask_path):
            continue
            
        fx, fy, cx, cy = read_intrinsics(sample_path)
        sam2_mask = np.load(sam2_mask_path)
        H, W = sam2_mask.shape
        
        for group in ["CGN_Native", "CGN_Mask"]:
            json_path = os.path.join(sample_path, f"result_group_{group}.json")
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # 1. 评估手掌中心 (Palm Center)
                palm_pts = np.array(data.get('translations', []))
                results[group]["counts"].append(len(palm_pts))
                if len(palm_pts) > 0:
                    u, v = project_points(palm_pts, fx, fy, cx, cy)
                    u = np.clip(u, 0, W - 1)
                    v = np.clip(v, 0, H - 1)
                    prec = np.mean(sam2_mask[v, u])
                    results[group]["palm_precs"].append(prec)
                else:
                    results[group]["palm_precs"].append(0.0)
                    
                # 2. 评估接触点 (Contact Point)
                contact_pts = np.array(data.get('contact_pts', []))
                if len(contact_pts) > 0:
                    u, v = project_points(contact_pts, fx, fy, cx, cy)
                    u = np.clip(u, 0, W - 1)
                    v = np.clip(v, 0, H - 1)
                    prec = np.mean(sam2_mask[v, u])
                    results[group]["contact_precs"].append(prec)
                else:
                    results[group]["contact_precs"].append(0.0)
            else:
                results[group]["counts"].append(0)
                results[group]["palm_precs"].append(0.0)
                results[group]["contact_precs"].append(0.0)

    # 统计数据汇总
    summary = {}
    for group in ["CGN_Native", "CGN_Mask"]:
        summary[group] = {
            "mean_grasps": float(np.mean(results[group]["counts"])),
            "mean_palm_precision": float(np.mean(results[group]["palm_precs"])),
            "std_palm_precision": float(np.std(results[group]["palm_precs"])),
            "mean_contact_precision": float(np.mean(results[group]["contact_precs"])),
            "std_contact_precision": float(np.std(results[group]["contact_precs"]))
        }

    print("\n" + "="*95)
    print("Contact-GraspNet 跨架构对比实验结果汇总 (Mean ± Std)")
    print("="*95)
    print(f"CGN-Native (原生整场景输入):")
    print(f"  - 手掌中心 Precision (Palm Center):   {summary['CGN_Native']['mean_palm_precision']:.4f} ± {summary['CGN_Native']['std_palm_precision']:.4f}")
    print(f"  - 接触点   Precision (Contact Point): {summary['CGN_Native']['mean_contact_precision']:.4f} ± {summary['CGN_Native']['std_contact_precision']:.4f}")
    print(f"  - 平均生成候选抓取数:                {summary['CGN_Native']['mean_grasps']:.1f}")
    print("-" * 95)
    print(f"CGN-Mask (Qd+SAM2 局部裁剪与过滤):")
    print(f"  - 手掌中心 Precision (Palm Center):   {summary['CGN_Mask']['mean_palm_precision']:.4f} ± {summary['CGN_Mask']['std_palm_precision']:.4f}")
    print(f"  - 接触点   Precision (Contact Point): {summary['CGN_Mask']['mean_contact_precision']:.4f} ± {summary['CGN_Mask']['std_contact_precision']:.4f}")
    print(f"  - 平均生成候选抓取数:                {summary['CGN_Mask']['mean_grasps']:.1f}")
    print("="*95)
    
    # 保存结果为 JSON
    out_json_path = os.path.join(data_dir, "../ablation_results/cgn_experiments_summary.json")
    with open(out_json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ 实验汇总 JSON 报告已保存至: {out_json_path}")

if __name__ == "__main__":
    main()
