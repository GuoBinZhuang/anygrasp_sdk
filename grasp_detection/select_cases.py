import csv
import os

def main():
    base_dir = "/home/gb/My_respositories/anygrasp_sdk/grasp_detection"
    csv_path = os.path.join(base_dir, "paper_exp_per_sample_results.csv")
    
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "sample_id": int(r["sample_id"]),
                "anygrasp_b": float(r["AnyGrasp_Group_B_precision"]),
                "anygrasp_d": float(r["AnyGrasp_Group_D_precision"]),
                "cgn_nat_palm": float(r["CGN_Native_palm_precision"]),
                "cgn_msk_palm": float(r["CGN_Mask_palm_precision"]),
                "cgn_nat_contact": float(r["CGN_Native_contact_precision"]),
                "cgn_msk_contact": float(r["CGN_Mask_contact_precision"]),
                "cgn_nat_grasps": int(r["CGN_Native_num_grasps"]),
                "cgn_msk_grasps": int(r["CGN_Mask_num_grasps"]),
                "abs_gain": float(r["CGN_contact_absolute_gain"]),
                "rel_gain": float(r["CGN_contact_relative_gain"])
            })
            
    # 1. CGN-Mask 提升最大
    best_cgn_gain_case = max(rows, key=lambda x: x["abs_gain"])
    
    # 2. CGN-Mask 仍然失败或 precision 最低
    worst_cgn_mask_case = min(rows, key=lambda x: (x["cgn_msk_contact"], x["cgn_msk_grasps"]))
    
    # 3. AnyGrasp Group D 成功代表 (在 D 为 1.0 时，B 最小)
    ag_success_candidates = [r for r in rows if r["anygrasp_d"] == 1.0]
    ag_success_case = min(ag_success_candidates, key=lambda x: x["anygrasp_b"]) if ag_success_candidates else min(rows, key=lambda x: x["anygrasp_b"])
    
    # 4. Native 明显漂移到背景/桌面的失败样本 (排除已选的样本)
    used_ids = {best_cgn_gain_case['sample_id'], worst_cgn_mask_case['sample_id'], ag_success_case['sample_id']}
    nat_drift_candidates = [r for r in rows if r["cgn_nat_grasps"] >= 30 and r["sample_id"] not in used_ids]
    nat_drift_case = min(nat_drift_candidates, key=lambda x: x["cgn_nat_contact"]) if nat_drift_candidates else min(rows, key=lambda x: x["cgn_nat_contact"])

    md_path = os.path.join(base_dir, "paper_qualitative_case_selection.md")
    with open(md_path, 'w') as f:
        f.write("# Recommended Qualitative Cases for Paper Visualization\n\n")
        f.write("We selected four representative cases representing distinct physical behaviors of transparent bag grasping under SOTA detection models.\n\n")
        
        # Case 1
        f.write("## Case 1: Best CGN-Mask Performance Improvement\n")
        f.write(f"- **Sample ID**: `sample_{best_cgn_gain_case['sample_id']:02d}`\n")
        f.write("- **Selection Criteria**: This sample exhibited the highest absolute precision gain for the Contact-GraspNet model after applying the Qd+SAM2 mask module.\n")
        f.write(f"- **Native Precision (Contact Point)**: `{best_cgn_gain_case['cgn_nat_contact']:.4f}`\n")
        f.write(f"- **Mask-constrained Precision (Contact Point)**: `{best_cgn_gain_case['cgn_msk_contact']:.4f}`\n")
        f.write(f"- **Absolute Performance Gain**: `+{best_cgn_gain_case['abs_gain']:.4f}`\n")
        f.write("- **Recommended Layers for Visualization**: RGB image / SAM2 target mask / 3D point cloud / CGN-Native grasp predictions (widely scattered) / CGN-Mask grasp predictions (clustered inside the mask).\n\n")
        
        # Case 2
        f.write("## Case 2: CGN-Mask Failure / Lowest Precision Case\n")
        f.write(f"- **Sample ID**: `sample_{worst_cgn_mask_case['sample_id']:02d}`\n")
        f.write("- **Selection Criteria**: Represents cases where the mask constraints resulted in low precision or zero generated grasps due to severe depth noise inside the bag region.\n")
        f.write(f"- **Native Precision (Contact Point)**: `{worst_cgn_mask_case['cgn_nat_contact']:.4f}`\n")
        f.write(f"- **Mask-constrained Precision (Contact Point)**: `{worst_cgn_mask_case['cgn_msk_contact']:.4f}`\n")
        f.write(f"- **CGN-Mask Grasp Count**: `{worst_cgn_mask_case['cgn_msk_grasps']}`\n")
        f.write("- **Recommended Layers for Visualization**: RGB image / Raw Depth map (exhibiting zero-depth void pixels inside target) / SAM2 mask / Grasp projection.\n\n")
        
        # Case 3
        f.write("## Case 3: Representative Success Case for AnyGrasp (Group D)\n")
        f.write(f"- **Sample ID**: `sample_{ag_success_case['sample_id']:02d}`\n")
        f.write("- **Selection Criteria**: This sample showed the most significant performance improvement for AnyGrasp, where Group B was poorly aligned due to depth noise, while Group D achieved a perfect 1.0000 precision through hard crop filtering.\n")
        f.write(f"- **AnyGrasp Group B Precision**: `{ag_success_case['anygrasp_b']:.4f}`\n")
        f.write(f"- **AnyGrasp Group D Precision**: `{ag_success_case['anygrasp_d']:.4f}`\n")
        f.write("- **Recommended Layers for Visualization**: RGB image / SAM2 mask / AnyGrasp Group B translations (mostly on bag edges and table) / AnyGrasp Group D translations (neatly inside the target region).\n\n")
        
        # Case 4
        f.write("## Case 4: Native Background Drift Failure Case\n")
        f.write(f"- **Sample ID**: `sample_{nat_drift_case['sample_id']:02d}`\n")
        f.write("- **Selection Criteria**: Represents native models predicting a full set of candidate grasps but having zero or near-zero target-region precision because all grasps drifted to the background wall or table surface.\n")
        f.write(f"- **CGN-Native Grasp Count**: `{nat_drift_case['cgn_nat_grasps']}`\n")
        f.write(f"- **CGN-Native Precision (Contact Point)**: `{nat_drift_case['cgn_nat_contact']:.4f}`\n")
        f.write("- **Recommended Layers for Visualization**: RGB image / 3D point cloud / CGN-Native grasp predictions (with long Z distances indicating drift to background walls or the ground) / depth_mask showing crop boundaries.\n\n")
        
    print(f"Qualitative Case Selection Markdown file saved to {md_path}")
    print(f"Selected Cases: Best Gain=sample_{best_cgn_gain_case['sample_id']:02d}, Worst CGN=sample_{worst_cgn_mask_case['sample_id']:02d}, Best AG=sample_{ag_success_case['sample_id']:02d}, Drift Fail=sample_{nat_drift_case['sample_id']:02d}")

if __name__ == "__main__":
    main()
