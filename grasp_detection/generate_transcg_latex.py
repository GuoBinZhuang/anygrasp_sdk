import os
import csv
import numpy as np

def main():
    base_dir = "/home/gb/My_respositories/anygrasp_sdk/grasp_detection"
    csv_path = os.path.join(base_dir, "transcg_multi_scene_per_scene.csv")
    
    if not os.path.exists(csv_path):
        print("Error: CSV file not found!")
        return

    scenes_data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            scenes_data.append({
                "scene": int(row["scene"]),
                "n_views": int(row["n_views"]),
                "native_prec_mean": float(row["native_prec_mean"]),
                "native_prec_std": float(row["native_prec_std"]),
                "native_cnt_mean": float(row["native_cnt_mean"]),
                "proposed_prec_mean": float(row["proposed_prec_mean"]),
                "proposed_prec_std": float(row["proposed_prec_std"]),
                "proposed_cnt_mean": float(row["proposed_cnt_mean"])
            })

    # Sort by scene ID
    scenes_data = sorted(scenes_data, key=lambda x: x["scene"])

    # 1. Output Overview Latex Table (Table 3)
    tex_overview_path = os.path.join(base_dir, "table_transcg_overview.tex")
    
    # Calculate global stats (weighted by views or unweighted scene average)
    total_views = sum(s["n_views"] for s in scenes_data)
    
    # Scene-level averages (unweighted)
    nat_precs = [s["native_prec_mean"] for s in scenes_data]
    prop_precs = [s["proposed_prec_mean"] for s in scenes_data]
    nat_cnts = [s["native_cnt_mean"] for s in scenes_data]
    prop_cnts = [s["proposed_cnt_mean"] for s in scenes_data]
    
    mean_nat_prec = np.mean(nat_precs)
    std_nat_prec = np.std(nat_precs, ddof=1)
    mean_prop_prec = np.mean(prop_precs)
    std_prop_prec = np.std(prop_precs, ddof=1)
    
    mean_nat_cnt = np.mean(nat_cnts)
    std_nat_cnt = np.std(nat_cnts, ddof=1)
    mean_prop_cnt = np.mean(prop_cnts)
    std_prop_cnt = np.std(prop_cnts, ddof=1)

    with open(tex_overview_path, 'w') as f:
        f.write("% LaTeX Table generated automatically for TransCG overview results (Scene-level N=25)\n")
        f.write("\\begin{tabular}{llcc}\n")
        f.write("\\hline\n")
        f.write("Method & Preprocessing Pipeline & 3D Alignment Precision & Average Candidates \\\\\n")
        f.write("\\hline\n")
        f.write("Baseline & CGN-Mask on Raw Depth & {:.4f} $\\pm$ {:.4f} & {:.1f} $\\pm$ {:.1f} \\\\\n".format(
            mean_nat_prec, std_nat_prec, mean_nat_cnt, std_nat_cnt))
        f.write("Proposed & $Q_d$ + Depth Jump + LCC & \\textbf{{{:.4f} $\\pm$ {:.4f}}} & \\textbf{{{:.1f} $\\pm$ {:.1f}}} \\\\\n".format(
            mean_prop_prec, std_prop_prec, mean_prop_cnt, std_prop_cnt))
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
    print(f"LaTeX overview table saved to {tex_overview_path}")

    # 2. Output Detailed Per-Scene Latex Table (for Appendix)
    tex_detail_path = os.path.join(base_dir, "table_transcg_detailed_appendix.tex")
    with open(tex_detail_path, 'w') as f:
        f.write("% LaTeX Table generated automatically for detailed TransCG per-scene results\n")
        f.write("\\begin{tabular}{cccccc}\n")
        f.write("\\hline\n")
        f.write("Scene ID & Views & Baseline Precision & Proposed Precision & Baseline Candidates & Proposed Candidates \\\\\n")
        f.write("\\hline\n")
        for s in scenes_data:
            # Highlight proposed if it is higher than native
            if s["proposed_prec_mean"] > s["native_prec_mean"]:
                prop_prec_str = "\\textbf{{{:.4f}}}".format(s["proposed_prec_mean"])
            else:
                prop_prec_str = "{:.4f}".format(s["proposed_prec_mean"])
                
            if s["proposed_cnt_mean"] > s["native_cnt_mean"]:
                prop_cnt_str = "\\textbf{{{:.1f}}}".format(s["proposed_cnt_mean"])
            else:
                prop_cnt_str = "{:.1f}".format(s["proposed_cnt_mean"])

            f.write("Scene {:02d} & {} & {:.4f} & {} & {:.1f} & {} \\\\\n".format(
                s["scene"], s["n_views"], s["native_prec_mean"], prop_prec_str, s["native_cnt_mean"], prop_cnt_str))
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
    print(f"LaTeX detailed appendix table saved to {tex_detail_path}")

if __name__ == "__main__":
    main()
