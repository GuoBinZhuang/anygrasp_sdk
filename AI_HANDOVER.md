# AnyGrasp SDK & CGN Preprocessing Pipeline - AI Handover Documentation

This document serves as an immediate context loader for any AI agent resuming development of the target-region preprocessing pipeline for transparent bag/garment grasping under the AnyGrasp and Contact-GraspNet (CGN) backbones.

---

## 1. Project Mission & Technical Background

### The Problem
Traditional 3D grasp detection models (e.g., AnyGrasp, Contact-GraspNet) suffer from severe failures when manipulating **transparent plastic bags** or **highly specular target garments** inside storage bins:
1. **Edge/Mask Leakage:** 2D semantic masks (such as SAM) easily leak into surrounding container walls (like blue plastic bins) due to reflections.
2. **Depth Measurement Voids:** Infrared reflections fail on transparent/specular surfaces, causing extreme zero-depth voids and boundary noise.
3. **Spatial Drift:** Grasp candidates often drift to background tables or container edges due to sparse surface geometry.

### Our Solution (The Proposed Preprocessing Pipeline)
We introduce a multimodal preprocessing pipeline combining color-depth spatial cues:
* **Multimodal HSV-RGB Blue Filter:** Isolates blue bin workspaces to prevent semantic leakage.
* **Local Depth Quality Map ($Q_d$):** Estimates pixel-level depth standard deviation to downweight noisy or invalid depth zones.
* **3D Depth Gradient Jump Detector:** Sobel-filtered boundaries defining strict physical barriers.
* **Largest Connected Component (LCC) Extraction:** Locks onto the core object body and rejects scattered outliers.
* **Adaptive SAM2 Closed-Loop Negative Prompting:** Centroid calculation on false positive leakage areas, automatically feeding back negative points to contract SAM boundaries.
* **3D Boundary Padding:** Restricts workspace point cloud density with a safety padding buffer ($\delta = 4\,\text{cm}$) to prevent edge cropping.

---

## 2. Directory Structure & Key Files

The workspace is organized as follows:
* `grasp_detection/`
  * `evaluate_transcg_multi_scene.py` [NEW/TRACKED]: The central evaluation script. Supports multi-scene batch evaluations over the TransCG test dataset, reporting native vs. proposed configurations.
  * `generate_transcg_latex.py` [NEW/TRACKED]: Pulls output statistics and generates LaTeX tables formatted for paper submissions.
  * `paper_methodology_section_draft.md` [NEW/TRACKED]: Mathematical formulations and flowchart descriptions of the preprocessing pipeline.
  * `paper_qualitative_case_selection.md` [MODIFIED]: Detailed records of selected cases for paper qualitative visualization (Sample 07, 12, 16, 13).
  * `paper_figure_captions.md` [MODIFIED]: LaTeX figure captions corresponding to the qualitative results.
  * `paper_experiment_section_draft_v2.md` [MODIFIED]: Experimental results write-up.
  * Generated outputs (untracked by Git but saved locally):
    * `transcg_multi_scene_per_scene.csv` & `transcg_multi_scene_per_view.csv`: Raw statistical outputs.
    * `transcg_multi_scene_summary.json`: Summary JSON of the evaluation run.
    * `table_transcg_overview.tex` & `table_transcg_detailed_appendix.tex`: Preformatted LaTeX code for tables.

---

## 3. Current Progress Snapshot

1. **Batch Evaluation Framework Completed:**
   * `evaluate_transcg_multi_scene.py` is capable of parsing multiple scenes in parallel, managing baseline/proposed modes, executing NMS (Non-Maximum Suppression) on proposals, and computing contact-point level precision metrics.
2. **Qualitative Case Studies Refined:**
   * Four typical failure-to-success scenarios have been mapped and verified to reach **1.0000 Precision** under our proposed configuration:
     * **Case 1 (Sample 07):** High absolute gain for Contact-GraspNet (Native `0.2000` $\to$ Proposed `1.0000`).
     * **Case 2 (Sample 12):** Rescue of zero candidates under extreme depth voids using 3D padding (Native `0.0200` $\to$ Proposed `1.0000`, 18 valid grasp proposals recovered).
     * **Case 3 (Sample 16):** AnyGrasp blue bin workspace exclusion (Native `0.3800` $\to$ Proposed `1.0000`).
     * **Case 4 (Sample 13):** Eliminating background table drift for CGN (Native `0.0200` $\to$ Proposed `1.0000`).

---

## 4. Environment & Execution Setup (For the next device)

### Path Dependencies
* **Contact-GraspNet Pytorch Repo:** Currently cloned at `/home/gb/My_respositories/contact_graspnet_pytorch`. The scripts append this path to `sys.path`. If cloned elsewhere on the new machine, update these lines:
  ```python
  sys.path.append('<YOUR_PATH_TO>/contact_graspnet_pytorch')
  sys.path.append('<YOUR_PATH_TO>/contact_graspnet_pytorch/contact_graspnet_pytorch')
  ```

### Datasets
* TransCG test dataset folder structure should be symlinked or placed under `grasp_detection/transcg_data` (ignored in `.gitignore`).

### Run Commands
1. Run multi-scene evaluation:
   ```bash
   python evaluate_transcg_multi_scene.py --scenes 1,5,7,8,11,12 --max-views 50
   ```
2. Generate LaTeX tables:
   ```bash
   python generate_transcg_latex.py
   ```

---

## 5. Next Steps for AI Agent on the New Device

1. **Path Alignment:** Update the hardcoded path references inside `evaluate_transcg_multi_scene.py` to match the directory structure of the new machine.
2. **Generate Qualitative Visualizations:**
   * Use the IDs listed in `paper_qualitative_case_selection.md` (`sample_07`, `sample_12`, `sample_16`, `sample_13`) to run visual rendering scripts.
   * Generate side-by-side comparison images of: RGB / SAM2 / Depth Void / Native Grasps / Mask-Constrained Grasps.
3. **Verify Table Integration:** Copy the output code from `table_transcg_overview.tex` and `table_transcg_detailed_appendix.tex` into the main Overleaf project or LaTeX document.
4. **Draft Revision:** Review and proofread the experimental findings in `paper_experiment_section_draft_v2.md` to ensure consistency with the generated CSV data.
