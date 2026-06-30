# Recommended Qualitative Cases for Paper Visualization

We selected four representative cases representing distinct physical behaviors of transparent bag grasping under SOTA detection models, highlighting how our proposed preprocessing pipeline corrects various failure modes.

## Case 1: Best CGN-Mask Performance Improvement
- **Sample ID**: `sample_07`
- **Selection Criteria**: This sample exhibited the highest absolute precision gain for the Contact-GraspNet model after applying the Qd+SAM2 mask module.
- **Native Precision (Contact Point)**: `0.2000`
- **Mask-constrained Precision (Contact Point)**: `1.0000`
- **Absolute Performance Gain**: `+0.8000`
- **Recommended Layers for Visualization**: RGB image / SAM2 target mask / 3D point cloud / CGN-Native grasp predictions (widely scattered) / CGN-Mask grasp predictions (clustered inside the mask).

## Case 2: Successful Rescue of Extreme Depth Voids via Adaptive ROI & 3D Padding
- **Sample ID**: `sample_12`
- **Selection Criteria**: Represents scenes that historically failed to generate any grasp candidates under narrow cropping, but are now successfully rescued by expanding the ROI and applying 3D padding bounds.
- **Native Precision (Contact Point)**: `0.0200` (due to predictions drifting to table edge)
- **Mask-constrained Precision (Contact Point)**: `1.0000`
- **CGN-Mask Grasp Count**: `18` (successfully recovered from 0)
- **Recommended Layers for Visualization**: RGB image / Raw Depth map (exhibiting zero-depth void pixels inside target) / SAM2 mask / Rescued grasp projection showing dense target-aligned proposals.

## Case 3: Representative Success Case for AnyGrasp (Group D) with Blue Basket Removal
- **Sample ID**: `sample_16`
- **Selection Criteria**: This sample shows a significant performance improvement for AnyGrasp. Under Group B, predictions drifted heavily to the surrounding blue bins, while Group D achieved a 1.0000 precision by utilizing HSV blue filtering, depth jump detection, and LCC locking.
- **AnyGrasp Group B Precision**: `0.3800`
- **AnyGrasp Group D Precision**: `1.0000`
- **Recommended Layers for Visualization**: RGB image / SAM2 mask / AnyGrasp Group B translations (partially on blue bins and table) / AnyGrasp Group D translations (neatly inside the target clothing region).

## Case 4: Successful Correction of Native Background Drift
- **Sample ID**: `sample_13`
- **Selection Criteria**: Represents cases where the native model predicted a full set of candidates but all drifted to the background table, which our method successfully corrected by restricting the workspace and cloud density.
- **CGN-Native Grasp Count**: `50`
- **CGN-Native Precision (Contact Point)**: `0.0200`
- **CGN-Mask Grasp Count**: `8`
- **CGN-Mask Precision (Contact Point)**: `1.0000`
- **Recommended Layers for Visualization**: RGB image / 3D point cloud / CGN-Native grasp predictions (Z-distance drift to background table) / CGN-Mask predictions (restricted to the target object surface).
