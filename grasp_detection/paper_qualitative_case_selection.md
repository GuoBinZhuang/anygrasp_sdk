# Recommended Qualitative Cases for Paper Visualization

We selected four representative cases representing distinct physical behaviors of transparent bag grasping under SOTA detection models.

## Case 1: Best CGN-Mask Performance Improvement
- **Sample ID**: `sample_07`
- **Selection Criteria**: This sample exhibited the highest absolute precision gain for the Contact-GraspNet model after applying the Qd+SAM2 mask module.
- **Native Precision (Contact Point)**: `0.0800`
- **Mask-constrained Precision (Contact Point)**: `1.0000`
- **Absolute Performance Gain**: `+0.9200`
- **Recommended Layers for Visualization**: RGB image / SAM2 target mask / 3D point cloud / CGN-Native grasp predictions (widely scattered) / CGN-Mask grasp predictions (clustered inside the mask).

## Case 2: CGN-Mask Failure / Lowest Precision Case
- **Sample ID**: `sample_12`
- **Selection Criteria**: Represents cases where the mask constraints resulted in low precision or zero generated grasps due to severe depth noise inside the bag region.
- **Native Precision (Contact Point)**: `0.0000`
- **Mask-constrained Precision (Contact Point)**: `0.0000`
- **CGN-Mask Grasp Count**: `0`
- **Recommended Layers for Visualization**: RGB image / Raw Depth map (exhibiting zero-depth void pixels inside target) / SAM2 mask / Grasp projection.

## Case 3: Representative Success Case for AnyGrasp (Group D)
- **Sample ID**: `sample_16`
- **Selection Criteria**: This sample showed a significant performance improvement for AnyGrasp, where Group B was poorly aligned due to depth noise, while Group D achieved a 0.9200 precision through hard crop filtering. Note that CGN-Mask produced no valid grasp candidates here due to depth voids.
- **AnyGrasp Group B Precision**: `0.3000`
- **AnyGrasp Group D Precision**: `0.9200`
- **Recommended Layers for Visualization**: RGB image / SAM2 mask / AnyGrasp Group B translations (mostly on bag edges and table) / AnyGrasp Group D translations (neatly inside the target region).

## Case 4: Native Background Drift Failure Case
- **Sample ID**: `sample_13`
- **Selection Criteria**: Represents native models predicting a full set of candidate grasps but having zero or near-zero target-region precision because all grasps drifted to the background wall or table surface.
- **CGN-Native Grasp Count**: `50`
- **CGN-Native Precision (Contact Point)**: `0.0000`
- **Recommended Layers for Visualization**: RGB image / 3D point cloud / CGN-Native grasp predictions (with long Z distances indicating drift to background walls or the ground) / depth_mask showing crop boundaries.
