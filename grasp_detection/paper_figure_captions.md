# Figure Captions for the Paper

This document contains the academic English captions for all 8 figures in the paper. The descriptions focus strictly on localization and detection precision rather than physical grasp success, and avoid any self-promotional or exaggerated language.

---

### Fig. 1. Quantitative comparison of target mask precision across different grasp detection configurations.
**Caption:**
Fig. 1. Quantitative comparison of target mask precision across different grasp detection configurations. The bar heights represent the mean precision calculated over the experimental dataset, with the error bars denoting the standard deviation ($\pm$ SD). Six configurations are evaluated: (a) AnyGrasp Group B (raw grasp proposals), (b) AnyGrasp Group D (mask-filtered proposals), (c) Contact-GraspNet (CGN) Native evaluating the palm center, (d) CGN Mask evaluating the palm center, (e) CGN Native evaluating the contact point, and (f) CGN Mask evaluating the contact point. Target mask precision measures the proportion of predicted grasp contact points or palm centers that fall within the ground-truth semantic mask of the target object, serving as a proxy for localization accuracy on the target object.

---

### Fig. 2. Paired analysis of localization precision between native and mask-constrained Contact-GraspNet.
**Caption:**
Fig. 2. Paired comparison of localization precision per sample between the native Contact-GraspNet (CGN) and the mask-constrained CGN (CGN Mask) evaluated at the contact point. Each line segment connects the precision of a single scene sample under native CGN (left) and CGN Mask (right). Green lines indicate an increase in target mask precision, red lines indicate a decrease, and gray lines represent no change. A marked shift towards high-precision values (1.00) is observed when the object mask constraint is applied, demonstrating the effectiveness of the proposed filtering in restricting candidate proposals to the target object region.

---

### Fig. 3. Comparison of candidate grasp proposal counts across configurations.
**Caption:**
Fig. 3. Comparison of the number of candidate grasp proposals generated per sample across different configurations. The box plot illustrates the distribution (median, quartiles, and range) of the number of grasp candidates. Configurations include AnyGrasp (Groups B and D) and Contact-GraspNet (Native and Mask, evaluated at the contact point). The introduction of target mask constraints reduces the number of background proposals, leading to a more focused set of candidates directed at the target object.

---

### Fig. 4. Visual and geometric explanation of the evaluation metrics: Palm Center vs. Contact Point.
**Caption:**
Fig. 4. Schematic illustration and projection comparison of the palm center and contact point metrics. (a) Geometric definition: The contact point (green) lies directly on the object surface, whereas the palm center (purple) is offset backwards along the approach vector to accommodate the gripper geometry. (b) 2D projection comparison on a sample image: The projected contact points align closely with the target object's mask. In contrast, the projected palm centers are systematically shifted upwards or outwards due to the depth offset. This geometric projection offset explains why the palm center metric yields lower target mask precision in 2D evaluation, despite the 3D grasp pose being physically aligned with the object.

---

### Fig. 5. Qualitative result of Case 1: Localization accuracy improvement on Contact-GraspNet.
**Caption:**
Fig. 5. Qualitative visualization of Case 1 (Sample 07) demonstrating localization accuracy improvement via target mask filtering. From left to right: input RGB image, depth map, Segment Anything 2 (SAM2) target mask, projected contact points from Native CGN, and projected contact points from CGN Mask. Green scatter points indicate projected contact locations falling within the target mask, while red scatter points represent those falling outside. The introduction of target mask constraints filters out extraneous background detections, improving target mask precision from 0.080 to 1.000.

---

### Fig. 6. Qualitative result of Case 2: Detection failure due to extreme depth voids.
**Caption:**
Fig. 6. Qualitative visualization of Case 2 (Sample 12) illustrating a failure mode under the mask constraint. Under extreme depth voids or missing sensor data within the target mask region, the mask-constrained CGN (CGN Mask) fails to generate any grasp candidates within the target region, resulting in a target mask precision of 0.000. This highlights a limitation of geometry-based grasp detection when sensor depth data is highly incomplete.

---

### Fig. 7. Qualitative result of Case 3: Drift correction on AnyGrasp.
**Caption:**
Fig. 7. Qualitative visualization of Case 3 (Sample 16) showing drift correction on the AnyGrasp model. From left to right: RGB, depth, SAM2 mask, AnyGrasp Group B (raw grasp proposals projected), and AnyGrasp Group D (mask-filtered proposals projected). In Sample 16, CGN-Mask produced no valid grasp candidates after depth-based filtering, whereas AnyGrasp retained a target-consistent prediction. The raw AnyGrasp model (Group B) outputs numerous grasp proposals scattered across the background. The mask-constrained configuration (Group D) filters out background proposals, ensuring all remaining candidates fall within the target object boundary (precision increases from 0.300 to 0.920).

---

### Fig. 8. Qualitative result of Case 4: Background drift failure on native Contact-GraspNet.
**Caption:**
Fig. 8. Qualitative visualization of Case 4 (Sample 13) showing background drift failure on the native Contact-GraspNet. The native CGN backbone predicts contact points entirely on the background table surface rather than the target object. Because all candidates fall outside the target mask, the resulting precision is 0.000 for both Native and Mask configurations (the latter having no valid candidates within the mask), highlighting the difficulty of localizing grasps on target objects when the base model's predictions suffer from severe spatial drift.
