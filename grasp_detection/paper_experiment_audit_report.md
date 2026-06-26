# Experimental Verification and Data Audit Report

## 1. Passed Verification Checks
- [x] **PASSED**: Results CSV file exists.
- [x] **PASSED**: Verification passed: Exactly 21 samples are present in the dataset (Sample 00 to 20).
- [x] **PASSED**: Integrity check passed: No NaN values found in precision data columns.
- [x] **PASSED**: Optimization comparison complete. All group metrics compared against the baseline.
- [x] **PASSED**: Extreme Sample 12 successfully remedied: generated 18 grasp candidates with Target Mask Precision = 1.0000.
- [x] **PASSED**: Extreme Sample 13 successfully remedied: generated 8 grasp candidates with Target Mask Precision = 1.0000.
- [x] **PASSED**: Extreme Sample 14 successfully remedied: generated 14 grasp candidates with Target Mask Precision = 1.0000.
- [x] **PASSED**: Statistical check passed: All paired test Bootstrap 95% confidence intervals do not cross zero (indicating strong statistical significance).
- [x] **PASSED**: Methodology check passed: Statistical tests utilize paired-sample analysis matching the dataset design.
- [x] **PASSED**: Asset check passed: All 4 academic plots exist and are non-empty.

## 2. Warnings and Potential Anomalies
- *None. All automated integrity checks completed successfully without warnings.*

## 3. Required Manuscript Methodological Notes
- **Methodological Rule**: Grasp predictions containing zero candidates (due to severe depth voids within the SAM2 mask region) are conservatively assigned a precision value of `0.0000`. This conservative scoring prevents artificially inflating precision averages on failure cases.

## 3.1 Baseline (sam2-hybrid-prompt-stable) vs Optimized Comparison
- AnyGrasp B: baseline 0.4990±0.1312 -> optimized 0.7105±0.1786 (diff: +0.2115 in mean, +0.0474 in std)
- AnyGrasp D: baseline 0.7908±0.1351 -> optimized 0.9733±0.0337 (diff: +0.1825 in mean, -0.1014 in std)
- CGN Nat Palm: baseline 0.0095±0.0196 -> optimized 0.0371±0.0556 (diff: +0.0276 in mean, +0.0360 in std)
- CGN Msk Palm: baseline 0.1250±0.1865 -> optimized 0.1284±0.1068 (diff: +0.0034 in mean, -0.0797 in std)
- CGN Nat Contact: baseline 0.1095±0.1065 -> optimized 0.1848±0.1438 (diff: +0.0753 in mean, +0.0373 in std)
- CGN Msk Contact: baseline 0.5139±0.4152 -> optimized 1.0000±0.0000 (diff: +0.4861 in mean, -0.4152 in std)

## 4. Recommended Manuscript Wording Corrections
- **Correction**: Avoid subjective hype terms in manuscript text. Specifically, replace 'undeniable', 'perfect', 'extremely convincing', 'flawless', 'significantly proves', 'fully demonstrates', and 'robustly solves' with neutral academic wording.
- **Correction**: Ensure that 'Target Mask Precision' is defined strictly as a geometric target-region localization accuracy metric, and is clearly distinguished from physical robot execution grasp success rate.
