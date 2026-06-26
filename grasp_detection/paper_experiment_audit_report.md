# Experimental Verification and Data Audit Report

## 1. Passed Verification Checks
- [x] **PASSED**: Results CSV file exists.
- [x] **PASSED**: Verification passed: Exactly 21 samples are present in the dataset (Sample 00 to 20).
- [x] **PASSED**: Integrity check passed: No NaN values found in precision data columns.
- [x] **PASSED**: Sample 12 verified: CGN-Mask grasp count is 0.
- [x] **PASSED**: Sample 12 verified: Empty prediction was scored as 0.0000 precision.
- [x] **PASSED**: Sample 13 verified: CGN-Mask grasp count is 0.
- [x] **PASSED**: Sample 13 verified: Empty prediction was scored as 0.0000 precision.
- [x] **PASSED**: Sample 14 verified: CGN-Mask grasp count is 0.
- [x] **PASSED**: Sample 14 verified: Empty prediction was scored as 0.0000 precision.
- [x] **PASSED**: Statistical check passed: All paired test Bootstrap 95% confidence intervals do not cross zero (indicating strong statistical significance).
- [x] **PASSED**: Methodology check passed: Statistical tests utilize paired-sample analysis matching the dataset design.
- [x] **PASSED**: Asset check passed: All 4 academic plots exist and are non-empty.

## 2. Warnings and Potential Anomalies
- [ ] **WARNING**: Inconsistency in AnyGrasp B: calculated 0.4295±0.1159, expected 0.5886±0.2798
- [ ] **WARNING**: Inconsistency in AnyGrasp D: calculated 0.6125±0.2755, expected 1.0000±0.0000
- [ ] **WARNING**: Inconsistency in CGN Nat Palm: calculated 0.0057±0.0157, expected 0.0324±0.0538
- [ ] **WARNING**: Inconsistency in CGN Msk Palm: calculated 0.0894±0.1662, expected 0.1747±0.2211
- [ ] **WARNING**: Inconsistency in CGN Nat Contact: calculated 0.0990±0.0971, expected 0.1905±0.1592
- [ ] **WARNING**: Inconsistency in CGN Msk Contact: calculated 0.4785±0.4211, expected 0.8571±0.3586

## 3. Required Manuscript Methodological Notes
- **Note**: Sample 12 empty prediction was conservatively scored as zero precision (methodological note).
- **Note**: Sample 13 empty prediction was conservatively scored as zero precision (methodological note).
- **Note**: Sample 14 empty prediction was conservatively scored as zero precision (methodological note).
- **Methodological Rule**: Grasp predictions containing zero candidates (due to severe depth voids within the SAM2 mask region) are conservatively assigned a precision value of `0.0000`. This conservative scoring prevents artificially inflating precision averages on failure cases.

## 4. Recommended Manuscript Wording Corrections
- **Correction**: Avoid subjective hype terms in manuscript text. Specifically, replace 'undeniable', 'perfect', 'extremely convincing', 'flawless', 'significantly proves', 'fully demonstrates', and 'robustly solves' with neutral academic wording.
- **Correction**: Ensure that 'Target Mask Precision' is defined strictly as a geometric target-region localization accuracy metric, and is clearly distinguished from physical robot execution grasp success rate.
