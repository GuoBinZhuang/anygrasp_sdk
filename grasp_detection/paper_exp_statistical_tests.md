# Statistical Significance Tests (Paired Analysis)

> [!NOTE]
> If any group possesses zero variance (e.g. AnyGrasp Group D having constant 1.0 precision across samples), it is handled robustly in the Wilcoxon and paired t-test procedures.

| Test Pair | Mean (Before) | Mean (After) | Abs Gain | Rel Gain | Std (Before) | Std (After) | Paired T-Test p-value | Wilcoxon p-value | Bootstrap 95% CI (Mean Diff) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| AnyGrasp Group B vs AnyGrasp Group D | 0.7105 | 0.9733 | 0.2629 | 0.3700 | 0.1786 | 0.0337 | 9.3808e-07 | 5.9340e-05 | [0.1914, 0.3390] |
| CGN Native Contact Point vs CGN Mask Contact Point | 0.1848 | 1.0000 | 0.8152 | 4.4124 | 0.1438 | 0.0000 | 6.9514e-17 | 5.8732e-05 | [0.7543, 0.8733] |
| CGN Native Palm Center vs CGN Mask Palm Center | 0.0371 | 0.1284 | 0.0913 | 2.4575 | 0.0556 | 0.1068 | 8.4589e-04 | 1.9310e-03 | [0.0477, 0.1372] |
