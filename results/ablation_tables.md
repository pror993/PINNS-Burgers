# Burgers PINN Ablation Study Results

## 1. Optimization Strategy Sweep

| Strategy | Relative $L_2$ Error (%) | In-Domain PDE Residual | Wall Time (s) |
|---|:---:|:---:|:---:|
| Adam Only | 29.40% | 2.1276e-01 | 15.0s |
| L-BFGS Only | 11.03% | 9.3542e-02 | 5.1s |
| Staged Adam + L-BFGS | 1.79% | 2.2956e-03 | 15.1s |

## 2. Loss Weight Configurations Sweep

| Weight Configuration ($w_{ic}:w_{bc}:w_{pde}$) | Relative $L_2$ Error (%) | PDE Loss | IC Loss | BC Loss |
|---|:---:|:---:|:---:|:---:|
| Equal (1:1:1) | 4.86% | 2.6855e-04 | 1.2656e-04 | 2.9978e-06 |
| Physics-Heavy (1:1:10) | 34.27% | 4.0239e-04 | 1.4127e-02 | 1.6384e-05 |
| Moderate IC/BC (10:10:1) | 6.07% | 8.4768e-04 | 1.2379e-05 | 1.1174e-06 |
| Notebook Baseline (50:10:1) | 1.79% | 5.8620e-04 | 1.1254e-06 | 8.6794e-07 |
| IC-Heavy (100:10:1) | 6.03% | 2.2194e-03 | 1.1956e-06 | 3.7563e-06 |

## 3. Collocation Point Density Sweep

| Collocation Count ($N_f$) | Relative $L_2$ Error (%) | In-Domain PDE Residual | Extrapolation Residual ($t > 1$) | Wall Time (s) |
|---|:---:|:---:|:---:|:---:|
| 500 | 5.63% | 8.4434e-01 | 2.3606e+00 | 5.2s |
| 1000 | 21.45% | 8.8240e-01 | 3.8009e-01 | 11.4s |
| 2500 | 1.79% | 2.2956e-03 | 7.5920e-02 | 14.9s |
| 5000 | 1.24% | 1.0401e-03 | 8.0441e-02 | 23.9s |
