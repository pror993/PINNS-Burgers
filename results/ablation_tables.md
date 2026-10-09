# Burgers PINN Ablation Study Results

## 1. Optimization Strategy Sweep

| Strategy | Relative $L_2$ Error (%) | In-Domain PDE Residual | Wall Time (s) |
|---|:---:|:---:|:---:|
| Adam Only (2,000 epochs) | 29.40% | $2.1276 \times 10^{-1}$ | 15.0s |
| L-BFGS Only (1,000 iter) | 11.03% | $9.3542 \times 10^{-2}$ | 5.1s |
| Staged Adam + L-BFGS (Baseline) | 1.79% | $2.2956 \times 10^{-3}$ | 15.1s |

## 2. Loss Weight Configurations Sweep

| Weight Configuration ($w_{\text{ic}} : w_{\text{bc}} : w_{\text{pde}}$) | Relative $L_2$ Error (%) | PDE Loss | IC Loss | BC Loss |
|---|:---:|:---:|:---:|:---:|
| Equal (1:1:1) | 4.86% | $2.6855 \times 10^{-4}$ | $1.2656 \times 10^{-4}$ | $2.9978 \times 10^{-6}$ |
| Physics-Heavy (1:1:10) | 34.27% | $4.0239 \times 10^{-4}$ | $1.4127 \times 10^{-2}$ | $1.6384 \times 10^{-5}$ |
| Moderate IC/BC (10:10:1) | 6.07% | $8.4768 \times 10^{-4}$ | $1.2379 \times 10^{-5}$ | $1.1174 \times 10^{-6}$ |
| Notebook Baseline (50:10:1) | 1.79% | $5.8620 \times 10^{-4}$ | $1.1254 \times 10^{-6}$ | $8.6794 \times 10^{-7}$ |
| IC-Heavy (100:10:1) | 6.03% | $2.2194 \times 10^{-3}$ | $1.1956 \times 10^{-6}$ | $3.7563 \times 10^{-6}$ |

## 3. Collocation Point Density Sweep

| Collocation Points ($N_f$) | Relative $L_2$ Error (%) | In-Domain PDE Residual | Extrapolation Residual ($t \in [1.0, 1.3]$) | Wall Time (s) |
|---|:---:|:---:|:---:|:---:|
| 500 | 5.63% | $8.4434 \times 10^{-1}$ | $2.3606 \times 10^{0}$ | 5.2s |
| 1,000 | 21.45% | $8.8240 \times 10^{-1}$ | $3.8009 \times 10^{-1}$ | 11.4s |
| 2,500 (Baseline) | 1.79% | $2.2956 \times 10^{-3}$ | $7.5920 \times 10^{-2}$ | 14.9s |
| 5,000 | 1.24% | $1.0401 \times 10^{-3}$ | $8.0441 \times 10^{-2}$ | 23.9s |
