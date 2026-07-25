
# 📡 SLP-TENN

[![Code (GitHub)](https://img.shields.io/badge/Code-GitHub-blue?logo=github)](https://github.com/zhangjinshuoseu-create/SLP_TENN)
[![Paper (arXiv)](https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv)](https://arxiv.org/abs/2510.02108)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/{user}/{repo}/blob/main/LICENSE)

This repository provides the open-source code for the paper **"[Unlocking Symbol-Level Precoding Efficiency Through Tensor Equivariant Neural Network](https://arxiv.org/abs/2510.02108)"**. Building on the **tensor equivariant neural network (TENN)**, the paper proposes an **attention-based multidimensional equivariant (AMDE)** module, and uses it to design a **unified deep-learning framework for symbol-level precoding (SLP)** — one that covers both the **CIZF** (SINR-balancing) and **CIMMSE** criteria, both **PSK and QAM** constellations, and both **perfect and imperfect CSI** (robust SLP under channel aging). The framework retains most of the performance gains of optimal SLP while cutting the online per-symbol complexity to **linear**, achieving roughly an **80× GPU speedup** over conventional SLP.

---

## 📄 Get the Paper

- **arXiv (with detailed appendix)** — [arXiv:2510.02108](https://arxiv.org/abs/2510.02108)

---

## 🧠 Core Concepts

This repository provides a **low-complexity, deep-learning-based framework for symbol-level precoding (SLP)**. Its core is a tensor-equivariant network that learns the SLP **perturbation factors** directly, replacing the expensive per-symbol iterative optimization used by conventional SLP. The paper first analyzes the SLP problem and proves that the mapping from the problem information to its optimal solution is **tensor equivariant (TE)** — permuting users or symbols at the input permutes the solution in exactly the same way. By matching the network's parameter-sharing pattern to this TE structure, the resulting networks obtain low complexity and strong generalization. Two networks are designed accordingly:

- **SLPN** — for **perfect CSI**.
- **RSLPN** — for **imperfect CSI**, a two-stage pipeline (RSLPN-A + RSLPN-B) that realizes the robust MMSE design while preserving the same TE.

**Tensor Equivariance (TE)** generalizes permutation equivariance to high-dimensional tensors, and the basic modules include:

- **Multidimensional Equivariance (MDE)**: permuting each tensor dimension independently produces the same permutation at the output.
- **High-Order Equivariance (HOE)**: the same permutation is applied simultaneously across multiple dimensions.
- **Multidimensional Invariance (MDI)**: the output is unchanged under permutations along specified dimensions.

---

## ✨ Key Features

- 🌱 **Scalable** — generalizes across different user numbers $K$ and block lengths $L$ without retraining; a single trained network adapts to unseen configurations.
- ⚡ **Efficient** — replacing the iterative NNLS solver; roughly an 80× GPU speedup over conventional SLP in typical settings.
- 🧱 **Model-driven** — the network only learns the perturbation factors; the transmit signal is recovered in closed form.
- 🔁 **Block-parallel** — an entire symbol block is processed in a single forward pass.
- 📶 **General** — one architecture covers CIZF & CIMMSE, PSK & QAM, and perfect & imperfect CSI.

---



## 🔄 Scope: From Network Output $\mathbf{D}$ to Transmit Signal

> [!IMPORTANT]
> **This repository implements only the learning core.** From the channel $\mathbf{H}$ and symbols $\mathbf{S}$, it builds the network inputs $(\mathbf{B},\mathbf{C})$ and trains/tests the network that outputs the **perturbation tensor $\mathbf{D}$** (the set of $\delta$ factors). Everything that turns $\mathbf{D}$ into an actual transmit signal — post-net refinement, closed-form precoding, block-level power reallocation, and SER evaluation — lives in a **separate MATLAB pipeline and is _not_ open-sourced here**. The steps below summarize that downstream so the full chain stays reproducible from the paper.

```text
This repo   :  H, S ──► build (B, C) ──► [ SLPN / RSLPN ] ──► D = [δ̂μ, δ̂ν]
MATLAB side :        D ──► (optional ρ refine) ──► s̃c ──► closed-form xc ──► block rescale ──► SER
```

Equation numbers below match [`paper/LCSLP.pdf`](paper/LCSLP.pdf).

**Step 1 — Non-negativity & decomposition (Eq. 48–49).**
$$\hat{\mathbf{D}} = \mathrm{ReLU}(\mathbf{D}), \qquad \hat{\boldsymbol{\delta}}_\mu[l] = \hat{\mathbf{D}}_{[:,l,1]}, \qquad \hat{\boldsymbol{\delta}}_\nu[l] = \hat{\mathbf{D}}_{[:,l,2]}.$$

**Step 2 — (Optional) post-net refinement (Eq. 50–52).** A scalar $\rho[l]\ge 0$ rescales the perturbation. With $\mathbf{p}[l] = \boldsymbol{\Lambda}_\mu[l]\hat{\boldsymbol{\delta}}_\mu[l] + \boldsymbol{\Lambda}_\nu[l]\hat{\boldsymbol{\delta}}_\nu[l]$,
$$\rho[l] = \max\left\{0,\ -\frac{\mathbf{s}_c^H[l]\boldsymbol{\Upsilon}\mathbf{p}[l] + \mathbf{p}^H[l]\boldsymbol{\Upsilon}\mathbf{s}_c[l]}{2\,\mathbf{p}^H[l]\boldsymbol{\Upsilon}\mathbf{p}[l]}\right\}, \qquad \tilde{\mathbf{s}}_c[l] = \mathbf{s}_c[l] + \rho[l]\,\mathbf{p}[l].$$
Without refinement, set $\rho[l]=1$, i.e. $\tilde{\mathbf{s}}_c[l] = \mathbf{s}_c[l] + \mathbf{p}[l]$ (Eq. 11 / 17).

**Step 3 — Closed-form precoding.**
- CIZF (Eq. 10): $\ \mathbf{x}_c^\star[l] = \gamma^\star[l]\,\mathbf{H}^\dagger\tilde{\mathbf{s}}_c^\star[l], \quad \gamma^\star[l] = \sqrt{P_T / \lVert \mathbf{H}^\dagger\tilde{\mathbf{s}}_c^\star[l]\rVert_2^2}.$
- CIMMSE (Eq. 15–16): $\ \mathbf{x}_c^\star[l] = \gamma^\star[l]\,\mathbf{H}^H\boldsymbol{\Upsilon}_{\mathrm{MMSE}}\tilde{\mathbf{s}}_c^\star[l], \quad \gamma^\star[l] = \sqrt{P_T / \lVert \mathbf{H}^H\boldsymbol{\Upsilon}_{\mathrm{MMSE}}\tilde{\mathbf{s}}_c^\star[l]\rVert_2^2}.$

**Step 4 — Block-level power reallocation (Eq. 18).** One rescaling factor per block:
$$\bar{\gamma}^\star = \sqrt{\frac{L}{\sum_{l=1}^{L} 1/(\gamma^\star[l])^2}}, \qquad \bar{\mathbf{x}}_c^\star[l] = \frac{\bar{\gamma}^\star}{\gamma^\star[l]}\,\mathbf{x}_c^\star[l].$$

**Step 5 — Demodulation & SER (Eq. 19).** Transmit $\bar{\mathbf{x}}_c^\star[l]$, scale the received signal by $\bar{\gamma}$, demodulate, and compute the SER.

> **Imperfect CSI (robust SLP).** The closed-form precoder $\mathbf{P}[l]$ is instead assembled from Eq. (58)–(61), using the auxiliary variable $\boldsymbol{\Psi}$ (from RSLPN-A) together with the perturbation factors $\mathbf{D}$ (from RSLPN-B).

---


## 🔧 Network / Module Introduction

The framework is built from tensor-equivariant building blocks and one composite attention module:

| Module 🧩 | Function ⚙️ | Shapes ♾️ | Code |
|:--|:--|:--|:--|
| **MDE layer** | Fully-connected layer sharing parameters so as to satisfy multidimensional equivariance. | **In** $\mathrm{bs}\times M_1\times\dots\times M_N\times D_I$ → **Out** $\mathrm{bs}\times M_1\times\dots\times M_N\times D_O$ | `MDE_Module`, `MDE_Module_LowFLOPs` in [`models/te_module.py`](models/te_module.py) |
| **HOE layer** | Equivariant layer for identical permutations across multiple input/output dimensions (2-1 order used here). | **In** $\mathrm{bs}\times K\times K\times L\times D_I$ → **Out** $\mathrm{bs}\times K\times L\times D_O$ | `HOE_2_1_Module` in [`models/te_module.py`](models/te_module.py) |
| **MDI module** | Attention-based module invariant along specified dimensions. | **In** $\mathrm{bs}\times M_1\times\dots\times M_N\times D_I$ → **Out** with invariant dims removed | `MDI_Module` in [`models/te_module.py`](models/te_module.py) (used by RSLPN-A) |
| **AMDE** | Attention-based MDE residual block (feature-attention + equivariant-dimension attention), the backbone of the network. | **In / Out** $\mathrm{bs}\times K\times L\times F$ (also 3-D axes for RSLPN-A) | `AMDE_Block` / `AMDE_Network` in [`models/te_models.py`](models/te_models.py); attention helpers in [`models/te_module.py`](models/te_module.py) |

**SLPN** (perfect CSI; class `SLPN` in [`models/prec_models.py`](models/prec_models.py)) stacks these into the mapping $G$. Feature dimensions below use $F$ for the hidden width (`d_hidden`) and $T$ for the number of AMDE blocks (`n_amde_layer`). Paper / default script setting for CIZF-DL & CIMMSE-DL: $F=4$, $T=4$.

$$
\begin{aligned}
\mathbf{C}' &= \mathrm{SiLU}(\mathrm{BN}(f_{\mathrm{HOE}}(\mathbf{C}))) && \in \mathbb{R}^{K\times L\times F} \\
\mathbf{C}'' &= \mathrm{PReLU}(\mathrm{BN}(\mathrm{FC}(\mathbf{C}'))) && \in \mathbb{R}^{K\times L\times F} \\
\mathbf{B}' &= \mathrm{PReLU}(\mathrm{BN}(f_{\mathrm{MDE}}(\mathbf{B}))) && \in \mathbb{R}^{K\times L\times F} \\
\mathbf{F} &= \mathrm{FC}([\mathbf{C}'', \mathbf{B}']) && \in \mathbb{R}^{K\times L\times F} \\
\mathbf{D} &= \mathrm{FC}(f_{\mathrm{AMDE}}^{\times T}(\mathbf{F})) && \in \mathbb{R}^{K\times L\times 2}
\end{aligned}
$$

In code, the forward call is `D = model(mat, vec)` where `mat` is $\mathbf{C}$ and `vec` is $\mathbf{B}$.

### RSLPN (imperfect CSI / channel aging)

This repository also includes the robust pipeline (RCIMMSE-DL):

- **RSLPN-A** (`RSLPN_A` in [`models/prec_models.py`](models/prec_models.py)): estimates the non-negative auxiliary variable $\boldsymbol{\Psi}$ (tensor `psi` of shape `[bs, K, L]`). Input features are built inside `input_layer` from channel, symbols, noise power, aging coefficients, and $\mathbf{U}$ (feature width $D_I=8$). Architecture: 3-D AMDE → MDI (pool antenna axis) → 2-D AMDE → Softplus. Script defaults: $F=16$, $T_1=T_2=2$.
- **RSLPN-B** (alias `RSLPN_B = SLPN`): same architecture as SLPN; predicts $\mathbf{D}$ from robust KKT features $(\mathbf{B},\mathbf{C})$ built via `get_RMMSE_kkt_Upsilon` + `rslpn_kkt_features_from_upsilon` in [`func/prec_func.py`](func/prec_func.py). Script defaults: $F=16$, $T=2$.

Training is sequential: train RSLPN-A first, freeze it, then train RSLPN-B.

---

## 🔄 From Network Output D to Transmit Signal

The repository stops at $\mathbf{D}$ (and `psi` for the robust case). For completeness, here is how the paper turns $\mathbf{D}$ into the actual transmit signal and, ultimately, an SER measurement. These steps are implemented separately (MATLAB) and are **not** part of this codebase, but you can reproduce them from the equations below. Equation numbers match [`paper/LCSLP.pdf`](paper/LCSLP.pdf).

**Step 1 — Non-negativity & decomposition (Eq. 48–49).**
$$\hat{\mathbf{D}} = \mathrm{ReLU}(\mathbf{D}), \qquad \hat{\boldsymbol{\delta}}_\mu[l] = \hat{\mathbf{D}}_{[:,l,1]}, \qquad \hat{\boldsymbol{\delta}}_\nu[l] = \hat{\mathbf{D}}_{[:,l,2]}.$$

**Step 2 — (Optional) post-net refinement (Eq. 50–52).** Introduce a scalar $\rho[l]\ge 0$ that rescales the perturbation; with $\mathbf{p}[l] = \boldsymbol{\Lambda}_\mu[l]\hat{\boldsymbol{\delta}}_\mu[l] + \boldsymbol{\Lambda}_\nu[l]\hat{\boldsymbol{\delta}}_\nu[l]$,
$$\rho[l] = \max\left\{0,\ -\frac{\mathbf{s}_c^H[l]\boldsymbol{\Upsilon}\mathbf{p}[l] + \mathbf{p}^H[l]\boldsymbol{\Upsilon}\mathbf{s}_c[l]}{2\,\mathbf{p}^H[l]\boldsymbol{\Upsilon}\mathbf{p}[l]}\right\}, \qquad \tilde{\mathbf{s}}_c[l] = \mathbf{s}_c[l] + \rho[l]\,\mathbf{p}[l].$$
Without refinement, set $\rho[l]=1$, i.e. $\tilde{\mathbf{s}}_c[l] = \mathbf{s}_c[l] + \mathbf{p}[l]$ (Eq. 11 / 17).

**Step 3 — Closed-form precoding.**
- CIZF (Eq. 10): $\quad \mathbf{x}_c^\star[l] = \gamma^\star[l]\,\mathbf{H}^\dagger\tilde{\mathbf{s}}_c^\star[l], \quad \gamma^\star[l] = \sqrt{P_T / \lVert \mathbf{H}^\dagger\tilde{\mathbf{s}}_c^\star[l]\rVert_2^2}.$
- CIMMSE (Eq. 15–16): $\quad \mathbf{x}_c^\star[l] = \gamma^\star[l]\,\mathbf{H}^H\boldsymbol{\Upsilon}_{\mathrm{MMSE}}\tilde{\mathbf{s}}_c^\star[l], \quad \gamma^\star[l] = \sqrt{P_T / \lVert \mathbf{H}^H\boldsymbol{\Upsilon}_{\mathrm{MMSE}}\tilde{\mathbf{s}}_c^\star[l]\rVert_2^2}.$

**Step 4 — Block-level power reallocation (Eq. 18).** Use a single rescaling factor per block:
$$\bar{\gamma}^\star = \sqrt{\frac{L}{\sum_{l=1}^{L} 1/(\gamma^\star[l])^2}}, \qquad \bar{\mathbf{x}}_c^\star[l] = \frac{\bar{\gamma}^\star}{\gamma^\star[l]}\,\mathbf{x}_c^\star[l].$$

**Step 5 — Demodulation & SER (Eq. 19).** Transmit $\bar{\mathbf{x}}_c^\star[l]$, scale the received signal by $\bar{\gamma}$ as in Eq. (19), demodulate, and compute SER.

> For **imperfect CSI (robust SLP)**, the closed-form precoder $\mathbf{P}[l]$ is instead assembled from Eq. (58)–(61), using the auxiliary variable $\boldsymbol{\Psi}$ (estimated by RSLPN-A) and the perturbation factors $\mathbf{D}$ (from RSLPN-B).

---

## 📌 Examples

- **Example 1 — CIZF-DL** (SINR-balancing, perfect CSI): `train_cizf_*.py` / `test_cizf_*.py` (e.g. `UE12TX12_QPSK`, `UE12TX14_QPSK`, `UE12TX14_16QAM`).
- **Example 2 — CIMMSE-DL** (MMSE, perfect CSI): `train_cimmse_*.py` / `test_cimmse_*.py` (QPSK / 16QAM variants).
- **Example 3 — RCIMMSE-DL** (robust MMSE under channel aging): `train_rcimmse_RSLPN_A.py` → `train_rcimmse_RSLPN_B.py` → `test_rcimmse_RSLPN.py`.

Each example learns the mapping to $\mathbf{D}$ (and, for the robust case, additionally $\boldsymbol{\Psi}$); the transmit-signal reconstruction of [Step 3–5](#from-network-output-d-to-transmit-signal) is left to the external MATLAB pipeline.

---

## 🏗️ Project Structure

```
LCSLP_github/
├── paper/                         # Paper PDF (+ TeX source)
│   ├── LCSLP.pdf
│   └── LCSLP.tex
├── models/                        # Networks / TE layers
│   ├── prec_models.py             # SLPN, RSLPN_A, RSLPN_B(=SLPN)
│   ├── te_models.py               # AMDE_Network / AMDE_Block, dim-list helpers
│   └── te_module.py               # MDE / HOE / MDI / FA-MDE / EA-MDE
├── func/                          # Pipeline helpers
│   ├── prec_func.py               # CIR thresholds; B,C construction; robust Υ
│   ├── train_func.py              # Train / test loops (MSE; save predictions)
│   └── init_func.py               # Param containers, DatasetFolder_*, init
├── train_cizf_*.py                # CIZF-DL training entry points
├── test_cizf_*.py                 # CIZF-DL testing entry points
├── train_cimmse_*.py              # CIMMSE-DL training entry points
├── test_cimmse_*.py               # CIMMSE-DL testing entry points
├── train_rcimmse_RSLPN_A.py       # RSLPN-A training (ψ)
├── train_rcimmse_RSLPN_B.py       # RSLPN-B training (D; needs frozen A)
├── test_rcimmse_RSLPN.py          # Joint RSLPN-A/B testing
├── train_data/                    # (not shipped) place .mat datasets here
├── save_data/                     # (created at runtime) checkpoints & preds
├── README.md
└── LICENSE                        # MIT
```

| File | Role in pipeline |
|:--|:--|
| `func/prec_func.py` | **Input construction**: CIR boundaries; `slp_kkt_features` / robust feature builders |
| `models/te_module.py`, `models/te_models.py` | **Network** building blocks (TE layers, AMDE) |
| `models/prec_models.py` | **Network** SLPN / RSLPN-A / RSLPN-B |
| `func/train_func.py`, `func/init_func.py` | **Training / testing** loops and configs |
| `train_*.py` / `test_*.py` | **Entry points** (paths, hyper-parameters, model construction) |

---

## 🚀 Usage

Dependencies used by the scripts include PyTorch, NumPy, SciPy (`loadmat` / `savemat`), and `einops`. Training / testing expect a CUDA device index via `gpu_id` (entry scripts use non-negative GPU ids; use CPU only if you change the device string in the training code yourself).

### 1. Prepare data

Data are **not** included in this repository. Place `.mat` files under the `in_folder` path configured in each entry script (default pattern: `./train_data/{EXPERIMENT_NAME}/`). Checkpoints and predictions are written to `./save_data/{EXPERIMENT_NAME}/`.

**CIZF-DL** (example `train_cizf_UE12TX12_QPSK.py`):

| File stem (key) | Expected array layout |
|:--|:--|
| `upsilon_cizf` | `[sample_num, K, K]` complex |
| `symbol_data` | `[sample_num, K, L]` complex |
| `delta_cizf` | `[sample_num, 2K, L]` (loader splits to `[sample_num, K, L, 2]`) |

**CIMMSE-DL** (multi-SNR): `upsilon_cimmse` is `[sample_num, K, K, n_snr]`; `delta_cimmse` is `[sample_num, n_snr, 2K, L]`. Default SNR list in scripts: `{0,5,10,15,20,25,30}` dB.

**RCIMMSE-DL**: `channel_data` `[sample_num, K, N_T]`, `symbol_data` `[sample_num, K, L]`, `U_data`, `omega_data`, plus `psi_data` (for A) or `delta_data` (for B). Scripts use `txlen=50`, `alpha_sim=0.995`, SNR list `{10,...,40}` dB.

Typical split in scripts: CIZF/CIMMSE `n_train=90000`, `n_test=10000`; RSLPN `n_train=55000`, `n_test=5000` (per channel realization before SNR tiling where applicable).

### 2. Train

Hyper-parameters are set **inside** each script (no CLI argparse). Minimal commands:

```bash
# CIZF-DL (QPSK, K=12, N_T=12) — SLPN with d_hidden=4, n_amde_layer=4
python train_cizf_UE12TX12_QPSK.py

# CIMMSE-DL (QPSK, K=12, N_T=14)
python train_cimmse_UE12TX14_QPSK.py

# RCIMMSE-DL: train A, then B
python train_rcimmse_RSLPN_A.py
python train_rcimmse_RSLPN_B.py
```

Checkpoints are saved as `./save_data/.../TE.pth.tar` (`net_name='TE'`).

### 3. Test

```bash
python test_cizf_UE12TX12_QPSK.py
python test_cimmse_UE12TX14_QPSK.py
python test_rcimmse_RSLPN.py
```

Tests load `TE.pth.tar` from the corresponding `out_folder`, print Test MSE, and write predicted `delta` (and `psi` for robust) `.mat` files for the external MATLAB pipeline.

### 4. Minimal API example

```python
import torch
from models.te_models import generate_amde_dim_list, generate_mde_dim_list, generate_ea_dim_list
from models.prec_models import SLPN
from func.prec_func import slp_cir_thresholds, slp_kkt_features

amde_dim_list = generate_amde_dim_list(n_amde_layer=4, n_dim=2)
mde_dim_list = generate_mde_dim_list(n_dim=2)
ea_dim = generate_ea_dim_list(n_dim=2)

net = SLPN(
    d_c=8, d_b=4, d_hidden=4, n_amde_layer=4,
    amde_dim_list=amde_dim_list, mde_dim_list=mde_dim_list,
    fa_dim=(1, 2), ea_dim=ea_dim,
)

# upsilon: [bs, K, K] complex; tx: [bs, K, L] complex
thr_u, thr_l = slp_cir_thresholds(tx, constellation='QPSK', pskOrder=2)
mat, vec = slp_kkt_features(upsilon, tx, thr_u, thr_l)  # C, B
# mat: [bs, K, K, L, 8]  |  vec: [bs, K, L, 4]
D = net(mat, vec)  # output shape [bs, K, L, 2]
```

---

## 📚 Citation

If you find this repository useful, please cite:

```bibtex
@ARTICLE{slptenn,
  author  = {Zhang, Jinshuo and Wang, Yafei and Yi, Xinping and Wang, Wenjin and Jin, Shi and Chatzinotas, Symeon and Ottersten, Bj\"orn},
  title   = {Unlocking Symbol-Level Precoding Efficiency Through Tensor Equivariant Neural Network},
  journal = {},
  year    = {},
  volume  = {},
  number  = {},
  pages   = {},
  doi     = {}
}
```


## 📝 License

Released under the [MIT License](LICENSE).

---


