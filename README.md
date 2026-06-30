# Neural ODE-LMM

**Neural ODE enhanced linear mixed effect models for estimating complex  association pattern of time-varying covariates with marker trajectory**

This repository contains the code accompanying the paper:

> Li Z.A., Clairon Q., Samieri C., Prague M.\*, Proust-Lima C.\* (2026). *Neural ODE enhanced linear mixed effect models for estimating complex  association pattern of time-varying covariates with marker trajectory.*

## Overview

Longitudinal cohort studies generate irregularly spaced, partially missing repeated measurements whose outcomes may depend on the full trajectory of time-varying exposures — not just their instantaneous values. Classical linear mixed-effects models (LMMs) can accommodate such effects but require the analyst to pre-specify the functional form linking covariate history to the outcome.

The **Neural ODE-LMM** embeds a Neural Ordinary Differential Equation within the mixed-effects framework:

- A learned vector field encodes covariate trajectories into a continuous-time latent state that drives both the fixed- and random-effect design, while preserving the standard LMM observation model.
- All parameters are estimated by maximising a penalised marginal likelihood.
- Covariate effects are quantified via the **trajectory-profile ΔPDP** (partial-dependence contrast), which compares predictions under counterfactual covariate paths. Variance is estimated via the delta method applied to the inverse empirical Fisher information within the penalised-likelihood framework of Commenges et al. (2015).

## Model Architecture

```
Xi(0), Xs_i, ti0 ──► Encoder ──► Λi(0)
                                    │
                     Neural ODE     │    dΛ/dt = f(Λ(t), X̄(t), M(t), t)
                                    ▼
                               Λi(tij) ──► ρ_ψ(·)ᵀβ  (fixed effects)
                                        ──► g_ξ(·)ᵀbi (random effects)
                                        ──► Ŷi(tij)
```

Baseline covariates are mapped to an initial latent state by the encoder. The Neural ODE integrates the latent state forward in continuous time, driven by interpolated time-varying covariates. At each visit, two learned networks produce fixed- and random-effect design vectors, preserving the standard LMM observation model. A group lasso penalty encourages parsimonious use of the direct (skip) pathway for time-varying covariates.

## Key Features

- **Continuous-time latent dynamics**: the ODE encoder captures cumulative, path-dependent covariate effects without requiring the analyst to pre-specify the functional form.
- **LMM observation model**: subject-specific random effects and marginal likelihood are preserved, enabling standard mixed-model inference.
- **ΔPDP inference**: trajectory-profile partial dependence contrasts with delta-method confidence intervals provide interpretable, uncertainty-equipped association measures.
- **Group lasso regularisation**: applied to the first-layer decoder weights to encourage covariates to route through the ODE pathway and induce group-wise sparsity (Yuan & Lin, 2006).
- **Variable selection**: dual-pathway architecture (ODE + skip connection) lets the data determine whether each covariate operates through its accumulated history or its current value.

## Application

The methodology is applied to assess the association between cardiometabolic health (BMI, fasting glucose, HDL cholesterol, blood pressure) and cognitive trajectory (IST subtest) in the population-based **Trois-Cités (3C) cohort** — 5,859 participants aged ≥ 65, followed up to 14 years with irregular visit schedules.

## Repository Structure

```
├── R/                  # R scripts
│   ├── HLME_*.R        # HLME comparison models (lcmm package)
│   └── simulation*.R   # Simulation pipeline (calibrate → define → loop → aggregate)
├── python/             # Python scripts
│   ├── model.py        # Neural ODE-LMM model (PyTorch + torchdiffeq)
│   ├── train.py        # Training loop (penalised marginal NLL)
│   └── pdp_analysis.py # ΔPDP, Fisher information, delta-method CIs
└── ...
```

> **Note**: the actual directory layout may differ; please refer to the contents of the repository.

## Requirements

**Python** (≥ 3.10):
- PyTorch ≥ 2.6.0
- torchdiffeq
- NumPy, SciPy

**R**:
- `lcmm` (for HLME comparison models)

## Computational Environment

All models were trained on a dual-socket Intel Cascade Lake node (2 × 18 cores, 192 GB RAM) without GPU acceleration. Training a single Neural ODE-LMM on the 3C cohort takes approximately 30 minutes; each simulation replicate completes in 30–40 minutes. The model can also be trained on a standard laptop.

## Simulation Study

The simulation study validates the ΔPDP estimator across contrasting scenarios:

- **Scenario S1** — Instantaneous linear BMI × age association: verifies that the model produces near-nominal coverage and bounded bias when the true effect lies within the hypothesis class.
- **Scenario S2** — Cumulative BMI burden: demonstrates the ODE encoder's ability to recover path-dependent effects that an LMM cannot capture without pre-specifying the correct accumulation functional.

## Citation

```bibtex
@article{li2026neuralodelmm,
  title={Neural ODE enhanced linear mixed effect models for 
        estimating complex  association pattern of time-varying covariates with marker trajectory},
  author={Li, Zhe Aurore and Clairon, Quentin and Samieri, C{\'e}cilia 
          and Prague, M{\'e}lanie and Proust-Lima, C{\'e}cile},
  journal={},
  year={2026}
}
```

## License

Please see the [LICENSE](LICENSE) file for details.
