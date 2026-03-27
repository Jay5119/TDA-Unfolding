
# Tunable Domain Adaptation Using Unfolding (TDA-Unfolding)

## Abstract
Machine learning models often struggle to generalize across domains with varying data distributions, such as differing noise levels, leading to degraded performance. Traditional strategies like personalized training, which trains separate models per domain, and joint training, which uses a single model for all domains, have significant limitations in flexibility and effectiveness. To address this, we propose two novel domain adaptation methods for regression tasks based on interpretable unrolled networks—deep architectures inspired by iterative optimization algorithms. These models leverage the functional dependence of select tunable parameters on domain variables, enabling controlled adaptation during inference. Our methods include Parametric Tunable-Domain Adaptation (P-TDA), which uses known domain parameters for dynamic tuning, and Data-Driven Tunable-Domain Adaptation (DD-TDA), which infers domain adaptation directly from input data. We validate our approach on compressed sensing problems involving noise-adaptive sparse signal recovery, domain-adaptive gain calibration, and domain-adaptive phase retrieval, demonstrating improved or comparable performance to domain-specific models while surpassing joint training baselines. This work highlights the potential of unrolled networks for effective, interpretable domain adaptation in regression settings.

Two adaptation settings are supported:
- **P-TDA (Parametric TDA):** adapts using *known* domain parameters at test time.
- **DD-TDA (Data-Driven TDA):** adapts by *inferring* domain-dependent quantities directly from input measurements.

The codebase includes experiments for (i) noise-adaptive sparse recovery (LISTA-style) and (ii) domain-adaptive gain calibration (Wiener/LISTA-style), along with datasets used in the experiments.

## Folder Structure
```
TDA_Unfolding/
├─ LICENSE
├─ README.md
│
├─ NA-LISTA/                               # Noise-adaptive sparse recovery (LISTA-style)
│  ├─ JT.py / PT.py                        # Baselines: joint vs personalized training
│  ├─ PTDA.py / DDTDA.py                   # Proposed: P-TDA and DD-TDA variants
│  ├─ broad_SNR/                           # Test sets across broad SNR range
│  ├─ narrow_SNR/                          # Test sets across narrow SNR range
│  ├─ generalization/                      # Generalization test sets
│  └─ MNIST/                               # MNIST compressed sensing + OCR evaluation
│
└─ gain_calib-LISTA/                       # Domain-adaptive gain calibration (Wiener/LISTA-style)
   ├─ wiener_normal_gain_calibration_*.py  # JT/PT baselines
   ├─ wiener_tunable_gain_calibration_*.py # PTDA/DDTDA variants
   ├─ random/                              # Random-gain test sets
   ├─ struc/                               # Structured-gain test sets
   └─ generalization/                      # Generalization experiments
```

## Authors
- **Snehaa Reddy** : snehaareddy192@gmail.com
- **Jayaprakash Katual** : katualjayaprakash@gmail.com
- **Satish Mulleti** : mulleti.satish@gmail.com

Department of Electrical Engineering, Indian Institute of Technology Bombay, Mumbai, India.

## Citation
If you use this code or results in your work, please cite:

```
<__>
```

