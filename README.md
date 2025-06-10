# Tunable Domain Adaptation Using Unfolding

## Authors
- Snehaa Reddy ([snehaareddy192@gmail.com](mailto:snehaareddy192@gmail.com))
- Jayaprakash Katual ([katualjayaprakash@gmail.com](mailto:katualjayaprakash@gmail.com))
- Satish Mulleti ([mulleti.satish@gmail.com](mailto:mulleti.satish@gmail.com))  
Department of Electrical Engineering,  
Indian Institute of Technology Bombay, Mumbai, India, 400076

## Abstract
Machine learning models often struggle to generalize across domains with varying data distributions, such as differing noise levels, leading to degraded performance. Traditional strategies like personalized training, which trains separate models per domain, and joint training, which uses a single model for all domains, have significant limitations in flexibility and effectiveness. To address this, we propose two novel domain adaptation methods for regression tasks based on interpretable unrolled networks—deep architectures inspired by iterative optimization algorithms. These models leverage the functional dependence of select tunable parameters on domain variables, enabling controlled adaptation during inference. Our methods include Parametric Tunable-Domain Adaptation (P-TDA), which uses known domain parameters for dynamic tuning, and Data-Driven Tunable-Domain Adaptation (DD-TDA), which infers domain adaptation directly from input data. We validate our approach on compressed sensing problems involving noise-adaptive sparse signal recovery and domain-adaptive gain calibration, demonstrating improved or comparable performance to domain-specific models while surpassing joint training baselines. This work highlights the potential of unrolled networks for effective, interpretable domain adaptation in regression settings.


Our methods include:
1. **Parametric Tunable-Domain Adaptation (P-TDA)**: Uses known domain parameters for dynamic tuning
2. **Data-Driven Tunable-Domain Adaptation (DD-TDA)**: Infers domain adaptation directly from input data


## Keywords
`Unrolling`, `LISTA`, `Domain-Adaptation`, `Compressive Sensing`, `Blind-Gain Calibration`, `Model-Based Learning`

## Repository Structure
### Core Components
| Component             | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `A.npy`               | Measurement matrix used in compressed sensing                               |
| `X.npy`               | Sparse signal representations                                               |
| `*_test.npy` files    | Test datasets (inputs and outputs) for various domains                      |

### Main Directories
| Directory              | Contents                                                                     | Key Files                                                                 |
|------------------------|------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| **gain_calib-LISTA**   | Gain calibration experiments                                                 | |
| ├─ generalization      | Cross-domain generalization tests                                            | `DDTDA.py`, `PTDA.py`, `net_*.pth` models                                |
| ├─ random              | Test data for random gain perturbations                                      | `X/Y{1,2,3}_test.npy`                                                    |
| ├─ struc               | Test data for structured gain patterns                                       | `X/Y{1,2,3}_test.npy`                                                    |
| └─ *root*              | Core calibration scripts                                                     | `wiener_*_calibration_*.py` (JT/PT/P-TDA/DD-TDA implementations)         |
| **NA-LISTA**           | Noise-adaptive LISTA experiments                                             | |
| ├─ broad_SNR           | Test data for wide SNR ranges (20-50 dB)                                     | `X/Y{1,2,3}_test.npy`                                                    |
| ├─ narrow_SNR          | Test data for narrow SNR ranges (35-38 dB)                                   | `X/Y{1,2,3}_test.npy`                                                    |
| ├─ generalization      | Multi-domain test sets (6 domains)                                           | `X/Y{1-6}_test.npy`                                                      |
| └─ *root*              | Adaptation method implementations                                            | `JT.py`, `PT.py`, `PTDA.py`, `DDTDA.py`                                   |

### Scripts Overview
| Experiment Type        | Training Scripts                                      | Adaptation Methods                          |
|------------------------|-------------------------------------------------------|---------------------------------------------|
| **Gain Calibration**   | `wiener_normal_gain_calibration_jt.py` (Joint)        | `wiener_tunable_gain_calibration_ptda.py` (P-TDA) |
|                        | `wiener_normal_gain_calibration_pt1.py` (Personalized)| `wiener_tunable_gain_calibration_ddtda.py` (DD-TDA)|
| **Noise Adaptation**   | `JT.py` (Joint Training)                              | `PTDA.py` (Parametric TDA)                  |
|                        | `PT.py` (Personalized Training)                       | `DDTDA.py` (Data-Driven TDA)                |


## Citation  
If you use this work, please cite:

