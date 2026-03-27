# Tunable Domain Adaptation Using Unfolding

## Authors
- Snehaa Reddy ([snehaareddy192@gmail.com](mailto:snehaareddy192@gmail.com))
- Jayaprakash Katual ([katualjayaprakash@gmail.com](mailto:katualjayaprakash@gmail.com))
- Satish Mulleti ([mulleti.satish@gmail.com](mailto:mulleti.satish@gmail.com))  
Department of Electrical Engineering,  
Indian Institute of Technology Bombay, Mumbai, India, 400076

## Abstract



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
This repository is organized into two main application modules for tunable domain adaptation using unrolled networks:
```text
TDA-Unfolding/

```
Each module contains:
- `.py` scripts implementing model variants.
- `.npy` files for matrices and datasets.
- Subdirectories with test and generalization data to evaluate robustness.
