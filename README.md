# Tunable Domain Adaptation Using Unfolding

## Authors
- Snehaa Reddy ([snehaareddy192@gmail.com](mailto:snehaareddy192@gmail.com))
- Jayaprakash Katual ([katualjayaprakash@gmail.com](mailto:katualjayaprakash@gmail.com))
- Satish Mulleti ([mulleti.satish@gmail.com](mailto:mulleti.satish@gmail.com))  
Department of Electrical Engineering,  
Indian Institute of Technology Bombay, Mumbai, India, 400076

## Abstract
Machine learning models often struggle to generalize across domains with varying data distributions, such as differing noise levels, leading to degraded performance. Traditional strategies like personalized training, which trains separate models per domain, and joint training, which uses a single model for all domains, have significant limitations in flexibility and effectiveness. To address this, we propose two novel domain adaptation methods for regression tasks based on interpretable unrolled networks—deep architectures inspired by iterative optimization algorithms. These models leverage the functional dependence of select tunable parameters on domain variables, enabling controlled adaptation during inference. Our methods include Parametric Tunable-Domain Adaptation (P-TDA), which uses known domain parameters for dynamic tuning, and Data-Driven Tunable-Domain Adaptation (DD-TDA), which infers domain adaptation directly from input data. We validate our approach on compressed sensing problems involving noise-adaptive sparse signal recovery, domain-adaptive gain calibration, and domain-adaptive phase retrieval, demonstrating improved or comparable performance to domain-specific models while surpassing joint training baselines. This work highlights the potential of unrolled networks for effective, interpretable domain adaptation in regression settings.


Our methods include:
1. **Parametric Tunable-Domain Adaptation (P-TDA)**: Uses known domain parameters for dynamic tuning
2. **Data-Driven Tunable-Domain Adaptation (DD-TDA)**: Infers domain adaptation directly from input data


## Keywords
`Unfolding`, `domain-adaptation`, `model-based learning`, `compressive sensing`, `blind-gain calibration`

## Repository Structure

```
TDA-Unfolding/
├── README.md                          # Project documentation
├── LICENSE                            # MIT License
│
├── NA-LISTA/                          # Noise-Adaptive Sparse Recovery Experiments
│   ├── A.npy                         # Measurement matrix (30×100)
│   ├── X.npy                         # Training dataset (100×43,000)
│   ├── PTDA.py                       # Parametric TDA implementation
│   ├── DDTDA.py                      # Data-Driven TDA implementation
│   ├── JT.py                         # Joint Training baseline
│   ├── PT.py                         # Personalized Training baseline
│   ├── broad_SNR/                    # Experiments across broad SNR range
│   ├── narrow_SNR/                   # Experiments within narrow SNR range
│   ├── generalization/               # Generalization tests
│   └── MNIST/                        # MNIST Image Reconstruction
│       ├── TDA_PTDA_MLP.py          # PTDA for image recovery
│       ├── TDA_DDTDA_MLP.py         # DDTDA for image recovery
│       ├── TDA_JT.py                # Joint Training for images
│       ├── TAIL_LISTA.py            # Tail-LISTA variant
│       ├── train_tda.py             # Training script
│       ├── test_tda.py              # Evaluation script
│       ├── train_TailLISTA.py       # Tail-LISTA training
│       ├── Analysis_all.py          # Comprehensive analysis
│       ├── DDIM_mnist_cs.py         # DDIM baseline
│       ├── Utils_tda.py             # Utility functions
│       └── Utils_all.py             # Additional utilities
│
└── gain_calib-LISTA/                 # Gain Calibration Experiments
    ├── A.npy                        # Measurement matrix
    ├── X.npy                        # Training dataset
    ├── Random_gains_close_to_zero.npy
    ├── Structured_gains_close_to_zero.npy
    ├── wiener_normal_gain_calibration_jt.py       # JT for gain calibration
    ├── wiener_normal_gain_calibration_pt1.py      # PT for gain calibration
    ├── wiener_tunable_gain_calibration_ptda.py    # PTDA for gain calibration
    ├── wiener_tunable_gain_calibration_ddtda.py   # DDTDA for gain calibration
    ├── generalization/              # Generalization experiments
    │   ├── PTDA.py
    │   ├── DDTDA.py
    │   ├── JT.py
    │   └── PT.py
    ├── random/                      # Random gain experiments
    └── struc/                       # Structured gain experiments
```

## Problem Domains

This repository addresses three main compressed sensing problems with domain adaptation:

### 1. Noise-Adaptive Sparse Signal Recovery (NA-LISTA)
Recovery of sparse signals from compressed measurements with varying noise levels:
- **Measurement model**: y = Ax + n, where n ~ N(0, σ²I)
- **Domain parameter**: Noise level σ (SNR variation)
- **Challenge**: Single model must handle multiple noise regimes
- **Solution**: Adaptive thresholding based on noise level

**SNR Ranges Tested**:
- Domain 1: σ = 0.1 (low SNR)
- Domain 2: σ = 0.03 (medium SNR)
- Domain 3: σ = 0.005 (high SNR)

### 2. Blind Gain Calibration (gain_calib-LISTA)
Recovery of sparse signals with unknown multiplicative gain factors:
- **Measurement model**: y = D·Ax + n, where D is diagonal gain matrix
- **Domain parameter**: Gain values (random or structured)
- **Challenge**: Simultaneous signal recovery and gain estimation
- **Solution**: Joint calibration and reconstruction

**Gain Types**:
- Random gain values
- Structured/systematic gain patterns
- Near-zero gain scenarios

### 3. Image Compressed Sensing (MNIST)
MNIST digit recovery from compressed measurements with noise:
- **Task**: Image reconstruction and digit classification
- **Metrics**: PSNR, SSIM, classification accuracy
- **Features**: Multi-channel support, OCR-based evaluation

## Methods

### Baseline Methods

1. **Joint Training (JT)**
   - Single model trained across all domains
   - Fixed threshold parameters
   - Lowest computational cost but limited adaptation

2. **Personalized Training (PT)**
   - Separate model per domain
   - Optimal per-domain performance
   - High computational and storage cost

### Proposed Methods

3. **Parametric Tunable-Domain Adaptation (P-TDA)**
   - Uses known domain parameters (e.g., noise level σ)
   - MLP-based threshold predictor
   - Dynamic adaptation at inference time
   - Architecture: 5-layer MLP (512→256→128→64→1)

4. **Data-Driven Tunable-Domain Adaptation (DD-TDA)**
   - Infers domain characteristics from measurements
   - No explicit domain parameter required
   - Learns noise level implicitly from data statistics
   - Similar MLP architecture to P-TDA

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.9+
- NumPy
- SciPy
- scikit-learn
- Matplotlib
- tqdm

### Setup
```bash
# Clone the repository
git clone https://github.com/Jay5119/TDA-Unfolding.git
cd TDA-Unfolding

# Install dependencies
pip install torch numpy scipy scikit-learn matplotlib tqdm
```

## Usage

### 1. Noise-Adaptive Sparse Recovery

#### Training PTDA Model
```bash
cd NA-LISTA
python PTDA.py
```

#### Training DDTDA Model
```bash
cd NA-LISTA
python DDTDA.py
```

#### Training Baseline Models
```bash
# Joint Training
python JT.py

# Personalized Training
python PT.py
```

### 2. Gain Calibration

#### Training Models
```bash
cd gain_calib-LISTA

# PTDA for gain calibration
python wiener_tunable_gain_calibration_ptda.py

# DDTDA for gain calibration
python wiener_tunable_gain_calibration_ddtda.py

# Joint Training baseline
python wiener_normal_gain_calibration_jt.py

# Personalized Training baseline
python wiener_normal_gain_calibration_pt1.py
```

### 3. MNIST Image Compressed Sensing

#### Training
```bash
cd NA-LISTA/MNIST

# Train PTDA model
python train_tda.py --model ptda --epochs 150 --lr 1e-4

# Train DDTDA model
python train_tda.py --model ddtda --epochs 150 --lr 1e-4

# Train Joint Training baseline
python train_tda.py --model jt --epochs 150 --lr 1e-4
```

#### Testing
```bash
# Evaluate trained model
python test_tda.py --model ptda --checkpoint path/to/checkpoint.pth

# Run comprehensive analysis
python Analysis_all.py
```

## Key Features

- **Interpretable Architecture**: Models based on unrolled ISTA iterations
- **Dynamic Adaptation**: Thresholds adapt to domain characteristics at inference
- **Efficiency**: Single model handles multiple domains (vs. PT requiring N models)
- **Flexibility**: Both parametric and data-driven adaptation strategies
- **Generalization**: Tested on unseen noise levels and gain patterns

## Model Architecture

### LISTA (Learned ISTA) Base
All models use the LISTA architecture, which unfolds K iterations of the Iterative Shrinkage-Thresholding Algorithm (ISTA):

```
x^(k+1) = soft_threshold(S·x^(k) + W·y, θ^(k))
```

Where:
- `W`: Feed-forward weight matrix (initialized as A^T/L)
- `S`: Recurrent weight matrix (initialized as I - A^T·A/L)
- `θ^(k)`: Layer-wise threshold parameters
- `K`: Number of unfolded layers (typically 15)

### TDA Modifications
- **JT**: Fixed threshold θ^(k) per layer
- **PT**: Domain-specific fixed thresholds
- **P-TDA**: θ^(k) = MLP(y, A, σ) - threshold adapted to noise level
- **DD-TDA**: θ^(k) = MLP(y, A) - threshold inferred from measurements

## Evaluation Metrics

- **MSE**: Mean Squared Error between recovered and true signals
- **PSNR**: Peak Signal-to-Noise Ratio (dB)
- **SSIM**: Structural Similarity Index (for images)
- **Hit Rate**: Proportion of correctly identified non-zero support
- **Classification Accuracy**: Digit recognition from recovered MNIST images
- **Calibrated Accuracy**: Classification accuracy on correctly predicted ground truth

## Training Configuration

### Typical Hyperparameters
- **Epochs**: 120-215
- **Batch Size**: 32
- **Learning Rate**: 6e-5 to 8e-5
- **Optimizer**: Adam (β₁=0.9, β₂=0.999)
- **Scheduler**: StepLR (step=5, gamma=0.6)
- **Loss Function**: MSE
- **Layers**: 15 unfolded ISTA iterations

### Dataset Details
- **Signal dimension**: N_x = 100
- **Measurement dimension**: N_y = 30
- **Sparsity**: K = 3
- **Training samples**: 43,000
- **Data split**: 60% train, 20% validation, 20% test

## Results Summary

Our methods demonstrate:
- **P-TDA**: Near-optimal performance matching PT with single model
- **DD-TDA**: Strong performance without requiring explicit domain parameters
- **Both**: Significant improvement over JT baseline
- **Generalization**: Good performance on unseen noise levels/gain patterns

## Citation

If you use this code in your research, please cite:

```bibtex
@article{tda-unfolding2025,
  title={Tunable Domain Adaptation Using Unfolding},
  author={Reddy, Snehaa and Katual, Jayaprakash and Mulleti, Satish},
  journal={},
  year={2025},
  institution={Indian Institute of Technology Bombay}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or collaborations, please contact:
- Snehaa Reddy: [snehaareddy192@gmail.com](mailto:snehaareddy192@gmail.com)
- Jayaprakash Katual: [katualjayaprakash@gmail.com](mailto:katualjayaprakash@gmail.com)
- Satish Mulleti: [mulleti.satish@gmail.com](mailto:mulleti.satish@gmail.com)

Department of Electrical Engineering
Indian Institute of Technology Bombay
Mumbai, India, 400076
