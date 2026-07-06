# Early Disease Detection in Chicken Using Audio Analysis and Deep Learning

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🐔 Project Overview

This project implements an automated **early disease detection system** for poultry farms using deep learning and audio signal processing. The system analyzes chicken vocalizations to classify their health status into three categories: **Healthy**, **Noise**, and **Unhealthy**.

Respiratory diseases cause **30-40% mortality** in poultry farms. Early detection is critical for:
- Preventing disease spread
- Reducing economic losses
- Minimizing antibiotic usage
- Improving animal welfare

### Key Features
✅ **Non-invasive monitoring** through audio analysis  
✅ **High accuracy** (>90%) with F1-scores >88% for all classes  
✅ **Comprehensive feature extraction** (243 features)  
✅ **Advanced preprocessing** (Hamming + Adaptive Kalman filtering)  
✅ **Deep CNN architecture** with attention mechanisms  
✅ **Ensemble learning** for robust predictions  
✅ **GPU-accelerated training** for practical deployment

---

## 📊 Dataset

**Source:** Poultry Vocalization Signal Dataset (Data in Brief, 2023)

| Class | Files | Percentage |
|-------|-------|------------|
| Healthy | 139 | 40.2% |
| Noise | 86 | 24.9% |
| Unhealthy | 121 | 35.0% |
| **Total** | **346** | **100%** |

### Audio Specifications
- **Sampling Rate:** 96 kHz
- **Bit Depth:** 24-bit
- **Format:** WAV (Mono)
- **Duration:** 2-10 seconds per file

---

## 🏗️ System Architecture

### 1. Preprocessing Pipeline

#### Hamming Window
Reduces spectral leakage in frequency analysis:
```
h(n) = 0.54 + 0.46 × cos(2πn/N)
```

#### Adaptive Kalman Filter
Denoises signals while preserving disease markers:
- **Process variance (Q):** 1e-5
- **Measurement variance (R):** 1e-2
- **Adaptation rate:** 0.01

**Algorithm Steps:**
1. **Prediction:**
   - State: `x̂(k|k-1) = F × x̂(k-1|k-1)`
   - Covariance: `P(k|k-1) = F × P(k-1|k-1) × F^T + Q`

2. **Update:**
   - Kalman Gain: `K(k) = P(k|k-1) × H^T × [H × P(k|k-1) × H^T + R]^(-1)`
   - State Update: `x̂(k|k) = x̂(k|k-1) + K(k) × (y(k) - H × x̂(k|k-1))`
   - Covariance Update: `P(k|k) = (I - K(k) × H) × P(k|k-1)`

### 2. Feature Extraction (243 Features Total)

| Feature Group | Count | Description |
|---------------|-------|-------------|
| **MFCC** | 80 | 40 coefficients × 2 (mean + std) |
| **Delta MFCC** | 80 | Temporal dynamics |
| **Chroma** | 24 | 12 pitch classes × 2 |
| **Mel Spectrogram** | 7 | Energy distribution (low/mid/high) |
| **Spectral Contrast** | 14 | 7 bands × 2 |
| **Tonnetz** | 12 | Harmonic content (6 × 2) |
| **Spectral Features** | 8 | Centroid, rolloff, bandwidth, ZCR |
| **Spectral Flatness** | 2 | Mean + std |
| **Spectral Entropy** | 1 | Frequency disorder measure |
| **Autocorrelation** | 3 | Periodicity (max, mean, std) |
| **Statistical** | 8 | Mean, std, var, max, min, RMS, skew, kurtosis |
| **RMS Energy** | 4 | Frame-level energy statistics |

**Key Finding:** Unhealthy chickens show distinctive **power spectrum spike at 0.2 rad/sample**

### 3. CNN Model Architecture

**Total Parameters:** 580,868 (2.22 MB)  
**Trainable:** 578,436 | **Non-trainable:** 2,432

```
Input: (243, 1) - 243 features reshaped for Conv1D

Block 1: Initial Convolution
├─ Conv1D(64 filters, kernel=5, padding='same')
├─ BatchNormalization
├─ ReLU Activation
└─ Dropout(0.2)

Block 2: Residual Block (128 filters)
├─ Conv1D(128, kernel=3) → BatchNorm → ReLU
├─ Conv1D(128, kernel=3) → BatchNorm
├─ Residual Connection (1×1 conv shortcut)
├─ ReLU Activation
├─ MaxPooling1D(pool_size=2)
└─ Dropout(0.3)

Block 3: Residual Block (256 filters)
├─ Conv1D(256, kernel=3) → BatchNorm → ReLU
├─ Conv1D(256, kernel=3) → BatchNorm
├─ Residual Connection (1×1 conv shortcut)
├─ ReLU Activation
└─ Dropout(0.3)

Attention Mechanism
├─ Dense(1, tanh) → Flatten → Softmax
├─ RepeatVector → Permute
└─ Multiply (attention weights × features)

Global Pooling
├─ GlobalAveragePooling1D
├─ GlobalMaxPooling1D
└─ Concatenate → 512 features

Dense Layers
├─ Dense(256, ReLU) → BatchNorm → Dropout(0.4)
├─ Dense(128, ReLU) → BatchNorm → Dropout(0.4)
└─ Dense(3, Softmax) - Output
```

### 4. Training Strategy

#### Data Augmentation (3 variations per sample)
1. **Gaussian Noise:** σ = 0.005
2. **Pitch Shifting:** +2 semitones
3. **Volume Scaling:** ×1.3

**Total Samples After Augmentation:**
- Original: 346
- Augmented: 1,038
- **Total: 1,384**

#### Class Balancing
- **SMOTE** (Synthetic Minority Over-sampling) with k_neighbors=5
- **Class weights:** Balanced
- **StandardScaler:** Features normalized and clipped to [-10, 10]

#### Training Configuration
- **Loss Function:** Categorical Crossentropy
- **Optimizer:** Adam (lr=1e-3)
- **Batch Size:** 16
- **Epochs:** 150
- **Callbacks:**
  - EarlyStopping (patience=15)
  - ModelCheckpoint (save best model)
  - ReduceLROnPlateau (factor=0.5, patience=5)

#### Ensemble Strategy
- Train **3 models** with different random seeds
- **Average predictions** for final output
- Improves robustness and accuracy

---

## 📈 Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | >90% |
| **Healthy F1-Score** | >88% |
| **Noise F1-Score** | >88% |
| **Unhealthy F1-Score** | >88% |
| **Training Time** | 10-15 min (GPU) |

### Model Outputs
Generated artifacts in `cnn_model/results/`:
- `poultry_cnn_model_v2.keras` - Best single model
- `poultry_cnn_ensemble_1/2/3.keras` - Ensemble models
- `poultry_cnn_model_package_v2.pkl` - Scaler + metadata
- `confusion_matrix.png` - Performance visualization
- `training_history.png` - Training curves

---

## 🚀 Installation & Usage

### Prerequisites
```bash
Python 3.9+
CUDA 11.x (for GPU support)
```

### Install Dependencies
```bash
pip install -r cnn_model/requirements_cnn.txt
```

**Key Packages:**
- tensorflow>=2.10.0
- librosa>=0.10.0
- scikit-learn>=1.3.0
- numpy>=1.24.0
- soundfile>=0.12.0

### Prediction on New Audio Files

#### Single File
```bash
cd cnn_model
python predict_cnn_v2.py path/to/audio.wav
```

#### Batch Prediction (Folder)
```bash
python predict_cnn_v2.py path/to/audio/folder
```

#### Example Output
```
File: sample_chicken_1.wav
  Prediction:  Unhealthy (87.3%)
  Healthy:     8.2%
  Noise:       4.5%
  Unhealthy:   87.3%
```

---

## 📁 Project Structure

```
.
├── cnn_model/
│   ├── cnn-training-final.ipynb      # Training notebook (Kaggle)
│   ├── predict_cnn_v2.py             # Prediction script
│   ├── requirements_cnn.txt          # Dependencies
│   └── results/
│       ├── poultry_cnn_model_v2.keras
│       ├── poultry_cnn_ensemble_*.keras
│       ├── poultry_cnn_model_package_v2.pkl
│       ├── confusion_matrix.png
│       └── training_history.png
│
├── src/
│   ├── preprocessing/
│   │   ├── audio_processor.py        # Audio preprocessing pipeline
│   │   └── filters.py                # Hamming & Kalman filters
│   ├── features/
│   │   └── feature_extractor.py      # 243-feature extraction
│   └── utils/
│       └── data_loader.py            # Dataset management
│
├── Healthy/                          # 139 healthy audio files
├── Noise/                            # 86 noise audio files
├── Unhealthy/                        # 121 unhealthy audio files
│
├── config.py                         # Configuration settings
├── main.tex                          # LaTeX presentation
├── Poultry_Health_Training_Colab.ipynb
├── predict_from_colab_model.py
└── README.md
```

---

## 🔬 Technical Details

### Preprocessing Parameters
```python
PREPROCESSING_CONFIG = {
    'frame_duration': 20,          # seconds
    'use_adaptive_kalman': True,
    'process_variance': 1e-5,      # Q matrix
    'measurement_variance': 1e-2,  # R matrix
}
```

### Feature Extraction Configuration
```python
FEATURE_CONFIG = {
    'n_mfcc': 40,
    'n_fft': 2048,
    'hop_length': 512,
}
```

### Model Training
```python
MODEL_CONFIG = {
    'input_shape': (243, 1),
    'filters': [64, 128, 256],
    'attention_heads': 2,
    'dropout_rates': [0.2, 0.3, 0.3, 0.4, 0.4],
    'dense_units': [256, 128],
}
```

---

## 🎯 Key Research Contributions

1. **Comprehensive Feature Set**  
   243 diverse features vs. traditional MFCC-only approaches

2. **Advanced Preprocessing**  
   Hamming + Adaptive Kalman for optimal noise reduction

3. **Novel Architecture**  
   CNN with residual connections + attention mechanism

4. **Practical System**  
   End-to-end pipeline ready for farm deployment

5. **Reproducible Research**  
   Complete documentation and code availability

---

## 📊 Comparison with Baseline

| Approach | Features | Accuracy | Notes |
|----------|----------|----------|-------|
| Traditional ML (SVM) | MFCC only | ~75% | Limited feature set |
| XGBoost | 102 features | ~90% | From paper baseline |
| **Our CNN (Single)** | **243 features** | **>90%** | **Deep learning** |
| **Our CNN (Ensemble)** | **243 features** | **>92%** | **Best performance** |

---

## 🔮 Future Work

1. **Real-time Processing**  
   Optimize for edge devices (Raspberry Pi, embedded systems)

2. **Larger Dataset**  
   Collect more samples for improved generalization

3. **Multi-disease Classification**  
   Extend beyond respiratory diseases

4. **Explainable AI**  
   Implement attention visualization and feature importance

5. **Mobile Application**  
   Develop farmer-friendly mobile interface

6. **Continuous Learning**  
   Implement online learning for model updates

---

## 📚 References

1. **Dataset Source:**  
   Poultry Vocalization Signal Dataset, *Data in Brief*, Volume 50, 2023, 109528  
   DOI: https://doi.org/10.1016/j.dib.2023.109528

2. **Deep Learning for Audio:**  
   Hershey, S., et al. "CNN architectures for large-scale audio classification." *ICASSP 2017*

3. **Residual Networks:**  
   He, K., et al. "Deep residual learning for image recognition." *CVPR 2016*

4. **Attention Mechanisms:**  
   Vaswani, A., et al. "Attention is all you need." *NIPS 2017*

5. **SMOTE:**  
   Chawla, N. V., et al. "SMOTE: Synthetic minority over-sampling technique." *JAIR 2002*

---

## 👤 Author

**Sohel Akhtar**  
Roll No.: 2301216  
Department of Computer Science and Engineering  
Indian Institute of Information Technology Guwahati

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- Dataset providers: Poultry Vocalization Signal Dataset team
- IIIT Guwahati for computational resources
- TensorFlow and Keras communities
- Librosa audio processing library developers

---

## 📞 Contact

For questions or collaborations:
- Email: sohel.akhtar@iiitg.ac.in
- GitHub: [Your GitHub Profile]

---

**⭐ If you find this project useful, please consider giving it a star!**