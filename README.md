# Early Detection of Diabetic Retinopathy Using Transfer Learning with ResNet50

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Kaggle](https://img.shields.io/badge/Dataset-APTOS_2019-20BEFF.svg)](https://www.kaggle.com/c/aptos2019-blindness-detection)

## Project Overview

This repository contains the implementation for a deep learning-based system to automatically detect and grade Diabetic Retinopathy (DR) severity from retinal fundus images. The project addresses the critical need for accessible and efficient DR screening, especially in regions with a shortage of ophthalmologists. Using transfer learning on the ResNet50 architecture and a novel two-phase training protocol with inverse-frequency class weighting, our model achieves clinically competitive performance (QWK 0.907) on the APTOS 2019 dataset using a single consumer-grade GPU.

**Project Guide:** Dr. Gokulakrishnan D.  
**Institution:** School of Computing, SRM Institute of Science and Technology, Chennai  
**Team:** Aditya Kumar Singh, Sameer Yadav

---

## Problem Statement

Diabetic Retinopathy (DR) is a leading cause of blindness among the working-age population. The current manual grading process is time-consuming, subjective, and limited by the scarcity of trained experts. The goal is to build an automated, accurate, and efficient multi-class grading system that can perform 5-class severity grading (from No DR to Proliferative DR) to assist in large-scale screening programs.

---

## Methodology & Key Contributions

This project focuses on tackling the challenge of **class imbalance** in medical imaging, where severe DR cases (Grade 3 & 4) are rare but clinically critical.

### Key Contributions:
1.  **Single-GPU Deployable System:** Achieves QWK 0.907 (almost perfect agreement) without needing expensive multi-GPU ensembles.
2.  **Inverse-Frequency Class Weighting:** A mathematically derived weighting scheme (`w3=3.79`, `w4=2.48`) that forces the model to focus on rare, severe cases.
3.  **Two-Phase Training Protocol:**
    - **Phase 1 (Head Warmup):** Trains only the new classification head to stabilize gradients.
    - **Phase 2 (Full Fine-Tuning):** Unfreezes the entire ResNet50 backbone with a lower learning rate to adapt features to the medical domain.
4.  **Rigorous Evaluation:** Reports per-grade Precision, Recall, F1-Score, and Quadratic Weighted Kappa (QWK) for clinical interpretability.

### Alternative Methodologies Rejected:
| Methodology | Reason for Rejection |
| :--- | :--- |
| Custom CNN from Scratch | Requires significantly more data; poorer generalization. |
| Traditional ML + Features | Feature engineering is subjective; misses deep patterns. |
| Vision Transformers (ViT) | Requires massive data (>100k images) and high compute. |
| Complex Ensembles | Impractical for real-time, resource-limited clinical settings. |

---

## Dataset

**Source:** [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection) (Kaggle)

- **Total Images:** 3,662 training images, 1,928 test images (held-out)
- **Classes:** 5 (Matching the International Clinical DR Severity Scale)
    - `0`: No DR
    - `1`: Mild NPDR
    - `2`: Moderate NPDR
    - `3`: Severe NPDR
    - `4`: Proliferative DR

### Data Distribution & Imbalance
The dataset is highly imbalanced, motivating our class-weighting strategy.

| Class | Severity | Count | Percentage |
| :---: | :--- | :---: | :---: |
| 0 | No DR | 1,805 | 49.3% |
| 1 | Mild | 370 | 10.1% |
| 2 | Moderate | 999 | 27.3% |
| 3 | Severe | 193 | 5.3% |
| 4 | Proliferative | 295 | 8.0% |

**Data Split:** Grade-stratified 80% Train, 10% Validation, 10% Test.

---

## System Architecture

The architecture follows a standard but highly optimized machine learning pipeline.

### 1. Data Pipeline
- **Preprocessing:** Resize to `320x320` (higher than standard `224x224` to preserve microaneurysm details), normalize pixels to [0,1].
- **Augmentation (on-the-fly):**
    - Random Rotation (0-360 degrees)
    - Horizontal Flip
    - Vertical Flip
    - *(Note: Color jitter is excluded as color is diagnostically important).*

### 2. Model Architecture (ResNet50 + Custom Head)
- **Backbone:** ResNet50 pre-trained on ImageNet.
- **Classification Head:**
    - Global Average Pooling 2D
    - Dropout (0.5)
    - Dense Layer (2048 units, ReLU activation)
    - Dropout (0.5)
    - Dense Layer (5 units, Softmax activation)

### 3. Training Pipeline
- **Optimizer:** Adam
- **Loss Function:** Categorical Cross-Entropy with **Inverse-Frequency Class Weights**.
    - Weight formula: `w_c = N / (K * n_c)`
    - `w = [0.41, 1.98, 0.73, 3.79, 2.48]` (Grades 0-4). Grade 3 is penalized ~9x more than Grade 0.
- **Two-Phase Schedule:**
    1.  **Warmup (2 epochs):** Backbone frozen, LR = 1e-3.
    2.  **Fine-Tuning (up to 40 epochs):** Backbone unfrozen, LR = 1e-4, Early Stopping (patience=5), ReduceLROnPlateau.

---

## 📊 Results & Performance

The model was evaluated on a held-out test set, achieving robust performance across all severity grades.

### Aggregate Performance
| Metric | Value |
| :--- | :--- |
| **Validation Accuracy** | 85.58% |
| **Test-Set Accuracy** | 84.97% |
| **Validation QWK** | **0.907** (Almost Perfect Agreement) |
| **Test-Set QWK** | 0.901 |
| **Full-Corpus QWK** | 0.926 |

### Per-Class Performance (Validation Set)
| Grade | Precision | Recall | F1-Score | Support |
| :---: | :---: | :---: | :---: | :---: |
| 0 (No DR) | 0.91 | 0.92 | **0.91** | 180 |
| 1 (Mild) | 0.82 | 0.84 | **0.83** | 37 |
| 2 (Moderate) | 0.85 | 0.86 | **0.85** | 100 |
| 3 (Severe) | 0.79 | 0.81 | **0.80** | 19 |
| 4 (Proliferative) | 0.76 | 0.78 | **0.77** | 30 |
| **Macro Avg** | 0.83 | 0.84 | **0.83** | 366 |

### Comparison with Published Methods
| Method | Architecture | Accuracy | QWK | Computational Cost |
| :--- | :--- | :---: | :---: | :--- |
| Pratt et al. (2016)† | Custom CNN | 75.0% | 0.852 | Low |
| Wang et al. (2019)† | DenseNet-121 | 81.5% | 0.893 | Medium |
| **Ours (no weighting)** | ResNet50 | 83.1% | 0.892 | **Low** |
| **Proposed** | **ResNet50** | **85.6%** | **0.907** | **Low** |
| APTOS Winner 2019 | Ensemble (10+ models) | 87.3% | 0.944 | Very High |
*† Different dataset; results not directly comparable but shown for context.*

---

## Getting Started

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- Kaggle API (to download the dataset)

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/diabetic-retinopathy-detection.git
    cd diabetic-retinopathy-detection
    ```

2.  **Install required packages:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download the APTOS 2019 dataset:**
    ```bash
    # Set up Kaggle API (if not already done)
    mkdir ~/.kaggle
    cp kaggle.json ~/.kaggle/
    kaggle competitions download -c aptos2019-blindness-detection
    ```

4.  **Run the training script:**
    ```bash
    python train.py --epochs 40 --batch_size 32 --img_size 320
    ```

### Project Structure
```
├── data/                   # Dataset directory
│   ├── train_images/
│   └── train.csv
├── src/
│   ├── data_preprocessing.py
│   ├── model.py            # ResNet50 architecture
│   ├── train.py            # Two-phase training loop
│   └── evaluate.py         # Metrics & Confusion Matrix
├── weights/                # Saved model checkpoints
├── results/                # Plots, logs, and CSV files
├── requirements.txt
└── README.md
```

---

## Research Gaps Addressed

| Gap ID | Problem from Literature | Our Solution |
| :---: | :--- | :--- |
| **G1** | Class imbalance for severe DR (poor recall on Grade 3 & 4). | **Inverse-frequency class weighting** (w3=3.79, w4=2.48). |
| **G2** | Binary classification (referable/non-referable) is common. | **5-class severity grading** matching clinical scale. |
| **G3** | High computational requirements (ensembles, ViTs). | **Single ResNet50 + Transfer Learning** for efficiency. |
| **G4** | Poor generalization across populations. | **APTOS dataset** (multi-center Indian clinics) + aggressive augmentation. |
| **G5** | Black-box models with no interpretability. | Detailed **per-class metrics & confusion matrices**. |
| **G6** | Speed-accuracy trade-off. | **83.7% accuracy with real-time inference** capability. |
| **G8** | QWK not optimized. | **Achieved QWK 0.907**, the primary clinical metric. |

---

## Limitations & Future Work

### Current Limitations:
- **Geographic Generalization:** Model trained primarily on Indian population data; performance may vary on other ethnicities or camera hardware.
- **F1 Gap:** A 14-point gap persists between Grade 0 (0.91) and Grade 4 (0.77).
- **Interpretability:** No saliency maps (e.g., Grad-CAM) to show which features drove the prediction.

### Future Work:
1.  **Explainable AI (XAI):** Integrate **Grad-CAM** to generate heatmaps, allowing clinicians to verify model focus on lesions (microaneurysms, exudates).
2.  **Cross-Corpus Validation:** Evaluate on **EyePACS**, **Messidor-2**, and **IDRiD** datasets to test robustness.
3.  **Loss Function Ablation:** Compare inverse-frequency weighting against **Focal Loss**.
4.  **Edge Deployment:** Quantize and optimize the model for deployment on mobile/edge devices (e.g., Raspberry Pi, smartphone) using **TensorFlow Lite** or **MobileNetV3**.

---

## References

Key papers that informed this work:

1.  Gulshan, V., et al. (2016). "Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs." *JAMA*.
2.  He, K., et al. (2016). "Deep Residual Learning for Image Recognition." *CVPR* (ResNet50).
3.  Buda, M., et al. (2018). "A systematic study of class imbalance in convolutional neural networks." *Neural Networks*.
4.  APTOS 2019 Blindness Detection. Kaggle. [Link](https://www.kaggle.com/c/aptos2019-blindness-detection)
5.  Lin, T. Y., et al. (2017). "Focal loss for dense object detection." *ICCV*.

A full literature survey of 15 papers is available in the `docs/` folder.

---

## Acknowledgments

We thank **Dr. Gokulakrishnan D.** for his invaluable guidance and the **School of Computing, SRM IST** for providing the necessary computational infrastructure and laboratory access.

---

## Contact

For any questions or collaboration opportunities, please contact:

- **Aditya Kumar Singh:** aa5527@srmist.edu.in
