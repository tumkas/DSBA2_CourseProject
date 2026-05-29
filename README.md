# DSBA2_CourseProject

**Diagnosis of pathologies of the cardiorespiratory system and main arteries based on the analysis of sound data**

## Overview
Cardiovascular and chronic respiratory diseases are major global health issues. Standard diagnostics rely heavily on expensive equipment and specialized personnel. This project aims to build an automated, "lab-in-a-pocket" diagnostic system using simple sound recordings (like PCGs and lung sounds) to detect cardiorespiratory pathologies. 

By combining Deep Learning architectures with advanced signal processing and eXplainable AI (LIME), this tool provides accurate and interpretable preliminary diagnostics.

## Features & Methodology

### 1. Data Processing
* **Dataset**: [PhysioNet/Computing in Cardiology (CinC) Challenge 2016 dataset](https://physionet.org/content/challenge-2016/1.0.0/).
* **Signal Cleaning**: Applied a 4th-order Butterworth digital bandpass filter (20–400 Hz) to remove environmental noise and stethoscope friction.
* **Feature Extraction**: Converted 1D sound waves into 2D Mel-spectrograms (128 Mel bands, capped at 800 Hz) to create high-contrast visual representations of heart sounds.

### 2. Neural Network Architectures
We compared generic audio models against visual transfer-learning models:
* **Custom CNN**: Baseline architecture (Accuracy: 84%, but suffered from severe class imbalance).
* **YAMNet (Audio-Native)**: Fine-tuned from Google's AudioSet (Accuracy: 78%). Struggled to isolate precise medical frequencies against general background sounds.
* **ResNet18 (Visual Transfer Learning) - Best Model**: Pre-trained on ImageNet, adapted for 1-channel spectrograms. Combined with **Focal Loss** (to handle class imbalance) and **SpecAugment** (data augmentation technique).
  * **Overall Accuracy**: 89%
  * **Recall (Abnormal)**: 92%
  * **F1-Score**: 93%

### 3. Interpretability (LIME)
To solve the "black box" problem of neural networks, we integrated the **LIME** (Local Interpretable Model-agnostic Explanations) framework. LIME generates visual heatmaps over the Mel-spectrograms to highlight the exact time and frequency areas driving the model's decision. This ensures the model relies on real physiological biomarkers (e.g., systolic murmurs) rather than memorizing hospital background noise.

### 4. Track B: Chaos Theory & Mathematical Modeling (Future Work)
The next phase of the project investigates non-linear mathematical modeling of bio-signals. Since the cardiorespiratory system is chaotic, we are implementing Phase Space Reconstruction, calculating chaotic invariants (like the Largest Lyapunov Exponent), and using clustering algorithms to develop an alternative, math-based diagnostic approach.

## Installation

Ensure you have Python 3.8+ installed. Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Usage
* **Data Preparation**: Run `prepare_data.py` to filter raw audio and generate spectrograms.
* **Training Models**: Use `train.py` (for CNN/ResNet18) or `train_yamnet.py` (for YAMNet).
* **Explaining Predictions**: Run `lime_explainer.py` to generate visual interpretability heatmaps for specific spectrograms.
* **Inference**: Use `predict.py` to classify a single audio `.wav` file.

---
*Fulfilled by George Kuznetsov. Developed as part of the DSBA Research Project, HSE University.*
