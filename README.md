# Ultrasound Image Segmentation Using U-Net

A deep learning project focused on fetal ultrasound image segmentation using a U-Net convolutional neural network (CNN) implemented with PyTorch.

## Overview

This project explores the use of deep learning techniques to perform semantic segmentation of fetal ultrasound images.

The main objective is to develop a model capable of identifying and segmenting relevant anatomical structures in ultrasound images through pixel-level classification.

The project follows an iterative development approach, covering dataset preparation, model implementation, training, and evaluation.

## Dataset and Image Acquisition

### Image Acquisition

The dataset consists of images of a fetus-shaped phantom submerged in another phantom with a different concentration.

### Image Annotation

Each ultrasound image is associated with a corresponding binary segmentation mask.

The masks represent the anatomical structures of interest and serve as ground-truth references during model training.

## Methodology

The current implementation is based on the following workflow:

1. **Dataset preparation:** Load ultrasound images and their corresponding binary segmentation masks.
2. **Preprocessing:** Convert images to grayscale, resize them to 256 × 256 pixels, and normalize pixel intensities.
3. **Dataset splitting:** Divide the dataset into training (80%) and validation (20%) subsets.
4. **Model architecture:** Implement a U-Net architecture with an encoder-decoder structure and skip connections.
5. **Training:** Optimize model parameters using the Adam optimizer and a combined loss function.
6. **Evaluation:** Monitor training loss and validation Dice Score.

## Model Architecture

The network follows a U-Net architecture consisting of:

- Four encoder blocks.
- A bottleneck layer.
- Four decoder blocks.
- Skip connections between corresponding encoder and decoder stages.
- A final convolutional layer for binary segmentation.

Each convolutional block contains two convolutional layers with ReLU activation functions.

## Technologies

- Python 2.14.1
- PyTorch
- NumPy
- Matplotlib
- Pillow

## Repository Structure

```text
FetalUS-Segmentation/
│
├── FRUS
│
├── Masks
|
├── Week1/
│   ├── segmentation.py
│   └── ... (models)
│
├── Week2/
│   ├── segmentation.ipynb
│   └── ... (models)
│
└── README.md
```

*Note: The directory structure above is illustrative and should be updated to reflect the actual repository contents.*

## Training Configuration

| Parameter | Value |
|---|---|
| Input image size | 256 × 256 |
| Batch size | 2 |
| Epochs | 30 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | BCE + Dice Loss |
| Evaluation metric | Dice Score |
| Training samples | 210 |
| Validation samples | 52 |

## Current Results

The current training implementation achieved a best validation Dice Score of approximately **0.4423**.

Model performance is still under evaluation, and further research and improvements are needed to obtain more consistent segmentation results.

## Project Status

**In development.**

Future work includes:

- Improving segmentation performance.
- Evaluating different preprocessing techniques.
- Experimenting with data augmentation.
- Investigating model generalization.
- Comparing predictions against ground-truth segmentation masks.

## Disclaimer

This project is intended for educational and research purposes. It is not designed or validated for clinical use or medical diagnosis.

