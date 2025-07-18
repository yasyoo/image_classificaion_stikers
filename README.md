# Image Classification Notebook: Telegram Emoji Dataset

## Overview

This notebook explores image classification techniques using a dataset of Telegram emoji images. The dataset contains approximately 100 classes with about 100 images per class, where each class is labeled by the corresponding emoji code. The notebook focuses on two main tasks:

1. Maximizing validation accuracy without using pre-trained models or image resizing
2. Maximizing validation accuracy using resizing and transfer learning (pre-trained models)

The notebook documents various experiments with different model architectures, optimization techniques, and data augmentation approaches to tackle these classification challenges.

## Key Features

- Implements custom CNN architectures (MyNet, SimpleNet) and pre-trained models (ResNet-34, ResNet-50)
- Explores different optimization strategies including learning rate adjustments and batch size modifications
- Tests various data augmentation techniques (including albumentations library)
- Implements mixed precision training to handle resource constraints
- Includes extensive error analysis and model evaluation
- Uses Weights & Biases (wandb) for experiment tracking

## Dataset Structure

The dataset follows this directory structure:
```
train/
    U+1F232/
        image1.png
        image2.png
        ...
    U+1F306/
    ...
val/
    U+1F232/
    U+1F306/
    ...
```

## Requirements

To run this notebook, you'll need:

- Python 3.10
- PyTorch
- PyTorch Lightning
- Torchmetrics
- WandB (Weights & Biases)
- Albumentations (for data augmentation experiments)
- Other standard ML/DL libraries (NumPy, Matplotlib, etc.)

## Installation

```bash
pip install wandb pytorch_lightning torchmetrics
```

## Key Findings

- Without pre-trained models or resizing, the best achieved accuracy was 0.2029 using ResNet-50
- Data augmentation initially hurt performance (best accuracy 0.136 with augmentation vs 0.2029 without)
- Mixed precision training was necessary to handle memory constraints in Google Colab
- Smaller batch sizes (16 vs 32) helped with resource limitations but slowed training

## Usage

1. Clone the repository
2. Download the dataset from the provided link
3. Set up the environment with required dependencies
4. Run the notebook cells sequentially

## Notebook Structure

1. Environment setup and installation
2. Data loading and preprocessing
3. Model architecture definitions
4. Training loops and optimization
5. Validation and evaluation
6. Experiment tracking with wandb

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on coursework from Applied Data Analysis tasks, HSE University
- Uses Telegram emoji dataset (link provided in notebook)
