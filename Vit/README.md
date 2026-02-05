# Vision Transformer (ViT) Implementation

A PyTorch implementation of the Vision Transformer (ViT) architecture from scratch, replicating the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Dosovitskiy et al.

## Overview

This project implements a Vision Transformer for image classification, demonstrating how to build and train a ViT model from the ground up. The implementation includes:

- **Custom ViT Architecture**: Building the complete ViT architecture including patch embeddings, positional embeddings, transformer encoder blocks, and classification head
- **Food Classification**: Training on a custom dataset (pizza, steak, sushi) to classify food images
- **Modular Code Structure**: Well-organized, reusable code components for data handling, training, and model building
- **Pre-trained Models**: Integration with PyTorch's pre-trained ViT models for transfer learning

## Architecture

The Vision Transformer architecture consists of:

1. **Patch Embedding Layer**: Splits images into fixed-size patches (16x16) and converts them into embeddings
2. **Positional Embeddings**: Adds learnable positional information to patch embeddings
3. **Class Token**: A learnable classification token prepended to the sequence
4. **Transformer Encoder**: Stack of transformer encoder blocks with multi-head self-attention
5. **MLP Head**: Classification head for final predictions

### Key Components

- **Patch Size**: 16x16 pixels
- **Image Size**: 224x224 pixels
- **Embedding Dimension**: 768 (ViT-Base)
- **Number of Transformer Layers**: 12 (configurable)
- **Number of Attention Heads**: 12

## Project Structure

```
Vit/
├── Paper Replication.ipynb    # Main notebook with ViT implementation and experiments
├── going_modular/             # Modular Python code
│   ├── data_setup.py          # Data loading and preprocessing utilities
│   ├── engine.py              # Training and evaluation engine
│   ├── model_builder.py       # Model architecture definitions
│   ├── train.py               # Training script
│   ├── predictions.py         # Prediction utilities
│   └── utils.py               # Helper utilities
├── helper_functions.py        # General helper functions
├── data/                      # Dataset directory
│   └── pizza_steak_sushi/     # Food classification dataset
│       ├── train/             # Training images
│       └── test/              # Test images
└── models/                    # Saved model checkpoints
```

## Getting Started

### Prerequisites

- Python 3.11+
- PyTorch 2.10.0+
- torchvision 0.25.0+

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Vit
```

2. Install dependencies using `uv` (recommended):
```bash
uv sync
```

Or install manually:
```bash
pip install torch torchvision torchinfo matplotlib numpy requests tqdm ipykernel
```

### Dataset

The project uses a food classification dataset with three classes:
- 🍕 Pizza
- 🥩 Steak
- 🍣 Sushi

The dataset is automatically downloaded when running the notebook, or you can manually place your data in the `data/pizza_steak_sushi/` directory with `train/` and `test/` subdirectories.

## Usage

### Running the Notebook

Open and run `Paper Replication.ipynb` to:
1. Explore the ViT architecture implementation
2. Train a custom ViT model from scratch
3. Fine-tune a pre-trained ViT model
4. Evaluate model performance
5. Make predictions on new images

### Training from Scratch

The notebook includes step-by-step implementation of:
- Patch embedding creation
- Transformer encoder blocks
- Complete ViT architecture assembly
- Training loop with loss tracking
- Model evaluation and visualization

### Using Pre-trained Models

The project also demonstrates how to use PyTorch's pre-trained ViT models:

```python
from torchvision import models

# Load pre-trained ViT-Base
pretrained_vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

# Modify classification head for your number of classes
pretrained_vit.heads = nn.Linear(in_features=768, out_features=num_classes)
```

### Modular Training Script

You can also use the modular training script:

```python
from going_modular import train

# Training is configured in train.py
# Modify hyperparameters as needed:
# - NUM_EPOCHS
# - BATCH_SIZE
# - LEARNING_RATE
```

## 🔧 Key Features

- **From Scratch Implementation**: Complete ViT architecture built from basic PyTorch components
- **Educational**: Step-by-step breakdown of each component with visualizations
- **Modular Design**: Reusable code components for easy experimentation
- **Transfer Learning**: Integration with pre-trained models for faster training
- **Visualization Tools**: Functions for plotting loss curves, predictions, and model architecture

## Model Architecture Summary

The custom ViT model includes:
- **Total Parameters**: ~152K (for 3-class classification)
- **Input Shape**: (batch_size, 3, 224, 224)
- **Output Shape**: (batch_size, num_classes)
- **Transformer Layers**: 12 encoder blocks
- **Embedding Dimension**: 768

## Results

The notebook demonstrates:
- Training custom ViT models on the food classification dataset
- Comparison between from-scratch and pre-trained models
- Visualization of training progress and model predictions
- Model checkpoint saving and loading

## References

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) - Original ViT paper
- [PyTorch Vision Transformer Documentation](https://pytorch.org/vision/stable/models.html#vision-transformer-vit)

## Development

### Code Organization

- **`going_modular/`**: Production-ready modular code
- **`Paper Replication.ipynb`**: Research and experimentation notebook
- **`helper_functions.py`**: Utility functions for visualization and data handling

### Contributing

Feel free to experiment with:
- Different patch sizes
- Various transformer configurations
- Alternative datasets
- Training strategies and hyperparameters

## License

This project is for educational purposes, implementing the Vision Transformer architecture as described in the original research paper.

## Acknowledgments

- Original ViT paper authors: Dosovitskiy et al.
- PyTorch and torchvision teams for excellent deep learning frameworks
- Food dataset from PyTorch Deep Learning course materials

