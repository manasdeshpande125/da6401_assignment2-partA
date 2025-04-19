# da6401_assignment2-partA

# Image Classification on iNaturalist_12K Dataset using PyTorch Lightning

This project performs image classification on the [iNaturalist_12K] dataset using Convolutional Neural Networks (CNNs) with various hyperparameters and visualizes predictions.

## Directory Structure
```
inaturalist_12K/
├── train/
└── val/
```

## Requirements

Install the required dependencies:

```bash
pip install torch torchvision pytorch-lightning matplotlib wandb
```

##  Model Training & Evaluation

We define and train three different CNN models using the `train_CNN_lightning` function, each with distinct configurations to compare performance.

### Best Model Configurations

| Model | Filter Mode | Activation | Batch Size | Dropout | Optimizer | LR     | Val Accuracy |
|-------|-------------|------------|------------|---------|-----------|--------|--------------|
| 1     | inc_dec     | ReLU       | 64         | No      | Adam      | 0.001  | 0.4095       |
| 2     | dec_inc     | Mish       | 64         | No      | Momentum  | 0.0001 | 0.4005       |
| 3     | dec_inc     | GELU       | 32         | No      | Nadam     | 0.0001 | 0.3960       |

Each model is trained for 13–20 epochs and evaluated on the validation set. The predictions are visualized using `matplotlib`.

## Functions Overview

### `load_test_data(test_dir, batch_size)`
Loads validation data from the iNaturalist dataset with resizing, normalization, and batching.

### `show_predictions(model, dataloader, classes)`
Displays a grid of predictions alongside ground truth labels using `matplotlib`.

### `train_CNN_lightning(...)`
Trains a CNN model based on dynamic architecture and optimizer choices using PyTorch Lightning.

## Sample Visualization

Each trained model produces a grid of predicted vs actual labels:

- Green ✅: Correct prediction
- Red ❌: Incorrect prediction

## Usage

In a Python script or Jupyter Notebook:

```python
# Load validation data
test_loader, test_classes = load_test_data()

# Train and visualize model predictions
model = train_CNN_lightning(
    epochs=20,
    filter_mode="inc_dec",
    act_str="RELU",
    batch_count=64,
    data_aug='y',
    batch_norm='y',
    drop='n',
    optimizer_name="adam",
    lr_rate=0.001,
    drop_value=0.2,
    kernel_sizes=[3, 3, 3, 3, 3],
    test_loader=test_loader
)

show_predictions(model, test_loader, test_classes)
```


