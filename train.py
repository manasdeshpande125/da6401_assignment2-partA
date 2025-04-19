# Imports
import os
os.system("pip install pytorch_lightning")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import pytorch_lightning as pl
import torch.optim as optim
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import argparse  # Missing import for argparse

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Optional: Enable DataParallel if multiple GPUs
def prepare_model_for_device(model):
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    return model.to(device)

# Dataset setup
if not os.path.exists("inaturalist_12K"):
    os.system("wget https://storage.googleapis.com/wandb_datasets/nature_12K.zip -O nature_12K.zip")
    os.system("unzip -q nature_12K.zip")
    os.system("rm nature_12K.zip")

def load_data(batch_count, data_aug='n', train_dir='inaturalist_12K/train'):
    if data_aug.lower() == 'y':
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=(0, 30)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    dataset = ImageFolder(root=train_dir, transform=transform)
    val_size = round(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(10))

    trainloader = DataLoader(train_ds, batch_size=batch_count, shuffle=True, num_workers=2)
    validationloader = DataLoader(val_ds, batch_size=batch_count, shuffle=False, num_workers=2)

    classes = dataset.classes
    return trainloader, validationloader, classes


class LightningNet(pl.LightningModule):
    def __init__(self,
                 input_shape=(3, 224, 224),
                 filters=[4, 16, 32, 64, 128],
                 kernel_size=[3, 3, 3, 3, 3],
                 activation=nn.ReLU,
                 batch_size=32,
                 use_batch_norm=True,
                 use_dropout=True,
                 dropout_rate=0.25,
                 learning_rate=1e-3,
                 num_classes=10):   #default parameters

        super().__init__()
        self.save_hyperparameters()

        self.activation = activation()
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(dropout_rate)
        self.pool = nn.MaxPool2d(2, 2)
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()

        in_channels = input_shape[0]
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        assert len(filters) == len(kernel_size), "filters and kernel_sizes must be the same length"

        for out_channels, k_size in zip(filters, kernel_size):
            self.conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=k_size, padding=k_size // 2)
            )
            if use_batch_norm:
                self.bn_layers.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels

        # Compute flattened size after conv stack
        self.flattened_size = self._get_conv_output(input_shape, batch_size)

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 84)
        self.fc2 = nn.Linear(84, num_classes)

    def _get_conv_output(self, shape, batch_size):
        dummy_input = torch.zeros(batch_size, *shape)
        dummy_output = self._forward_features(dummy_input)
        return dummy_output.view(batch_size, -1).size(1)

    def _forward_features(self, x):
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            if self.use_batch_norm:
                x = self.bn_layers[i](x)
            x = self.activation(x)
            x = self.pool(x)
            if self.use_dropout:
                x = self.dropout(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = torch.flatten(x, 1)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('train_acc', acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('val_acc', acc, prog_bar=True, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log('test_loss', loss, prog_bar=True,on_epoch=True, on_step=False)
        self.log('test_acc', acc, prog_bar=True,on_epoch=True, on_step=False)

        return {'test_loss': loss, 'test_acc': acc}

    def configure_optimizers(self, optimizer_type='adam'):
        if optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)

        elif optimizer_type == 'momentum':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)

        elif optimizer_type == 'nesterov':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, nesterov=True)

        elif optimizer_type == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        elif optimizer_type == 'nadam':
            optimizer = torch.optim.NAdam(self.parameters(), lr=self.learning_rate)

        return optimizer


def train_CNN_lightning(epochs, filter_mode, act_str, batch_count, data_aug,
                        batch_norm, drop, optimizer_name, lr_rate, drop_value,kernel_sizes,test_loader=None):

    # 1. Load dataset
    train_loader, val_loader, classes = load_data(batch_count, data_aug)
    #test_loader, test_classes = load_test_data(batch_count)

    # 2. Define filter configurations
    filter_map = {
        'all_32':   [32, 32, 32, 32, 32],
        'inc':      [16, 32, 64, 128, 256],
        'dec':      [128, 64, 32, 16, 8],
        'inc_dec':  [32, 64, 128, 64, 32],
        'dec_inc':  [128, 64, 32, 64, 128],
    }
    filters = filter_map.get(filter_mode, [32, 64, 128, 64, 32])
    if len(kernel_sizes) != len(filters):
        raise ValueError("Length of kernel_sizes must match number of filters.")

    # 3. Activation functions
    activation_map = {
        'relu': nn.ReLU,
        'sigmoid': nn.Sigmoid,
        'tanh': nn.Tanh,
        'gelu': nn.GELU,
        'silu': nn.SiLU,
        'mish': nn.Mish
    }
    activation_fn = activation_map.get(act_str.lower(), nn.ReLU)

    # 4. Optimizer mapping passed as string to LightningNet
    optimizer_name = optimizer_name.lower()  # For consistency

    # 5. Model initialization
    model = LightningNet(
        input_shape=(3, 224, 224),
        filters=filters,
        kernel_size=kernel_sizes,
        activation=activation_fn,
        batch_size=batch_count,
        use_batch_norm=(batch_norm == 'y'),
        use_dropout=(drop == 'y'),
        dropout_rate=drop_value,
        learning_rate=lr_rate,
        num_classes=len(classes)
    )

    # 6. WandB logger (Optional, comment if not using wandb)
    wandb_logger = WandbLogger(project=args.wandb_project, entity=args.wandb_entity)

    # 7. Callbacks (optional: Early stopping, Checkpointing)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, mode='min'),
        ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, filename='{epoch}-{val_loss:.2f}')
    ]

    # 8. Trainer setup
    trainer = Trainer(
        max_epochs=epochs,
        accelerator='auto',
        devices=1 if torch.cuda.is_available() else None,
        callbacks=callbacks,
        logger=wandb_logger
    )

    # 9. Train the model
    trainer.fit(model, train_loader, val_loader)
    return model


# Main entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN on iNaturalist_12K with PyTorch Lightning")

    # Default values chosen from the best-performing model
    parser.add_argument("--wandb_entity", "-we",help = "Wandb Entity used to track experiments in the Weights & Biases dashboard.", default="cs24m024")
    parser.add_argument("--wandb_project", "-wp",help="Project name used to track experiments in Weights & Biases dashboard", default="Trial")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--filter_mode", type=str, choices=["inc_dec", "dec_inc"], default="inc_dec")
    parser.add_argument("--act_str", type=str, choices=["RELU", "Mish", "GELU"], default="RELU")
    parser.add_argument("--batch_count", type=int, default=64)
    parser.add_argument("--data_aug", type=str, choices=['y', 'n'], default='y')
    parser.add_argument("--batch_norm", type=str, choices=['y', 'n'], default='y')
    parser.add_argument("--drop", type=str, choices=['y', 'n'], default='n')
    parser.add_argument("--optimizer_name", type=str, choices=["adam", "momentum", "nadam"], default="adam")
    parser.add_argument("--lr_rate", type=float, default=0.001)
    parser.add_argument("--drop_value", type=float, default=0.2)
    parser.add_argument("--kernel_sizes", nargs='+', type=int, default=[3, 3, 3, 3, 3])
    parser.add_argument("--train_dir", type=str, default="inaturalist_12K/train")

    args = parser.parse_args()
    # print(args.epochs)
    import wandb
    wandb.login()
    wandb.init(project=args.wandb_project, entity=args.wandb_entity)

    train_CNN_lightning(
        epochs=args.epochs,
        filter_mode=args.filter_mode,
        act_str=args.act_str,
        batch_count=args.batch_count,
        data_aug=args.data_aug,
        batch_norm=args.batch_norm,
        drop=args.drop,
        optimizer_name=args.optimizer_name,
        lr_rate=args.lr_rate,
        drop_value=args.drop_value,
        kernel_sizes=args.kernel_sizes
        )
