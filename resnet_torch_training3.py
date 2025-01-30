import os
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from datetime import datetime
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchinfo import summary
from PIL import Image

import shutil
import glob
import re


# from lightning_sdk import Studio

# Set up logging
def setup_logging():
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)

    # Function to get the next available number for a log type
    def get_next_number(prefix):
        existing_files = glob.glob(f'logs/{prefix}*.log')
        if not existing_files:
            return 1
        numbers = []
        for file in existing_files:
            try:
                # Extract number from filename (e.g., 'verbose1.log' -> 1)
                num = int(re.search(rf'{prefix}(\d+)\.log$', file).group(1))
                numbers.append(num)
            except (AttributeError, ValueError):
                continue
        return max(numbers, default=0) + 1

    # Get next available numbers for each log type
    verbose_num = get_next_number('verbose')
    progress_num = get_next_number('training_progress')

    # Set up verbose logger
    verbose_logger = logging.getLogger('verbose')
    verbose_logger.setLevel(logging.DEBUG)
    verbose_handler = logging.FileHandler(f'logs/verbose{verbose_num}.log')
    verbose_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    verbose_logger.addHandler(verbose_handler)

    # Set up training progress logger
    progress_logger = logging.getLogger('progress')
    progress_logger.setLevel(logging.INFO)
    progress_handler = logging.FileHandler(f'logs/training_progress{progress_num}.log')
    progress_handler.setFormatter(logging.Formatter('%(message)s'))
    progress_logger.addHandler(progress_handler)

    # Add console handler only for progress logger to keep console output clean
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    progress_logger.addHandler(console_handler)

    # Add minimal console output for critical verbose logs
    minimal_console_handler = logging.StreamHandler()
    minimal_console_handler.setFormatter(logging.Formatter('%(message)s'))
    minimal_console_handler.setLevel(logging.WARNING)  # Only show warnings and errors
    verbose_logger.addHandler(minimal_console_handler)

    return verbose_logger, progress_logger


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    verbose_logger.debug(f"Using device: {device}")
    return device


def transform_and_save_dataset(input_dir, output_dir, transform, expected_classes=8, batch_size=32):
    # Check if tensors already exist
    if os.path.exists(output_dir):
        if len(os.listdir(output_dir)) == expected_classes:
            verbose_logger.info(f"Tensors already exist in {output_dir}, skipping transformation")
            return
        else:
            verbose_logger.warning(f"Found incomplete tensor directory, recreating tensors")
            shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    dataset = ImageFolder(input_dir)

    if len(dataset.classes) != expected_classes:
        raise ValueError(f"Expected {expected_classes} classes, found {len(dataset.classes)}")

    class_counts = {cls: 0 for cls in dataset.classes}

    for class_name in dataset.classes:
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

    batches = [dataset.samples[i:i + batch_size] for i in range(0, len(dataset.samples), batch_size)]

    for batch in tqdm.tqdm(batches, desc=f"Processing {os.path.basename(input_dir)}"):
        for img_path, label in batch:
            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    tensor = transform(img)

                class_name = dataset.classes[label]
                class_counts[class_name] += 1

                filename = os.path.basename(img_path).split('.')[0] + '.pt'
                save_path = os.path.join(output_dir, class_name, filename)

                torch.save(tensor, save_path)
            except Exception as e:
                verbose_logger.error(f"Error processing {img_path}: {e}")

    verbose_logger.info("Class distribution:")
    for cls, count in class_counts.items():
        verbose_logger.info(f"{cls}: {count} images")


class TensorDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Preload all file paths at init
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            self.samples.extend([
                (os.path.join(class_dir, file), class_idx)
                for file in os.listdir(class_dir)
                if file.endswith('.pt')
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        tensor = torch.load(path)
        return tensor, label


def get_data_loaders(train_dir, val_dir, test_dir, batch_size=16):
    train_dataset = TensorDataset(train_dir)
    val_dataset = TensorDataset(val_dir)
    test_dataset = TensorDataset(test_dir)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                   num_workers=4, pin_memory=True, persistent_workers=True,
                   prefetch_factor=2),
        DataLoader(val_dataset, batch_size=batch_size,
                   num_workers=4, pin_memory=True, persistent_workers=True),
        DataLoader(test_dataset, batch_size=batch_size,
                   num_workers=4, pin_memory=True, persistent_workers=True)
    )


class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.backbone = models.resnet50(weights="IMAGENET1K_V2")

        self.backbone.layer2.add_module('dropout', nn.Dropout(0.2))
        self.backbone.layer3.add_module('dropout', nn.Dropout(0.3))
        self.backbone.layer3.add_module('dropout', nn.Dropout(0.4))
        self.backbone.use_checkpointing = True
        self.backbone.layer4.use_checkpointing = True

        # Memory cleanup hook
        self.backbone.layer4.register_forward_hook(lambda *args: torch.cuda.empty_cache())

        for param in self.backbone.parameters():
            param.requires_grad = False

        verbose_logger.debug(f"Initializing classifier with {num_classes} classes")
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.backbone.fc.requires_grad_(True)

    def forward(self, x):
        if self.backbone.use_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(self.backbone, x, use_reentrant=False)
        return self.backbone(x)


def train_model(model, train_loader, val_loader, num_epochs, device, save_path: str):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=0.0005)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)
    scaler = torch.cuda.amp.GradScaler()

    model.to(device)
    best_loss = float('inf')
    accumulation_steps = 4

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        optimizer.zero_grad(set_to_none=True)

        print()
        print(f'Epoch {epoch}/{num_epochs}')
        for batch_idx, (inputs, labels) in enumerate(tqdm.tqdm(train_loader, 'train: ')):
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels) / accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item() * inputs.size(0) * accumulation_steps
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # Calculate epoch-level metrics
        train_loss /= total
        train_accuracy = 100.0 * correct / total  # Calculate train accuracy here

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad(), torch.cuda.amp.autocast():
            for inputs, labels in tqdm.tqdm(val_loader, 'val:'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss /= total
        val_accuracy = 100.0 * correct / total

        # Log epoch results with both accuracies
        epoch_summary = (
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}% - "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%"
        )
        progress_logger.info(epoch_summary)
        verbose_logger.info(f"Epoch {epoch + 1} completed: {epoch_summary}")

        scheduler.step(val_loss)

        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        verbose_logger.debug(f"Current learning rate: {current_lr}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)
            progress_logger.info(f"New best model saved! (Val Loss: {val_loss:.4f})")
            verbose_logger.info(f"Model saved with validation loss: {val_loss}")
            verbose_logger.info(f"Model saved with val_loss: {val_loss:.4f}")
            verbose_logger.info(f"Model path: {save_path}")


def main():
    global verbose_logger, progress_logger
    verbose_logger, progress_logger = setup_logging()

    # Add memory pinning for faster data transfer
    torch.backends.cudnn.benchmark = True  # Optimize CUDA operations

    # Optional: Enable TF32 for faster training on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Define directories
    base_dir = 'datasets'
    img_dirs = {
        'train': os.path.join(base_dir, 'images/train'),
        'val': os.path.join(base_dir, 'images/val'),
        'test': os.path.join(base_dir, 'images/test')
    }
    tensor_dirs = {
        'train': os.path.join(base_dir, 'tensors/train'),
        'val': os.path.join(base_dir, 'tensors/val'),
        'test': os.path.join(base_dir, 'tensors/test')
    }

    # Transform and save images as tensors
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for split, img_dir in img_dirs.items():
        verbose_logger.info(f'Processing {split} dataset')
        transform_and_save_dataset(img_dir, tensor_dirs[split], transform)

    # Rest of your training code remains the same
    batch_size = 64
    num_epochs = 50
    num_classes = 8
    save_path = 'resnet_codebrim.pth'
    device = get_device()

    train_loader, val_loader, test_loader = get_data_loaders(
        tensor_dirs['train'], tensor_dirs['val'], tensor_dirs['test'],
        batch_size
    )

    model = Classifier(num_classes)
    model_summary = summary(model, input_size=(batch_size, 3, 224, 224))
    verbose_logger.info(f"Model Summary:\n{model_summary}")

    train_model(model, train_loader, val_loader, num_epochs, device, save_path)

    verbose_logger.info("Training completed successfully!")
    progress_logger.info("\n=== Training Completed ===")


if __name__ == "__main__":
    main()
