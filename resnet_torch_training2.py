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

# from lightning_sdk import Studio

# Set up logging
def setup_logging():
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    # Generate timestamp for log files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Set up verbose logger
    verbose_logger = logging.getLogger('verbose')
    verbose_logger.setLevel(logging.DEBUG)
    verbose_handler = logging.FileHandler(f'logs/verbose_{timestamp}.log')
    verbose_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    verbose_logger.addHandler(verbose_handler)

    # Set up training progress logger
    progress_logger = logging.getLogger('progress')
    progress_logger.setLevel(logging.INFO)
    progress_handler = logging.FileHandler(f'logs/training_progress_{timestamp}.log')
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


def transform_and_save_dataset(input_dir, output_dir, transform, expected_classes=5):
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

    for img_path, label in tqdm.tqdm(dataset.samples, desc=f"Processing {os.path.basename(input_dir)}"):
        try:
            img = Image.open(img_path).convert('RGB')
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


def get_data_loaders(train_dir, val_dir, batch_size=16):
    # Modified to load pre-transformed tensors
    class TensorDataset(Dataset):
        def __init__(self, root_dir):
            self.samples = []
            self.classes = sorted(os.listdir(root_dir))
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

            for class_name in self.classes:
                class_dir = os.path.join(root_dir, class_name)
                for file in os.listdir(class_dir):
                    if file.endswith('.pt'):
                        self.samples.append((
                            os.path.join(class_dir, file),
                            self.class_to_idx[class_name]
                        ))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, label = self.samples[idx]
            tensor = torch.load(path)
            return tensor, label

    train_dataset = TensorDataset(train_dir)
    val_dataset = TensorDataset(val_dir)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size),
    )

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.backbone = models.resnet50(weights="IMAGENET1K_V2")

        self.backbone.layer1.add_module('dropout', nn.Dropout(0.2))
        self.backbone.layer2.add_module('dropout', nn.Dropout(0.2))
        self.backbone.layer3.add_module('dropout', nn.Dropout(0.2))
        self.backbone.layer4.add_module('dropout', nn.Dropout(0.2))

        for param in self.backbone.parameters():
            param.requires_grad = False

        verbose_logger.debug(f"Initializing classifier with {num_classes} classes")
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

def train_model(model, train_loader, val_loader, num_epochs, device, save_path: str):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)

    model.to(device)
    verbose_logger.debug(f"Model moved to device: {device}")

    best_loss = float('inf')

    # Log training start
    progress_logger.info("\n=== Training Started ===")
    progress_logger.info(f"Total epochs: {num_epochs}")
    progress_logger.info(f"Save path: {save_path}")

    for epoch in range(num_epochs):
        # Training phase
        verbose_logger.info(f'Epoch {epoch + 1}/{num_epochs} started')
        progress_logger.info(f'\nEpoch {epoch + 1}/{num_epochs}:')
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(tqdm.tqdm(train_loader)):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Log batch progress
            if batch_idx % 10 == 0:  # Log every 10 batches
                verbose_logger.debug(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

        train_loss /= total
        train_accuracy = 100.0 * correct / total

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        verbose_logger.info('Starting validation phase')
        with torch.no_grad():
            for inputs, labels in tqdm.tqdm(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss /= total
        val_accuracy = 100.0 * correct / total

        # Log epoch results
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

    # Define directories
    base_dir = 'datasets2'
    img_dirs = {
        'train': os.path.join(base_dir, 'images/train'),
        'val': os.path.join(base_dir, 'images/val'),
    }
    tensor_dirs = {
        'train': os.path.join(base_dir, 'tensors2/train'),
        'val': os.path.join(base_dir, 'tensors2/val'),
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
    batch_size = 128
    num_epochs = 50
    num_classes = 5
    save_path = 'output/resnet_torch_training_5_classes.pth'
    device = get_device()

    train_loader, val_loader, = get_data_loaders(
        tensor_dirs['train'], tensor_dirs['val'],
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