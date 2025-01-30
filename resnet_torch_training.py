import os
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from datetime import datetime
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import default_collate
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchinfo import summary
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

def get_data_loaders(train_dir, val_dir, test_dir, batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    verbose_logger.info('Transforming train dataset')
    train_dataset = ImageFolder(train_dir, transform=transform)
    verbose_logger.info('Transforming val dataset')
    val_dataset = ImageFolder(val_dir, transform=transform)

    verbose_logger.debug(f'Train dataset size: {len(train_dataset)}')
    verbose_logger.debug(f'Val dataset size: {len(val_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=default_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=default_collate)

    return train_loader, val_loader

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.backbone = models.resnet50(weights="IMAGENET1K_V2")
        for param in self.backbone.parameters():
            param.requires_grad = False

        # verbose_logger.debug(f"Initializing classifier with {num_classes} classes")
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

def train_model(model, train_loader, val_loader, num_epochs, device, save_path: str):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
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

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    verbose_logger.info(f"Num GPUs Available: {torch.cuda.device_count()}")

    train_dir = 'datasets/images/train'
    val_dir = 'datasets/images/val'
    test_dir = 'datasets/images/test'

    batch_size = 256
    num_epochs = 50
    num_classes = 8
    save_path = 'output/resnet_codebrim.pth'

    verbose_logger.info("=== Training Configuration ===")
    verbose_logger.info(f"Batch size: {batch_size}")
    verbose_logger.info(f"Number of epochs: {num_epochs}")
    verbose_logger.info(f"Number of classes: {num_classes}")
    verbose_logger.info(f"Save path: {save_path}")

    device = get_device()
    train_loader, val_loader = get_data_loaders(train_dir, val_dir, test_dir, batch_size)
    model = Classifier(num_classes)

    model_summary = summary(model, input_size=(batch_size, 3, 224, 224))
    verbose_logger.info(f"Model Summary:\n{model_summary}")

    train_model(model, train_loader, val_loader, num_epochs, device, save_path)

    verbose_logger.info("Training completed successfully!")
    progress_logger.info("\n=== Training Completed ===")

if __name__ == "__main__":
    main()