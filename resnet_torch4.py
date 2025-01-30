from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm
import logging

logging.basicConfig(filename='resnet_torch4_1.log', level=logging.INFO)
logger = logging.getLogger(__name__)

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.backbone = models.resnet50()
        self.backbone.layer1.add_module('dropout1', nn.Dropout(0.1))
        self.backbone.layer2.add_module('dropout2', nn.Dropout(0.2))
        self.backbone.layer3.add_module('dropout3', nn.Dropout(0.3))
        self.backbone.layer4.add_module('dropout4', nn.Dropout(0.4))

        # Replace the final FC layer before freezing backbone
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, 8)

        # Freeze all backbone layers except FC
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:  # Don't freeze FC layer
                param.requires_grad = False

    def forward(self, x):
        if self.training:
            return checkpoint(self.backbone, x, use_reentrant=False)
        return self.backbone(x)


class MyDataset(ImageFolder):
    def __init__(self, root_dir, transform=None, target_transform=None, Q1=200):
        super().__init__(root_dir, transform=transform, target_transform=target_transform)

        self._Q1 = Q1
        self.root_dir = root_dir
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if target in self.underrepresented_classes:
            sample = self.underrepresented_class_transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    @property
    def Q1(self):
        if self._Q1 is not None:
            #print('q1 is not none')
            return self._Q1
        
        #return np.percentile(list(self.class_counts.values()),25)
        return 300

    @property
    def class_counts(self):
        return Counter([label for _, label in self.samples])

    @property
    def underrepresented_classes(self):
        out = []
        for c, count in self.class_counts.items():
            if count < self.Q1:
                out.append(c)
        #print('Underrepresented classes:', c)
        #print('class counts: ', self.class_counts)
        return out

    @property
    def underrepresented_class_transform(self):
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            # transforms.AutoAugment()
            transforms.A
            
        ])


def get_data_loaders(train_dir, val_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = MyDataset(train_dir, transform=transform)
    val_dataset = MyDataset(val_dir, transform=transform)

    return (DataLoader(train_dataset, batch_size=batch_size, num_workers=4,
                       prefetch_factor=2, pin_memory=True),
            DataLoader(val_dataset, batch_size=batch_size, num_workers=4,
                       prefetch_factor=2, pin_memory=True))


def train_model(model, train_loader: DataLoader, val_loader, epochs, device, save_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), weight_decay=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    scaler = torch.amp.GradScaler()

    # Debug: Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}")

    model = model.to(device)
    best_loss = float('inf')
    accumulation_steps = 4
    
    ds: MyDataset = train_loader.dataset
    print(ds.underrepresented_classes)
    
    print(train_loader.dataset)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        optimizer.zero_grad()
        

        print(f'Epoch {epoch}/{epochs}')
        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, 'train')):
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.amp.autocast(device_type=str(device)):
                outputs = model(inputs)
                loss = criterion(outputs, labels) / accumulation_steps

            # Debug: Print loss value
            if batch_idx == 0:
                print(f"Loss value: {loss.item()}")

                # Check if outputs require grad
                print(f"Outputs require grad: {outputs.requires_grad}")

                # Print gradient status of parameters
                for name, param in model.named_parameters():
                    print(f"{name}: requires_grad = {param.requires_grad}")

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)

                # Debug: Print gradients after backward
                if batch_idx == 0:
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            print(f"{name} grad: {param.grad is not None}")

                optimizer.step()
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * inputs.size(0) * accumulation_steps
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            #Epoch level metrics
            train_loss /= total
            train_accuracy = 100 * correct / total

            model.eval()
            val_loss = 0
            correct = 0
            total = 0

            with torch.no_grad(), torch.amp.autocast(str(device)):
                for inputs, labels in tqdm(val_loader, 'val: '):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                val_loss /= total
                val_accuracy = 100 * correct / total

                print(f'train loss: {train_loss: .4f}, val loss: {val_loss: .4f},'
                      f' train accuracy: {train_accuracy: .4f} val accuracy: {val_accuracy}')

                logger.info(f'{epoch} ,{train_loss}, {val_loss}, {train_accuracy}, {val_accuracy}')

                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), save_path)


def main():
    batch_size = 64
    epochs = 100
    save_path = 'resnet_torch4_1.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: {}'.format(device))

    train_loader, val_loader = get_data_loaders(
        'datasets/images/train',
        'datasets/images/val',
        batch_size=batch_size
    )

    train_model(MyNet(), train_loader, val_loader, epochs, device, save_path)

if __name__ == '__main__':
    main()