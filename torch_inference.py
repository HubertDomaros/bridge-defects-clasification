import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets.folder import ImageFolder
from torchvision import models, transforms

import pandas as pd
from tqdm import tqdm

import utils.constants as c

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.backbone = models.resnet50(weights="IMAGENET1K_V2")
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


device = 'cpu'
src_dir = r'D:\0-Code\PG\2_sem\0_Dyplom\bridge-defects-clasification\datasets\images\test'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

dataset = ImageFolder(src_dir, transform=transform)

dataloader = DataLoader(dataset, batch_size=32)


def load_model(pth_path):
    model = Classifier(8)
    state_dict = torch.load(pth_path, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model

my_model = load_model('resnet_codebrim.pth')


def predict(model: nn.Module, test_dataloader: DataLoader) -> pd.DataFrame:
    results = {
        'img': [],
        'pred_labels': []
    }
    with torch.inference_mode():
        for images, labels in tqdm(test_dataloader):
            images = images.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            results['img'].extend(os.path.basename(test_dataloader.dataset.samples[i][0])
                                        for i in range(len(predictions)))
            results['pred_labels'].extend(predictions.cpu().numpy())
    return pd.DataFrame(results)

if __name__ == '__main__':
    predicted_df = predict(my_model, dataloader)
    actual = pd.read_json('datasets/labels/test.json').drop(['Background', 'Crack', 'Efflorescence',
       'Spallation_ExposedBars_CorrosionStain'], axis=1)
    actual.rename(columns={'combination_id': 'ground_truth'}, inplace=True)
    print(actual.columns)

    print(actual.merge(predicted_df, on='img'))


