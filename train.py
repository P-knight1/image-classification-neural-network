import torchvision
import torchvision.transforms as T
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class PetDataset(Dataset):
    def __init__(self, root, split, transform=None):
        self.base = torchvision.datasets.OxfordIIITPet(
            root=root, split=split, download=True,
            target_types=['category'])
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(image_size):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class PetNet(nn.Module):
    def __init__(self, num_classes=37):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier  = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        return self.classifier(x)


if __name__ == '__main__':
    IMAGE_SIZE = 224
    BATCH_SIZE = 64
    EPOCHS     = 30
    MAX_LR     = 1e-3
    save_path  = 'best_model.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading datasets...")
    train_set    = PetDataset('./data', 'trainval', transform=get_transforms(IMAGE_SIZE))
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model     = PetNet(num_classes=37).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=MAX_LR)

    print(f"\n{'Epoch':>5} {'Train Loss':>11} {'Train Acc':>10}")
    print("-" * 32)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimiser.zero_grad()
            logits = model(images)
            loss   = criterion(logits, labels)
            correct += (logits.argmax(1) == labels).sum().item()
            loss.backward()
            optimiser.step()
            total_loss += loss.item()
            total      += labels.size(0)

        print(f"{epoch:>5} {total_loss/len(train_loader):>11.4f} {100.*correct/total:>9.1f}%")

    torch.save(model.state_dict(), save_path)
    print(f"\nTraining complete. Model saved to: {save_path}")
