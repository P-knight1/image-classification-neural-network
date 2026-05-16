import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image


class PetDataset(Dataset):
    def __init__(self, root, split, transform=None):
        self.base = torchvision.datasets.OxfordIIITPet(
            root=root, split=split, download=True,
            target_types=['category', 'segmentation'])
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, (label, trimap) = self.base[idx]
        img_arr    = np.array(img)
        trimap_arr = np.array(trimap)
        if img_arr.shape[:2] == trimap_arr.shape[:2]:
            img_arr[trimap_arr == 2] = [124, 116, 104]
        img = Image.fromarray(img_arr)
        if self.transform:
            img = self.transform(img)
        return img, label


def get_train_transforms(image_size):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def get_eval_transforms(image_size):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv_path = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        return F.relu(self.conv_path(x) + self.skip(x))


class PetNet(nn.Module):
    def __init__(self, num_classes=37):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.stage1 = ConvBlock(32,  64)
        self.stage2 = ConvBlock(64,  128, stride=2)
        self.stage3 = ConvBlock(128, 256, stride=2)
        self.stage4 = ConvBlock(256, 512, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier  = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        return self.classifier(x)


def eval_epoch(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
            total   += labels.size(0)
    return 100.0 * correct / total


if __name__ == '__main__':
    IMAGE_SIZE = 224
    BATCH_SIZE = 64
    EPOCHS     = 30
    MAX_LR     = 1e-3
    ETA_MIN    = 1e-6
    save_path  = 'best_model.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading datasets...")
    train_set = PetDataset('./data', 'trainval', transform=get_train_transforms(IMAGE_SIZE))
    test_set  = PetDataset('./data', 'test',     transform=get_eval_transforms(IMAGE_SIZE))

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    model     = PetNet(num_classes=37).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=EPOCHS * len(train_loader), eta_min=ETA_MIN)

    best_test_acc = 0.0

    print(f"\n{'Epoch':>5} {'Train Loss':>11} {'Train Acc':>10} {'Test Acc':>9}")
    print("-" * 42)

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
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            scheduler.step()
            total_loss += loss.item()
            total      += labels.size(0)

        if epoch % 5 == 0 or epoch == EPOCHS:
            test_acc = eval_epoch(model, test_loader, device)
            print(f"{epoch:>5} {total_loss/len(train_loader):>11.4f} {100.*correct/total:>9.1f}% {test_acc:>8.1f}%")
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(model.state_dict(), save_path)
        else:
            print(f"{epoch:>5} {total_loss/len(train_loader):>11.4f} {100.*correct/total:>9.1f}%")

    print(f"\nBest test accuracy: {best_test_acc:.1f}%")
    print(f"Model saved to: {save_path}")
