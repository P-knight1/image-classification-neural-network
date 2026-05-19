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
        T.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
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


def mixup(x, y, alpha=0.4):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam



def mixed_loss(criterion, logits, ya, yb, lam):
    return lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)

# Skip connection design inspired by He et al. (2015), "Deep Residual Learning for Image Recognition"
# https://arxiv.org/abs/1512.03385
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv_path = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),  # first conv
            nn.BatchNorm2d(out_ch),                                               # normalise
            nn.ReLU(inplace=True),                                                # activation
            nn.Conv2d(out_ch, out_ch, 3, padding=2, dilation=2, bias=False),     # dilated conv
            nn.BatchNorm2d(out_ch),                                               # normalise
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
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),  #224×224 -> 112×112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),                  #112×112 -> 56×56
        )

        self.stage1 = nn.Sequential(ConvBlock(32,  64),  ConvBlock(64,  64))            #56×56×64
        self.stage2 = nn.Sequential(ConvBlock(64,  128, stride=2), ConvBlock(128, 128)) #28×28×128
        self.stage3 = nn.Sequential(ConvBlock(128, 256, stride=2), ConvBlock(256, 256)) #14×14×256
        self.stage4 = ConvBlock(256, 512, stride=2)                                     #7×7×512

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  #7×7×512 -> 512

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),  #512 -> 37 class scores
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


def train_epoch(model, loader, criterion, optimiser, scheduler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if np.random.rand() < 0.50:
            images, ya, yb, lam = mixup(images, labels)
            optimiser.zero_grad()
            logits = model(images)
            loss   = mixed_loss(criterion, logits, ya, yb, lam)
            dominant = ya if lam >= 0.5 else yb
            correct += (logits.argmax(1) == dominant).sum().item()
        else:
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

    return total_loss / len(loader), 100.0 * correct / total


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

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\nLoading datasets...")
    train_set    = PetDataset('./data', 'trainval', transform=get_train_transforms(IMAGE_SIZE))
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)

    model     = PetNet(num_classes=37).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimiser = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=EPOCHS * len(train_loader), eta_min=ETA_MIN)
    
    print(f"Device:{device}")

    print(f"\n{'Epoch':>5} {'Train Loss':>11} {'Train Acc':>10}")
    print("-" * 32)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimiser, scheduler, device)
        print(f"{epoch:>5} {train_loss:>11.4f} {train_acc:>9.1f}%")

    torch.save(model.state_dict(), save_path)
    print(f"\nTraining complete. Model saved to: {save_path}")
