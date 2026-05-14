import torchvision
import torchvision.transforms as T
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
    ])


if __name__ == '__main__':
    IMAGE_SIZE = 164
    BATCH_SIZE = 64

    print("Loading datasets...")
    train_set = PetDataset('./data', 'trainval', transform=get_transforms(IMAGE_SIZE))
    test_set  = PetDataset('./data', 'test',     transform=get_transforms(IMAGE_SIZE))

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Train images: {len(train_set)}")
    print(f"Test  images: {len(test_set)}")
    print(f"Classes:      37")
    print("Dataset loaded successfully.")
