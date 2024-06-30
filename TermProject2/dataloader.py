import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset


class CustomDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = self.data_frame.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label


def load_data(train_csv, val_csv, test_csv, img_dir, batch_size=32, base_transform=None, augment_transform=None):
    base_dataset = CustomDataset(train_csv, img_dir, transform=base_transform)
    if augment_transform:
        augmented_dataset = CustomDataset(train_csv, img_dir, transform=augment_transform)
        train_dataset = ConcatDataset([base_dataset, augmented_dataset])
    else:
        train_dataset = base_dataset

    val_dataset = CustomDataset(val_csv, img_dir, transform=base_transform)
    test_dataset = CustomDataset(test_csv, img_dir, transform=base_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
