# data_loader.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(data_dir, batch_size=16):
    """
    Loads images from 'data_dir' and returns train/test DataLoaders.
    Expected folder structure:
    data/
      ├── train/
      │   ├── NORMAL/
      │   └── PNEUMONIA/
      └── test/
          ├── NORMAL/
          └── PNEUMONIA/
    """

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    train_data = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
    test_data = datasets.ImageFolder(root=f"{data_dir}/test", transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
