# dataset.py
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir, batch_size=16):
    """
    Loads image data from train, val, test folders.
    Applies transformations for resizing and normalization.
    """

    # Common image size for CNN
    image_size = (224, 224)

    # Transformations for images (resize, convert to tensor, normalize)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Load data from folder structure
    train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    val_data = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)
    test_data = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Class names (Normal, Pneumonia)
    classes = train_data.classes

    return train_loader, val_loader, test_loader, classes
