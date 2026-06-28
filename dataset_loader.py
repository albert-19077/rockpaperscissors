import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import config


def get_data_loader(train=True):
    """
    Creates and returns a PyTorch DataLoader for either the training or validation set.
    For the training set, it applies data augmentation techniques (random rotation,
    horizontal flips, and lighting/color jitter) to help prevent overfitting.
    """
    if train:
        # Training Pipeline (With Data Augmentation to prevent model confusion)
        transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            # Randomly rotate hand by up to 15 degrees
            transforms.RandomRotation(15),
            # Randomly flip hand horizontally (handles left/right hand variations)
            transforms.RandomHorizontalFlip(p=0.5),
            # Randomly alter brightness, contrast, and saturation (handles lighting variations)
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        path = config.TRAIN_DIR
        shuffle = True
    else:
        # Validation/Inference Pipeline (Plain preprocessing, no modifications)
        transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        path = config.VAL_DIR
        shuffle = False
    if not os.path.exists(path):
        raise FileNotFoundError(f"Directory '{path}' not found.")
    dataset = datasets.ImageFolder(root=path, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0
    )

    return loader
