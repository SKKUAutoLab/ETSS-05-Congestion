import torchvision.transforms as standard_transforms
from .SHHA import SHHA

def loading_data(data_root, type_dataset):
    transform = standard_transforms.Compose([standard_transforms.ToTensor(),  standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_set = SHHA(data_root, train=True, transform=transform, patch=True, flip=True, type_dataset=type_dataset)
    val_set = SHHA(data_root, train=False, transform=transform, type_dataset=type_dataset)
    return train_set, val_set