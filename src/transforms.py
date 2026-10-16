"""Shared image transforms. MLP works on small flattened images; ResNet on 224px, ImageNet-normalised."""
from __future__ import annotations

from torchvision import transforms

from src.config import MLP_IMG_SIZE, RESNET_IMG_SIZE

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def mlp_transforms(train: bool):
    aug = [transforms.RandomHorizontalFlip()] if train else []
    return transforms.Compose(
        [transforms.Resize((MLP_IMG_SIZE, MLP_IMG_SIZE)), *aug, transforms.ToTensor()]
    )


def resnet_transforms(train: bool):
    if train:
        pipe = [
            transforms.Resize((RESNET_IMG_SIZE, RESNET_IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        pipe = [transforms.Resize((RESNET_IMG_SIZE, RESNET_IMG_SIZE))]
    return transforms.Compose(
        [*pipe, transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )
