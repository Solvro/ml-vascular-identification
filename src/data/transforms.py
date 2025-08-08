from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def build_transforms(img_size=256, train=True, hflip_p=0.5):
    """Return torchvision Compose for train/eval."""
    if train:
        augs = [
            transforms.Grayscale(num_output_channels=3),    # L→RGB (3ch)
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(hflip_p),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    else:
        augs = [
            transforms.Grayscale(num_output_channels=3),    # L→RGB (3ch)
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    return transforms.Compose(augs)
