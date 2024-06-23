from torchvision import transforms, datasets
import torch
from supervising_task.settings import batch_size, mean_std_normalization, IMAGE_SIZE


def load_15SceneData(batch_size=batch_size, crop=True):

    if crop:
        transform_train = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomRotation(20),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean_std_normalization[0], mean_std_normalization[1])])
    else:
        transform_train = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                    transforms.RandomHorizontalFlip(p=0.5),

                                    transforms.RandomRotation(20),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean_std_normalization[0], mean_std_normalization[1])])
    
    transform_test = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean_std_normalization[0], mean_std_normalization[1])])
    # Train data loader
    trainset = datasets.ImageFolder('data/train', transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # Validation data loader
    testset = datasets.ImageFolder('data/validation', transform=transform_test)
    validation_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

    return train_loader, validation_loader

def simple_15SceneData(batch_size=batch_size, mean_std_normalization=mean_std_normalization):

    transform_train = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean_std_normalization[0], mean_std_normalization[1])])
    
    transform_test = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean_std_normalization[0], mean_std_normalization[1])])
    # Train data loader
    trainset = datasets.ImageFolder('data/train', transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # Validation data loader
    testset = datasets.ImageFolder('data/validation', transform=transform_test)
    validation_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

    return train_loader, validation_loader


def validation_15SceneData(batch_size=batch_size):

    transform_test = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                    transforms.ToTensor()])

    # Validation data loader
    testset = datasets.ImageFolder('data/validation', transform=transform_test)
    validation_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

    return validation_loader
