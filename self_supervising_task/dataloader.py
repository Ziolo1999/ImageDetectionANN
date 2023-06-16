from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
from self_supervising_task.settings import batch_size, mean_std_normalization, IMAGE_SIZE
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as F
import numpy as np
import torch

class RotatedDataset(ImageFolder):
    def __init__(self, root, transform=None, angle=None):
        super().__init__(root, transform=transform)
        self.angle = angle

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        
        if self.angle is None:
            random_angle = np.random.randint(low=0, high=4) * 90
            rotated_img = F.rotate(img, random_angle)
            label = random_angle // 90
        else:
           rotated_img = F.rotate(img, self.angle)
           label = self.angle // 90

        return rotated_img, label
    
class PreturbationDataset(ImageFolder):
    def __init__(self, root, color=None, transform=None):
        super().__init__(root, transform=transform)
        self.color = color

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        
        region_indx = np.random.randint(low=0, high=IMAGE_SIZE[0]-10)

        if self.color is None:
            random_color = np.random.choice([0,255])
            img[:, region_indx : region_indx+10, region_indx : region_indx+10] = torch.ones((3,10,10)) * random_color
            label = int(random_color/255)
        else:
            img[:, region_indx : region_indx+10, region_indx : region_indx+10] = torch.ones((3,10,10)) * self.color
            label = int(self.color/255)

        return img, label


def train_rotation_loader(batch_size=batch_size, transform="simple", mean_std_normalization=mean_std_normalization):
    if transform=="simple":
        transform = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean_std_normalization[0], mean_std_normalization[1])])
    else:
        transform = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                    transforms.RandomResizedCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean_std_normalization[0], mean_std_normalization[1])])
    
    # Train data loader
    train_0 = RotatedDataset(root='data/train', transform=transform, angle=0)
    train_90 = RotatedDataset(root='data/train', transform=transform, angle=90)
    train_180 = RotatedDataset(root='data/train', transform=transform, angle=180)
    train_270 = RotatedDataset(root='data/train', transform=transform, angle=270)
    train_dataset = ConcatDataset([train_0, train_90, train_180, train_270])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader

def valid_rotation_loader(batch_size=batch_size, mean_std_normalization=mean_std_normalization):
    transform = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean_std_normalization[0], mean_std_normalization[1])])

    # Train data loader
    validation_dataset = RotatedDataset(root='data/validation', transform=transform, angle=None)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    return validation_loader

def train_preturbation_loader(batch_size=batch_size, transform="simple", mean_std_normalization=mean_std_normalization):
    if transform=="simple":
        transform = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean_std_normalization[0], mean_std_normalization[1])])
    else:
        transform = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomRotation(20),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean_std_normalization[0], mean_std_normalization[1])])
    # Train data loader
    train_0 = PreturbationDataset(root='data/train', transform=transform, color=0)
    train_255 = PreturbationDataset(root='data/train', transform=transform, color=255)
    train_dataset = ConcatDataset([train_0, train_255])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader

def valid_preturbation_loader(batch_size=batch_size, mean_std_normalization=mean_std_normalization):
    transform = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean_std_normalization[0], mean_std_normalization[1])])
        
    # Train data loader
    validation_dataset = PreturbationDataset(root='data/validation', transform=transform, color=None)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    return validation_loader
