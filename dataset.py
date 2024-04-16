# import some packages you need here
import torch
from torchvision import transforms
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

class MNIST(Dataset):
    """ MNIST dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        data_dir: directory path containing images

    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
            - Substract mean of 0.1307, and divide by std 0.3081
            - These preprocessing can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png
    """

    def __init__(self, data_path, image_transform=None):
        # write your codes here
        self.dataset = pd.read_csv(data_path)
        self.image_transform = image_transform

    def __len__(self):
        # write your codes here
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        # name = row['name']
        img_path = row['img_path']
        # Switched to using PIL.Image for loading images, which is more standard with PyTorch and torchvision.
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        label = torch.tensor(row['label'])

        if self.image_transform is not None:
            img = self.image_transform(img)

        return img, label

def get_loader(train_df, test_df, batch_size, test_batch_size):
        
        image_transform = transforms.Compose([
            transforms.ToTensor(), # Convert the values of all values to a range between 0 and 1
            transforms.Normalize((0.1307,), (0.3081,)) # Normalize with mean and std
        ])
        def get_aug_transforms():
            train_transform = transforms.Compose([
                transforms.RandomRotation(10),             # Rotate images by up to 18 degrees
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)), # Moving the images from left to right
                transforms.ToTensor(),                     # Convert the values of all values to a range between 0 and 1
                transforms.Normalize((0.1307,), (0.3081,)) # Normalize with mean and std
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            return train_transform, test_transform
        
        # normal dataset
        train_dataset = MNIST(data_path=train_df, image_transform=image_transform)
        test_dataset = MNIST(data_path=test_df, image_transform=image_transform)
        
        # # augmentation aplied dataset
        # train_transform, test_transform = get_aug_transforms()
        # train_dataset = MNIST(data_path=train_df, image_transform=train_transform)
        # test_dataset = MNIST(data_path=test_df, image_transform=test_transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, generator=torch.Generator().manual_seed(725))
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=8, pin_memory=True)
        return train_loader, test_loader        

if __name__ == '__main__':

    # write test codes to verify your implementations

    # Note:
    # 1) Each image should be preprocessed as follows:
    #     - First, all values should be in a range of [0,1]
    #     - Substract mean of 0.1307, and divide by std 0.3081
    train, test = get_loader('train_df.csv', 'test_df.csv', 1, 1)
    

    # # test dataloader
    # for images, labels in train:
    #     print(type(images), type(labels))
    #     print(images, labels)
    #     break