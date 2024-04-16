import torch
import torch.nn as nn

class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """

    def __init__(self):
        # write your codes here
        super().__init__()
        self.feature = nn.Sequential(
            #1 layer
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2), # MNIST Image size is 28, model input size is 32. so, 28*28->32*32-->28*28
            nn.ReLU(),
            # nn.Dropout(0.25), # To normalize model, activate Dropout
            #2 subsampling
            nn.MaxPool2d(kernel_size=2, stride=2), # input size : 28x28x6, output size : 14x14x6
            #3 layer
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1), # input size : 14x14x6, output size : 10x10x16
            nn.ReLU(),
            # nn.Dropout(0.25), # To normalize model, activate Dropout
            #4 subsampling
            nn.MaxPool2d(kernel_size=2, stride=2), # input_size : 10x10x16, output_size : 5x5x16
            #5 layer
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.ReLU()

        )
        self.classifier = nn.Sequential(
            #1 FC
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            #2 FC
            nn.Linear(in_features=84, out_features=10),
        )

    def forward(self, img):

        img = self.feature(img)
        img = torch.flatten(img, 1)
        logits = self.classifier(img)
        return logits


class CustomMLP(nn.Module):
    """ Your custom MLP model

        - Note that the number of model parameters should be about the same
          with LeNet-5
    """

    def __init__(self):
        super().__init__()
        # Final attempt to closely match parameter count
        self.fc1 = nn.Linear(784, 80)  
        self.fc2 = nn.Linear(80, 20)   
        self.fc3 = nn.Linear(20, 10)    
        
        self.activations = nn.ReLU()

    def forward(self, img):
        img = torch.flatten(img, 1)  # Flatten the image tensor
        x = self.activations(self.fc1(img))
        x = self.activations(self.fc2(x))
        output = self.fc3(x)
        return output
