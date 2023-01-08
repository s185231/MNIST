import torch
from torch import nn


class MyAwesomeModel(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        # Defining some variables that might come in handy later
        KERNEL_SIZE = 3
        n_out_features1 = 6
        n_out_features2 = 16
        n_out_features3 = 20
        pool_kernel_size = 2
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=n_out_features1, kernel_size=KERNEL_SIZE, stride=1, padding=1),
            nn.BatchNorm2d(n_out_features1),
            nn.ReLU(),
            nn.MaxPool2d(pool_kernel_size)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=n_out_features1, out_channels=n_out_features2, kernel_size=KERNEL_SIZE, stride=1, padding=1),
            nn.BatchNorm2d(n_out_features2),
            nn.ReLU(),
            nn.MaxPool2d(pool_kernel_size)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=n_out_features2, out_channels=n_out_features3, kernel_size=KERNEL_SIZE, stride=1, padding=1),
            nn.BatchNorm2d(n_out_features3),
            nn.ReLU(),
            nn.MaxPool2d(pool_kernel_size)
        )
        
        self.fc = nn.Sequential(
            # Found those 320, simply through errors. I know... very scientific
            nn.Linear(180, 128),
            nn.Linear(128, 64),
            nn.Linear(64, n_classes),
            )

    def forward(self, x):
        #print(x.dtype)
        # swaps 1st and 2nd dimension to get batches in 1st
        # and also switches type to float32
        x = x.reshape(-1, 1, 28, 28).type(torch.float32)  
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

