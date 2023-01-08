import torch
import numpy as np
import os

NUM_DATA_SETS = 5

def mnist():
    # exchange with the corrupted mnist dataset
    print(os.getcwd())
    PATH = "./corruptmnist/"
    train_sets = []
    for i in range(NUM_DATA_SETS):
        train_set = np.load(PATH + f"train_{i}.npz")
        #images = 
        train_sets.append(
            torch.utils.data.TensorDataset(
                torch.tensor(train_set["images"]),
                torch.tensor(train_set["labels"])
            )
        )
        
    
    test_set = np.load(PATH + "test.npz")
    
    test = torch.utils.data.TensorDataset(torch.tensor(test_set["images"]), torch.tensor(test_set["labels"]))
    train = torch.utils.data.ConcatDataset(train_sets)
    
    return train, test
