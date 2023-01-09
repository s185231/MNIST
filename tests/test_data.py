#from tests import _PATH_DATA
from torchvision import datasets
import numpy as np 
from PIL import Image
import torch  
from src.data.data import mnist


train, test = mnist()
N_train = len(train)
N_test = len(test)

assert len(train) == N_train
assert len(test) == N_test

for i in range(len(train)):
    assert (train[i][0]).size()== torch.empty(28,28).size()