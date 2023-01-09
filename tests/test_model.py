from src.data.data import mnist
from src.models.model import MyAwesomeModel



def model_test(idx):
    train, test = mnist()
    im = test[0][idx]
    num_classes = 5
    model = MyAwesomeModel(num_classes)
    out = model(im)
    return len(out)

assert model_test(0) == 5