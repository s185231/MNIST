from src.data.data import mnist
from src.models.model import MyAwesomeModel

def test_model(idx):
    train, test = mnist()
    im = test[0][idx]
    num_classes = 5
    model = MyAwesomeModel(num_classes)
    out = model(im)
    assert out