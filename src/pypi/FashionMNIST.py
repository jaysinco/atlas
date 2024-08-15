import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class ClassifyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.stack(x)


def train(learning_rate, batch_size, epochs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = ClassifyNet().to(device)
    for name, param in model.named_parameters():
        print(f"Layer: {name} Size: {param.size()}")

    train_data = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=transforms.ToTensor())

    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for X, y in train_dataloader:
        print(X.shape, y.shape)
        break


if __name__ == "__main__":
    learning_rate = 1e-3
    batch_size = 64
    epochs = 5
    train(learning_rate, batch_size, epochs)
