import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os


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
        )

    def forward(self, x):
        return self.stack(x)


def train(device, learning_rate, batch_size, epochs):
    model = ClassifyNet().to(device)
    for name, param in model.named_parameters():
        print(f"Layer: {name} Size: {param.size()}")

    curr_file_dir = os.path.dirname(os.path.realpath(__file__))
    data_root_dir = os.path.join(curr_file_dir, "data")
    train_data = datasets.FashionMNIST(
        root=data_root_dir, train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.FashionMNIST(
        root=data_root_dir, train=False, download=True, transform=transforms.ToTensor())

    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}/{epochs}\n-------------------------------")
        train_loop(device, train_dataloader, model, loss_fn, optimizer)
        test_loop(device, test_dataloader, model, loss_fn)
    print("Done!")


def train_loop(device, dataloader: DataLoader, model: nn.Module, loss_fn, optimizer: torch.optim.Optimizer):
    model.train()
    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        pred: torch.Tensor = model(X)
        loss: torch.Tensor = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(device, dataloader: DataLoader, model: nn.Module, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred: torch.Tensor = model(X)
            loss: torch.Tensor = loss_fn(pred, y)
            test_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    learning_rate = 1e-3
    batch_size = 64
    epochs = 5
    num_cores = 1

    # inter-op parallelism
    torch.set_num_interop_threads(num_cores)

    # intra-op parallelism
    torch.set_num_threads(num_cores)
    # os.environ["OMP_NUM_THREADS"] = str(num_cores)
    # os.environ["MKL_NUM_THREADS"] = str(num_cores)

    print(f"Using {device} device")
    train(device, learning_rate, batch_size, epochs)
