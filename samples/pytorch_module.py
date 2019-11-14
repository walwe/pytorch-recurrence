import torch
from recurrence_matrix import recurrence_matrix
import numpy as np


class RecurrenceFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        return recurrence_matrix(x)


class RecurrenceLayer(torch.nn.Module):

    @staticmethod
    def forward(x):
        return RecurrenceFunction.apply(x)


def main():
    device = "cuda"
    image_size = 224
    data = np.arange(100_000).reshape((10000, 10))
    tensor = torch.tensor(data, dtype=torch.float).to(device)
    layer = torch.nn.Sequential(
        RecurrenceLayer(),
        torch.nn.AdaptiveAvgPool2d(image_size),
        torch.nn.Flatten(),
        torch.nn.Linear(image_size*image_size, 2)
    ).to(device)

    output = layer(tensor)
    print(data.shape, tensor.shape, output.shape)


if __name__ == '__main__':
    main()
