from time import time
import torch
import recurrence_matrix
import numpy as np


def run(tensor, iterations):
    total_time = 0
    for _ in range(iterations):
        t0 = time()
        _ = recurrence_matrix.recurrence_matrix(tensor)
        total_time += time() - t0

    return total_time


def main():
    data = np.arange(10_000).reshape((1000, 10))
    tensor = torch.tensor(data, device='cuda')
    iterations = 1000
    total_time = run(tensor, iterations)

    print("Total Time: ", total_time)
    print("Average: ", total_time / iterations)


if __name__ == '__main__':
    main()
