import torch
import numpy as np
from unittest import TestCase, main


class TestRecu(TestCase):

    def setUp(self) -> None:
        self.data = np.array([[1, 2, 3, 4], [1, 1, 1, 1]])

    def test_can_import(self):
        import recurrence_matrix

    def test_values_correct_cpu(self):
        import recurrence_matrix

        device = torch.device('cuda')
        t0 = torch.tensor(self.data, device=device)
        out = recurrence_matrix.recurrence_matrix(t0)
        real = np.array([self.pairwise_distance(d) for d in self.data])
        np.testing.assert_almost_equal(out.cpu().numpy(), real)

    @staticmethod
    def pairwise_distance(d):
        res = np.zeros((1, d.shape[0], d.shape[0]))
        for i in range(d.shape[0]):
            res[0, i, :] = abs(d[i] - d)
        return res


if __name__ == '__main__':
    main()
