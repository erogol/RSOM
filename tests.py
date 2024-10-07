import unittest

import torch

from som import SOM


class TestSOM(unittest.TestCase):
    def setUp(self):
        self.data = torch.randn(100, 10)
        self.som = SOM(self.data, num_units=25, height=5, width=5)

    def test_init(self):
        self.assertEqual(self.som.num_units, 25)
        self.assertEqual(self.som.height, 5)
        self.assertEqual(self.som.width, 5)
        self.assertEqual(self.som.W.shape, (25, 10))

    def test_normalize_weights(self):
        self.som._normalize_weights()
        norms = torch.norm(self.som.W.data, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-6))

    def test_unit_cords(self):
        self.assertEqual(self.som.unit_cords(7), (2, 1))
        self.assertEqual(self.som.unit_cords(24), (4, 4))

    def test_euq_dist(self):
        X = self.data[:5]
        X2 = (X**2).sum(1).unsqueeze(1)
        D = self.som._euq_dist(X2, X)
        self.assertEqual(D.shape, (25, 5))

    def test_find_neighbors(self):
        neighbors = self.som.find_neighbors(12, 1)
        self.assertEqual(neighbors.shape, (1, 25))
        self.assertEqual(neighbors[0, 12].item(), 0)

    def test_best_match(self):
        X = self.data[:5]
        BMU, D = self.som.best_match(X)
        self.assertEqual(BMU.shape, (5, 25))
        self.assertEqual(D.shape, (25, 5))

    def test_assing_to_units(self):
        self.som.assing_to_units()
        self.assertEqual(self.som.ins_unit_assign.shape, (100,))
        self.assertEqual(self.som.ins_unit_dist.shape, (100,))

    def test_set_params(self):
        U = self.som.set_params(10)
        self.assertEqual(len(U["alphas"]), 10)
        self.assertEqual(len(U["H_maps"]), 10)
        self.assertEqual(len(U["radiuses"]), 10)

    def test_train_batch(self):
        self.som.train_batch(num_epoch=5, batch_size=20, verbose=False)
        self.assertIsNotNone(self.som.ins_unit_assign)
        self.assertIsNotNone(self.som.ins_unit_dist)

    def test_update_unit_saliency(self):
        win_counts = torch.ones(25)
        update_rate = torch.ones(25, 25)
        self.som._update_unit_saliency(win_counts, update_rate, 0.1)
        self.assertGreater(self.som.unit_saliency_coeffs.sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
