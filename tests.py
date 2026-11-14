import logging
import os
import subprocess
import sys
import unittest
import warnings

import numpy as np
import torch
from sklearn.datasets import load_digits

from rsom import RSOM

torch.set_num_threads(1)


def _digits(standardize: bool = True) -> torch.Tensor:
    raw = torch.from_numpy(load_digits().data).float()
    if not standardize:
        return raw
    std = raw.std(0)
    return (raw - raw.mean(0)) / torch.where(std > 0, std, torch.ones_like(std))


def _qe(som: RSOM, X: torch.Tensor) -> float:
    """Mean squared distance of each sample to its nearest unit."""
    return torch.cdist(X, som.W.detach()).min(1)[0].pow(2).mean().item()


def _build(X: torch.Tensor, seed: int, **kw) -> RSOM:
    torch.manual_seed(seed)
    return RSOM(X, **kw)


class TestSOM(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.data = torch.randn(100, 10)
        self.som = RSOM(self.data, num_units=25, height=5, width=5)

    def test_init(self):
        self.assertEqual(self.som.num_units, 25)
        self.assertEqual(self.som.height, 5)
        self.assertEqual(self.som.width, 5)
        self.assertEqual(self.som.W.shape, (25, 10))

    def test_num_units_follows_height_and_width(self):
        som = RSOM(self.data, height=5, width=5)
        self.assertEqual(som.num_units, 25)
        self.assertEqual(som.W.shape, (25, 10))
        som.train_batch(num_epoch=3, verbose=False)
        self.assertEqual(som.ins_unit_assign.shape, (100,))

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
        D = self.som._euclidean_distance(X, X2)
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


class TestRadiusSchedule(unittest.TestCase):
    """The radius follows the 2014 schedule: ceil(r0 * (N - t) / (N - 1)) - 1."""

    def test_radius_on_a_7x7_map(self):
        som = _build(torch.randn(60, 4), 0, num_units=49, height=7, width=7)
        radiuses = np.array(som.set_params(300)["radiuses"])
        self.assertEqual(radiuses[0], 3)
        self.assertEqual(radiuses[-1], 0)
        self.assertEqual(int((radiuses >= 1).sum()), 201)

    def test_radius_on_a_4x13_map(self):
        som = _build(torch.randn(60, 4), 0, num_units=52, height=4, width=13)
        radiuses = np.array(som.set_params(100)["radiuses"])
        self.assertEqual(int((radiuses >= 1).sum()), 51)

    def test_one_epoch_does_not_divide_by_zero(self):
        som = _build(torch.randn(60, 4), 0, num_units=25, height=5, width=5)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            radiuses = som.set_params(1)["radiuses"]
        self.assertTrue(np.isfinite(radiuses).all())
        som.train_batch(num_epoch=1, verbose=False)
        self.assertTrue(torch.isfinite(som.W).all())
