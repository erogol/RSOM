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


class TestAlphaAndMiniBatch(unittest.TestCase):
    """Mini-batch rule: W <- W + alpha * (batch mean - W). Full batch: step 1."""

    @classmethod
    def setUpClass(cls):
        cls.X = _digits()

    def _train(self, seed=0, num_epoch=2, batch_size=128, **kw):
        som = _build(self.X, seed, num_units=49, height=7, width=7, **kw)
        w0 = som.W.detach().clone()
        som.train_batch(num_epoch=num_epoch, batch_size=batch_size, verbose=False)
        return som, w0

    def test_alpha_changes_minibatch_weights(self):
        a, _ = self._train(alpha_max=0.05, alpha_min=0.001)
        b, _ = self._train(alpha_max=0.9, alpha_min=0.5)
        self.assertGreater((a.W - b.W).abs().max().item(), 1e-2)

    def test_alpha_is_the_minibatch_step_size(self):
        # For a very small alpha, the weights move in proportion to alpha.
        a, w0 = self._train(num_epoch=1, alpha_max=1e-4, alpha_min=1e-4)
        b, _ = self._train(num_epoch=1, alpha_max=2e-4, alpha_min=2e-4)
        ratio = (b.W.detach() - w0).norm() / (a.W.detach() - w0).norm()
        self.assertAlmostEqual(ratio.item(), 2.0, delta=0.05)

    def test_full_batch_weights_do_not_use_alpha(self):
        a, _ = self._train(batch_size=None, num_epoch=10, alpha_max=0.05, alpha_min=0.001)
        b, _ = self._train(batch_size=None, num_epoch=10, alpha_max=0.9, alpha_min=0.5)
        self.assertLess((a.W - b.W).abs().max().item(), 1e-4)

    def test_minibatch_quantization_error_is_close_to_full_batch(self):
        qe = {}
        for batch_size in (None, 128, 32):
            runs = [
                _qe(self._train(seed, 100, batch_size)[0], self.X) for seed in range(3)
            ]
            qe[batch_size] = float(np.mean(runs))
        # 23.73 is the full-batch value of the code before this change.
        self.assertLess(qe[None], 23.73 * 1.03)
        self.assertLess(qe[128], 1.10 * qe[None])
        self.assertLess(qe[32], 1.10 * qe[None])

    def test_last_batch_with_one_sample(self):
        som = _build(self.X[:101], 0, num_units=25, height=5, width=5)
        som.train_batch(num_epoch=6, batch_size=4, verbose=False)
        self.assertTrue(torch.isfinite(som.W).all())


class TestActivationCounts(unittest.TestCase):
    def test_each_win_is_counted_one_time(self):
        X = _digits()
        som = _build(
            X,
            0,
            num_units=49,
            height=7,
            width=7,
            set_outlier_unit_det=False,
            set_inunit_outlier_det=False,
        )
        wins = []
        calc_distance = som._calc_distance

        def spy(*args, **kw):
            D = calc_distance(*args, **kw)
            wins.append(torch.bincount(D.argmin(0), minlength=som.num_units))
            return D

        som._calc_distance = spy
        som.train_batch(num_epoch=2, batch_size=128, verbose=False)
        # The last call is the final assignment, after the training.
        true_wins = torch.stack(wins[:-1]).sum(0).float()
        self.assertEqual(true_wins.sum().item(), 2 * X.shape[0])
        expected = true_wins / true_wins.sum()
        self.assertLess((som.activations - expected).abs().max().item(), 1e-6)


class TestUnitSaliency(unittest.TestCase):
    """Paper Eq. 3-4: e_j += rho * (beta_j + z_j), rho = 1 / learning rate."""

    def test_excitation_follows_paper_eq_3_and_4(self):
        som = _build(torch.randn(40, 3), 0, num_units=4, height=2, width=2)
        h = torch.tensor(
            [
                [1.0, 0.5, 0.5, 0.0],
                [0.5, 1.0, 0.0, 0.5],
                [0.5, 0.0, 1.0, 0.5],
                [0.0, 0.5, 0.5, 1.0],
            ]
        )
        z1 = torch.tensor([4.0, 0.0, 2.0, 0.0])
        z2 = torch.tensor([1.0, 3.0, 0.0, 2.0])
        som._update_unit_saliency(z1, h, 0.5)
        som._update_unit_saliency(z2, h, 0.25)
        # beta_j: sum of h(j, v) * z_v over the winners v other than j.
        beta1 = torch.tensor([1.0, 2.0, 2.0, 1.0])
        beta2 = torch.tensor([1.5, 1.5, 1.5, 1.5])
        expected = (beta1 + z1) / 0.5 + (beta2 + z2) / 0.25
        self.assertTrue(torch.allclose(som.unit_saliency_coeffs, expected))

    def test_unit_scores_are_normalized_after_training(self):
        som = _build(_digits(), 0, num_units=49, height=7, width=7)
        som.train_batch(num_epoch=5, batch_size=256, verbose=False)
        self.assertAlmostEqual(som.unit_saliency_coeffs.sum().item(), 1.0, places=4)
        self.assertEqual(som.unit_saliency.shape, (49,))


class TestInUnitOutliers(unittest.TestCase):
    """Default in-unit rule: Tukey upper whisker. Percentile rule on request."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        centers = rng.uniform(-6, 6, size=(5, 10))
        X = centers[rng.integers(0, 5, 2000)] + rng.normal(0, 1, (2000, 10))
        cls.X = torch.from_numpy(X.astype(np.float32))

    def _flagged_share(self, **kw):
        som = _build(
            self.X, 0, num_units=49, height=7, width=7, set_outlier_unit_det=False, **kw
        )
        som.train_batch(num_epoch=50, verbose=False)
        return (~som.inst_saliency).float().mean().item()

    def test_default_rule_flags_few_samples_of_clean_data(self):
        self.assertLess(self._flagged_share(), 0.03)

    def test_percentile_rule_is_still_available(self):
        share = self._flagged_share(inunit_outlier_thresh=95)
        self.assertGreater(share, 0.04)
        self.assertLess(share, 0.08)

    def test_larger_whisker_factor_flags_fewer_samples(self):
        self.assertLess(
            self._flagged_share(inunit_whisker_factor=3.0),
            self._flagged_share(inunit_whisker_factor=0.5),
        )


class TestMapShape(unittest.TestCase):
    def test_shape_estimate_centers_the_data(self):
        raw = _digits(standardize=False)
        som = _build(raw, 0, num_units=49)
        self.assertEqual((som.height, som.width, som.num_units), (7, 7, 49))
        self.assertTrue(torch.equal(som.X, _digits(standardize=False)))

    def test_changed_unit_count_is_logged(self):
        torch.manual_seed(0)
        X = torch.randn(300, 4) * torch.tensor([4.0, 1.0, 1.0, 1.0])
        logger = logging.getLogger(RSOM.__module__)
        with self.assertLogs(logger, level="WARNING") as logs:
            som = RSOM(X, num_units=49)
        self.assertEqual(som.num_units, 52)
        self.assertIn("52", "\n".join(logs.output))


class TestSmallPoints(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.data = torch.randn(100, 10)
        self.logger = logging.getLogger(RSOM.__module__)

    def test_cost_moving_average_keeps_most_of_the_old_value(self):
        som = RSOM(self.data, num_units=4, height=2, width=2)
        som._print_cost(None, torch.full((4, 5), 1.0), 0, 1, 0, 2)
        som._print_cost(None, torch.full((4, 5), 2.0), 0, 1, 1, 2)
        self.assertAlmostEqual(float(som.avg_cost), 0.99 * 1.0 + 0.01 * 2.0, places=5)

    def test_verbose_false_is_silent(self):
        records = []
        handler = logging.Handler()
        handler.emit = records.append
        old_level = self.logger.level
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
        try:
            som = RSOM(self.data, num_units=25, height=5, width=5)
            records.clear()
            som.train_batch(num_epoch=3, batch_size=20, verbose=False)
            self.assertEqual(len(records), 0)
            som.train_batch(num_epoch=1, verbose=True)
            self.assertGreater(len(records), 0)
        finally:
            self.logger.removeHandler(handler)
            self.logger.setLevel(old_level)

    def test_best_match_uses_the_cosine_distance(self):
        som = RSOM(self.data, num_units=25, height=5, width=5, dist="cosine")
        X = self.data[:7] * 3
        BMU, D = som.best_match(X)
        W = som.W.detach()
        expected = 1 - (W / W.norm(dim=1, keepdim=True)) @ (
            X / X.norm(dim=1, keepdim=True)
        ).t()
        self.assertTrue(torch.allclose(D, expected, atol=1e-5))
        self.assertTrue(torch.equal(BMU.argmax(1), expected.argmin(0)))

    def test_set_params_does_not_store_one_map_per_epoch(self):
        som = RSOM(self.data, num_units=25, height=5, width=5)
        U = som.set_params(200)
        self.assertNotIn("H_maps", U)
        self.assertEqual(len(U["alphas"]), 200)
        self.assertEqual(len(U["radiuses"]), 200)
        neigh_map = som._neighborhood_map(U["dist_map"], U["radiuses"][0])
        self.assertEqual(neigh_map.shape, (25, 25))
        # The map does not include the learning rate.
        self.assertTrue(torch.equal(neigh_map.diagonal(), torch.ones(25)))

    def test_import_does_not_configure_logging(self):
        module = sys.modules[RSOM.__module__]
        code = f"import logging, {module.__name__}; print(len(logging.getLogger().handlers))"
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(module.__file__)),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "0")
