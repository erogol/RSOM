"""
Rectifying Self Organazing Maps a.k.a RSOM

RSOM is a clustering and outlier detection method that is predicated with
old Self Organazing Maps.

This module is a PyTorch implementation. It includes full-batch and
mini-batch learning rules.

For more detail about RSOM refer to http://arxiv.org/abs/1312.4384

AUTHOR:
    Eren Golge
    erengolge@gmail.com
    www.erengolge.com
"""

"""
TO DO:
-> Try dot product distance instead of Euclidean
-> Normzalize only updated weight vectors in that epoch
-> compare code with https://github.com/JustGlowing/minisom/blob/master/minisom.py
-> print resulting objective values
-> write bookeeping for best objective value
-> learning rate is already decreasing so radius might be good to keep it constant
-> UPDATE only winners
"""

import logging
from typing import Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class RSOM(torch.nn.Module):
    def __init__(
        self,
        data: torch.Tensor,
        num_units: int = 10,
        height: Optional[int] = None,
        width: Optional[int] = None,
        alpha_max: float = 0.05,
        alpha_min: float = 0.001,
        set_count_activations: bool = True,
        set_outlier_unit_det: bool = True,
        set_inunit_outlier_det: bool = True,
        outlier_unit_thresh: float = 0.5,
        inunit_outlier_thresh: Optional[float] = None,
        dist: str = "euclidean",
        log_full_data_cost: bool = False,
        steps_to_full_data_cost: int = -1,  # TODO: number of steps to compute full data cost
        inunit_whisker_factor: float = 1.5,
        outlier_group_share: Optional[float] = None,
    ):
        """Rectifying Self Organizing Maps. RSOM is a clustering and outlier detection method that is predicated with old Self Organizing Maps.

        Args:
            data: Input data.
            num_units: Number of units. When height and width are given, it is set to height * width.
            height: Height of the map.
            width: Width of the map.
            alpha_max: Learning rate at the first epoch.
            alpha_min: Learning rate at the end of the linear decrease. In mini-batch mode, each weight
                update is W <- W + step * (M - W), where M is the neighbourhood-weighted mean of the batch
                and step = alpha + (1 - alpha) * B / N (B: batch size, N: number of samples). A larger
                alpha follows the last batches more closely. Full-batch mode does not use alpha for the
                weights (see train_batch).
            set_count_activations: Whether to count activations.
            set_outlier_unit_det: Whether to detect outlier units.
            set_inunit_outlier_det: Whether to detect in-unit outliers.
            outlier_unit_thresh: Threshold for outlier unit detection.
            inunit_outlier_thresh: Percentile (0-100) for the in-unit percentile rule: a sample is an outlier
                when its distance to its unit is above this percentile of the distances in the unit. This rule
                always flags about (100 - percentile) % of each unit. The default None selects the box-plot
                rule (see inunit_whisker_factor).
            dist: Distance metric. Can be "euclidean" or "cosine".
            log_full_data_cost: Whether to log full data cost or use exponential moving average. Computing full data
                cost is expensive and might cause OOM.
            inunit_whisker_factor: Factor of the box-plot rule, which is the default in-unit rule: a sample is
                an outlier when its distance to its unit is above the Tukey upper whisker
                Q3 + factor * (Q3 - Q1) of the distances in the unit. The rule uses the distances as stored in
                ins_unit_dist (the squared distance for dist="euclidean").
            outlier_group_share: If given, also flag dense unrelated groups. The rule builds a minimum
                spanning tree over the units that hold samples and cuts the edges longer than
                Q3 + 3 * (Q3 - Q1) of the edge lengths. A part of the tree with less than this share of
                the samples is an outlier group. A small real cluster can also be flagged. None turns the
                rule off.

        """

        super(RSOM, self).__init__()
        self.X = data
        self.num_units = num_units
        self.height = height
        self.width = width
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.set_count_activations = set_count_activations
        self.set_outlier_unit_det = set_outlier_unit_det
        self.set_inunit_outlier_det = set_inunit_outlier_det
        self.outlier_unit_thresh = outlier_unit_thresh
        self.inunit_outlier_thresh = inunit_outlier_thresh
        self.inunit_whisker_factor = inunit_whisker_factor
        self.outlier_group_share = outlier_group_share
        self.log_full_data_cost = log_full_data_cost

        self._estimate_map_shape()
        self.data_dim = self.X.shape[1]

        self.W = torch.nn.Parameter(torch.randn(self.num_units, self.data_dim))
        self._normalize_weights()

        self.distance_metric = dist
        if self.distance_metric not in ["euclidean", "cosine"]:
            raise ValueError("distance_metric must be either 'euclidean' or 'cosine'")

        self.activations = torch.zeros(self.num_units)
        self.unit_saliency_coeffs = torch.zeros(self.num_units)
        self.unit_saliency = torch.ones(self.num_units, dtype=torch.bool)
        self.inst_saliency = torch.tensor([])
        self.ins_unit_assign = torch.tensor([])
        self.ins_unit_dist = torch.tensor([])
        self.unit_coher = torch.tensor([])

    def _normalize_weights(self):
        self.W.data = self.W.data / torch.norm(self.W.data, dim=1, keepdim=True)

    def _estimate_map_shape(self):
        if self.height is None or self.width is None:
            # Center the data for the shape estimate only. Without this, the
            # first singular value follows the data mean, not the data spread.
            s = torch.linalg.svdvals(self.X - self.X.mean(0))
            ratio = s[0] / s[1]
            self.height = min(
                self.num_units, int(np.ceil(np.sqrt(self.num_units / ratio)))
            )
            self.width = int(np.ceil(self.num_units / self.height))
            if self.height * self.width != self.num_units:
                logger.warning(
                    f"The map has {self.height * self.width} units (height = {self.height}, "
                    f"width = {self.width}), not the requested {self.num_units}."
                )
        self.num_units = self.height * self.width

    def unit_cords(self, index: int) -> Tuple[int, int]:
        return index % self.width, index // self.width

    def _calc_distance(self, X, X2=None):
        if self.distance_metric == "euclidean":
            return self._euclidean_distance(X, X2)
        elif self.distance_metric == "cosine":
            return self._cosine_distance(X)

    def _euclidean_distance(self, X, X2=None):
        if X2 is None:
            X2 = (X**2).sum(1)[:, None]
        W2 = (self.W**2).sum(1)[:, None]
        return -2 * torch.mm(self.W, X.t()) + W2 + X2.t()

    def _cosine_distance(self, X):
        X_norm = X / torch.norm(X, dim=1, keepdim=True)
        W_norm = self.W / torch.norm(self.W, dim=1, keepdim=True)
        return 1 - torch.mm(W_norm, X_norm.t())

    def find_neighbors(self, unit_id: int, radius: int) -> torch.Tensor:
        neighbors = torch.zeros(1, self.num_units)
        unit_x, unit_y = self.unit_cords(unit_id)

        min_y = max(int(unit_y - radius), 0)
        max_y = min(int(unit_y + radius), self.height - 1)
        min_x = max(int(unit_x - radius), 0)
        max_x = min(int(unit_x + radius), self.width - 1)

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                dist = abs(y - unit_y) + abs(x - unit_x)
                neighbors[0, x + (y * self.width)] = dist

        return neighbors

    def best_match(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if X.dim() == 1:
            X = X.unsqueeze(0)
        D = self._calc_distance(X)
        BMU = (D == D.min(0)[0]).float().t()
        return BMU, D

    def assing_to_units(self, X=None):
        if X is None:
            D = self._calc_distance(self.X)
            self.ins_unit_assign = D.argmin(axis=0)
            self.ins_unit_dist = D[self.ins_unit_assign, torch.arange(self.X.shape[0])]
        else:
            D = self._calc_distance(X)
            ins_unit_assign = D.argmin(axis=0)
            ins_unit_dist = D[ins_unit_assign, torch.arange(X.shape[0])]
            return ins_unit_assign, ins_unit_dist

    @staticmethod
    def _neighborhood_map(dist_map: torch.Tensor, radius: int) -> torch.Tensor:
        # The neighbourhood map does not include the learning rate.
        neigh_updt_map = 1 - dist_map / float(1 + radius)
        neigh_updt_map[dist_map > radius] = 0
        return neigh_updt_map

    def set_params(self, num_epoch: int) -> dict:
        U = {"alphas": [], "radiuses": []}

        dist_map = torch.zeros(self.num_units, self.num_units)
        max_radius = np.ceil(1 + np.floor(min(self.width, self.height) - 1) / 2) - 1
        for u in range(self.num_units):
            dist_map[u, :] = self.find_neighbors(u, self.num_units)

        for epoch in range(num_epoch):
            alpha = self.alpha_max - self.alpha_min
            alpha = alpha * (num_epoch - epoch) / num_epoch + self.alpha_min
            # 2014 schedule: the radius is a whole number that decreases to 0.
            radius = max_radius * (num_epoch - epoch) / max(num_epoch - 1, 1)
            radius = max(int(np.ceil(radius)) - 1, 0)

            U["alphas"].append(alpha)
            U["radiuses"].append(radius)

        # train_batch computes the neighbourhood map of each epoch from dist_map.
        U["dist_map"] = dist_map
        return U

    def train_batch(
        self,
        num_epoch: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: bool = True,
    ):
        """
        Full-batch mode (batch_size is None or not below the number of samples) is the classic batch SOM:
        each epoch sets each unit to the neighbourhood-weighted mean of the data. This mode does not use
        alpha_max and alpha_min for the weights. In the two modes, alpha sets the weight of each epoch
        in the unit saliency scores (rho = 1 / alpha, paper Eq. 3), so later epochs count more.

        Mini-batch mode moves each unit toward the neighbourhood-weighted mean M of the batch:
        W <- W + step * (M - W), with step = alpha + (1 - alpha) * B / N. alpha decreases linearly
        from alpha_max toward alpha_min. The B / N term keeps large batches (few steps per epoch) from
        moving the units too little.

        Args:
            num_epoch: number of epochs to train
            batch_size: number of samples to train in one batch
            verbose: if True, log the progress of training at level INFO with the logging module. The
                module does not configure logging: call logging.basicConfig(level=logging.INFO) to see it.
        """
        if num_epoch is None:
            num_epoch = 500 * self.num_units

        if batch_size is None:
            batch_size = self.X.shape[0]

        if verbose:
            logger.info("Learning...")
        U = self.set_params(num_epoch)

        X2 = None
        full_batch = batch_size >= self.X.shape[0]
        if full_batch:
            X2 = (self.X**2).sum(1).unsqueeze(1)

        for epoch in range(num_epoch):
            if verbose:
                logger.info(f"Epoch --- {epoch}")
            update_rate = self._neighborhood_map(U["dist_map"], U["radiuses"][epoch])
            learn_rate = U["alphas"][epoch]

            shuffle_indices = torch.randperm(self.X.shape[0])
            win_counts = torch.zeros(self.num_units)

            batches = torch.split(shuffle_indices, batch_size)
            num_steps = len(batches)

            for step, batch_indices in enumerate(batches):
                batch_data = self.X[batch_indices, :]
                D = self._calc_distance(batch_data, X2)
                # One winner for each sample, so a tie is not counted two times.
                BMU = torch.nn.functional.one_hot(D.argmin(0), self.num_units).float()
                batch_wins = BMU.sum(dim=0)
                win_counts += batch_wins

                if self.set_count_activations:
                    self.activations += batch_wins

                A = torch.mm(BMU, update_rate)
                S = A.sum(0)
                non_zeros = S > 0
                means = torch.mm(A[:, non_zeros].t(), batch_data)
                means = means / S[non_zeros].unsqueeze(1)

                # The batch share B / N raises the step for large batches, so few steps per epoch
                # still move the units. A batch of all samples gets step 1, as in full-batch mode.
                share = batch_data.shape[0] / self.X.shape[0]
                step_size = 1.0 if full_batch else learn_rate + (1 - learn_rate) * share
                self.W.data[non_zeros] = torch.lerp(
                    self.W.data[non_zeros], means, step_size
                )

                if verbose:
                    self._print_cost(X2, D, epoch, num_epoch, step, num_steps)

            if self.set_outlier_unit_det:
                # h of paper Eq. 2 includes the learning rate.
                self._update_unit_saliency(
                    win_counts, learn_rate * update_rate, learn_rate
                )

        if self.set_count_activations:
            self.activations /= self.activations.sum()

        self.assing_to_units()

        if self.set_outlier_unit_det:
            self._find_outlier_units()

        if self.outlier_group_share is not None:
            self._find_outlier_groups()

        if self.set_inunit_outlier_det:
            self._find_inunit_outliers()

    def _print_cost(
        self,
        X2: torch.Tensor,
        D: torch.Tensor,
        epoch: int,
        num_epoch: int,
        step: int,
        num_steps: int,
    ):
        batch_cost = D.min(0)[0].mean()

        if self.log_full_data_cost:
            # TODO: handle when data is too large
            if self.distance_metric == "euclidean":
                D = self._calc_distance(self.X, X2)
                cost = torch.norm(D.min(0)[0], p=1) / self.X.shape[0]
            elif self.distance_metric == "cosine":
                D = self._calc_distance(self.X)
                cost = D.min(0)[0].mean()  # Average minimum cosine distance
        else:
            # exponential moving average cost
            if not hasattr(self, "avg_cost"):
                self.avg_cost = batch_cost
            else:
                self.avg_cost = self.avg_cost * 0.99 + batch_cost * 0.01
            cost = self.avg_cost

        logger.info(
            f"epoch {epoch} / {num_epoch} -- step {step} / {num_steps} -- avg-cost: {cost.item():.6f} -- batch-cost: {batch_cost.item():.6f}"
        )

    def _update_unit_saliency(
        self, win_counts: torch.Tensor, update_rate: torch.Tensor, learn_rate: float
    ):
        """Add the excitation of one epoch, paper Eq. 3-4: e_j += rho * (beta_j + z_j).

        rho is 1 / learn_rate. z_j (win_counts) is the win count of unit j in the epoch. beta_j is the
        sum of h(j, v) * z_v over the winners v other than unit j, where h is update_rate.
        """
        neigh = update_rate.clone()
        neigh.fill_diagonal_(0)
        beta = torch.mv(neigh, win_counts)
        self.unit_saliency_coeffs += (beta + win_counts) / learn_rate

    def _find_outlier_units(self):
        self.unit_saliency_coeffs /= self.unit_saliency_coeffs.sum()
        self.unit_saliency = (
            self.unit_saliency_coeffs > self.outlier_unit_thresh / self.num_units
        )

        self.inst_saliency = torch.ones(self.X.shape[0], dtype=torch.bool)
        outlier_units = torch.where(self.unit_saliency == False)[0]
        for i in outlier_units:
            self.inst_saliency[torch.where(self.ins_unit_assign == i)[0]] = False

    def _find_outlier_groups(self):
        """Flag the units of each small, far group (see outlier_group_share) and their samples."""
        if self.inst_saliency.numel() == 0:
            self.inst_saliency = torch.ones(self.X.shape[0], dtype=torch.bool)
        # The tree has at most num_units nodes, so compute it on the CPU.
        sizes = torch.bincount(self.ins_unit_assign.cpu(), minlength=self.num_units)
        used = torch.where(sizes > 0)[0]
        if len(used) < 3:
            return
        W = self.W.data.cpu()[used]
        dist = torch.cdist(W, W)
        # Prim's algorithm: edges of the minimum spanning tree as (parent, child, length).
        in_tree = torch.zeros(len(used), dtype=torch.bool)
        in_tree[0] = True
        best, parent = dist[0].clone(), torch.zeros(len(used), dtype=torch.long)
        edges = []
        for _ in range(len(used) - 1):
            child = int(torch.where(in_tree, torch.inf, best).argmin())
            edges.append((int(parent[child]), child, float(best[child])))
            in_tree[child] = True
            closer = dist[child] < best
            parent[closer] = child
            best = torch.minimum(best, dist[child])
        q1, q3 = np.quantile([e[2] for e in edges], [0.25, 0.75])
        cut = q3 + 3.0 * (q3 - q1)
        root = list(range(len(used)))

        def find(a: int) -> int:
            while root[a] != a:
                a = root[a]
            return a

        for a, b, length in edges:
            if length <= cut:
                root[find(a)] = find(b)
        groups = torch.tensor([find(i) for i in range(len(used))])
        group_ids = torch.unique(groups)
        group_sizes = torch.stack([sizes[used[groups == g]].sum() for g in group_ids])
        largest = group_ids[group_sizes.argmax()]
        for g, size in zip(group_ids, group_sizes):
            # The largest group is the main data and is never an outlier group.
            if g == largest or size >= self.outlier_group_share * self.X.shape[0]:
                continue
            members = used[groups == g]
            self.unit_saliency[members.to(self.unit_saliency.device)] = False
            mask = torch.isin(self.ins_unit_assign, members.to(self.ins_unit_assign.device))
            self.inst_saliency[mask.to(self.inst_saliency.device)] = False

    def _find_inunit_outliers(self):
        if self.inst_saliency.numel() == 0:
            self.inst_saliency = torch.ones(self.X.shape[0], dtype=torch.bool)

        for i in torch.unique(self.ins_unit_assign):
            indices = torch.where(self.ins_unit_assign == i)[0]
            dists = self.ins_unit_dist[indices]
            if self.inunit_outlier_thresh is None:
                q1, q3 = torch.quantile(dists, torch.tensor([0.25, 0.75]))
                unit_thresh = q3 + self.inunit_whisker_factor * (q3 - q1)
            else:
                unit_thresh = torch.quantile(dists, self.inunit_outlier_thresh / 100)
            outlier_insts = indices[dists > unit_thresh]
            self.inst_saliency[outlier_insts] = False

    def salient_inst_index(self) -> torch.Tensor:
        return torch.where(self.inst_saliency == True)[0]

    def salient_unit_index(self) -> torch.Tensor:
        return torch.where(self.unit_saliency == True)[0]

    def salient_insts(self) -> torch.Tensor:
        return self.X[self.inst_saliency]

    def salient_units(self) -> torch.Tensor:
        return self.W[self.unit_saliency]

    def inst_to_unit_mapping(self) -> torch.Tensor:
        return torch.stack((torch.arange(self.X.shape[0]), self.ins_unit_assign))

