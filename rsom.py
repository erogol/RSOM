"""
Rectifying Self Organazing Maps a.k.a RSOM

RSOM is a clustering and outlier detection method that is predicated with
old Self Organazing Maps.

It includes Batch and Stochastic learning rules. There are two different
implementations. One is based on Numpy and tthe other is Theano. If you have
tall and wide data matrix, we suggest to use Theano version. Otherwise
Numpy version is faster. You can also use GPU with Theano but you need to
set Theano configurations.

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

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
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
        inunit_outlier_thresh: float = 95,
        dist: str = "euclidean",
        log_full_data_cost: bool = False,
        steps_to_full_data_cost: int = -1,  # TODO: number of steps to compute full data cost
    ):
        """Rectifying Self Organizing Maps. RSOM is a clustering and outlier detection method that is predicated with old Self Organizing Maps.

        Args:
            data: Input data.
            num_units: Number of units.
            height: Height of the map.
            width: Width of the map.
            alpha_max: Maximum learning rate.
            alpha_min: Minimum learning rate.
            set_count_activations: Whether to count activations.
            set_outlier_unit_det: Whether to detect outlier units.
            set_inunit_outlier_det: Whether to detect in-unit outliers.
            outlier_unit_thresh: Threshold for outlier unit detection.
            inunit_outlier_thresh: Threshold for in-unit outlier detection.
            dist: Distance metric. Can be "euclidean" or "cosine".
            log_full_data_cost: Whether to log full data cost or use exponential moving average. Computing full data
                cost is expensive and might cause OOM.

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
            u, s, v = torch.svd(self.X)
            ratio = s[0] / s[1]
            self.height = min(
                self.num_units, int(np.ceil(np.sqrt(self.num_units / ratio)))
            )
            self.width = int(np.ceil(self.num_units / self.height))
            self.num_units = self.height * self.width
        logging.info(
            f"Estimated map size is -> height = {self.height}, width = {self.width}"
        )

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
        X2 = (X**2).sum(1).unsqueeze(1)
        D = -2 * torch.mm(self.W, X.t()) + (self.W**2).sum(1).unsqueeze(1) + X2.t()
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

    def set_params(self, num_epoch: int) -> dict:
        U = {"alphas": [], "H_maps": [], "radiuses": []}

        dist_map = torch.zeros(self.num_units, self.num_units)
        radius = np.ceil(1 + np.floor(min(self.width, self.height) - 1) / 2) - 1
        for u in range(self.num_units):
            dist_map[u, :] = self.find_neighbors(u, self.num_units)

        for epoch in range(num_epoch):
            alpha = self.alpha_max - self.alpha_min
            alpha = alpha * (num_epoch - epoch) / num_epoch + self.alpha_min
            radius = np.ceil(1 + np.floor(min(self.width, self.height) - 1) / 2) - 1
            radius = radius * (num_epoch - epoch) / (num_epoch - 1) - 1
            radius = max(radius, 0)

            neigh_updt_map = alpha * (1 - dist_map / float((1 + radius)))
            neigh_updt_map[dist_map > radius] = 0

            U["H_maps"].append(neigh_updt_map)
            U["alphas"].append(alpha)
            U["radiuses"].append(radius)

        return U

    def train_batch(
        self,
        num_epoch: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: bool = True,
    ):
        """
        Args:
            num_epoch: number of epochs to train
            batch_size: number of samples to train in one batch
            verbose: if True, print the progress of training
        """
        if num_epoch is None:
            num_epoch = 500 * self.num_units

        if batch_size is None:
            batch_size = self.X.shape[0]

        logger.info("Learning...")
        U = self.set_params(num_epoch)

        X2 = None
        if batch_size == self.X.shape[0]:
            X2 = (self.X**2).sum(1).unsqueeze(1)

        for epoch in range(num_epoch):
            logger.info(f"Epoch --- {epoch}")
            update_rate = U["H_maps"][epoch]
            learn_rate = U["alphas"][epoch]

            shuffle_indices = torch.randperm(self.X.shape[0])
            win_counts = torch.zeros(self.num_units)

            batches = torch.split(shuffle_indices, batch_size)
            num_steps = len(batches)

            for step, batch_indices in enumerate(batches):
                batch_data = self.X[batch_indices, :]
                D = self._calc_distance(batch_data, X2)
                BMU = (D == D.min(0)[0][None, :]).float().t()

                win_counts += BMU.sum(dim=0)

                if self.set_count_activations:
                    self.activations += win_counts

                A = torch.mm(BMU, update_rate)
                S = A.sum(0)
                non_zeros = S.nonzero().squeeze()

                self.W.data[non_zeros] = torch.mm(A[:, non_zeros].t(), batch_data) / S[
                    non_zeros
                ].unsqueeze(1)

                self._print_cost(X2, D, epoch, num_epoch, step, num_steps)

            if self.set_outlier_unit_det:
                self._update_unit_saliency(win_counts, update_rate, learn_rate)

        if self.set_count_activations:
            self.activations /= self.activations.sum()

        self.assing_to_units()

        if self.set_outlier_unit_det:
            self._find_outlier_units()

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
                self.avg_cost = self.avg_cost * 0.01 + batch_cost * 0.99
            cost = self.avg_cost

        logger.info(
            f"epoch {epoch} / {num_epoch} -- step {step} / {num_steps} -- avg-cost: {cost.item():.6f} -- batch-cost: {batch_cost.item():.6f}"
        )

    def _update_unit_saliency(
        self, win_counts: torch.Tensor, update_rate: torch.Tensor, learn_rate: float
    ):
        excitations = (update_rate * win_counts.unsqueeze(1)).sum(dim=0) / learn_rate
        excitations = excitations / excitations.sum()
        single_excitations = win_counts * learn_rate
        single_excitations = single_excitations / single_excitations.sum()
        self.unit_saliency_coeffs += excitations + single_excitations

    def _find_outlier_units(self):
        self.unit_saliency_coeffs /= self.unit_saliency_coeffs.sum()
        self.unit_saliency = (
            self.unit_saliency_coeffs > self.outlier_unit_thresh / self.num_units
        )

        self.inst_saliency = torch.ones(self.X.shape[0], dtype=torch.bool)
        outlier_units = torch.where(self.unit_saliency == False)[0]
        for i in outlier_units:
            self.inst_saliency[torch.where(self.ins_unit_assign == i)[0]] = False

    def _find_inunit_outliers(self):
        if self.inst_saliency.numel() == 0:
            self.inst_saliency = torch.ones(self.X.shape[0], dtype=torch.bool)

        for i in torch.unique(self.ins_unit_assign):
            indices = torch.where(self.ins_unit_assign == i)[0]
            unit_thresh = torch.quantile(
                self.ins_unit_dist[indices], self.inunit_outlier_thresh / 100
            )
            outlier_insts = indices[self.ins_unit_dist[indices] > unit_thresh]
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import som_plot
    import torch
    from sklearn.datasets import load_digits
    from sklearn.preprocessing import StandardScaler

    # Load the digits dataset
    digits = load_digits()
    X = digits.data
    y = digits.target

    # Preprocess the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert to PyTorch tensor
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Initialize and train SOM
    som = SOM(X_tensor, num_units=100, alpha_max=0.05, alpha_min=0.01)
    som.train_batch(num_epoch=1000, batch_size=32, verbose=True)

    # Get the weights and assign instances to units
    W = som.W.detach().numpy()
    som.assing_to_units()

    # Plot scatter plot
    som_plot.som_plot_scatter(W, X_scaled, som.activations.numpy())

    # Plot outlier scatter plot
    som_plot.som_plot_outlier_scatter(
        W,
        X_scaled,
        som.unit_saliency.numpy(),
        som.inst_saliency.numpy(),
        som.activations.numpy(),
    )

    # Plot mapping
    distance_map = (
        som._euq_dist(torch.sum(X_tensor**2, dim=1).unsqueeze(1), X_tensor)
        .detach()
        .numpy()
    )
    distance_map = distance_map.reshape(som.height, som.width)
    som_plot.som_plot_mapping(distance_map)

    plt.show()
