import math
import torch
import torch.nn as nn
import pandas as pd
from typing import Optional, Tuple, Union
from pytorch_lightning import callbacks
from torchmetrics import MetricCollection, MeanSquaredError, MeanAbsoluteError


class SinusoidalRBFLayer(nn.Module):
    """
    Sinusoidal Radial Basis Function.
    """

    def __init__(self, num_basis: int = 20, cutoff_dist: float = 5.0) -> None:
        """
        Args:
            num_basis: Number of radial basis functions to use.
            cutoff_dist: Euclidean distance threshold for determining whether
                two nodes (atoms) are neighbours.
        """
        super().__init__()
        self.num_basis = num_basis
        self.cutoff_dist = cutoff_dist

        self.register_buffer(
            "freqs", math.pi * torch.arange(1, self.num_basis + 1) / self.cutoff_dist
        )

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Computes sinusoidal radial basis functions for a tensor of distances.

        Args:
            distances: torch.Tensor of distances (any size).

        Returns:
            A torch.Tensor of radial basis functions with size [*, num_basis]
                where * is the size of the input (the distances).
        """
        distances = distances.unsqueeze(-1)
        return torch.sin(self.freqs * distances) / distances


class CosineCutoff(nn.Module):
    """
    Cosine cutoff function.
    """

    def __init__(self, cutoff_dist: float = 5.0) -> None:
        """
        Args:
            cutoff_dist: Euclidean distance threshold for determining whether
                two nodes (atoms) are neighbours.
        """
        super().__init__()
        self.cutoff_dist = cutoff_dist

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Applies cosine cutoff function to input.

        Args:
            distances: torch.Tensor of distances (any size).

        Returns:
            torch.Tensor of distances that has been cut with the cosine cutoff
            function.
        """
        return torch.where(
            distances < self.cutoff_dist,
            0.5 * (torch.cos(distances * math.pi / self.cutoff_dist) + 1),
            0,
        )


class GatedEquivariantBlock(nn.Module):
    """
    Gated Equivariant Block.
    """

    def __init__(
        self,
        num_features: int = 128,
        num_out_features: int = 128,
        hidden_dim: int = 128,
        scalar_activation: Optional[nn.Module] = None,
    ) -> None:
        """
        Args:
            num_features: Size of the node embeddings (scalar features) and
                vector features.
            num_out_features: Number of output features.
            hidden_dim: Size of the hidden layer in the subnetwork that
                transforms both the scalar and vector features.
            scalar_activation: Optional final activation function to use for
                scalar features.
        """
        super().__init__()
        self.num_features = num_features
        self.num_out_features = num_out_features
        self.hidden_dim = hidden_dim
        self.scalar_activation = (
            None if scalar_activation is None else scalar_activation()
        )

        self.W_1 = nn.Linear(
            in_features=self.num_features,
            out_features=self.num_out_features,
            bias=False,
        )
        self.W_2 = nn.Linear(
            in_features=self.num_features,
            out_features=self.num_out_features,
            bias=False,
        )
        self.scalar_vector_network = nn.Sequential(
            nn.Linear(
                in_features=self.num_features + self.num_out_features,
                out_features=self.hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                in_features=self.hidden_dim, out_features=2 * self.num_out_features
            ),
        )

    def forward(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of Gated Equivariant block.

        Args:
            features: A tuple with scalar features and vector features, i.e.,
                tensors with sizes [num_nodes, num_features] and
                [num_nodes, num_features, 3], respectively.

        Returns:
            A tuple with scalar features and vector features, i.e., tensors
            with sizes [num_nodes, num_out_features] and
            [num_nodes, num_out_features, 3], respectively.
        """
        scalar_features, vector_features = features

        W_1_vector_features = self.W_1(vector_features.movedim(-2, -1)).movedim(
            -2, -1
        )  # [num_nodes, num_out_features, 3]
        W_2_vector_features = self.W_2(vector_features.movedim(-2, -1)).movedim(
            -2, -1
        )  # [num_nodes, num_out_features, 3]

        tmp = self.scalar_vector_network(  # [num_nodes, 2*num_out_features]
            torch.cat(
                [
                    torch.linalg.vector_norm(W_2_vector_features, dim=-1),
                    scalar_features,
                ],
                dim=-1,
            )
        )
        tmp_vector, scalar_features = torch.split(
            tmp, self.num_out_features, dim=-1
        )  # [num_nodes, num_out_features]

        vector_features = W_1_vector_features * tmp_vector.unsqueeze(-1)

        if self.scalar_activation is not None:
            scalar_features = self.scalar_activation(scalar_features)

        return scalar_features, vector_features


def build_fully_connected_graphs(
    graph_indexes: Union[torch.IntTensor, torch.LongTensor]
) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Builds edge index where all graphs in the batch are fully connected.

    Args:
        graph_indexes: torch.Tensor of size [num_nodes] with the graph
            index each node belongs to.

    Returns:
        A tensor of size [2, num_possible_edges], i.e., an edge index with
        all possible edges (fully connected graphs).
    """
    # Number of nodes per graph
    _, num_nodes_per_graph = torch.unique(graph_indexes, return_counts=True)

    # Each adjacency matrix is all ones except along the diagonal where there
    # are only zeros
    adjacency_matrices = [
        torch.ones(num_nodes, num_nodes, dtype=int) - torch.eye(num_nodes, dtype=int)
        for num_nodes in num_nodes_per_graph
    ]
    # Create edge index
    edge_index = torch.block_diag(*adjacency_matrices).nonzero().t()
    edge_index = edge_index.to(graph_indexes.device)

    return edge_index


def build_readout_network(
    num_in_features: int,
    num_out_features: int = 1,
    num_layers: int = 2,
    activation: nn.Module = nn.SiLU,
):
    """
    Build readout network.

    Args:
        num_in_features: Number of input features.
        num_out_features: Number of output features (targets).
        num_layers: Number of layers in the network.
        activation: Activation function as a nn.Module.

    Returns:
        The readout network as a nn.Module.
    """
    # Number of neurons in each layer
    num_neurons = [
        num_in_features,
        *[
            max(num_out_features, num_in_features // 2 ** (i + 1))
            for i in range(num_layers - 1)
        ],
        num_out_features,
    ]

    # Build network
    readout_network = nn.Sequential()
    for i, (n_in, n_out) in enumerate(zip(num_neurons[:-1], num_neurons[1:])):
        readout_network.append(nn.Linear(n_in, n_out))
        if i < num_layers - 1:
            readout_network.append(activation())

    return readout_network


def build_gated_equivariant_readout_network(
    num_in_features: int,
    num_out_features: int = 1,
    num_layers: int = 2,
    activation: Optional[nn.Module] = None,
    pyramidal=True,
):
    """
    Build gated equivariant readout network.

    Args:
        num_in_features: Number of input features.
        num_out_features: Number of output features (targets).
        num_layers: Number of layers in the network.
        activation: Optional final activation function to use for
            scalar features in gated equivariant block.
        pyramidal: Boolean indicating whether the network should have a
            pyramidal structure, i.e., that the hidden dim gets halved for each
            layer.

    Returns:
        The gated equivariant readout network as a nn.Module.
    """
    # Number of neurons in each layer
    if pyramidal:
        num_neurons = [
            num_in_features,
            *[
                max(num_out_features, num_in_features // 2 ** (i + 1))
                for i in range(num_layers - 1)
            ],
            num_out_features,
        ]
    else:
        num_neurons = [num_in_features for _ in range(num_layers)] + [num_out_features]

    # Build network
    readout_network = nn.Sequential()
    for i, (n_in, n_out) in enumerate(zip(num_neurons[:-1], num_neurons[1:])):
        scalar_activation = activation if i < num_layers - 1 else None
        readout_network.append(
            GatedEquivariantBlock(
                num_features=n_in,
                num_out_features=n_out,
                hidden_dim=n_in,
                scalar_activation=scalar_activation,
            )
        )

    return readout_network


class ScaleAndShift(nn.Module):
    """
    Module for scaling and shifting (e.g. to undo standardization).
    """

    def __init__(
        self,
        scale: torch.Tensor = torch.tensor(1.0),
        shift: torch.Tensor = torch.tensor(0.0),
        eps: float = 1e-8,
    ) -> None:
        """
        Args:
            scale: torch.Tensor with scale value(s).
            shift: torch.Tensor with shift value(s).
            eps: Small constant to add to scale.
        """
        super().__init__()
        self.register_buffer("scale", scale + eps)
        self.register_buffer("shift", shift)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Scales and shifts input.

        Args:
            input_: torch.Tensor to scale and shift.

        Returns:
            Input tensor with values scaled and shifted.
        """
        return input_ * self.scale + self.shift


class AddAtomicReferences(nn.Module):
    """ "
    Module for adding single-atom reference energies.
    """

    def __init__(self, atom_refs: torch.Tensor) -> None:
        """
        Args:
            atom_refs: torch.Tensor of size [num_atom_types, 1] with atomic
                reference values.
        """
        super().__init__()
        self.atom_refs = nn.Embedding.from_pretrained(atom_refs, freeze=True)

    def forward(
        self,
        atomwise_energies: torch.Tensor,
        atoms: Union[torch.IntTensor, torch.LongTensor],
    ) -> torch.Tensor:
        """
        Add single atom energies.

        Args:
            atomwise_energies: torch.Tensor of size [num_nodes, num_targets]
                with atomwise energies / predictions.
            atoms: torch.Tensor of size [num_nodes] with atom type of each node
                in the graph.

        Returns:
            A torch.Tensor of energies / predictions for each atom where the
            single atom energies have been added.
        """
        return atomwise_energies + self.atom_refs(atoms)


class FinalizePredictions(nn.Module):
    """
    Scales and shifts atomwise predictions with standard deviation and mean of
    training targets and adds atomic reference values.
    """

    def __init__(
        self,
        atom_refs: Optional[torch.Tensor] = None,
        mean: torch.Tensor = torch.tensor(0.0),
        std: torch.Tensor = torch.tensor(1.0),
        eps: float = 1e-8,
    ) -> None:
        """
        Args:
            atom_refs: torch.Tensor of size [num_atom_types, 1] with atomic
                reference values.
            mean: torch.Tensor with mean value to shift predictions by.
            std: torch.Tensor with standard deviation to scale predictions by.
            eps: Small constant to add to scale.
        """
        super().__init__()
        if std.item() == 1.0:
            eps = 0.0

        if mean != 0.0 or std != 1.0:
            self.scale_and_shift = ScaleAndShift(scale=std, shift=mean, eps=eps)
        else:
            self.scale_and_shift = None

        if atom_refs is not None:
            self.add_atom_refs = AddAtomicReferences(atom_refs=atom_refs)
        else:
            self.add_atom_refs = None

    def forward(
        self,
        atomwise_predictions: torch.Tensor,
        atoms: Union[torch.IntTensor, torch.LongTensor],
    ):
        """
        Finalizes atomwise predictions / energies.

        Args:
            atomwise_predictions: torch.Tensor of size [num_nodes, num_targets]
                with atomwise predictions / energies.
            atoms: torch.Tensor of size [num_nodes] with atom type of each node
                in the graph.
        Returns:
            A torch.Tensor of predictions / energies for each atom where the
            predictions have been scaled and shifted with the training mean and
            standard deviation of the target and the atomic energies have been
            added.
        """
        preds = atomwise_predictions
        if self.scale_and_shift is not None:
            preds = self.scale_and_shift(preds)

        if self.add_atom_refs is not None:
            preds = self.add_atom_refs(preds, atoms)

        return preds


class AtomwisePrediction(nn.Module):
    """
    Module for predicting properties as sums of atomic contributions.
    """

    def __init__(
        self,
        num_features: int = 128,
        num_outputs: int = 1,
        num_layers: int = 2,
        mean: torch.Tensor = torch.tensor(0.0),
        std: torch.Tensor = torch.tensor(1.0),
        atom_refs: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Args:
            num_features: Size of the node embeddings (scalar features) and
                vector features.
            num_outputs: Size of output, i.e., the number of targets/properties.
            num_layers: Number of layers in the fully-connected feedforward
                network.
            mean: torch.Tensor with mean value to shift atomwise contributions
                by.
            std: torch.Tensor with standard deviation to scale atomwise
                contributions by.
            atom_refs: torch.Tensor of size [num_atom_types, 1] with atomic
                reference values.
        """
        super().__init__()
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.mean = mean
        self.std = std
        self.atom_refs = atom_refs

        self.readout_network = build_readout_network(
            num_in_features=self.num_features,
            num_out_features=self.num_outputs,
            num_layers=self.num_layers,
            activation=nn.SiLU,
        )
        self.prediction_finalizer = FinalizePredictions(
            atom_refs=self.atom_refs, mean=self.mean, std=self.std, eps=0.0
        )

    def forward(
        self,
        scalar_features: torch.Tensor,
        atoms: Union[torch.IntTensor, torch.LongTensor],
        graph_indexes: Union[torch.IntTensor, torch.LongTensor],
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass of atomwise readout network.

        Args:
            scalar_features: torch.Tensor of size [num_nodes, num_features]
                with scalar features of each node.
            atoms: torch.Tensor of size [num_nodes] with atom type of each node
                in the graph.
            graph_indexes: torch.Tensor of size [num_nodes] with the graph
                index each node belongs to.

        Returns:
            A tensor of size [num_graphs, num_outputs] with predictions for
            each graph.
        """
        num_graphs = torch.unique(graph_indexes).shape[0]
        # Get atomwise contributions
        atomwise_contributions = self.readout_network(
            scalar_features
        )  # [num_nodes, num_outputs]
        atomwise_contributions = self.prediction_finalizer(
            atomwise_contributions,
            atoms,
        )
        # Sum contributions for each graph
        output_per_graph = torch.zeros(
            (num_graphs, self.num_outputs), device=scalar_features.device
        )
        output_per_graph.index_add_(
            dim=0,
            index=graph_indexes,
            source=atomwise_contributions,
        )

        return output_per_graph


class DipoleMomentPrediction(nn.Module):
    """
    Module for predicting the dipole moment molecular property.
    """

    def __init__(
        self,
        num_features: int = 128,
        num_layers: int = 2,
    ) -> None:
        """
        Args:
            num_features: Size of the node embeddings (scalar features) and
                vector features.
            num_layers: Number of layers in the fully-connected feedforward
                network.
        """
        super().__init__()
        self.num_features = num_features
        self.num_layers = num_layers

        self.readout_network = build_gated_equivariant_readout_network(
            num_in_features=self.num_features,
            num_out_features=1,
            num_layers=self.num_layers,
            activation=nn.SiLU,
            pyramidal=False,
        )

    def forward(
        self,
        scalar_features: torch.Tensor,
        vector_features: torch.Tensor,
        atom_positions: torch.Tensor,
        graph_indexes: Union[torch.IntTensor, torch.LongTensor],
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass of gated equivariant readout network.

        Args:
            scalar_features: torch.Tensor of size [num_nodes, num_features]
                with scalar features of each node.
            vector_features: torch.Tensor of size [num_nodes, num_features, 3]
                with vector features of each node.
            atom_positions: torch.Tensor of size [num_nodes, 3] with euclidean
                coordinates of each node / atom.
            graph_indexes: torch.Tensor of size [num_nodes] with the graph
                index each node belongs to.

        Returns
            A tensor of size [num_graphs, 1] with the magnitude of the dipole
                moment of each graph.
        """
        num_graphs = torch.unique(graph_indexes).shape[0]
        scalar_features, vector_features = (
            self.readout_network(  # [num_nodes, 1], [num_nodes, 1, 3]
                (scalar_features, vector_features)
            )
        )
        mu_per_atom = (  # [num_nodes, 3]
            vector_features.squeeze(-2) + scalar_features * atom_positions
        )
        # Sum contributions for each graph                                          # [num_graphs, 3]
        mu_per_graph = torch.zeros((num_graphs, 3), device=scalar_features.device)
        mu_per_graph.index_add_(  # [num_graphs, 3]
            dim=0,
            index=graph_indexes,
            source=mu_per_atom,
        )
        mu_magnitude = torch.linalg.vector_norm(  # [num_graphs, 1]
            mu_per_graph, dim=-1, keepdim=True
        )
        return mu_magnitude


class ElectronicSpatialExtentPrediction(AtomwisePrediction):
    """
    Module for predicting the electronic spatial extent.
    """

    def __init__(
        self,
        num_features: int = 128,
        num_layers: int = 2,
    ) -> None:
        """
        Args:
            num_features: Size of the node embeddings (scalar features) and
                vector features.
            num_layers: Number of layers in the fully-connected feedforward
                network.
        """
        super().__init__(
            num_features=num_features, num_outputs=1, num_layers=num_layers
        )

    def forward(
        self,
        scalar_features: torch.Tensor,
        atom_positions: torch.Tensor,
        graph_indexes: Union[torch.IntTensor, torch.LongTensor],
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass of readout network for predicting the electronic spatial
        extent.

        Args:
            scalar_features: torch.Tensor of size [num_nodes, num_features]
                with scalar features of each node.
            atom_positions: torch.Tensor of size [num_nodes, 3] with euclidean
                coordinates of each node / atom.
            graph_indexes: torch.Tensor of size [num_nodes] with the graph
                index each node belongs to.

        Returns:
            A tensor of size [num_graphs, 1] with predictions for each graph.
        """
        num_graphs = torch.unique(graph_indexes).shape[0]
        atomwise_contributions = self.readout_network(scalar_features)  # [num_nodes, 1]
        norms_squared = (
            torch.linalg.vector_norm(  # [num_nodes, 1]
                atom_positions, dim=-1, keepdim=True
            )
            ** 2
        )

        # Sum contributions for each graph
        output_per_graph = torch.zeros((num_graphs, 1), device=scalar_features.device)
        output_per_graph.index_add_(
            dim=0,
            index=graph_indexes,
            source=atomwise_contributions * norms_squared,
        )

        return output_per_graph


class PredictionWriter(callbacks.BasePredictionWriter):

    def __init__(self, dataloaders=["train", "val", "test"]):
        super().__init__(write_interval="batch")
        self.dataloaders = dataloaders

        self.preds = list()
        self.targets = list()
        self.split = list()
        self.dataloader_idx_to_split = {
            i: split for i, split in enumerate(self.dataloaders)
        }
        self.metrics = {}

    def init_metrics(self, split):
        self.metrics[split] = MetricCollection(
            {
                f"rmse_{split}": MeanSquaredError(squared=False),
                f"mae_{split}": MeanAbsoluteError(),
            }
        )

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        # Get data slit, predictions, and targets.
        split = self.dataloader_idx_to_split[dataloader_idx]
        preds = outputs
        targets = batch.y

        # Move to cpu
        preds = preds.detach().cpu()
        targets = targets.detach().cpu()

        # Initialize metrics
        if batch_idx == 0:
            self.init_metrics(split=split)

        # Update metrics
        self.metrics[split].update(preds, targets)

        # Store metrics and targets
        self.preds.append(preds)
        self.targets.append(targets)
        self.split.extend([split] * len(preds))

    def on_predict_end(self, trainer, pl_module):
        # Compute and save metrics
        merged_metrics = {}
        for metrics in self.metrics.values():
            merged_metrics.update(metrics.compute())

        for key, value in merged_metrics.items():
            merged_metrics[key] = value.detach().cpu().item()

        df_metrics = pd.DataFrame(merged_metrics, index=[trainer.datamodule.target])
        df_metrics.to_csv(f"{trainer.logger.log_dir}/metrics.csv")

        # Collect results in dataframe
        preds = torch.cat(self.preds, dim=0)
        targets = torch.cat(self.targets, dim=0)
        columns = [
            f"pred_{trainer.datamodule.target:02}",
            f"target_{trainer.datamodule.target:02}",
        ]
        df = pd.DataFrame(
            torch.cat([preds, targets], dim=1).detach().cpu().tolist(), columns=columns
        )

        # Add one-hot encoding to indicate data split
        df = df.join(pd.get_dummies(self.split, dtype=float))
        df.to_csv(f"{trainer.logger.log_dir}/predictions.csv", index=False)
