import torch
import torch.nn as nn
from typing import Union, Tuple
from scaling_model.models.utils import (
    build_fully_connected_graphs,
    SinusoidalRBFLayer,
    CosineCutoff,
)


class PaiNNMessageBlock(nn.Module):
    """
    Message block in PaiNN.
    """

    def __init__(self, num_features: int = 128, num_rbf_features: int = 20) -> None:
        """
        Args:
            num_features: Size of the node embeddings (scalar features) and
                vector features.
            num_rbf_features: Number of radial basis functions to represent
                distances.
        """
        super().__init__()
        self.num_features = num_features
        self.num_rbf_features = num_rbf_features

        self.scalar_network = nn.Sequential(
            nn.Linear(in_features=self.num_features, out_features=self.num_features),
            nn.SiLU(),
            nn.Linear(
                in_features=self.num_features, out_features=3 * self.num_features
            ),
        )
        self.rbf_network = nn.Linear(
            in_features=self.num_rbf_features, out_features=3 * self.num_features
        )

    def forward(
        self,
        idx_i: Union[torch.IntTensor, torch.LongTensor],
        idx_j: Union[torch.IntTensor, torch.LongTensor],
        rel_dir: torch.Tensor,
        rel_dist_cut: torch.Tensor,
        rbf_features: torch.Tensor,
        scalar_features: torch.Tensor,
        vector_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of PaiNN message block.

        Args:
            idx_i: torch.Tensor of size [num_edges] with node indexes.
            idx_j: torch.Tensor of size [num_edges] with neighbor indexes.
            rel_dir: torch.Tensor of size [num_edges, 3] with directions
                between nodes.
            rel_dist_cut: torch.Tensor of size [num_edges] with cosine cutted
                distances between nodes.
            rbf_features: torch.Tensor of size [num_edges, num_rbf_features].
            scalar_features: torch.Tensor of size [num_nodes, num_features]
                with scalar features of each node.
            vector_features: torch.Tensor of size [num_nodes, num_features, 3]
                with vector features of each node.

        Returns:
            A tuple with scalar features and vector features, i.e., tensors
            with sizes [num_nodes, num_features] and
            [num_nodes, num_features, 3], respectively.
        """
        # Propagate scalar features
        phi = self.scalar_network(scalar_features)  # [num_nodes, 3*num_features]

        # Propagate rbf features
        W = self.rbf_network(rbf_features) * rel_dist_cut.unsqueeze(
            -1
        )  # [num_edges, 3*num_features]

        # Multiply phi and W and split
        phi_W = phi[idx_j] * W  # [num_edges, 3*num_features]
        phi_W_vv, phi_W_ss, phi_W_vs = torch.split(  # [num_edges, num_features]
            phi_W, self.num_features, dim=-1
        )

        # Compute scalar residuals
        scalar_residuals = torch.zeros_like(
            scalar_features
        )  # [num_nodes, num_features]
        scalar_residuals.index_add_(
            dim=0, index=idx_i, source=phi_W_ss
        )  # [num_nodes, num_features]

        # FARLIG KODE
        index_counts = torch.bincount(idx_i).float() 
        idx_cnt = index_counts.unsqueeze(-1).expand_as(scalar_residuals)
        scalar_residuals = scalar_residuals/idx_cnt

        # Compute vector residuals
        vector_residuals = torch.zeros_like(
            vector_features
        )  # [num_nodes, num_features, 3]
        vector_residuals_per_edge = vector_features[  # [num_edges, num_features, 3]
            idx_j
        ] * phi_W_vv.unsqueeze(-1) + phi_W_vs.unsqueeze(-1) * rel_dir.unsqueeze(-2)

        vector_residuals.index_add_(  # [num_nodes, num_features, 3]
            dim=0, index=idx_i, source=vector_residuals_per_edge
        )

        #Aggregation for sum_j over Vector features. Verify wheter the counter should be over i or j
        vec_cnt = torch.concat([idx_cnt.unsqueeze(-1)]*3,axis=2)
        vector_residuals = vector_residuals/vec_cnt

        scalar_features = (
            scalar_features + scalar_residuals
        )  # [num_nodes, num_features]
        vector_features = (
            vector_features + vector_residuals
        )  # [num_nodes, num_features, 3]

        return scalar_features, vector_features


class PaiNNUpdateBlock(nn.Module):
    """
    Update block in PaiNN.
    """

    def __init__(self, num_features: int = 128) -> None:
        """
        Args:
            num_features: Size of the node embeddings (scalar features) and
                vector features.
        """
        super().__init__()
        self.num_features = num_features

        self.U = nn.Linear(
            in_features=self.num_features, out_features=self.num_features, bias=False
        )
        self.V = nn.Linear(
            in_features=self.num_features, out_features=self.num_features, bias=False
        )
        self.scalar_vector_network = nn.Sequential(
            nn.Linear(
                in_features=2 * self.num_features, out_features=self.num_features
            ),
            nn.SiLU(),
            nn.Linear(
                in_features=self.num_features, out_features=3 * self.num_features
            ),
        )

    def forward(
        self, scalar_features: torch.Tensor, vector_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of PaiNN update block.

        Args:
            scalar_features: torch.Tensor of size [num_nodes, num_features]
                with scalar features of each node.
            vector_features: torch.Tensor of size [num_nodes, num_features, 3]
                with vector features of each node.

        Returns:
            A tuple with scalar features and vector features, i.e., tensors
            with sizes [num_nodes, num_features] and
            [num_nodes, num_features, 3], respectively.
        """
        U_vector_features = self.U(vector_features.movedim(-2, -1)).movedim(
            -2, -1
        )  # [num_nodes, num_features, 3]
        V_vector_features = self.V(vector_features.movedim(-2, -1)).movedim(
            -2, -1
        )  # [num_nodes, num_features, 3]

        a = self.scalar_vector_network(  # [num_nodes, 3*num_features]
            torch.cat(
                [torch.linalg.vector_norm(V_vector_features, dim=-1), scalar_features],
                dim=-1,
            )
        )
        a_vv, a_sv, a_ss = torch.split(
            a, self.num_features, dim=-1
        )  # [num_nodes, num_features]

        vector_residuals = U_vector_features * a_vv.unsqueeze(
            -1
        )  # [num_nodes, num_features, 3]
        scalar_residuals = a_ss + a_sv * torch.sum(
            U_vector_features * V_vector_features, dim=-1
        )
        scalar_features = (
            scalar_features + scalar_residuals
        )  # [num_nodes, num_features]
        vector_features = (
            vector_features + vector_residuals
        )  # [num_nodes, num_features, 3]

        return scalar_features, vector_features


class PaiNN(nn.Module):
    """
    Polarizable Atom Interaction Neural Network with PyTorch.
    """

    def __init__(
        self,
        num_message_passing_layers: int = 3,
        num_features: int = 128,
        num_rbf_features: int = 20,
        num_unique_atoms: int = 118,
        cutoff_dist: float = 3.0,
    ) -> None:
        """
        Args:
            num_message_passing_layers: Number of message passing layers in
                the PaiNN model.
            num_features: Size of the node embeddings (scalar features) and
                vector features.
            num_rbf_features: Number of radial basis functions to represent
                distances.
            num_unique_atoms: Number of unique atoms in the data that we want
                to learn embeddings for.
            cutoff_dist: Euclidean distance threshold for determining whether
                two nodes (atoms) are neighbours.
        """
        super().__init__()
        self.num_message_passing_layers = num_message_passing_layers
        self.num_features = num_features
        self.num_rbf_features = num_rbf_features
        self.num_unique_atoms = num_unique_atoms
        self.cutoff_dist = cutoff_dist

        self.atom_embedding = nn.Embedding(
            num_embeddings=self.num_unique_atoms + 1,
            embedding_dim=num_features,
            padding_idx=0,
        )
        self.cosine_cut = CosineCutoff(cutoff_dist=self.cutoff_dist)
        self.radial_basis = SinusoidalRBFLayer(
            num_basis=self.num_rbf_features, cutoff_dist=self.cutoff_dist
        )
        self.message_blocks = nn.ModuleList()
        self.update_blocks = nn.ModuleList()
        for _ in range(self.num_message_passing_layers):
            self.message_blocks.append(
                PaiNNMessageBlock(
                    num_features=self.num_features,
                    num_rbf_features=self.num_rbf_features,
                )
            )
            self.update_blocks.append(PaiNNUpdateBlock(num_features=self.num_features))

    def forward(
        self,
        atoms: Union[torch.IntTensor, torch.LongTensor],
        atom_positions: torch.Tensor,
        graph_indexes: Union[torch.IntTensor, torch.LongTensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of PaiNN.

        Args:
            atoms: torch.Tensor of size [num_nodes] with atom type of each node
                in the graph.
            atom_positions: torch.Tensor of size [num_nodes, 3] with euclidean
                coordinates of each node / atom.
            graph_indexes: torch.Tensor of size [num_nodes] with the graph
                index each node belongs to.

        Returns:
            A tuple with scalar features and vector features, i.e., tensors
            with sizes [num_nodes, num_features] and
            [num_nodes, num_features, 3], respectively.
        """
        scalar_features = self.atom_embedding(atoms)  # [num_nodes, num_features]
        vector_features = torch.zeros(  # [num_nodes, num_features, 3]
            scalar_features.size() + (3,),
            dtype=scalar_features.dtype,
            device=scalar_features.device,
        )

        # Create edge index with fully connected graphs
        # idx_i: node indexes
        # idx_j: neighbor indexes
        idx_i, idx_j = build_fully_connected_graphs(
            graph_indexes
        )  # [num_possible_edges]

        # Relative_positions pos_ij = pos_j - pos_i and distances
        rel_pos = (
            atom_positions[idx_j] - atom_positions[idx_i]
        )  # [num_possible_edges, 3]
        rel_dist = torch.linalg.vector_norm(rel_pos, dim=1)  # [num_possible_edges]

        # Keep only edges shorter than the cutoff
        short_edges = rel_dist < self.cutoff_dist  # [num_possible_edges]
        idx_i, idx_j = idx_i[short_edges], idx_j[short_edges]  # [num_edges]
        rel_pos = rel_pos[short_edges]  # [num_edges, 3]
        rel_dist = rel_dist[short_edges]  # [num_edges]

        # Relative directions, cosine cutted distances, and rbf features
        rel_dir = rel_pos / rel_dist.unsqueeze(-1)  # [num_edges, 3]
        rel_dist_cut = self.cosine_cut(rel_dist)  # [num_edges]
        rbf_features = self.radial_basis(rel_dist)  # [num_edges, num_rbf_features]

        for message, update in zip(self.message_blocks, self.update_blocks):
            scalar_features, vector_features = message(  # ([num_nodes, num_features],
                idx_i,  #  [num_nodes, num_features, 3])
                idx_j,
                rel_dir,
                rel_dist_cut,
                rbf_features,
                scalar_features,
                vector_features,
            )
            scalar_features, vector_features = update(  # ([num_nodes, num_features],
                scalar_features, vector_features  #  [num_nodes, num_features, 3])
            )

        return scalar_features, vector_features
