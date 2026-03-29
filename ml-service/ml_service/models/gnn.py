from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GraphSAGE, to_hetero
from torch_geometric.transforms import ToUndirected


class MLPDecoder(nn.Module):
    """MLP decoder: concatenate (cv, job) embeddings -> scalar score."""

    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.lin1 = nn.Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)

    def forward(self, z_cv: torch.Tensor, z_job: torch.Tensor) -> torch.Tensor:
        z = torch.cat([z_cv, z_job], dim=-1)
        z = torch.relu(self.lin1(z))
        return self.lin2(z).squeeze(-1)


def prepare_data_for_gnn(data: HeteroData) -> HeteroData:
    """Add reverse edges so every node type is a message destination.

    This must be called once before building the model (to get metadata)
    and before every forward pass.
    """
    return ToUndirected()(data)


class HeteroGraphSAGE(nn.Module):
    """Heterogeneous GNN for CV-Job matching.

    Architecture:
    1. Per-type linear projections to common ``hidden_channels``
    2. PyG ``GraphSAGE`` wrapped with ``to_hetero()`` for message passing
    3. ``MLPDecoder`` to produce (cv, job) match scores

    Note: input data **must** be preprocessed with ``prepare_data_for_gnn()``
    (adds reverse edges) before calling ``encode()`` or ``forward()``.
    """

    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        hidden_channels: int = 128,
        num_layers: int = 2,
        node_dims: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        if node_dims is None:
            node_dims = {"cv": 386, "job": 386, "skill": 385, "seniority": 6}

        # Per-type projection to common hidden_dim
        self.projections = nn.ModuleDict(
            {ntype: nn.Linear(dim, hidden_channels) for ntype, dim in node_dims.items()}
        )

        # PyG built-in GraphSAGE wrapped for heterogeneous graph
        backbone = GraphSAGE(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=hidden_channels,
        )
        self.gnn = to_hetero(backbone, metadata, aggr="mean")

        self.decoder = MLPDecoder(hidden_channels)

    def encode(self, data: HeteroData) -> dict[str, torch.Tensor]:
        """Compute node embeddings for all node types."""
        x_dict = {}
        for ntype, proj in self.projections.items():
            x_dict[ntype] = proj(data[ntype].x)

        z_dict = self.gnn(x_dict, data.edge_index_dict)
        return z_dict

    def decode(
        self,
        z_dict: dict[str, torch.Tensor],
        cv_indices: torch.Tensor,
        job_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Decode (cv, job) pairs into scores."""
        z_cv = z_dict["cv"][cv_indices]
        z_job = z_dict["job"][job_indices]
        return self.decoder(z_cv, z_job)

    def forward(
        self,
        data: HeteroData,
        cv_indices: torch.Tensor,
        job_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Full forward: encode + decode."""
        z_dict = self.encode(data)
        return self.decode(z_dict, cv_indices, job_indices)
