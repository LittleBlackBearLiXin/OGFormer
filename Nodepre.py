import torch
from torch_geometric.data import Data
import numpy as np



def preprocess(data,loop=False):
    adj_sparse = torch.sparse_coo_tensor(data.edge_index,
                                         torch.ones(data.edge_index.shape[1]),
                                         (data.num_nodes, data.num_nodes))
    adj_dense = adj_sparse.to_dense()
    if loop:
        adj_dense = adj_dense + torch.eye(adj_dense.shape[0])
    data = Data(x=data.x,
                adjacency=adj_dense,
                y=data.y,
                train_mask=data.train_mask,
                val_mask=data.val_mask,
                test_mask=data.test_mask)
    return data


def convert_and_create_data_objects(x, y, edge_index, train_mask, val_mask, test_mask):
    """
    Convert the dataset into cross-validation folds and create Data objects for each fold.

    Args:
        x (torch.Tensor): Node feature matrix of shape [num_nodes, num_features].
        y (torch.Tensor): Node labels of shape [num_nodes].
        edge_index (torch.Tensor): Edge indices of shape [2, num_edges].
        train_mask (torch.Tensor): Training mask of shape [num_nodes, num_folds].
        val_mask (torch.Tensor): Validation mask of shape [num_nodes, num_folds].
        test_mask (torch.Tensor): Test mask of shape [num_nodes, num_folds].

    Returns:
        List[Data]: A list of PyTorch Geometric Data objects, one for each fold.
    """
    num_folds = train_mask.shape[1]
    data_splits = []

    for fold in range(num_folds):
        # Extract masks for the current fold
        current_train_mask = train_mask[:, fold].bool()
        current_val_mask = val_mask[:, fold].bool()
        current_test_mask = test_mask[:, fold].bool()

        # Create adjacency matrix (assuming undirected graph)
        num_nodes = x.size(0)
        adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
        row, col = edge_index
        adjacency[row, col] = True
        adjacency[col, row] = True  # Assuming undirected graph

        # Create the Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=current_train_mask,
            val_mask=current_val_mask,
            test_mask=current_test_mask
        )

        data_splits.append(data)

    return data_splits




