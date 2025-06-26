from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork,Amazon,Coauthor,WikiCS
from typing import  Any
from torch_geometric.data import Data

def load_planetoid(name: str, use_geom: bool) -> Any:
    if use_geom:
        if name == 'Cora':
            return Planetoid(root=r'./tmp/cora-geom', name="Cora", split='geom-gcn')
        elif name == 'Citeseer':
            return Planetoid(root=r'./tmp/citeseer-geom', name="Citeseer", split='geom-gcn')
        elif name == 'Pubmed':
            return Planetoid(root=r'./tmp/pubmed-geom', name="Pubmed", split='geom-gcn')
    else:
        if name == 'Cora':
            return Planetoid(root=r'./tmp/cora', name="Cora")
        elif name == 'Citeseer':
            return Planetoid(root=r'./tmp/citeseer', name="Citeseer")
        elif name == 'Pubmed':
            return Planetoid(root=r'./tmp/pubmed', name="Pubmed")
    raise ValueError(f"Not '{name}'  Planetoid dataset")

def load_webkb(name: str) -> Any:
    if name == 'WebKB':
        return WebKB(root=r'./tmp/WebKB', name="Texas")
    elif name == 'Cornell':
        return WebKB(root=r'./tmp/Cornell', name="Cornell")
    elif name == 'Wisconsin':
        return WebKB(root=r'./tmp/Wisconsin', name="Wisconsin")
    raise ValueError(f"Not '{name}'  WebKB dataset")

def load_wikipedia_network(name: str) -> Any:
    if name == 'Chameleon':
        return WikipediaNetwork(root=r'./tmp/Chameleon', name='chameleon',geom_gcn_preprocess=True)
    elif name == 'Squirrel':
        return WikipediaNetwork(root=r'./tmp/Squirrel', name='squirrel',geom_gcn_preprocess=True)
    raise ValueError(f"Not '{name}'  WikipediaNetwork dataset")




def load_Amazon(name: str) -> Any:
    if name == 'Computers':
        return Amazon(root=r'./tmp/Computers', name='Computers')
    elif name == 'Photo':
        return Amazon(root=r'./tmp/Photo', name="Photo")
    raise ValueError(f"not '{name}'  Amazon dataset。")

def load_Coauthor(name: str) -> Any:
    if name == 'CS':
        return Coauthor(root=r'./tmp/CS', name='CS')
    elif name == 'Physics':
        return Coauthor(root=r'./tmp/Physics', name='Physics')
    raise ValueError(f"not '{name}'  Coauthor datset。")

def load_WikiCS() -> Any:
    return WikiCS(root=r'./tmp/WikiCS')

def load_data(name: str,
              use_planetoid: bool = True,
              use_geom: bool = False,
              use_webkb: bool = False,
              use_wikipedia_network: bool = False,
              use_Amazon: bool=False,
              use_Coauthor: bool=False,
              use_WikiCS: bool=False,
              ) -> Any:
    if use_planetoid:
        return load_planetoid(name, use_geom)
    elif use_webkb:
        return load_webkb(name)
    elif use_wikipedia_network:
        return load_wikipedia_network(name)
    elif use_Amazon:
        return load_Amazon(name)
    elif use_Coauthor:
        return load_Coauthor(name)
    elif use_WikiCS:
        return load_WikiCS()
    else:
        raise ValueError(f"not '{name}' ")

import torch
from torch_geometric.utils import index_to_mask
import numpy as np
def random_planetoid_splits(data, num_classes,randomstate):
    # Set new random planetoid splits:
    # * 20 * num_classes labels for training
    # * 500 labels for validation
    # * 1000 labels for testing
    np.random.seed(randomstate)
    torch.manual_seed(randomstate)
    np.random.seed(randomstate)


    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)

    rest_index = torch.cat([i[20:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:500], size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index[500:1500], size=data.num_nodes)

    return data


def load_large_dataset(data_dir, name):

    if name == 'amazon-photo':
        torch_dataset = load_data(name='Photo', use_planetoid=False, use_geom=False, use_webkb=False, use_actor=False,
                  use_wikipedia_network=False,
                  use_Amazon=True,use_Coauthor=False)
    elif name == 'amazon-computer':
        torch_dataset = load_data(name='Computers', use_planetoid=False, use_geom=False, use_webkb=False, use_actor=False,
                  use_wikipedia_network=False,
                  use_Amazon=True)
    elif name == 'coauthor-physics':
        torch_dataset = load_data(name='Physics', use_planetoid=False, use_geom=False, use_webkb=False, use_actor=False,
                  use_wikipedia_network=False,
                  use_Amazon=False,use_Coauthor=True)
    elif name == 'coauthor-cs':
        torch_dataset = load_data(name='CS', use_planetoid=False, use_geom=False, use_webkb=False, use_actor=False,
                  use_wikipedia_network=False,
                  use_Amazon=False, use_Coauthor=True)


    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes

    splits = load_fixed_splits(data_dir, name)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    valid_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[splits['train']] = True
    valid_mask[splits['valid']] = True
    test_mask[splits['test']] = True

    dataset = Data(x=node_feat, edge_index=edge_index, y=label,
                   train_mask=train_mask, val_mask=valid_mask, test_mask=test_mask)

    return dataset


def load_fixed_splits(data_dir, name):

    splits = {}
    idx = np.load(f'{data_dir}/{name}_split.npz')
    splits['train'] = torch.from_numpy(idx['train'])
    splits['valid'] = torch.from_numpy(idx['valid'])
    splits['test'] = torch.from_numpy(idx['test'])
    return splits



import os
def load_chameleon_squirrel_dataset(data_dir, dataset_name):

    DATAPATH = data_dir
    print(f"Loading dataset: {dataset_name}")

    def load_wiki_new(name):
        path = os.path.join(DATAPATH, f'{data_dir}/{dataset_name}_filtered.npz')  # 修正路径
        data = np.load(path)

        node_feat = data['node_features']
        labels = data['node_labels']
        edges = data['edges']
        if edges is None:
            raise ValueError(f"Edges data not found in {dataset_name} dataset.")

        edge_index = edges.T

        assert edge_index.shape[0] == 2, f"Invalid edge_index shape: {edge_index.shape}"


        edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        node_feat = torch.as_tensor(node_feat, dtype=torch.float)
        labels = torch.as_tensor(labels, dtype=torch.long)

        dataset = Data(x=node_feat, edge_index=edge_index, y=labels)

        return dataset

    def load_hetero_dataset(name):
        path = os.path.join(DATAPATH, f'{data_dir}/{dataset_name}.npz')  # 修正路径
        data = np.load(path)

        node_feat = data['node_features']
        labels = data['node_labels']
        edges = data['edges']
        if edges is None:
            raise ValueError(f"Edges data not found in {dataset_name} dataset.")

        edge_index = edges.T

        assert edge_index.shape[0] == 2, f"Invalid edge_index shape: {edge_index.shape}"


        edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        node_feat = torch.as_tensor(node_feat, dtype=torch.float)
        labels = torch.as_tensor(labels, dtype=torch.long)

        dataset = Data(x=node_feat, edge_index=edge_index, y=labels)

        return dataset

    def load_fixed_splits(name):
        splits_lst = []

        if name in ["chameleon", "squirrel"]:
            file_path = os.path.join(DATAPATH, f'{data_dir}/{dataset_name}_filtered.npz')
            data = np.load(file_path)

            train_masks = data["train_masks"]  # (10, N)，
            val_masks = data["val_masks"]
            test_masks = data["test_masks"]

            N = train_masks.shape[1]  #
            node_idx = np.arange(N)

            print(f"Train mask size: {train_masks.shape}, Expected size: ({N}, )")
            print(f"Validation mask size: {val_masks.shape}, Expected size: ({N}, )")
            print(f"Test mask size: {test_masks.shape}, Expected size: ({N}, )")

            assert train_masks.shape[1] == N, f"Train mask size mismatch: {train_masks.shape[1]} != {N}"
            assert val_masks.shape[1] == N, f"Validation mask size mismatch: {val_masks.shape[1]} != {N}"
            assert test_masks.shape[1] == N, f"Test mask size mismatch: {test_masks.shape[1]} != {N}"

            train_mask = torch.zeros((N, 10), dtype=torch.bool)
            val_mask = torch.zeros((N, 10), dtype=torch.bool)
            test_mask = torch.zeros((N, 10), dtype=torch.bool)

            for i in range(train_masks.shape[0]):
                train_mask[:, i] = torch.zeros(N, dtype=torch.bool)
                train_mask[train_masks[i], i] = 1  #

                #
                val_mask[:, i] = torch.zeros(N, dtype=torch.bool)
                val_mask[val_masks[i], i] = 1

                test_mask[:, i] = torch.zeros(N, dtype=torch.bool)
                test_mask[test_masks[i], i] = 1

            splits_lst.append({
                "train_mask": train_mask,
                "val_mask": val_mask,
                "test_mask": test_mask
            })
        elif name in ["roman_empire", "amazon_ratings"]:
            file_path = os.path.join(DATAPATH, f'{data_dir}/{dataset_name}.npz')  #
            data = np.load(file_path)

            train_masks = data["train_masks"]  # (10, N)，
            val_masks = data["val_masks"]  #
            test_masks = data["test_masks"]  #

            N = train_masks.shape[1]  #
            node_idx = np.arange(N)

            print(f"Train mask size: {train_masks.shape}, Expected size: ({N}, )")
            print(f"Validation mask size: {val_masks.shape}, Expected size: ({N}, )")
            print(f"Test mask size: {test_masks.shape}, Expected size: ({N}, )")

            assert train_masks.shape[1] == N, f"Train mask size mismatch: {train_masks.shape[1]} != {N}"
            assert val_masks.shape[1] == N, f"Validation mask size mismatch: {val_masks.shape[1]} != {N}"
            assert test_masks.shape[1] == N, f"Test mask size mismatch: {test_masks.shape[1]} != {N}"

            train_mask = torch.zeros((N, 10), dtype=torch.bool)
            val_mask = torch.zeros((N, 10), dtype=torch.bool)
            test_mask = torch.zeros((N, 10), dtype=torch.bool)

            for i in range(train_masks.shape[0]):
                train_mask[:, i] = torch.zeros(N, dtype=torch.bool)
                train_mask[train_masks[i], i] = 1

                val_mask[:, i] = torch.zeros(N, dtype=torch.bool)
                val_mask[val_masks[i], i] = 1

                test_mask[:, i] = torch.zeros(N, dtype=torch.bool)
                test_mask[test_masks[i], i] = 1

            splits_lst.append({
                "train_mask": train_mask,
                "val_mask": val_mask,
                "test_mask": test_mask
            })


        else:
            raise ValueError('Invalid dataset name')

        return splits_lst

    if dataset_name in ['chameleon', 'squirrel']:

        dataset = load_wiki_new(dataset_name)
    elif dataset_name in ["roman_empire", "amazon_ratings"]:
        dataset = load_hetero_dataset(dataset_name)
    else:
        raise ValueError('Invalid dataset name')

    splits = load_fixed_splits(dataset_name)
    dataset.train_mask = splits[0]["train_mask"]
    dataset.val_mask = splits[0]["val_mask"]
    dataset.test_mask = splits[0]["test_mask"]

    return dataset




