from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import scipy
import scipy.io
from sklearn.preprocessing import label_binarize
import torch_geometric.transforms as T

from data_utils import even_quantile_labels, to_sparse_tensor

from torch_geometric.datasets import Planetoid, Amazon, Coauthor, Twitch, PPI, Reddit
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import Data
from torch_geometric.utils import stochastic_blockmodel_graph, subgraph, homophily
from os import path

def load_dataset(args):
    '''
    dataset_ind: in-distribution training dataset
    dataset_ood_tr: ood-distribution training dataset as ood exposure
    dataset_ood_te: a list of ood testing datasets or one ood testing dataset
    '''

    if args.dataset in 'arxiv':
        dataset_ind, dataset_ood_tr, dataset_ood_te = load_arxiv_dataset(args.data_dir)

    elif args.dataset in ('cora', 'products', 'wikics'):
        dataset_ind, dataset_ood_tr, dataset_ood_te = load_graph_dataset(args.data_dir, args.dataset, args.ood_type)

    else:
        raise ValueError('Invalid dataname')
    return dataset_ind, dataset_ood_tr, dataset_ood_te


def load_arxiv_dataset(data_dir, time_bound=[2015,2017], inductive=True):
    from ogb.nodeproppred import NodePropPredDataset

    ogb_dataset = NodePropPredDataset(name='ogbn-arxiv', root=f'{data_dir}/ogb')
    edge_index = torch.as_tensor(ogb_dataset.graph['edge_index'])
    node_feat = torch.as_tensor(ogb_dataset.graph['node_feat'])
    
    ### TODO
    print("Loading pretrained LM features (title and abstract) ...")
    LM_emb_path = f"arxiv/arxiv.emb"
    print(f"LM_emb_path: {LM_emb_path}")
    node_feat = torch.from_numpy(np.array(
        np.memmap(LM_emb_path, mode='r',
                    dtype=np.float16,
                    shape=(node_feat.shape[0], 768)))
    ).to(torch.float32)
       
    label = torch.as_tensor(ogb_dataset.labels).reshape(-1, 1)
    year = ogb_dataset.graph['node_year']

    year_min, year_max = time_bound[0], time_bound[1]
    test_year_bound = [2017, 2018, 2019, 2020]

    center_node_mask = (year <= year_min).squeeze(1)
    # import ipdb; ipdb.set_trace()
    if inductive:
        ind_edge_index, _ = subgraph(center_node_mask, edge_index)
    else:
        ind_edge_index = edge_index

    dataset_ind = Data(x=node_feat, edge_index=ind_edge_index, y=label)
    idx = torch.arange(label.size(0))
    dataset_ind.node_idx = idx[center_node_mask]

    center_node_mask = (year <= year_max).squeeze(1) * (year > year_min).squeeze(1)
    if inductive:
        all_node_mask = (year <= year_max).squeeze(1)
        ood_tr_edge_index, _ = subgraph(all_node_mask, edge_index)
    else:
        ood_tr_edge_index = edge_index

    dataset_ood_tr = Data(x=node_feat, edge_index=ood_tr_edge_index, y=label)
    idx = torch.arange(label.size(0))
    dataset_ood_tr.node_idx = idx[center_node_mask]

    dataset_ood_te = []
    for i in range(len(test_year_bound)-1):
        center_node_mask = (year <= test_year_bound[i+1]).squeeze(1) * (year > test_year_bound[i]).squeeze(1)
        if inductive:
            all_node_mask = (year <= test_year_bound[i+1]).squeeze(1)
            ood_te_edge_index, _ = subgraph(all_node_mask, edge_index)
        else:
            ood_te_edge_index = edge_index

        dataset = Data(x=node_feat, edge_index=ood_te_edge_index, y=label)
        idx = torch.arange(label.size(0))
        dataset.node_idx = idx[center_node_mask]
        dataset_ood_te.append(dataset)

    return dataset_ind, dataset_ood_tr, dataset_ood_te


def load_graph_dataset(data_dir, dataname, ood_type):
    transform = T.NormalizeFeatures()
    if dataname in ('cora'):
        # import ipdb; ipdb.set_trace()
        torch_dataset = Planetoid(root=f'{data_dir}Planetoid', split='public',
                              name=dataname, transform=transform)
        dataset = torch_dataset[0]
        tensor_split_idx = {}
        idx = torch.arange(dataset.num_nodes)
        tensor_split_idx['train'] = idx[dataset.train_mask]
        tensor_split_idx['valid'] = idx[dataset.val_mask]
        tensor_split_idx['test'] = idx[dataset.test_mask]
        dataset.splits = tensor_split_idx
        
        ### TODO
        num_nodes = 2708
        tensor_split_idx = {}
        idx = torch.arange(num_nodes)
        from custom_cora import get_raw_text_cora as get_raw_text
        if dataname == "cora":
            dataset, text = get_raw_text(use_text=True, seed=0)
            tensor_split_idx['train'] = idx[dataset.train_mask]
            tensor_split_idx['valid'] = idx[dataset.val_mask]
            tensor_split_idx['test'] = idx[dataset.test_mask]
            dataset.splits = tensor_split_idx
        ### text emb: finetuned debert
        
        print("Loading pretrained LM features (title and abstract) ...")
        LM_emb_path = f"cora/cora.emb"
        print(f"LM_emb_path: {LM_emb_path}")
        node_feat = torch.from_numpy(np.array(
            np.memmap(LM_emb_path, mode='r',
                        dtype=np.float16,
                        shape=(num_nodes, 64)))
        ).to(torch.float32)
        
        dataset.x = node_feat
        ### TODO
        
    elif dataname == 'products':
        from custom_products import load_products
        dataset, ordered_desc = load_products()
        num_nodes = dataset.x.shape[0]
        idx = torch.arange(num_nodes)
        tensor_split_idx = {}
        tensor_split_idx['train'] = idx[dataset.splits['train']]
        tensor_split_idx['valid'] = idx[dataset.splits['valid']]
        tensor_split_idx['test'] = idx[dataset.splits['test']]
        dataset.splits = tensor_split_idx
        # import ipdb; ipdb.set_trace()
        LM_emb_path = f"products/products.emb" # 64
        node_feat = torch.from_numpy(np.array(
        np.memmap(LM_emb_path, mode='r',
                    dtype=np.float16,
                    shape=(num_nodes, 64)))
        ).to(torch.float32)
        # dataset.x = node_feat
    elif dataname == 'wikics':
        from custom_wikics import load_wikics
        dataset, ordered_desc = load_wikics()
        num_nodes = dataset.x.shape[0]
        idx = torch.arange(num_nodes)
        tensor_split_idx = {}
        tensor_split_idx['train'] = idx[dataset.splits['train']]
        tensor_split_idx['valid'] = idx[dataset.splits['valid']]
        tensor_split_idx['test'] = idx[dataset.splits['test']]
        dataset.splits = tensor_split_idx
        # import ipdb; ipdb.set_trace()
        LM_emb_path = f"wikics/wikics.emb" # 64
        node_feat = torch.from_numpy(np.array(
        np.memmap(LM_emb_path, mode='r',
                    dtype=np.float16,
                    shape=(num_nodes, 64)))
        ).to(torch.float32)

    else:
        raise NotImplementedError

    dataset.node_idx = torch.arange(dataset.num_nodes)
    dataset_ind = dataset

    if ood_type == 'label':
        if dataname == 'cora':
            class_t = 3 # class_t 变量作为分界点
        elif dataname == 'arxiv':
            class_t = 10
        elif dataname == 'products':
            class_t = 10
        elif dataname == 'wikics':
            class_t = 3
        label = dataset.y

        center_node_mask_ind = (label > class_t)
        # center_node_mask_ind = (label != 0 )
        idx = torch.arange(label.size(0))
        dataset_ind.node_idx = idx[center_node_mask_ind]

        if dataname in ('cora', 'citeseer', 'pubmed', 'products'):
            split_idx = dataset.splits
        if dataname in ('cora', 'citeseer', 'pubmed', 'arxiv', 'products'):
            tensor_split_idx = {}
            idx = torch.arange(label.size(0))
            for key in split_idx:
                mask = torch.zeros(label.size(0), dtype=torch.bool)
                mask[torch.as_tensor(split_idx[key])] = True
                tensor_split_idx[key] = idx[mask * center_node_mask_ind]
            dataset_ind.splits = tensor_split_idx

        dataset_ood_tr = Data(x=dataset.x, edge_index=dataset.edge_index, y=dataset.y)
        dataset_ood_te = Data(x=dataset.x, edge_index=dataset.edge_index, y=dataset.y)
        # import ipdb; ipdb.set_trace()
        center_node_mask_ood_tr = (label == class_t)
        center_node_mask_ood_te = (label < class_t)
        # center_node_mask_ood_te, center_node_mask_ood_tr = (label == 0 ), (label == 0 )
        # dataset_ood_tr, dataset_ood_te 完全一致，只有node_idx不同， dataset_ood_tr 只有class_t 对应的node
        # dataset_ood_te 是小于class_t 的 node
        # dataset_ind 是大于class_t 的 node
        dataset_ood_tr.node_idx = idx[center_node_mask_ood_tr] 
        dataset_ood_te.node_idx = idx[center_node_mask_ood_te]
        
        # import ipdb; ipdb.set_trace()
        ### TODO, 重构dataset_ood_tr
        # ratio = 0.19 # 0.05， 0.1, 0.2
        
        import random
        # random.sample
        
        ood_x_ = torch.load('./cora/ood_embs.pth') # (#ood_node, 64), (539, 64)
        new_node_num = len(ood_x_)
        indices = list(range(new_node_num))
        random.shuffle(indices)
        
        # sample_cnt = int(ratio * dataset.x.shape[0])
        sample_cnt = new_node_num 
        sample_idx = sorted(indices[:sample_cnt])
        
        # import ipdb; ipdb.set_trace()
        ood_x_ = ood_x_[sample_idx]
        new_edges_ = load_new_edges(offset=dataset.x.shape[0], idx=sample_idx)
        # import ipdb; ipdb.set_trace()
        
        new_x  = torch.cat([dataset.x, ood_x_], dim=0)
        
        new_edges = torch.cat([new_edges_, dataset.edge_index], dim=1)
        new_ood_mask_tr = torch.ones(sample_cnt, dtype=torch.bool) 
        center_node_mask_ood_tr = torch.cat([center_node_mask_ood_tr, new_ood_mask_tr])
        
        dataset_ood_tr = Data(x=new_x, edge_index=new_edges, y=-torch.ones(len(new_x)))
        idx = torch.arange(len(new_x))
        dataset_ood_tr.node_idx = idx[center_node_mask_ood_tr]
        ### TODO
        # import ipdb; ipdb.set_trace()
    else:
        raise NotImplementedError

    return dataset_ind, dataset_ood_tr, dataset_ood_te

import sys
sys.path.append('..')
from utils.util import get_text_data

def load_new_edges(dataset='cora', offset=2708, idx=None):
    ood_text, id_text = get_text_data(dataset)
    m, n = len(ood_text), len(id_text)
    edge_scores = torch.from_numpy(np.load(f'{dataset}' + '/' + dataset + '_edge_scores.npy'))
    top_k_edges = []
    assert offset == n
    for i in range(m):
        scores = edge_scores[i]
        scores[i] = -float('inf')
        top_k_indices = torch.topk(scores, 3).indices

        for idx in top_k_indices:
            top_k_edges.append([n + i, idx])
            top_k_edges.append([idx, n + i])
    top_k_edges = np.unique(top_k_edges, axis=0).transpose()
        
    return torch.from_numpy(top_k_edges)
    