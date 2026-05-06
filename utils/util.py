import json
from torch_geometric.utils import k_hop_subgraph, to_networkx
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from torch.nn import functional as F
# from zhipuai import ZhipuAI
import numpy as np
import random 
import os

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['node_id'] = idx
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])
    
    
def get_response(prompt=None):
    client = ZhipuAI(api_key="2845fb7472a796c29f8fb8d377e99eef.NorzFqCXVXgeKNst")  # 请填写您自己的APIKey
    response = client.chat.completions.create(
        model="glm-4-plus",  # 请填写您要调用的模型名称
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    print(response.choices[0].message)
    return response

def load_json_str(json_str):
    data = json.loads(json_str)
    return data


def extract_k_hops(data, texts, center_node, num_hops=2, max_neighbors=10):
    subset, edge_index, _, _ = k_hop_subgraph(
        center_node, 
        num_hops=num_hops, 
        edge_index=data.edge_index, 
    )
       
    non_center_nodes = subset[subset != center_node]
    sampled_nodes = torch.cat((
        torch.tensor([center_node]),  # 保证中心节点始终在
        non_center_nodes[torch.randperm(len(non_center_nodes))[:max_neighbors]]
    ))
    # print("max neighbors", max_neighbors)
    mask = torch.isin(edge_index[0], sampled_nodes) & torch.isin(edge_index[1], sampled_nodes)
    subset = sampled_nodes
    sampled_edge_index = edge_index[:, mask]
    
    # relabel
    node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(subset)}
    new_edge_index = sampled_edge_index.clone()
    new_edge_index[0] = torch.tensor([node_mapping[i.item()] for i in sampled_edge_index[0]])
    new_edge_index[1] = torch.tensor([node_mapping[i.item()] for i in sampled_edge_index[1]])
    new_texts = [texts[node] for node in subset.tolist()]
    
    # print(len(node_mapping))
    # print(new_edge_index.shape)
    # print(len(new_texts))
    return new_edge_index, new_texts, subset.tolist()

def prompt_background(texts):
    n = len(texts)
    bg = ""
    for i in range(n):
        bg += f"Paper {i+1}:\n"
        bg += texts[i]
        bg += '\n\n'  
    # print(bg)
    return bg

def get_bg_papers(dataset, text, center_id, max_nei=5):
    edges, texts, ids = extract_k_hops(dataset, text, center_id, max_neighbors=max_nei)
    bgs = prompt_background(texts[1:])
    return bgs, ids[1:]


def get_cora_casestudy(SEED=0):
    data_X, data_Y, data_citeid, data_edges = parse_cora()
    # data_X = sklearn.preprocessing.normalize(data_X, norm="l1")

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)  # Numpy module.
    random.seed(SEED)  # Python random module.

    # load data
    data_name = 'cora'
    # path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')
    dataset = Planetoid('../../data/Planetoid', data_name,
                        transform=T.NormalizeFeatures())
    data = dataset[0]

    data.x = torch.tensor(data_X).float()
    data.edge_index = torch.tensor(data_edges).long()
    data.y = torch.tensor(data_Y).long()
    data.num_nodes = len(data_Y)

    # split data
    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)

    data.train_id = np.sort(node_id[:int(data.num_nodes * 0.6)])
    data.val_id = np.sort(
        node_id[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)])
    data.test_id = np.sort(node_id[int(data.num_nodes * 0.8):])

    data.train_mask = torch.tensor(
        [x in data.train_id for x in range(data.num_nodes)])
    data.val_mask = torch.tensor(
        [x in data.val_id for x in range(data.num_nodes)])
    data.test_mask = torch.tensor(
        [x in data.test_id for x in range(data.num_nodes)])

    return data, data_citeid

# credit: https://github.com/tkipf/pygcn/issues/27, xuhaiyun


def parse_cora():
    path = 'cora/cora_orig/cora'
    idx_features_labels = np.genfromtxt(
        "{}.content".format(path), dtype=np.dtype(str))
    data_X = idx_features_labels[:, 1:-1].astype(np.float32)
    labels = idx_features_labels[:, -1]
    class_map = {x: i for i, x in enumerate(['Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
                                            'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory'])}
    data_Y = np.array([class_map[l] for l in labels])
    data_citeid = idx_features_labels[:, 0]
    idx = np.array(data_citeid, dtype=np.dtype(str))
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(
        "{}.cites".format(path), dtype=np.dtype(str))
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(
        edges_unordered.shape)
    data_edges = np.array(edges[~(edges == None).max(1)], dtype='int')
    data_edges = np.vstack((data_edges, np.fliplr(data_edges)))
    return data_X, data_Y, data_citeid, np.unique(data_edges, axis=0).transpose()


def get_raw_text_cora(use_text=False, seed=0):
    data, data_citeid = get_cora_casestudy(seed)
    if not use_text:
        return data, None

    with open('cora/cora_orig/mccallum/cora/papers')as f:
        lines = f.readlines()
    pid_filename = {}
    for line in lines:
        pid = line.split('\t')[0]
        fn = line.split('\t')[1]
        pid_filename[pid] = fn

    path = 'cora/cora_orig/mccallum/cora/extractions/'
    text = []
    # import ipdb; ipdb.set_trace()
    for pid in data_citeid:
        fn = pid_filename[pid]
        with open(path+fn) as f:
            lines = f.read().splitlines()

        for line in lines:
            if 'Title:' in line:
                ti = line
            if 'Abstract:' in line:
                ab = line
        text.append(ti+'\n'+ab)
    
    return data, text

import sys
def get_text_data(dataname='cora', seed=0):
    dataset, id_text = get_raw_text_cora(use_text=True, seed=seed)
    ood_data = []

    for ood_idx, filename in enumerate(os.listdir("cora/ood_data")):
        # import ipdb; ipdb.set_trace()
        with open(f'cora/ood_data/{filename}', 'r') as f:
            string = f.read()
        parsed_data = json.loads(string)
        data = 'Title: ' + parsed_data['title'] + '\n' + \
                'Abstract: ' + parsed_data['abstract']
        ood_data.append(data)
    return ood_data, id_text