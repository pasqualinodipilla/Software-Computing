import pandas as pd
import networkx as nx
import pickle
import numpy as np
import datetime 
import os
import random
from collections import Counter 
import copy
import pytest
import sys
sys.path.append('../functions')
from configurations import (
    STOR_DIR,
    PATH_VACCINE,
    PATH_COM_OF_USER
)

@pytest.fixture
def define_k():
    k = 5000
    return k

@pytest.fixture
def read_file_G(define_k):
    G=nx.read_weighted_edgelist(STOR_DIR+PATH_VACCINE,
                                       delimiter='\t',
                                       create_using=nx.DiGraph,
                                       nodetype=str)
    sampled_nodes = random.sample(G.nodes, define_k)
    sampled_graph = G.subgraph(sampled_nodes)
    G = sampled_graph
    return G
        
@pytest.fixture
def read_file_com_of_user():
    with open(STOR_DIR+PATH_COM_OF_USER,'rb') as f: 
        com_of_user=pickle.load(f)
    return com_of_user
        
@pytest.fixture
def create_edgelist(read_file_G, read_file_com_of_user):
    G = read_file_G
    com_of_user = read_file_com_of_user
    nx.set_node_attributes(G, com_of_user, "community")
    list_combination = ['AA','AB','BA','BB']
    edgelist = list(read_file_G.edges())
    edgelist=[(node1,node2) for node1,node2 in edgelist if read_file_G.nodes[node1]["community"]+read_file_G.nodes[node2]["community"] in list_combination]
    return edgelist

@pytest.fixture
def compute_weights(read_file_G, create_edgelist):
    edge_weight = nx.get_edge_attributes(read_file_G,'weight')
    edge_weight = [edge_weight[(node1, node2)] for (node1, node2) in create_edgelist]
    return edge_weight

@pytest.fixture
def n_swaps():
    N=1000
    return N

@pytest.fixture
def get_groupA_groupB(read_file_G):
    G = read_file_G
    N = len(nx.nodes(G))
    group_A = []
    group_B = []
    for n, user in enumerate(nx.nodes(G)):
        if n < N/2:
            group_A.append(user)
        else:
            group_B.append(user)
    return {'G':G, 'group_A': group_A, 'group_B' : group_B}