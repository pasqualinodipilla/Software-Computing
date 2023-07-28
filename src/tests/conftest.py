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
    k = 2000
    return k

@pytest.fixture
def read_file_G(define_k):
    G=nx.read_weighted_edgelist(STOR_DIR+PATH_VACCINE,
                                       delimiter='\t',
                                       create_using=nx.DiGraph,
                                       nodetype=str)
    sampled_nodes = random.sample(G.nodes, define_k)
    sampled_graph = G.subgraph(sampled_nodes)
    return sampled_graph
        
@pytest.fixture
def read_file_com_of_user():
    with open(STOR_DIR+PATH_COM_OF_USER,'rb') as f: 
        com_of_user=pickle.load(f)
    return com_of_user
        
@pytest.fixture
def create_G_with_community(read_file_G, read_file_com_of_user):
    G = read_file_G
    com_of_user = read_file_com_of_user
    for node in nx.nodes(G):
        com_of_user.setdefault(node,'')
    nx.set_node_attributes(G, com_of_user, "community")
    return G

@pytest.fixture
def create_edgelist(create_G_with_community):
    list_combination = ['AA','AB','BA','BB']
    edgelist = list(create_G_with_community.edges())
    edgelist=[(node1,node2) for node1,node2 in edgelist if create_G_with_community.nodes[node1]["community"]+create_G_with_community.nodes[node2]["community"] in list_combination]
    return edgelist

@pytest.fixture
def compute_weights(read_file_G, create_edgelist):
    edge_weight = nx.get_edge_attributes(read_file_G,'weight')
    edge_weight = [edge_weight[(node1, node2)] for (node1, node2) in create_edgelist]
    return edge_weight

@pytest.fixture
def n_swaps():
    N=100
    return N

@pytest.fixture
def get_groupA_groupB(create_G_with_community):
    G = create_G_with_community
    group_A = []
    group_B = []
    for n, user in enumerate(nx.nodes(G)):
        if G.nodes[user]['community']=='A':
            group_A.append(user)
        elif G.nodes[user]['community']=='B':
            group_B.append(user)
    
    G_subgraph = G.subgraph(list(set(group_A)|set(group_B)))
    
    list_nodes_selected = [node for node in nx.nodes(G_subgraph) 
        if (G_subgraph.in_degree(node)>0 or G_subgraph.out_degree(node)>0)]
    G_subgraph = G.subgraph(list_nodes_selected)
    
    group_A = list(set(group_A) & set(list_nodes_selected))
    group_B = list(set(group_B) & set(list_nodes_selected))
     
    return {'G_subgraph':G_subgraph, 'group_A': group_A, 'group_B' : group_B}

@pytest.fixture
def create_list_edges():
    return [(1,2), (5, 6), (3,5), (2,7), (1,2), (9,3), (3,4), (4,1), (5,7), (9,2)]

    