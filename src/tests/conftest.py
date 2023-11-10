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
    PATH_EDGELIST
)

@pytest.fixture
def define_stor_dir():
    return './test_data/edgelist_test.txt'

@pytest.fixture
def define_com_of_user():
    return './test_data/com_of_user.pkl'

@pytest.fixture
def define_com_of_user_days():
    return './test_data/com_of_user_days.pkl'

@pytest.fixture
def define_path_day():
    return './test_data/edgelist_days/'

@pytest.fixture
def define_path_day1():
    return './test_data/edgelist_days/edgelist_w1_2022-02-01.txt'

@pytest.fixture
def define_path_day2():
    return './test_data/edgelist_days/edgelist_w1_2022-02-02.txt'

@pytest.fixture
def define_path_day3():
    return './test_data/edgelist_days/edgelist_w1_2022-02-03.txt'

@pytest.fixture
def define_path_war():
    return './test_data/WarTweets.pkl.gz'

@pytest.fixture
def define_path_betweenness():
    return './test_data/betweenness.csv'

@pytest.fixture
def read_betweenness(define_path_betweenness):
    df = pd.read_csv(define_path_betweenness)
    list_betweenness = df['betweenness'].to_list()
    return list_betweenness

@pytest.fixture
def read_com_of_user_days(define_com_of_user_days):
    with open(define_com_of_user_days,'rb') as f: 
        com_of_user_days=pickle.load(f)
    return com_of_user_days
    
@pytest.fixture
def read_file_G_days(define_path_day1, define_path_day2, define_path_day3):
    list_days = [define_path_day1, define_path_day2, define_path_day3]
    G_days = []
    for path in list_days:
        G=nx.read_weighted_edgelist(path,delimiter='\t',create_using=nx.DiGraph,nodetype=str)
        G_days.append(G)
    return G_days

@pytest.fixture
def read_file_G(define_stor_dir):
    G=nx.read_weighted_edgelist(define_stor_dir,
                                       delimiter='\t',
                                       create_using=nx.DiGraph,
                                       nodetype=str)
    return G
        
@pytest.fixture
def read_file_components_weak():
    G=nx.read_weighted_edgelist('./test_data/component_weak.txt',
                                delimiter='\t',
                                create_using=nx.DiGraph,nodetype=str)
    return G

@pytest.fixture
def read_file_com_of_user(define_com_of_user):
    with open(define_com_of_user,'rb') as f: 
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

@pytest.fixture
def create_input_for_df(): #input_test_create_df
    col_names = ['A','B','C']
    list_content = [[1,2],[7,2],[3,6]]
    return {'col_names':col_names, 'list_content': list_content}

@pytest.fixture
def input_test_filter_top_users():
    df_users = pd.DataFrame({'community': ['A','A','B','B','A','B', 'B', 'A', 'B', 'B'],
                            'total-degree':[4,13,2,11,8,9,6,22,1,5]})
    return df_users
                            
@pytest.fixture
def input_mixing_matrix_manipulation():
    df = pd.DataFrame({'A':[1, 2],
                       'B':[3, 4]})
    df.index=['A','B']
    return df

@pytest.fixture
def df_top():
    df = pd.DataFrame({'user':[1,3], 'community':['A','A']})
    return df

@pytest.fixture
def dataframe_retweet():
    df = pd.DataFrame({'user.id':[1,1,1,5,3,3,1],
                       'retweeted_status.user.id':[4,4,4,np.nan,2,2,2],
                       'text':['hi','hi','apple','bye','table table table , e o . https ...',np.nan,'cat cat house ; !'],
                       'created_at_days':['2022-06-10','2022-06-12','2022-03-02','2021-10-28','2022-06-10','2022-03-02','2022-03-02'],
                       'created_at':[2,5,4,10,20,21,7]})
    df['created_at_days'] = pd.to_datetime(df['created_at_days'])
    return df

@pytest.fixture
def dataframe_retweet2():
    df = pd.DataFrame({'user.id':[1,3,1],
                       'retweeted_status.user.id':[4.,2.,2.],
                       'weight':[3,2,1]})
    df = df.sort_values(by=['user.id','retweeted_status.user.id']).reset_index(drop = True)
    return df