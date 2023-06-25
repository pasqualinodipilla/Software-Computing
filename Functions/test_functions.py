import pandas as pd
import networkx as nx
import pickle
import numpy as np
import datetime 
import os
from collections import Counter 
import copy
import random
from functions import assign_communities, mixing_matrix, randomize_network, compute_randomized_modularity
from functions import compute_connected_component, compute_weak_connected_component, gini

class cache:  #we use caching for storing the most frequently accessed data in the temporary cache                   
              #location so that the data can be accessed quickly
    pass

def test_assign_communities(ReadFileG, ReadFileComOfUser):
    '''
    This function is testing if the output of assign_communities() function are of the correct type:
    group_A and group_B must be lists, Gvac_subgraph, Gvac_A and Gvac_B must be direct graphs. Then       
    it tests that the set list of nodes belonging to Gvac_A (i.e. the subgraph containing the nodes
    belonging to group A) actually corresponds to group_A and that the set list of nodes belonging to
    Gvac_B (i.e. the subgraph containing the nodes belonging to group B) actually corresponds to 
    group_B. At the end it tests that the set list of nodes belonging to Gvac_subgraph is given by
    the set list of nodes belonging to Gvac_A or Gvac_B
    :param ReadFileG: variable into which we store the network G_vac in networkX format.
    :param ReadFileComOfUser: variable into which we store com_of_user that is a dictionary having 
    the Id user as key and the community as value.
    '''

    cache.G_vac = ReadFileG
    cache.com_of_user = ReadFileComOfUser
    k = 5000
    sampled_nodes = random.sample(cache.G_vac.nodes, k)
    sampled_graph = cache.G_vac.subgraph(sampled_nodes)
    cache.G_vac = sampled_graph
    Gvac_subgraph, Gvac_A, Gvac_B, group_A, group_B=assign_communities(cache.G_vac, cache.com_of_user)
    cache.Gvac_subgraph = Gvac_subgraph
    cache.Gvac_A = Gvac_A
    cache.Gvac_B = Gvac_B
    cache.group_A = group_A
    cache.group_B = group_B
    assert type(group_A) == list
    assert type(group_B) == list
    assert type(Gvac_subgraph) == type(nx.DiGraph())
    assert type(Gvac_A) == type(nx.DiGraph())
    assert type(Gvac_B) == type(nx.DiGraph())
    
    set_nodes_Gvac_A = set(Gvac_A.nodes())
    set_nodes_Gvac_B = set(Gvac_B.nodes())
    set_nodes_Gvac_subgraph = set(Gvac_subgraph.nodes())
    
    assert set_nodes_Gvac_A & set(group_A) == set_nodes_Gvac_A
    assert set_nodes_Gvac_B & set(group_B) == set_nodes_Gvac_B
    assert set_nodes_Gvac_A | set_nodes_Gvac_B == set_nodes_Gvac_subgraph
    
def test_mixing_matrix():
    '''
    This function is used to test if the 2 columns that we get in the mixing matrix for the unweighted and weighted cases are actually the number of edges coming respectively from users of group A and users of group B. In addition we test if the sum of all the elements of the 2 matrices corresponds to the overall number of edges of our graph, in both the cases unweighted and weighted.
    '''
    
    df, df_weight = mixing_matrix(cache.G_vac, cache.com_of_user)
    list_combination = ['AA','AB','BA','BB']
    edgelist = list(cache.G_vac.edges())
    edgelist=[(node1,node2) for node1,node2 in edgelist if cache.G_vac.nodes[node1]["community"]+cache.G_vac.nodes[node2]["community"] in list_combination]
    edge_weight = nx.get_edge_attributes(cache.G_vac,'weight')
    edge_weight = [edge_weight[(node1, node2)] for (node1, node2) in edgelist]
    assert set(df.columns) == set(['A','B'])
    assert set(df_weight.columns) == set(['A','B'])
    assert int(df.sum().sum()) == len(edgelist)
    assert int(df_weight.sum().sum()) == int(sum(edge_weight))
    
def test_randomize_network():
    '''
    This function is used to test that, regardless of the shuffling, the number of edges connecting
    the nodes in the original graph Gvac_subgraph is actually equal to the overall number of edges
    connecting the nodes in the shuffled network, Gvac_shuffle.
    In addition we test if modularity_unweighted and modularity_weighted, outputs of the c
    orresponding function, are float type or not.
    '''
    
    N = 1000
    modularity_unweighted, modularity_weighted, Gvac_shuffle = randomize_network(N, cache.Gvac_subgraph, cache.group_A, cache.group_B)
    edge_weight1 = nx.get_edge_attributes(cache.Gvac_subgraph,'weight')
    edge_weight1 = [edge_weight1[(node1, node2)] for (node1, node2) in edge_weight1]
    edge_weight2 = nx.get_edge_attributes(Gvac_shuffle,'weight')
    edge_weight2 = [edge_weight2[(node1, node2)] for (node1, node2) in edge_weight2]
    assert int(sum(edge_weight1)) == int(sum(edge_weight2))
    assert type(modularity_unweighted) == float
    assert type(modularity_weighted) == float
    
def test_compute_randomized_modularity():
    '''
    This function is used to test the type of the output of compute_randomized_modularity() function:
    list_modularity_unweighted, list_modularity_weighted must be lists. Then, since in our case we 
    perform 10 randomizations, we test if the lenght of the two lists is equal to 10.
    '''
    list_modularity_unweighted, list_modularity_weighted =compute_randomized_modularity(cache.Gvac_subgraph,cache.group_A,cache.group_B)
    assert type(list_modularity_unweighted) == list
    assert type(list_modularity_weighted) == list
    assert len(list_modularity_unweighted) == 10 
    assert len(list_modularity_weighted) == 10
    
def test_compute_connected_component():
    '''
    This function is used to test the type of the output of compute_connected_component() function:
    group_A_G0, group_B_G0, group_A_G1, group_B_G1 must be lists, G0 and G1 must be direct graphs.
    Then the functions tests that the nodes of the graph G0 (representing the first strongly 
    connected component) actually correspond to the set list of nodes belonging to group_A_G0 or
    group_B_G0. Similarly it tests if the nodes of the graph G1 (representing the second strongly 
    connected component) actually correspond to the set list of nodes belonging to group_A_G1 or
    group_B_G1.
    '''
    group_A_G0, group_B_G0, group_A_G1, group_B_G1, G0, G1 = compute_connected_component(cache.Gvac_subgraph,cache.group_A, cache.group_B)
    cache.G0 = G0
    assert type(group_A_G0) == list
    assert type(group_B_G0) == list
    assert type(group_A_G1) == list
    assert type(group_B_G1) == list
    assert type(G0) == type(nx.DiGraph())
    assert type(G1) == type(nx.DiGraph())
    assert set(group_A_G0) | set(group_B_G0) == set(G0.nodes())
    assert set(group_A_G1) | set(group_B_G1) == set(G1.nodes())
    
    
def test_compute_weak_connected_component():
    '''
    This function is used to test the type of the output of compute_connected_component() function:
    group_A_G0_weak, group_B_G0_weak, group_A_G1_weak, group_B_G1_weak must be lists, G0_weak and
    G1_weak must be direct graphs.
    Then the functions tests if the nodes of the graph G0_weak (representing the first weakly 
    connected component) actually correspond to the set list of nodes belonging to group_A_G0_weak or
    group_B_G0_weak. Similarly it tests if the nodes of the graph G1_weak (representing the second 
    weakly connected component) actually correspond to the set list of nodes belonging to 
    group_A_G1_weak or group_B_G1_weak. In addition it tests if the first strongly connected 
    component is smaller than the first weakly connected component, as it should be.
    '''
    (group_A_G0_weak, group_B_G0_weak, group_A_G1_weak,
     group_B_G1_weak,G0_weak, G1_weak)=compute_weak_connected_component(cache.Gvac_subgraph,
                                                                        cache.group_A,
                                                                        cache.group_B)
    
    assert type(group_A_G0_weak) == list
    assert type(group_B_G0_weak) == list
    assert type(group_A_G1_weak) == list
    assert type(group_B_G1_weak) == list
    assert type(G0_weak) == type(nx.DiGraph())
    assert type(G1_weak) == type(nx.DiGraph())
    assert set(group_A_G0_weak) | set(group_B_G0_weak) == set(G0_weak.nodes())
    assert set(group_A_G1_weak) | set(group_B_G1_weak) == set(G1_weak.nodes())
    assert len(cache.G0.nodes()) <= len(G0_weak.nodes())
    
def test_gini():
    '''
    This function firstly tests if the type of the Gini index value given in output from the corresponding function is a float type (in both the cases Gini_in, Gini_out) and than, as a further confirmation, if we assume a pure equidistribution, for example with all values of the x array in input equal to 1, we test if we will obtain a gini index equal to 0, as it should be by definition.
    '''
    in_degree_original = [cache.G_vac.in_degree(node) for node in nx.nodes(cache.G_vac)]
    out_degree_original = [cache.G_vac.out_degree(node) for node in nx.nodes(cache.G_vac)]
    
    Gini_in = gini(in_degree_original)
    Gini_out = gini(out_degree_original)

    Gini_0 = gini([1,1,1,1])
    
    assert type(Gini_in) == np.float64
    assert type(Gini_out) == np.float64
    assert Gini_0 == 0