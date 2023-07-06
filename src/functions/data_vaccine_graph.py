import pandas as pd
import networkx as nx
import pickle
from collections import Counter 
from functions import assign_communities, mixing_matrix, randomize_network, compute_randomized_modularity
from functions import compute_connected_component, compute_weak_connected_component, gini
from configurations import (
    STOR_DIR,
    PATH_VACCINE,
    PATH_MIXING_MATRIX,
    PATH_COM_OF_USER,
    PATH_DEGREE
)


def main():  
    '''
    Load the network built from the retweets on Italian "vaccine" tweets since October 2020. 
    The variable is a   NetworkX DiGraph object (directed graph), having users as nodes and 
    the edge weights are cumulative numbers of retweets over the full time span. 
    Edges in direction x->y 
    means that user x retweeted user y a number of times equal to the weight. 
    So the in-degree of a user is the total number of retweets it received in the time span.
    '''
    Gvac=nx.read_weighted_edgelist(STOR_DIR+PATH_VACCINE,
                                       delimiter='\t',
                                       create_using=nx.DiGraph,
                                       nodetype=str)
    #print(nx.info(Gvac))
    
    with open(STOR_DIR+PATH_COM_OF_USER,'rb') as f:
        com_of_user=pickle.load(f)
        
    Gvac_subgraph, Gvac_A, Gvac_B, group_A, group_B = assign_communities(Gvac, com_of_user)
    df, df_weight = mixing_matrix(Gvac, com_of_user)
    
    '''
    I save a file for each table.
    Table representing the number of links from A to A, from A to B, from B to A and from B to B 
    (mixing matrix).
    '''
    df.head()
    df.to_csv(PATH_MIXING_MATRIX+'MixingVaccine1.csv', index=False)
    
    '''
    Each entry is divided by the total number of links taken into account.
    '''
    total_links = df.sum().sum()
    df1 = df/float(total_links)
    df1.to_csv(PATH_MIXING_MATRIX+'MixingVaccine2.csv', index=False)
    
    '''
    We take into account the total number of links starting from community A, i.e. the sum of the
    elements of the first row of the mixing matrix, and we divide each element of the first row by
    this number. Then we repeat the procedure for the second row.
    In this way we get the average behaviour if a link starts from community A or from community B.
    '''
    df.loc['A']=df.loc['A']/df.sum(axis=1).to_dict()['A']
    df.loc['B']=df.loc['B']/df.sum(axis=1).to_dict()['B']
    df2 = df
    df2.to_csv(PATH_MIXING_MATRIX+'MixingVaccine3.csv', index=False)
    
    '''
    Mixing matrix in weighted case.
    '''
    df_weight.head()
    df3 = df_weight
    df3.to_csv(PATH_MIXING_MATRIX+'MixingVaccine4.csv', index=False)
    
    '''
    Each entry is divided by the total number of links taken into account, in the weighted case.
    '''
    total_links = df_weight.sum().sum()
    df4 = df_weight/float(total_links)
    df4.to_csv(PATH_MIXING_MATRIX+'MixingVaccine5.csv', index=False)
    
    '''
    Referring to the weighted case, we take into account the total number of links starting 
    from community A, i.e. the sum of the elements of the first row of the mixing matrix, 
    and we divide each element of the first row by this number. Then we repeat the procedure 
    for the second row. In this way we get the average behaviour if a link starts 
    from community A or from community B.
    '''
    df_weight.loc['A']=df_weight.loc['A']/df_weight.sum(axis=1).to_dict()['A']
    df_weight.loc['B']=df_weight.loc['B']/df_weight.sum(axis=1).to_dict()['B']
    df5 = df_weight
    df5.to_csv(PATH_MIXING_MATRIX+'MixingVaccine6.csv', index=False)
    
    '''
    Number of nodes belonging to the communities in the original vaccine network.
    '''
    Counter(list(com_of_user.values())) 
    
    '''
    We create 6 lists to store the in- out-degree of the nodes belonging to the whole network, 
    group A and group B and we save them in a corresponding file.
    '''
    in_degree_original = [Gvac.in_degree(node) for node in nx.nodes(Gvac)]
    out_degree_original = [Gvac.out_degree(node) for node in nx.nodes(Gvac)]
    
    in_degree_group_A = [Gvac_A.in_degree(node) for node in nx.nodes(Gvac_A)]
    out_degree_group_A = [Gvac_A.out_degree(node) for node in nx.nodes(Gvac_A)]
    
    in_degree_group_B = [Gvac_B.in_degree(node) for node in nx.nodes(Gvac_B)]
    out_degree_group_B = [Gvac_B.out_degree(node) for node in nx.nodes(Gvac_B)]
    
    df_original = pd.DataFrame({'in_degree_original': in_degree_original,
                                'out_degree_original': out_degree_original})
    df_original.head()
    df_original.to_csv(PATH_DEGREE+"DegreeOriginal.csv", index = False)
    
main()