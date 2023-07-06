import pandas as pd
import networkx as nx
import pickle
from collections import Counter 
from scipy import stats
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import datetime
import string
from sklearn.metrics import roc_auc_score
from configurations import SEED, top_user_fraction, PATH_WAR

def assign_communities(G, com_of_user):
    '''
    This function returns the subgraph containing the nodes belonging to group A 
    and group B with a degree>0 and two lists containing the users belonging to group A
    and the users belonging to group B, respectively, taking in input the graph G and
    'com_of_user' into which we save and load our data by using pickle module. 
    :param G:network in networkX format.
    :param com_of_user: com_of_user is a dictionary having the Id user as key 
    and the community as value.
    :return Gvac_subgraph: subgraph of G in networkX format containing the nodes belonging 
    to group A and group B with a degree>0
    :return Gvac_A: subgraph of G in networkX format containing the nodes belonging to group A
    :return Gvac_B: subgraph of G in networkX format containing the nodes belonging to group B
    :return group_A: list of strings containing the Id users of group A
    :return group_B: list of strings containing the Id users of group B
    '''
    #We want to assign a node attribute to store the value of that
    #property for each node: the attribute in question is the belonging to a community.
    for node in nx.nodes(G):
        com_of_user.setdefault(node,'')
    nx.set_node_attributes(G, com_of_user, "community")
    
    #We define group_A if the user belongs to community A, group_B  if  the user belongs to community B
    #and group_null if the users don't belong to one of the two groups.
    group_A = set()
    group_B = set()
    group_null = set()
    
    for user in nx.nodes(G):
        if com_of_user[user]=='A':
            group_A.add(user)
        elif com_of_user[user]=='B':
            group_B.add(user)
        elif com_of_user[user]=='':
            group_null.add(user)
    
    #We consider the subgraph including the users belonging to group A or group B and the users
    #with a degree greater than 0.
    Gvac_subgraph = G.subgraph(list(group_A|group_B))
    
    list_nodes_selected = [node for node in nx.nodes(Gvac_subgraph) 
        if (Gvac_subgraph.in_degree(node)>0 or Gvac_subgraph.out_degree(node)>0)]
    Gvac_subgraph = G.subgraph(list_nodes_selected)
    #print(nx.info(Gvac_subgraph))
    
    #We get two lists including the users belonging to groupA and to group B, respectively, with
    #degree greater than 0.
    group_A = list(set(group_A) & set(list_nodes_selected))
    group_B = list(set(group_B) & set(list_nodes_selected))
    
    Gvac_A = G.subgraph(list(group_A)) 
    #print(nx.info(Gvac_A))
    
    Gvac_B = G.subgraph(list(group_B))
    return Gvac_subgraph, Gvac_A, Gvac_B, group_A, group_B


def mixing_matrix(G, com_of_user):
    '''
    This function returns the mixing matrix, thus the 2x2 matrix that has as entries the 
    number of edges starting from A and ending in A, starting from A and ending in B, 
    starting from B and ending in A and starting from B and ending in B, taking as input 
    the graph G and 'com_of_user' into which we save and load our data by using pickle module.
    :param G:network in networkX format.
    :param com_of_user: com_of_user is a dictionary having the Id user as key and 
    the community as value.
    :return df: dataframe representing the 2x2 mixing matrix in the unweighted case.
    :return df_weight: dataframe representing the 2x2 mixing matrix in the weighted case.
    '''
    
    com_of_edges={}
    for edge in nx.edges(G):
        com_of_edges[edge]=G.nodes[edge[0]]["community"]+G.nodes[edge[1]]["community"]
    
    mixing_matrix=Counter(list(com_of_edges.values()))
    #Unweighted case
    df = pd.DataFrame({'A':[mixing_matrix['AA'], mixing_matrix['BA']],
                       'B':[mixing_matrix['AB'], mixing_matrix['BB']]})
    df.index = ['A','B']
    
    #Get edge attributes from graph
    edge_weight = nx.get_edge_attributes(G,'weight')
    mixing_matrix_weight={'AA':0, 'AB':0, 'BA':0, 'BB':0}
    for edge in nx.edges(G):
        if com_of_edges[edge]== 'AA':
            mixing_matrix_weight['AA']+=edge_weight[edge]
        elif com_of_edges[edge]== 'AB':
            mixing_matrix_weight['AB']+=edge_weight[edge]
        elif com_of_edges[edge]== 'BA':
            mixing_matrix_weight['BA']+=edge_weight[edge]
        elif com_of_edges[edge]== 'BB':
            mixing_matrix_weight['BB']+=edge_weight[edge]
    
    #Weighted case
    df_weight = pd.DataFrame({'A':[mixing_matrix_weight['AA'], mixing_matrix_weight['BA']], 
                              'B':[mixing_matrix_weight['AB'], mixing_matrix_weight['BB']]})
    df_weight.index = ['A','B']
    return df, df_weight

def randomize_network(seed, n_swaps, Gvac, group_A, group_B):
    '''
    This function creates a random network and compute the modularity in the 
    unweighted and weighted case, taking in input the number of swaps, 
    the graph Gvac and the lists containing the users
    belonging to group A or group B.
    :param n_swaps: integer number representing the number of swaps.
    :param Gvac: network in networkX format.
    :param group_A: list of strings containing the Id users of group A
    :param group_B: list of strings containing the Id users of group B
    :return modularity_unweighted: float value representing the modularity in the unweighted case.
    :return modularity_weighted: float value representing the modularity in the weighted case.
    '''
    list_edges = []
    #We get edge attributes from the graph, i.e. we get a dictionary of attributes keyed by edge.
    edge_weight = nx.get_edge_attributes(Gvac,'weight')

    for edge in edge_weight:
        #We take into account any possible repetition: if I have a weight equal to 5 
        #we repeat this command 5 times
        for j in range (int(edge_weight[edge])):
            list_edges.append(edge)
    i=0

    #Let's create an array of indexes in which each index corresponds to an element of list_edges
    array_index = np.arange(len(list_edges))
    #specify random seed
    np.random.seed(seed)
    while(i<n_swaps):
        #We choose two indexes randomly
        edges_to_swap = np.random.choice(array_index,2)
        edge_1 = list_edges[edges_to_swap[0]]
        edge_2 = list_edges[edges_to_swap[1]]
    
        #We have to verify that all the nodes belonging to these edges are different.
        nodes = [edge_1[0],edge_1[1], edge_2[0], edge_2[1]]
        if len(set(nodes))==4:
            newedge_1 = (edge_1[0],edge_2[1])
            newedge_2 = (edge_2[0],edge_1[1])
            #Let's change the edges in listedges
            list_edges[edges_to_swap[0]]=newedge_1
            list_edges[edges_to_swap[1]]=newedge_2
            
            i = i+1
    #By using Counter(), which counts the number of times an element appears in a list, 
    #we redefine the edge weight    
    edge_weight = Counter(list_edges)
    
    #We create the randomized network and we evaluate the modularity in the unweighted 
    #and weighted cases.
    Gvac_shuffle = nx.from_edgelist(list(set(list_edges)),nx.DiGraph)
    modularity_unweighted = nx.community.modularity(Gvac_shuffle, [group_A,group_B])
    nx.set_edge_attributes(Gvac_shuffle, edge_weight, "weight")
    modularity_weighted = nx.community.modularity(Gvac_shuffle, 
                                                  [group_A,group_B], weight = 'weight')
    return modularity_unweighted, modularity_weighted, Gvac_shuffle

def compute_randomized_modularity(Gvac_subgraph, group_A, group_B):
    '''
    Function that returns two lists containing the modularity in the unweighted and weighted case,
    taking in input the subgraph Gvac_subgraph (that will be obtained from Assign_Communities)
    and two lists (that will contain the users belonging to group A or group B).
    :param Gvac_subgraph: subgraph of G in networkX format containing the nodes belonging 
    to group A and group B with a degree>0
    :param group_A: list of strings containing the Id users of group A.
    :param group_B: list of strings containing the Id users of group B.
    :return list_modularity_unweighted: list containing the modularity values evaluated after
    a certain number of randomization, in this case 10, in the unweighted case.
    :return list_modularity_weighted: list containing the modularity values evaluated after
    a certain number of randomization, in this case 10, in the weighted case.
    '''
    list_modularity_unweighted = []
    list_modularity_weighted = []
    N = len(Gvac_subgraph.edges())
    #We repeat the randomization a certain number of times, in this case 10.
    for i in range(10):
        start_time = datetime.datetime.now()
        modularity_unweighted, modularity_weighted, Gvac_shuffle = randomize_network(SEED,N,Gvac_subgraph,
                                                                       group_A, group_B)
        list_modularity_unweighted.append(modularity_unweighted)
        list_modularity_weighted.append(modularity_weighted)
        #If we want to know how long it takes to perform a randomization.
        #print(datetime.datetime.now()-start_time)
    return list_modularity_unweighted, list_modularity_weighted

def compute_connected_component(Gvac_subgraph, group_A, group_B):
    '''
    Function that returns G0 and G1 that will include the nodes of group A 
    or group B belonging to the first and second strongly connected component, 
    being the first strongly connected component G0, the second strongly connected
    component G1 and four lists group_A_G0, group_B_G0, group_A_G1, group_B_G1 which contain
    the users of group A and group B belonging to the first or second strongly connected component,
    giving in input the subgraph Gvac_subgraph (that will be obtained from Assign_Communities)
    and two lists (that will contain the users belonging to group A or group B).
    :param Gvac_subgraph: subgraph of G in networkX format containing the nodes belonging 
    to group A and group B with a degree>0
    :param group_A: list of strings containing the Id users of group A.
    :param group_B: list of strings containing the Id users of group B.
    :return group_A_G0: list of strings containing the Id users of group A belonging to the 
    first strongly connected component.
    :return group_B_G0: list of strings containing the Id users of group B belonging to the 
    first strongly connected component.
    :return group_A_G1: list of strings containing the Id users of group A belonging to the 
    second strongly connected component.
    :return group_B_G1: list of strings containing the Id users of group B belonging to the 
    second strongly connected component.
    :return G0: A generator of sets of nodes, one for each strongly connected component 
    of Gvac_subgraph, in this case the first one.
    :return G1: A generator of sets of nodes, one for each strongly connected component 
    of Gvac_subgraph, in this case the second one.
    '''
    Gcc = sorted(nx.strongly_connected_components(Gvac_subgraph), key=len, reverse=True)
    G0 = Gvac_subgraph.subgraph(Gcc[0])
    G1 = Gvac_subgraph.subgraph(Gcc[1])
    
    group_A_G0 = list(set(group_A) & set(list(G0.nodes())))
    group_B_G0 = list(set(group_B) & set(list(G0.nodes())))
    group_A_G1 = list(set(group_A) & set(list(G1.nodes())))
    group_B_G1 = list(set(group_B) & set(list(G1.nodes())))
    return group_A_G0, group_B_G0, group_A_G1, group_B_G1, G0, G1

def compute_weak_connected_component(Gvac_subgraph, group_A, group_B):
    '''
    Function that returns G0 and G1 that will include the nodes of group A 
    or group B belonging to the first and second strongly connected component, 
    being the first strongly connected component G0, the second strongly connected
    component G1 and four lists group_A_G0, group_B_G0, group_A_G1, group_B_G1 which contain
    the users of group A and group B belonging to the first or second strongly connected component,
    giving in input the subgraph Gvac_subgraph (that will be obtained from Assign_Communities)
    and two lists (that will contain the users belonging to group A or group B).
    :param Gvac_subgraph: subgraph of G in networkX format containing the nodes belonging 
    to group A and group B with a degree>0
    :param group_A: list of strings containing the Id users of group A.
    :param group_B: list of strings containing the Id users of group B.
    :return group_A_G0: list of strings containing the Id users of group A belonging to the 
    first weakly connected component.
    :return group_B_G0: list of strings containing the Id users of group B belonging to the 
    first weakly connected component.
    :return group_A_G1: list of strings containing the Id users of group A belonging to the 
    second weakly connected component.
    :return group_B_G1: list of strings containing the Id users of group B belonging to the 
    second weakly connected component.
    :return G0: A generator of sets of nodes, one for each weakly connected component 
    of Gvac_subgraph, in this case the first one.
    :return G1: A generator of sets of nodes, one for each weakly connected component 
    of Gvac_subgraph, in this case the second one.
    '''
    Gcc = sorted(nx.weakly_connected_components(Gvac_subgraph), key=len, reverse=True)
    G0 = Gvac_subgraph.subgraph(Gcc[0])
    G1 = Gvac_subgraph.subgraph(Gcc[1])
    group_A_G0 = list(set(group_A) & set(list(G0.nodes())))
    group_B_G0 = list(set(group_B) & set(list(G0.nodes())))
    group_A_G1 = list(set(group_A) & set(list(G1.nodes())))
    group_B_G1 = list(set(group_B) & set(list(G1.nodes())))
    return group_A_G0, group_B_G0, group_A_G1, group_B_G1, G0, G1


def compute_strong_or_weak_components(Gvac_subgraph, group_A, group_B, isweak):
    nodes_group_A_G0 = []
    nodes_group_B_G0 = []
    nodes_group_A_G1 = []
    nodes_group_B_G1 = []
    if isweak:
        x = compute_weak_connected_component(Gvac_subgraph, group_A, group_B)
    else:
        x = compute_connected_component(Gvac_subgraph, group_A, group_B)
    
    nodes_group_A_G0.append(len(x[0]))
    nodes_group_B_G0.append(len(x[1]))
    nodes_group_A_G1.append(len(x[2]))
    nodes_group_B_G1.append(len(x[3]))
    return nodes_group_A_G0, nodes_group_B_G0, nodes_group_A_G1, nodes_group_B_G1
    
    
def gini(x): 
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    #x = x.flatten()
    if np.amin(x) < 0:
        # Values cannot be negative:
        x -= np.amin(x)
    # Values cannot be 0:
    x = x + 0.0000001
    # Values must be sorted:
    x = np.sort(x)
    # Index per array element:
    index = np.arange(1,x.shape[0]+1)
    # Number of array elements:
    n = x.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * x)) / (n * np.sum(x)))

def create_df(col_names, lists):
    
    df = pd.DataFrame()
    for i, name in enumerate(col_names):
        df[name] = lists[i]
    
    return df


def filter_top_users(df_users, label_community):
    df_users_community = df_users[df_users.community==label_community]
    number_top_users = int(len(df_users_community)*top_user_fraction)
    df_top = df_users_community.sort_values(by='total-degree', ascending=False)[:number_top_users]
    return df_top

def read_cleaned_war_data(PATH_WAR):
    
    '''
    Here we read the file with all the retweets and we store it in df.
    '''
    
    df=pd.read_pickle(PATH_WAR)
    
    '''
    We delete all the rows with at least a null value.
    '''
    df=df.dropna(how='any')
    
    '''
    We create the column 'created_at_days' and we use it for all the dataframe
    '''
    df['created_at_days']=[datetime.datetime(t.year,t.month,t.day) for t in df['created_at'].tolist()]
    
    '''
    We rename the df column from user.id to user.
    '''
    df=df.rename(columns={'user.id': 'user'})
    return df
    
def n_tweets_over_time(df, df_top, label_community):
    '''
    Let's create a dataframe in which we have group A over time: we use a 'left join' 
    where the reference is the user we are interested in and on the other side we have
    the time during which he's active.
    '''
    dGroup_time = df_top.set_index('user').join(df.set_index('user'))
    '''
    Our goal is to establish it there is a change in percentage with respect to the total 
    number of retweets.
    '''
    
    '''
    Here we create other 3 dataframes in which I have the total number of tweets,
    the number of tweets in group A and in group B and I rename the number of the column
    to make it more explanatory.
    '''
    df_tweets = dGroup_time[dGroup_time['created_at_days']<(dGroup_time['created_at_days'].max()-pd.Timedelta('1 days'))].groupby('created_at_days').count()[['community']]
    df_tweets.columns = [label_community]
    return df_tweets

                                      
