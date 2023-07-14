import pandas as pd
import networkx as nx
import pickle
from collections import Counter 
from scipy import stats
import numpy as np
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import datetime
import string
import copy
from sklearn.metrics import roc_auc_score
from configurations import SEED, top_user_fraction, PATH_WAR, DIR_FILES

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
    if isweak:
        x = compute_weak_connected_component(Gvac_subgraph, group_A, group_B)
    else:
        x = compute_connected_component(Gvac_subgraph, group_A, group_B)
    return len(x[0]), len(x[1]), len(x[2]), len(x[3])
    
    
def gini(x): 
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    #x = x.flatten()
    '''
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
    '''
    total = 0
    if len(x) >= 1:
        i = 0
        for num in x:
            y = num * (2*i-len(x)-1) / len(x)*sum(x)
            total += y
            i += 1
        return total
    else:
        return 0

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

def age_of_activity(Gvac,i, nodes_age_in, nodes_age_out):
    '''
    In order to compute the average age of activity we have to use lists of dictionaries because I have to store the information
    of the previous day referring to that particular node. We want a list that contains the different days but within each day 
    we want to maintain the track of each user because or I have to add a new user or I have to consider a user who was active 
    also in the past. If our datestore is equal to 1 it means that we're dealing with the first day, thus we define our 
    dictionaries as empty. Otherwise we take the dictionary of the day before.
    In these for loop we read all the nodes and we divide the process into two steps:
    - we verify if each node has an in-degree>0, thus if the user taken into account joined actively the temporal graph of that
    day. In this case we add a day to the activity of that node;
    - we repeat the same procedure in the case of the out-degree.
    '''
    if i==0:
        dict_nodes_in = {}
        dict_nodes_out = {}
    #if this step is satisfied they will be equal to the last element of the nodes list.
    else:
        dict_nodes_in = copy.deepcopy(nodes_age_in[-1])
        dict_nodes_out = copy.deepcopy(nodes_age_out[-1])
    
    for node in nx.nodes(Gvac):
        if Gvac.in_degree(node)>0:
            dict_nodes_in.setdefault(node,0)
            dict_nodes_in[node]+=1
        if Gvac.out_degree(node)>0:
            dict_nodes_out.setdefault(node,0)
            dict_nodes_out[node]+=1
            
    #here we have our lists of dictionaries.  
    nodes_age_in.append(dict_nodes_in)
    nodes_age_out.append(dict_nodes_out)
    return nodes_age_in, nodes_age_out

def create_date_store(DIR_FILES):
    listfiles=[file for file in os.listdir(DIR_FILES) if file [-3:] == 'txt'] #let's select all the .txt files.
    date_store = []
    Gvac_days = []
    #We order listfiles with np.sort, i.e. the days one after the other.
    for file in np.sort(listfiles):

        if file[17:19]+'-'+file[20:22] =='02-01':
            date_store.append(file[18:19]+'-'+file[20:22])
        elif file[17:19] =='03':
            date_store.append(file[18:19]+'-'+file[20:22])
        else:
            date_store.append(file[20:22])

        #The first step is to read the edgelist.
        Gvac=nx.read_weighted_edgelist(DIR_FILES+file,
                                            delimiter='\t',create_using=nx.DiGraph,nodetype=str)
        
        Gvac_days.append(Gvac)
    return date_store, Gvac_days

def mixing_matrix_manipulation(df):
    '''
    df will represent the table with the number of links from A to A, from A to B, from B to A and from B to B 
    (mixing matrix).
    Each entry is divided by the total number of links taken into account.
    '''
    tot_links = df.sum().sum()
    df1 = df/float(tot_links)
    '''
    We take into account the total number of links starting from community A, i.e. 
    the sum of the elements of the first row of the mixing matrix, and we divide each 
    element of the first row by this number. Then we repeat the procedure for the second row.
    In this way we get the average behaviour if a link starts from community A or from community B. We will do these step
    in the weighted and unweighted cases.
    '''
    df.loc['A']=df.loc['A']/df.sum(axis=1).to_dict()['A']
    df.loc['B']=df.loc['B']/df.sum(axis=1).to_dict()['B']
    df2 = df
    return df1, df2

def degree_distributions(G):
    '''
    We create 6 lists to store the in- out-degree of the nodes belonging to the whole network, group A and group B a
    '''
    in_degree = [G.in_degree(node) for node in nx.nodes(G)]
    out_degree = [G.out_degree(node) for node in nx.nodes(G)]
    return in_degree, out_degree 

def words_frequency(df, group):
    '''
    Here we are going to evaluate the frequency of the mostly used words within the two groups
    in order to understand if, as in the vaccine network where group A is pro vaccine and group
    B is against vaccine, also in the war network the users belonging to group A and group B share
    a similar opinion about a different topic, for example we would expect that users belonging
    to group A are pro Ukraine and users belonging to group B are pro Russia.
    '''
    
    df_sel=df[['text', 'retweeted_status.user.id']].drop_duplicates() 
    #It returns DataFrame with duplicate rows removed.
    df_group = pd.DataFrame({'user': group})
    #with set_index we set the DataFrame index (row labels) using one or more existing columns or arrays (of the correct length).
    #The join() method takes all items in an iterable and joins them into one string 
    #(in this way we get user and corresponding text of the retweeted status).
    df_group_retweet = df_group.set_index('user').join(df_sel.set_index('retweeted_status.user.id')[['text']])
    df_group_retweet = df_group_retweet[df_group_retweet['text'].isnull()==False] #Detect missing values
    
    list_text = df_group_retweet['text'].tolist() 
    #let's convert df_group_retweet['text'] to an ordinary list with the same elements.
    text = ' '.join(list_text) #all items in list_text are joined into one string
    #let's delete some irrelevant words or characters.
    stop = set(stopwords.words('italian') + list(string.punctuation) + ['https', '...', '”', '“', '``', "''", '’',])
    #tokenizers can be used to find the words in a string. Then we go ahead with counting
    #the word occurrence.
    listToken = [i  for i in word_tokenize(text.lower()) if i not in stop]
    counterToken = Counter(listToken)
    key_list = list(counterToken.keys())
    values_list = [counterToken[key] for key in key_list]
    values_list, key_list = zip(*sorted(zip(values_list,key_list)))
    
    return value_list, key_list

def get_daily_nodes(Gvac_days):
    nodes_original = []
    for Gvac in Gvac_days:
        nodes_original.append(len(Gvac.nodes()))
    return nodes_original

def get_daily_Gini_in_out(Gvac_days):
    Gini_in_values = []
    Gini_out_values = []
    for i, Gvac in enumerate(Gvac_days):   
        #Here we save all the users who receive retweets and the users who retweets, respectively.
        in_degree_original, out_degree_original = degree_distributions(Gvac) 
        Gini_in = gini(in_degree_original)
        Gini_out = gini(out_degree_original)
        Gini_in_values.append(Gini_in)
        Gini_out_values.append(Gini_out)
    return Gini_in_values, Gini_out_values