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
from configurations import SEED, top_user_fraction, PATH_WAR, DIR_FILES, min_rt

def assign_communities(G, com_of_user):
    '''
    This function returns the subgraph containing the nodes belonging to group A and group B with a degree>0 and two lists
    containing the users belonging to group A and the users belonging to group B, respectively, taking in input the graph G and
    'com_of_user' into which we save and load our data by using pickle module. 
    
    :param G:network in networkX format.
    :param com_of_user: com_of_user is a dictionary having the Id user as key 
    and the community as value.
    
    :return Gvac_subgraph: subgraph of G in networkX format containing the nodes belonging to group A and group B with a degree>0
    :return Gvac_A: subgraph of G in networkX format containing the nodes belonging to group A
    :return Gvac_B: subgraph of G in networkX format containing the nodes belonging to group B
    :return group_A: list of strings containing the Id users of group A
    :return group_B: list of strings containing the Id users of group B
    '''
    #We want to assign a node attribute to store the value of that property for each node: the attribute in question is the
    #belonging to a community.
    for node in nx.nodes(G):
        com_of_user.setdefault(node,'')
    nx.set_node_attributes(G, com_of_user, "community")
    
    #We define group_A if the user belongs to community A, group_B  if  the user belongs to community B and group_null if the
    #users don't belong to one of the two groups.
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
    
    #We consider the subgraph including the users belonging to group A or group B and the users with a degree greater than 0.
    Gvac_subgraph = G.subgraph(list(group_A|group_B))
    
    list_nodes_selected = [node for node in nx.nodes(Gvac_subgraph) 
        if (Gvac_subgraph.in_degree(node)>0 or Gvac_subgraph.out_degree(node)>0)]
    Gvac_subgraph = G.subgraph(list_nodes_selected)
    
    #We get two lists including the users belonging to groupA and to group B, respectively, with degree greater than 0.
    group_A = list(set(group_A) & set(list_nodes_selected))
    group_B = list(set(group_B) & set(list_nodes_selected))
    
    Gvac_A = G.subgraph(list(group_A)) 
    Gvac_B = G.subgraph(list(group_B))
    return Gvac_subgraph, Gvac_A, Gvac_B, group_A, group_B


def mixing_matrix(G, com_of_user):
    '''
    This function returns the mixing matrix, thus the 2x2 matrix that has as entries the number of edges starting from A and
    ending in A, starting from A and ending in B, starting from B and ending in A and starting from B and ending in B, taking as
    input the graph G and 'com_of_user' into which we save and load our data by using pickle module.
    
    :param G:network in networkX format.
    :param com_of_user: com_of_user is a dictionary having the Id user as key and the community as value.
    
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
    This function creates a random network and compute the modularity in the unweighted and weighted case, taking in input the
    number of swaps, the graph Gvac and the lists containing the users belonging to group A or group B.
    
    :param n_swaps: integer number representing the number of swaps.
    :param Gvac: network in networkX format.
    :param group_A: list of strings containing the Id users of group A.
    :param group_B: list of strings containing the Id users of group B.
    
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
    #i=0

    #Let's create an array of indexes in which each index corresponds to an element of list_edges
    array_index = np.arange(len(list_edges))
    #specify random seed
    np.random.seed(seed)
    '''
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
    '''
    list_edges, edge_weight = swapping(n_swaps, array_index, list_edges)
    
    #We create the randomized network and we evaluate the modularity in the unweighted 
    #and weighted cases.
    Gvac_shuffle = nx.from_edgelist(list(set(list_edges)),nx.DiGraph)
    modularity_unweighted = nx.community.modularity(Gvac_shuffle, [group_A,group_B])
    nx.set_edge_attributes(Gvac_shuffle, edge_weight, "weight")
    modularity_weighted = nx.community.modularity(Gvac_shuffle, 
                                                  [group_A,group_B], weight = 'weight')
    return modularity_unweighted, modularity_weighted, Gvac_shuffle

def swapping(n_swaps, array_index, list_edges):
    i=0
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
    return list_edges, edge_weight

def compute_randomized_modularity(Gvac_subgraph, group_A, group_B):
    '''
    Function that returns two lists containing the modularity in the unweighted and weighted case, taking in input the subgraph
    Gvac_subgraph (that will be obtained from Assign_Communities) and two lists (that will contain the users belonging to group A 
    or group B).
    
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

def compute_connected_component(Gvac_subgraph, group_A, group_B, weak_or_strong):
    '''
    Function that returns G0 and G1 that will include the nodes of group A or group B belonging to the first and second strongly
    connected component, being the first strongly connected component G0, the second strongly connected component G1 and four
    lists group_A_G0, group_B_G0, group_A_G1, group_B_G1 which contain the users of group A and group B belonging to the first or
    second strongly connected component, giving in input the subgraph Gvac_subgraph (that will be obtained from
    Assign_Communities) and two lists (that will contain the users belonging to group A or group B).
    
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
    if weak_or_strong == "strong":
        Gcc = sorted(nx.strongly_connected_components(Gvac_subgraph), key=len, reverse=True)
    elif weak_or_strong == "weak":
        Gcc = sorted(nx.weakly_connected_components(Gvac_subgraph), key=len, reverse=True)
    else:
        raise ValueError("weak_or_strong parameter should be either strong or weak.")
    G0 = Gvac_subgraph.subgraph(Gcc[0])
    G1 = Gvac_subgraph.subgraph(Gcc[1])
    
    group_A_G0 = list(set(group_A) & set(list(G0.nodes())))
    group_B_G0 = list(set(group_B) & set(list(G0.nodes())))
    group_A_G1 = list(set(group_A) & set(list(G1.nodes())))
    group_B_G1 = list(set(group_B) & set(list(G1.nodes())))
    return group_A_G0, group_B_G0, group_A_G1, group_B_G1, G0, G1

def gini(x): 
    '''
    Function returning the Gini coefficient on a list x.
    
    Param x: list that will contain the daily in- or out-degree.
    
    return: float value reprenting the Gini index based on the bottom eq in the following link:
    http://www.statsdirect.com/help/generatedimages/equations/equation154.svg, from
    http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm .
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
    
def compute_betweeness(G0, G0_weak):
    betweenness = nx.betweenness_centrality(G0, k=500)
    betweenness_weak = nx.betweenness_centrality(G0_weak, k=60)
    in_degree_G0 = [G0.in_degree(node) for node in nx.nodes(G0)]
    out_degree_G0 = [G0.out_degree(node) for node in nx.nodes(G0)]
    in_degree_G0_weak = [G0_weak.in_degree(node) for node in nx.nodes(G0_weak)]
    out_degree_G0_weak = [G0_weak.out_degree(node) for node in nx.nodes(G0_weak)]
    
    return betweenness, betweenness_weak, in_degree_G0, out_degree_G0, in_degree_G0_weak, out_degree_G0_weak
    
    
def sort_data(G0, betweenness):
    nodes = []
    in_degreeG0 = []
    out_degreeG0 = []
    betweenessG0 = []
    
    for node in G0.nodes():
        nodes.append(node)
        in_degreeG0.append(G0.in_degree(node))
        out_degreeG0.append(G0.out_degree(node))
        betweenessG0.append(betweenness[node])
    
    #The zip() function takes the iterables, aggregates them in a tuple, and returns it.
    #The sorted() function sorts all the elements of the tuple in ascending order.
    nodes, in_degreeG0, out_degreeG0, betweenessG0 = zip(*sorted(zip(nodes,in_degreeG0,out_degreeG0,betweenessG0)))
    return nodes, in_degreeG0, out_degreeG0, betweenessG0

    
def create_df(col_names, lists):
    '''
    This function creates a dataframe.
    
    param col_names: string containing the name of each column of the dataframe
    param lists: lists containing specific information, for example the in-degree or the out-degree of each user.
    
    return df: general dataframe.
    '''
    df = pd.DataFrame()
    for i, name in enumerate(col_names):
        df[name] = lists[i]
    return df


def filter_top_users(df_users, label_community):
    '''
    This function returns a dataframe containing the top-users by taking into account the the total degree as in-degree+out-
    degree.
    
    param df_users: dataframe containing the user Id, the community, the in- and out-degree.
    param label_community: string representing the community: A or B.
    
    return df_top: dataframe containing the top-users
    '''
    df_users_community = df_users[df_users.community==label_community]
    number_top_users = int(len(df_users_community)*top_user_fraction)
    df_top = df_users_community.sort_values(by='total-degree', ascending=False)[:number_top_users]
    return df_top

def read_cleaned_war_data(PATH_WAR):    
    '''
    Function where we read the file with all the retweets and we store it in df, then we delete all the rows with at least a 
    null value, we create the column 'created_at_days' and we use it for all the dataframe, we rename the df column from user.id
    to user. Then we create a dataframe in which we have group A over time: we use a 'left join' where the reference is the user
    we are interested in and on the other side we have the time during which he's active. The goal is to establish it there is a
    change in percentage with respect to the total number of retweets and we we create other 3 dataframes in which I have the
    total number of tweets, the number of tweets in group A and in group B and I rename the number of the column to make it more
    explanatory.
    
    param PATH_WAR: file with all the retweets.
    
    return df: manipulated dataframe df 
    '''
    
    df=pd.read_pickle(PATH_WAR)
    df=df.dropna(how='any')
    df['created_at_days']=[datetime.datetime(t.year,t.month,t.day) for t in df['created_at'].tolist()]
    df=df.rename(columns={'user.id': 'user'})
    return df
    
def n_tweets_over_time_selected_community(df, df_top, label_community):
    '''
    Function where we create a dataframe in which we have group A or B over time: we use a 'left join' where the reference is the
    user we are interested in and on the other side we have the time during which he's active. Our goal is to establish it there
    is a change in percentage with respect to the total number of retweets. Here we create other 3 dataframes in which I have the
    total number of tweets, the number of tweets in group A and in group B and I rename the number of the column to make it more
    explanatory.
    
    param df: dataframe obtained from read_cleaned_war_data function.
    param df_top: dataframe containing the top-users
    param label_community: string corresponding to the community A or B.
    
    return df_tweets: dataframe containing the number of tweets over time.
    '''
    dGroup_time = df_top.set_index('user').join(df.set_index('user'))
    dGroup_time = dGroup_time[dGroup_time['created_at_days']<(dGroup_time['created_at_days'].max()-pd.Timedelta('1 days'))]
    df_tweets = dGroup_time.groupby('created_at_days').count()[['community']]
    df_tweets.columns = [label_community]
    return df_tweets

def n_tweets_over_time(df):
    df_tweets_minus_last_day = df[df['created_at_days']<(df['created_at_days'].max()-pd.Timedelta('1 days'))]
    df_tweets = df_tweets_minus_last_day.groupby('created_at_days').count()[['created_at']]
    df_tweets.columns = ['Ntweets'] 
    return df_tweets
    
    

def age_of_activity(Gvac_days):
    '''
    The age is defined in the following way: age 0 is referred to the 1st day (all users have age 0). During the second day there 
    will be users that have age 1, since a day passed, and new users that will have age 0, and so on. So, after the second day
    there will be some users of the first day that are active or not, there will be some new users and there will be some users
    that had not been active at all.
    In order to compute the average age of activity we have to use lists of dictionaries because I have to store the information
    of the previous day referring to that particular node. We want a list that contains the different days but within each day 
    we want to maintain the track of each user because or I have to add a new user or I have to consider a user who was active 
    also in the past. If our datestore is equal to 1 it means that we're dealing with the first day, thus we define our 
    dictionaries as empty. Otherwise we take the dictionary of the day before.
    In these for loop we read all the nodes and we divide the process into two steps:
    - we verify if each node has an in-degree>0, thus if the user taken into account joined actively the temporal graph of that
    day. In this case we add a day to the activity of that node;
    - we repeat the same procedure in the case of the out-degree.
    
    param Gvac_days: list containing the users data day by day in the time span considered.
    
    return nodes_age_in: list containing the age of activity of each node taking into account the in-degree.
    return nodes_age_out: list containing the age of activity of each node taking into account the out-degree.
    '''
    #The following lists will be lists of dictionaries in order to evaluate the
    #average age of activity.
    nodes_age_in = []
    nodes_age_out = []
    
    
    for i, Gvac in enumerate(Gvac_days):
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
    '''
    Function that, after having read the txt files day by day,  orders them in time and creates a list (date_store) containing
    all the dates in the time span considered. Then we read the edgelists day by day and we store them in a list (Gvac_days).
    
    param DIR_FILES: file containing all the dates to be selected.
    
    return date_store: list contanining all the dates in the time span considered.
    return Gvac_days: list containing the users data day by day in the time span considered.
    '''
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
    Taking in input the dataframe df, that represent the table with the number of links from A to A, from A to B, from B to A 
    and from B to B (mixing matrix), each entry is divided by the total number of links taken into account. Then e take into 
    account the total number of links starting from community A, i.e. the sum of the elements of the first row of the mixing
    matrix, and we divide each element of the first row by this number. Then we repeat the procedure for the second row.
    In this way we get the average behaviour if a link starts from community A or from community B. We will do these step
    in the weighted and unweighted cases.
    
    param df: dataframe representing the mixing matrix.
    
    return df1: dataframe representing a first manipulation of the mixing matrix.
    return df2: dataframe representing a second manipulation of the mixing matrix.
    '''
    tot_links = df.sum().sum()
    df1 = df/float(tot_links)
    df.loc['A']=df.loc['A']/df.sum(axis=1).to_dict()['A']
    df.loc['B']=df.loc['B']/df.sum(axis=1).to_dict()['B']
    df2 = df
    return df1, df2

def degree_distributions(G):
    '''
    We create 6 lists to store the in- out-degree of the nodes belonging to the whole network, group A and group B.
    
    param G: network in networkX format.
    
    return in_degree: list containing the in-degree values.
    return out_degree: list containing the out-degree values.
    '''
    in_degree = [G.in_degree(node) for node in nx.nodes(G)]
    out_degree = [G.out_degree(node) for node in nx.nodes(G)]
    return in_degree, out_degree 

def words_frequency(df, group):
    '''
    Here we are going to evaluate the frequency of the mostly used words within the two groups in order to understand if, as in
    the vaccine network where group A is pro vaccine and group B is against vaccine, also in the war network the users belonging
    to group A and group B share a similar opinion about a different topic, for example we would expect that users belonging
    to group A are pro Ukraine and users belonging to group B are pro Russia.
    
    Param df: dataframe containing the useful information as user id, retweeted status of each user and so on.
    param group: list of strings containing the Id users of a certain group (group A or group B).
    
    return value_list: list containing the occurrence or frequency of the last mostly used words.
    return key_list: list containing the last mostly used words.
    '''
    
    df_sel=df[['text', 'retweeted_status.user.id']].drop_duplicates() #It returns DataFrame with duplicate rows removed.
    df_group = pd.DataFrame({'user': group})
    #With set_index we set the DataFrame index (row labels) using one or more existing columns or arrays (of the correct length).
    #The join() method takes all items in an iterable and joins them into one string (in this way we get user and corresponding
    #text of the retweeted status).
    df_group_retweet = df_group.set_index('user').join(df_sel.set_index('retweeted_status.user.id')[['text']])
    df_group_retweet = df_group_retweet[df_group_retweet['text'].isnull()==False] #Detect missing values
    
    list_text = df_group_retweet['text'].tolist() #let's convert df_group_retweet['text'] to an ordinary list with same elements.
    text = ' '.join(list_text) #all items in list_text are joined into one string
    #let's delete some irrelevant words or characters.
    stop = set(stopwords.words('italian') + list(string.punctuation) + ['https', '...', '”', '“', '``', "''", '’',])
    #tokenizers can be used to find the words in a string. Then we go ahead with counting the word occurrence.
    listToken = [i  for i in word_tokenize(text.lower()) if i not in stop]
    counterToken = Counter(listToken)
    key_list = list(counterToken.keys())
    values_list = [counterToken[key] for key in key_list]
    values_list, key_list = zip(*sorted(zip(values_list,key_list)))
    
    return values_list, key_list

def get_daily_nodes(Gvac_days):
    '''
    This function returns the daily number of nodes for each network day by day in the time span considered.
    
    param Gvac_days: list containing the users data day by day in the time span considered.
    
    return nodes_original: list containing the number of nodes or users belonging to the network day by day.
    '''
    nodes_original = []
    for Gvac in Gvac_days:
        nodes_original.append(len(Gvac.nodes()))
    return nodes_original

def get_daily_Gini_in_out(Gvac_days):
    '''
    This function returns the values of Gini index day by day in the time span considered taking into account respectively the 
    in-degree or the out-degree distribution.
    
    param Gvac_days: list containing the users data day by day in the time span considered.
    
    return Gini_in_values: list containing the Gini index values day by day calculated taking into account the in-degree.
    return Gini_out_values: list containing the Gini index values day by day calculated taking into account the out-degree.
    '''
    Gini_in_values = []
    Gini_out_values = []
    for Gvac in Gvac_days:   
        #Here we save all the users who receive retweets and the users who retweets, respectively.
        in_degree_original, out_degree_original = degree_distributions(Gvac) 
        Gini_in = gini(in_degree_original)
        Gini_out = gini(out_degree_original)
        Gini_in_values.append(Gini_in)
        Gini_out_values.append(Gini_out)
    return Gini_in_values, Gini_out_values

def get_daily_assortativity(Gvac_days):
    '''
    This function returns the values of assortativity day by day in the time span considered.
    
    param Gvac_days: list containing the users data day by day in the time span considered.
    
    return assortativity_values: list containing the assortativity values day by day.
    '''
    assortativity_values = []
    for Gvac in Gvac_days:
        assortativity=nx.degree_assortativity_coefficient(Gvac)
        assortativity_values.append(assortativity)    
    return assortativity_values

def get_daily_modularity(Gvac_days, com_of_user):
    '''
    This function allows to evaluate the modularity of the real and shuffled networks day by day either in the unweighted case
    or in the weighted case. In addition it returns the number of nodes belonging to group A and the number of nodes belonging to
    group B day by day.
    
    param Gvac_days: list containing the users data day by day in the time span considered.
    param com_of_user: com_of_user is a dictionary having the Id user as key and the community as value.
    
    return mod_unweighted_file: list containing the modularity of the real network day by day in the unweighted case.
    return mod_weighted_file: list containing the modularity of the real network day by day in the weighted case.
    return random_mod_unweighted_file: list containing the modularity of the shuffled network day by day in the unweighted case.
    return random_mod_weighted_file: list containing the modularity of the shuffled network day by day in the weighted case.
    return nodes_group_A: list containing the number of nodes belonging to group A day by day.
    return nodes_group_B: list containing the number of nodes belonging to group B day by day.
    '''
    mod_unweighted_file = []
    mod_weighted_file = []
    random_mod_unweighted_file = []
    random_mod_weighted_file = []
    nodes_group_A = []
    nodes_group_B = []
    
    for Gvac in Gvac_days:
        Gvac_subgraph, Gvac_A, Gvac_B, group_A, group_B = assign_communities(Gvac, com_of_user)
        list_modularity_unweighted,list_modularity_weighted=compute_randomized_modularity(Gvac_subgraph, group_A, group_B)
        mod_unweighted=nx.community.modularity(Gvac_subgraph, [group_A,group_B], weight = None)
        mod_weighted=nx.community.modularity(Gvac_subgraph, [group_A,group_B])
        mod_unweighted_file.append(mod_unweighted)
        mod_weighted_file.append(mod_weighted)
        random_mod_unweighted_file.append(list_modularity_unweighted)
        random_mod_weighted_file.append(list_modularity_weighted)
        nodes_group_A.append(len(group_A))
        nodes_group_B.append(len(group_B))
        
    return mod_unweighted_file, mod_weighted_file, random_mod_unweighted_file, random_mod_weighted_file, nodes_group_A, nodes_group_B

def get_daily_components(Gvac_days, com_of_user):
    '''
    This function returns the first strongly and weakly connected components including nodes of group A or group B and the 
    second strongly and weakly connected components including nodes of group A or group B.
    
    param Gvac_days: list containing the users data day by day in the time span considered.
    param com_of_user: com_of_user is a dictionary having the Id user as key and the community as value.
    
    return nodes_group_A_G0: list containing the nodes of group A belonging to the first strongly connected component.
    return nodes_group_B_G0: list containing the nodes of group B belonging to the first strongly connected component.
    return nodes_group_A_G1: list containing the nodes of group A belonging to the second strongly connected component.
    return nodes_group_B_G1: list containing the nodes of group B belonging to the second strongly connected component.
    return nodes_group_A_G0_weak: list containing the nodes of group A belonging to the first weakly connected component.
    return nodes_group_B_G0_weak: list containing the nodes of group B belonging to the first weakly connected component.
    return nodes_group_A_G1_weak: list containing the nodes of group A belonging to the second weakly connected component.
    return nodes_group_B_G1_weak: list containing the nodes of group A belonging to the second weakly connected component.
    '''
    nodes_group_A_G0 = []
    nodes_group_B_G0 = []
    nodes_group_A_G1 = []
    nodes_group_B_G1 = []
    
    nodes_group_A_G0_weak = []
    nodes_group_B_G0_weak = []
    nodes_group_B_G1_weak = []
    nodes_group_A_G1_weak = []
    
    for Gvac in Gvac_days:
        Gvac_subgraph, Gvac_A, Gvac_B, group_A, group_B = assign_communities(Gvac, com_of_user)
        nodes_group_A_G0_1, nodes_group_B_G0_1, nodes_group_A_G1_1, nodes_group_B_G1_1, G0,G1 = compute_connected_component(Gvac_subgraph, group_A, group_B, 'strong')
        nodes_group_A_G0_weak_1, nodes_group_B_G0_weak_1, nodes_group_A_G1_weak_1, nodes_group_B_G1_weak_1, G0_weak, G1_weak = compute_connected_component(Gvac_subgraph, group_A, group_B, 'weak')
        
        nodes_group_A_G0.append(len(nodes_group_A_G0_1))
        nodes_group_B_G0.append(len(nodes_group_B_G0_1))
        nodes_group_A_G1.append(len(nodes_group_A_G1_1))
        nodes_group_B_G1.append(len(nodes_group_B_G1_1))
        
        nodes_group_A_G0_weak.append(len(nodes_group_A_G0_weak_1))
        nodes_group_B_G0_weak.append(len(nodes_group_B_G0_weak_1))
        nodes_group_A_G1_weak.append(len(nodes_group_A_G1_weak_1))
        nodes_group_B_G1_weak.append(len(nodes_group_B_G1_weak_1))
      
    return nodes_group_A_G0, nodes_group_B_G0, nodes_group_A_G1, nodes_group_B_G1, nodes_group_A_G0_weak, nodes_group_B_G0_weak, nodes_group_A_G1_weak, nodes_group_B_G1_weak

def col_retweet_network(df, min_rt):
    list_col = ['user.id','retweeted_status.user.id']
    df_edgelist=df[list_col].copy()
    df_edgelist=df_edgelist.dropna(how='any')
    df_edgelist=df_edgelist.groupby(list_col).size()
    df_edgelist = df_edgelist.reset_index().rename(columns={0:'weight'})
    
    return df_edgelist

def compute_clustering(Gvac):
    '''
    Here we evaluate the clustering coefficient
    '''
    lcc = nx.clustering(Gvac)
    nodes = []
    clustering = []
    for node in Gvac.nodes():
        nodes.append(node)
        clustering.append(lcc[node])
    return nodes, clustering