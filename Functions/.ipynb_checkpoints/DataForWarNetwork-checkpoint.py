import pandas as pd
import networkx as nx
import pickle
from collections import Counter 
from scipy import stats
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from functions import assign_communities
from functions import mixing_matrix
from functions import randomize_network
from functions import compute_randomized_modularity
from functions import compute_connected_component
from functions import compute_weak_connected_component
from functions import gini

STOR_DIR='/mnt/stor/users/francesco.durazzi2/twitter/' #Paths used
PATH_EDGELIST='./data/edgelist_w1.txt'
PATH_COM_OF_USER = 'vaccines/data/v_com_of_user_w1_2020-10-07_2022-03-10.pkl'
PATH_MIXING_MATRIX='../Data/MixingMatrixTables/'
PATH_DEGREE_WAR = "../Data/DegreeDistributionsWar/"
DATA_SPEARMAN = '../Data/TableSpearman/'
DATA_BETWEENESS = '../Data/Betweeness/'
DATA_CLUSTERING = '../Data/ClusteringDistribution/'
PATH_WAR = '../Data/WarTweets.pkl'
DATA_FREQUENCY = '../Data/Frequency/'


def main():
    Gvac=nx.read_weighted_edgelist(PATH_EDGELIST,
                                   delimiter='\t',create_using=nx.DiGraph,nodetype=str)
    #print(nx.info(Gvac))
    
    with open(STOR_DIR+PATH_COM_OF_USER,'rb') as f:
        com_of_user=pickle.load(f)
    
    Gvac_subgraph, Gvac_A, Gvac_B, group_A, group_B = assign_communities(Gvac, com_of_user)
    df, df_weight = mixing_matrix(Gvac, com_of_user)
    
    '''
    I save a file for each table.
    Table representing the number of links from A to A, from A to B, from B to A and from B 
    to B (mixing matrix).
    '''
    
    df.head()
    df.to_csv(PATH_MIXING_MATRIX+'MixingWar1.csv', index=False)
    
    '''
    Each entry is divided by the total number of links taken into account.
    '''
    total_links = df.sum().sum()
    df1 = df/float(total_links)
    df1.to_csv(PATH_MIXING_MATRIX+'MixingWar2.csv', index=False)
    
    '''
    We take into account the total number of links starting from community A, i.e. 
    the sum of the elements of the first row of the mixing matrix, and we divide each 
    element of the first row by this number. Then we repeat the procedure for the second row.
    In this way we get the average behaviour if a link starts from community A or from community B.
    '''
    df.loc['A']=df.loc['A']/df.sum(axis=1).to_dict()['A']
    df.loc['B']=df.loc['B']/df.sum(axis=1).to_dict()['B']
    df2 = df
    df2.to_csv(PATH_MIXING_MATRIX+'MixingWar3.csv', index=False)
    
    '''
    Mixing matrix in weighted case.
    '''
    df_weight.head()
    df3 = df_weight
    df3.to_csv(PATH_MIXING_MATRIX+'MixingWar4.csv', index=False)
    
    '''
    Each entry is divided by the total number of links taken into account, in the weighted case.
    '''
    total_links = df_weight.sum().sum()
    df4 = df_weight/float(total_links)
    df4.to_csv(PATH_MIXING_MATRIX+'MixingWar5.csv', index=False)
    
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
    df5.to_csv(PATH_MIXING_MATRIX+'MixingWar6.csv', index=False)
    
    com_of_usersWar={}
    for node in nx.nodes(Gvac):
        com_of_usersWar[node]=com_of_user[node]
        
    '''
    Number of nodes belonging to the communities in the war network.
    '''
    Counter(list(com_of_usersWar.values()))
    
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
    df_original.to_csv(PATH_DEGREE_WAR+"DegreeOriginal.csv", index = False)
    
    df_degree_group_A = pd.DataFrame({'in_degree_group_A': in_degree_group_A,
                                      'out_degree_group_A': out_degree_group_A})
    df_degree_group_A.head()
    df_degree_group_A.to_csv(PATH_DEGREE_WAR+"DegreeGroupA.csv", index = False)
    
    df_degree_group_B = pd.DataFrame({'in_degree_group_B': in_degree_group_B,
                                      'out_degree_group_B': out_degree_group_B})
    df_degree_group_B.head()
    df_degree_group_B.to_csv(PATH_DEGREE_WAR+"DegreeGroupB.csv", index = False)
    
    '''
    Since we want to visualize the percentage occupied by each strongly (or weakly) 
    connected component we define the lists Gcc and Gcc_weak by using the sorted() 
    function to get the components in descending order. 
    '''
    Gcc = sorted(nx.strongly_connected_components(Gvac_subgraph), key=len, reverse=True)
    
    #Here we can see the percentage occupied by each strongly connected component.
    n_nodes = len(Gvac_subgraph.nodes())
    #for i, value in enumerate(Gcc):
        #print(100*len(Gcc[i])/float(n_nodes))
        
    Gcc_w = sorted(nx.weakly_connected_components(Gvac_subgraph), key=len, reverse=True)
    
    #Here we can see the percentage occupied by each weakly connected component.
    n_nodes = len(Gvac_subgraph.nodes())
    #for i, value in enumerate(Gcc_w):
        #print(100*len(Gcc_w[i])/float(n_nodes), len(Gcc_w[i]))
        
    '''
    We call the following two functions to get the nodes of group A or group B belonging 
    to the first and second strong (weak) connected component, the first strong (weak) 
    connected component G0 and the second strong (weak) connected component G1. After 
    that we need an undirected representation of the digraph in order to compute the betweeness.
    '''
    group_A_G0, group_B_G0, group_A_G1, group_B_G1, G0, G1 = compute_connected_component(Gvac_subgraph,
                                                                                       group_A,
                                                                                       group_B)
    (group_A_G0_weak, group_B_G0_weak, group_A_G1_weak, group_B_G1_weak, G0_weak, G1_weak) = compute_weak_connected_component(Gvac_subgraph,group_A,group_B)
    Gvac_subgraph.to_undirected()
    G0_weak_undirected = Gvac_subgraph.to_undirected()
    betweenness = nx.betweenness_centrality(G0, k=500)
    betweenness_weak = nx.betweenness_centrality(G0_weak_undirected, k=60)
    in_degree_G0 = [G0.in_degree(node) for node in nx.nodes(G0)]
    out_degree_G0 = [G0.out_degree(node) for node in nx.nodes(G0)]
    in_degree_G0_weak = [Gvac_subgraph.in_degree(node) for node in nx.nodes(Gvac_subgraph)]
    out_degree_G0_weak = [Gvac_subgraph.out_degree(node) for node in nx.nodes(Gvac_subgraph)]
    
    '''
    In order to plot betweeness vs in-degree and betweeness vs out-degree we have to sort the
    data in the same order.
    '''
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
    nodes, in_degreeG0, out_degreeG0, betweenessG0 = zip(*sorted(zip(nodes,
                                                                     in_degreeG0,
                                                                     out_degreeG0,
                                                                     betweenessG0)))
    nodesW = []
    in_degreeG0_weak = []
    out_degreeG0_weak = []
    betweenessG0_weak = []
    for node in Gvac_subgraph.nodes():
        nodesW.append(node)
        in_degreeG0_weak.append(Gvac_subgraph.in_degree(node))
        out_degreeG0_weak.append(Gvac_subgraph.out_degree(node))
        betweenessG0_weak.append(betweenness_weak[node])

    #The zip() function takes the iterables, aggregates them in a tuple, and returns it.
    #The sorted() function sorts all the elements of the tuple in ascending order.
    nodesW, in_degreeG0_weak, out_degreeG0_weak, betweenessG0_weak = zip(*sorted(zip(nodesW,in_degreeG0_weak, out_degreeG0_weak, betweenessG0_weak)))
    
    df_spearman = pd.DataFrame({'SpearmanName':['in-degree strong',
            'out-degree strong', 'in-degree weak', 'out-degree weak'], 
            'SpearmanValue':[stats.spearmanr(betweenessG0,in_degreeG0), 
                             stats.spearmanr(betweenessG0,out_degreeG0),
                             stats.spearmanr(betweenessG0_weak,in_degreeG0_weak),
                             stats.spearmanr(betweenessG0_weak,out_degreeG0_weak)]})
    #df_spearman.head()
    df_spearman.to_csv(DATA_SPEARMAN+'Spearman.csv', index=False)
    
    df_A = pd.DataFrame({'In-degree strong': in_degreeG0, 'Betweeness strong': betweenessG0})
    df_A.to_csv(DATA_BETWEENESS+'PanelA.csv', index=False)
    df_B = pd.DataFrame({'Out-degree strong': out_degreeG0, 'Betweeness strong': betweenessG0})
    df_B.to_csv(DATA_BETWEENESS+'PanelB.csv', index=False)
    df_C = pd.DataFrame({'In-degree weak': in_degreeG0_weak, 'Betweeness weak': betweenessG0_weak})
    df_C.to_csv(DATA_BETWEENESS+'PanelC.csv', index=False)
    df_D = pd.DataFrame({'Out-degree weak': out_degreeG0_weak,
                         'Betweeness weak': betweenessG0_weak})
    df_D.to_csv(DATA_BETWEENESS+'PanelD.csv', index=False)
    
    #here we evaluate the clustering coefficient
    lcc = nx.clustering(Gvac)
    nodes = []
    clustering = []
    for node in Gvac.nodes():
        nodes.append(node)
        clustering.append(lcc[node])
        
    df_clustering = pd.DataFrame({'Nodes': nodes, 'Clustering coefficient': clustering})
    df_clustering.to_csv(DATA_CLUSTERING+'ClusteringDistribution.csv', index=False)
    
    df_clust_indegree = pd.DataFrame({'In-degree': in_degree_original,
                                      'Clustering coefficient': clustering})
    df_clust_indegree.to_csv(DATA_CLUSTERING+'ClusteringInDegree.csv', index=False)
    
    df_clust_outdegree = pd.DataFrame({'Out-degree': out_degree_original,
                                       'Clustering coefficient': clustering})
    df_clust_outdegree.to_csv(DATA_CLUSTERING+'ClusteringOutDegree.csv',
                              index=False)
    
    '''
    Here we read the file with all the retweets and we store it in df.
    '''
    df=pd.read_pickle(PATH_WAR) 
    #print(df.count())
    
    nltk.download('stopwords')
    nltk.download('punkt')
    
    '''
    Here we are going to evaluate the frequency of the mostly used words within the two groups
    in order to understand if, as in the vaccine network where group A is pro vaccine and group
    B is against vaccine, also in the war network the users belonging to group A and group B share
    a similar opinion about a different topic, for example we would expect that users belonging
    to group A are pro Ukraine and users belonging to group B are pro Russia.
    '''
    df_sel=df[['text', 'retweeted_status.user.id']].drop_duplicates() 
    #It returns DataFrame with duplicate rows removed.
    df_A = pd.DataFrame({'user': group_A})
    #with set_index we set the DataFrame index (row labels) using one or more 
    #existing columns or arrays (of the correct length).
    #The join() method takes all items in an iterable and joins them into one string 
    #(in this way we get user and corresponding text of the retweeted status).
    df_A_retweet = df_A.set_index('user').join(df_sel.set_index('retweeted_status.user.id')[['text']])
    df_A_retweet = df_A_retweet[df_A_retweet['text'].isnull()==False] #Detect missing values
    
    list_text = df_A_retweet['text'].tolist() 
    #let's convert df_A_retweet['text'] to an ordinary list with the same elements.
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
    
    key_list[-20:] #last 20 mostly used words in groupA
    
    #Let's repeat the same proceduro for group B.
    df_sel=df[['text', 'retweeted_status.user.id']].drop_duplicates()
    df_B = pd.DataFrame({'user': group_B})
    df_B_retweet = df_B.set_index('user').join(df_sel.set_index('retweeted_status.user.id')[['text']])
    df_B_retweet = df_B_retweet[df_B_retweet['text'].isnull()==False]
    
    list_textB = df_B_retweet['text'].tolist()
    textB = ' '.join(list_textB)
    stopB = set(stopwords.words('italian') + list(string.punctuation) + ['https', '...', '”', '“', '``', "''", '’', '..'])
    listTokenB = [i  for i in word_tokenize(textB.lower()) if i not in stopB]
    counterTokenB = Counter(listTokenB)
    key_listB = list(counterTokenB.keys())
    values_listB = [counterTokenB[key] for key in key_listB]
    values_listB, key_listB = zip(*sorted(zip(values_listB,key_listB)))
    
    key_listB[-20:] #last 20 mostly used words in group B.
    
    df_frequencyA = pd.DataFrame({'key_list': key_list, 'values_list': values_list})
    df_frequencyB = pd.DataFrame({'key_listB': key_listB, 'values_listB': values_listB})
    df_frequencyA.to_csv(DATA_FREQUENCY+'Figure15_1.csv', index=False)
    df_frequencyB.to_csv(DATA_FREQUENCY+'Figure15_2.csv', index=False)
    
    
main()
    
    
    
    