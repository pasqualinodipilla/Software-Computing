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
from functions import assign_communities, mixing_matrix, randomize_network, compute_randomized_modularity, degree_distributions
from functions import compute_connected_component, compute_weak_connected_component, gini, mixing_matrix_manipulation, create_df
from functions import words_frequency
from configurations import (
    STOR_DIR,
    PATH_EDGELIST,
    PATH_MIXING_MATRIX,
    PATH_COM_OF_USER,
    PATH_DEGREE_WAR,
    DATA_SPEARMAN,
    DATA_BETWEENESS,
    DATA_CLUSTERING,
    PATH_WAR,
    DATA_FREQUENCY
)

def main():
    Gvac=nx.read_weighted_edgelist(PATH_EDGELIST,
                                   delimiter='\t',create_using=nx.DiGraph,nodetype=str)
    
    with open(STOR_DIR+PATH_COM_OF_USER,'rb') as f:
        com_of_user=pickle.load(f)
    
    Gvac_subgraph, Gvac_A, Gvac_B, group_A, group_B = assign_communities(Gvac, com_of_user)
    df, df_weight = mixing_matrix(Gvac, com_of_user) #mixing matrix in unweighted and weighted case
    df1, df2 = mixing_matrix_manipulation(df)
    df3, df4 = mixing_matrix_manipulation(df_weight)
    '''
    I save a file for each table.
    '''
    df.to_csv(PATH_MIXING_MATRIX+'MixingWar1.csv', index=False)
    df1.to_csv(PATH_MIXING_MATRIX+'MixingWar2.csv', index=False)
    df2.to_csv(PATH_MIXING_MATRIX+'MixingWar3.csv', index=False)
    df_weight.to_csv(PATH_MIXING_MATRIX+'MixingWar4.csv', index=False)
    df3.to_csv(PATH_MIXING_MATRIX+'MixingWar5.csv', index=False)
    df4.to_csv(PATH_MIXING_MATRIX+'MixingWar6.csv', index=False)
    
    com_of_usersWar={}
    for node in nx.nodes(Gvac):
        com_of_usersWar[node]=com_of_user[node]    
    '''
    Number of nodes belonging to the communities in the war network.
    '''
    Counter(list(com_of_usersWar.values()))
    
    '''
    We create 6 lists to store the in- out-degree of the nodes belonging to the whole network, group A and group B and we save
    them in a corresponding file.
    '''
    in_degree_original, out_degree_original = degree_distributions(Gvac)
    in_degree_group_A, out_degree_group_A = degree_distributions(Gvac_A)
    in_degree_group_B, out_degree_group_B = degree_distributions(Gvac_B)
    
    df_original = create_df(['in_degree_original', 'out_degree_original'],[in_degree_original, out_degree_original])
    df_degree_group_A = create_df(['in_degree_group_A', 'out_degree_group_A'],[in_degree_group_A, out_degree_group_A])
    df_degree_group_B = create_df(['in_degree_group_B', 'out_degree_group_B'],[in_degree_group_B, out_degree_group_B])

    df_original.to_csv(PATH_DEGREE_WAR+"DegreeOriginal.csv", index = False)
    df_degree_group_A.to_csv(PATH_DEGREE_WAR+"DegreeGroupA.csv", index = False)
    df_degree_group_B.to_csv(PATH_DEGREE_WAR+"DegreeGroupB.csv", index = False)
    
    '''
    We call the following two functions to get the nodes of group A or group B belonging to the first and second strong (weak)
    connected component, the first strong (weak) connected component G0 and the second strong (weak) connected component G1.
    After that we need an undirected representation of the digraph in order to compute the betweeness.
    '''
    group_A_G0, group_B_G0, group_A_G1, group_B_G1, G0, G1 = compute_connected_component(Gvac_subgraph,group_A,group_B)
    (group_A_G0_weak, group_B_G0_weak, group_A_G1_weak, group_B_G1_weak, G0_weak, G1_weak)= compute_weak_connected_component(Gvac_subgraph,group_A,group_B)
    
    G0_weak_undirected = Gvac_subgraph.to_undirected()
    betweenness = nx.betweenness_centrality(G0, k=500)
    betweenness_weak = nx.betweenness_centrality(Gvac_subgraph, k=60)
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
    df_spearman.to_csv(DATA_SPEARMAN+'Spearman.csv', index=False)
    
    df_A = create_df(['In-degree strong', 'Betweeness strong'],[in_degreeG0, betweenessG0])
    df_B = create_df(['Out-degree strong', 'Betweeness strong'],[out_degreeG0, betweenessG0])
    df_C = create_df(['In-degree weak', 'Betweeness weak'],[in_degreeG0_weak, betweenessG0_weak])
    df_D = create_df(['Out-degree weak', 'Betweeness weak'],[out_degreeG0_weak, betweenessG0_weak])
    
    df_A.to_csv(DATA_BETWEENESS+'PanelA.csv', index=False)
    df_B.to_csv(DATA_BETWEENESS+'PanelB.csv', index=False)
    df_C.to_csv(DATA_BETWEENESS+'PanelC.csv', index=False)
    df_D.to_csv(DATA_BETWEENESS+'PanelD.csv', index=False)
    
    '''
    Here we evaluate the clustering coefficient
    '''
    lcc = nx.clustering(Gvac)
    nodes = []
    clustering = []
    for node in Gvac.nodes():
        nodes.append(node)
        clustering.append(lcc[node])
        
    df_clustering = create_df(['Nodes', 'Clustering coefficient'],[nodes, clustering])
    df_clust_indegree = create_df(['In-degree', 'Clustering coefficient'],[in_degree_original, clustering])
    df_clust_outdegree = create_df(['Out-degree', 'Clustering coefficient'],[out_degree_original, clustering])
    
    df_clustering.to_csv(DATA_CLUSTERING+'ClusteringDistribution.csv', index=False)
    df_clust_indegree.to_csv(DATA_CLUSTERING+'ClusteringInDegree.csv', index=False)
    df_clust_outdegree.to_csv(DATA_CLUSTERING+'ClusteringOutDegree.csv',index=False)
    
    '''
    Here we read the file with all the retweets and we store it in df.
    ''' 
    df = pd.read_pickle(PATH_WAR, compression='gzip')
    
    nltk.download('stopwords')
    nltk.download('punkt')
    
    '''
    Here we are going to evaluate the frequency of the mostly used words within the two groups
    in order to understand if, as in the vaccine network where group A is pro vaccine and group
    B is against vaccine, also in the war network the users belonging to group A and group B share
    a similar opinion about a different topic, for example we would expect that users belonging
    to group A are pro Ukraine and users belonging to group B are pro Russia.
    
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
    '''
   
    key_list, value_list = words_frequency(df, group_A)
    key_listB, value_listB = words_frequency(df, group_B)
    
    df_frequencyA = create_df(['key_list', 'values_list'],[key_list, values_list])
    df_frequencyB = create_df(['key_listB', 'values_listB'],[key_listB, values_listB])
    df_frequencyA.to_csv(DATA_FREQUENCY+'Figure15_1.csv', index=False)
    df_frequencyB.to_csv(DATA_FREQUENCY+'Figure15_2.csv', index=False) 
main()
    
    
    
    