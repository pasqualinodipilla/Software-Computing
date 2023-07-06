import pandas as pd
import networkx as nx
import pickle
import numpy as np
import datetime 
import os
from collections import Counter 
import copy
from functions import assign_communities, mixing_matrix, randomize_network, compute_randomized_modularity
from functions import compute_connected_component, compute_weak_connected_component, gini
from configurations import (
    STOR_DIR,
    PATH_COM_OF_USER,
    PATH_S_COMPONENTS,
    PATH_W_COMPONENTS,
    PATH_ASSORT,
    PATH_GINI,
    PATH_N_NODES,
    PATH_AGE,
    PATH_MODULARITY,
    PATH_FREQUENCY,
    PATH_WAR
    DIR_FILES
)

'''
Network analysis over time.
Here we want to analyze the main properties of our network day by day from 01-02-2022 to 11-03-2022; in particular we compare day
by day the modularity of our real network with the modularity of a random network obtained with a shuffling of the edges
(configuration model), in order to demonstrate that our network is actually clustered. Then we evaluate the assortativity
coefficient, that is often used as a correlation measure between nodes, the Gini index, that is a measure of statistical
dispersion used to represent the inequality within a social group, and we identify the first two strongly and weakly connected
components in order to find out if the giant component is made up of users belonging to a single community or not.
'''

def main():
    '''
    We read the data of the "war network" by considering the same communities 
    of the "vaccine network" 
    in order to find out if the two networks share the same clusterization or not,
    or in other terms, to understand if the users belonging to a certain community 
    share a similar opinion about the two topics or not.
    '''
    with open(STOR_DIR+PATH_COM_OF_USER,'rb') as f:
        com_of_user=pickle.load(f)
    
    #let's save the directory in a variable (dir_files).
    #dir_files='./data/days/'
    #let's select all the .txt files.
    listfiles=[file for file in os.listdir(DIR_FILES) if file [-3:] == 'txt']
    #let's create lists in which we save the information we need
    mod_unweighted_file = []
    mod_weighted_file = []
    random_mod_unweighted_file = []
    random_mod_weighted_file = []
    assortativity_values = []
    Gini_in_values = []
    Gini_out_values = []
    #In the following list we save all the nodes belonging to our war network.
    nodes_original = []
    #Here we save the nodes belonging to a single community of the war network, 
    #where the communities are
    #taken from the vaccine network.
    nodes_group_A = []
    nodes_group_B = []
    #The following lists will be lists of dictionaries in order to evaluate the
    #average age of activity.
    nodes_age_in = []
    nodes_age_out = []
    #We create also a list in which we store the dates.
    date_store = []
    #We order listfiles with np.sort, i.e. the days one after the other.
    for file in np.sort(listfiles):
        #print(file)
        
        if file[17:19]+'-'+file[20:22] =='02-01':
            date_store.append(file[18:19]+'-'+file[20:22])
        elif file[17:19] =='03':
            date_store.append(file[18:19]+'-'+file[20:22])
        else:
            date_store.append(file[20:22])
    
        #print("Date =", date_store[-1])
        #The first step is to read the edgelist.
        Gvac=nx.read_weighted_edgelist(DIR_FILES+file,
                                           delimiter='\t',create_using=nx.DiGraph,nodetype=str)
        #print(nx.info(Gvac))
        nodes_original.append(len(Gvac.nodes()))
    
        #Here we save all the users who receive retweets and the users who retweets, respectively.
        in_degree_original = [Gvac.in_degree(node) for node in nx.nodes(Gvac)]
        out_degree_original = [Gvac.out_degree(node) for node in nx.nodes(Gvac)]
   
        #In order to compute the average age of activity we have to use lists of
        #dictionaries because I have to
        #store the information of the previous day referring to that particular node.
        #We want a list that contains the different days but within each day we want 
        #to maintain the track of
        #each user because or I have to add a new user or I have to consider a user who was active 
        #also in the past.

        #If our datestore is equal to 1 it means that we're dealing with the first day, 
        #thus we define our dictionaries as empty. Otherwise we take the dictionary 
        #of the day before.
        if len(date_store)==1:
            dict_nodes_in = {}
            dict_nodes_out = {}
            #if this step is satisfied they will be equal to the last element of the nodes list.
        else:
            dict_nodes_in = copy.deepcopy(nodes_age_in[-1])
            dict_nodes_out = copy.deepcopy(nodes_age_out[-1])
   
        #In these for loop we read all the nodes and we divide the process into two steps:
        #we verify if each node
        #has an in-degree>0, thus if the user taken into account joined actively 
        #the temporal graph of that day.
        #In this case we add a day to the activity of that node; we repeat the same 
        #procedure in the case of the out-degree.
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
    
    
    
        #We evaluate assortativity coefficient and Gini index for each day.
        assortativity=nx.degree_assortativity_coefficient(Gvac)
        #print("Assortativity coefficient =", assortativity)
        Gini_in = gini(in_degree_original)
        Gini_out = gini(out_degree_original)
        #print("Gini index in-degree =", Gini_in)
        #print("Gini index out-degree =", Gini_out)
    
        assortativity_values.append(assortativity)
        Gini_in_values.append(Gini_in)
        Gini_out_values.append(Gini_out)
    
        #We assigned the two communities A and B.
        Gvac_subgraph, Gvac_A, Gvac_B, group_A, group_B = assign_communities(Gvac, com_of_user)
        nodes_group_A.append(len(group_A))
        nodes_group_B.append(len(group_B))
    
        #We evaluate the modularity of the real network and the randomized one.
        list_modularity_unweighted,list_modularity_weighted=compute_randomized_modularity(Gvac_subgraph,
                                                                                          group_A,
                                                                                          group_B)
        mod_unweighted=nx.community.modularity(Gvac_subgraph, [group_A,group_B], weight = None)
        mod_weighted=nx.community.modularity(Gvac_subgraph, [group_A,group_B])
        #print("Modularity real network, Modularity randomized network, Standard deviation
        #(Unweighted case)")
        #print(mod_unweighted,np.mean(list_modularity_unweighted),np.std(list_modularity_unweighted))
        #print("Modularity real network, Modularity randomized network, Standard deviation
        #(Weighted case)")
        #print(mod_weighted,np.mean(list_modularity_weighted),np.std(list_modularity_weighted))
        mod_unweighted_file.append(mod_unweighted)
        mod_weighted_file.append(mod_weighted)
        random_mod_unweighted_file.append(list_modularity_unweighted)
        random_mod_weighted_file.append(list_modularity_weighted)
        #print(" ")
    
        #I want to plot the fraction of nodes of the first two (strongly and weakly) 
        #connected components
        #belonging to community A or B, thus I create a temporary variable x that represents 
        #a list whose elements
        #correspond to the output of the function at issue. Therefore x[0] will contain 
        #the users of group A 
        #belonging to the first strongly connected component, x[1] the users of group B 
        #belonging to the first
        #strongly connected component and so on. Then we save len(x[]) in one of the proper
        #4 lists in order
        #to get the number of nodes of group A or group B that belong to the first and second 
        #(strongly or weakly) connected connected components.
        
        isweak = False
        nodes_group_A_G0, nodes_group_B_G0, nodes_group_A_G1, nodes_group_B_G1 = compute_strong_or_weak_components(Gvac_subgraph,
                                                                                                                   group_A,
                                                                                                                   group_B, 
                                                                                                                   isweak)
        isweak = True
        nodes_group_A_G0_weak, nodes_group_B_G0_weak, nodes_group_A_G1_weak, nodes_group_B_G1_weak = compute_strong_or_weak_components(Gvac_subgraph,group_A,group_B, isweak)
    
    '''
    We create a set of dataframes and we save them in order to perform the plots 
    in Plot_Graph.ipynb.
    '''

    '''
    In this dataframe we store the dates (date_store), the nodes of group A belonging to the first
    strongly connected component (nodes_group_A_G0), the nodes of group B belonging to the first
    strongly connected component (nodes_group_B_G0), the nodes of group A belonging to the second
    strongly connected component (nodes_group_A_G1), the nodes of group B belonging to the second
    strongly connected component (nodes_group_B_G1).
    '''
    df_components = create_df(['date_store', 'nodes_group_B_G0', 'nodes_group_A_G0', 'nodes_group_B_G1', 'nodes_group_A_G1'],
                             [date_store, nodes_group_B_G0, nodes_group_A_G0, nodes_group_B_G1, nodes_group_A_G1])
    df_components.to_csv(PATH_S_COMPONENTS+'Figure8.csv', index=False)
    
    '''
    In this dataframe we store the dates (date_store), the nodes of group A belonging to the first
    weakly connected component (nodes_group_A_G0_weak), the nodes of group B belonging to the first
    weakly connected component (nodes_group_B_G0_weak), the nodes of group A belonging to 
    the second weakly connected component (nodes_group_A_G1_weak), the nodes of group B 
    belonging to the second weakly connected component (nodes_group_B_G1_weak).
    '''
    df_components_weak = create_df_G0_G1(date_store,
                                         nodes_group_B_G0_weak,
                                         nodes_group_A_G0_weak,
                                         nodes_group_B_G1_weak,
                                         nodes_group_A_G1)
    df_components_weak.to_csv(PATH_W_COMPONENTS+'Figure9.csv', index=False)
    
    '''
    In this dataframe we store the dates (date_store) and the assortativity values.
    '''
    df_assortativity = pd.DataFrame({'date_store': date_store,
                                     'assortativity_values': assortativity_values})
    df_assortativity.to_csv(PATH_ASSORT+'Figure10.csv', index=False)
    
    '''
    In this dataframe we store the dates (date_store), the Gini index values for in- and 
    out- distributions.
    '''
    df_gini = pd.DataFrame({'date_store': date_store,
                            'Gini_in_values': Gini_in_values,
                            'Gini_out_values': Gini_out_values})
    df_gini.to_csv(PATH_GINI+'Figure11.csv', index=False)
    
    '''
    In this dataframe we store the dates (date_store), the nodes belonging to a single community 
    of the war network, where the communities are taken from the vaccine network 
    (nodes_group_A, nodes_group_B) and all the nodes belonging to our war network (nodes_original).
    '''
    df_n_of_nodes = pd.DataFrame({'date_store': date_store,
                                  'nodes_group_A':nodes_group_A, 
                                  'nodes_group_B': nodes_group_B,
                                  'nodes_original': nodes_original})
    df_n_of_nodes.to_csv(PATH_N_NODES+'Figure12.csv', index=False)
    
    '''
    In this dataframe we store the dates (date_store), the modularity of the real network and the 
    modularity of the radomized one.
    '''
    df_modularity = pd.DataFrame({'date_store': date_store,
                                  'mod_weighted_file': mod_weighted_file,
                                  'random_mod_weighted_file': random_mod_weighted_file})
    df_modularity.to_csv(PATH_MODULARITY+'Figure13.csv', index=False)
    
    '''
    In this dataframe we store the dates (date_store) and the age of the users in the case 
    of the in and out distributions. The age is defined in the following way: age 0 is referred 
    to the 1st day (all users have age 0). During the second day there will be users that 
    have age 1, since a day passed, and new users that will have age 0, and so on. So, after 
    the second day there will be some users of the first day that are active or not, there will
    be some new users and there will be some users that had not been active at all.
    '''
    
    df_age = pd.DataFrame({'date_store': date_store,
                           'nodes_age_in': nodes_age_in,
                           'nodes_age_out': nodes_age_out})
    df_age.to_csv(PATH_AGE+'Figure14.csv', index=False)
    
    '''
    We want to evaluate the behaviour of the retweets for the top-scoring nodes:
    I need 4 lists, the list of nodes in the order I read them, the list containing the community 
    to which each node belongs, the list with the corresponding in-degree and the list with the 
    corresponding out-degree.
    '''
    
    nodes = [node for node in nx.nodes(Gvac_subgraph)]
    community = [Gvac_subgraph.nodes[node]["community"] for node in nodes]
    in_degree = [Gvac_subgraph.in_degree(node) for node in nodes]
    out_degree = [Gvac_subgraph.out_degree(node) for node in nodes]
    
    '''
    Let's create a dataframe where we put these 4 lists.
    '''
    df_users = pd.DataFrame({'user': nodes,
                             'community': community,
                             'in-degree': in_degree,
                             'out-degree': out_degree})
    
    '''
    Now it depends on how we want to consider the top-users: we are analyzing them by taking 
    into account the degree, in particular we decide to consider the total degree as 
    in-degree+out-degree, thus we define another column.
    '''
    df_users['total-degree']=df_users['in-degree']+df_users['out-degree']
    
    '''
    We define the top users of group A on the basis of the total-degree (we are taking the 1%).
    '''
    df_topA = filter_top_users(df_users, 'A')
    df_topA.to_csv(PATH_FREQUENCY+'topUsersGroupA.csv', index=False)
    
    '''
    We define the top users of group B on the basis of the total-degree (we are taking the 1%).
    '''
    df_topB = filter_top_users(df_users, 'B')
    df_topB.to_csv(PATH_FREQUENCY+'topUsersGroupB.csv', index=False)
    
    df = read_cleaned_war_data(PATH_WAR)
    df.to_csv(PATH_FREQUENCY+'totalNofRetweets.csv.gz', index=False, compression='gzip')
    
    df_tweets_A = n_tweets_over_time(df, df_top_A, 'NtweetsGroupA')
    df_tweets_B = n_tweets_over_time(df, df_top_B, 'NtweetsGroupB')
    
    df4 = df[df['created_at_days']<(df['created_at_days'].max()-pd.Timedelta('1 days'))].groupby('created_at_days').count()[['created_at']]
    df4.columns = ['Ntweets']
    
    '''
    In order to join together the three dataframes we use an outer join: in the left join 
    we took as reference only the dataframe on the 'left part' and we took into account 
    only the indexes of the dataframe on the 'left part'. With the outer join we take 
    into account the indexes of all the dataframes considered.
    '''
    df5 = df2.join(df3,how='outer').join(df4,how='outer').fillna(0) #Fill NA/NaN values with 0
    
    '''
    We are interested in the fraction of retweets with respect to the total number of retweets
    (or the percentage if we multiply by 100). Thus we create other 2 columns: the fraction 
    of retweets in the case of group A and in the case of group B
    '''
    df5['fractionTweetsGroupB'] = df5['NtweetsGroupB']/df5['Ntweets']
    df5['fractionTweetsGroupA'] = df5['NtweetsGroupA']/df5['Ntweets']
    
    '''
    We save these data in order to plot of the behaviour over time of the fraction of 
    retweets in group A and group B with respect to the total number of retweets.
    '''
    df5.to_csv(PATH_FREQUENCY+'Fraction.csv', index=False)
    
    
main()
    
    
    
    
    
    
    
    