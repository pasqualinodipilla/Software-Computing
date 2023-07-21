import pandas as pd
import networkx as nx
import pickle
import numpy as np
import datetime 
import os
from collections import Counter 
import copy
from functions import assign_communities, mixing_matrix, randomize_network, compute_randomized_modularity
from functions import compute_connected_component, compute_weak_connected_component, gini, compute_strong_or_weak_components
from functions import create_df, filter_top_users, read_cleaned_war_data, n_tweets_over_time, age_of_activity, create_date_store
from functions import degree_distributions, get_daily_nodes, get_daily_Gini_in_out, get_daily_assortativity, get_daily_modularity
from functions import get_daily_components, n_tweets_over_time_selected_community
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
    PATH_WAR,
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
    We read the data of the "war network" by considering the same communities of the "vaccine network" 
    '''
    with open(STOR_DIR+PATH_COM_OF_USER,'rb') as f:
        com_of_user=pickle.load(f)
    
    date_store, Gvac_days = create_date_store(DIR_FILES)
    nodes_original = get_daily_nodes(Gvac_days)
    Gini_in_values, Gini_out_values = get_daily_Gini_in_out(Gvac_days)
    assortativity_values = get_daily_assortativity(Gvac_days)
    nodes_age_in, nodes_age_out = age_of_activity(Gvac_days)
    mod_unweighted_file, mod_weighted_file, random_mod_unweighted_file, random_mod_weighted_file, nodes_group_A, nodes_group_B = get_daily_modularity(Gvac_days, com_of_user)
    nodes_group_A_G0, nodes_group_B_G0, nodes_group_A_G1, nodes_group_B_G1, nodes_group_A_G0_weak, nodes_group_B_G0_weak, nodes_group_A_G1_weak, nodes_group_B_G1_weak= get_daily_components(Gvac_days, com_of_user)
   
    '''
    In the following 2 dataframes we store firstly for the strong components and then for the weak components the dates 
    (date_store), the nodes of group A belonging to the first connected component, the nodes of group B belonging to the
    first connected component, the nodes of group A belonging to the second connected component, the nodes of group B belonging
    to the second strongly connected component.
    '''
    df_components = create_df(['date_store', 'nodes_group_B_G0', 'nodes_group_A_G0', 'nodes_group_B_G1', 'nodes_group_A_G1'],
                             [date_store, nodes_group_B_G0, nodes_group_A_G0, nodes_group_B_G1, nodes_group_A_G1])
    df_components_weak = create_df(['date_store','nodes_group_B_G0_weak','nodes_group_A_G0_weak','nodes_group_B_G1_weak',
                                    'nodes_group_A_G1_weak'],[date_store, nodes_group_B_G0_weak, nodes_group_A_G0_weak,
                              nodes_group_B_G1_weak, nodes_group_A_G1_weak])
    
    df_components.to_csv(PATH_S_COMPONENTS+'Figure8.csv', index=False)
    df_components_weak.to_csv(PATH_W_COMPONENTS+'Figure9.csv', index=False)
    
    '''
    In the following 4 dataframes we store respectively 
    - the dates (date_store) and the assortativity values,
    - the dates (date_store) and the Gini index values for in- and out- distributions,
    - the dates (date_store), the nodes belonging to a single community of the war network, where the communities are taken from
    the vaccine network (nodes_group_A, nodes_group_B) and all the nodes belonging to our war network (nodes_original),
    - the dates (date_store), the modularity of the real network and the modularity of the radomized one.
    '''
    df_assortativity = create_df(['date_store', 'assortativity_values'],[date_store, assortativity_values])
    df_gini = create_df(['date_store', 'Gini_in_values', 'Gini_out_values'], [date_store, Gini_in_values, Gini_out_values])
    df_n_of_nodes = create_df(['date_store', 'nodes_group_A', 'nodes_group_B', 'nodes_original'], [date_store, nodes_group_A,
                                                                                                  nodes_group_B, nodes_original])
    df_modularity = create_df(['date_store', 'mod_weighted_file', 'random_mod_weighted_file'],[date_store, mod_weighted_file,
                                                                                              random_mod_weighted_file])
    
    df_assortativity.to_csv(PATH_ASSORT+'Figure10.csv', index=False)
    df_gini.to_csv(PATH_GINI+'Figure11.csv', index=False)
    df_n_of_nodes.to_csv(PATH_N_NODES+'Figure12.csv', index=False)
    df_modularity.to_csv(PATH_MODULARITY+'Figure13.csv', index=False)
    
    '''
    In this dataframe we store the dates (date_store) and the age of the users in the case of the in- and out- distributions.
    The age is defined in the following way: age 0 is referred to the 1st day (all users have age 0). During the second day there 
    will be users that have age 1, since a day passed, and new users that will have age 0, and so on. So, after the second day
    there will be some users of the first day that are active or not, there will be some new users and there will be some users
    that had not been active at all.
    '''   
    df_age = create_df(['date_store', 'nodes_age_in', 'nodes_age_out'],[date_store, nodes_age_in, nodes_age_out])
    df_age.to_csv(PATH_AGE+'Figure14.csv', index=False)
    
    '''
    In order to evaluate the behaviour of the retweets for the top-scoring nodes we need 4 lists, the list of nodes in the order
    we read them, the list containing the community to which each node belongs, the lists with the corresponding in-degree and
    out-degree.
    '''   
    Gvac_subgraph, Gvac_A, Gvac_B, group_A, group_B = assign_communities(Gvac, com_of_user)
    nodes = [node for node in nx.nodes(Gvac_subgraph)]
    community = [Gvac_subgraph.nodes[node]["community"] for node in nodes]
    in_degree = [Gvac_subgraph.in_degree(node) for node in nodes]
    out_degree = [Gvac_subgraph.out_degree(node) for node in nodes]
    
    '''
    Let's create a dataframe where we put these 4 lists and we consider the top-users by taking into account the the total degree
    as in-degree+out-degree, thus we define another column.
    '''
    df_users = create_df(['user', 'community', 'in-degree', 'out-degree'],[nodes, community, in_degree, out_degree])
    df_users['total-degree']=df_users['in-degree']+df_users['out-degree']
    
    '''
    We define the top users of group A and group B on the basis of the total-degree (we are taking the 1%).
    '''
    df_topA = filter_top_users(df_users, 'A')
    df_topB = filter_top_users(df_users, 'B')
    df_topA.to_csv(PATH_FREQUENCY+'topUsersGroupA.csv', index=False)
    df_topB.to_csv(PATH_FREQUENCY+'topUsersGroupB.csv', index=False)
    
    df = read_cleaned_war_data(PATH_WAR)
    df.to_csv(PATH_FREQUENCY+'totalNofRetweets.csv.gz', index=False, compression='gzip')
    
    df_tweets_A = n_tweets_over_time_selected_community(df, df_topA, 'NtweetsGroupA')
    df_tweets_B = n_tweets_over_time_selected_community(df, df_topB, 'NtweetsGroupB')
    df_tweets = n_tweets_over_time(df)
    '''
    In order to join together the three dataframes we use an outer join: in the left join we took as reference only the dataframe 
    on the 'left part' and we took into account only the indexes of the dataframe on the 'left part'. With the outer join we take
    into account the indexes of all the dataframes considered.
    '''
    df_final = df_tweets_B.join(df_tweets_A,how='outer').join(df_tweets,how='outer').fillna(0) #Fill NA/NaN values with 0 
    '''
    Here we obtain and we save the fraction of retweets with respect to the total number of retweets in order to plot its
    behaviour over time: we create other 2 columns, the fraction of retweets in the case of group A and in the case of group B.
    '''
    df_final['fractionTweetsGroupB'] = df_final['NtweetsGroupB']/df_final['Ntweets']
    df_final['fractionTweetsGroupA'] = df_final['NtweetsGroupA']/df_final['Ntweets']
    df_final.to_csv(PATH_FREQUENCY+'Fraction.csv', index=False)   
main()
    
    
    
    
    
    
    
    