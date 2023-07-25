import networkx as nx
import pickle
from functions import (
    assign_communities,
    create_df,
    filter_top_users,
    read_cleaned_war_data,
    n_tweets_over_time,
    age_of_activity,
    create_date_store,
    get_daily_nodes,
    get_daily_Gini_in_out,
    get_daily_assortativity,
    get_daily_modularity,
    get_daily_components,
    n_tweets_over_time_selected_community
)
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

def main():
    #We read the data of the "war network" by considering the same communities of the "vaccine network" 
    with open(STOR_DIR+PATH_COM_OF_USER,'rb') as f:
        com_of_user=pickle.load(f)
    
    date_store, Gvac_days = create_date_store(DIR_FILES)
    nodes_original = get_daily_nodes(Gvac_days)
    Gini_in_values, Gini_out_values = get_daily_Gini_in_out(Gvac_days)
    assortativity_values = get_daily_assortativity(Gvac_days)
    nodes_age_in, nodes_age_out = age_of_activity(Gvac_days)
    mod_unweighted_file, mod_weighted_file, random_mod_unweighted_file, random_mod_weighted_file, nodes_group_A, nodes_group_B = get_daily_modularity(Gvac_days, com_of_user)
    nodes_group_A_G0, nodes_group_B_G0, nodes_group_A_G1, nodes_group_B_G1, nodes_group_A_G0_weak, nodes_group_B_G0_weak, nodes_group_A_G1_weak, nodes_group_B_G1_weak= get_daily_components(Gvac_days, com_of_user)
   
    
    #we store for the strong components and for the weak components the dates, the nodes of group A and B belonging to the first     #connected component, the nodes of group A and B belonging to the second connected component.
    
    df_components = create_df(['date_store', 'nodes_group_B_G0', 'nodes_group_A_G0', 'nodes_group_B_G1', 'nodes_group_A_G1'],
                             [date_store, nodes_group_B_G0, nodes_group_A_G0, nodes_group_B_G1, nodes_group_A_G1])
    df_components_weak = create_df(['date_store','nodes_group_B_G0_weak','nodes_group_A_G0_weak','nodes_group_B_G1_weak',
                                    'nodes_group_A_G1_weak'],[date_store, nodes_group_B_G0_weak, nodes_group_A_G0_weak,
                              nodes_group_B_G1_weak, nodes_group_A_G1_weak])
    
    df_components.to_csv(PATH_S_COMPONENTS+'Figure8.csv', index=False)
    df_components_weak.to_csv(PATH_W_COMPONENTS+'Figure9.csv', index=False)
    
    #we store respectively the assortativity, the Gini index for in- and out- distributions, the nodes belonging to a single         #community of the war network and all the nodes belonging to our war network, the modularity of the real network and of the
    #radomized one
   
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
    
    #In this dataframe we store the dates (date_store) and the age of the users in the case of the in- and out- distributions.
    #The meaning of age of activity is explained in the corresponding function.
 
    df_age = create_df(['date_store', 'nodes_age_in', 'nodes_age_out'],[date_store, nodes_age_in, nodes_age_out])
    df_age.to_csv(PATH_AGE+'Figure14.csv', index=False)
    
    #To evaluate the behaviour of the retweets for the top-scoring nodes we need 4 lists, the list of nodes in the order we read
    #them, the list containing the community to which each node belongs, the lists with the corresponding in- and out- degree.
  
    for Gvac in Gvac_days:
        Gvac_subgraph, Gvac_A, Gvac_B, group_A, group_B = assign_communities(Gvac, com_of_user)
    nodes = [node for node in nx.nodes(Gvac_subgraph)]
    community = [Gvac_subgraph.nodes[node]["community"] for node in nodes]
    in_degree = [Gvac_subgraph.in_degree(node) for node in nodes]
    out_degree = [Gvac_subgraph.out_degree(node) for node in nodes]
    
    #Dataframe where we put the 4 lists and we consider the top-users in terms of the total degree, so we define another column.
 
    df_users = create_df(['user', 'community', 'in-degree', 'out-degree'],[nodes, community, in_degree, out_degree])
    df_users['total-degree']=df_users['in-degree']+df_users['out-degree']
    
    #We define the top users of group A and group B on the basis of the total-degree (we are taking the 1%).
    
    df_topA = filter_top_users(df_users, 'A')
    df_topB = filter_top_users(df_users, 'B')
    df_topA.to_csv(PATH_FREQUENCY+'topUsersGroupA.csv', index=False)
    df_topB.to_csv(PATH_FREQUENCY+'topUsersGroupB.csv', index=False)
    
    df = read_cleaned_war_data(PATH_WAR)
    df.to_csv(PATH_FREQUENCY+'totalNofRetweets.csv.gz', index=False, compression='gzip')
    
    df_tweets_A = n_tweets_over_time_selected_community(df, df_topA, 'NtweetsGroupA')
    df_tweets_B = n_tweets_over_time_selected_community(df, df_topB, 'NtweetsGroupB')
    df_tweets = n_tweets_over_time(df)
    
    #To join together the three dataframes we use an outer join: in the left join we took as reference only the dataframe on the
    #'left part' and we took into account only the indexes of the dataframe on the 'left part'. With the outer join we consider
    #the indexes of all the dataframes.
    
    df_final = df_tweets_B.join(df_tweets_A,how='outer').join(df_tweets,how='outer').fillna(0) #Fill NA/NaN values with 0 
    
    #Fraction of retweets with respect to the total number of retweets in order to plot its behaviour over time: we create other
    #2 columns, the fraction of retweets in the case of group A and in the case of group B.
    
    df_final['fractionTweetsGroupB'] = df_final['NtweetsGroupB']/df_final['Ntweets']
    df_final['fractionTweetsGroupA'] = df_final['NtweetsGroupA']/df_final['Ntweets']
    df_final.to_csv(PATH_FREQUENCY+'Fraction.csv', index=False)   
main()
    
    
    
    
    
    
    
    