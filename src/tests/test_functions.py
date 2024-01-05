import pandas as pd
import networkx as nx
import pickle
import numpy as np
import datetime 
import os
from collections import Counter 
import copy
import random
import sys
sys.path.append('../functions')
from configurations import SEED, n_rand
from functions import (
    assign_communities,
    mixing_matrix,
    randomize_network,
    swapping,
    compute_randomized_modularity,
    compute_connected_component,
    gini,
    compute_betweeness,
    sort_data,
    create_df,
    filter_top_users,
    read_cleaned_war_data,
    n_tweets_over_time_selected_community,
    n_tweets_over_time,
    age_of_activity,
    create_date_store,
    mixing_matrix_manipulation,
    degree_distributions,
    words_frequency,
    get_daily_nodes,
    get_daily_Gini_in_out,
    get_daily_assortativity,
    get_daily_modularity,
    get_daily_components,
    col_retweet_network,
    compute_clustering
)


def test_assign_communities(read_file_G, read_file_com_of_user):
    '''
    This function is testing if the output of assign_communities() function are of the correct type:
    group_A and group_B must be lists, Gvac_subgraph, Gvac_A and Gvac_B must be direct graphs. Then it
    tests that the set list of nodes belonging to Gvac_A (i.e. the subgraph containing the nodes
    belonging to group A) actually corresponds to group_A and that the set list of nodes belonging to
    Gvac_B (i.e. the subgraph containing the nodes belonging to group B) actually corresponds to
    group_B. At the end it tests that the set list of nodes belonging to Gvac_subgraph is given by the
    set list of nodes belonging to Gvac_A or Gvac_B.
    
    :param read_file_G: variable into which we store the network G_vac in networkX format.
    :param read_file_com_of_user: variable into which we store com_of_user that is a dictionary having
    the Id user as key and the community as value.
    '''
   
    Gvac_subgraph, Gvac_A, Gvac_B, group_A, group_B=assign_communities(read_file_G, read_file_com_of_user)
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
    
def test_mixing_matrix(read_file_G, read_file_com_of_user, create_edgelist, compute_weights):
    '''
    This function is used to test if the 2 columns that we get in the mixing matrix for the unweighted
    and weighted cases are actually the number of edges coming respectively from users of group A and
    users of group B. In addition we test if the sum of all the elements of the 2 matrices corresponds
    to the overall number of edges of our graph, in both the cases unweighted and weighted.
    
    :param read_file_G: variable into which we store the network G_vac in networkX format.
    :param read_file_com_of_user: variable into which we store com_of_user that is a dictionary having
    the Id user as key and the community as value.
    :param create_edgelist: edgelist containing a possible combination AA, AB, BA, BB for each couple
    of nodes that are connected by a link.
    :param compute_weights: dictionary containing the weights of each link.
    '''
    
    df, df_weight = mixing_matrix(read_file_G, read_file_com_of_user)
    assert set(df.columns) == set(['A','B'])
    assert set(df_weight.columns) == set(['A','B'])
    assert int(df.sum().sum()) == len(create_edgelist)
    assert int(df_weight.sum().sum()) == int(sum(compute_weights))
    
def test_randomize_network(n_swaps, get_groupA_groupB):
    '''
    This function is used to test that, regardless of the shuffling, the number of edges connecting
    the nodes in the original graph Gvac_subgraph is actually equal to the overall number of edges
    connecting the nodes in the shuffled network, Gvac_shuffle.
    In addition we test if modularity_unweighted and modularity_weighted, outputs of the c
    orresponding function, are float type or not.
    
    :param n_swaps: integer number representing the number of swaps.
    :param get_groupA_groupB: dictionary having as values a network G, and two lists group_A and
    group_B containing the users belonging to group_A and group_B respectively.
    '''
    modularity_unweighted, modularity_weighted, Gvac_shuffle = randomize_network(SEED,n_swaps, get_groupA_groupB['G_subgraph'], get_groupA_groupB['group_A'], get_groupA_groupB['group_B'])
   
    assert len(Gvac_shuffle.nodes()) == len(get_groupA_groupB['G_subgraph'].nodes())
    sum(list(nx.get_edge_attributes(Gvac_shuffle,'weight').values())) == sum(list(nx.get_edge_attributes(get_groupA_groupB['G_subgraph'],'weight').values()))
    assert type(modularity_unweighted) == float
    assert type(modularity_weighted) == float
    
def test_compute_randomized_modularity(get_groupA_groupB):
    '''
    This function is used to test the type of the output of compute_randomized_modularity() function:
    list_modularity_unweighted, list_modularity_weighted must be lists. Then, since in our case we 
    perform n_rand =10 randomizations, we test if the lenght of the two lists is equal to n_rand.
    
    :param get_groupA_groupB: :param get_groupA_groupB: dictionary having as values a network G, and
    two lists group_A and group_B containing the users belonging to group_A and group_B respectively.
    '''
    list_modularity_unweighted,list_modularity_weighted=compute_randomized_modularity(get_groupA_groupB['G_subgraph'], get_groupA_groupB['group_A'],get_groupA_groupB['group_B'])
    
    assert type(list_modularity_unweighted) == list
    assert type(list_modularity_weighted) == list
    assert len(list_modularity_unweighted) == n_rand 
    assert len(list_modularity_weighted) == n_rand
    
def test_compute_connected_component(get_groupA_groupB):
    '''
    This function is used to test the type of the output of compute_connected_component() function:
    group_A_G0, group_B_G0, group_A_G1, group_B_G1 must be lists, G0 and G1 must be direct graphs.
    Then the functions tests that the nodes of the graph G0 (representing the first strongly 
    connected component) actually correspond to the set list of nodes belonging to group_A_G0 or
    group_B_G0. Similarly it tests if the nodes of the graph G1 (representing the second strongly 
    connected component) actually correspond to the set list of nodes belonging to group_A_G1 or
    group_B_G1.
    
    :param get_groupA_groupB: :param get_groupA_groupB: dictionary having as values a network G, and
    two lists group_A and group_B containing the users belonging to group_A and group_B respectively.
    '''
  
    #test weak_or_strong = strong
    group_A_G0, group_B_G0, group_A_G1, group_B_G1, G0, G1 = compute_connected_component(get_groupA_groupB['G_subgraph'],get_groupA_groupB['group_A'], get_groupA_groupB['group_B'], 'strong')
    
    assert type(group_A_G0) == list
    assert type(group_B_G0) == list
    assert type(group_A_G1) == list
    assert type(group_B_G1) == list
    assert type(G0) == type(nx.DiGraph())
    assert type(G1) == type(nx.DiGraph())
    assert set(group_A_G0) | set(group_B_G0) == set(G0.nodes())
    assert set(group_A_G1) | set(group_B_G1) == set(G1.nodes())
    
    #test weak_or_strong = weak
    group_A_G0, group_B_G0, group_A_G1, group_B_G1, G0, G1 = compute_connected_component(get_groupA_groupB['G_subgraph'],get_groupA_groupB['group_A'], get_groupA_groupB['group_B'], 'weak')
    
    assert type(group_A_G0) == list
    assert type(group_B_G0) == list
    assert type(group_A_G1) == list
    assert type(group_B_G1) == list
    assert type(G0) == type(nx.DiGraph())
    assert type(G1) == type(nx.DiGraph())
    assert set(group_A_G0) | set(group_B_G0) == set(G0.nodes())
    assert set(group_A_G1) | set(group_B_G1) == set(G1.nodes())
    
    
def test_gini(read_file_G, read_file_com_of_user):
    '''
    This function firstly tests if the type of the Gini index value given in output from the
    corresponding function is a float type (in both the cases Gini_in, Gini_out) and than, as a
    further confirmation, if we assume a pure equidistribution, for example with all values of the x
    list in input equal to 1, we test if we will obtain a gini index equal to 0, as it should be by
    definition. If instead we have the maximum of inequality, for example by assuming the first
    element equal to 1 and the others equal to 0, we will get a gini index almost equal to 1. 
    Eventually a Gini Index of 0.5 shows that there is equal distribution of elements across some
    classes.
    
    :param read_file_G: variable into which we store the network G_vac in networkX format.
    :param read_file_com_of_user: variable into which we store com_of_user that is a dictionary having
    the Id user as key and the
    community as value.
    '''
    in_degree_original = [read_file_G.in_degree(node) for node in nx.nodes(read_file_G)]
    out_degree_original = [read_file_G.out_degree(node) for node in nx.nodes(read_file_G)]
    
    Gini_in = gini(in_degree_original)
    Gini_out = gini(out_degree_original)

    Gini_0 = gini([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
    Gini_05 = gini([1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0])
    Gini_1 = gini([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    
    assert type(Gini_in) == float 
    assert type(Gini_out) == float 
    assert Gini_0 == 0
    assert Gini_1 >= 0.95 and Gini_1 <= 1.
    assert Gini_05 == 0.5
    
def test_swapping(n_swaps, create_list_edges):
    '''
    This function tests that the lenght of the list containing the edges of the original network is
    equal to the lenght of the same list in the case of the shuffled network. The same is done by
    considering the weights and then it tests that the set of edges of the original network is
    different from the set of edges of the shuffled network.
    
    :param n_swaps: integer number representing the number of swaps.
    :param create_list_edges: edgelist containing a possible combination AA, AB, BA, BB for each
    couple of nodes that are connected by a link.
    '''
    list_edges, edge_weight = swapping(n_swaps, copy.deepcopy(create_list_edges))
    assert len(list_edges)==len(create_list_edges)
    assert set(list_edges) != set(create_list_edges)
    assert sum(list(edge_weight.values())) == len(create_list_edges)
    
def test_create_df(create_input_for_df):
    '''
    This function tests that the dataframe in input (create_input_for_df), as an example, has 2 rows
    and 2 columns.
    
    :param create_input_for_df: dataframe used as an example in conftest.py.
    '''
    df = create_df(create_input_for_df['col_names'], create_input_for_df['list_content'])
    
    assert len(df) == 2
    assert len(df.columns) == 3 
    
def test_filter_top_users(input_test_filter_top_users):
    '''
    This function tests that the dataframe that is output of the filter_top_users function
    has two rows in the case of community A and top_users_fraction parameter = 0.5 and three rows
    in the case of community B and top_users_fraction parameter = 0.5. In addition by running this
    function in another notebook with the same input (input_test_filter_top_users) we tested the
    expected values in the two cases.
    
    :param input_test_filter_top_users: dataframe in input, defined in conftest.py.
    '''
    #test_filter_top_users for community A and top_users_fraction parameter = 0.5
    df_top = filter_top_users(input_test_filter_top_users, 'A', 0.5)
    assert len(df_top) == 2
    assert set(df_top['total-degree'].tolist()) == {22,13}
    
    #test_filter_top_users for community B and top_users_fraction parameter = 0.5
    df_top = filter_top_users(input_test_filter_top_users, 'B', 0.5)
    assert len(df_top) == 3
    assert set(df_top['total-degree'].tolist()) == {11,9,6} 
    
def test_mixing_matrix_manipulation(input_mixing_matrix_manipulation):
    '''
    This function tests that the dataframes in output of the mixing_matrix_manipulation function have
    2 rows and 2 columns and by defining with df3 and df4 the expected output for the first and second
    run, we impose that they actually coincide with df1 and df2.
    
    :param input_mixing_matrix_manipulation: dataframe in input defined in conftest.py.
    '''
    df1, df2 = mixing_matrix_manipulation(input_mixing_matrix_manipulation)
    assert len(df1.columns) == 2
    assert len(df1) == 2
    df3 = pd.DataFrame({'A':[0.1, 0.2],'B':[0.3, 0.4]}, index = ['A', 'B'])
    assert df1.loc['A', 'A'] == df3.loc['A', 'A']
    assert df1.loc['A', 'B'] == df3.loc['A', 'B']
    assert df1.loc['B', 'A'] == df3.loc['B', 'A']
    assert df1.loc['B', 'B'] == df3.loc['B', 'B']
    
    assert len(df2.columns) == 2
    assert len(df2) == 2
    df4 = pd.DataFrame({'A':[(1/4), (1/3)],'B':[(3/4), (2/3)]}, index = ['A', 'B'])
    assert df2.loc['A', 'A'] == df4.loc['A', 'A']
    assert df2.loc['A', 'B'] == df4.loc['A', 'B']
    assert df2.loc['B', 'A'] == df4.loc['B', 'A']
    assert df2.loc['B', 'B'] == df4.loc['B', 'B']
    
def test_compute_clustering(read_file_G):
    '''
    Since with compute_clustering we are going to evaluate the clustering value for each node
    according to a graph, that in our simple case is read_file_G, and since in our case we would
    expect 10 nodes that have a label from 0 to 9 in strings, we assert that the nodes belonging 
    to the graph have these labelled nodes and in addition we assert that the total number of nodes
    and of clustering values is 10.
  
    :param read_file_G: graph used as an example and defined in conftest.py.
    '''
    nodes, clustering = compute_clustering(read_file_G)
    
    assert set(nodes) == set([str(i) for i in range(10)])
    for value in clustering: 
        assert value >= 0
    assert len(nodes) == 10
    assert len(clustering) == 10
    
def test_col_retweet_network(dataframe_retweet, dataframe_retweet2):
    '''
    In this case we test that the output of col_retweet_network has 3 rows and that the expected
    output, given by dataframe_retweet2, is actually equal to df.
    
    :param dataframe_retweet: dataframe defined in conftest.py
    :param dataframe_retweet2: dataframe defined in conftest.py
    '''
    df = col_retweet_network(dataframe_retweet)
    assert len(df) == 3
    assert df.equals(dataframe_retweet2)
    
def test_get_daily_components(read_file_G_days,read_com_of_user_days):
    '''
    This function tests that the lenght of the lists containing the nodes belonging to the first
    and second strong e weak components are all equal to three. then we test that the output of the
    get_daily_components function actually corresponds to the expected values.
    
    :param read_file_G_days: list contaning 3 networks, each one for a different day, defined in
    conftest.py as an example to test the function.
    :param read_com_of_user_days: variable into which we store com_of_user that is a dictionary having
    the Id user as key and the community as value for each day.
    '''
    nodes_group_A_G0, nodes_group_B_G0, nodes_group_A_G1, nodes_group_B_G1, nodes_group_A_G0_weak, nodes_group_B_G0_weak, nodes_group_A_G1_weak, nodes_group_B_G1_weak = get_daily_components(read_file_G_days,read_com_of_user_days)
    
    assert (len(nodes_group_A_G0)==3)&(len(nodes_group_B_G0)==3)&(len(nodes_group_A_G1)==3)&(len(nodes_group_B_G1)==3)&(len(nodes_group_A_G0_weak)==3)&(len(nodes_group_B_G0_weak)==3)&(len(nodes_group_A_G1_weak)==3)&(len(nodes_group_B_G1_weak)==3)
    assert set(nodes_group_A_G0) == {0,1}
    assert set(nodes_group_B_G0) == {0,1}
    assert set(nodes_group_A_G1) == {0}
    assert set(nodes_group_B_G1) == {1}
    assert set(nodes_group_A_G0_weak) == {84, 268, 214}
    assert set(nodes_group_B_G0_weak) == {282, 213, 79}
    assert set(nodes_group_A_G1_weak) == {32, 26, 19}
    assert set(nodes_group_B_G1_weak) == {15, 38, 31}
    
def test_get_daily_modularity(read_file_G_days,read_com_of_user_days):
    '''
    This function tests that the lenght of the lists that are output of the get_daily_modularity
    function is actually equal to three and that the output values correspond to the expected values.
    
    :param read_file_G_days: list contaning 3 networks, each one for a different day, defined in
    conftest.py as an example to test the function.
    :param read_com_of_user_days: variable into which we store com_of_user that is a dictionary having
    the Id user as key and the community as value for each day.
    '''
    mod_unweighted_file, mod_weighted_file, random_mod_unweighted_file, random_mod_weighted_file, nodes_group_A, nodes_group_B = get_daily_modularity(read_file_G_days,read_com_of_user_days)
        
    assert len(mod_unweighted_file) == 3
    assert len(mod_weighted_file) == 3
    assert len(random_mod_unweighted_file) == 3
    assert len(random_mod_weighted_file) == 3
    assert len(nodes_group_A) == 3
    assert len(nodes_group_B) == 3
    assert set(mod_unweighted_file) == {0.004, -0.019, 0.019}
    assert set(mod_weighted_file) == {0.008, -0.02, 0.018}
    assert {len(random_mod_unweighted_file[0]),len(random_mod_unweighted_file[1]), len(random_mod_unweighted_file[2])} == {10}
    assert {len(random_mod_weighted_file[0]),len(random_mod_weighted_file[1]), len(random_mod_weighted_file[2])} == {10}
    assert set(nodes_group_A) == {480, 314, 403}
    assert set(nodes_group_B) == {419, 332, 487}
    
def test_get_daily_assortativity(read_file_G_days):
    '''
    This function tests that the list output of the get_daily_assortativity function has lenght equal
    to three and that the output values correspond to the expected values.
    
    :param read_file_G_days: list contaning 3 networks, each one for a different day, defined in
    conftest.py as an example to test the function.
    '''
    assortativity_values = get_daily_assortativity(read_file_G_days)
    assert len(assortativity_values) == 3
    assert set(assortativity_values) == {-0.19, -0.15, -0.17}
    
def test_daily_Gini_in_out(read_file_G_days):
    '''
    This function tests that the lists output of the get_daily_Gini_in_out function have lenght equal
    to three and that the output values correspond to the expected values.
    
    :param read_file_G_days: list contaning 3 networks, each one for a different day, defined in
    conftest.py as an example to test the function.
    '''
    Gini_in_values, Gini_out_values = get_daily_Gini_in_out(read_file_G_days)
    assert len(Gini_in_values) == 3
    assert len(Gini_out_values) == 3
    assert set(Gini_in_values) == {0.842, 0.931, 0.937}
    assert set(Gini_out_values) == {0.422, 0.304, 0.294}
    
def test_get_daily_nodes(read_file_G_days):
    '''
    This function tests that the list output of the get_daily_nodes function has lenght equal
    to three and that the output values correspond to the expected values.
    
    :param read_file_G_days: list contaning 3 networks, each one for a different day, defined in
    conftest.py as an example to test the function.
    '''
    nodes_original = get_daily_nodes(read_file_G_days)
    assert len(nodes_original) == 3
    assert set(nodes_original) == {646, 822, 967}
    
def test_words_frequency(dataframe_retweet):
    '''
    This function tests that the two lists output of the function words_frequency have lenght 3, and
    then we test that the expected values and strings correspond to the actual output of the function.
    
    :dataframe_retweet: dataframe defined in conftest.py
    '''
    values_list, key_list = words_frequency(dataframe_retweet, [2])
    assert len(key_list) == 3
    assert len(values_list) == 3
    assert set(key_list) == {'table', 'cat', 'house'}
    assert set(values_list) == {3,2,1}
    
def test_degree_distributions(read_file_G):
    '''
    This function tests that the lenght of the lists that are output of the degree_distributions 
    function is equal to 10 and then we test that the expected values correspond to the output of
    the function.
    
    :read_file_G: variable into which we store the network used for testing in networkX format.
    '''
    in_degree, out_degree = degree_distributions(read_file_G)
    assert len(in_degree) == 10
    assert len(out_degree) == 10
    assert set(in_degree) == {1, 2, 3}
    assert set(out_degree) == {1, 2, 3, 4}
    
def test_create_date_store(define_path_day):
    '''
    This function tests that the lenght of the two lists output of create_date_store have lenght 3,
    that the expected values for the first list correspond to the right dates that are output of the
    function, and that the elements of the second list are graphs.
    
    :define_path_day: path to follow in order to test this function.
    '''
    date_store, Gvac_days = create_date_store(define_path_day)
    
    assert len(date_store) == 3
    assert len(Gvac_days) == 3
    assert set(date_store) == {'02', '03', '2-01'}
    assert type(Gvac_days[0]) == nx.DiGraph
    assert type(Gvac_days[1]) == nx.DiGraph
    assert type(Gvac_days[2]) == nx.DiGraph
    
def test_age_of_activity(read_file_G_days):
    '''
    This function tests that the lenght of the output of the age_of_activity function is equal to 3.
    Then we test that the expected values correspond to the output, in particular we are calculating
    the number of active nodes in a certain day by selecting only the key of the dictionary.
    
    :param read_file_G_days: list contaning 3 networks, each one for a different day, defined in
    conftest.py as an example to test the function.
    '''
    nodes_age_in, nodes_age_out = age_of_activity(read_file_G_days)
    assert len(nodes_age_in) == 3
    assert len(nodes_age_out) == 3
    assert len(nodes_age_in[0].keys()) == 211 
    assert len(nodes_age_in[1].keys()) == 348
    assert len(nodes_age_in[2].keys()) == 516
    assert len(nodes_age_out[0].keys()) == 440
    assert len(nodes_age_in[1].keys()) == 348
    assert len(nodes_age_in[2].keys()) == 516
    
def test_n_tweets_over_time(dataframe_retweet):
    '''
    This function tests that the column of the dataframe (output of the function n_tweets_over_time) 
    is only one and has also a specific label and then we test the relevant index that is represented
    by the date. then we test the expected values during the different days.
    
    :dataframe_retweet: dataframe defined in conftest.py
    '''
    df_tweets = n_tweets_over_time(dataframe_retweet)
    assert set(df_tweets.columns) == {'Ntweets'}
    assert set(df_tweets.index) == {pd.to_datetime('2022-06-10'), pd.to_datetime('2022-03-02'), pd.to_datetime('2021-10-28')}
    assert set(df_tweets['Ntweets'].to_list()) == {1,2,3}
    
def test_n_tweets_over_time_selected_community(dataframe_retweet, df_top):
    '''
    This function tests the same things that have been tested in the previous function.
    
    :dataframe_retweet: dataframe defined in conftest.py
    :df_top: dataframe defined in conftest.py
    '''
    df_tweets = n_tweets_over_time_selected_community(dataframe_retweet.rename(columns = {'user.id':'user'}), df_top, 'A')
    assert set(df_tweets.columns) == {'A'}
    assert set(df_tweets.index) == {pd.to_datetime('2022-06-10'), pd.to_datetime('2022-03-02')}
    assert set(df_tweets['A'].to_list()) == {2,3}
    
def test_read_cleaned_war_data(define_path_war):
    '''
    This function tests the right lenght of the dataframe output of the function and then it tests
    that the label of each column is the correct one.
    
    :define_path_war: path to follow in order to test this function.
    '''
    df = read_cleaned_war_data(define_path_war)
    assert len(df) == 354059
    assert set(df.columns) == {'created_at', 'created_at_days', 'id', 'lang', 'retweeted_status.id',
'retweeted_status.user.id', 'retweeted_text', 'text', 'url', 'url_domain', 'user', 'user.screen_name'}
    
def test_sort_data(read_file_G,read_betweenness):
    '''
    This function tests that the lenght of the lists output of the sort_data function is 10 as 
    expected.
    
    :read_file_G: variable into which we store the network used for testing in networkX format.
    :read_betweenness: list containing the betweenness values.
    '''
    nodes, in_degreeG0, out_degreeG0, betweenessG0 = sort_data(read_file_G,read_betweenness)
    assert len(nodes) == 10
    assert len(in_degreeG0) == 10
    assert len(out_degreeG0) == 10
    assert len(betweenessG0) == 10
    
def test_compute_betweeness(read_file_components_weak):
    '''
    This function tests that the lenght of all the lists that are output of the compute_betweeness
    function is 550 as expected.
    
    :read_file_components_weak: graph in networkX format representing the weak connected components
    of a network taken as example.
    '''
    betweenness, betweenness_weak, in_degree_G0, out_degree_G0, in_degree_G0_weak, out_degree_G0_weak = compute_betweeness(read_file_components_weak, read_file_components_weak)
   
    assert len(betweenness) == 550
    assert len(betweenness_weak) == 550
    assert len(in_degree_G0) == 550
    assert len(out_degree_G0) == 550
    assert len(in_degree_G0_weak) == 550
    assert len(out_degree_G0_weak) == 550
    