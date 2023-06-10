# ReadMe
The goal of the project is to develop a community detection method in a social network analysis. In our case we focus on Twitter network, in which the interactions between units - Twitter users - aim at finding different individuals in that network with similar interests. The challenge is therefore to detect social interactions between individuals with comparable considerations and desires from a large social network.

The social networks taken into consideration are a network built from the retweets on Twitter about the italian Covid vaccine and a second network built from a random sample of about 10000 Twitter users, that were present also in the vaccine network, without considering a specific topic this time. We would expect these users to have expressed their opinion also about the war in Ukraine. Therefore, the goal of the project is to compare these two networks by using the same communities identified in the vaccine network: we want to establish if users belonging to the same communities within the two networks share a similar opinion about the two topics taken into account or not.

These repository contains all the useful information to generate and work with data.

The steps to follow in order to obtain the outputs are:
-First of all open a terminal, 
-go in Software-Computing directory (cd Software-Computing),
-go in Functions directory (cd Functions/),
-run open_variables.py (python open_variables.py),
-run functions.py (python functions.py),
-run DataForVaccineGraph.py (python DataForVaccineGraph.py),
-run DataForWarNetwork.py (python DataForWarNetwork.py),
-run analysis_over_time.py (python analysis_over_time.py),
-run conftest.py (python conftest.py), test the good working of the previous codes with test_functions.py (pytest test_functions.py).
-Eventually obtain the plots by entering in Plot_Graph.ipynb notebook and running each cell.
 

As explained below you will find all the Plots in 'Plots' repository and all the data generated in 'Data' repository.
Folder subdivision is organized as follows:
 
- Functions: each one import the proper libraries to be used (we avoid to discuss each one).
             Here in 'Functions' I decided to put another repository ('data') where we store
             the edgelists used by the several codes. Being them not the real output of the
             project we preferred to put them in Functions repository.

 1) open_variables.py:
            here we load the network built from the retweets on Italian "vaccine" tweets since October 2020. The variable Gvac is a NetworkX DiGraph object (directed graph), having users as nodes and the edge weights are cumulative numbers of retweets over the full time span. Edges in direction x->y means that user x retweeted user y a number of times equal to the weight. So the in-degree of a user is the total number of retweets it received in the time span. Then the communities were calculated with the leading eigenvector methods of python-igraph (https://igraph.org/c/doc/igraph-Community.html#igraph_community_leading_eigenvector), asking for 2 communities. The variable com_of_user is a dictionary were each key is a user and the value is the community it was assigned to. For example, com_of_user[\<node name\>] will be 'A' or 'B'.
            For about 14K users which were active on Twitter during February 2022, we downloaded up to 3000 tweets and retweets written from 1/2/2022 to 10/3/2022. We generate a dataframe that contains this data: each row is a tweet with different information. Also the original tweets they retweeted were included as rows of the DataFrame.Then we build a retweet network, and to do that the relevant columns are user.id and retweeted_status.user.id. In fact, the first one is the retweeting user (x) and the second one the retweeted user (y) of the edge x->y.
            
2) functions.py:
           -it implements 7 functions that will be used in DataForVaccineGraph.py,
            DataForWarNetwork.py, analysis_over_time.py and test_functions.py. The functions are:
             1) assign_communities(G, com_of_user)
            taking in input a network in networkX format (G) and a dictionary having the 
            Id user as key and the community as value (com_of_user), this functions returns a subgraph of G in networkX format containing the nodes belonging to group A and group B with a degree>0 (Gvac_subgraph), a subgraph of G in networkX format containing the nodes belonging to group A (Gvac_A), a subgraph of G in networkX format containing the nodes belonging to group B (Gvac_B), a list of strings containing the Id users of group A (group_A) and a list of strings containing the Id users of group B (group_B).
            2) mixing_matrix(G, com_of_user)
            taking in input a network in networkX format (G) and a dictionary having the 
            Id user as key and the community as value (com_of_user), this function returns a dataframe representing the 2x2 mixing matrix in the unweighted case and a dataframe representing the 2x2 mixing matrix in the weighted case.
            3) randomize_network(n_swaps, Gvac, group_A, group_B)
            taking in input the integer number representing the number of swaps (n_swaps), a network in networkX format (Gvac), a list of strings containing the Id users of group A (group_A) and  a list of strings containing the Id users of group B (group_B), this function returns a float value representing the modularity in the unweighted case and a float value representing the modularity in the weighted case.
            4) compute_randomized_modularity(Gvac_subgraph, group_A, group_B)
            taking in input the subgraph of G in networkX format containing the nodes belonging to group A and group B with a degree>0 (Gvac_subgraph), a list of strings containing the Id users of group A (group_A) and a list of strings containing the Id users of group B (group_B) this function returns a list containing the modularity values evaluated after a certain number of randomization, in this case 10, in the unweighted case and  a list containing the modularity values evaluated after a certain number of randomization, in this case 10, in the weighted case.
            5) compute_connected_component(Gvac_subgraph, group_A, group_B)
            taking in input the subgraph of G in networkX format containing the nodes belonging to group A and group B with a degree>0 (Gvac_subgraph), a list of strings containing the Id users of group A (group_A) and a list of strings containing the Id users of group B (group_B) this function returns a list of strings containing the Id users of group A belonging to the first strongly connected component (group_A_G0), a list of strings containing the Id users of group B belonging to the first strongly connected  component (group_B_G0), a list of strings containing the Id users of group A belonging to the second strongly connected component (group_A_G1), a list of strings containing 
            the Id users of group B belonging to the second strongly connected component
            (group_B_G1), a generator of sets of nodes, one for each strongly connected component of Gvac_subgraph, in this case the first one (G0) and a generator of sets of nodes, one for each strongly connected component of Gvac_subgraph, in this case the first one (G1).
            6) compute_weak_connected_component(Gvac_subgraph, group_A, group_B)
            taking in input the subgraph of G in networkX format containing the nodes belonging to group A and group B with a degree>0 (Gvac_subgraph), a list of strings containing the Id users of group A (group_A) and a list of strings containing the Id users of group B (group_B) this function returns a list of strings containing the Id users of group A belonging to the first weakly connected component (group_A_G0), a list of strings containing the Id users of group B belonging to the first weakly connected component (group_B_G0), a list of strings containing the Id users of group A belonging to the second weakly connected component (group_A_G1), a list of strings containing the Id users of group B belonging to the second weakly connected component (group_B_G1), a generator of sets of nodes, one for each weakly connected component of 
            Gvac_subgraph, in this case the first one (G0) and a generator of sets of nodes, one for each weakly connected component of Gvac_subgraph, in this case the first one (G1).
            7) gini(x, w=None)
            taking in input a array or list on which I want to calculate the Gini index and array or list representig the weight of values, this function returns a float value representing the Gini index.
            
3) DataForVaccineGraph.py:
           first of all it reads the network built from the retweets on Italian "vaccine" tweets since October 2020. The variable is a   NetworkX DiGraph object (directed graph), having users as nodes and the edge weights are cumulative numbers of retweets over the full time span. Edges in direction x->y means that user x retweeted user y a number of times equal to the weight. So the in-degree of a user is the total number of retweets it received in the time span. Then it saves the files for each table: a table representing the number of links from A to A, from A to B, from B to A and from B to B (mixing matrix), a table obtaining by dividing each entry of the previous table by the total number of links taken
           into account; the 3rd dataframe is obtained in the following way: we take into account the total number of links starting from community A, i.e. the sum of the elements of the first row of the mixing matrix, and we divide each element of the first row by this number. Then we repeat the procedure for the second row. In this way we get the average behaviour if a link starts from community A or from community B. The same is repeated in the weighted case. Moreover it counts the number of nodes belonging to the communities in the original vaccine network and, eventually, it creates 6 lists to store the in- out-degree of the nodes belonging to the whole network, group A and group B and we save them in a corresponding file.
            
4) DataForWarNetwork.py:
            First of all it loads the second network built from a random sample of Twitter users, that were present also in the vaccine network, without considering a specific topic this time, assuming that these users have expressed their opinion also about the war in Ukraine. Then we repeat the same steps that we performed in DataForVaccineGraph.py but for the new network. In addition, we get the nodes of group A or group B belonging to the first and second strong (and weak) connected components, we obtain and we store the betweeness data together with the in-degree and out-degree data by sorting them in the same order. We get a table with the corresponding spearman coefficients, we evaluate and store the clustering coefficient data. Eventually, we evaluate the frequency of the
            mostly used words within the two groups in order to understand if, as in the vaccine network where group A is pro vaccine and group B is against vaccine, also in the war network the users belonging to group A and group B share a similar opinion about a different topic, for example we would expect that users belonging to group A are pro Ukraine and users belonging to group B are pro Russia.
            
5) analysis_over_time.py:
            after having read the data of the "war network" by considering the same communities of the "vaccine network", the goal of this part is to analyze the main properties of our network day by day from 01-02-2022 to 11-03-2022; in particular we compare day by day the modularity of our real network with the modularity of a random network obtained with a shuffling of the edges (configuration model), in order to demonstrate that our network is actually clustered. Then we evaluate the assortativity coefficient, that is often used as a correlation measure between nodes, the Gini index, that is a measure of statistical dispersion used to represent the inequality within a social group, the average age of activity and we identify the first two strongly and weakly connected components in order to find out if the giant component is made up of users belonging to a single community or not. The last thing is to evaluate the behaviour of the retweets for the top-scoring nodes, by considering the top users of group A and B on the basis of the total-degree. The goal is to establish if there is a change in percentage with respect to the total number of retweets. Obviously we create a set of dataframes and we save them in order to perform the plots in Plot_Graph.ipynb.
            
6) conftest.py:
           here we use Pytest fixtures, i.e. functions that can be used to manage our apps states and dependencies. In particular they can provide data for testing and a wide range of value types when explicitly called by our testing software. They will be used in test_functions.py. We define two functions: ReadFileG(), ReadFileComOfUser(): in the first one we read the network built from the retweets on Italian "vaccine" tweets since October 2020. The variable G, output of the functions ReadFileG(), is a NetworkX DiGraph object (directed graph), having users as nodes and the edge weights are cumulative numbers of retweets over the full time span. Instead in the second function,  the variable com_of_user, output of the function ReadFileComOfUser(), is a dictionary where each key is a user and the value is the community it was assigned to.
            
7) test_functions.py:
            It implements 7 functions that are used to test the proper working of the functions implemented in functions.py:
            1) test_assign_communities(ReadFileG, ReadFileComOfUser):
            by taking in input a variable into which we store the network G_vac in networkX format (ReadFileG) and a variable into which we store com_of_user that is a dictionary having the Id user as key and the community as value (ReadFileComOfUser), this function tests if the output of assign_communities() function are of the correct type: group_A and group_B must be lists, Gvac_subgraph, Gvac_A and Gvac_B must be direct graphs. Then it tests that the set list of nodes belonging to Gvac_A (i.e. the subgraph containing the nodes belonging to group A) actually corresponds to group_A and that the set list of nodes belonging to Gvac_B (i.e. the subgraph containing the nodes belonging to group B) actually corresponds to group_B. At the end it tests that the set list of nodes belonging toGvac_subgraph is given by the set list of nodes belonging to Gvac_A or Gvac_B.
            2) test_mixing_matrix(): 
            this function is used to test if the 2 columns that we get in the 
            mixing matrix for the unweighted and weighted cases are actually the number of edges coming respectively from users of group A and users of group B. In addition we test if the sum of all the elements of the 2 matrices corresponds to the overall number of edges of our graph, in both the cases unweighted and weighted.
            3) test_randomize_network(): 
            this function is used to test that, regardless of the shuffling, the number of edges connecting the nodes in the original graph Gvac_subgraph is actually equal to the overall number of edges connecting the nodes in the shuffled network, Gvac_shuffle. In addition we test if modularity_unweighted and modularity_weighted, outputs of the corresponding function, are float type or not.
            4) test_compute_randomized_modularity(): this function is used to test the type of the output of compute_randomized_modularity() function: list_modularity_unweighted, list_modularity_weighted must be lists. Then, since in our case we perform 10 randomizations, we test if the lenght of the two lists is equal to 10.
            5) test_compute_connected_component():
            this function is used to test the type of the output of compute_connected_component() function: group_A_G0, group_B_G0, group_A_G1, group_B_G1 must be lists, G0 and G1 must be direct graphs. Then the functions tests that the nodes of the graph G0 (representing the first strongly connected component) actually correspond to the set list of nodes belonging to group_A_G0 or group_B_G0. Similarly it tests if the nodes of the graph G1 (representing the second strongly connected component) actually correspond to the set list of nodes belonging to group_A_G1 or group_B_G1.
            6) test_compute_weak_connected_component():
            this function is used to test the type of the output of compute_connected_component() function: group_A_G0_weak,  group_B_G0_weak,  group_A_G1_weak, group_B_G1_weak must be lists, G0_weak and G1_weak must be direct graphs. Then the functions tests if the nodes of the graph G0_weak (representing the first weakly connected component) actually correspond to the set list of nodes belonging to group_A_G0_weak or group_B_G0_weak. Similarly it tests if the nodes of the graph G1_weak (representing the second weakly connected component) actually correspond to the set list of nodes belonging to group_A_G1_weak or group_B_G1_weak. In addition it tests if the first strongly connected component is smaller than the first weakly connected component, as it should be.
            7) test_gini():
            this function firstly tests if the type of the Gini index value given in
            output from the corresponding function is a float type (in both the cases Gini_in, Gini_out) and than, as a further confirmation, if we assume a pure equidistribution, for example with all values of the x array in input equal to 1, we test if we will obtain a gini index equal to 0, as it should be by definition.
            
8) Plot_Graph.ipynb:
            it takes in input all the data from 'Data' and gives in output the plots described in the following (see 'Plots').
            
            
- Data: here we store all the data that we obtain in output from DataForVaccineGraph.py, 
        DataForWarNetwork.py and analysis_over_time.py in order to obtain the plots from the notebook Plot_Graph.ipynb.


-Age : Figure14.csv.           : dataframe where we store the dates (date_store) and the age of the users in the case of the in and out distributions. The meaning of age is explained in analysis_over_time.py.
           
-Assortativity : Figure10.csv : here we store the assortativity values for each date (date_store).
           
-Betweeness: here we store the betweeness values.
    -PanelA.csv : betweeness values and corresponding in degree values in the case of 
    the 1st strongly connected components.
    -PanelB.csv : betweeness values and corresponding out degree values in the case of 
the 1st strongly connected components.
-PanelC.csv : betweeness values and corresponding in degree values in the case of the 1st weakly connected components.
-PanelD.csv : betweeness values and corresponding out degree values in the case of the 1st weakly connected components.
                                            
-ClusteringDistribution: ClusteringDistribution.csv: here we store the values of clustering for each node.
           
-DegreeDistributionVaccine: here we store in- and out-degree distribution values for the whole vaccine network, groupA and groupB of the vaccine network.
-DegreeGroupA.csv: here we store in- and out-degree distribution values for groupA of the vaccine network.
-DegreeGroupB.csv: here we store in- and out-degree distribution values for groupB of the vaccine network.
-DegreeOriginal.csv: here we store in- and out-degree distribution values for the whole vaccine network.
           
-DegreeDistributionWar: here we store in- and out-degree distribution values for the whole war network, groupA and groupB of the war network.
-DegreeGroupA.csv: here we store in- and out-degree distribution values for groupA of the war network.
-DegreeGroupB.csv: here we store in- and out-degree distribution values for groupB of the war network.
-DegreeOriginal.csv: here we store in- and out-degree distribution values for the whole war network.
           
-Frequency                     :
-Figure15_1.csv : table representing the key list and the corresponding value list (the word appearing and the corresponding number of times it appeared) in the case of group A
for the war network.
                  
-Figure15_2.csv : table representing the key list and the corresponding value list (the word appearing and the corresponding number of times it appeared) in the case of group B for the war network.
                  
-Fraction.csv : the total number of tweets, the number of tweets in group A and in group B, fraction of retweets in group  A and group B with respect to the total number of retweets.
                  
-topUsersGroupA.csv: here we store the top users of group A (20) on the basis of the total-degree, in fact we save the users, the community to which he belongs, in-degree, out-degree and total degree.
                  
-topUsersGroupB.csv: here we store the top users of group B (20) on the basis of the total-degree, in fact we save the users, the community to which he belongs, in-degree, out-degree and total degree.
                  
-totalNofRetweets.csv   : table with the total number of retweets for each user.
                  
-Gini_Index:  Figure11.csv  : here we store the dates (date_store), the gini index values in the case of the in- and out-degree distributions.
           
-InOutDegreeVsClustering:
-ClusteringInDegree.csv  : here we store in-degree values and corresponding clustering coefficient values.
-ClusteringOutDegree.csv : here we store out-degree values and corresponding clustering coefficient values.
                                          
           
-MixingMatrixTables:
-MixingVaccine1.csv: Table representing the number of links from A to A, from A to B, from B to A and from B to B (mixing matrix).
                                     
-MixingVaccine2.csv: each entry of the previous table is divided by the total number of links taken into account.
                                     
-MixingVaccine3.csv: Table obtained by taking into account the total number of links starting from community A, i.e. the sum of the elements of the first row of the mixing matrix, and we divide each element of the first row by this number. Then we repeat the procedure for the second row. In this way we get the average behaviour if a link starts from community A or from community B.

-MixingVaccine4.csv: Mixing matrix in weighted case.

-MixingVaccine5.csv: each entry of the previous table is divided by the total number of links taken into account.

-MixingVaccine6.csv: Table obtained by taking into account the total number of links starting from community A, i.e. the sum of the elements of the first row of the mixing matrix, and we divide each element of the first row by this number. Then we repeat the procedure for the second row, thus as before but in the weighted case. The following tables are the same tables described above but in the case of the war network:
-MixingWar1.csv    
-MixingWar2.csv    
-MixingWar3.csv    
-MixingWar4.csv    
-MixingWar5.csv    
-MixingWar6.csv    
           
-Modularity:  Figure13.csv:  here we store the dates (date_store), the modularity of the real network and the modularity of the radomized one.
           
-NumberOfNodesAB:  Figure12.csv: we store the dates (date_store), the nodes belonging to a single community of the war network, where the communities are taken from the vaccine 
network (nodes_group_A, nodes_group_B) and all the nodes belonging to our war network (nodes_original).
           
-StronglyConnectedComponents   :
Figure8.csv: here we store the dates (date_store), the nodes of group A belonging to the first strongly connected component (nodes_group_A_G0), the nodes of group B belonging to 
the first strongly connected component (nodes_group_B_G0),the nodes of group A belonging to 
the second strongly connected component (nodes_group_A_G1), the nodes of group B belonging to the secondstrongly connected component (nodes_group_B_G1).
           
-TableSpearman:   Spearman.csv : Table with the Spearman coefficient values.
           
-WeaklyConnectedComponents     : 
Figure9.csv:  here we store the dates (date_store), the nodes of group A belonging to the first weakly connected component (nodes_group_A_G0_weak), the nodes of group B belonging to the first weakly connected component (nodes_group_B_G0_weak), the nodes of group A belonging
to the second weakly connected component (nodes_group_A_G1_weak), the nodes of group B
belonging to the second weakly connected component (nodes_group_B_G1_weak).
           
-WarTweets.pkl                 : file with all the retweets

- Plots: 
-Figure1.png  : in- out-degree distribution for Vaccine Network.
-Figure2.png  : in- out-degree distribution for War Network.
-Figure3.png  : in- or -out degree as a function of the betweeness centrality.
-Figure4.png  : plot of the clustering coefficient distribution.
-Figure5.png  : plot of the in- and out-degree as a function of the clustering coefficient.
-Figure6.png  : plots of dimension of the first and second strongly connected components (1st), fraction of nodes belonging to group A (2nd) group B (3rd), given the dimension of the first and second strongly connected components.
-Figure7.png  : plots of dimension of the first and second weakly connected components
(1st), fraction of nodes belonging to group A (2nd) group B (3rd), given the dimension of the first and second weakly connected components.
-Figure8.png  : plot of the assortativity coefficient day by day compared with the assortativity value for the static network.
-Figure9.png  : plot of the Gini index day by day for the in- and out-degree distributions compared with the corresponding values for the static network. 
-Figure10.png : plot of the number of nodes belonging to war network day by day (1st) and 
the fraction of nodes belonging to group A or to group B day by day, that are present in the war network, but that belong also to the vaccine network.
-Figure11.png : plot representing a comparison of modularity day by day between the real network and the shuffled one.
-Figure12.png : plot representing the average age of activity day-by-day considering 
separetely the in- and out-distributions, in other terms it is the plot of the average age of activity of users who receive a retweet (in-degree) and of users who retweets (out-degree).
-Figure13.png : plot of the the behaviour of the frequency of the mostly 10 words used in group A and group B.
-Figure14.png : plot of the total number of retweets over the period considered.
           
    -Fitting:
    -Figure1aInDegreeGroupA.png: fit of in-degree distribution for group A of the vaccine network by using a series of well-known functions.
    -Figure1bOutDegreeGroupA.png: fit of out-degree distribution for group A of the vaccine network by using a series of well-known functions.
    -Figure1cInDegreeGroupB.png: fit of in-degree distribution for group B of the vaccine network by using a series of well-known functions.
    -Figure1dOutDegreeGroupB.png: fit of out-degree distribution for group B of the vaccine network by using a series of well-known functions.
    -Figure1eInDegreeOriginal.png   : fit of in-degree distribution for the whole vaccine network by using a series of well-known functions.
    -Figure1fOutDegreeOriginal.png  : fit of out-degree distribution for the whole vaccine network by using a series of well-known functions.
    -Figure2aInDegreeGroupAWar.png  : fit of in-degree distribution for group A of the war network by using a series of well-known functions.
    -Figure2bOutDegreeGroupAWar.png : fit of out-degree distribution for group A of the war network by using a series of well-known functions.
    -Figure2cInDegreeGroupBWar.png  : fit of in-degree distribution for group B of the war network by using a series of well-known functions.
    -Figure2dOutDegreeGroupBWar.png : fit of out-degree distribution for group B of the war network by using a series of well-known functions.
    -Figure2eInDegreeWarGraph.png   : fit of in-degree distribution for the whole war network by using a series of well-known functions.
    -Figure2fOutDegreeWarGraph.png  : fit of out-degree distribution for the whole war network by using a series of well-known functions.