# ReadMe
The goal of the project is to develop a community detection method in a social network analysis. In our case we focus on Twitter network, in which the interactions between units - Twitter users - aim at finding different individuals in that network with similar interests. The challenge is therefore to detect social interactions between individuals with comparable considerations and desires from a large social network.

The social networks taken into consideration are a network built from the retweets on Twitter about the italian Covid vaccine and a second network built from a random sample of about 10000 Twitter users, that were present also in the vaccine network, without considering a specific topic this time. We would expect these users to have expressed their opinion also about the war in Ukraine. Therefore, the goal of the project is to compare these two networks by using the same communities identified in the vaccine network: we want to establish if users belonging to the same communities within the two networks share a similar opinion about the two topics taken into account or not.

## Features
The steps to follow in order to obtain the outputs are:
- First of all open a terminal, 
- go in github_try directory and then in Software-Computing directory:
```sh
cd github_try
cd Software-Computing
```
Here you can find a repository called 'data' containing all the data, including either those used to run the python codes or the ones generated as output of the project; we can also find a repository called 'tutorial' including all the plots and the jupyter notebook used to get the plots, being them the real output of our project. Eventually we can find an other repository called 'src' including the python codes used for our project and testing. Thus:
- go in src directory:
```sh
cd src
```
In src directory we can find other two directories 'functions' and 'tests': in 'functions' are stored all the codes used to get the outputs of our project, therefore after entering in 'functions', the codes must be run in the following order:
```sh
cd functions
python configurations.py
python functions.py
python open_variables.py
python data_vaccine_graph.py
python data_war_network.py
python analysis_over_time.py
```
| Code name | Explanation |
| ------ | ------ |
| configurations.py | It contains all the parameters, preferences, and alternative options that allow the selection of various features and settings. |
| functions.py | Here we store all the functions used to run the following codes. |
| open_variables.py | Here we load the network built from the retweets on Italian "vaccine" tweets since October 2020. The variable Gvac is a NetworkX DiGraph object (directed graph), having users as nodes and the edge weights are cumulative numbers of retweets over the full time span. Edges in direction x->y means that user x retweeted user y a number of times equal to the weight. Then the communities were calculated with the leading eigenvector methods of python-igraph asking for 2 communities. For about 14K users which were active on Twitter during February 2022, after having downloaded up to 3000 tweets and retweets written from 1/2/2022 to 10/3/2022 we built a retweet network, and to do that the relevant columns are user.id and retweeted_status.user.id.  |
| data_vaccine_graph.py |  We read the network built from the retweets on Italian "vaccine" tweets since October 2020. Then we compute the mixing matrix, a table representing the number of links from A to A, from A to B, from B to A and from B to B, and some manipulations of it. This is done in the unweighted and weighted cases. Moreover we study the in-degree and out-degree distributions of our vaccine network. |
|data_war_network.py | We load the second network built from a random sample of Twitter users, that were present also in the vaccine network, without considering a specific topic this time, assuming that these users have expressed their opinion also about the war in Ukraine. Then we repeat the same steps that we performed in DataForVaccineGraph.py but for the new network. In addition, we get the nodes of group A or group B belonging to the first and second strong (and weak) connected components, we study the betweeness centrality as a function of the in-degree and out-degree. We get a table with the corresponding spearman coefficients, we evaluate the clustering coefficient data. Eventually, we evaluate the frequency of the mostly used words within the two groups in order to understand if, as in the vaccine network where group A is pro vaccine and group B is against vaccine, also in the war network the users belonging to group A and group B share a similar opinion about a different topic, for example we would expect that users belonging to group A are pro Ukraine and users belonging to group B are pro Russia. |
|analysis_over_time.py | The goal of this part is to analyze the main properties of our network day by day from 01-02-2022 to 11-03-2022; in particular we compare day by day the modularity of our real network with the modularity of a random network obtained with a shuffling of the edges (configuration model), in order to demonstrate that our network is actually clustered. Then we evaluate the assortativity coefficient, the Gini index, the average age of activity and we identify the first two strongly and weakly connected components in order to find out if the giant component is made up of users belonging to a single community or not. The last thing is to evaluate the behaviour of the retweets for the top-scoring nodes, by considering the top users of group A and B on the basis of the total-degree. The goal is to establish if there is a change in percentage with respect to the total number of retweets. |
 At this point, from src directory we go in 'tests' directory and we run firstly 'conftest.py' and then 'test_functions.py':
 ```sh
cd tests
pytest conftest.py
pytest test_functions.py
```
| Code name | Explanation |
| ------ | ------ |
| conftest.py | It contains auxiliary functions thanks to the use of pytest fixtures, thus in this way  instead of running the same code for every test, we can attach fixture function to the tests and it will run and return the data to the test before executing each test. |
| test_functions.py | It contains all the functions used to test the functions contained in functions.py. |
Now, from Software-Computing directory we can check that all the data have been correctly generated in 'data' directory, and by entering in 'tutorial' directory we can run the jupyter notebook plot_graph.ipynb. The plots will be saved in 'plots' directory that is included in 'tutorial'.