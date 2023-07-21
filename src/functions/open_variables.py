import pandas as pd
import networkx as nx
import pickle
from configurations import (
    STOR_DIR,
    PATH_VACCINE,
    PATH_COM_OF_USER,
    PATH_UKRAINE,
    min_rt
)


def main():
    '''
    Vaccine network
    Load the network built from the retweets on Italian "vaccine" tweets since October 2020. 
    The variable is a NetworkX DiGraph object (directed graph), having users as nodes and the 
    edge weights are cumulative numbers of retweets over the full time span. Edges in direction
    x->y means that user x retweeted user y a number of times equal to the weight. So the i
    n-degree of a user is the total number of retweets it received in the time span.
    '''
    Gvac=nx.read_weighted_edgelist(STOR_DIR+PATH_VACCINE,
                                   delimiter='\t',create_using=nx.DiGraph,nodetype=str)
    
    '''
    Vaccine network communities  
    The communities were calculated with the leading eigenvector methods of 
    python-igraph 
    (https://igraph.org/c/doc/igraph-Community.html#igraph_community_leading_eigenvector),
    asking for 2 communities. The variable is a dictionary were each key is a user and the value
    is the community it was assigned to. For example, com_of_user[\<node name\>] will be 'A' or 'B'.
    '''
    with open(STOR_DIR+PATH_COM_OF_USER,'rb') as f:
        com_of_user=pickle.load(f)
    
    '''
    Timelines data  
    For about 14K users which were active on Twitter during February 2022, we downloaded up 
    to 3000 tweets and retweets written from 1/2/2022 to 10/3/2022. This dataframe contains 
    this data: each row is a tweet with different information. Also the original tweets 
    they retweeted were included as rows of the DataFrame.  
    '''
    with open(STOR_DIR+PATH_UKRAINE,'rb') as f:
        df=pickle.load(f)
    
    '''
    To build a retweet network, the relevant columns are user.id and retweeted_status.user.id.
    In fact, the first one is the retweeting user (x) and the second one the retweeted user 
    (y) of the edge x->y.  
    '''
    # Set the minimum the weight threshold for edges to be considered (1 is the 
    #minimum -> keep edges with at least 1 retweet from the users)
    
    list_col = ['user.id','retweeted_status.user.id']
    df_edgelist=df[list_col].copy()
    df_edgelist=df_edgelist.dropna(how='any')
    df_edgelist=df_edgelist.groupby(list_col).size()
    df_edgelist = df_edgelist.reset_index().rename(columns={0:'weight'})
    
    '''
    df_edgelist=df[['user.id','retweeted_status.user.id']].copy()
    df_edgelist=df_edgelist.dropna(how='any')
    df_edgelist=df_edgelist.groupby(df_edgelist.columns.tolist()).size().reset_index().rename(
        columns={0:'weight'})
    df_edgelist=df_edgelist[df_edgelist.weight>=min_rt]
    '''
    df_edgelist.to_csv('data/edgelist_w{}.txt'.format(min_rt), header=None, index=None, sep='\t')