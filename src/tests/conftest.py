import pandas as pd
import networkx as nx
import pickle
import numpy as np
import datetime 
import os
from collections import Counter 
import copy
import pytest

STOR_DIR='/mnt/stor/users/francesco.durazzi2/twitter/' #Paths used
EDGELIST='vaccines/data/edgelist_w1.txt'
PATH_COM_OF_USER = 'vaccines/data/v_com_of_user_w1_2020-10-07_2022-03-10.pkl'

@pytest.fixture
def ReadFileG():
    G=nx.read_weighted_edgelist(STOR_DIR+EDGELIST,
                                       delimiter='\t',
                                       create_using=nx.DiGraph,
                                       nodetype=str)
    return G
        
@pytest.fixture
def ReadFileComOfUser():
    with open(STOR_DIR+PATH_COM_OF_USER,'rb') as f: 
        com_of_user=pickle.load(f)
    return com_of_user
        