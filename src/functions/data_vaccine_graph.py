import networkx as nx
import pickle 
from functions import (
    assign_communities,
    mixing_matrix,
    create_df,
    degree_distributions,
    mixing_matrix_manipulation
)
from configurations import (
    STOR_DIR,
    PATH_VACCINE,
    PATH_MIXING_MATRIX,
    PATH_COM_OF_USER,
    PATH_DEGREE
)

def main():  
    #Load the network built from the retweets on Italian "vaccine" tweets since October 2020. 
   
    Gvac=nx.read_weighted_edgelist(STOR_DIR+PATH_VACCINE,
                                       delimiter='\t',
                                       create_using=nx.DiGraph,
                                       nodetype=str)
    
    with open(STOR_DIR+PATH_COM_OF_USER,'rb') as f:
        com_of_user=pickle.load(f)
        
    Gvac_subgraph, Gvac_A, Gvac_B, group_A, group_B = assign_communities(Gvac, com_of_user)
    df, df_weight = mixing_matrix(Gvac, com_of_user) #mixing matrix in unweighted and weighted case
    df1, df2 = mixing_matrix_manipulation(df)
    df3, df4 = mixing_matrix_manipulation(df_weight)
    
    #I save a file for each table.
    
    df.to_csv(PATH_MIXING_MATRIX+'MixingVaccine1.csv', index=False)
    df1.to_csv(PATH_MIXING_MATRIX+'MixingVaccine2.csv', index=False)
    df2.to_csv(PATH_MIXING_MATRIX+'MixingVaccine3.csv', index=False)
    df_weight.to_csv(PATH_MIXING_MATRIX+'MixingVaccine4.csv', index=False)
    df3.to_csv(PATH_MIXING_MATRIX+'MixingVaccine5.csv', index=False)
    df4.to_csv(PATH_MIXING_MATRIX+'MixingVaccine6.csv', index=False)
    
    #We create 6 lists to store the in- out-degree of the nodes belonging to the whole network, group A and B. We save them in a
    #corresponding file.    
    
    in_degree_original, out_degree_original = degree_distributions(Gvac)
    in_degree_group_A, out_degree_group_A = degree_distributions(Gvac_A)
    in_degree_group_B, out_degree_group_B = degree_distributions(Gvac_B)
    
    df_original = create_df(['in_degree_original', 'out_degree_original'],[in_degree_original, out_degree_original])
    df_original.to_csv(PATH_DEGREE+"DegreeOriginal.csv", index = False)
main()