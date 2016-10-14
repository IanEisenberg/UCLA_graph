# -*- coding: utf-8 -*-
import bct
import fancyimpute
import numpy as np
import pandas as pd
import pickle
from util import calc_connectivity_mat, community_reorder, get_subgraph, get_visual_style, Graph_Analysis
from util import plot_graph, print_community_members, simulate, threshold_proportional_sign

    
    
#**********************************
# Load Data
#**********************************  
    
datasets = pickle.load(open('../Data/subset_data.pkl','rb'))
data = datasets['all_data']
task_data = datasets['task_data']
survey_data = datasets['survey_data']
verbose_lookup = datasets['verbose_lookup']


# ************************************
# ************ Imputation *******************
# ************************************
data.drop(['ptid','gender','age'], axis=1, inplace=True)
data_complete = fancyimpute.SoftImpute().complete(data)
data_complete = pd.DataFrame(data_complete, index = data.index, columns = data.columns)

# ************************************
# ************ Connectivity Matrix *******************
# ************************************

spearman_connectivity = calc_connectivity_mat(data_complete, edge_metric = 'spearman')
distance_connectivity = calc_connectivity_mat(data_complete, edge_metric = 'distance')

# ************************************
# ********* Graphs *******************
# ************************************

def get_fully_connected_threshold(connectivity_matrix, initial_value = .1):
    '''Get a threshold above the initial value such that the graph is fully connected
    '''
    if type(connectivity_matrix) == pd.DataFrame:
        connectivity_matrix = connectivity_matrix.as_matrix()
    threshold = initial_value
    thresholded_mat = bct.threshold_proportional(connectivity_matrix,threshold)
    while np.any(np.max(thresholded_mat, axis = 1)==0):
        threshold += .01
        thresholded_mat = bct.threshold_proportional(connectivity_matrix,threshold)
    return threshold
    
# threshold positive graph
t = .5
plot_t = get_fully_connected_threshold(spearman_connectivity, .1)
t_f = bct.threshold_proportional
c_a = bct.community_louvain

G_w, connectivity_adj, threshold_visual_style = Graph_Analysis(spearman_connectivity, community_alg = c_a, 
                                                    thresh_func = t_f, threshold = t, plot_threshold = plot_t,
                                                     print_options = {'lookup': {}}, 
                                                    plot_options = {'inline': False})
                                                                                                  
# distance graph
t = 1
plot_t = get_fully_connected_threshold(distance_connectivity, .1)
t_f = bct.threshold_proportional
c_a = lambda x: bct.community_louvain(x, gamma = 1)

G_w, connectivity_adj, visual_style = Graph_Analysis(distance_connectivity, community_alg = c_a, thresh_func = t_f,
                                                      threshold = t, plot_threshold = plot_t,
                                                     print_options = {'lookup': {}}, 
                                                    plot_options = {'inline': False})

# signed graph
t = 1
t_f = threshold_proportional_sign
c_a = bct.modularity_louvain_und_sign                                               

# circle layout                                                  
G_w, connectivity_mat, visual_style = Graph_Analysis(spearman_connectivity, community_alg = c_a, thresh_func = t_f,
                                                     reorder = True, threshold = t,  layout = 'circle', 
                                                     plot_threshold = plot_t, print_options = {'lookup': {}}, 
                                                    plot_options = {'inline': False})
                                                    

subgraph = community_reorder(get_subgraph(G_w,3))
print_community_members(subgraph)
subgraph_visual_style = get_visual_style(subgraph, vertex_size = 'eigen_centrality')
plot_graph(subgraph, visual_style = subgraph_visual_style, layout = 'circle', inline = False)



#**********************************
# Threshold and Stability Analysis
#**********************************    
 
# plot number of communities as a function of threshold
thresholds = np.arange(.15,.35,.01)
partition_distances = []
cluster_size = []
for t in thresholds:
    # simulate threhsold 100 times
    clusters = simulate(100, fun = lambda: Graph_Analysis(corr_data, threshold = t, weight = True, display = False)[0].vs['community'])
    distances = []
    for i,c1 in enumerate(clusters):
        for c2 in clusters[i:]:
            distances.append(bct.partition_distance(c1,c2)[0])
    partition_distances += zip([t]*len(distances), distances)
    cluster_size += (zip([t]*len(clusters), [max(c) for c in clusters]))
    

plt.figure(figsize = (16,12))
sns.stripplot(x = list(zip(*cluster_size))[0], y = list(zip(*cluster_size))[1], jitter = .4)
plt.ylabel('Number of detected communities', size = 20)
plt.xlabel('Threshold', size = 20)

plt.figure(figsize = (16,12))
#sns.stripplot(x = zip(*partition_distances)[0], y = zip(*partition_distances)[1], jitter = .2)
sns.boxplot(x = list(zip(*partition_distances))[0], y = list(zip(*partition_distances))[1])
plt.ylabel('Partition distance over 100 repetitions', size = 20)
plt.xlabel('Threshold', size = 20)



#**********************************
# Practice functions
#**********************************

# participation coefficient definition. Unneeded, just ust bct.participation_coef
def get_participation_coefficient(G):
    adj_list = G_bin.get_adjlist()
    part_coef = []
    for j in range(len(G_bin.vs)):
        summ = 0
        for c in np.unique(comm):
            connections_within = float(len([i for i in adj_list[j] if comm[i] == c]))
            summ += (connections_within/G_bin.vs[j].degree())**2
        part_coef.append(1-summ)
    return part_coef























