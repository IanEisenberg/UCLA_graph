# -*- coding: utf-8 -*-
import bct
import pickle
from util import calc_small_world, community_reorder, get_subgraph, get_visual_style, \
            Graph_Analysis, plot_graph, print_community_members, \
            threshold_proportional_sign
    
    
#**********************************
# Load Data
#**********************************  
    
datasets = pickle.load(open('../Data/subset_data.pkl','rb'))
data = datasets['all_data']
task_data = datasets['task_data']
survey_data = datasets['survey_data']
verbose_lookup = datasets['verbose_lookup']

#**********************************
# Prepare adjacency matrix
#**********************************

# create correlation matrix
corr_data = data.copy()
corr_data.drop(['ptid','gender','age'], axis=1, inplace=True)






# Hubs
# get "high degree" threshold = 1 std above mean degree
degree_threshold = np.mean(G.degree()) + np.std(G.degree())
high_degree_nodes = G.degree() > degree_threshold
G.vs['hub'] = high_degree_nodes
connector_hubs = G.vs.select(lambda v: v['hub'] == True and v['part_coef'] > .3)
provinicial_hubs = G.vs.select(lambda v: v['hub'] == True and v['part_coef'] <= .3)

#**********************************
# Weighted Analysis
#**********************************  
method = 'positive'
if method == 'signed': 
    #threshold and thresh_Func
    t = 1
    t_f = threshold_proportional_sign
    #edge metric
    em = 'spearman'
    #community algorithm
    gamma = 1
    c_a = lambda x: bct.modularity_louvain_und_sign(x, gamma = gamma)
elif method == 'positive':
    #threshold and thresh_Func
    t = .3
    t_f = bct.threshold_proportional
    #edge metric
    em = 'spearman'
    #community algorithm
    gamma = 1
    c_a = lambda x: bct.modularity_louvain_und(x, gamma = gamma)
    
G_w, connectivity_adj, visual_style = Graph_Analysis(corr_data, community_alg = c_a, thresh_func = t_f, edge_metric = em, 
                                                     reorder = True, threshold = t, weight = True, layout = 'circle', 
                                                     print_options = {'lookup': verbose_lookup}, plot_options = {'inline': False})

subgraph = community_reorder(get_subgraph(G_w,2))
print_community_members(subgraph)
subgraph_visual_style = get_visual_style(subgraph, vertex_size = 'eigen_centrality')
plot_graph(subgraph, visual_style = subgraph_visual_style, layout = 'circle', inline = False)

#subgraph analysis                                            
Sigma, Gamma, Lambda = calc_small_world(G_w)

#**********************************
# Save Graphs
#**********************************
visual_style = None
for metric in ['pearson','spearman','abs_pearson','abs_spearman','MI']:
    if not visual_style:
        layout = 'kk'
    else:
        layout = visual_style['layout']
    G_w, connectivity_adj, visual_style = Graph_Analysis(corr_data, edge_metric = metric, weight = True, 
                                                              reorder = False, layout = layout, 
                                                              print_options = {'lookup': verbose_lookup, 'file': '../Plots/weighted_' + metric + '.txt'},
                                                              plot_options = {'inline': False, 'target': '../Plots/weighted_' + metric + '.pdf'})


#**********************************
# Binary Analysis
#**********************************
t = .3
#edge metric
em = 'spearman'
#community algorithm
c_a = bct.community_louvain
G_bin, connectivity_adj, visual_style = Graph_Analysis(corr_data, community_alg = c_a, edge_metric = em,
                                                       threshold = t, weight = False, layout = 'kk', inline = False)
Sigma, Gamma, Lambda = calc_small_world(G)


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























