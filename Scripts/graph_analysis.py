# -*- coding: utf-8 -*-

import bct
import pickle
import igraph
import numpy as np
import pandas as pd
from pprint import pprint
import seaborn as sns
import matplotlib.pyplot as plt

# Utilities/Small World
def simulate(rep = 1000, fun = lambda: gen_random_graph(100,100)):
    output = []
    for _ in range(rep):
        output.append(fun())
    return output
        
def gen_random_graph(n = 10, m = 10, template_graph = None):
    if template_graph:
        G = template_graph.copy()
        G.rewire() # bct.randomizer_bin_und works on binary adjrices
    else:
        #  Generates a random binary graph with n vertices and m edges
        G = igraph.Graph.Erdos_Renyi(n = n, m = m)    
    # get cluster coeffcient. Transitivity is closed triangles/total triplets
    c = G.transitivity_undirected() 
    # get average (shortest) path length
    l = G.average_path_length()
    return (G,c,l,c/l)

def calc_small_world(G):
    # simulate random graphs with same number of nodes and edges
    sim_out = simulate(rep = 10000, fun = lambda: gen_random_graph(n = len(G.vs), m = len(G.es)))
    # get average C and L for random
    C_random = np.mean([i[1] for i in sim_out])
    L_random = np.mean([i[2] for i in sim_out])
    # calculate relative clustering and path length vs random networks
    Gamma = G.transitivity_undirected()/C_random
    Lambda = G.average_path_length()/L_random
    # small world coefficient
    Sigma = gamma/lam   
    return (Sigma, Gamma, Lambda)
    
# Visualization
def plot_graph(G, visual_style = None, inline = True):
    if not visual_style:
        visual_style = {}
        if 'weight' in G.es:
            visual_style['edge_width'] = [10 * weight for weight in G.es['weight']]
        visual_style['layout'] = G_bin.layout("kk")
    fig = igraph.plot(G, inline = inline, **visual_style)
    return fig

def plot_mat(mat, labels = []):
    fig = plt.figure(figsize = [16,12])
    ax = fig.add_axes([.25,.15,.7,.7]) 
    sns.heatmap(mat)
    if len(labels) > 0:
        ax.set_yticklabels(labels, rotation = 0, fontsize = 'large')
    return fig
    
def print_community_members(G, lookup = {}):
    assert set(['community','id','within_module_degree','name',  'eigen_centrality']) <=  set(G.vs.attribute_names()), \
        "Missing some required vertex attributes. Vertices must have id, name, part_coef, eigen_centrality, community and within_module_degree"
        
    print('Key: Node index, Within Module Degree, Measure, Eigenvector centrality')
    for community in np.unique(G.vs['community']):
        #find members
        members = [lookup.get(v['name'],v['name']) for v in G.vs if v['community'] == community]
        # ids and total degree
        ids = [v['id'] for v in G.vs if v['community'] == community]
        eigen = [round(v['eigen_centrality'],2) for v in G.vs if v['community'] == community]
        #sort by within degree
        within_degrees = [round(v['within_module_degree'],2) for v in G.vs if v['community'] == community]
        to_print = zip(ids, within_degrees,  members, eigen)
        to_print.sort(key = lambda x: -x[1])
        
        print('Members of community ' + str(community) + ':')
        pprint(to_print)
        print('')

# Main graph analysis function
def Graph_Analysis(data, threshold = .3, weight = True, layout = 'kk', reorder = True, display = True, filey = None):
    connectivity_matrix = data.corr().as_matrix()
    # remove diagnoal (required by bct) and uppder triangle
    np.fill_diagonal(connectivity_matrix,0)
    
    #threshold
    graph_mat = bct.threshold_proportional(connectivity_matrix,threshold)
    # make a binary version if not weighted
    if not weight:
        graph_mat = np.ceil(graph_mat)
        G = igraph.Graph.Adjacency(graph_mat.tolist(), mode = 'undirected')
    else:
        G = igraph.Graph.Weighted_Adjacency(graph_mat.tolist(), mode = 'undirected')

    # community detection
    # using louvain but also bct.modularity_und which is "Newman's spectral community detection"
    comm, mod = bct.community_louvain(graph_mat)
    
    #if reorder, reorder vertices by community membership
    if reorder:
        data = data.iloc[:,np.argsort(comm)]
        comm = np.sort(comm)
        connectivity_matrix = data.corr().as_matrix()
        # remove diagnoal (required by bct) and uppder triangle
        np.fill_diagonal(connectivity_matrix,0)
        
        #threshold
        graph_mat = bct.threshold_proportional(connectivity_matrix,threshold)
        # make a binary version if not weighted
        if not weight:
            graph_mat = np.ceil(graph_mat)
            G = igraph.Graph.Adjacency(graph_mat.tolist(), mode = 'undirected')
        else:
            G = igraph.Graph.Weighted_Adjacency(graph_mat.tolist(), mode = 'undirected')
    
    G.vs['community'] = comm
    G.vs['id'] = range(len(G.vs))
    G.vs['name'] = data.columns 
    G.vs['within_module_degree'] = bct.module_degree_zscore(graph_mat,comm)
    G.vs['part_coef'] = bct.participation_coef(graph_mat, comm)
    if weight:
        G.vs['eigen_centrality'] = G.eigenvector_centrality(directed = False, weights = G.es['weight'])
    else:
        G.vs['eigen_centrality'] = G.eigenvector_centrality(directed = False)
    
    if display:
        # plot community structure
        # make layout:
        layout = G.layout(layout)
        
        
        
        # color by community and within-module-centrality
        # each community is a different color palette, darks colors are more central to the module
        num_colors = 20.0
        palettes = ['Blues','Reds','Greens','Greys','Purples','Oranges']
        
        min_degree = np.min(G.vs['within_module_degree'])
        max_degree = np.max(G.vs['within_module_degree']-min_degree)
        within_degree = [(v-min_degree)/max_degree for v in G.vs['within_module_degree']]
        within_degree = np.digitize(within_degree, bins = np.arange(0,1,1/num_colors))
        
        vertex_color = [sns.color_palette(palettes[v['community']-1], int(num_colors)+1)[within_degree[i]] for i,v in enumerate(G.vs)]
        
        visual_style = {'layout': layout, 
                        'vertex_color': vertex_color, 
                        'vertex_size': [c*50+20 for c in G.vs['eigen_centrality']], 
                        'vertex_label': G.vs['id'],
                        'vertex_label_size': 20,
                        'bbox': (1000,1000),
                        'margin': 50}
        if weight:
            visual_style['edge_width'] = [w*4 for w in G.es['weight']]
            
        #visualize
        print_community_members(G, lookup = verbose_lookup)
        if filey:
            plot_graph(G, filey, visual_style = visual_style, inline = False)
        else:
            plot_graph(G, visual_style = visual_style, inline = False)
    return (G, graph_mat)
    
#**********************************
# Load Data
#**********************************  
    
datasets = pickle.load(open('../Data/subset_data.pkl','r'))
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
corr_mat = corr_data.corr().as_matrix()


# absolute value of correlations (not used)
valence_mat = np.ceil(corr_mat)+np.floor(corr_mat)
abs_mat = np.abs(corr_mat)

#**********************************
# Binary Analysis
#**********************************
G_bin, connectivity_adj = Graph_Analysis(corr_data, weight = False, layout = 'kk')
Sigma, Gamma, Lambda = calc_small_world(G)



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
G_w, connectivity_mat = Graph_Analysis(corr_data, threshold = .3, weight = True, reorder = True)
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
sns.stripplot(x = zip(*cluster_size)[0], y = zip(*cluster_size)[1], jitter = .4)
plt.ylabel('Number of detected communities', size = 20)
plt.xlabel('Threshold', size = 20)

plt.figure(figsize = (16,12))
#sns.stripplot(x = zip(*partition_distances)[0], y = zip(*partition_distances)[1], jitter = .2)
sns.boxplot(x = zip(*partition_distances)[0], y = zip(*partition_distances)[1])
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























