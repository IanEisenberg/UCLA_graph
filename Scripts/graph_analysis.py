# -*- coding: utf-8 -*-

import bct
import pickle
import igraph
import numpy as np
import pandas as pd
from pprint import pprint
import seaborn as sns
import matplotlib.pyplot as plt

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
    
def simulate(rep = 1000, fun = lambda: gen_random_graph(100,100)):
    output = []
    for _ in range(rep):
        output.append(fun())
    return output
        
def plot_graph(G, visual_style = None):
    if not visual_style:
        visual_style = {}
        if 'weight' in G.es:
            visual_style['edge_width'] = [10 * weight for weight in G.es['weight']]
        visual_style['layout'] = G_bin.layout("kk")
    fig = igraph.plot(G, **visual_style)
    return fig

def plot_mat(mat, labels = []):
    fig = plt.figure(figsize = [16,12])
    ax = fig.add_axes([.25,.15,.7,.7]) 
    sns.heatmap(mat)
    if len(labels) > 0:
        ax.set_yticklabels(labels, rotation = 0, fontsize = 'large')
    return fig
    
def print_community_members(G):
    for community in np.unique(G_bin.vs['community']):
        members = [v['name'] for v in G_bin.vs if v['community'] == community]
        print('Members of community ' + str(community) + ':')
        pprint(members)
        print('')
        
    
#**********************************
# Load Data
#**********************************  
    
datasets = pickle.load(open('../Data/subset_data.pkl','r'))
data = datasets['all_data']
task_data = datasets['task_data']
survey_data = datasets['survey_data']

#**********************************
# Prepare adjacency matrix
#**********************************

# create correlation matrix
corr_data = data.copy()
corr_data.drop(['ptid','gender','age'], axis=1, inplace=True)
corr_mat = corr_data.corr().as_matrix()
# remove diagnoal (required by bct) and uppder triangle
np.fill_diagonal(corr_mat,0)
# absolute value of correlations (not used)
valence_mat = np.ceil(corr_mat)+np.floor(corr_mat)
abs_mat = np.abs(corr_mat)

#**********************************
# Binary Analysis
#**********************************
threshold = .3

#threshold
thresh_mat = bct.threshold_proportional(corr_mat,threshold)
# make a binary version
adj = np.ceil(thresh_mat)

# make a binary graph
G_bin = igraph.Graph.Adjacency(adj.tolist(), mode = 'undirected')
G_bin.vs['name'] = corr_data.columns

# simulate random graphs with same number of nodes and edges
sim_out = simulate(rep = 10000, fun = lambda: gen_random_graph(n = len(G_bin.vs), m = len(G_bin.es)))
# get average C and L for random
C_random = np.mean([i[1] for i in sim_out])
L_random = np.mean([i[2] for i in sim_out])
# calculate relative clustering and path length vs random networks
gamma = G_bin.transitivity_undirected()/C_random
lam = G_bin.average_path_length()/L_random
# small world coefficient
sigma = gamma/lam

# community detection
# using louvain but also bct.modularity_und which is "Newman's spectral community detection"
comm, mod = bct.community_louvain(adj)
G_bin.vs['community'] = comm
part_coef = bct.participation_coef(adj, comm)
G_bin.vs['part_coef'] = part_coef

# plot community structure
color_palette = sns.color_palette('hls', np.max(comm))
color_dict = {i+1:color_palette[i] for i in range(np.max(comm))}
community_color = [color_dict[v] for v in G_bin.vs['community']]
visual_style = {'layout': 'kk', 'vertex_color': community_color, 'vertex_size': [p*60 for p in part_coef]}
plot_graph(G_bin, visual_style = visual_style)
print_community_members(G_bin)

# Hubs
# get "high degree" threshold = 1 std above mean degree
degree_threshold = np.mean(G_bin.degree()) + np.std(G_bin.degree())
high_degree_nodes = G_bin.degree() > degree_threshold
G_bin.vs['hub'] = high_degree_nodes
connector_hubs = G_bin.vs.select(lambda v: v['hub'] == True and v['part_coef'] > .3)
provinicial_hubs = G_bin.vs.select(lambda v: v['hub'] == True and v['part_coef'] <= .3)

#**********************************
# Weighted Analysis
#**********************************
G_weighted = igraph.Graph.Weighted_Adjacency(thresh_adj.tolist(), mode="undirected")
G_weighted.vs['name'] = corr_data.columns





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























