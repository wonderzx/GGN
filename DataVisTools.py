import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import random

def three_dim_np_to_list(np_array):
	result = []
	dim_2_result = []
	for dim_1_list in np_array:
		dim_2_result.clear()
		for dim_2_list in dim_1_list:
			dim_2_result.append(list(dim_2_list))
		result.append(dim_2_result.copy())
	return result

def two_dim_np_to_list(np_array):
	result = []
	for dim_1_list in np_array:
		result.append(list(dim_1_list))
	return result

def np_adj_to_txt_link(adj):
	link_str = ''
	for node1,node1_list in enumerate(adj):
		for node2,exist_edge in enumerate(node1_list):
			if exist_edge==1:
				link_str += str(node1)+' '+str(node2) + ' 0\n'
	with open("adj_to_net.txt", 'wb') as f:
		f.write(link_str.encode())
		f.close()

def np_adj_to_df_link(adj):
	node_from_list = []
	node2_to_list = []
	edge_weight_list = []
	for node1,node1_list in enumerate(adj):
		for node2,edge_weight in enumerate(node1_list):
			if abs(edge_weight)>0:
				node_from_list.append(node1)
				node2_to_list.append(node2)
				edge_weight_list.append(edge_weight*random.random())
	df = pd.DataFrame()
	df['ind1'] = node_from_list
	df['ind2'] = node2_to_list
	df['C'] = edge_weight_list
	df.index = range(1,len(node_from_list)+1)
	df.to_csv('np_adj_to_df_link.csv')


adj_path = './data/bn/mark-13826-adjmat.pickle'
# adj_path = './logs/exp2022-06-29T22;04;51.240862/standard_adj.pickle'
with open(adj_path,'rb') as f:
	adj_matrix = pickle.load(f)

# G = nx.from_numpy_array(adj_matrix)
G = nx.from_numpy_matrix(adj_matrix, create_using=nx.DiGraph)

subax1 = plt.subplot(121)
nx.draw(G, with_labels=True, font_weight='bold')
subax2 = plt.subplot(122)
nx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold')
nx.draw(G,pos = nx.drawing.layout.spring_layout(G),node_color='#0000CD', edge_color='#000000',width=0.01, node_size = 0.5,edge_cmap=plt.cm.gray, with_labels=False)
plt.savefig('stockNet.png',dpi=500,bbox_inches = 'tight', facecolor='white', edgecolor='red')
plt.show()

