import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import random
import pyttsx3
import time


def talkWith(engine, content):
    """ 朗读内容 """
    engine.say(content)
    engine.runAndWait()


def talkContent(content):
    """ 朗读字符串内容 使用系统文字转语音 """

    engine = pyttsx3.init()
    # 设置朗读速度
    engine.setProperty('rate', 160)
    # 如果字符串过长 通过句号分隔 循环读取
    if len(content) > 20:
        con_list = content.split('。')
        for item in con_list:
            time.sleep(1)
            talkWith(engine, item)
    else:
        talkWith(engine, content)

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

def np_adj_to_df_link(adj, node_label='code',stock_dict={}):
	"""
	为了环形可视化的数据需求
	"""
	code_list = list(stock_dict.keys())
	name_list = list(stock_dict.values())
	node_from_list = []
	node2_to_list = []
	edge_weight_list = []
	for node1,node1_list in enumerate(adj):
		for node2,edge_weight in enumerate(node1_list):
			if abs(edge_weight) > 0:
				if node_label == 'index':
					node_from_list.append(node1)
					node2_to_list.append(node2)
				if node_label == 'code':
					node_from_list.append(code_list[node1])
					node2_to_list.append(code_list[node2])
				if node_label == 'name':
					node_from_list.append(name_list[node1])
					node2_to_list.append(name_list[node2])
				edge_weight_list.append(edge_weight)
	df = pd.DataFrame()
	df['ind1'] = node_from_list
	df['ind2'] = node2_to_list
	df['C'] = edge_weight_list
	df.index = range(1,len(node_from_list)+1)
	df.to_csv('np_adj_to_df_link.csv',encoding='utf-8')


def plot3d_matrix_bar(matrix):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	node_y = range(len(matrix))
	for  node_y_k in node_y:
		# Generate the random data for the y=k 'layer'.
		node_x = np.arange(len(matrix))
		zs = matrix[node_y_k]
		# You can provide either a single color or an array with the same length as
		# xs and ys. To demonstrate this, we color the first bar of each set cyan.
		# Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
		ax.bar(node_x, zs, zs=node_y_k, zdir='y', alpha=0.8)

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	# On the y axis let's only label the discrete values that we have data for.
	ax.set_yticks(node_y)
	plt.show()


if __name__ == '__main__':
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

