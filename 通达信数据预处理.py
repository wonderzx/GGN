import os
import bs4 as bs # 导入 beautiful soup4 包，用于抓取网页信息
import requests # 导入 request 用于获取网站上的源码
import seaborn as sns  # 可视化相似度矩阵
import matplotlib as plt
from matplotlib.collections import LineCollection   # 见参考Matplotlib.collections.LineCollection结构及用法
import numpy as np
import pandas as pd
import datetime as dt
import time
from sklearn import cluster, covariance, manifold
import pickle
import json
from DataVisTools import * # 数据可视化辅助工具
from mpl_toolkits.mplot3d import Axes3D


# 通达信数据预处理工具
class TdxDataPreprocesser():
    # 专门用来处理通达信导出数据的类
    def __init__(self):
        self.original_data_folder = './day/'
        self.preprocessed_data_folder = './tdx_pickle/'
        self.boolean_network_data_folder = './bn_json/'
        self.vibrate__network_data_folder = './vibrate_json/'
        self.boolean_network_np = None
        self.vibrate_network_np = None
        self.vibrate_network_np_dict = {}
        self.data_folder_stock_code_list = []
        self.ap_cluster_result_dict = {}
        self.column_names_list = ['date', 'open', 'high', 'low', 'close', 'trd_volume', 'trd_value']
        self.column_type_dict = {'date': str, 'open': float, 'high': float, 'low': float, 'close': float, 'trd_volume': float, 'trd_value': float}

    def read_one_file_data(self,filename):
        """
        pickle去掉了csv的开头两行和末尾一行
        把字符串的日期改成了datetime日期
        """
        df = pd.read_csv(self.original_data_folder + filename, sep=',', header=None, names=self.column_names_list,
                         dtype=self.column_type_dict, skiprows=2, encoding='gbk')
        df.drop([len(df) - 1], inplace=True)
        df['date'] = df['date'].apply(lambda x: dt.datetime.strptime(x, '%Y/%m/%d'))
        # df = df.set_index('date')
        # df['times '] = df['times '].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))   # 日期转字符串
        # df['date'] = df['date'].apply(lambda x: dt.time.mktime(dt.time.strptime(x, '%Y-%m-%d %H:%M:%S'))) # 字串串转日期
        # df['date'] = df['date'].apply(lambda x: time.mktime(time.strptime(x, '%Y-%m-%d')))
        return df

    def read_one_file_info(self,filename):
        """
        提取股票代码和名称
        """
        df = pd.read_csv(self.original_data_folder+filename, sep=' ',nrows=0,encoding='gbk')
        return df

    def get_stock_list_from_internet(self, re_request_list=False):
        if re_request_list:
            # https://q.stock.sohu.com/cn/bk.shtml 所有板块在这里查看
            # resp = requests.get('https://q.stock.sohu.com/cn/bk_4507.shtml')  # 上证50
            # resp = requests.get('https://q.stock.sohu.com/cn/bk_4506.shtml')  # 上证380
            # resp = requests.get('https://q.stock.sohu.com/cn/bk_4489.shtml')# 融资融券
            resp = requests.get('https://q.stock.sohu.com/cn/bk_4839.shtml')  # 富时罗素
            resp.encoding = 'gb2312'
            soup = bs.BeautifulSoup(resp.text, 'lxml')
            table = soup.find('table', {'id': 'BIZ_MS_plstock'})
            stock_dict = {}
            for row in table.findAll('tr')[2:]:
                code = row.findAll('td')[0].text
                name = row.findAll('td')[1].text
                stock_dict[code] = name

            with open('./data/stock_dict.pickle', 'wb') as f:
                pickle.dump(stock_dict, f)
        else:
            with open('./data/stock_dict.pickle', 'rb') as f:
                stock_dict = pickle.load(f)
        return stock_dict

    def get_stock_list_from_file(self, re_request_list=False,filename='指数板块.txt',class_code='000300'):
        if re_request_list:
            class_stock_df = pd.read_csv(filename,header=None,names=['class_code','class_name','stock_code','stock_name']
                                         ,dtype={'class_code' : str,'class_name' : str,'stock_code' : str,'stock_name' : str}
                                         , encoding='gbk')
            class_df = class_stock_df[class_stock_df['class_code']==class_code]
            stock_dict = class_df.set_index('stock_code')['stock_name'].to_dict()
            with open('./data/stock_dict.pickle', 'wb') as f:
                pickle.dump(stock_dict, f)
        else:
            with open('./data/stock_dict.pickle', 'rb') as f:
                stock_dict = pickle.load(f)
        return stock_dict

    def tdx_data_to_pickle_files(self):
        """
        提取股票代码和名称
        pickle去掉了csv的开头两行和末尾一行
        把字符串的日期改成了datetime日期
        """
        dir_list = os.listdir(self.original_data_folder)
        for i,single_dir in enumerate(dir_list):
            stock_info_df = self.read_one_file_info(single_dir)
            stock_code = stock_info_df.columns[0]
            stock_name = stock_info_df.columns[1]
            k_line_df = self.read_one_file_data(single_dir)
            single_csv_data_dict = {'stock_code': stock_code, 'stock_name': stock_name, 'df': k_line_df}
            with open(self.preprocessed_data_folder+stock_code+'.pickle','wb') as f:
                pickle.dump(single_csv_data_dict,f)
                f.close()
            print(str(i)+'th file preprocessed!')
            # print('\x1b[2J') # 清屏效果

    def plot_similarity_matrix(self,similarity_matrix):
        # self.plot_matrix(similarity_matrix)
        # plot3d_matrix_bar(similarity_matrix)
        plt.figure()
        similarity_dis = similarity_matrix.reshape(len(similarity_matrix)**2)
        similarity_dis = np.extract(similarity_dis != 1,similarity_dis)
        similarity_dis = np.sort(similarity_dis)
        print("max in similarity_dis is %f" % similarity_dis.max())
        print("min in similarity_dis is %f" % similarity_dis.min())
        print("averge in similarity_dis is %f" % np.average(similarity_dis))
        print("midnum in similarity_dis is %f" % np.median(similarity_dis))
        plt.bar(range(len(similarity_dis)),similarity_dis)
        plt.show()

    def plot_matrix(self,similarity_matrix):
        ax = sns.heatmap(similarity_matrix, cmap="YlGn",
                         square=True,  # 正方形格仔
                         cbar=False,  # 去除 color bar
                         xticklabels=False, yticklabels=False)  # 去除纵、横轴 label
        fig = ax.get_figure()
        fig.show()
        fig.savefig('similarity_matrix.jpg', bbox_inches='tight',pad_inches=0.0)  # 去白边

    def pickle_to_ap_bn_adj_matrix(self,stock_dict={}, start='2020/2/20', end='2022/2/20', edge_orientation='to_center',
                                   inter_cluster_edge_exist=True,result_save_name=''):
        start_date = dt.datetime.strptime(start, '%Y/%m/%d')
        end_date = dt.datetime.strptime(end, '%Y/%m/%d')
        stock_code_np, stock_name_np = np.array(sorted(stock_dict.items())).T  # 将symbol_dict转换维（key,value）形式的列，并排序，然后转为2×50数组。最后进行拆包，返回两个numpy.array
        rose = []  # 实例化list，用于承载变化信息
        boolean_series = pd.DataFrame()
        for stock_code in stock_code_np:
            with open(self.preprocessed_data_folder + stock_code + '.pickle', 'rb') as f:
                now_stock_dict = pickle.load(f)
                f.close()
                now_stock_df = now_stock_dict['df']
                now_stock_df['index'] = now_stock_df['date']
                now_stock_df = now_stock_df.set_index('index')
                between_df = now_stock_df[now_stock_df.date.between(start_date, end_date)].copy()
                between_df['rose'] = between_df['close'] > between_df['open']  # 没涨归为跌
                between_df['boolean'] = between_df['rose'].apply(lambda x: 1 if x else 0)
                boolean_series[stock_code] = between_df['boolean'].copy()

        boolean_series.fillna(0, inplace=True)  # 没数据归为跌，
        row_num = boolean_series.shape[0]
        col_num = boolean_series.shape[1]
        similarity_matrix = np.ones([col_num,col_num],float) # col_num * col_num 的矩阵！
        for row_i, code1 in enumerate(stock_code_np):
            for col_i, code2 in enumerate(stock_code_np):
                same_series = (boolean_series[code1] == boolean_series[code2])
                same_series = same_series.apply(lambda x: 1 if x else 0)
                similarity = same_series.sum()
                similarity_matrix[row_i][col_i] = similarity/row_num
        self.plot_similarity_matrix(similarity_matrix)
        print('similarity_matrix',similarity_matrix)
        cluster_center, labels, iter_times, A_matrix, R_matirx = cluster.affinity_propagation(similarity_matrix,random_state=0, return_n_iter=True)  # 返回划分好的聚类中心的索引和聚类中心的标签
        self.ap_cluster_result_dict = {'cluster_center': cluster_center, 'labels': labels, 'iter_times': iter_times,
                                  'A_matrix': A_matrix, 'R_matirx': R_matirx}
        n_labels = labels.max()  # 返回标签中的最大值，标签默认是数字递增形式的
        with open("boolean_ap_cluster_result.txt",'w') as f:
            for i in range(n_labels + 1):
                print('聚类类别 %i: %s ;;聚类中心是 %s' % ((i + 1), ', '.join(stock_name_np[labels == i]),stock_name_np[cluster_center[i]]))  # 列出聚类后分类信息
                print('聚类类别 %i: %s ;;聚类中心是 %s\n' % ((i + 1),str(np.where(labels == i)), cluster_center[i]))  # 列出聚类后分类信息
                f.write('聚类类别 %i: %s ;;聚类中心是 %s\n' % ((i + 1), ', '.join(stock_name_np[labels == i]),stock_name_np[cluster_center[i]]))  # 列出聚类后分类信息
                f.write('聚类类别 %i: %s ;;聚类中心是 %s\n' % ((i + 1),str(np.where(labels == i)), cluster_center[i]))  # 列出聚类后分类信息
            f.close()
        cluster_tree_matrix = np.zeros([col_num, col_num], float)  # col_num * col_num 的矩阵！
        for leaf_node, cluster_index in enumerate(labels):
            center_node = cluster_center[cluster_index]
            if edge_orientation == 'to_center':
                cluster_tree_matrix[leaf_node][center_node] = 1
            else:
                cluster_tree_matrix[center_node][leaf_node] = 1
        if inter_cluster_edge_exist:
            for center_node_from in cluster_center:
                for center_node_to in cluster_center:
                    if center_node_from != center_node_to:
                        cluster_tree_matrix[center_node_from][center_node_to] = 1
        print("iter times:",iter_times)
        print("A_matrix:",A_matrix)
        print("R_matirx:",R_matirx)
        self.plot_matrix(A_matrix)
        self.plot_matrix(R_matirx)
        np_adj_to_df_link(cluster_tree_matrix,node_label='index',stock_dict=stock_dict)
        adj_address = self.boolean_network_data_folder + 'AP-adjmat.pickle'
        with open(adj_address, 'wb') as f:
            pickle.dump(cluster_tree_matrix, f)
            f.close()
        with open(self.boolean_network_data_folder+ result_save_name + '-ap_boolean_cluster_result_dict.pickle',  'wb') as f:
            pickle.dump(self.ap_cluster_result_dict, f)
            f.close()
        with open(self.boolean_network_data_folder + result_save_name + '-between_boolean_series_df.pickle', 'wb') as f:
            pickle.dump(boolean_series, f)
            f.close()

    # 测试用例
# code_list=['000001', '000002', '600601', '000012', '600612', '600651', '000009', '000568', '600660', '000004']
# start='2022/5/1';end='2022/5/30'
    def pickle_to_boolean_net_series(self,code_list=['000001'],start='2020/2/20',end='2022/2/20'):
        start_date = dt.datetime.strptime(start, '%Y/%m/%d')
        end_date = dt.datetime.strptime(end, '%Y/%m/%d')
        boolean_dict = {}
        boolean_series = pd.DataFrame()
        stock_name_list = []
        for i,stock_code in enumerate(code_list):
            with open(self.preprocessed_data_folder + stock_code + '.pickle', 'rb') as f:
                now_stock_dict = pickle.load(f)
                f.close()
            stock_name_list.append(now_stock_dict['stock_name'])
            now_stock_df = now_stock_dict['df']
            now_stock_df['index'] = now_stock_df['date']
            now_stock_df = now_stock_df.set_index('index')
            between_df = now_stock_df[now_stock_df.date.between(start_date, end_date)].copy()
            between_df['rose'] = between_df['close'] > between_df['open']  # 没涨归为跌
            between_df['boolean'] = between_df['rose'].apply(lambda x: 1 if x else 0)
            boolean_series[stock_code] = between_df['boolean'].copy()
        boolean_series.fillna(0, inplace=True)    # 没数据归为跌，放在这里因为在前面比较可以自动补充无数据，保持维度统一
        boolean_series = np.array(boolean_series)[:, :, np.newaxis]
        self.boolean_network_np = boolean_series
        boolean_dict['boolean_series'] = boolean_series
        boolean_dict['code_list'] = code_list
        boolean_dict['stock_name_list'] = stock_name_list
        with open(self.boolean_network_data_folder + 'boolean_dict.pickle', 'wb') as f:
            pickle.dump(boolean_dict, f)
            f.close()
        boolean_dict['boolean_series'] = boolean_series.tolist()
        print(self.boolean_network_np.shape)
        with open(self.boolean_network_data_folder+'bn.json', "w", encoding='utf-8') as f:
            # json.dump(dict_, f)  # 写为一行
            json.dump(boolean_dict, f, indent=2, sort_keys=True, ensure_ascii=False)  # 写为多行
            f.close()
        print('boolean_net creating processed!')

    def pickle_to_vibrate_net_series(self,code_list=['000001'],start='2020/2/20',end='2022/2/20'):
        start_date = dt.datetime.strptime(start, '%Y/%m/%d')
        end_date = dt.datetime.strptime(end, '%Y/%m/%d')
        vibrate_dict = {}
        vibrate_series = pd.DataFrame()
        stock_name_list = []
        for i,stock_code in enumerate(code_list):
            with open(self.preprocessed_data_folder + stock_code + '.pickle', 'rb') as f:
                now_stock_dict = pickle.load(f)
                f.close()
            stock_name_list.append(now_stock_dict['stock_name'])
            now_stock_df = now_stock_dict['df']
            now_stock_df['index'] = now_stock_df['date']
            now_stock_df = now_stock_df.set_index('index')
            between_df = now_stock_df[now_stock_df.date.between(start_date, end_date)].copy()
            between_df['vibrate'] = between_df['close'] - between_df['open']  # 一天之内的变化
            vibrate_series[stock_code] = between_df['vibrate'].copy()
        vibrate_series.fillna(0, inplace=True)    # 没数据归为变化为0，放在这里因为在前面比较可以自动补充无数据，保持维度统一
        vibrate_series = np.array(vibrate_series)[:, :, np.newaxis]
        self.vibrate_network_np = vibrate_series
        vibrate_dict['vibrate_series'] = vibrate_series
        vibrate_dict['code_list'] = code_list
        vibrate_dict['stock_name_list'] = stock_name_list
        with open(self.boolean_network_data_folder + 'vibrate_dict.pickle', 'wb') as f:
            pickle.dump(vibrate_dict, f)
            f.close()
        vibrate_dict['boolean_series'] = vibrate_series.tolist()
        print(self.boolean_network_np.shape)
        with open(self.boolean_network_data_folder+'vibrate.json', "w", encoding='utf-8') as f:
            # json.dump(dict_, f)  # 写为一行
            json.dump(vibrate_dict, f, indent=2, sort_keys=True, ensure_ascii=False)  # 写为多行
            f.close()
        print('boolean_net creating processed!')


    def pickle_to_ap_vibrate_adj_matrix(self,stock_dict={}, start='2020/2/20', end='2022/2/20', edge_orientation='to_center'
                                        ,inter_cluster_edge_exist=True,result_save_name='-rzrq-'):
        start_date = dt.datetime.strptime(start, '%Y/%m/%d')
        end_date = dt.datetime.strptime(end, '%Y/%m/%d')
        stock_code_np, stock_name_np = np.array(sorted(stock_dict.items())).T  # 将symbol_dict转换维（key,value）形式的列，并排序，然后转为2×50数组。最后进行拆包，返回两个numpy.array
        rose = []  # 实例化list，用于承载变化信息
        vibrate_series = pd.DataFrame()
        for stock_code in stock_code_np:
            with open(self.preprocessed_data_folder + stock_code + '.pickle', 'rb') as f:
                now_stock_dict = pickle.load(f)
                f.close()
                now_stock_df = now_stock_dict['df']
                now_stock_df['index'] = now_stock_df['date']
                now_stock_df = now_stock_df.set_index('index')
                between_df = now_stock_df[now_stock_df.date.between(start_date, end_date)].copy()
                between_df['vibrate'] = between_df['close'] - between_df['open']  # 一天之内的变化
                vibrate_series[stock_code] = between_df['vibrate'].copy()

        vibrate_series.fillna(0, inplace=True)    # 没数据归为变化为0，放在这里因为在前面比较可以自动补充无数据，保持维度统一
        self.vibrate_network_np_dict['vibrate_series'] = vibrate_series.copy()
        vibrate_series = np.array(vibrate_series)
        # Learn a graphical structure from the correlations
        edge_model = covariance.GraphicalLassoCV(cv=5)  # 实例化一个GraphicalLassoCV对象，关于Lasso见参考10，关于GraphicalLassoCV见参考14

        # standardize the time series: using correlations rather than covariance
        # is more efficient for structure recovery
        X = vibrate_series.copy()
        X /= X.std(axis=0)
        edge_model.fit(X)
        self.plot_similarity_matrix(edge_model.covariance_)
        print('similarity_matrix',edge_model.covariance_)
        cluster_center, labels, iter_times, A_matrix, R_matirx = cluster.affinity_propagation(edge_model.covariance_,random_state=0, return_n_iter=True)  # 返回划分好的聚类中心的索引和聚类中心的标签
        self.ap_cluster_result_dict = {'cluster_center': cluster_center, 'labels': labels, 'iter_times': iter_times,
                                  'A_matrix': A_matrix, 'R_matirx': R_matirx}
        n_labels = labels.max()  # 返回标签中的最大值，标签默认是数字递增形式的
        with open("vibrate_ap_cluster_result.txt",'w') as f:
            for i in range(n_labels + 1):
                print('聚类类别 %i: %s ;;聚类中心是 %s' % ((i + 1), ', '.join(stock_name_np[labels == i]),stock_name_np[cluster_center[i]]))  # 列出聚类后分类信息
                print('聚类类别 %i: %s ;;聚类中心是 %s\n' % ((i + 1),str(np.where(labels == i)), cluster_center[i]))  # 列出聚类后分类信息
                f.write('聚类类别 %i: %s ;;聚类中心是 %s\n' % ((i + 1), ', '.join(stock_name_np[labels == i]),stock_name_np[cluster_center[i]]))  # 列出聚类后分类信息
                f.write('聚类类别 %i: %s ;;聚类中心是 %s\n' % ((i + 1),str(np.where(labels == i)), cluster_center[i]))  # 列出聚类后分类信息
            f.close()
        col_num = edge_model.covariance_.shape[1]
        cluster_tree_matrix = np.zeros([col_num, col_num], float)  # col_num * col_num 的矩阵！
        for leaf_node, cluster_index in enumerate(labels):
            center_node = cluster_center[cluster_index]
            if edge_orientation == 'to_center':
                cluster_tree_matrix[leaf_node][center_node] = 1
            else:
                cluster_tree_matrix[center_node][leaf_node] = 1
        if inter_cluster_edge_exist:
            for center_node_from in cluster_center:
                for center_node_to in cluster_center:
                    if center_node_from != center_node_to:
                        cluster_tree_matrix[center_node_from][center_node_to] = 1
        print("iter times:",iter_times)
        print("A_matrix:",A_matrix)
        print("R_matirx:",R_matirx)
        self.plot_similarity_matrix(A_matrix)
        self.plot_similarity_matrix(R_matirx)
        np_adj_to_df_link(cluster_tree_matrix,node_label='index',stock_dict=stock_dict)
        adj_address = self.boolean_network_data_folder + 'AP-adjmat.pickle'
        with open(adj_address, 'wb') as f:
            pickle.dump(cluster_tree_matrix, f)
            f.close()
        with open(self.boolean_network_data_folder+result_save_name + '-ap_vibrate_cluster_result_dict.pickle',  'wb') as f:
            pickle.dump(self.ap_cluster_result_dict, f)
            f.close()
        with open(self.boolean_network_data_folder+result_save_name + '-between_vibrate_series_df.pickle', 'wb') as f:
            pickle.dump(self.vibrate_network_np_dict['vibrate_series'], f)
            f.close()

    def build_nn_for_boolean_net(self):
        with open("",'wb') as f:
            self.ap_cluster_result_dict = pickle.load(f)


if __name__ == '__main__':
    # code_list=['000001', '000002', '600601', '000012', '600612', '600651', '000009', '000568', '600660', '000004']
    start='2021/5/30';end='2022/5/30'
    class_code = 'fsls'
    do = TdxDataPreprocesser()
    new_stock_dict = do.get_stock_list_from_internet(re_request_list=True)
    # new_stock_dict = do.get_stock_list_from_file(re_request_list=True,filename='指数板块.txt',class_code=class_code)
    code_list = list(new_stock_dict.keys())
    print(len(code_list),code_list)
    do.pickle_to_ap_bn_adj_matrix(new_stock_dict, start, end,edge_orientation='to_leaf',result_save_name=class_code)
    do.pickle_to_ap_vibrate_adj_matrix(new_stock_dict, start, end,edge_orientation='to_leaf',result_save_name=class_code)
    # do.pickle_to_boolean_net_series(code_list, start, end)
    # do.tdx_data_to_pickle_files()
    talkContent("完成了完成了!")
