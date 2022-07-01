import os
import bs4 as bs # 导入 beautiful soup4 包，用于抓取网页信息
import requests # 导入 request 用于获取网站上的源码
import numpy as np
import pandas as pd
import datetime as dt
import time
from sklearn import cluster, covariance, manifold
import pickle
import json


# 通达信数据预处理工具
class TdxDataPreprocesser():
    # 专门用来处理通达信导出数据的类
    def __init__(self):
        self.original_data_folder = './day/'
        self.preprocessed_data_folder = './tdx_pickle/'
        self.boolean_network_data_folder = './bn_json/'
        self.boolean_network_np = None
        self.data_folder_stock_code_list = []
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

    def get_stock_list(self, re_request_list=False):
        if re_request_list:
            resp = requests.get('https://q.stock.sohu.com/cn/bk_4507.shtml')
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

    def pickle_to_ap_bn_adj_matrix(self,code_list=['000001'],start='2020/2/20',end='2022/2/20'):
        start_date = dt.datetime.strptime(start, '%Y/%m/%d')
        end_date = dt.datetime.strptime(end, '%Y/%m/%d')
        stock_dict = self.get_stock_list(re_request_list=False)  # SSE50，dict结构
        stock_code_np, stock_name_np = np.array(sorted(stock_dict.items())).T  # 将symbol_dict转换维（key,value）形式的列，并排序，然后转为2×50数组。最后进行拆包，返回两个numpy.array
        rose = []  # 实例化list，用于承载“报价”
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

        open_prices = np.vstack([q['boolean'] for q in rose])
        close_prices = np.vstack([q['boolean'] for q in rose])  # 通过numpy聚合功能，获取一个n行1列的数组，每个记录是一个股票的收盘价的list
        # The daily variations of the quotes are what carry most information
        variation = close_prices - open_prices  # 收盘价-开盘价，作为信息载体
        # #############################################################################
        # 从相关性矩阵中获得一个相关性网络
        edge_model = covariance.GraphicalLassoCV(cv=5)  # 实例化一个GraphicalLassoCV对象

        # standardize the time series: using correlations rather than covariance
        # is more efficient for structure recovery
        X = variation.copy().T
        X /= X.std(axis=0)
        edge_model.fit(X)

        # #############################################################################
        # 聚类划分

        _, labels = cluster.affinity_propagation(edge_model.covariance_)  # 返回划分好的聚类中心的索引和聚类中心的标签
        n_labels = labels.max()  # 返回标签中的最大值，标签默认是数字递增形式的

        for i in range(n_labels + 1):  # 此处是[0,1,2)
            print('Cluster %i: %s' % ((i + 1), ', '.join(stock_name_np[labels == i])))  # 列出聚类后分类信息

        # #############################################################################
        # 为可视化找到一个低维度的嵌入:在二维平面上找到节点(股票)的最佳位置
        # 使用稠密的特征解算器来实现再现性(arpack是由不控制的随机向量启动的)。此外，我们使用大量的邻居捕捉大规模的结构。
        node_position_model = manifold.LocallyLinearEmbedding(
            n_components=2, eigen_solver='dense', n_neighbors=6)  # 近邻选6个，降维后得到2个

        embedding = node_position_model.fit_transform(X.T).T  # 训练模型并执行降维，返回降维后的样本集

        # Display a graph of the partial correlations
        partial_correlations = edge_model.precision_.copy()  # 偏相关分析
        d = 1 / np.sqrt(np.diag(partial_correlations))
        partial_correlations *= d
        partial_correlations *= d[:, np.newaxis]  # 转为n*1结构的二维数组
        non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)  # 取上三角矩阵，判断与0.02大小获取True/False布尔值

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
        boolean_series.fillna(0, inplace=True)
        boolean_series = np.array(boolean_series)[:, :, np.newaxis]
        self.boolean_network_np = boolean_series
        boolean_dict['boolean_series'] = boolean_series.tolist()
        boolean_dict['code_list'] = code_list
        boolean_dict['stock_name_list'] = stock_name_list
        print(self.boolean_network_np.shape)
        with open(self.boolean_network_data_folder+'bn.json', "w", encoding='utf-8') as f:
            # json.dump(dict_, f)  # 写为一行
            json.dump(boolean_dict, f, indent=2, sort_keys=True, ensure_ascii=False)  # 写为多行
            f.close()
        with open(self.boolean_network_data_folder + 'bn.pickle', 'wb') as f:
            pickle.dump(boolean_dict['boolean_net'], f)
            f.close()
        print('boolean_net creating processed!')


if __name__ == '__main__':
    # code_list=['000001', '000002', '600601', '000012', '600612', '600651', '000009', '000568', '600660', '000004']
    start='1990/5/1';end='2022/5/30'
    do = TdxDataPreprocesser()
    new_stock_dict = do.get_stock_list()
    code_list = list(new_stock_dict.keys())
    do.pickle_to_boolean_net_series(code_list, start, end)
    # do.tdx_data_to_pickle_files()
