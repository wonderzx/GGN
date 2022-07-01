'''
    ind1 ind2         C
11     M    B  0.344116
16     A    C       0.0
17     B    C  0.044794
22     H    C  0.051657
25     K    C  0.022004
...前两列为拓扑端点，最后一列为权重（权重取值范围为0 ~ 1）
有向图

nx.from_numpy_matrix(np.array(data), create_using=nx.DiGraph)

无向图

nx.from_numpy_matrix(np.array(data))
'''

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

namespace = globals()


def directed_graph(source, title,has_weight=False):
    # graph type
    G1 = nx.DiGraph()
    source = source.reset_index(drop=True)

    # load
    for i in range(len(source)):
        code1 = source.loc[i, 'ind1']
        code2 = source.loc[i, 'ind2']
        w = source.loc[i, title]
        G1.add_edge(str(code1), str(code2), weight=w)

    # weight classify
    for i in [0, 20, 40, 60, 80]:
        ii = i / 100
        if has_weight :
            namespace['E%d' % (i)] = [(u, v) for (u, v, d) in G1.edges(data=True) if (d['weight'] >= ii) & (d['weight'] < ii + 0.2)]
        # 有权重
        else:
            namespace['E%d' % (i)] = [(u, v) for (u, v, d) in G1.edges(data=True)] # 无权重

    # position
    pos = nx.shell_layout(G1)
    plt.rcParams['figure.figsize'] = (10, 10)

    # nodes
    nx.draw_networkx_nodes(G1, pos, node_size=1000, alpha=0.4,
                           node_color='dodgerblue', node_shape='o')

    # lines
    for i in [0, 20, 40, 60, 80]:
        ii = i / 100 + 0.05
        nx.draw_networkx_edges(G1, pos,
                               edgelist=namespace['E%d' % (i)],
                               width=5, edge_color='dodgerblue', alpha=ii,
                               arrowstyle="->", arrowsize=50)

    # texts
    nx.draw_networkx_labels(G1, pos, font_size=25,
                            font_family='sans-serif',
                            font_color='k')

    # show and save
    plt.axis('off')
    plt.show()
    plt.savefig(title + ".png")


if __name__ == '__main__':
    df = pd.read_csv("./np_adj_to_df_link.csv")
    directed_graph(df, 'C')
