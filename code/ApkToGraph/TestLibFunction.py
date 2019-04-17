# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 16:06:55 2018
测试各种库函数的用法
@author: ZhangXin
"""
######################################################
# test gexf format

#import networkx as nx
#""" Create a graph with three nodes"""
#graph = nx.Graph()
#graph.add_node('red')
#graph.add_node('green')
#graph.add_node('blue')
#""" Add color data """
#graph.nodes['red']['viz'] = {'color': {'r': 255, 'g': 0, 'b': 0, 'a': 0}}
#graph.nodes['green']['viz'] = {'color': {'r': 0, 'g': 255, 'b': 0, 'a': 0}}
#graph.nodes['blue']['viz'] = {'color': {'r': 0, 'g': 0, 'b': 255, 'a': 0}}
#""" Write to GEXF """
## Use 1.2draft so you do not get a deprecated warning in Gelphi
#nx.write_gexf(graph, "file.gexf")
#graph = nx.read_gexf("file.gexf")
#nx.write_gexf(graph, "newFile.gexf")




######################################################
# test connected_components

#import matplotlib.pyplot as plt
#import networkx as nx
#G=nx.path_graph(4)
#G.add_path([10,11,12])
#nx.draw(G,with_labels=True,label_size=1000,node_size=1000,font_size=20)
#plt.show()
#
#connectedGraph = sorted(nx.connected_components(G), key=len, reverse=True)
#print(connectedGraph)





####################################################
# test python-louvain

#import community
#import networkx as nx
#import matplotlib.pyplot as plt
#
#better with karate_graph() as defined in networkx example.
#erdos renyi don't have true community structure
#G = nx.erdos_renyi_graph(30, 0.05)
#print("G:")
#print(type(G))
#print(G)
#
##first compute the best partition
## partition is a dict with node_id to community_id.
#partition = community.best_partition(G)
#print("partition:")
#print(type(partition))
#print(partition)
#
#
##drawing 
#size = float(len(set(partition.values()))) # number of communities
#print("size: %s" % size)
#
#pos = nx.spring_layout(G)
#count = 0.
#for com in set(partition.values()) :
#    count = count + 1.
#    list_nodes = [nodes for nodes in partition.keys()
#                                if partition[nodes] == com]
#    nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 20,
#                                node_color = str(count / size))
#
#
#nx.draw_networkx_edges(G, pos, alpha=0.5)
#plt.show()




#######################################################
# test json

#import json
#from collections import OrderedDict
#
#test_dict = OrderedDict()
#test_dict[3] = "A"
#test_dict[2] = "B"
#test_dict[1] = "C"
#print(test_dict)
#print(type(test_dict))
#
##dumps
#print("dumps:")
#
#json_str = json.dumps(test_dict)
#print(json_str)
#print(type(json_str))
#
##loads
#print("loads:")
#new_dict = json.loads(json_str, object_pairs_hook=OrderedDict)
#print(new_dict)
#print(type(new_dict))
#
#new_dict[0] = "D"
#
#print(new_dict["3"])
#
##dump
#with open("test_info.json","w") as f:
#    json.dump(new_dict,f)
#    
##load
#print("load:")
#with open("test_info.json","r") as load_f:
#    load_dict = json.load(load_f, object_pairs_hook=OrderedDict)
#    
#print(load_dict)
#print(type(load_dict))




##########################################################
# test DBSCAN

#import numpy as np
#import matplotlib.pyplot as plt
#from sklearn import datasets
#import sklearn.cluster as skc
##matplotlib inline
#X1, y1=datasets.make_circles(n_samples=5000, factor=.6,
#                                      noise=.05)
#
#print(type(X1))
#print(X1.shape)
#print(X1)
#X2, y2 = datasets.make_blobs(n_samples=1000, n_features=2, centers=[[1.2,1.2]], cluster_std=[[.1]],
#               random_state=9)
#print(X2.shape)
#print(X2)
#X = np.concatenate((X1, X2))
#print(X.shape)
#plt.scatter(X[:, 0], X[:, 1], marker='o')
#plt.show()
#
#y_pred = skc.DBSCAN(eps = 0.1, min_samples = 10).fit_predict(X)
#print(y_pred)
#plt.scatter(X[:, 0], X[:, 1], c=y_pred)
#plt.show()
#
##db = skc.DBSCAN(eps = 0.1, min_samples = 10).fit(X)
##print(db.labels_)



##########################################################
# test multiprocessing
#
#from pathos.multiprocessing import ProcessingPool as Pool
#
#def work(x, y):
#    return x + y
#
#x = [1,2,3,4,5,6]
#y = [1,1,1,1,1,1]
#pool = Pool(4)
#results = pool.map(work, x, y)
#print(results)



##########################################################
# test boxplot

#import numpy as np
#import matplotlib.pyplot as plt
#np.random.seed(100)#生成随机数
#data=np.random.normal(size=(1000,4),loc=0,scale=1) #1000个值得4维数组
#lables = ['A','B','C','D']
#plt.boxplot(data,labels=lables,sym='o',whis=1.5)
#plt.show()


##########################################################
# test sort
#com0 = [0, 0, 1]
#com1 = [4, 5, 6]
#com2 = [1, 2, 3]
#
#comCOS = [1.0, 15.0, 6.0]
#comList = [com0, com1, com2]
#
#sortedcomTupleList = sorted(enumerate(comList), key=lambda comTuple:comCOS[comTuple[0]], reverse=True)
#
#sortedComList = [comTuple[1] for comTuple in sortedcomTupleList]
#
#print(sortedComList)


###########################################################
# test json list
#
#import json
#
#test_list = [[[1, 2], [4]], [[3], [5, 6]], [[7]]]
#
#print("dump:")
#
#with open("test_list.json","w") as f:
#    json.dump(test_list,f)
#
#print("load:")
#
#with open("test_list.json","r") as load_f:
#    load_list = json.load(load_f)
#    
#    
#print(load_list)
#print(load_list[0])
#print(type(load_list[0]))

##########################################################
# test matplotlib hist

#import matplotlib.pyplot as plt
#import numpy as np
#import matplotlib
#
## 设置matplotlib正常显示中文和负号
#matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
#matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号
## 随机生成（10000,）服从正态分布的数据
#test_list = [24, 28, 24, 22, 14, 6, 11, 14, 7, 9, 14, 9, 4, 1, 6, 8, 7, 1, 5, 2, 5, 3, 2, 3, 4, 3, 2, 2, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 19, 19, 9, 13, 25, 18, 20, 10, 18, 9, 13, 7, 2, 8, 5, 7, 3, 7, 8, 6, 6, 6, 7, 1, 2, 9, 1, 2, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 18, 18, 19, 19, 16, 17, 15, 11, 3, 16, 10, 6, 7, 7, 5, 6, 3, 3, 10, 4, 3, 5, 5, 6, 5, 2, 3, 1, 1, 1, 2, 2, 3, 2, 4, 3, 1, 1, 2, 2, 11, 13, 14, 6, 12, 9, 9, 6, 3, 6, 6, 6, 4, 4, 2, 3, 4, 7, 6, 3, 4, 2, 3, 3, 1, 4, 3, 3, 1, 2, 2, 1, 1, 1, 36, 10, 28, 27, 21, 34, 17, 18, 9, 12, 10, 14, 11, 12, 8, 7, 12, 6, 6, 9, 5, 4, 3, 6, 5, 3, 4, 2, 2, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 4, 1, 1, 1, 1]
#data = np.array(test_list)
#"""
#绘制直方图
#data:必选参数，绘图数据
#bins:直方图的长条形数目，可选项，默认为10
#normed:是否将得到的直方图向量归一化，可选项，默认为0，代表不归一化，显示频数。normed=1，表示归一化，显示频率。
#facecolor:长条形的颜色
#edgecolor:长条形边框的颜色
#alpha:透明度
#"""
#plt.hist(data, bins=40, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
## 显示横轴标签
#plt.xlabel("区间")
## 显示纵轴标签
#plt.ylabel("频数/频率")
## 显示图标题
#plt.title("频数/频率分布直方图")
#plt.show()

##########################################################
# test matplotlib hist

#import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
#
#x = [0,5,9,10,15]
#y = [0,1,2,3,4]
#
#tick_spacing = 1
## tick_spacing = 5
##通过修改tick_spacing的值可以修改x轴的密度
##1的时候1到16，5的时候只显示几个
#fig, ax = plt.subplots(1,1)
#ax.plot(x,y)
#ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
#plt.show()



##########################################################
# test pandas

#import pandas as pd
#import numpy as np
#
#list1 = [1, 2, 3, 4, 100]
#list2 = [6, 7, 8, 900]
#
#def status(x) : 
#    return pd.Series([x.count(),x.min(),x.quantile(.25),x.median(),
#                      x.quantile(.75),x.mean(),x.max(),x.mad(),x.var(),x.std()],
#                      index=['数目','最小值','25%分位数','中位数','75%分位数','均值',
#                             '最大值','平均绝对偏差','方差','标准差'])
#
#dataList = [list1, list2]
#print(pd.Series(list1))    
#print(status(pd.Series(list1)))
#
#df = pd.DataFrame(dataList)
#df = df.T
#print(df)
#
#
#df_des = df.describe()
#print(df_des)
#df_des.columns = ["list1", "list2"]
#df_des.index = ['count', 'mean', 'std', 'min', '25%', 'median', '75%', 'max']
#
#print(df_des)
#
##df_status = df.apply(status)
##print(df_status)



##########################################################
# test torch load



