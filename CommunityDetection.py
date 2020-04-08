# -*- coding: utf-8 -*-
"""
完成敏感度的计算、社区检测、按照敏感度排序节点，最终生成特征列表
参数：
-b 良性应用调用图数据集的路径
-m 恶意应用调用图数据集的路径

功能：
1. apkDecompileDatasetToFeatureList()
   包装的顶层函数接口，多线程处理APK调用图数据集生成对应的特征数据集
   
2. oneApkToFeatureList()
   对调用图进行社交检测，然后提取出特征

3. getSenNodeCommunityList()
   读取一个调用图gexf文件，进行两次社区划分，得到敏感节点三维列表

4. getSortedCommunityList()
   对社区内节点按照敏感度进行排序
"""
import os
import argparse
import json
import shutil
import networkx as nx
import igraph as ig
from multiprocessing import Pool
from functools import partial

from Utils import getEntity
from SourceAndSink import readSourceAndSink, getEntityIdDict
from Statistics import getCOS, statisticsOfCommunity

# 将networkx格式的图转变为igraph格式，额外返回节点的索引和节点标签的字典
def networkxTograph(graph):
    G = graph.copy()
    mapping = dict(zip(G.nodes(),range(G.number_of_nodes())))
    reverse_mapping = dict(zip(range(G.number_of_nodes()),G.nodes()))
    G = nx.relabel_nodes(G,mapping)
    G_ig = ig.Graph(len(G), list(zip(*list(zip(*nx.to_edgelist(G)))[:2])))
    return G_ig, reverse_mapping, mapping 

# 对第一次划分的社团再进行一次划分, 并返回划分后社区中的敏感节点
def getSenNodeClusterList(graph):
    senNodeClusterList = []
    clusterList = []
    # 转变为igraph格式
    G_ig, idToLabelDict, labelToIdDict = networkxTograph(graph)
    # igraph-louvain算法进行社交网络检测划分
    vertexCluster = G_ig.community_multilevel()
    for cluster in vertexCluster:
        senNodeList = []
        nodeList = []
        for nodeId in cluster:
            node = idToLabelDict[nodeId]
            nodeList.append(node)
            if node.split(",")[-3] != "Normal":
                senNodeList.append(node)
        if len(senNodeList) != 0:
            senNodeClusterList.append(senNodeList)
            clusterList.append(nodeList)
    return senNodeClusterList, clusterList

# 读取一个调用图gexf文件，进行两次社区划分，得到敏感节点三维列表
# 第一维是第一次划分得到的大社区，第二维是大社区划分得到的小社区，第三维是小社区中的敏感节点列表
def getSenNodeCommunityList(graphGexfFilePath):
    # 读取调用图
    graph = nx.read_gexf(graphGexfFilePath)
    # 去除干扰节点
    superNodeList = ["{<java.lang.RuntimeException: void init(java.lang.String)>,Normal,,}"]
    graph.remove_nodes_from(superNodeList)
    # 转变为igraph格式
    G_ig, idToLabelDict, labelToIdDict = networkxTograph(graph)
    # igraph-louvain算法进行社交网络检测划分
    vertexCluster = G_ig.community_multilevel()
    # 将社区划分得到的子图提取到列表中
    communitySubgraphList = []
    # 将第一次社区划分得到社区进行第二次划分，并敏感节点提取到列表中
    senNodeCommunityList = []
    
    communityList = []
    for com in vertexCluster:
        iSSensitive = False
        communityNodesList = []
        for nodeId in com:
            node = idToLabelDict[nodeId]
            communityNodesList.append(node)
            if node.split(",")[-3] != "Normal":
                iSSensitive = True
        if iSSensitive == True:        
            communitySubgraph = graph.subgraph(communityNodesList).copy()
            communitySubgraphList.append(communitySubgraph)
            senNodeClusterList, clusterList  = getSenNodeClusterList(communitySubgraph)
            senNodeCommunityList.append(senNodeClusterList)
            communityList.append(clusterList)
    #print([len(c) for com in senNodeCommunityList for c in com ])  
    return senNodeCommunityList, communityList

# 对提取得到的敏感列表，按照敏感度进行排序
def getSortedCommunityList(senNodeCommunityList, entityToIdDict, idToCOSDict):
    # 最后排好序的列表
    sortedCommunityList = []
    # 临时变量保存内部排好序的大社区
    communityList = []
    # 对应大社区敏感度列表
    communityCOSList = []
    # 每个大社区内部进行排序
    for senNodeCommunity in senNodeCommunityList:
        # 对大社区中小社区内部进行排序
        community, communityCOS = getSortedClusterList(senNodeCommunity, 
                                                       entityToIdDict, 
                                                       idToCOSDict)
        # 对应敏感度和社区列表
        communityList.append(community)
        communityCOSList.append(communityCOS)   
    # 对内部排好序的大社区，按照大社区敏感度重新排序
    sortedCommunityTupleList = sorted(enumerate(communityList), 
                                      key=lambda x:communityCOSList[x[0]], 
                                      reverse=True)
    sortedCommunityList = [communityTuple[1] for communityTuple 
                           in sortedCommunityTupleList]
    return sortedCommunityList



# 对大社区内小社区进行排序
def getSortedClusterList(senNodeClusterList, entityToIdDict, idToCOSDict):
    # 最后排好序的小社区列表
    sortedClusterList = []
    # 临时变量保存内部排好序的小社区
    clusterList = []
    clusterCOSList = []
    # 大社区敏感度
    communityCOS = 0
    for senNodeCluster in senNodeClusterList:
        cluster, clusterCOS = getSortedCluster(senNodeCluster, 
                                               entityToIdDict, 
                                               idToCOSDict)
        clusterList.append(cluster)
        clusterCOSList.append(clusterCOS)
        # 小社区累加得到大社区的敏感度
        communityCOS += clusterCOS
    # 对内部排好序的小社区，按照小社区敏感度重新排序
    sortedClusterTupleList = sorted(enumerate(clusterList), 
                                    key=lambda x:clusterCOSList[x[0]], 
                                    reverse=True)
    sortedClusterList = [clusterTuple[1] for clusterTuple 
                           in sortedClusterTupleList]
    return sortedClusterList, communityCOS

# 对社区内节点按照敏感度进行排序
def getSortedCluster(senNodeCluster, entityToIdDict, idToCOSDict):
    cluster = []
    clusterCOS = 0
    for senNode in senNodeCluster:
        if isDigit(senNode):
            senNodeId = int(senNode)
        else:
            senNodeId = entityToIdDict[getEntity(senNode)]
        cluster.append(senNodeId)
        clusterCOS += idToCOSDict[senNodeId]
    sortedCluster = sorted(cluster, key=lambda senNodeId: idToCOSDict[senNodeId],
                           reverse=True)
    return sortedCluster, clusterCOS

# 判断字符串是否为整型数
def isDigit(string):
    try:
        digit=int(string)
        return isinstance(digit,int)
    except ValueError:
        return False

# 测试排序算法是否正确       
def testSorted():
    testList = [[['{<android.os.HandlerThread: android.os.Looper getLooper()>,Source,,NO_CATEGORY}', '{<android.os.Handler: boolean sendEmptyMessage(java.lang.Integer)>,Sink,,NO_CATEGORY}'], ['{<android.view.View: void setBackgroundColor(java.lang.Integer)>,Sink,,NO_CATEGORY}']], [['{<android.widget.TabHost$TabSpec: java.lang.String getTag()>,Source,,NO_CATEGORY}'], ['{<android.widget.TabHost$TabSpec: android.widget.TabHost$TabSpec setContent(android.widget.TabHost$TabContentFactory)>,Sink,,NO_CATEGORY}']]]
    sourceDict, sinkDict = readSourceAndSink()
    idToEntityDict, entityToIdDict = getEntityIdDict(sourceDict, sinkDict)
    idToCOSDict = {11140: 0.1, -7576: 0.2, -6181: 0.4, 13696: 0.3, -5086: 0.7}
    #预期输出结果
    # predicted results:
    # [[[-5086], [13696]], [[-6181], [-7576,11140]]]
    sortedTestList = getSortedCommunityList(testList, entityToIdDict, idToCOSDict)
    print(sortedTestList)

# 对调用图进行社交检测，然后提取出特征
def oneApkToFeatureList(apkDecompileDirPath, entityToIdDict, idToCOSDict=[], superNodeList=[]):
    # 判断目录下是否已经提取过特征，即之前已经进行过社交网络网络检测
    featureListJsonFilePath = os.path.join(apkDecompileDirPath, "featureList.json")
    comListJsonFilePath = os.path.join(apkDecompileDirPath, "comList.json")
    if not os.path.isfile(featureListJsonFilePath):
        # 调用图社交检测
        graphGexfFilePath = os.path.join(apkDecompileDirPath, "sourceGraph.gexf")
        senNodeCommunityList, communityList = getSenNodeCommunityList(graphGexfFilePath)
    else:
        # 加载之前提取的列表
        with open(featureListJsonFilePath, "r") as featureListJsonFile:
            senNodeCommunityList = json.load(featureListJsonFile)
        with open(comListJsonFilePath, "r") as comListJsonFile:
            communityList = json.load(comListJsonFile)
    # 按照敏感度进行排序
    if len(idToCOSDict) != 0:
        sortedSenNodeCommunityList = getSortedCommunityList(senNodeCommunityList, 
                                                            entityToIdDict, idToCOSDict)
    else:
        sortedSenNodeCommunityList = senNodeCommunityList
    # 提取的特征保存为featureList.json
    with open(featureListJsonFilePath, "w") as featureListJsonFile:
        json.dump(sortedSenNodeCommunityList, featureListJsonFile)
    
    with open(comListJsonFilePath, "w") as comListJsonFile:
        json.dump(communityList, comListJsonFile)
    
#    cleanDecompileDir(apkDecompileDirPath)

# 清理文件夹，只保留列表中的内容
def cleanDecompileDir(apkDecompileDirPath):
    remainList = ["featureList.json", "information.json"]
    for fileName in os.listdir(apkDecompileDirPath):
        if fileName not in remainList:
            filePath = os.path.join(apkDecompileDirPath, fileName)
            if os.path.isfile(filePath):
                os.unlink(filePath)
            if os.path.isdir(filePath):
                shutil.rmtree(filePath)

# 包装的顶层函数接口，多线程处理APK调用图数据集生成对应的特征数据集
def apkDecompileDatasetToFeatureList(apkDecompileDatesetDirPath, entityToIdDict, idToCOSDict=[]):
    apkDecompileDirPathList = []
    for apkDecompileDirBasename in os.listdir(apkDecompileDatesetDirPath):
        apkDecompileDirPath = os.path.join(apkDecompileDatesetDirPath, apkDecompileDirBasename)
        if os.path.isdir(apkDecompileDirPath):
            apkDecompileDirPathList.append(apkDecompileDirPath)
    pool = Pool()
    pool.map(partial(oneApkToFeatureList,entityToIdDict=entityToIdDict, idToCOSDict=idToCOSDict), 
             apkDecompileDirPathList)
    
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Call Graphs To Feature List.")
    parser.add_argument("-b", "--benign_graph_dateset", help="a benign directory with the APKs Call Graph to analyze.", type=str, required=True)
    parser.add_argument("-m", "--malware_graph_dateset", help="a malware directory with the APKs Call Graph to analyze.", type=str, required=True)
    args = parser.parse_args()
    
    benignDecompileDatesetDirPath = args.benign_graph_dateset
    malwareDecompileDatesetDirPath = args.malware_graph_dateset
    
    if not os.path.isdir(benignDecompileDatesetDirPath):
        raise OSError("Benign graph dataset directory %s does not exist." % (args.benign_graph_dateset))

    if not os.path.isdir(malwareDecompileDatesetDirPath):
        raise OSError("Benign graph dataset directory %s does not exist." % (args.malware_graph_dateset))
    
    sourceDict, sinkDict = readSourceAndSink()
    idToEntityDict, entityToIdDict = getEntityIdDict(sourceDict, sinkDict)
    
    # 获取敏感度
    print("Get coefficient of sensitivity...")
    idToCOSDict, entityToCOSDict = getCOS(benignDecompileDatesetDirPath, 
                                          malwareDecompileDatesetDirPath, 
                                          idToEntityDict, entityToIdDict)
    # 社交网络检测，并对社区按照敏感度进行排序，提取特征列表
    print("Extract benign dataset feature list...")
    apkDecompileDatasetToFeatureList(benignDecompileDatesetDirPath, entityToIdDict, idToCOSDict)
    print("Extract malware dataset feature list...")
    apkDecompileDatasetToFeatureList(malwareDecompileDatesetDirPath, entityToIdDict, idToCOSDict)
    
    # 对社交网络的检测结果进行统计分析（正式流程可省略）
    print("Benign dataset community statistics...")
    statisticsOfCommunity(benignDecompileDatesetDirPath, isMalware=False)
    
    print("Malware dataset community statistics...")
    statisticsOfCommunity(malwareDecompileDatesetDirPath)