# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 17:08:42 2019
探索不同的方法的效率和准确性
聚类方式
1. 直接计算敏感节点间距离进行聚类
2. 先进行社区聚类，分出子图，计算子图内敏感节点间的聚类进行聚类


排列方式
1. 按照敏感性大小排列
2. 把最大敏感度的放在最前面，然后按照点间距离排序
3. 把最大敏感度的放在最前面，然后加权敏感度和点间距离排序
@author: ZhangXin
"""
import os
import community
import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import sklearn.cluster as skc
from sklearn import metrics
from multiprocessing import Pool
from functools import partial

from Statistics import getEntity, statistics, getCoefficientOfSensitivityDict
from SourceAndSink import readSourceAndSink, getEntityIdDict

def getSenNodeCommunityList(graphGexfFilePath, superNodeList=[]):
    communitySubgraphList = []
    graph = nx.read_gexf(graphGexfFilePath)
    undirectedGraph = graph.to_undirected()
    # partitionDict : {nodeId : communityId}
    partitionDict = community.best_partition(undirectedGraph)    # hyperparameter1
    numOfCommunities = len(set(partitionDict.values()))
#    print("number of communities: %s" % numOfCommunities)
    
    for communityId in range(numOfCommunities):
        communityNodesList = [nodes for nodes in partitionDict.keys()
                                    if partitionDict[nodes] == communityId]
        """
        TODO
        生成图耗时大 需要优化 不含敏感节点的社区 不用生成图
        """
        communitySubgraph = graph.subgraph(communityNodesList).copy()
        communitySubgraphList.append(communitySubgraph)
    
    
    senNodeCommunityList = []
    for i, communitySubgraph in enumerate(communitySubgraphList):
#        print("*" * 30)
#        numOfSenNodes = len(getSenNodeListInGraph(communitySubgraph))
#        if numOfSenNodes:
#            communitySubgraphGexfFilePath = os.path.join(os.path.split(graphGexfFilePath)[0],
#                                                         "communitySubgraph" + str(i) + "_" + str(numOfSenNodes)+ ".gexf")
#            nx.write_gexf(communitySubgraph, communitySubgraphGexfFilePath)
        senNodeClusterList  = getSenNodeClusterList(communitySubgraph)
        if len(senNodeClusterList) != 0:
            senNodeCommunityList.append(senNodeClusterList)
    
    return senNodeCommunityList

    
def getSenNodeClusterList(graph):
    """
    聚类耗时巨大,继续用社区发现算法,不用聚类的方法
    """
    
    senNodeClusterList = []
        
    undirectedGraph = graph.to_undirected()
    partitionDict = community.best_partition(undirectedGraph, resolution = 0.5)
    
    numOfClusters = len(set(partitionDict.values()))

    for clusterId in range(numOfClusters):
        cluster = [node for node in partitionDict.keys() 
                                    if (partitionDict[node] == clusterId and isSenNode(node))]
        if len(cluster) != 0:
            senNodeClusterList.append(cluster)
            
        
        # find super node!
        if len(cluster) > 16:
            findSuperNode(cluster, undirectedGraph)
         
#    print(senNodeClusterList)
    return senNodeClusterList


def findSuperNode(cluster, graph):
    neighbourNodeDict = dict()
    
    for senNode in cluster:
        for neighbourNode in list(graph.adj[senNode]):
            if neighbourNode not in neighbourNodeDict.keys():
                neighbourNodeDict[neighbourNode] = 1
            else:
                neighbourNodeDict[neighbourNode] += 1
    maxCount = 0
    maxCountNodeLabel = ""
    for node, count in neighbourNodeDict.items():
        if count > maxCount:
            maxCountNodeLabel = node
            maxCount = count
    
    writeStringLine = str(maxCountNodeLabel) + " " + str(maxCount) + " " + str(len(cluster)) + "\n"
    
    statisticsSuperNodeFilePath = "DataStatistics/statisticsSuperNode.txt"
    with open(statisticsSuperNodeFilePath, "a+") as statisticsSuperNodeFile:
        statisticsSuperNodeFile.write(writeStringLine)
    


#def getSenNodeClusterList(graph):
#    
#    senNodeClusterList = []
#    
#    senNodeList = getSenNodeListInGraph(graph)
#    numOfSenNodes = len(senNodeList)
##    print("number of sensitive nodes in graph: %s" % numOfSenNodes)
#    
#    if numOfSenNodes == 0:
##        print("This graph does not contain sensitive nodes!")
#        return senNodeClusterList
#    
#    if numOfSenNodes == 1:
#        senNodeClusterList.append(senNodeList)
#        return senNodeClusterList
#    
#    X = np.zeros([numOfSenNodes, numOfSenNodes], dtype=int)
#    
#    sumOfShortestPathLength = 0
#    
#    """
#    TODO
#    耗时巨大,需要在这一步之前进一步减小图的大小 考虑把社区数变多 减小社区大小 去掉不含敏感节点的社区
#    或者继续用社区发现算法 不用聚类的方法
#    """
#    for i, senSourceNode in enumerate(senNodeList):
#        for j, senTargetNode in enumerate(senNodeList):
#            if i < j:
#                shortestPathLength = getShortestPathLength(graph, 
#                                                           senSourceNode, senTargetNode)
#            elif i == j:
#                shortestPathLength = 0
#            else:
#                shortestPathLength = X[j, i]
##            shortestPathLength = getShortestPathLength(graph, 
##                                                       senSourceNode, senTargetNode)
#            X[i, j] = shortestPathLength
#            sumOfShortestPathLength += shortestPathLength
#    
#    averageShortestPathLength = sumOfShortestPathLength / (numOfSenNodes ** 2)
##    print(X)
##    print("averageShortestPathLength = %s" % averageShortestPathLength)
#    db=skc.DBSCAN(eps=averageShortestPathLength * 1.5, min_samples=1).fit(X) # hyperparameter2
#    labels = db.labels_
#    
##    print("Labels:")
##    print(labels)
#    
##    raito=len(labels[labels[:] == -1]) / len(labels)
##    print('Noise raito:', format(raito, '.2%'))
##    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#    numOfClusters = len(set(labels))
##    print('Estimated number of clusters: %d' % numOfClusters)
##    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))
#    for i in range(numOfClusters):
#        senNodeCluster = [senNodeList[j] for j, label in enumerate(labels) if label == i ]
#        senNodeClusterList.append(senNodeCluster)
#        
##    print(senNodeClusterList)
#    return senNodeClusterList

            
def getSenNodeListInGraph(graph):
    senNodeList = []
    for node in graph:
        nodeType = node.split(",")[-3]
        if nodeType == "Sink" or nodeType == "Source":
            senNodeList.append(node)
    return senNodeList

def isSenNode(nodeLabel):
    nodeType = nodeLabel.split(",")[-3]
    if nodeType == "Sink" or nodeType == "Source":
        return True
    else:
        return False

#def getSenNodeIdListInGraph(graph, entityToIdDict):
#    senNodeIdList = []
#    for node in graph:
#        nodeType = node.split(",")[-3]
#        if nodeType == "Sink" or nodeType == "Source":
#            senNodeId = getEntity(node)
#            senNodeIdList.append(senNodeId)
#    return senNodeIdList    

def getShortestPathLength(graph, sourceNode, targetNode):
    try:
        length = nx.shortest_path_length(graph.to_undirected(), 
                                         source=sourceNode, target=targetNode)
    except nx.NetworkXNoPath:
        length = graph.number_of_nodes()
    return length 


def getSortedCommunityList(senNodeCommunityList, entityToIdDict, idToCOSDict):
    sortedCommunityList = []
    communityCOSList = []
    communityList = []
    
    for senNodeCommunity in senNodeCommunityList:
        community, communityCOS = getSortedClusterList(senNodeCommunity, 
                                                       entityToIdDict, 
                                                       idToCOSDict)
        communityList.append(community)
        communityCOSList.append(communityCOS)
    
#    print("communityList:")
#    print(communityList)
#    print("communityCOSList:")
#    print(communityCOSList)  
    sortedCommunityTupleList = sorted(enumerate(communityList), 
                                      key=lambda x:communityCOSList[x[0]], 
                                      reverse=True)

#    print("sortedCommunityTupleList:")
#    print(sortedCommunityTupleList)
    sortedCommunityList = [communityTuple[1] for communityTuple 
                           in sortedCommunityTupleList]
    
#    print("sortedCommunityList:")
#    print(sortedCommunityList)
    return sortedCommunityList



def getSortedClusterList(senNodeClusterList, entityToIdDict, idToCOSDict):
    sortedClusterList = []
    clusterCOSList = []
    clusterList = []
    communityCOS = 0
    
    # clusterCOSList
    # clusterList
    for senNodeCluster in senNodeClusterList:
        cluster, clusterCOS = getSortedCluster(senNodeCluster, 
                                               entityToIdDict, 
                                               idToCOSDict)
        clusterList.append(cluster)
        clusterCOSList.append(clusterCOS)
        communityCOS += clusterCOS
    
#    print("clusterList:")
#    print(clusterList)
#    print("clusterCOSList:")
#    print(clusterCOSList)    
#    print("communityCOS:")
#    print(communityCOS)
    sortedClusterTupleList = sorted(enumerate(clusterList), 
                                    key=lambda x:clusterCOSList[x[0]], 
                                    reverse=True)
#    print("sortedClusterTupleList:")
#    print(sortedClusterTupleList)
    sortedClusterList = [clusterTuple[1] for clusterTuple 
                           in sortedClusterTupleList]
#    print("sortedClusterList:")
#    print(sortedClusterList)
    return sortedClusterList, communityCOS



def getSortedCluster(senNodeCluster, entityToIdDict, idToCOSDict):
    
    cluster = []
    clusterCOS = 0
    for senNode in senNodeCluster:
        senNodeId = entityToIdDict[getEntity(senNode)]
        cluster.append(senNodeId)
        clusterCOS += idToCOSDict[senNodeId]
#    print("cluster:")
#    print(cluster)
#    print("clusterCOS:")
#    print(clusterCOS)
    sortedCluster = sorted(cluster, key=lambda senNodeId: idToCOSDict[senNodeId],
                           reverse=True)
    
#    print("sortedCluster:")
#    print(sortedCluster)
    return sortedCluster, clusterCOS
       
def testSorted():
    
    # testList shape
    # [ [ [ *, *],    
    #     [*]     ],
    #   [ [*],   
    #     [*]     ]
    #               ]
    testList = [[['{<android.os.HandlerThread: android.os.Looper getLooper()>,Source,,NO_CATEGORY}', '{<android.os.Handler: boolean sendEmptyMessage(java.lang.Integer)>,Sink,,NO_CATEGORY}'], ['{<android.view.View: void setBackgroundColor(java.lang.Integer)>,Sink,,NO_CATEGORY}']], [['{<android.widget.TabHost$TabSpec: java.lang.String getTag()>,Source,,NO_CATEGORY}'], ['{<android.widget.TabHost$TabSpec: android.widget.TabHost$TabSpec setContent(android.widget.TabHost$TabContentFactory)>,Sink,,NO_CATEGORY}']]]
    # print(testList)
    sourceDict, sinkDict = readSourceAndSink()
    idToEntityDict, entityToIdDict = getEntityIdDict(sourceDict, sinkDict)
#    for com in testList:
#        for cluster in com:
#            for node in cluster:
#                print(entityToIdDict[getEntity(node)])
    idToCOSDict = {11140: 0.1, -7576: 0.2, -6181: 0.4, 13696: 0.3, -5086: 0.7}
    # predicted results:
    # [[[-5086], [13696]], [[-6181], [-7576,11140]]]
    
    sortedTestList = getSortedCommunityList(testList, entityToIdDict, idToCOSDict)
#    print("*" * 10)
    print(sortedTestList)


def oneApkToFeatureList(apkDecompileDirPath, entityToIdDict, idToCOSDict, superNodeList=[]):
    graphGexfFilePath = os.path.join(apkDecompileDirPath, "maxSensitiveSubGraph.gexf")
    senNodeCommunityList = getSenNodeCommunityList(graphGexfFilePath)
    sortedSenNodeCommunityList = getSortedCommunityList(senNodeCommunityList, 
                                                        entityToIdDict, idToCOSDict)
    featureListJsonFilePath = os.path.join(apkDecompileDirPath, "featureList.json")
    
    with open(featureListJsonFilePath, "w") as featureListJsonFile:
        json.dump(sortedSenNodeCommunityList, featureListJsonFile)
        


        
def apkDecompileDatasetToFeatureList(apkDecompileDatesetDirPath, entityToIdDict, idToCOSDict):
    apkDecompileDirPathList = []
    
    for apkDecompileDirBasename in os.listdir(apkDecompileDatesetDirPath):
        apkDecompileDirPath = os.path.join(apkDecompileDatesetDirPath, apkDecompileDirBasename)
        if os.path.isdir(apkDecompileDirPath):
            apkDecompileDirPathList.append(apkDecompileDirPath)
    pool = Pool()
    pool.map(partial(oneApkToFeatureList,entityToIdDict=entityToIdDict, idToCOSDict=idToCOSDict), 
             apkDecompileDirPathList)
    
            
if __name__ == "__main__":
    senNodeCommunityList = getSenNodeCommunityList("test\\maxSensitiveSubGraph.gexf")
    print(senNodeCommunityList)
    # testSorted()  