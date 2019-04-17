# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 16:42:39 2019

@author: ZhangXin
"""
import os
import re
import json
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
import pandas as pd

from SourceAndSink import readSourceAndSink, getEntityIdDict


def statistics(apkDecompileDatesetDirPath, idToEntityDict, entityToIdDict, 
               isMalware=True):
    
    # "SG" means "source graph" 
    # "SSG" means "sensitive subgraph"
    # "MSSG" means "max sensitive subgraph", the max weakly connected component subgraph of sensitive subgraph
    
    MSSGNodePercentageList = []
    MSSGEdgePercentageList = []
    MSSGSenNodePercentageList = []
    
    idToCountDict = dict()
    for id in idToEntityDict.keys():
        idToCountDict[id] = 0
    
    numOfApks = 0
    for apkDecompileDir in os.listdir(apkDecompileDatesetDirPath):
        infoJsonFilePath = os.path.join(apkDecompileDatesetDirPath, 
                                        apkDecompileDir, "information.json")
        if os.path.isfile(infoJsonFilePath):
            numOfApks += 1
            infoOrderedDict = getOrderedDictFromJsonFile(infoJsonFilePath)
            percentageOfNodeMSSGToSG = (infoOrderedDict["numOfMSSGNodes"] / 
                                        infoOrderedDict["numOfSGNodes"])
            percentageOfEdgeMSSGToSG = (infoOrderedDict["numOfMSSGEdges"] / 
                                        infoOrderedDict["numOfSGEdges"])
            percentageofSenNodeMSSGToSG = (infoOrderedDict["numOfSenNodesInMSSG"] / 
                                           infoOrderedDict["numOfSenNodesInSG"])
            MSSGNodePercentageList.append(percentageOfNodeMSSGToSG)
            MSSGEdgePercentageList.append(percentageOfEdgeMSSGToSG)
            MSSGSenNodePercentageList.append(percentageofSenNodeMSSGToSG)
            
            sensitiveNodeDict = infoOrderedDict["sensitiveNodeDict"]
            
            sourceAPICount, sinkAPICount = 0, 0
            for (sensitiveNodeLabel, count) in sensitiveNodeDict.items():
                entity = getEntity(sensitiveNodeLabel)
                if entity not in entityToIdDict:
                    raise Exception("Entity not in entityToIdDict !")
                id = entityToIdDict[entity]
                idToCountDict[id] += count
                if id > 0:
                    sourceAPICount += count
                else: 
                    sinkAPICount += count
    statisticalData = OrderedDict()
    statisticalData["MSSGNodePCT"] = MSSGNodePercentageList
    statisticalData["MSSGEdgePCT"] = MSSGEdgePercentageList
    statisticalData["MSSGSenNodePCT"] = MSSGSenNodePercentageList
    
    if isMalware:
        statisticalDataJsonFilePath = "DataStatistics/malwareMSSGPCT.json"
    else:
        statisticalDataJsonFilePath = "DataStatistics/benignMSSGPCT.json"
    with open(statisticalDataJsonFilePath, "w") as statisticalDataJsonFile:
        json.dump(statisticalData, statisticalDataJsonFile)
    
#    print("MSSGNodePercentageList:")
#    print(MSSGNodePercentageList)
#    print("MSSGEdgePercentageList:")
#    print(MSSGEdgePercentageList)
#    print("MSSGSenNodePercentageList:")
#    print(MSSGSenNodePercentageList)
    
#    print("idToCountDict:")
#    print(idToCountDict)
#    print("sourceAPICount = %s" % sourceAPICount)
#    print("sinkAPICount = %s" % sinkAPICount)  
    
    statisticalDataPlot(statisticalData, isMalware)
    
    return idToCountDict, numOfApks


def statisticalDataPlot(statisticalData, isMalware=True):
    # boxplot 
    numOfApks = len(list(statisticalData.values())[0])
    items = len(statisticalData.keys())
    boxplotData = np.zeros([numOfApks, items], dtype=float)
    boxplotLabels = statisticalData.keys()
#    print(boxplotData)
#    print(boxplotData.shape)
    for i, percentageList in enumerate(statisticalData.values()):
        boxplotData[:, i] = np.array(percentageList)
#    print(boxplotData)
    
    boxplotData = boxplotData * 100
    
#    plt.figure(figsize=(9, 4))    
    plt.figure()
    plt.ylabel("Percentage (%)")
    yticks = ticker.FormatStrFormatter("%d")
    plt.gca().yaxis.set_major_formatter(yticks)
    
    if isMalware:
        boxplotTitle = "Malware MSSGPCT Statistics"
        boxplotFilePath = "DataStatistics/malwareMSSGPCT.pdf"
    else:
        boxplotTitle = "Benign MSSGPCT Statistics"
        boxplotFilePath = "DataStatistics/benignMSSGPCT.pdf"
    
    plt.title(boxplotTitle)   
    plt.boxplot(boxplotData,labels=boxplotLabels,sym='o',whis=1.5, showmeans=True)
    plt.savefig(boxplotFilePath)
#    plt.show()


def getEntity(nodeLabel):
    entity = ""
    pattern = re.compile("{<(.*?): (.*?) (.*?)\((.*?)\)>(.*)}")
    matcher = pattern.match(nodeLabel)
    if matcher:
        packageName = matcher.group(1)
        methodName = matcher.group(3)
        entity = packageName + ": " + methodName
    return entity


def getOrderedDictFromJsonFile(jsonFilePath):
    with open(jsonFilePath, "r") as jsonFile:
        OrderedDictFromJsonFile = json.load(jsonFile, object_pairs_hook=OrderedDict)
    return OrderedDictFromJsonFile


def getCoefficientOfSensitivityDict(malwareIdToCountDict, numOfMalwareApks, 
                                    benignIdToCountDict, numOfBenignApks, 
                                    idToEntityDict, entityToIdDict):
    # "COS" means "Coefficient Of Sensitivity"
    idToCOSDict = OrderedDict()
    entityToCOSDict = OrderedDict()
    for (id, entity) in idToEntityDict.items():
        reciprocalOfAverageBenignCount = numOfBenignApks / (1 + benignIdToCountDict[id])
        averageMalwareCount = malwareIdToCountDict[id] / numOfMalwareApks
        # COS = averageMalwareCount * math.log10(reciprocalOfAverageBenignCount)
        """
        TODO
        这里敏感系数没有加log平滑，需要做实验测试
        """
        COS = averageMalwareCount * reciprocalOfAverageBenignCount
        idToCOSDict[id] = COS
        entityToCOSDict[entity] = COS
    
    idToCOSDictJsonFilePath = "DataStatistics/idToCOSDict.json"
    with open(idToCOSDictJsonFilePath, "w") as idToCOSDictJsonFile:
        json.dump(idToCOSDict, idToCOSDictJsonFile)
        
    entityToCOSDictJsonFilePath = "DataStatistics/entityToCOSDict.json"
    
    with open(entityToCOSDictJsonFilePath, "w") as entityToCOSDictJsonFile:
        json.dump(entityToCOSDict, entityToCOSDictJsonFile)
        
#    print(getOrderedDictFromJsonFile(idToCOSDictJsonFilePath))
    
    return idToCOSDict, entityToCOSDict


def statisticsOfCommunity(apkDecompileDatesetDirPath, isMalware=True):
    
    
    numOfCommunitiesInEachApkList = []
    numOfClustersInEachCommunityList = []
    numOfSenNodesInEachClusterList = []
    numOfSenNodesInEachCommunityList = []
    numOfSenNodesInEachApkList = []
    numOfClustersInEachApkList = []
        
    for apkDecompileDir in os.listdir(apkDecompileDatesetDirPath):
        listJsonFilePath = os.path.join(apkDecompileDatesetDirPath, 
                                        apkDecompileDir, "featureList.json")
        if os.path.isfile(listJsonFilePath):
            # print("name of apk: %s" % apkDecompileDir)
            with open(listJsonFilePath, "r") as listJsonFile:
                featureList = json.load(listJsonFile)
            
            
            numOfCommunitiesInEachApkList.append(len(featureList))
            numOfSenNodesInEachApk = 0
            numOfClustersInEachApk = 0
            for community in featureList:
                numOfClustersInEachCommunityList.append(len(community))
                numOfClustersInEachApk += len(community)
                numOfSenNodesInEachCommunity = 0
                for cluster in community:
                    numOfSenNodesInEachClusterList.append(len(cluster))
                    numOfSenNodesInEachCommunity += len(cluster)
                
                numOfSenNodesInEachCommunityList.append(numOfSenNodesInEachCommunity)
                numOfSenNodesInEachApk += numOfSenNodesInEachCommunity
            
            numOfSenNodesInEachApkList.append(numOfSenNodesInEachApk)
            numOfClustersInEachApkList.append(numOfClustersInEachApk)
    labels = ["NumOfCommunitiesInEachApk", "NumOfClustersInEachCommunity",
              "NumOfSenNodesInEachCluster","NumOfSenNodesInEachCommunity",
              "NumOfSenNodesInEachApk","NumOfClustersInEachApk"]
    dataLists = [numOfCommunitiesInEachApkList, numOfClustersInEachCommunityList,
              numOfSenNodesInEachClusterList, numOfSenNodesInEachCommunityList,
              numOfSenNodesInEachApkList, numOfClustersInEachApkList]
    
    for label, dataList in zip(labels, dataLists):
        if isMalware:
            label = "Malware" + label
        else: label = "Benign" + label
        histAndBoxPlot(dataList, label)
        
        
    dataFrame = pd.DataFrame(dataLists)
    dataFrame = dataFrame.T
    statisticsDataFrame = dataFrame.describe()
    statisticsDataFrame.columns = labels
    print(statisticsDataFrame)
    if isMalware:
        statisticsCSVFilePath = "DataStatistics/malwareCommunityStatistics.csv"
    else:
        statisticsCSVFilePath = "DataStatistics/benignCommunityStatistics.csv"
        
    statisticsDataFrame.to_csv(statisticsCSVFilePath)
    
#    print(numOfCommunitiesInEachApkList)
#    print(numOfClustersInEachCommunityList)
#    print(numOfSenNodesInEachClusterList)
#    print(numOfSenNodesInEachCommunityList)
#    print(numOfSenNodesInEachApkList)
#    print(numOfClustersInEachApkList)

def histAndBoxPlot(dataList, dataLabel):
    
    # plt.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
    data = np.array(dataList)
    fig = plt.figure(figsize =(9,5))
    
    # boxplot 
    axBoxplot = fig.add_subplot(1,2,1)
    axBoxplot.set_ylabel(dataLabel)
    axBoxplot.yaxis.set_major_locator(ticker.MultipleLocator(int(max(data) / 15)))
    # axBoxplot.set_title("箱型图")
    axBoxplot.boxplot(data,sym='o',whis=1.5, showmeans=True)
    
    # hist
    axhist = fig.add_subplot(1,2,2)
    axhist.set_xlabel(dataLabel)
    axhist.xaxis.set_major_locator(ticker.MultipleLocator(int(max(data) / 10)))
    axhist.set_ylabel("Frequency")
    # axhist.set_title("直方图")
    axhist.hist(data,bins=40, density=0, facecolor="blue", edgecolor="black", alpha=0.7)
    
    fig.tight_layout()
    
    figurePdfFilePath ="DataStatistics/" + dataLabel + ".pdf"
    plt.savefig(figurePdfFilePath)
    # plt.show()


if __name__ == "__main__":
##    entity = getEntity("{<android.media.AudioRecord: java.lang.Integer getRecordingState()>,Sink,,NO_CATEGORY}")
##    print(entity)
#    sourceDict, sinkDict = readSourceAndSink()
##    print(sourceDict)
#    idToEntityDict, entityToIdDict = getEntityIdDict(sourceDict, sinkDict)
##    print(idToEntityDict)
#    malwareIdToCountDict, numOfMalwareApks = statistics("F:\\test\\decompileDateset\\malware", 
#                                     idToEntityDict, entityToIdDict, isMalware=True)
##    print(malwareIdToCountDict)
#    benignIdToCountDict, numOfBenignApks = statistics("F:\\test\\decompileDateset\\benign", 
#                                      idToEntityDict, entityToIdDict, isMalware=False)
#    getCoefficientOfSensitivityDict(malwareIdToCountDict, numOfMalwareApks, 
#                                    benignIdToCountDict, numOfBenignApks, 
#                                    idToEntityDict, entityToIdDict)
    
    # test statisticsOfCommunity
#    # windows
#    statisticsOfCommunity("F:\\test\\decompileDataset\\malware")
#    statisticsOfCommunity("F:\\test\\decompileDataset\\benign", isMalware=False)
    
    # linux
    statisticsOfCommunity("/mount/zx/MyDroid/Dataset/2012/malware")
    statisticsOfCommunity("/mount/zx/MyDroid/Dataset/2012/benign", isMalware=False)
    
    
    
    
    
    
    