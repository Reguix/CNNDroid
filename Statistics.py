# -*- coding: utf-8 -*-
"""
功能：
1. 获取敏感节点的敏感度
2. 对社交检测的结果进行统计分析
"""
import os
import json
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
import pandas as pd

from SourceAndSink import readSourceAndSink, getEntityIdDict
from Utils import getOrderedDictFromJsonFile, getEntity
# 统计数据集中的APK的数目和敏感API出现的次数
def statistics(apkDecompileDatesetDirPath, idToEntityDict, entityToIdDict, 
               isMalware=True):
    
    # 敏感API编号和其出现次数的字典，先初始化为0
    idToCountDict = dict()
    for id in idToEntityDict.keys():
        idToCountDict[id] = 0
    
    # 遍历数据集中的每个文件夹中的information.json文件进行统计
    numOfApks = 0
    for apkDecompileDir in os.listdir(apkDecompileDatesetDirPath):
        infoJsonFilePath = os.path.join(apkDecompileDatesetDirPath, 
                                        apkDecompileDir, "information.json")
        if os.path.isfile(infoJsonFilePath):
            numOfApks += 1
            infoOrderedDict = getOrderedDictFromJsonFile(infoJsonFilePath)
            sensitiveNodeIdDict = infoOrderedDict["sensitiveNodeIdDict"]
            for (sensitiveNodeId, count) in sensitiveNodeIdDict.items():
                idToCountDict[int(sensitiveNodeId)] += count
    
    return idToCountDict, numOfApks

# 计算敏感度
def calculateCOS(malwareIdToCountDict, numOfMalwareApks, 
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
        把两个值对比一下，是否需要把倒数的值取log
        """
        COS = averageMalwareCount * math.log10(reciprocalOfAverageBenignCount + 1)
        idToCOSDict[id] = COS
        entityToCOSDict[entity] = COS
    return idToCOSDict, entityToCOSDict

# 获取敏感度
def getCOS(benignDecompileDatesetDirPath, malwareDecompileDatesetDirPath, idToEntityDict, entityToIdDict):
    
    if os.path.isfile("idToCOSDict.json") and os.path.isfile("entityToCOSDict.json"):
        # 从原来保存的json文件中加载敏感度
        print("Attention! Load coefficient of sensitivity from 'idToCOSDict.json' file!")
        idToCOSTempDict = getOrderedDictFromJsonFile("idToCOSDict.json")
        entityToCOSDict = getOrderedDictFromJsonFile("entityToCOSDict.json")
        idToCOSDict = OrderedDict()
        for key in idToCOSTempDict.keys():
            idToCOSDict[int(key)] = idToCOSTempDict[key]
    else:
        print("Calculate coefficient of sensitivity...")
        # 统计良性数据集中敏感API出现的次数
        benignIdToCountDict, numOfBenignApks = statistics(benignDecompileDatesetDirPath, 
                                                          idToEntityDict, entityToIdDict,
                                                          isMalware=False)
        # 保存良性数据集的统计信息
        saveDatasetDict(benignIdToCountDict, numOfBenignApks, 
                        benignDecompileDatesetDirPath, isMalware=False)
        # 统计恶意数据集中敏感API出现的次数
        malwareIdToCountDict, numOfMalwareApks = statistics(malwareDecompileDatesetDirPath, 
                                                            idToEntityDict, entityToIdDict,
                                                            isMalware=True)
        # 保存恶意数据集的统计信息
        saveDatasetDict(malwareIdToCountDict, numOfMalwareApks, 
                        malwareDecompileDatesetDirPath, isMalware=True)
        # 计算所有敏感API的敏感度
        idToCOSDict, entityToCOSDict = calculateCOS(malwareIdToCountDict,
                                                    numOfMalwareApks, 
                                                    benignIdToCountDict, 
                                                    numOfBenignApks, 
                                                    idToEntityDict, 
                                                    entityToIdDict)
        idToCOSDictJsonFilePath = "idToCOSDict.json"
        with open(idToCOSDictJsonFilePath, "w") as idToCOSDictJsonFile:
            json.dump(idToCOSDict, idToCOSDictJsonFile)
        entityToCOSDictJsonFilePath = "entityToCOSDict.json"
        with open(entityToCOSDictJsonFilePath, "w") as entityToCOSDictJsonFile:
            json.dump(entityToCOSDict, entityToCOSDictJsonFile)
    return idToCOSDict, entityToCOSDict

# 保存数据集的统计信息(这里数据不够，还需要统计节点，边，敏感节点)
def saveDatasetDict(idToCountDict, numOfApks, decompileDatesetDirPath, isMalware=True):
    datasetDict = OrderedDict()
    datasetDict["numOfApks"] = numOfApks 
    datasetDict["idToCountDict"] = idToCountDict
    datasetDictFileName = "_".join(decompileDatesetDirPath.split(os.path.sep)[-2:])
    if isMalware:
        datasetDictFileName += "_malware_datasetDict.json"
    else:
        datasetDictFileName += "_benign_datasetDict.json"
    
    datasetDictFilePath = datasetDictFileName
    with open(datasetDictFilePath, "w") as datasetDictFile:
            json.dump(datasetDict, datasetDictFile)
    
    
# 统计数据集在社交检测后的社区分布情况
def statisticsOfCommunity(apkDecompileDatesetDirPath, isMalware=True):
    # 统计六个指标的分布情况
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
    # 对每个指标画图展示其分布规律
    for label, dataList in zip(labels, dataLists):
        if isMalware:
            label = "Malware" + label
        else: label = "Benign" + label
        histAndBoxPlot(dataList, label)
    # 保存六个指标的原始数据
    dataFrame = pd.DataFrame(dataLists)
    dataFrame = dataFrame.T
    dataFrame.columns = labels
    if isMalware:
        dataFrame.to_csv("DataStatistics/malware_dataset.csv")
    else:
        dataFrame.to_csv("DataStatistics/benign_dataset.csv")
    # 分析六个指标的个数，平均值，最大值，最小值，方差等
    statisticsDataFrame = dataFrame.describe()
    statisticsDataFrame.columns = labels
    #print(statisticsDataFrame)
    if isMalware:
        statisticsCSVFilePath = "DataStatistics/malwareCommunityStatistics.csv"
    else:
        statisticsCSVFilePath = "DataStatistics/benignCommunityStatistics.csv"
    statisticsDataFrame.to_csv(statisticsCSVFilePath)

# 画直方图和箱型图
def histAndBoxPlot(dataList, dataLabel):
    # 用黑体显示中文
    # plt.rcParams['font.sans-serif']=['SimHei']   
    data = np.array(dataList)
    fig = plt.figure(figsize =(9,5))
    # boxplot 
    axBoxplot = fig.add_subplot(1,2,1)
    axBoxplot.set_ylabel(dataLabel)
    axBoxplot.yaxis.set_major_locator(ticker.MultipleLocator(int(max(data) / 15)))
    # axBoxplot.set_title("box")
    axBoxplot.boxplot(data,sym='o',whis=1.5, showmeans=True)
    # hist
    axhist = fig.add_subplot(1,2,2)
    axhist.set_xlabel(dataLabel)
    axhist.xaxis.set_major_locator(ticker.MultipleLocator(int(max(data) / 10)))
    axhist.set_ylabel("Frequency")
    # axhist.set_title("hist")
    axhist.hist(data,bins=40, density=0, facecolor="blue", edgecolor="black", alpha=0.7)
    fig.tight_layout()
    figurePdfFilePath ="DataStatistics/" + dataLabel + ".pdf"
    plt.savefig(figurePdfFilePath)
    # plt.show()