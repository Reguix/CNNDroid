# -*- coding: utf-8 -*-
"""
主函数
参数固定在代码里面，运行前需要更改代码。
参数为五个文件夹路径:
--benignDatesetDirPath   良性数据集，文件夹中存储着.apk后缀的APK文件。
--malwareDatesetDirPath  恶意数据集，文件夹中存储着.apk后缀的APK文件。
--benignDecompileDatesetDirPath   良性反汇编数据集，经过反汇编后每个APK文件会对应一个同名的文件夹。
--malwareDecompileDatesetDirPath  恶意反汇编数据集，经过反汇编后每个APK文件会对应一个同名的文件夹。
--imageDatasetDirPath 生成的APK图像数据集。


功能：
1. 反汇编APK，并提取调用图
2. 社交网络检测调用图提取敏感节点列表
3. 根据敏感度对敏感节点列表进行重新排序
4. 将特征转变为规则图像

注：首次运行需要统计数据集中敏感API出现的次数，计算得到敏感度
"""
import csv
import torch
import time
import os
import json
from queue import Queue
import networkx as nx
from ApkToGraph import oneApkToGraph
from SourceAndSink import readSourceAndSink, getEntityIdDict
from Statistics import getCOS, statisticsOfCommunity
from CommunityDetection import oneApkToFeatureList
from GenerateImage import generateImage
from Utils import getEntity

from multiprocessing import Pool
from functools import partial
from CNN.models import AlexNet
from CNN.grad_cam import gradcam
import numpy as np
from collections import OrderedDict
np.set_printoptions(threshold=np.inf)
#参数
# windows

benignDatesetDirPath = "F:\\test\\dataset\\benign"
malwareDatesetDirPath = "F:\\test\\dataset\\malware"
benignDecompileDatesetDirPath = "F:\\test\\decompileDataset\\benign"
malwareDecompileDatesetDirPath = "F:\\test\\decompileDataset\\malware"
imageDatasetDirPath = "F:\\test\\decompileDataset\\image"

# linux

#benignDatesetDirPath = "/home/zx/Dataset/2018/benign"
#malwareDatesetDirPath = "/home/zx/Dataset/2018/malware"
#benignDecompileDatesetDirPath = "/home/zx/CNNDroid/Dataset/2018/benign"
#malwareDecompileDatesetDirPath = "/home/zx/CNNDroid/Dataset/2018/malware"
#imageDatasetDirPath = "/home/zx/CNNDroid/Dataset/2018/image_800"

def classify(pretrained_model, image):
    with torch.no_grad():
        pretrained_model.eval()
        score = pretrained_model(image)
    label = score.max(dim = 1)[1].squeeze().cpu().numpy()
    return label

def getIdFromPixel(pixel, lenOfSourceDict, lenOfSinkDict):
    id = 0
    if pixel > 0:
        id = int(pixel * lenOfSourceDict)
    if pixel < 0:
        id = int(pixel * lenOfSinkDict)
    return id

def tranformImage(imageFilePath,lenOfSourceDict=17361, lenOfSinkDict=7784, 
                       width=4, height=800):
    imageFile = open(imageFilePath, "rb")
    image = torch.load(imageFile)
    for heightIndex in range(height):
        for widthIndex in range(width):
            pixel = image[heightIndex][widthIndex]
            image[heightIndex][widthIndex] = getIdFromPixel(pixel, lenOfSourceDict, lenOfSinkDict)
    image_np = image.numpy()
    image_np = image_np.astype(int)
    return image_np

def findPattern(cam, image, idToEntityDict, winSize=5, width=4, height=800):
    rowCamList = []
    
    border = height
    for i in range(height):
        isAllZero = True
        for j in range(width):
            if image[i][j] != 0:
                isAllZero = False
        if isAllZero:
            border = i
            break
    #print("border: ", border)
    for i in range(height):
        rowCam = 0
        for j in range(width):
            rowCam += cam[i][j]
        rowCamList.append(rowCam)
    
    winCam = sum(rowCamList[:winSize])
    maxRowIndex = 0
    maxWinCam = winCam
    for i in range(winSize, border):
        winCam = winCam - rowCamList[i - winSize] + rowCamList[i]
        if winCam > maxWinCam:
            maxWinCam = winCam
            maxRowIndex = i - winSize
    #print("maxRowIndex: ", maxRowIndex)
    entityList = []
    for i in range(winSize):
        for j in range(width):
            id = image[maxRowIndex + i][j]
            if id != 0:
                entityList.append(idToEntityDict[id])
    return entityList

def bfs(node, DG):
    if node is None:
        return
    queue = Queue()
    nodeSet = set()
    queue.put(node)
    nodeSet.add(node)
    while not queue.empty():
        curNode = queue.get()               
        for nextNode in list(DG.predecessors(curNode)):
            if nextNode not in nodeSet:
                nodeSet.add(nextNode)
                queue.put(nextNode)
    return nodeSet

def patternGraph(imageFilePathPrefix, patternList, apkDecompileDirPath):
    graphGexfFilePath = os.path.join(apkDecompileDirPath, "sourceGraph.gexf")
    sourceGraph = nx.read_gexf(graphGexfFilePath)
    
    comListJsonFilePath = os.path.join(apkDecompileDirPath, "comList.json")
    with open(comListJsonFilePath, "r") as comListJsonFile:
        comList = json.load(comListJsonFile)
    
    patternGraph = None
    for com in comList:
        for cluster in com:
            cnt = 0
            for label in cluster:
                if label.split(",")[-3] != "Normal":
                    for entity in patternList:
                        if getEntity(label) == entity:
                            cnt += 1
            if cnt >= len(patternList) * 0.8:
                patternGraph = sourceGraph.subgraph(cluster).copy()
                nx.write_gexf(patternGraph, imageFilePathPrefix + "_patternGraph.gexf")
    
    if patternGraph is not None:
        senNodeList = list()
        for node in patternGraph.nodes():
            if node.split(",")[-3] != "Normal":
                senNodeList.append(node)
        preNodeSet = set()
        for senNode in senNodeList:
            nodeSet = bfs(senNode, patternGraph)
            preNodeSet = preNodeSet | nodeSet
        patternSubgraph = patternGraph.subgraph(list(preNodeSet)).copy()
        nx.write_gexf(patternSubgraph, imageFilePathPrefix + "_patternSubgraph.gexf")
                

# 处理一个APK文件，生成调用图，提取特征，排列为图像
def handleOneApk(apkFilePath, apkDecompileDatesetDirPath, imageDatasetDirPath, sourceDict, sinkDict, 
                 idToEntityDict, entityToIdDict, isMalware, idToCOSDict=[]):
    # 生成调用图
    # timeDict = OrderedDict()
    timeDict = oneApkToGraph(apkFilePath, apkDecompileDatesetDirPath, sourceDict, sinkDict,
                                 idToEntityDict, entityToIdDict)
    apkFileBasename = os.path.basename(apkFilePath)
    apkFilename, _ = os.path.splitext(apkFileBasename)
    apkDecompileDirPath = os.path.join(apkDecompileDatesetDirPath, apkFilename)
    # 反汇编失败，直接返回
    if not os.path.isdir(apkDecompileDirPath):
        return timeDict
    
    # 社交网络检测，提取特征
    timeStart = time.time()
    oneApkToFeatureList(apkDecompileDirPath, entityToIdDict, idToCOSDict,
                        superNodeList=[])
    # 将特征排列为图像
    if len(idToCOSDict) == 0:
        return timeDict
    rawImage, rawHeight = generateImage(apkDecompileDirPath, imageDatasetDirPath, 
                                17361, 7784, isMalware)
    timeDict["rawHeight"] = rawHeight
    timeDict["imageTime"] = time.time() - timeStart
    
    # 分类
    timeStart = time.time()
    imageFilePathPrefix = os.path.join(imageDatasetDirPath, apkFilename)
    if isMalware:
        imageFilePath = imageFilePathPrefix + "_1.pickle"
    else:
        imageFilePath = imageFilePathPrefix + "_0.pickle"
    
    # 加载模型
    model = AlexNet()
    model.load("CNN/pth/AlexNet.pth")
    model.eval()
    
    # 预处理图像
    imageFile = open(imageFilePath, "rb")
    image = torch.load(imageFile)
    imageFile.close()
    image = torch.unsqueeze(image, dim=0)
    image = torch.unsqueeze(image, dim=0).float()
    
    # 分类器
    label = classify(model, image)
    #print("label: ", label)
    timeDict["classifyTime"] = time.time() - timeStart
    
    # 梯度可视化
    timeStart = time.time()
    cam = gradcam(model, image, label, imageFilePathPrefix)
    
    # 发现恶意调用模式
    image_np = tranformImage(imageFilePath)
    np.savetxt(imageFilePathPrefix + "_image.csv", image_np, fmt="%d", delimiter=",")
    patternList = findPattern(cam, image_np, idToEntityDict, winSize=5, width=4, height=800)
    with open(imageFilePathPrefix + "_pattern.txt", "w") as patternFile:
        for p in patternList:
            patternFile.write(p + "\n")
    
    # 提取恶意调用模式的调用图
    patternGraph(imageFilePathPrefix, patternList, apkDecompileDirPath)
    
    timeDict["gradcamTime"] = time.time() - timeStart
    #print(entityList)
    return  timeDict


def handleApkDataset(apkDatesetDirPath, apkDecompileDatesetDirPath, imageDatasetDirPath,
                     sourceDict, sinkDict, idToEntityDict, entityToIdDict, 
                     isMalware=True, idToCOSDict=[]):  
    # 如果该目录不存在，则新建目录
    if not os.path.isdir(apkDecompileDatesetDirPath):
        os.makedirs(apkDecompileDatesetDirPath)
    if not os.path.isdir(imageDatasetDirPath):
        os.makedirs(imageDatasetDirPath)  
    # 要反汇编的APK文件路径列表
    apkFilePathList = []    
    for apkFileBasename in os.listdir(apkDatesetDirPath):
        if apkFileBasename.endswith(".apk"):
            apkFilePath = os.path.join(apkDatesetDirPath, apkFileBasename)
            apkFilePathList.append(apkFilePath)
    # 多线程处理
    pool = Pool()
    timeDictList = pool.map(partial(handleOneApk, apkDecompileDatesetDirPath=apkDecompileDatesetDirPath,
                            imageDatasetDirPath=imageDatasetDirPath,
                            sourceDict=sourceDict, sinkDict=sinkDict,
                            idToEntityDict=idToEntityDict, 
                            entityToIdDict=entityToIdDict,
                            isMalware=isMalware, idToCOSDict=idToCOSDict), apkFilePathList)
    if isMalware:
        runtimeFile = "malware_runtime.csv"
    else:
        runtimeFile = "benign_runtime.csv"
    itemList = ["apkFilename", "edges", "nodes", "senNodes", "apktoolTime", 
                "graphTime", "imageTime", "classifyTime", "gradcamTime", "rawHeight"]
    with open(runtimeFile, "w", newline="") as F:
        for timeDict in timeDictList:
            strList = []
            strList.clear()
            if len(timeDict.keys()) == len(itemList):
                for item in itemList:
                    strList.append(str(timeDict[item]))
                F.write(",".join(strList) + "\n")
    #print(runtime_list)
    
def main():
    # 检查数据集文件夹是否存在
    if not os.path.isdir(benignDatesetDirPath):
        raise OSError("Benign apk directory %s does not exist." % (benignDatesetDirPath))
    if not os.path.isdir(malwareDatesetDirPath):
        raise OSError("Malware apk directory %s does not exist." % (malwareDatesetDirPath))
    
    # 从Source.txt和Sink.txt解析敏感API
    sourceDict, sinkDict = readSourceAndSink()
    # 对敏感API进行编号
    idToEntityDict, entityToIdDict = getEntityIdDict(sourceDict, sinkDict)    
    
    # 如果没有提供计算好的敏感度，则先处理数据集
    idToCOSDict = []
    if not os.path.isfile("idToCOSDict.json"):
        # 先对数据集进行预处理
        print("Not Find idToCOSDict.json! Start pretreatment...")
        print("Benign dataset pretreatment...")
        handleApkDataset(benignDatesetDirPath, benignDecompileDatesetDirPath, imageDatasetDirPath,
                         sourceDict, sinkDict, idToEntityDict, entityToIdDict, False, idToCOSDict)
        print("Malware dataset pretreatment...")
        handleApkDataset(malwareDatesetDirPath, malwareDecompileDatesetDirPath, imageDatasetDirPath,
                         sourceDict, sinkDict, idToEntityDict, entityToIdDict, True, idToCOSDict)
    # 获取敏感度
    print("Get coefficient of sensitivity...")
    idToCOSDict, entityToCOSDict = getCOS(benignDecompileDatesetDirPath, 
                                          malwareDecompileDatesetDirPath, 
                                          idToEntityDict, entityToIdDict)
#    print("Benign dataset in processing...")
#    handleApkDataset(benignDatesetDirPath, benignDecompileDatesetDirPath, imageDatasetDirPath,
#                     sourceDict, sinkDict, idToEntityDict, entityToIdDict, False, idToCOSDict)
    print("Malware dataset in processing...")
    handleApkDataset(malwareDatesetDirPath, malwareDecompileDatesetDirPath, imageDatasetDirPath,
                     sourceDict, sinkDict, idToEntityDict, entityToIdDict, True, idToCOSDict)
    
    # 对社交网络的检测结果进行统计分析（正式流程可省略）
#    print("Benign dataset community statistics...")
#    statisticsOfCommunity(benignDecompileDatesetDirPath, isMalware=False)
#    
#    print("Malware dataset community statistics...")
#    statisticsOfCommunity(malwareDecompileDatesetDirPath)
    
    print("Analyse done!")
    
    
if __name__ == "__main__":
    time_start=time.time()
    main()
    time_end=time.time()
    print('time cost',time_end-time_start,'s')
#    sourceDict, sinkDict = readSourceAndSink()
#    # 对敏感API进行编号
#    idToEntityDict, entityToIdDict = getEntityIdDict(sourceDict, sinkDict) 
    
#    handleOneApk("CNN/pth/DC6DBFA3413637AADCD15FC6B941B44ADFB213234102C1DBF64A75C5F61BD361.apk", "CNN/pth/", "CNN/pth/",
#                 sourceDict, sinkDict, idToEntityDict, entityToIdDict, 
#                 isMalware=False, idToCOSDict=[])
#
#    handleOneApk("CNN/pth/989321326DAC3661A0B28CFAFA16B3B96D0F6E42772F1AD787CBF9E4F2C4AF18.apk", "CNN/pth/", "CNN/pth/",
#                 sourceDict, sinkDict, idToEntityDict, entityToIdDict, 
#                 isMalware=True, idToCOSDict=[])