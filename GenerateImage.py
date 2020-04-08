# -*- coding: utf-8 -*-
"""
功能：将提取的特征列表排列为规则的张量图像
"""
import os
import torch
import json
import pandas as pd
import numpy as np

from Statistics import histAndBoxPlot

# 排列特征像素点为张量图像
def featureListToImageTensor(featureListJsonFilePath, lenOfSourceDict=17361, lenOfSinkDict=7784, 
                       width=4, height=800, maxHeight=3000):
    with open(featureListJsonFilePath, "r") as featureListJsonFile:
        featureList = json.load(featureListJsonFile)
    rawImage = np.zeros((maxHeight, width))
    imageTensor = torch.zeros(maxHeight, width).float()
    widthIndex, heightIndex = 0, 0
    for community in featureList:
        if widthIndex != 0:
            widthIndex = 0
            heightIndex += 1
            #if heightIndex >= height:
                #return imageTensor[:height]
        for cluster in community:
            if widthIndex != 0 and widthIndex + len(cluster) > width:
                widthIndex = 0
                heightIndex += 1
                #if heightIndex >= height:
                    #return imageTensor[:height]
            for nodeId in cluster:
                pixel = getNormalizedPixel(nodeId, lenOfSourceDict, lenOfSinkDict)
                imageTensor[heightIndex][widthIndex] = pixel
                rawImage[heightIndex][widthIndex] = nodeId
                widthIndex += 1
                if widthIndex >= width:
                    widthIndex = 0
                    heightIndex += 1
                    #if heightIndex >= height:
                        #return imageTensor[:height]
    if widthIndex != 0:
        rawHeight = heightIndex + 1
    else:
        rawHeight = heightIndex
    # print(imageTensor[:rawHeight])
    return rawImage[:height], imageTensor[:height], rawHeight
            
def getNormalizedPixel(nodeId, lenOfSourceDict, lenOfSinkDict):
    if nodeId > 0:
        return nodeId / lenOfSourceDict
    else:
        return nodeId / lenOfSinkDict

# 生成图像
def generateImage(apkDecompileDirPath, imageDatasetDirPath,
                  lenOfSourceDict=17361, lenOfSinkDict=7784, isMalware=True):
    featureListJsonFilePath = os.path.join(apkDecompileDirPath, "featureList.json")
    apkDecompileDir = apkDecompileDirPath.split(os.path.sep)[-1];
    if os.path.isfile(featureListJsonFilePath):
        rawImage, imageTensor, imageRawHeight = featureListToImageTensor(featureListJsonFilePath)
        if isMalware:
            imageTensorFilePath = os.path.join(imageDatasetDirPath, 
                                               apkDecompileDir + "_1.pickle")
        else:
            imageTensorFilePath = os.path.join(imageDatasetDirPath,
                                                   apkDecompileDir + "_0.pickle")
            
        with open(imageTensorFilePath, "wb") as imageTensorFile:
            torch.save(imageTensor, imageTensorFile)
    
        return rawImage, imageRawHeight
    else:
        raise OSError("%s does not exist." % (featureListJsonFilePath))


    
# 生成图像数据集
def generateImageDataset(apkDecompileDatesetDirPath, imageDatasetDirPath, 
                         lenOfSourceDict=17361, lenOfSinkDict=7784, isMalware=True):
    imageRawHeightList = []
    if not os.path.isdir(imageDatasetDirPath):
        os.makedirs(imageDatasetDirPath)
    for apkDecompileDir in os.listdir(apkDecompileDatesetDirPath):
        apkDecompileDirPath = os.path.join(apkDecompileDatesetDirPath, apkDecompileDir)
        _, imageRawHeight = generateImage(apkDecompileDirPath, imageDatasetDirPath, lenOfSourceDict, lenOfSinkDict, isMalware)
        imageRawHeightList.append(imageRawHeight)
        
    # 原始高度分布统计（正式流程省略）
    heightStatistics(imageRawHeightList, isMalware)

# 原始高度分布统计，用于选取合适的图像高度
def heightStatistics(imageRawHeightList, isMalware):   
    dataFrame = pd.DataFrame(imageRawHeightList)
    statisticsDataFrame = dataFrame.describe()
    statisticsDataFrame.columns = ["imageRawHeight"]
    if isMalware:
        statisticsCSVFilePath = "DataStatistics/malwareImageRawHeightStatistics.csv"
        plotLabel = "malwareImageRawHeight"
        dataFrame.to_csv("DataStatistics/malwareImageHeight.csv")
    else:
        statisticsCSVFilePath = "DataStatistics/benignImageRawHeightStatistics.csv"
        plotLabel = "benignImageRawHeight"
        dataFrame.to_csv("DataStatistics/benignImageHeight.csv")
    statisticsDataFrame.to_csv(statisticsCSVFilePath)
    histAndBoxPlot(imageRawHeightList, plotLabel)
    # print(statisticsDataFrame)
        
if __name__ == "__main__":
    # windows
#    generateImageDataset("F:\\test\\decompileDataset\\malware", "F:\\test\\decompileDataset\\image")
#    generateImageDataset("F:\\test\\decompileDataset\\benign", "F:\\test\\decompileDataset\\image")
    
    # linux
    generateImageDataset("/home/zhangxin/CNNDroid/Dataset/2017/benign",
                         "/home/zhangxin/CNNDroid/Dataset/2017/image_800", isMalware=False)
    generateImageDataset("/home/zhangxin/CNNDroid/Dataset/2017/malware", 
                         "/home/zhangxin/CNNDroid/Dataset/2017/image_800")