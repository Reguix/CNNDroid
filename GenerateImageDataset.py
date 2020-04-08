# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 16:04:53 2019

@author: ZhangXin
"""
import os
import torch
import json
import pandas as pd

from Statistics import histAndBoxPlot

def featureListToImageTensor(featureListJsonFilePath, lenOfSourceDict=17372, lenOfSinkDict=7784, 
                       width=4, height=300, maxHeight=2000):
    
    with open(featureListJsonFilePath, "r") as featureListJsonFile:
        featureList = json.load(featureListJsonFile)
    
    imageTensor = torch.zeros(maxHeight, width).double()
    
    widthIndex, heightIndex = 0, 0
    
    for community in featureList:
        if widthIndex != 0:
            widthIndex = 0
            heightIndex += 1
        for cluster in community:
            if widthIndex != 0 and widthIndex + len(cluster) > width:
                widthIndex = 0
                heightIndex += 1
            for nodeId in cluster:
                pixel = getNormalizedPixel(nodeId, lenOfSourceDict, lenOfSinkDict)
                imageTensor[heightIndex][widthIndex] = pixel
                # imageTensor[heightIndex][widthIndex] = nodeId
                # print("height: %s width: %s nodeId: %s" % (heightIndex, widthIndex, nodeId))
                widthIndex += 1
                if widthIndex >= width:
                    widthIndex = 0
                    heightIndex += 1
    if widthIndex != 0:
        rawHeight = heightIndex + 1
    else:
        rawHeight = heightIndex
    
    # print(imageTensor[:rawHeight])
    return imageTensor[:height], rawHeight
            
def getNormalizedPixel(nodeId, lenOfSourceDict, lenOfSinkDict):
    if nodeId > 0:
        return nodeId / lenOfSourceDict
    else:
        return nodeId / lenOfSinkDict
    

def generateImageDataset(apkDecompileDatesetDirPath, imageDatasetDirPath, 
                         lenOfSourceDict=17372, lenOfSinkDict=7784, isMalware=True):
    imageRawHeightList = []
    if not os.path.isdir(imageDatasetDirPath):
        os.makedirs(imageDatasetDirPath)
    for apkDecompileDir in os.listdir(apkDecompileDatesetDirPath):
        featureListJsonFilePath = os.path.join(apkDecompileDatesetDirPath, 
                                        apkDecompileDir, "featureList.json")
        if os.path.isfile(featureListJsonFilePath):
            imageTensor, imageRawHeight = featureListToImageTensor(featureListJsonFilePath, 
                                                           lenOfSourceDict=17372, lenOfSinkDict=7784, 
                                                           width=4, height=300, maxHeight=2000)
            imageRawHeightList.append(imageRawHeight)
            

            
            if isMalware:
                imageTensorFilePath = os.path.join(imageDatasetDirPath, 
                                                   apkDecompileDir + "_0.pickle")
            else:
                imageTensorFilePath = os.path.join(imageDatasetDirPath,
                                                   apkDecompileDir + "_1.pickle")
            
            with open(imageTensorFilePath, "wb") as imageTensorFile:
                torch.save(imageTensor, imageTensorFile)
    
    # print(imageRawHeightList)
    dataFrame = pd.DataFrame(imageRawHeightList)
    # print(dataFrame)
    statisticsDataFrame = dataFrame.describe()
    # print(statisticsDataFrame)
    
    statisticsDataFrame.columns = ["imageRawHeight"]
    if isMalware:
        statisticsCSVFilePath = "DataStatistics/malwareImageRawHeightStatistics.csv"
        plotLabel = "malwareImageRawHeight"
    else:
        statisticsCSVFilePath = "DataStatistics/benignImageRawHeightStatistics.csv"
        plotLabel = "benignImageRawHeight"
    statisticsDataFrame.to_csv(statisticsCSVFilePath)
    histAndBoxPlot(imageRawHeightList, plotLabel)
    
    print(statisticsDataFrame)
        
if __name__ == "__main__":
#    featureListJsonFilePath = "F:\\test\\decompileDataset\\benign\\com.lenderprolink.sstewart\\featureList.json"
#    featureListToImage(featureListJsonFilePath)
    # windows
#    generateImageDataset("F:\\test\\decompileDataset\\malware", "F:\\test\\decompileDataset\\image")
    
    # linux
    generateImageDataset("/home/zhangxin/MyDroid/Dataset/2012/benign",
                         "/home/zhangxin/MyDroid/Dataset/2012/image", isMalware=False)
    generateImageDataset("/home/zhangxin/MyDroid/Dataset/2012/malware", 
                         "/home/zhangxin/MyDroid/Dataset/2012/image")