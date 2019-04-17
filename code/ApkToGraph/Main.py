# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 20:26:44 2019
顶层主函数
1. 处理每个apk（多线程）
2. 统计(良性、恶意)
3. 计算得到敏感度
4. 再次处理每个apk得到featureList（多线程）
5. 统计社区信息（根据社区信息，选取合适的图片形状） 
*****************
7. 根据统计的的社区信息，设计合适的CNN结构模型 处理 分类
8. 发掘统计敏感模式
@author: ZhangXin
"""
import os
from ApkToGraph import apkDatasetToGraphDataset
from SourceAndSink import readSourceAndSink, getEntityIdDict
from Statistics import statistics, getCoefficientOfSensitivityDict, statisticsOfCommunity
from CommunityDetection import apkDecompileDatasetToFeatureList
from GenerateImageDataset import generateImageDataset


# windows

#benignDatesetDirPath = "F:\\test\\dataset\\benign"
#malwareDatesetDirPath = "F:\\test\\dataset\\malware"
#benignDecompileDatesetDirPath = "F:\\test\\decompileDataset\\benign"
#malwareDecompileDatesetDirPath = "F:\\test\\decompileDataset\\malware"
#imageDatasetDirPath = "F:\\test\\decompileDataset\\image"

# linux

benignDatesetDirPath = "/home/zhangxin/Dataset/2013/benign"
malwareDatesetDirPath = "/home/zhangxin/Dataset/2013/malware"
benignDecompileDatesetDirPath = "/home/zhangxin/MyDroid/Dataset/2013/benign"
malwareDecompileDatesetDirPath = "/home/zhangxin/MyDroid/Dataset/2013/malware"
imageDatasetDirPath = "/home/zhangxin/MyDroid/Dataset/2013/image"

if __name__ == "__main__":
    if not os.path.isdir(benignDatesetDirPath):
        raise OSError("Benign apk directory %s does not exist." % (benignDatesetDirPath))
    if not os.path.isdir(malwareDatesetDirPath):
        raise OSError("Malware apk directory %s does not exist." % (malwareDatesetDirPath))
    print("Generate benign graph...")
    sourceDict, sinkDict = readSourceAndSink()
    apkDatasetToGraphDataset(benignDatesetDirPath, 
                             benignDecompileDatesetDirPath,
                             sourceDict, sinkDict)
    print("Generate malware graph...")
    apkDatasetToGraphDataset(malwareDatesetDirPath, 
                             malwareDecompileDatesetDirPath,
                             sourceDict, sinkDict)
    
    idToEntityDict, entityToIdDict = getEntityIdDict(sourceDict, sinkDict)
    print("Benign statistics...")
    benignIdToCountDict, numOfBenignApks = statistics(benignDecompileDatesetDirPath, 
                                                      idToEntityDict, entityToIdDict,
                                                      isMalware=False)
    print("Malware statistics...")
    malwareIdToCountDict, numOfMalwareApks = statistics(malwareDecompileDatesetDirPath, 
                                                        idToEntityDict, entityToIdDict,
                                                        isMalware=True)
    print("Calculate coefficient of sensitivity...")
    idToCOSDict, entityToCOSDict = getCoefficientOfSensitivityDict(malwareIdToCountDict,
                                                                   numOfMalwareApks, 
                                                                   benignIdToCountDict, 
                                                                   numOfBenignApks, 
                                                                   idToEntityDict, 
                                                                   entityToIdDict)
    print("Extract benign feature list...")
    apkDecompileDatasetToFeatureList(benignDecompileDatesetDirPath, entityToIdDict, idToCOSDict)
    print("Extract malware feature list...")
    apkDecompileDatasetToFeatureList(malwareDecompileDatesetDirPath, entityToIdDict, idToCOSDict)
    
    print("Benign community statistics...")
    statisticsOfCommunity(benignDecompileDatesetDirPath, isMalware=False)
    
    print("Malware community statistics...")
    statisticsOfCommunity(malwareDecompileDatesetDirPath)
    
    print("Generate benign image dataset...")
    generateImageDataset(benignDecompileDatesetDirPath, imageDatasetDirPath, 
                         len(sourceDict), len(sinkDict), isMalware=False)
    print("Generate malware image dataset...")
    generateImageDataset(malwareDecompileDatesetDirPath, imageDatasetDirPath, 
                         len(sourceDict), len(sinkDict))
    
    print("Analyse done!")







