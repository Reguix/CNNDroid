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
2. 统计数据集中敏感API出现的次数，计算得到敏感度（只需要执行一次）
3. 社交网络检测调用图提取特征（社交网络划分进行统计分析，确定选取图像尺寸）
4. 将特征转变为规则图像

"""
import torch
import time
import os
from ApkToGraph import apkDatasetToGraphDataset
from SourceAndSink import readSourceAndSink, getEntityIdDict
from Statistics import getCOS, statisticsOfCommunity
from CommunityDetection import apkDecompileDatasetToFeatureList
from GenerateImage import generateImageDataset
from Utils import cleanDir

#参数
# windows

#benignDatesetDirPath = "F:\\test\\dataset\\benign"
#malwareDatesetDirPath = "F:\\test\\dataset\\malware"
#benignDecompileDatesetDirPath = "F:\\test\\decompileDataset\\benign"
#malwareDecompileDatesetDirPath = "F:\\test\\decompileDataset\\malware"
#imageDatasetDirPath = "F:\\test\\decompileDataset\\image"

# linux

benignDatesetDirPath = "/home/zx/Dataset/2018/benign"
malwareDatesetDirPath = "/home/zx/Dataset/2018/malware"
benignDecompileDatesetDirPath = "/home/zx/CNNDroid/Dataset/2018/benign"
malwareDecompileDatesetDirPath = "/home/zx/CNNDroid/Dataset/2018/malware"
imageDatasetDirPath = "/home/zx/CNNDroid/Dataset/2018/statistics_image_800"


def main():
    # 检查数据集文件夹是否存在
#    if not os.path.isdir(benignDatesetDirPath):
#        raise OSError("Benign apk directory %s does not exist." % (benignDatesetDirPath))
#    if not os.path.isdir(malwareDatesetDirPath):
#        raise OSError("Malware apk directory %s does not exist." % (malwareDatesetDirPath))
    
    # 从Source.txt和Sink.txt解析敏感API
    sourceDict, sinkDict = readSourceAndSink() 
    
    # 对敏感API进行编号
    idToEntityDict, entityToIdDict = getEntityIdDict(sourceDict, sinkDict)
    
    # 生成调用图
#    print("Generate benign dataset graph...")
#    apkDatasetToGraphDataset(benignDatesetDirPath, 
#                             benignDecompileDatesetDirPath,
#                             sourceDict, sinkDict,
#                             idToEntityDict, entityToIdDict)
#    print("Generate malware dataset graph...")
#    apkDatasetToGraphDataset(malwareDatesetDirPath, 
#                             malwareDecompileDatesetDirPath,
#                             sourceDict, sinkDict,
#                             idToEntityDict, entityToIdDict)
    # 删除旧的文件
#    if os.path.isfile("idToCOSDict.json"):
#        print("Delete old idToCOSDict.json")
#        os.unlink("idToCOSDict.json")
#        os.unlink("entityToCOSDict.json")
    # 清空之前的统计数据
    cleanDir("DataStatistics/")
    # 获取敏感度
    print("Get coefficient of sensitivity...")
    idToCOSDict, entityToCOSDict = getCOS(benignDecompileDatesetDirPath, 
                                          malwareDecompileDatesetDirPath, 
                                          idToEntityDict, entityToIdDict)
    
    # 社交网络检测，提取特征
    print("Extract benign dataset feature list...")
    apkDecompileDatasetToFeatureList(benignDecompileDatesetDirPath, entityToIdDict, idToCOSDict)
    print("Extract malware dataset feature list...")
    apkDecompileDatasetToFeatureList(malwareDecompileDatesetDirPath, entityToIdDict, idToCOSDict)
    
    # 对社交网络的检测结果进行统计分析（正式流程可省略）
    print("Benign dataset community statistics...")
    statisticsOfCommunity(benignDecompileDatesetDirPath, isMalware=False)
    
    print("Malware dataset community statistics...")
    statisticsOfCommunity(malwareDecompileDatesetDirPath)
  
#    # 将特征转变为图像
    print("Generate benign image dataset...")
    generateImageDataset(benignDecompileDatesetDirPath, imageDatasetDirPath, isMalware=False)
    print("Generate malware image dataset...")
    generateImageDataset(malwareDecompileDatesetDirPath, imageDatasetDirPath)
    
    print("Analyse done!")

if __name__ == "__main__":
    time_start=time.time()
    main()
    time_end=time.time()
#    print('time cost',time_end-time_start,'s')
#    with open("F:\\test\\decompileDataset\\image\\com.just4fun.spiderinphone_0.pickle", "rb") as f:
#        image = torch.load(f)
#        print(image[:8])
#        print(image[-8:])
#        print(image.shape)