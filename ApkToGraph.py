# -*- coding: utf-8 -*-
"""
参数：
-i 包含APK文件的目录路径
-o 输出的目录路径

功能：
1. apkDatasetToGraphDataset()
   包装的顶层函数接口，多线程处理APK数据集生成对应的调用图数据集

2. oneApkToGraph()
   对一个APK文件进行反汇编，提取调用图，并生成每个调用图的信息概述文件information.json
   information.json保存了调用图的节点数目，边数目，敏感节点数目，和敏感节点出现次数统计

3. apktool()
   调用apktool工具，对APK文件进行反汇编

4. smaliToGraph()
   对反编译后的文件进行处理，得到调用图对象，保存文件为sourceGraph.gexf
   
   节点的标签格式为{<_: _ _(_)>,_,_,_}，具体如下：
   
   {<packageName: returnValue methodName(methodArgs)>,methodType,methodPermssion,methodcategory}

"""
import sys
import json
import os
import re
import send2trash
import argparse
import shutil
from collections import OrderedDict
import networkx as nx
from multiprocessing import Pool
from functools import partial

from SourceAndSink import readSourceAndSink
from Utils import getEntity
import time

# 多线程处理APK数据集,生成调用图数据集
def apkDatasetToGraphDataset(apkDatesetDirPath, apkDecompileDatesetDirPath, 
                             sourceDict, sinkDict,
                             idToEntityDict, entityToIdDict):
    # 如果该目录不存在，则新建目录
    if not os.path.isdir(apkDecompileDatesetDirPath):
        os.makedirs(apkDecompileDatesetDirPath)
    # 要反汇编的APK文件路径列表
    apkFilePathList = []    
    for apkFileBasename in os.listdir(apkDatesetDirPath):
        if apkFileBasename.endswith(".apk"):
            apkFilename, _ = os.path.splitext(apkFileBasename)
            outputPath = os.path.join(apkDecompileDatesetDirPath, apkFilename)
            # 如果当前apk的反汇编文件夹已经存在，就不添加该apk文件任务，避免重复
            if not os.path.exists(outputPath):
                apkFilePath = os.path.join(apkDatesetDirPath, apkFileBasename)
                apkFilePathList.append(apkFilePath)
    # 多线程处理
    pool = Pool()
    pool.map(partial(oneApkToGraph, apkDecompileDatesetDirPath=apkDecompileDatesetDirPath, 
                     sourceDict=sourceDict, sinkDict=sinkDict,
                     idToEntityDict=idToEntityDict, 
                     entityToIdDict=entityToIdDict), apkFilePathList)

# 处理单个APK文件，进行反汇编，并提取调用图
def oneApkToGraph(apkFilePath, apkDecompileDatesetDirPath, sourceDict, sinkDict,
                  idToEntityDict, entityToIdDict):
    apkFileBasename = os.path.basename(apkFilePath)
    apkFilename, _ = os.path.splitext(apkFileBasename)
    outputPath = os.path.join(apkDecompileDatesetDirPath, apkFilename)
    timeDict = OrderedDict()
    timeDict["apkFilename"] = apkFilename
    # 如果目录已经存在就不再处理，直接返回
    if os.path.isdir(outputPath):
        return timeDict
    # 调用apktool反汇编
    timeStart = time.time()
    apktool(apkFilePath, outputPath)
    timeDict["apktoolTime"] = time.time() - timeStart
    # 生成调用图
    timeStart = time.time()
    sourceGraph, sensitiveNodeDict = smaliToGraph(outputPath, sourceDict, sinkDict) 
    timeDict["graphTime"] = time.time() - timeStart
    # 反汇编失败的话，将文件夹删除
    if sourceGraph.number_of_nodes == 0:
        shutil.rmtree(outputPath)
        return timeDict
    # 没有敏感节点的话，将文件夹删除
    # print(sensitiveNodeDict)
    sensitiveNodeSet = sensitiveNodeDict.keys()
    if len(sensitiveNodeSet) == 0:
        shutil.rmtree(outputPath)
        return timeDict
    # "info" means "information"
    # 记录调用图的信息
    infoOrderedDict = OrderedDict()
    # "SG" means "source graph" 
    # 记录调用图的节点数目，边数目，敏感节点数目
    infoOrderedDict["numOfSGNodes"] = sourceGraph.number_of_nodes()
    infoOrderedDict["numOfSGEdges"] = sourceGraph.number_of_edges()
    infoOrderedDict["numOfSenNodesInSG"] = len(sensitiveNodeSet)
    # 把敏感节点映射为id
    sensitiveNodeIdDict = OrderedDict()
    for sensitiveNodeId in idToEntityDict.keys():
        sensitiveNodeIdDict[sensitiveNodeId] = 0
    for (sensitiveNodeLabel, count) in sensitiveNodeDict.items():
        sensitiveNodeEntity = getEntity(sensitiveNodeLabel)
        sensitiveNodeId = entityToIdDict[sensitiveNodeEntity]
        sensitiveNodeIdDict[sensitiveNodeId] += count
        
    infoOrderedDict["sensitiveNodeIdDict"] = sensitiveNodeIdDict
    # infoOrderedDict["sensitiveNodeDict"] = sensitiveNodeDict
    # 将调用图的信息：节点数目、边数目、敏感节点数目、敏感节点字典，放在一个json格式的文件中
    infoJsonFilePath = os.path.join(outputPath, "information.json")
    with open(infoJsonFilePath, "w") as infoJsonFile:
        json.dump(infoOrderedDict, infoJsonFile)
    timeDict["nodes"] = infoOrderedDict["numOfSGNodes"]
    timeDict["edges"] = infoOrderedDict["numOfSGEdges"]
    timeDict["senNodes"] = infoOrderedDict["numOfSenNodesInSG"]
    return timeDict
     
# 反汇编APK文件    
def apktool(apkFilePath, outputPath):
    if os.path.exists(outputPath):
        send2trash.send2trash(outputPath)
        #shutil.rmtree(outputPath)
    cmd = "java -Djava.awt.headless=true -jar apktool_2.4.1.jar d "\
          + apkFilePath + " -o " + outputPath
    if sys.platform.startswith("win"):
        cmd = cmd + " > nul 2>&1"
    else:
        cmd = cmd + " > /dev/null 2>&1"
    os.system(cmd)

# 从反汇编代码中提取调用图
def smaliToGraph(apkDecompileDirPath, sourceDict, sinkDict):
    # 所有的smali文件的调用边列表
    methodClassList = []
    # 遍历解析所有的.smali文件
    smaliDirPath = os.path.join(apkDecompileDirPath, "smali")
    for root, dirs, files in os.walk(smaliDirPath):
        for file in files:
            if file.endswith(".smali"):
                smaliFilePath = os.path.join(root, file)
                # print(smaliFilePath)
                # 得到一个类（一个.smail文件）中的方法调用列表
                methodCallingList = handleSmaliFile(smaliFilePath,
                                                         sourceDict, sinkDict)
                # 汇总所有的smali文件的调用列表
                methodClassList.append(methodCallingList)
    # 将解析的结果保存为gexf格式
    graphGexfFilePath = os.path.join(apkDecompileDirPath, "sourceGraph.gexf")
    # 将解析的结果保存为调用边的txt格式
    #graphTxtFilePath = os.path.join(apkDecompileDirPath, "sourceGraph.txt")
    #graphTxtFile = open(graphTxtFilePath, "w")
    # networkx格式的调用图
    DG = nx.DiGraph()
    # 敏感节点出现的次数字典
    sensitiveNodeDict = dict()
    for methodCallingList in methodClassList:
        for (callerNodeLabel, calleeNodeLabel) in methodCallingList:
            # 将边写入sourceGraph.txt文件
            #callingEdgeLabel = callerNodeLabel + " ==> " + calleeNodeLabel
            #graphTxtFile.write(callingEdgeLabel + "\n")
            # networkx格式的调用图添加边
            DG.add_edge(callerNodeLabel, calleeNodeLabel)
            # 设置gexf节点的颜色，正常节点天蓝色，source节点红色，sink节点橘色
            normalColor = {'color': {'r': 30, 'g': 144, 'b': 255, 'a': 0}}
            sourceColor = {'color': {'r': 255, 'g': 69, 'b': 0, 'a': 0}}
            sinkColor = {'color': {'r': 255, 'g': 158, 'b': 53, 'a': 0}}
            # 设置调用节点的颜色
            callerNodeType = callerNodeLabel.split(",")[-3]
            if callerNodeType == "Sink" or callerNodeType == "Source":
                if callerNodeType == "Sink":
                    DG.nodes[callerNodeLabel]['viz'] = sinkColor
                if callerNodeType == "Source":
                    DG.nodes[callerNodeLabel]['viz'] = sourceColor
                # 统计敏感API出现的次数
                if callerNodeLabel not in sensitiveNodeDict.keys():
                    sensitiveNodeDict[callerNodeLabel] = 1
                else:
                    sensitiveNodeDict[callerNodeLabel] += 1 
            else:
                DG.nodes[callerNodeLabel]['viz'] = normalColor
            # 设置被调用节点的颜色
            calleeNodeType = calleeNodeLabel.split(",")[-3]
            if calleeNodeType == "Sink" or calleeNodeType == "Source":
                if calleeNodeType == "Sink":
                    DG.nodes[calleeNodeLabel]['viz'] = sinkColor
                if calleeNodeType == "Source":
                    DG.nodes[calleeNodeLabel]['viz'] = sourceColor
                # 统计敏感API出现的次数
                if calleeNodeLabel not in sensitiveNodeDict.keys():
                    sensitiveNodeDict[calleeNodeLabel] = 1
                else:
                    sensitiveNodeDict[calleeNodeLabel] += 1
            else:
                DG.nodes[calleeNodeLabel]['viz'] = normalColor
    #graphTxtFile.close()
    nx.write_gexf(DG, graphGexfFilePath)
    # 清理反汇编的文件夹，只留下调用图
#    cleanDecompileDir(apkDecompileDirPath)
    return DG, sensitiveNodeDict

# 清理文件夹，只保留列表中的内容
def cleanDecompileDir(apkDecompileDirPath):
    remainList = ["sourceGraph.gexf"]
    for fileName in os.listdir(apkDecompileDirPath):
        if fileName not in remainList:
            filePath = os.path.join(apkDecompileDirPath, fileName)
            if os.path.isfile(filePath):
                os.unlink(filePath)
            if os.path.isdir(filePath):
                shutil.rmtree(filePath)

# 将smail语法中的关键字进行替换
def replaceKeyWords(string):
    if string.startswith("L"):
        string = string[1:]
    if string == "I":
        string = "java.lang.Integer"
    if string == "Z":
        string = "boolean"
    if string == "V":
        string = "void"
    string = string.replace("/", ".").replace(";", "").replace("<", "").replace(">", "")
    return string

# 获取该方法的类型、权限和类别    
def getSourceAndSinkInfo(entityMethod, sourceDict, sinkDict):
    methodType, methodPermssion, methodCategory = "", "", ""
    if entityMethod in sourceDict.keys():
        methodType = "Source"
        methodPermssion = sourceDict[entityMethod][0]
        methodCategory = sourceDict[entityMethod][1]
    elif entityMethod in sinkDict.keys():
        methodType = "Sink"
        methodPermssion = sinkDict[entityMethod][0]
        methodCategory = sinkDict[entityMethod][1]
    else:
        methodType = "Normal"
    
    return methodType + "," + methodPermssion + "," + methodCategory

# 解析method语句
def handleSmaliMethod(line, packageName, sourceDict, sinkDict):
    callerNodeLabel = ""
    pattern = re.compile(".method(.*) (.*?)\\((.*?)\\)(.*)")
    matcher = pattern.match(line)

    if matcher:
        argList = matcher.group(3).split(";")
        argList = [arg for arg in argList if arg != ""]
        for idx in range(len(argList)):
            argList[idx] = replaceKeyWords(argList[idx])
        
        methodArgs = ",".join(argList)
        methodName = replaceKeyWords(matcher.group(2))
        returnValue = replaceKeyWords(matcher.group(4))
        
        callerMethod = ("<" + packageName + ": " + returnValue + " "
                        + methodName + "(" + methodArgs + ")>")
        entityMethod =  packageName + ": "+ methodName
        sourceAndSinkInfo = getSourceAndSinkInfo(entityMethod, 
                                                 sourceDict, sinkDict)
        callerNodeLabel = "{" + callerMethod + "," + sourceAndSinkInfo + "}"    
    
    return callerNodeLabel

# 解析invoke语句
def handleSmaliInvoke(line, sourceDict, sinkDict):
    calleeNodeLabel = ""
    pattern = re.compile("invoke-(.*?) \\{.*\\}, (.*?);->(.*?)\\((.*?)\\)(.*)")
    matcher = pattern.match(line)
    
    if matcher:
        argList = matcher.group(4).split(";")
        argList = [arg for arg in argList if arg != ""]
        for idx in range(len(argList)):
            argList[idx] = replaceKeyWords(argList[idx])
        
        packageName = replaceKeyWords(matcher.group(2))
        methodArgs = ",".join(argList)
        methodName = replaceKeyWords(matcher.group(3))
        returnValue = replaceKeyWords(matcher.group(5))
        
        calleeMethod = ("<" + packageName + ": " + returnValue + " "
                        + methodName + "(" + methodArgs + ")>")
        entityMethod =  packageName + ": " + methodName
        sourceAndSinkInfo = getSourceAndSinkInfo(entityMethod, 
                                                 sourceDict, sinkDict)
        calleeNodeLabel = "{" + calleeMethod + "," + sourceAndSinkInfo + "}"
    
    return calleeNodeLabel

# 解析整个smali文件中的调用列表
def handleSmaliFile(smaliFilePath, sourceDict, sinkDict):
    methodCallingList = []
    packageName = ""
    with open(smaliFilePath, "r", encoding='UTF-8') as smaliFile:
        lines = smaliFile.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith(".class"):
            packageName = replaceKeyWords(line.split(" ")[-1])
        if line.startswith(".method"):
            callerNodeLabel = handleSmaliMethod(line, packageName, sourceDict, sinkDict)
            while line.startswith(".end method") == False:
                if line.startswith("invoke"):
                    calleeNodeLabel = handleSmaliInvoke(line, sourceDict, sinkDict)
                    if calleeNodeLabel != "":
                        methodCallingList.append((callerNodeLabel, calleeNodeLabel))
                i += 1
                line = lines[i].strip()
        i += 1
    return methodCallingList                   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Android Apks To Call Graph.")
    parser.add_argument("-i", "--input", help="a directory with the original APKs to analyze.", type=str, required=True) 
    parser.add_argument("-o", "--output", help="a directory with the output APKs.", type=str, required=True)
    args = parser.parse_args()
    
    apkDatesetDirPath = args.input
    apkDecompileDatesetDirPath = args.output
    
    if not os.path.isdir(apkDatesetDirPath):
        raise OSError("Input directory %s does not exist." % (args.input))
    if not os.path.isdir(apkDecompileDatesetDirPath):
        raise OSError("Output directory %s does not exist." % (args.output))
    
    sourceDict, sinkDict = readSourceAndSink()
    apkDatasetToGraphDataset(apkDatesetDirPath, apkDecompileDatesetDirPath, sourceDict, sinkDict) 
                
    
