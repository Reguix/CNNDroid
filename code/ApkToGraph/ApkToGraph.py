# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 09:55:55 2018
1.输入apk文件路径，调用apktool工具，对APK文件进行反编译
2.对反编译后的文件进行处理，得到调用图文件，以及networkx的graph对象
  调用图的形式为{<_: _ _(_)>,_,_,_} ==> {<_: _ _(_)>,_,_,_}
  {<packageName: returnValue methodName(methodArgs)>,methodType,methodPermssion,methodcategory}
    
@author: ZhangXin
"""
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


def oneApkToGraph(apkFilePath, outputDirPath, sourceDict, sinkDict):
    
    apkFileBasename = os.path.basename(apkFilePath)
    apkFilename, _ = os.path.splitext(apkFileBasename)
    outputPath = os.path.join(outputDirPath, apkFilename)
    apktool(apkFilePath, outputPath)
    
    sourceGraph, sensitiveNodeDict = smaliToGraph(outputPath, sourceDict, sinkDict) 
    if sourceGraph.number_of_nodes == 0:
        shutil.rmtree(outputPath)
        return
#    print(sensitiveNodeDict)
    sensitiveNodeSet = sensitiveNodeDict.keys()
    if len(sensitiveNodeSet) == 0:
        shutil.rmtree(outputPath)
        return
#    print("source graph:")
#    print("number of sensitive nodes in sourceGraph: %s " % len(sensitiveNodeSet))
#    print("number of nodes in sourceGraph: %s" % sourceGraph.number_of_nodes())
#    print("number of edges in sourceGraph: %s" % sourceGraph.number_of_edges())

    # "wccsg" means "weakly connected component subgraph"
    sourceGraphWccsgList = sorted(nx.weakly_connected_component_subgraphs(sourceGraph), key=len, reverse=True)
#    sourceGraphWccsgLenList = [len(c) for c in sourceGraphWccsgList]
#    print("number of weakly connected component subgraph in source graph: %s " % len(sourceGraphWccsgLenList))
#    print("each weakly connected component subgraph size in source graph:")
#    print(sourceGraphWccsgLenList)
    
    # "sen" mean "sensitive", "num" means "number"
    numOfSenNodesInSourceGraphWccsgList = []

    sensitiveGraph = sourceGraph.copy()
    
    sensitiveGraphWccsgList = []
    sensitiveGraphWccsgLenList = []
    numOfSenNodesInSenGraphWccsgList = []
    
    
    for subGraph in sourceGraphWccsgList:
        num = getNumOfSenNodesInGraph(subGraph, sensitiveNodeSet)
        numOfSenNodesInSourceGraphWccsgList.append(num)
        if num != 0:
            sensitiveGraphWccsgList.append(subGraph)
            sensitiveGraphWccsgLenList.append(len(subGraph))
            numOfSenNodesInSenGraphWccsgList.append(num)
        else:
            sensitiveGraph.remove_nodes_from(subGraph.nodes)
    
#    print("number of sensitive in each weakly connected component subgraph of source graph:")
#    print(numOfSenNodesInSourceGraphWccsgList)
    
#    print("sensitive graph:")
#    print("number of sensitive nodes in sourceGraph: %s " % getNumOfSenNodesInGraph(sensitiveGraph, sensitiveNodeSet))     
#    print("number of nodes in sensitive graph: %s" % sensitiveGraph.number_of_nodes())
#    print("number of edges in sensitive graph: %s" % sensitiveGraph.number_of_edges())
#    print("number of weakly connected component subgraph in sensitive graph: %s " % len(sensitiveGraphWccsgLenList))
#    print("each weakly connected component subgraph size in sensitive graph:")
#    print(sensitiveGraphWccsgLenList)
#    print("number of sensitive in each weakly connected component subgraph of sensitive graph:")
#    print(numOfSenNodesInSenGraphWccsgList)
    
#    sensitiveGraphWccsgLenList2 = [len(c) for c in sorted(nx.weakly_connected_component_subgraphs(sensitiveGraph), key=len, reverse=True)]
#    if sensitiveGraphWccsgLenList2 == sensitiveGraphWccsgLenList:
#        print("True!")
#    else: print("False!")
    
    if len(sensitiveGraphWccsgList) == 0:
        shutil.rmtree(outputPath)
        return
    
    sensitiveGraphGexfFilePath = os.path.join(outputPath, "sensitiveGraph.gexf")
    nx.write_gexf(sensitiveGraph, sensitiveGraphGexfFilePath)
    
    maxSenWccsgGecfFilePath = os.path.join(outputPath, "maxSensitiveSubGraph.gexf")
    nx.write_gexf(sensitiveGraphWccsgList[0], maxSenWccsgGecfFilePath)
    
    # "info" means "information"
    infoOrderedDict = OrderedDict()
    
    # "SG" means "source graph" 
    # "SSG" means "sensitive subgraph"
    # "MSSG" means "max sensitive subgraph", the max weakly connected component subgraph of sensitive subgraph
    
    infoOrderedDict["numOfSGNodes"] = sourceGraph.number_of_nodes()
    infoOrderedDict["numOfSGEdges"] = sourceGraph.number_of_edges()
    infoOrderedDict["numOfSenNodesInSG"] = len(sensitiveNodeSet)
    
    infoOrderedDict["numOfSSGNodes"] = sensitiveGraph.number_of_nodes()
    infoOrderedDict["numOfSSGEdges"] = sensitiveGraph.number_of_edges()
    infoOrderedDict["numOfSenNodesInSSG"] = len(sensitiveNodeSet)
    
#    infoOrderedDict["percentageOfSSGNode"] = sensitiveGraph.number_of_nodes() / sourceGraph.number_of_nodes()
#    infoOrderedDict["percentageOfSSGEdge"] = sensitiveGraph.number_of_edges() / sourceGraph.number_of_nodes()
    
    infoOrderedDict["numOfMSSGNodes"] = sensitiveGraphWccsgList[0].number_of_nodes()
    infoOrderedDict["numOfMSSGEdges"] = sensitiveGraphWccsgList[0].number_of_edges()
    infoOrderedDict["numOfSenNodesInMSSG"] = numOfSenNodesInSenGraphWccsgList[0]
    
#    infoOrderedDict["percentageOfMSSGNode"] = sensitiveGraphWccsgList[0].number_of_nodes() / sourceGraph.number_of_nodes()
#    infoOrderedDict["percentageOfMSSGEdge"] = sensitiveGraphWccsgList[0].number_of_edges() / sourceGraph.number_of_nodes()
    
#    infoOrderedDict["percentageOfSenNodeInMSSG"] = numOfSenNodesInSenGraphWccsgList[0] / len(sensitiveNodeSet)
    infoOrderedDict["sensitiveNodeDict"] = sensitiveNodeDict
    
    infoJsonFilePath = os.path.join(outputPath, "information.json")
    with open(infoJsonFilePath, "w") as infoJsonFile:
        json.dump(infoOrderedDict, infoJsonFile)
     

def getNumOfSenNodesInGraph(graph, sensitiveNodeSet):
    num = 0
    for node in graph.nodes:
        if node in sensitiveNodeSet:
            num += 1
    return num
    
def isSensitiveGraph(graph, sensitiveNodeSet):
    for node in graph.nodes:
        if node in sensitiveNodeSet:
            return True
    return False
    
def apktool(apkFilePath, outputPath):
    if os.path.exists(outputPath):
        send2trash.send2trash(outputPath)
        # shutil.rmtree(outputPath)
    cmd = "java -Djava.awt.headless=true -jar apktool_2.3.4.jar d " + apkFilePath + " -o " + outputPath
    os.system(cmd)

def smaliToGraph(apkDecompileDirPath, sourceDict, sinkDict):
    
    normalColor = {'color': {'r': 30, 'g': 144, 'b': 255, 'a': 0}}
    # sensitiveColor = {'color': {'r': 255, 'g': 0, 'b': 0, 'a': 0}}
    sourceColor = {'color': {'r': 255, 'g': 69, 'b': 0, 'a': 0}}
    sinkColor = {'color': {'r': 255, 'g': 158, 'b': 53, 'a': 0}}
    
    DG = nx.DiGraph()
    graphTxtFilePath = os.path.join(apkDecompileDirPath, "sourceGraph.txt")
    graphGexfFilePath = os.path.join(apkDecompileDirPath, "sourceGraph.gexf")
    graphTxtFile = open(graphTxtFilePath, "w")
    
    methodClassList = []
    
    # sensitiveNodeSet = set()
    sensitiveNodeDict = dict()
    
    smaliDirPath = os.path.join(apkDecompileDirPath, "smali")
    for root, dirs, files in os.walk(smaliDirPath):
        for file in files:
            if file.endswith(".smali"):
                smaliFilePath = os.path.join(root, file)
                # print(smaliFilePath)
                methodCallingList = handleSmaliFile(smaliFilePath,
                                                         sourceDict, sinkDict)
                methodClassList.append(methodCallingList)
    # print("number of smali files : %s" % len(methodClassList))
    for methodCallingList in methodClassList:
        for (callerNodeLabel, calleeNodeLabel) in methodCallingList:
            callingEdgeLabel = callerNodeLabel + " ==> " + calleeNodeLabel
            graphTxtFile.write(callingEdgeLabel + "\n")
            DG.add_edge(callerNodeLabel, calleeNodeLabel)
            callerNodeType = callerNodeLabel.split(",")[-3]
            calleeNodeType = calleeNodeLabel.split(",")[-3]
            if callerNodeType == "Sink" or callerNodeType == "Source":
                if callerNodeType == "Sink":
                    DG.nodes[callerNodeLabel]['viz'] = sinkColor
                if callerNodeType == "Source":
                    DG.nodes[callerNodeLabel]['viz'] = sourceColor
                
                # sensitiveNodeSet.add(callerNodeLabel)
                if callerNodeLabel not in sensitiveNodeDict.keys():
                    sensitiveNodeDict[callerNodeLabel] = 1
                else:
                    sensitiveNodeDict[callerNodeLabel] += 1 
            else:
                DG.nodes[callerNodeLabel]['viz'] = normalColor
            if calleeNodeType == "Sink" or calleeNodeType == "Source":
                if calleeNodeType == "Sink":
                    DG.nodes[calleeNodeLabel]['viz'] = sinkColor
                if calleeNodeType == "Source":
                    DG.nodes[calleeNodeLabel]['viz'] = sourceColor
                # sensitiveNodeSet.add(calleeNodeLabel)
                if calleeNodeLabel not in sensitiveNodeDict.keys():
                    sensitiveNodeDict[calleeNodeLabel] = 1
                else:
                    sensitiveNodeDict[calleeNodeLabel] += 1
            else:
                DG.nodes[calleeNodeLabel]['viz'] = normalColor
    graphTxtFile.close()
    nx.write_gexf(DG, graphGexfFilePath)
    
    cleanDecompileDir(apkDecompileDirPath)

    return DG, sensitiveNodeDict
    
def cleanDecompileDir(apkDecompileDirPath):
    fileList = ["AndroidManifest.xml", "sourceGraph.gexf", "sourceGraph.txt"]
    for fileName in os.listdir(apkDecompileDirPath):
        if fileName not in fileList:
            filePath = os.path.join(apkDecompileDirPath, fileName)
            if os.path.isfile(filePath):
                os.unlink(filePath)
            if os.path.isdir(filePath):
                shutil.rmtree(filePath)


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

def handleSmaliFile(smaliFilePath, sourceDict, sinkDict):
    methodCallingList = []
    packageName = ""
    with open(smaliFilePath, "r") as smaliFile:
        lines = smaliFile.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # print(line)
        if line.startswith(".class"):
            packageName = replaceKeyWords(line.split(" ")[-1])
            # print("packageName is %s" % packageName)
        if line.startswith(".method"):
            callerNodeLabel = handleSmaliMethod(line, packageName, sourceDict, sinkDict)
            # print(callerNodeLabel)
            while line.startswith(".end method") == False:
                if line.startswith("invoke"):
                    calleeNodeLabel = handleSmaliInvoke(line, sourceDict, sinkDict)
                    # print(callerNodeLabel + " ==> " + calleeNodeLabel)
                    if calleeNodeLabel != "":
                        methodCallingList.append((callerNodeLabel, calleeNodeLabel))
                i += 1
                line = lines[i].strip()
        i += 1
    return methodCallingList                   

def apkDatasetToGraphDataset(apkDatesetDirPath, apkDecompileDatesetDirPath, sourceDict, sinkDict):
    
    apkFilePathList = []
    
    for apkFileBasename in os.listdir(apkDatesetDirPath):
        if apkFileBasename.endswith(".apk"):
            apkFilename, _ = os.path.splitext(apkFileBasename)
            apkFilePath = os.path.join(apkDatesetDirPath, apkFileBasename)
            outputPath = os.path.join(apkDecompileDatesetDirPath, apkFilename)
            if not os.path.exists(outputPath):
                apkFilePathList.append(apkFilePath)
    pool = Pool()
    pool.map(partial(oneApkToGraph, outputDirPath=apkDecompileDatesetDirPath, 
                     sourceDict=sourceDict, sinkDict=sinkDict), apkFilePathList)


if __name__ == "__main__":
#    parser = argparse.ArgumentParser(description = "Handle Android Apps.")
#    parser.add_argument("-i", "--input", help="a directory with the original APKs to analyze.", type=str, required=True) 
#    parser.add_argument("-o", "--output", help="a directory with the output APKs.", type=str, required=True)
#    args = parser.parse_args()
#    
#    apkDatesetDirPath = args.input
#    apkDecompileDatesetDirPath = args.output
#    
#    if not os.path.isdir(apkDatesetDirPath):
#        raise OSError("Input directory %s does not exist." % (args.input))
#    if not os.path.isdir(apkDecompileDatesetDirPath):
#        raise OSError("Output directory %s does not exist." % (args.output))
#    
#    sourceDict, sinkDict = readSourceAndSink()
#    apkDatasetToGraphDataset(apkDatesetDirPath, apkDecompileDatesetDirPath, sourceDict, sinkDict)

    

    # test handleApkDataset
    
#    sourceDict, sinkDict = readSourceAndSink()
#    print("number of source methods : %s" % len(sourceDict))
#    print("number of sink methods : %s" % len(sinkDict))    
#    apkDatasetToGraphDataset("F:\\test\\dataset\\benign", 
#                             "F:\\test\\decompileDateset\\benign",
#                             sourceDict, sinkDict)

    # test one apk
    sourceDict, sinkDict = readSourceAndSink()
    print("number of source methods : %s" % len(sourceDict))
    print("number of sink methods : %s" % len(sinkDict))  
    oneApkToGraph("F:\\nograph.apk", "F:\\nograph", sourceDict, sinkDict)       
        
        
            
                
    
