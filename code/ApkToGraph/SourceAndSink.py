# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 16:25:32 2019

@author: ZhangXin
"""
import re
from collections import OrderedDict

def readSourceAndSink(sourceFilePath="Source.txt", sinkFilePath="Sink.txt"):
    sourceDict = dict() # {entityMethodString:(permission, category)}
    sinkDict = dict()
    sourceDict = handleSourceAndSinkFile(sourceFilePath)
    sinkDict = handleSourceAndSinkFile(sinkFilePath)
    return sourceDict, sinkDict

def handleSourceAndSinkFile(filePath):
    sensitiveDict = dict() # {entityMethodString:(permission, category)}
    with open(filePath, "r") as file:
        lines = file.readlines()
        
    pattern = re.compile("<(.*?): (.*?) (.*?)\\((.*?)\\)> (.*?)\\((.*?)\\)")
    packageName, methodName, methodPermssion, methodcategory = "", "", "", ""
    for line in lines:
        line = line.strip()
        if line.startswith("<"):
            matcher = pattern.match(line)
            if matcher:
                packageName = matcher.group(1)
                methodName = matcher.group(3)
                methodPermssion = matcher.group(5)
                methodcategory = matcher.group(6)
                entityMethod =  packageName + ": " + methodName
                if len(methodName) > 0 and len(methodcategory) > 0:
                    sensitiveDict[entityMethod] = (methodPermssion, methodcategory)
    return sensitiveDict



def getEntityIdDict(sourceDict, sinkDict):
    
    idToEntityDict = OrderedDict()
    entityToIdDict = OrderedDict()
    id = 0
    for sourceEntity in sourceDict.keys():
        id += 1
        idToEntityDict[id] = sourceEntity
        entityToIdDict[sourceEntity] = id
    id = 0
    for sinkEntity in sinkDict.keys():
        id -= 1
        idToEntityDict[id] = sinkEntity
        entityToIdDict[sinkEntity] = id
    
    return idToEntityDict, entityToIdDict


if __name__ == "__main__":
    sourceDict, sinkDict = readSourceAndSink("Source.txt", "Sink.txt")
    print("number of source methods : %s" % len(sourceDict))
    print("number of sink methods : %s" % len(sinkDict))
#    print("sourceDict:")
#    print(sourceDict)
#    print("sinkDict:")
#    print(sinkDict)
#    idToEntityDict, entityToIdDict = getEntityIdDict(sourceDict, sinkDict)
#    print(idToEntityDict)
    
    
    
    
    