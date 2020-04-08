# -*- coding: utf-8 -*-
"""
无参数

功能：
1. readSourceAndSink()是从Source.txt和Sink.txt解析得到两类敏感API
   返回两个字典sourceDict和sinkDict。
   返回值字典格式：键为字符串entityMethod，值为元组(methodPermssion, methodcategory)
   entityMethod为方法实体字符串，唯一标识一个API，格式为“packageName: methodName”，即“包名: 方法名”
   (methodPermssion, methodcategory)为（方法权限，方法类别）
   

2. getEntityIdDict()是将解析得到敏感API字典按顺序编号
   对于Source类API，从1开始递增编号，为1~17372
   对于Sink类从API，从-1开始递减编号，为-1~-7784
   返回两个字典idToEntityDict和entityToIdDict
   idToEntityDict的键为编号，值为方法实体
   entityToIdDict的键为方法实体，值为编号

3. getNumOfCategory()是统计敏感类别中包含的敏感API的数目

4. genCategoryCSV()是将敏感类别的数目写入csv文件
"""
import re
from collections import OrderedDict

def readSourceAndSink(sourceFilePath="Source.txt", sinkFilePath="Sink.txt"):
    sourceDict = dict() # {entityMethodString:(permission, category)}
    sinkDict = dict()
    sourceDict = handleSourceAndSinkFile(sourceFilePath)
    sinkDict = handleSourceAndSinkFile(sinkFilePath)
    

    return sourceDict, sinkDict

# 从.txt文件中解析得到敏感API，返回字典
def handleSourceAndSinkFile(filePath):
    sensitiveDict = dict() # {entityMethodString:(permission, category)}
    with open(filePath, "r") as file:
        lines = file.readlines()
        
    pattern = re.compile("<(.*?): (.*?) (.*?)\\((.*?)\\)> (.*?)\\((.*?)\\)")
    packageName, methodName, methodPermssion, methodcategory = "", "", "", ""
    
    for line in lines:
        line = line.strip()
        # 解析以<开头的行
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

#统计不同类别的敏感API的数目
def getNumOfCategory(sensitiveDict):
    category = ""
    categoryToNumDict = dict()
    for (_ , category) in sensitiveDict.values():
        if category not in categoryToNumDict.keys():
            categoryToNumDict[category] = 1
        else:
            categoryToNumDict[category] += 1
    return categoryToNumDict

#将类别的敏感API的数目的写入CSV文件
def genCategoryCSV(sourceCategoryDict, sinkCategoryDict, CSVFilePath):
    CSVFile = open(CSVFilePath, "w")
    categorySet = set(sourceCategoryDict.keys()) | set(sinkCategoryDict.keys())
    CSVFile.write("Category,Number of Source,Number of Sink\n")
    for category in categorySet:
        CSVFile.write(category + ",")
        if category in sourceCategoryDict.keys():
            CSVFile.write(str(sourceCategoryDict[category])+",")
        else:
            CSVFile.write("-,")
        if category in sinkCategoryDict.keys():
            CSVFile.write(str(sinkCategoryDict[category])+"\n")
        else:
            CSVFile.write("-\n")
    CSVFile.close()

#将解析得到敏感API字典按顺序编号
def getEntityIdDict(sourceDict, sinkDict):
    
    idToEntityDict = OrderedDict()
    entityToIdDict = OrderedDict()
    id = 0
    for sourceEntity in sourceDict.keys():
        if sourceEntity not in sinkDict.keys():
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
    idToEntityDict, entityToIdDict = getEntityIdDict(sourceDict, sinkDict)
    print("number of source methods : %s" % len(sourceDict))
    print("number of sink methods : %s" % len(sinkDict))
    print("number of idToEntityDict: %s" % len(idToEntityDict))
    print(len(entityToIdDict))
    genCategoryCSV(getNumOfCategory(sourceDict), 
                   getNumOfCategory(sinkDict), "Category.csv")
    
    
    
    
    