# -*- coding: utf-8 -*-
"""
工具函数
"""
import os
import shutil
import json
import re
from collections import OrderedDict
# 从节点标签中获取函数实体
def getEntity(nodeLabel):
    entity = ""
    pattern = re.compile("{<(.*?): (.*?) (.*?)\((.*?)\)>(.*)}")
    matcher = pattern.match(nodeLabel)
    if matcher:
        packageName = matcher.group(1)
        methodName = matcher.group(3)
        entity = packageName + ": " + methodName
    return entity

# 从json文件中恢复有序字典
def getOrderedDictFromJsonFile(jsonFilePath):
    with open(jsonFilePath, "r") as jsonFile:
        OrderedDictFromJsonFile = json.load(jsonFile, object_pairs_hook=OrderedDict)
    return OrderedDictFromJsonFile

# 清理文件夹，只保留列表中的内容
def cleanDir(dirPath, remainList=[]):
    for fileName in os.listdir(dirPath):
        if fileName not in remainList:
            filePath = os.path.join(dirPath, fileName)
            if os.path.isfile(filePath):
                os.unlink(filePath)
            if os.path.isdir(filePath):
                shutil.rmtree(filePath)