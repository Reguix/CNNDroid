# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 09:05:21 2019

@author: ZhangXin
"""

import re
import os

statisticsSuperNodeFilePath = "DataStatistics/statisticsSuperNode.txt"

with open(statisticsSuperNodeFilePath, "r") as statisticsSuperNodeFile:
    lines = statisticsSuperNodeFile.readlines()
    print(len(lines))

superNodeDict = dict()
superNodeSet = set()
    
for line in lines:
    pattern = re.compile("(\{.*?\}) (.*?) (.*?)\n")
    matcher = pattern.match(line)
    if matcher:
        nodeLabel = matcher.group(1)
        adjCount = matcher.group(2)
        numOfCluster = matcher.group(3)
#        print(nodeLabel)
#        print(adjCount)
#        print(numOfCluster)
        
        if nodeLabel not in superNodeDict.keys():
            superNodeDict[nodeLabel] = [int(adjCount), int(numOfCluster)]
        else:
            countList = superNodeDict[nodeLabel]
            countList[0] += int(adjCount)
            countList[1] += int(numOfCluster)
            superNodeDict[nodeLabel] = countList
            
        if adjCount == numOfCluster:
            superNodeSet.add(nodeLabel)
            # print(nodeLabel)

print(len(superNodeDict))
print(superNodeSet)
        
    
        
        