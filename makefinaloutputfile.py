# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 21:29:35 2016

@author: priyanka
"""

#make o/p file for autolab

from __future__ import division
import urllib
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row
import re
from pyspark.mllib.feature import HashingTF, IDF
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.sql import SQLContext

from pyspark.ml.feature import NGram

def main():

    sc = SparkContext(conf = SparkConf().setAppName("make output file"))
#    nameTestPath="/Users/priyanka/Desktop/X_test_small.txt"
#    bytePath = "/Users/priyanka/Desktop/project2files/all"
    nameTestPath="s3n://eds-uga-csci8360/data/project2/labels/X_test.txt"
    bytePath = "s3n://eds-uga-csci8360/data/project2/binaries"
    op = sc.textFile("output.txt").map(lambda x:x.encode("utf-8")).map(lambda x: x.split()).map(lambda (x,y):(x,float(y)))

    #<file,pred label>
    op1 = op.map(lambda (x,y):(x,int(y)))
    
    #<file,index>
    xtest = sc.textFile(nameTestPath).map(lambda x: "file:"+bytePath+"/"+x+".bytes").zipWithIndex().map(lambda (x,y):(x.encode("utf-8"),y))
    
    #<index,label>
    joinOpXtest = xtest.join(op1).map(lambda (x,y):y)
    
    #Sort by index - <index,label>
    j = joinOpXtest.sortByKey()
    
    #keep only label
    final = j.map(lambda (x,y):y)
    
    #save to Text File
    final.saveAsTextFile("finaloutput.txt")
    