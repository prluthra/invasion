# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 20:36:26 2016

@author: priyanka
"""

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
from ast import literal_eval as make_tuple



def main():
    sc = SparkContext(conf = SparkConf().setAppName("Check accuracy"))
    bytePath = "/Users/priyanka/Desktop/project2files/all"
    origLabel = sc.textFile("/Users/priyanka/Desktop/y_test_small.txt").zipWithIndex().map(lambda (x,y):(y,float(x.encode("utf-8"))))
    
    origFile = sc.textFile("/Users/priyanka/Desktop/X_test_small.txt").map(lambda x:"file:"+bytePath+"/"+x+".bytes").zipWithIndex().map(lambda (x,y):(y,x.encode("utf-8")))
    
    orig = origFile.join(origLabel).map(lambda (x,y):y)
    
    pred = sc.textFile("/Users/priyanka/Desktop/output6.txt").map(lambda x:x.encode("utf-8"))
    
    predTuple = pred.map(lambda x: tuple(x.split())).map(lambda (x,y):(x,float(y)))
    
    joinOrigPred = orig.join(predTuple)
    
    joinFinal = joinOrigPred.map(lambda (x,y):y)
    
    noOfWrongRows= joinFinal.filter(lambda (x,y): x!=y).count()
    
    error = noOfWrongRows/origLabel.count()
    
    print error
    
    accuracy = 1- error
    
    print accuracy
    