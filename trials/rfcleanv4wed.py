# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 10:24:52 2016

@author: priyanka
"""

#rfclean for large data set

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

def clean(x):
    l= []
    for item in x:
        if len(item) <= 2:
            l.append(item)
    return l 
    
#removed bytetestpath and classtestpath
    
def main():
    sc = SparkContext(conf = SparkConf().setAppName("Random Forest"))
    sqlContext = SQLContext(sc)
    bytePath = "s3n://eds-uga-csci8360/data/project2/binaries"
    namePath = "s3n://eds-uga-csci8360/data/project2/labels/X_train_small.txt"
    nameTestPath="s3n://eds-uga-csci8360/data/project2/labels/X_test_small.txt"
    classPath = "s3n://eds-uga-csci8360/data/project2/labels/y_train_small.txt"

#bytePath =  "/Users/priyanka/Desktop/project2files/all"
#namePath = "/Users/priyanka/Desktop/X_train_small.txt"
#nameTestPath="/Users/priyanka/Desktop/X_test_small.txt"
#classPath = "/Users/priyanka/Desktop/y_train_small.txt"

#docData Output: ('file:/Users/priyanka/Desktop/project2files/train/04mcPSei852tgIKUwTJr.bytes', '00401000 20 FF 58 C0 20 FE 5C 01 F8 00 0F 8B 50 FC 06 01\r\n00401010 8C 01 FF")
    docData= sc.wholeTextFiles(bytePath, 25).map(lambda (x,y):(x.encode("utf-8"),y.encode("utf-8")))
    print ("docData frankie")
    docData.take(1)


#clean docData here - remove 1st word from line and remove /r/n  
    cleanDocData = docData.map(lambda (x,y): (x,clean(y.split())))


#try calculating tf here (filename,tf)
    x = 16**2 +1 
    hashingTF = HashingTF(x) 
    tfDocData = cleanDocData.map(lambda (x,y): (x,hashingTF.transform(y)))
    tfDocData.take(1)
#Output format : (index,filename)
    nameData = sc.textFile(namePath,25).map(lambda x: "file:"+bytePath+"/"+x+".bytes").zipWithIndex().map(lambda (x,y):(y,x))
#nameData.take(5)   

#Output format: (index,label)
    labelData = sc.textFile(classPath,25).zipWithIndex().map(lambda (x,y):(y,str(int(x)-1)))

#Output format: (filename,label)
    joinNameLabel = nameData.join(labelData).map(lambda (x,y):y)
#joinNameLabel.take(5) 

#Output: (label,tfidf)
    joinCleanDocLabel = joinNameLabel.join(tfDocData).map(lambda (x,y):y) 

#Output: (label,tfidf)
    hashData = joinCleanDocLabel.map(lambda (label, text): LabeledPoint(label,text))
    print "hashing TF done"
 

    print ("generating model fliss")
    model1 = RandomForest.trainClassifier(hashData, numClasses=9, categoricalFeaturesInfo={},
                                     numTrees=50, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=8, maxBins=32)

#==============================================================================
# Testing starts here
#==============================================================================
#Output: (filename,index)
    nameTestData=sc.textFile(nameTestPath,25).map(lambda x: "file:"+bytePath+"/"+x+".bytes").zipWithIndex()

#Output: (index,tfidf)
    joinTestDocLabel=nameTestData.join(tfDocData).map(lambda (x,y):y)

    print ("hashing test kenny")
    hashTestData = joinTestDocLabel.map(lambda (label, text): LabeledPoint(label, text))
    hashTestData.persist()

#Random forest prediction and labels and accuracy
    print "prediction part lyndz"
    prediction1 = model1.predict(hashTestData.map(lambda x: x.features))

    prediction1.saveAsTextFile("/Users/priyanka/Desktop/pred.txt")


if __name__ == "__main__":
    main()





